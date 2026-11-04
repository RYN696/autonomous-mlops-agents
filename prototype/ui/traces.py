"""Traçabilité : le journal d'actions regroupé en parcours.

Un parcours suit une demande de bout en bout, à travers tous les agents :
  - demande de réentraînement : lignes dont details.request_id est le même
    (orchestrateur, Coûts, Retrain), plus le contrôle Qualité qui l'a
    déclenchée (details.check_id) et les messages envoyés à l'humain ;
  - incident : lignes dont details.incident_id est le même (Fiabilité),
    plus la décision du LLM qui l'a ouvert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import String, cast, func, or_, select

from common import config
from common.action_log import action_log
from ui.catalog import BY_LOG_NAME, HUMAN, AgentInfo, agent_of
from ui.summaries import summarize

RETRAIN, INCIDENT = "retrain", "incident"
SCAN_ROWS = 3000  # lignes récentes lues pour lister les parcours


def utc(ts: datetime) -> datetime:
    """SQLite rend des dates sans fuseau, Postgres avec : tout en UTC."""
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def trace_key(details: dict | None) -> tuple[str, str] | None:
    d = details or {}
    if d.get("request_id"):
        return RETRAIN, str(d["request_id"])
    if d.get("incident_id"):
        return INCIDENT, str(d["incident_id"])
    return None


@dataclass
class Step:
    ts: datetime
    agent: AgentInfo
    action: str
    summary: str
    details: dict[str, Any]
    row_id: int | None = None
    is_message: bool = False
    offset: float = 0.0  # secondes depuis le début du parcours


@dataclass
class Trace:
    id: str
    kind: str
    title: str = ""
    status: str = "En cours"
    level: str = "run"  # ok, warn, bad, wait, run, muted (couleur du badge)
    started: datetime | None = None
    ended: datetime | None = None
    agents: list[AgentInfo] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.ended - self.started).total_seconds() if self.started and self.ended else 0.0

    @property
    def actions(self) -> list[str]:
        return [s.action for s in self.steps]


def _stale(trace: Trace, minutes: float, now: datetime | None) -> bool:
    """Plus rien depuis longtemps : l'agent s'est arrêté en cours de route (redémarrage, panne)."""
    now = now or datetime.now(timezone.utc)
    return trace.ended is not None and now - trace.ended > timedelta(minutes=minutes)


def _status(trace: Trace, now: datetime | None = None) -> tuple[str, str]:
    acts = set(trace.actions)
    if trace.kind == INCIDENT:
        if "incident_resolved" in acts:
            return f"Résolu en {trace.duration:.0f} s", "ok"
        if _stale(trace, config.INCIDENT_HOLD_MINUTES, now):
            return "Sans clôture", "muted"
        if "human_notified" in acts:
            return "Humain prévenu", "wait"
        return "En cours", "run"
    label, level = _retrain_status(trace, acts)
    # une question peut attendre l'humain longtemps : seul un travail "en cours" expire
    if level == "run" and _stale(trace, config.RETRAIN_TIMEOUT_MINUTES, now):
        return "Sans suite", "muted"
    return label, level


def _retrain_status(trace: Trace, acts: set[str]) -> tuple[str, str]:
    for action, label, level in (
        ("model_promoted", "Nouveau modèle en production", "ok"),
        ("promotion_blocked", "Promotion bloquée", "warn"),
        ("retrain_failed", "Entraînement en échec", "bad"),
        ("request_failed", "Demande en échec", "bad"),
        ("request_expired", "Expirée (redémarrage)", "muted"),
        ("retrain_refused_corrupted_data", "Données corrompues", "bad"),
        ("retrain_skipped", "Ignoré par Retrain", "muted"),
        ("retrain_refused", "Refusé par un humain", "muted"),
        ("drift_ignored", "Mise de côté", "muted"),
        ("retrain_requested", "Entraînement en cours", "run"),
    ):
        if action in acts:
            return label, level
    decisions = [s for s in trace.steps if s.action == "retrain_decision"]
    if decisions and decisions[-1].details.get("rule") == "approval":
        return "En attente d'un humain", "wait"
    if "retrain_decision" in acts and decisions[-1].details.get("rule") == "forbidden":
        return "Interdit par les règles", "muted"
    return "En attente du coût", "run"


def _title(trace: Trace) -> str:
    d = next((s.details for s in trace.steps if s.action in ("drift_received", "drift_detected")), None)
    if trace.kind == RETRAIN:
        if d and d.get("reason") == "corrupted_input_data":
            return f"Données corrompues {d.get('period') or ''}".strip()
        return f"Dérive {d.get('period')}" if d and d.get("period") else "Demande de réentraînement"
    d = next((s.details for s in trace.steps if s.action == "incident_opened"), {})
    return f"{d.get('cause', 'incident')} sur {d.get('component', '?')}"


def _finish(trace: Trace) -> Trace:
    trace.steps.sort(key=lambda s: (s.ts, s.row_id or 0))
    if trace.steps:
        trace.started, trace.ended = trace.steps[0].ts, trace.steps[-1].ts
        for s in trace.steps:
            s.offset = (s.ts - trace.started).total_seconds()
            if s.agent.key not in [a.key for a in trace.agents]:
                trace.agents.append(s.agent)
    trace.title = _title(trace)
    trace.status, trace.level = _status(trace)
    return trace


def _step(row: dict) -> Step:
    details = row.get("details") or {}
    return Step(utc(row["timestamp"]), agent_of(row["agent"]), row["action"],
                summarize(row["action"], details), details, row_id=row.get("id"))


def message_step(n: dict) -> Step:
    """Message affiché à l'humain (et sa réponse s'il s'agit d'une question)."""
    text = n["text"]
    if n.get("kind") == "question":
        answer = {"pending": "en attente de réponse", "approved": f"validé par {n.get('answered_by')}",
                  "refused": f"refusé par {n.get('answered_by')}",
                  "expired": "expirée (orchestrateur redémarré)"}.get(n.get("status"), n.get("status"))
        text = f"Question : {text} — {answer}"
    return Step(utc(n["timestamp"]), HUMAN, "message" if n.get("kind") != "question" else "question",
                text, {k: v for k, v in n.items() if k != "timestamp"}, is_message=True)


def group(rows: list[dict]) -> list[Trace]:
    """Parcours présents dans ces lignes, du plus récent au plus ancien."""
    traces: dict[tuple[str, str], Trace] = {}
    for row in rows:
        key = trace_key(row.get("details"))
        if key:
            traces.setdefault(key, Trace(id=key[1], kind=key[0])).steps.append(_step(row))
    return sorted((_finish(t) for t in traces.values()), key=lambda t: t.ended, reverse=True)


def _rows(conn, stmt) -> list[dict]:
    return [dict(r._mapping) for r in conn.execute(stmt)]


def recent_traces(engine, limit: int = 50, kind: str | None = None) -> list[Trace]:
    with engine.connect() as conn:
        rows = _rows(conn, select(action_log).order_by(action_log.c.id.desc()).limit(SCAN_ROWS))
    traces = [t for t in group(rows) if kind in (None, "", t.kind)]
    return traces[:limit]


def load_trace(engine, trace_id: str, notifier=None) -> Trace | None:
    details = action_log.c.details
    with engine.connect() as conn:
        rows = _rows(conn, select(action_log).where(or_(
            details["request_id"].as_string() == trace_id,
            details["incident_id"].as_string() == trace_id)).order_by(action_log.c.id))
        if not rows:
            return None
        kind = trace_key(rows[0]["details"])[0]
        extra: list[dict] = []
        check_ids = {r["details"].get("check_id") for r in rows if (r["details"] or {}).get("check_id")}
        for check_id in check_ids:  # le contrôle Qualité qui a déclenché la demande
            extra += _rows(conn, select(action_log).where(
                action_log.c.agent == "agent_qualite", details["check_id"].as_string() == check_id))
        opened = next((r for r in rows if r["action"] == "incident_opened"), None)
        if opened:  # la décision du LLM qui a ouvert l'incident (pas encore d'incident_id à ce moment)
            ts = opened["timestamp"]
            extra += _rows(conn, select(action_log).where(
                action_log.c.agent == "agent_fiabilite", action_log.c.action == "llm_decision",
                action_log.c.timestamp <= ts, action_log.c.timestamp >= ts - timedelta(minutes=2),
                action_log.c.id < opened["id"]).order_by(action_log.c.id.desc()).limit(1))
    trace = Trace(id=trace_id, kind=kind)
    trace.steps = [_step(r) for r in extra + rows]
    if notifier is not None:
        trace.steps += [message_step(n) for n in notifier.for_request(trace_id)]
    return _finish(trace)


def journal(engine, agent: str | None = None, action: str | None = None, q: str | None = None,
            before_id: int | None = None, limit: int = 100) -> list[dict]:
    """Journal brut filtré, du plus récent au plus ancien, avec résumé et parcours."""
    stmt = select(action_log).order_by(action_log.c.id.desc()).limit(limit)
    if agent:
        stmt = stmt.where(action_log.c.agent == agent)
    if action:
        stmt = stmt.where(action_log.c.action == action)
    if q:
        stmt = stmt.where(or_(cast(action_log.c.details, String).ilike(f"%{q}%"),
                              action_log.c.action.ilike(f"%{q}%")))
    if before_id:
        stmt = stmt.where(action_log.c.id < before_id)
    with engine.connect() as conn:
        rows = _rows(conn, stmt)
    for r in rows:
        r["timestamp"] = utc(r["timestamp"])
        r["info"] = agent_of(r["agent"])
        r["summary"] = summarize(r["action"], r["details"])
        key = trace_key(r["details"])
        r["trace_id"] = key[1] if key else None
    return rows


def actions_of(engine, log_name: str) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(select(action_log.c.action).where(action_log.c.agent == log_name)
                                           .group_by(action_log.c.action).order_by(action_log.c.action))]


def agent_activity(engine, now: datetime | None = None) -> dict[str, dict]:
    """Par agent : dernière action, et nombre d'actions sur 24 h."""
    now = now or datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    with engine.connect() as conn:
        day = conn.execute(select(action_log.c.agent, func.count()).where(
            action_log.c.timestamp >= now - timedelta(hours=24)).group_by(action_log.c.agent))
        counts = dict(day.all())
        for log_name in BY_LOG_NAME:
            last = conn.execute(select(action_log).where(action_log.c.agent == log_name)
                                .order_by(action_log.c.id.desc()).limit(1)).first()
            out[log_name] = {
                "count_24h": counts.get(log_name, 0),
                "last": ({"ts": utc(last.timestamp), "action": last.action,
                          "summary": summarize(last.action, last.details)} if last else None),
            }
    return out
