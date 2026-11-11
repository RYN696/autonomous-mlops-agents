"""« Qui a décidé ? » : les décisions des dernières 24 h, classées par origine.

C'est le message central du projet : les décisions critiques sont des règles écrites dans le
code, le LLM propose ou explique, l'humain garde la main sur ce qui coûte ou ce qui est risqué.
Ce module ne compte que des DÉCISIONS (pas les mesures ni les coûts enregistrés), lues dans le
journal d'actions, et dit pour chacune qui l'a prise :
  - regle  : un seuil ou une règle de code (rules/autonomy.yaml, config.py, guards.py) ;
  - llm    : une proposition du LLM, acceptée par le juge et les garde-fous ;
  - humain : une validation, un refus, un retour arrière ou une alerte ignorée.
Les « garde-fous » comptent les fois où le code ou le juge a écarté une proposition du LLM.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from common.action_log import action_log
from ui.summaries import summarize

# (agent, action) -> origine fixe de la décision
RULE_DECISIONS = {
    ("orchestrator", "retrain_decision"), ("orchestrator", "drift_ignored"),
    ("orchestrator", "retrain_refused_corrupted_data"), ("orchestrator", "suspect_data_alert"),
    ("agent_retrain", "model_promoted"), ("agent_retrain", "promotion_blocked"),
    ("agent_retrain", "retrain_skipped"),
    ("cost", "budget_exceeded"), ("cost", "cost_anomaly"), ("cost", "cost_recommendation"),
    ("agent_qualite", "drift_detected"),
}
GUARDS = {"verdict_corrected", "incident_rejected_by_rule", "judge_rejected"}
HUMAN_AGENT = "humain"
LABELS = {"regle": "Règle du code", "llm": "LLM (validé)", "humain": "Humain"}


def origin(agent: str, action: str, details: dict | None) -> str | None:
    """Origine d'une ligne du journal si c'est une décision, sinon None."""
    d = details or {}
    if agent == HUMAN_AGENT:
        return "humain"
    if (agent, action) in RULE_DECISIONS:
        return "regle"
    if agent == "agent_fiabilite" and action.startswith("action:"):
        return "regle" if str(d.get("by", "")).startswith("regle") else "llm"
    if agent == "orchestrator" and action == "chat_understood" and not d.get("problem"):
        return "llm" if d.get("understood_by") == "llm" else "regle"
    return None


def who_decides(engine, hours: float = 24, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    counts = {k: 0 for k in LABELS}
    latest: dict[str, dict] = {}
    guards = 0
    with engine.connect() as conn:
        rows = conn.execute(select(action_log.c.agent, action_log.c.action, action_log.c.details,
                                   action_log.c.timestamp)
                            .where(action_log.c.timestamp >= since)
                            .order_by(action_log.c.id.desc())).all()
    for agent, action, details, ts in rows:
        if action in GUARDS:
            guards += 1
            continue
        o = origin(agent, action, details)
        if o is None:
            continue
        counts[o] += 1
        if o not in latest:
            latest[o] = {"summary": summarize(action, details), "action": action, "ts": ts}
    total = sum(counts.values())
    return {
        "hours": hours, "total": total, "guards": guards,
        "parts": [{"key": k, "label": LABELS[k], "count": counts[k],
                   "share": counts[k] / total if total else 0.0, "latest": latest.get(k)} for k in LABELS],
    }
