"""Interface de la pipeline (http://localhost:8080), servie par l'orchestrateur.

/                 vue d'ensemble : agents, services, questions en attente, parcours récents
/parcours         toutes les demandes de réentraînement et tous les incidents
/parcours/<id>    chronologie d'un parcours, étape par étape et agent par agent
/agents/<clé>     ce que fait un agent, ce qu'il écoute et publie, ses dernières actions
/journal          journal d'actions complet, filtrable
/graph            les graphes LangGraph de l'orchestrateur (Mermaid)
/scenarios        banc de test : boutons pour provoquer dérives et pannes, sans terminal
POST /answer      réponse humaine à une question (publie approval_given)
Les gabarits et le style sont dans prototype/ui/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request

from common.events import Event, EventType
from ui import traces as tr
from ui.catalog import AGENTS, BY_KEY
from ui.scenarios import SCENARIOS, ScenarioRunner
from ui.status import PipelineStatus

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def create_app(notifier, bus, log, orchestrator=None, status: PipelineStatus | None = None,
               runner: ScenarioRunner | None = None) -> Flask:
    app = Flask("orchestrator-web", template_folder=str(UI_DIR / "templates"),
                static_folder=str(UI_DIR / "static"))
    app.secret_key = "banc-de-test-local"  # messages flash seulement (aucune session utilisateur)
    status = status or PipelineStatus(log.engine)
    runner = runner or ScenarioRunner(log)

    @app.context_processor
    def common_values():
        return {"agents": AGENTS, "has_graph": orchestrator is not None,
                "pending_count": len(notifier.pending()), "path": request.path}

    @app.template_filter("iso")
    def iso(ts: datetime | None) -> str:
        return tr.utc(ts).isoformat() if ts else ""

    @app.template_filter("pretty")
    def pretty(value) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    @app.template_filter("duration")
    def duration(seconds: float) -> str:
        seconds = int(round(seconds or 0))
        if seconds < 60:
            return f"{seconds} s"
        if seconds < 3600:
            return f"{seconds // 60} min {seconds % 60:02d} s"
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"

    def in_progress() -> list[dict]:
        """Fils LangGraph en pause, avec l'étape attendue."""
        if orchestrator is None:
            return []
        waiting = {"cost_estimated": "attend le coût", "approval_given": "attend un humain"}
        return [{"id": rid, "step": waiting.get(getattr(orchestrator.waiting_for(rid), "value", ""), "?")}
                for rid in sorted(orchestrator.pending)]

    @app.get("/")
    def index():
        containers = status.containers()
        activity = tr.agent_activity(log.engine)
        cards = [{"info": a, "container": containers.get(a.service), **activity.get(a.log_name, {})}
                 for a in AGENTS]
        recent = tr.recent_traces(log.engine, limit=8)
        incidents = [t for t in tr.recent_traces(log.engine, limit=50, kind=tr.INCIDENT) if t.level in ("run", "wait")]
        return render_template("overview.html", cards={c["info"].key: c for c in cards}, services=status.services(),
                               production=status.production(), costs=status.month_cost(),
                               questions=notifier.pending(), messages=notifier.latest(12), recent=recent,
                               open_incidents=incidents, in_progress=in_progress(),
                               docker_ok=bool(containers))

    @app.get("/parcours")
    def traces():
        kind = request.args.get("type") or None
        return render_template("traces.html", traces=tr.recent_traces(log.engine, limit=200, kind=kind), kind=kind)

    @app.get("/parcours/<trace_id>")
    def trace(trace_id: str):
        t = tr.load_trace(log.engine, trace_id, notifier)
        if t is None:
            abort(404)
        return render_template("trace.html", t=t)

    @app.get("/agents/<key>")
    def agent(key: str):
        info = BY_KEY.get(key) or abort(404)
        action = request.args.get("action") or None
        rows = tr.journal(log.engine, agent=info.log_name, action=action, limit=80)
        activity = tr.agent_activity(log.engine).get(info.log_name, {})
        return render_template("agent.html", info=info, rows=rows, action=action, activity=activity,
                               container=status.containers().get(info.service),
                               actions=tr.actions_of(log.engine, info.log_name))

    @app.get("/journal")
    def journal():
        args = {k: request.args.get(k) or None for k in ("agent", "action", "q")}
        before = request.args.get("before", type=int)
        rows = tr.journal(log.engine, before_id=before, limit=100, **args)
        return render_template("journal.html", rows=rows, args=args,
                               next_before=rows[-1]["id"] if len(rows) == 100 else None)

    @app.get("/graph")
    def graph():
        if orchestrator is None:
            return "Orchestrateur non branché sur cette page.", 404
        return render_template("graph.html", graphs=orchestrator.mermaid(), in_progress=in_progress())

    @app.get("/scenarios")
    def scenarios():
        runs = []
        for r in runner.runs:
            trace_id = None
            if r.check_id:
                trace_id = tr.trace_of_check(log.engine, r.check_id)
            elif r.scenario.target == "reliability" and r.scenario.key != "recover":
                trace_id = tr.incident_after(log.engine, r.started)
            runs.append({"run": r, "trace_id": trace_id})
        groups: dict[str, list] = {}
        for s in SCENARIOS:
            groups.setdefault(s.group, []).append(s)
        return render_template("scenarios.html", groups=groups, runs=runs, runner=runner,
                               trained=tr.trained_months(log.engine), agents_by_key=BY_KEY)

    @app.post("/scenarios/run")
    def run_scenario():
        try:
            runner.start(request.form["scenario"], {"month": request.form.get("month")})
        except (PermissionError, RuntimeError, KeyError) as exc:
            flash(str(exc))
        return redirect("/scenarios")

    @app.post("/scenarios/traffic")
    def traffic():
        try:
            if request.form.get("action") == "start":
                runner.start_traffic(request.form.get("rps", type=float) or 5.0)
            else:
                runner.stop_traffic()
        except PermissionError as exc:
            flash(str(exc))
        return redirect("/scenarios")

    @app.post("/answer")
    def answer():
        request_id = request.form["request_id"]
        approved = request.form.get("choice") == "yes"
        who = request.form.get("who") or "équipe"
        if notifier.answer(request_id, approved, who):  # False si quelqu'un a déjà répondu
            bus.publish(Event(type=EventType.APPROVAL_GIVEN, source="page_web", payload={
                "request_id": request_id, "approved": approved, "approved_by": who}))
        return redirect(request.form.get("next") or "/")

    @app.template_global()
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    return app
