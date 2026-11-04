"""Interface de la pipeline (http://localhost:8080), servie par l'orchestrateur.

/                 vue d'ensemble : agents, services, questions en attente, parcours récents
/parcours         toutes les demandes de réentraînement et tous les incidents
/parcours/<id>    chronologie d'un parcours, étape par étape et agent par agent
/agents/<clé>     ce que fait un agent, ce qu'il écoute et publie, ses dernières actions
/journal          journal d'actions complet, filtrable
/graph            les graphes LangGraph de l'orchestrateur (Mermaid)
/scenarios        banc de test : boutons pour provoquer dérives et pannes, sans terminal
/couts            rapport de coûts du mois (chiffres du code, explication par le LLM)
/storyboard       les 4 scénarios du storyboard, joués pour de vrai et suivis case par case
/chat             canal #mlops-agents : commandes en langage naturel pour l'orchestrateur
/donnees          aperçu d'un fichier de données reçu (bouton « Voir les données »)
POST /answer      réponse humaine à une question (publie approval_given, comme le bouton Discord)
POST /actions/... gestes humains depuis la page : revenir à une version, ignorer une alerte
POST /budget      budget du mois réglé par un responsable (lu en direct par l'agent Coûts)
Les gabarits et le style sont dans prototype/ui/.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request

from agents.cost import report as cost_report
from agents.orchestrator.human import HumanActionError, HumanActions
from common import budget as live_budget
from common import config
from ui import cost_charts, data_preview
from ui import traces as tr
from ui.catalog import AGENTS, BY_KEY, agent_of
from ui.decisions import who_decides
from ui.scenarios import SCENARIOS, ScenarioRunner
from ui.status import PipelineStatus
from ui.storyboard import STORIES, StoryBoard, actor_icon
from ui.summaries import summarize

HOT_SECONDS = 120  # vue d'ensemble : un agent qui a agi depuis moins de 2 min est « actif »
ROUTINE = {"cost_recorded", "quality_ok", "llm_decision"}  # mesures de fond : n'animent pas la carte

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def create_app(notifier, bus, log, orchestrator=None, status: PipelineStatus | None = None,
               runner: ScenarioRunner | None = None, human: HumanActions | None = None,
               discord=None, board: StoryBoard | None = None) -> Flask:
    app = Flask("orchestrator-web", template_folder=str(UI_DIR / "templates"),
                static_folder=str(UI_DIR / "static"))
    app.secret_key = "banc-de-test-local"  # messages flash seulement (aucune session utilisateur)
    status = status or PipelineStatus(log.engine)
    runner = runner or ScenarioRunner(log)
    human = human or HumanActions(notifier, bus, log)
    board = board or StoryBoard(runner, log, notifier, orchestrator, status)

    def services() -> list[dict]:
        if discord is None:
            return status.services()
        return status.services() + [discord.state()]

    @app.context_processor
    def common_values():
        return {"agents": AGENTS, "has_graph": orchestrator is not None,
                "pending_count": len(notifier.pending()), "path": request.path,
                "actor_icon": actor_icon}

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
        now = datetime.now(timezone.utc)
        # agents actifs dans les 2 dernières minutes : la carte anime leurs liens
        hot = [c["info"].key for c in cards
               if c.get("last") and c["last"]["action"] not in ROUTINE
               and (now - c["last"]["ts"]).total_seconds() < HOT_SECONDS]
        return render_template("overview.html", cards={c["info"].key: c for c in cards}, services=services(),
                               decisions=who_decides(log.engine), hot=hot,
                               production=status.production(), costs=status.month_cost(),
                               questions=notifier.pending(), messages=notifier.latest(12), recent=recent,
                               open_incidents=incidents, in_progress=in_progress(),
                               docker_ok=bool(containers), containers=containers)

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

    report_job = {"running": False, "error": None}

    def make_cost_report() -> None:
        try:
            report = cost_report.make_report(log.engine)
            log.record(cost_report.AGENT, "cost_report", config.MODEL_NAME, report)
            notifier.notify(cost_report.headline(report), level="info", actions=("costs",))
            report_job["error"] = None
        except Exception as exc:  # Postgres injoignable... : affiché sur la page
            report_job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            report_job["running"] = False

    @app.get("/rapports")
    def reports():
        """Rapports de fin de mission des agents Qualité et Retrain (écrits par le code)."""
        which = request.args.get("agent") or None
        alerts = request.args.get("tous") is None   # par défaut : sans les contrôles « conformes »
        rows = []
        for agent, action in (("agent_retrain", "retrain_report"), ("agent_qualite", "quality_report")):
            if which and which != agent:
                continue
            for r in tr.journal(log.engine, agent=agent, action=action, limit=60):
                if alerts and r["details"].get("level") == "ok" and action == "quality_report":
                    continue
                rows.append(r)
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return render_template("reports.html", rows=rows[:30], which=which, alerts=alerts)

    @app.get("/couts")
    def costs():
        reports = tr.journal(log.engine, agent=cost_report.AGENT, action="cost_report", limit=6)
        try:
            live = cost_report.figures(log.engine)
        except Exception:
            live = None
        try:
            charts = cost_charts.charts(log.engine, live["budget"], live["projection"]) if live else None
        except Exception:                      # un graphique en erreur ne casse pas la page
            charts = None
        return render_template("costs.html", reports=reports, live=live, job=report_job, charts=charts,
                               model=config.COST_LLM_MODEL)

    @app.post("/couts/rapport")
    def new_cost_report():
        if not report_job["running"]:  # un seul rapport à la fois (le LLM est lent)
            report_job["running"] = True
            threading.Thread(target=make_cost_report, daemon=True).start()
        return redirect("/couts")

    # ------------------------------------------------ storyboard
    def budget_info() -> dict:
        live = live_budget.current()
        costs = status.month_cost()
        return {"live": live, "value": live if live is not None else (costs or {}).get("budget"),
                "spent": (costs or {}).get("total")}

    def journal_rows(rows: list[dict]) -> list[dict]:
        for r in rows:
            r["timestamp"], r["info"] = tr.utc(r["timestamp"]), agent_of(r["agent"])
            r["summary"] = summarize(r["action"], r["details"])
            key = tr.trace_key(r["details"])
            r["trace_id"] = key[1] if key else None
        return rows

    @app.get("/storyboard")
    def storyboard():
        runs = [{"run": r, "progress": board.progress(r), "finished": board.finished(r)} for r in board.runs]
        return render_template("storyboard.html", stories=STORIES, runs=runs, board=board,
                               trained=tr.trained_months(log.engine), budget=budget_info())

    @app.post("/storyboard/run")
    def run_story():
        params = {k: request.form.get(k) for k in ("month", "text", "who")}
        try:
            run = board.start(request.form["story"], params)
        except (PermissionError, RuntimeError, KeyError) as exc:
            flash(str(exc))
            return redirect("/storyboard")
        return redirect(f"/storyboard/{run.id}")

    @app.get("/storyboard/<run_id>")
    def story(run_id: str):
        run = board.get(run_id) or abort(404)
        trace_id = board.trace_id(run)
        questions = [n for n in notifier.pending() if trace_id and n["request_id"] == trace_id]
        return render_template("story.html", run=run, progress=board.progress(run), finished=board.finished(run),
                               trace_id=trace_id, questions=questions, rows=journal_rows(board.related_rows(run)),
                               budget=budget_info(), grafana=config.GRAFANA_PUBLIC_URL)

    @app.post("/budget")
    def set_budget():
        who = request.form.get("who") or "équipe"
        try:
            if request.form.get("action") == "reset":
                live_budget.clear()
                log.record("humain", "budget_changed", config.MODEL_NAME, {
                    "budget": None, "by": who, "reason": "retour au budget de configuration de l'agent Coûts"})
            else:
                value = live_budget.set_budget(float(request.form["budget"].replace(",", ".")))
                log.record("humain", "budget_changed", config.MODEL_NAME, {"budget": value, "by": who})
        except (ValueError, KeyError) as exc:
            flash(f"budget invalide ({exc})")
        except Exception as exc:  # Redis injoignable
            flash(f"budget non enregistré : {type(exc).__name__}: {exc}")
        return redirect(request.form.get("next") or "/storyboard")

    # ------------------------------------------------ chat
    @app.get("/chat")
    def chat():
        conversations = []
        for row in tr.journal(log.engine, action="chat_received", limit=15):
            rid = row["details"].get("request_id")
            understood = tr.journal(log.engine, action="chat_understood", q=rid, limit=1)
            conversations.append({"row": row, "id": rid, "understood": understood[0]["details"] if understood else None,
                                  "messages": notifier.for_request(rid)})
        return render_template("chat.html", conversations=conversations, enabled=orchestrator is not None)

    @app.post("/chat")
    def post_chat():
        text = (request.form.get("text") or "").strip()
        who = request.form.get("who") or "équipe"
        if not text or orchestrator is None:
            flash("message vide" if not text else "orchestrateur non branché sur cette page")
            return redirect("/chat")
        request_id = uuid.uuid4().hex[:8]
        # le LLM peut mettre plusieurs secondes à comprendre : la page se met à jour toute seule
        threading.Thread(target=orchestrator.on_chat, args=(text[:500], who, request_id), daemon=True).start()
        return redirect("/chat")

    # ------------------------------------------------ gestes humains (comme les boutons Discord)
    @app.post("/actions/rollback")
    def human_rollback():
        f = request.form
        try:
            flash(human.rollback(f["target"], f["current"], f.get("request_id") or None,
                                 f.get("who") or "équipe", "page_web"))
        except HumanActionError as exc:
            flash(f"Rien n'a été changé : {exc}")
        return redirect(f.get("next") or "/")

    @app.post("/actions/ignore")
    def human_ignore():
        f = request.form
        flash(human.ignore(f.get("request_id") or None, f.get("who") or "équipe", "page_web"))
        return redirect(f.get("next") or "/")

    @app.get("/donnees")
    def data_view():
        name = request.args.get("fichier", "")
        try:
            preview, error = data_preview.preview(name), None
        except (ValueError, OSError) as exc:
            preview, error = None, str(exc)
        return render_template("data.html", preview=preview, error=error, name=name)

    @app.post("/answer")
    def answer():
        request_id = request.form["request_id"]
        approved = request.form.get("choice") == "yes"
        who = request.form.get("who") or "équipe"
        human.answer(request_id, approved, who)  # sans effet si quelqu'un a déjà répondu (ici ou sur Discord)
        return redirect(request.form.get("next") or "/")

    @app.template_global()
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    return app
