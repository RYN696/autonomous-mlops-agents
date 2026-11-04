"""Interface de la pipeline : parcours (traçabilité), résumés et pages web."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert

from agents.orchestrator.agent import Orchestrator
from agents.orchestrator.notifier import Notifier
from agents.orchestrator.rules import Rules
from agents.orchestrator.web import create_app
from common.action_log import ActionLog, action_log
from common.events import Event, EventType, Severity
from ui import traces as tr
from ui.summaries import summarize

T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)  # parcours récents


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def last(self, type_):
        return [e for e in self.events if e.type == type_][-1]


class FakeStatus:
    """Pas de Docker, MLflow ni Redis dans les tests."""

    def containers(self):
        return {"quality": {"status": "running", "name": "q", "started_at": T0.isoformat(), "restarts": 0},
                "reliability": {"status": "exited", "name": "r", "started_at": T0.isoformat(), "restarts": 2}}

    def services(self):
        return [{"name": "API de prédiction", "ok": True, "detail": "modèle v14", "warn": False}]

    def production(self):
        return {"version": "14", "rmse": 5.26, "created": T0}

    def month_cost(self):
        return {"month": "2026-11", "total": 0.3, "training": 0.29, "serving": 0.01, "budget": 20.0, "share": 0.015}


@pytest.fixture
def log(tmp_path):
    return ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}")


@pytest.fixture
def notifier(tmp_path):
    return Notifier(f"sqlite:///{tmp_path / 'notif.db'}")


def write(log, seconds, agent, act, /, **details):
    with log.engine.begin() as conn:
        conn.execute(insert(action_log).values(timestamp=T0 + timedelta(seconds=seconds), agent=agent,
                                               action=act, model="taxi-duration", details=details))


def retrain_story(log, rid="r1"):
    """Une dérive suivie jusqu'à la promotion bloquée, comme l'écrivent les agents."""
    write(log, 0, "agent_qualite", "drift_detected", reason="distribution_drift", drift_score=0.7,
          drift_threshold=0.5, prediction_drift=0.4, period="2023-02", check_id="c1")
    write(log, 1, "orchestrator", "drift_received", request_id=rid, check_id="c1",
          reason="distribution_drift", period="2023-02")
    write(log, 1, "orchestrator", "cost_requested", request_id=rid)
    write(log, 2, "cost", "cost_estimated", request_id=rid, options={"standard": {"cost": 0.3}, "spot": {"cost": 0.12}},
          month_total=0.0, budget=20.0, exceeds_budget=False)
    write(log, 2, "orchestrator", "retrain_decision", request_id=rid, action="retrain", rule="auto", estimated_cost=0.3)
    write(log, 3, "orchestrator", "retrain_requested", request_id=rid, period="2023-02")
    write(log, 4, "agent_retrain", "retrain_started", request_id=rid, period=["2023-02"])
    write(log, 20, "agent_retrain", "challenger_registered", request_id=rid, version="26", rmse=5.46, training_seconds=14)
    write(log, 21, "cost", "cost_recorded", request_id=rid, kind="training", cpu_seconds=14.0, cost=0.007)
    write(log, 22, "agent_retrain", "promotion_blocked", request_id=rid, challenger_version="26", challenger_rmse=5.46,
          production_version="14", production_rmse=5.51, improvement=0.009, min_improvement=0.02)


def incident_story(log):
    write(log, 100, "agent_fiabilite", "llm_decision", incident_id=None, phase="detection",
          reponse={"pensee": "L'API ne répond plus", "incident": True, "cause": "api_arretee"})
    write(log, 101, "agent_fiabilite", "incident_opened", incident_id="inc-1", cause="api_arretee",
          component="api", level="critical", evidence=["conteneur exited"])
    write(log, 120, "agent_fiabilite", "action:restart_api", incident_id="inc-1", action="restart_api",
          success=True, detail="conteneur mlops-api redémarré", by="llm")
    write(log, 170, "agent_fiabilite", "incident_resolved", incident_id="inc-1", duration_seconds=69,
          resume="API rétablie", judge={"score": 9})


# ---------- parcours ----------
def test_un_parcours_relie_tous_les_agents_d_une_demande(log, notifier):
    retrain_story(log)
    notifier.notify("⛔ Promotion bloquée : v26…", "r1")

    t = tr.load_trace(log.engine, "r1", notifier)
    assert t.kind == tr.RETRAIN and t.title == "Dérive 2023-02"
    assert (t.status, t.level) == ("Promotion bloquée", "warn")
    assert [a.name for a in t.agents] == ["Qualité", "Orchestrateur", "Coûts", "Retrain", "Humain"]
    assert t.steps[0].action == "drift_detected"  # retrouvé par check_id
    assert t.steps[-1].is_message  # message à l'humain (horodaté maintenant, après le journal simulé)
    assert next(s for s in t.steps if s.action == "promotion_blocked").offset == 22.0
    assert t.steps[4].offset == 2.0


def test_un_incident_garde_la_decision_qui_l_a_ouvert(log):
    incident_story(log)
    t = tr.load_trace(log.engine, "inc-1")
    assert t.actions == ["llm_decision", "incident_opened", "action:restart_api", "incident_resolved"]
    assert t.status == "Résolu en 70 s" and t.title == "api_arretee sur api"


def test_liste_des_parcours_du_plus_recent_au_plus_ancien(log):
    retrain_story(log)
    incident_story(log)
    traces = tr.recent_traces(log.engine)
    assert [t.id for t in traces] == ["inc-1", "r1"]
    assert [t.id for t in tr.recent_traces(log.engine, kind=tr.RETRAIN)] == ["r1"]


def test_etats_des_demandes_en_cours(log):
    write(log, 0, "orchestrator", "cost_requested", request_id="a")
    write(log, 0, "orchestrator", "cost_requested", request_id="b")
    write(log, 1, "orchestrator", "retrain_decision", request_id="b", action="retrain_over_budget", rule="approval")
    states = {t.id: t.status for t in tr.recent_traces(log.engine)}
    assert states == {"a": "En attente du coût", "b": "En attente d'un humain"}


def test_parcours_abandonnes_sans_suite(log):
    """Agent arrêté en cours de route : le parcours ne reste pas "en cours" pour toujours."""
    old = -3 * 3600
    write(log, old, "orchestrator", "retrain_requested", request_id="vieux")
    write(log, old, "agent_fiabilite", "incident_opened", incident_id="inc-vieux", cause="surcharge", component="api")
    write(log, old, "orchestrator", "retrain_decision", request_id="attend", action="retrain_over_budget", rule="approval")
    states = {t.id: t.status for t in tr.recent_traces(log.engine)}
    assert states == {"vieux": "Sans suite", "inc-vieux": "Sans clôture", "attend": "En attente d'un humain"}


def test_parcours_inconnu(log):
    assert tr.load_trace(log.engine, "nope") is None


def test_l_orchestrateur_ecrit_un_parcours_complet(log, notifier):
    """Bout en bout avec le vrai orchestrateur : dérive -> coût -> validation humaine -> retrain."""
    orch = Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules({"retrain_over_budget": "approval"}))
    orch.handle(Event(type=EventType.DRIFT_DETECTED, source="agent_qualite", severity=Severity.WARNING,
                      payload={"reason": "distribution_drift", "drift_score": 0.7, "check_id": "c9",
                               "period": "2023-02"}))
    rid = orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]
    orch.handle(Event(type=EventType.COST_ESTIMATED, source="agent_cost", payload={
        "request_id": rid, "exceeds_budget": True, "options": {"standard": {"cost": 0.3}},
        "month_total": 19.9, "budget": 20.0}))
    assert tr.load_trace(log.engine, rid, notifier).status == "En attente d'un humain"

    notifier.answer(rid, True, "ismail")
    orch.handle(Event(type=EventType.APPROVAL_GIVEN, source="page_web",
                      payload={"request_id": rid, "approved": True, "approved_by": "ismail"}))
    t = tr.load_trace(log.engine, rid, notifier)
    assert t.status == "Entraînement en cours"
    assert "validé par ismail" in " ".join(s.summary for s in t.steps)
    assert orch.bus.last(EventType.RETRAIN_REQUESTED).payload["period"] == "2023-02"


# ---------- résumés ----------
def test_resumes_lisibles():
    assert summarize("promotion_blocked", {"challenger_version": "26", "challenger_rmse": 5.46,
                                           "production_rmse": 5.51, "production_version": "14",
                                           "improvement": 0.009, "min_improvement": 0.02}).startswith(
        "Promotion bloquée : v26")
    assert summarize("retrain_decision", {"action": "retrain", "rule": "approval"}).startswith(
        "Règle « retrain » : validation humaine")
    assert summarize("action:rollback_model", {"action": "rollback_model", "success": False,
                                               "detail": "aucune version"}).startswith("Action rollback_model échec")


def test_resume_d_une_action_inconnue_ou_mal_formee():
    assert summarize("truc_nouveau", {"a": 1, "b": "x", "c": [1]}) == "a : 1, b : x"
    assert summarize("cost_estimated", {"options": "pas un dict"})  # ne lève pas


# ---------- pages ----------
@pytest.fixture
def client(log, notifier):
    orch = Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules({}))
    return create_app(notifier, FakeBus(), log, orch, status=FakeStatus()).test_client()


def test_toutes_les_pages_s_affichent(client, log, notifier):
    retrain_story(log)
    incident_story(log)
    notifier.ask("On réentraîne malgré le budget ?", "r1")
    for url in ["/", "/parcours", "/parcours?type=incident", "/parcours/r1", "/parcours/inc-1", "/journal",
                "/journal?agent=cost&q=training", "/agents/quality", "/agents/reliability?action=incident_opened",
                "/graph"]:
        assert client.get(url).status_code == 200, url

    home = client.get("/").get_data(as_text=True)
    assert "On réentraîne malgré le budget ?" in home and "v14" in home and "arrêté" in home
    trace = client.get("/parcours/r1").get_data(as_text=True)
    assert "Promotion bloquée" in trace and "Challenger v26 enregistré" in trace
    assert 'name="next" value="/parcours/r1"' in trace  # la question peut être validée depuis le parcours
    assert client.get("/parcours/inconnu").status_code == 404
    assert client.get("/agents/inconnu").status_code == 404


def test_journal_filtre(client, log):
    retrain_story(log)
    page = client.get("/journal?agent=agent_retrain").get_data(as_text=True)
    assert "challenger_registered" in page and "cost_estimated" not in page
    page = client.get("/journal?q=c1").get_data(as_text=True)
    assert "drift_received" in page and "promotion_blocked" not in page


def test_valider_depuis_un_parcours_revient_sur_le_parcours(client, notifier):
    notifier.ask("On réentraîne ?", "r1")
    resp = client.post("/answer", data={"request_id": "r1", "choice": "yes", "who": "omar", "next": "/parcours/r1"})
    assert resp.headers["Location"].endswith("/parcours/r1")
    assert notifier.pending() == []


# ---------- banc de test (page Scénarios) ----------
from ui.scenarios import ScenarioRunner  # noqa: E402


def fake_runner(log, outputs=None):
    outputs = outputs or {}

    def action(run):
        if run.scenario.key in outputs:
            raise outputs[run.scenario.key]
        return f"{run.scenario.key} ok"
    return ScenarioRunner(log, enabled=True, actions={k: action for k in
                                                      ("drift", "corrupted", "normal", "crash", "errors",
                                                       "latency", "bad-model", "mlflow-down", "recover")})


def test_un_scenario_est_trace_dans_le_journal_et_relie_au_parcours(log, notifier):
    runner = fake_runner(log)
    run = runner.start("drift", {"month": "3"}, wait=True)
    assert run.status == "ok" and run.params["month"] == 3 and run.check_id
    assert [r["action"] for r in log.latest()] == ["scenario_finished", "scenario_started"]

    # l'orchestrateur reçoit la dérive de ce contrôle : le parcours commence par le bouton
    log.record("orchestrator", "drift_received", "taxi-duration",
               {"request_id": "r9", "check_id": run.check_id, "period": "2023-03"})
    assert tr.trace_of_check(log.engine, run.check_id) == "r9"
    t = tr.load_trace(log.engine, "r9")
    assert t.agents[0].name == "Banc de test" and t.steps[0].summary == "Scénario lancé : Dérive des données (2023-03)"


def test_scenario_en_echec_et_desactive(log):
    runner = fake_runner(log, {"crash": SystemExit("Injection désactivée")})
    assert runner.start("crash", wait=True).status == "error"
    assert "Injection désactivée" in runner.runs[0].output
    with pytest.raises(PermissionError):
        ScenarioRunner(log, enabled=False).start("crash")


def test_page_scenarios_et_boutons(log, notifier):
    runner = fake_runner(log)
    orch = Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules({}))
    client = create_app(notifier, FakeBus(), log, orch, status=FakeStatus(), runner=runner).test_client()
    write(log, 0, "agent_retrain", "retrain_started", request_id="x", period=["2023-02"])

    page = client.get("/scenarios").get_data(as_text=True)
    assert "Mauvais modèle en production" in page and "2023-02 · déjà entraîné" in page

    assert client.post("/scenarios/run", data={"scenario": "corrupted", "month": "5"}).status_code == 302
    for _ in range(50):  # le scénario tourne dans un thread
        if runner.runs[0].status != "running":
            break
        import time; time.sleep(0.02)
    page = client.get("/scenarios").get_data(as_text=True)
    assert "Données corrompues" in page and "terminé" in page

    page = client.post("/scenarios/run", data={"scenario": "inconnu"}, follow_redirects=True).get_data(as_text=True)
    assert "inconnu" in page  # message d'erreur affiché, pas de plantage


def test_trafic_simule_demarre_et_s_arrete(log, monkeypatch):
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return {"degraded": False}

    monkeypatch.setattr("ui.scenarios.requests.post", lambda *a, **k: calls.append(1) or Resp())
    runner = fake_runner(log)
    runner.start_traffic(20)
    import time
    time.sleep(0.3)
    runner.stop_traffic()
    assert runner.traffic.ok >= 2 and not runner.traffic.running
    assert [r["action"] for r in log.latest()] == ["traffic_stopped", "traffic_started"]
