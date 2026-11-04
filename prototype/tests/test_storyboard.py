"""Les 4 scénarios du storyboard : la mécanique de chaque agent, puis le suivi case par case dans l'interface."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import insert

from agents.cost.agent import CostAgent
from agents.cost.store import InMemoryCostStore
from agents.orchestrator import chat
from agents.orchestrator.agent import Orchestrator
from agents.orchestrator.notifier import Notifier
from agents.orchestrator.rules import Rules
from agents.orchestrator.web import create_app
from agents.quality.__main__ import apply_scenario
from agents.quality.checker import build_event
from agents.quality.validation import validate_current_data
from agents.reliability import guards
from agents.reliability.signals import Signals
from common import config
from common.action_log import ActionLog, action_log
from common.events import Event, EventType, Severity
from common.llm import LLMError
from pipeline.data import clean
from ui.scenarios import ScenarioRunner
from ui.storyboard import DONE, VARIANT, StoryBoard

RULES = {"retrain": "auto", "retrain_over_budget": "approval"}


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]

    def last(self, type_):
        return [e for e in self.events if e.type == type_][-1]


class FakeLLM:
    def __init__(self, answer=None):
        self.answer = answer

    def chat_json(self, system, user, schema=None):
        if self.answer is None:
            raise LLMError("Ollama injoignable")
        return self.answer


def ev(type_, source="test", **payload):
    return Event(type=type_, source=source, payload=payload)


@pytest.fixture
def log(tmp_path):
    return ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}")


@pytest.fixture
def notifier(tmp_path):
    return Notifier(f"sqlite:///{tmp_path / 'notif.db'}")


@pytest.fixture
def orch(log, notifier):
    return Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules(RULES),
                        chat=chat.ChatParser(FakeLLM()))  # LLM indisponible : règles


# ---------------------------------------------------------------- scénario 3 : chat et budget
def test_la_phrase_du_storyboard_est_comprise_sans_llm():
    cmd = chat.parse_rules("@agents retrain le modèle taxi sur les 3 derniers mois")
    assert (cmd.action, cmd.model, cmd.months, cmd.ok) == ("retrain", config.MODEL_NAME, 3, True)
    assert cmd.period_label == "les 3 derniers mois"
    assert chat.parse_rules("Réentraîne sur les trois derniers mois").months == 3
    assert not chat.parse_rules("quelle heure est-il ?").ok
    assert "inconnu" in chat.parse_rules("retrain le modèle fraude").problem
    assert "refusée" in chat.parse_rules("retrain le modèle taxi sur 24 mois").problem


def test_le_llm_comprend_mais_le_code_verifie():
    parser = chat.ChatParser(FakeLLM({"action": "retrain", "modele": "taxi", "mois": 3}))
    cmd = parser.parse("@agents retrain le modèle taxi sur les 3 derniers mois")
    assert cmd.understood_by == "llm" and cmd.months == 3 and cmd.ok
    # le LLM se trompe sur le nombre écrit dans la phrase : les règles prennent le relais
    parser = chat.ChatParser(FakeLLM({"action": "retrain", "modele": "taxi", "mois": 12}))
    cmd = parser.parse("retrain le modèle taxi sur les 3 derniers mois")
    assert cmd.understood_by == "règles" and cmd.months == 3
    # action inventée par le LLM : rejetée
    assert chat.ChatParser(FakeLLM({"action": "delete", "modele": "", "mois": 0})).parse("retrain").understood_by == "règles"


def test_commande_chat_budget_depasse_puis_validation(orch, notifier):
    rid = orch.on_chat("@agents retrain le modèle taxi sur les 3 derniers mois", "ines", request_id="c1")
    assert rid == "c1"
    asked = orch.bus.last(EventType.COST_ESTIMATE).payload
    assert asked == {"request_id": "c1", "requested_by": "orchestrator", "months": 3}

    orch.handle(ev(EventType.COST_ESTIMATED, "cost", request_id="c1", exceeds_budget=True,
                   options={"standard": {"cost": 0.06}}, month_total=0.19, budget=0.22))
    question = notifier.pending()[0]["text"]
    assert "demandé par ines" in question and "les 3 derniers mois" in question
    assert "au-dessus du budget de 0.0300 €" in question

    orch.handle(ev(EventType.APPROVAL_GIVEN, "page_web", request_id="c1", approved=True, approved_by="sarah"))
    retrain = orch.bus.last(EventType.RETRAIN_REQUESTED).payload
    assert retrain["months"] == 3 and retrain["requested_by"] == "ines" and retrain["approved_by"] == "sarah"
    assert retrain["force"] is True and "period" not in retrain
    assert [r["action"] for r in orch.log.latest()][::-1][:2] == ["chat_received", "chat_understood"]


def test_commande_chat_incomprise_n_est_pas_lancee(orch, notifier):
    orch.on_chat("supprime la production", "ines", request_id="c2")
    assert EventType.COST_ESTIMATE not in orch.bus.types()
    assert "Je n'ai pas lancé" in notifier.latest()[0]["text"]


def test_le_budget_regle_en_direct_est_lu_par_l_agent_couts():
    budget = {"value": 0.25}
    agent = CostAgent(InMemoryCostStore(), FakeBus(), ActionLog(url="sqlite://"), budget=lambda: budget["value"])
    assert agent.handle_cost_estimate({"model": "taxi-duration", "months": 3})["budget"] == 0.25
    budget["value"] = 20.0
    assert agent.handle_cost_estimate({"model": "taxi-duration", "months": 3})["budget"] == 20.0


# ---------------------------------------------------------------- scénario 4 : le mauvais challenger
def raw_month(n=1000):
    t = pd.Timestamp("2023-03-01 10:00")
    return pd.DataFrame({
        "lpep_pickup_datetime": [t] * n, "lpep_dropoff_datetime": [t + pd.Timedelta(minutes=12)] * n,
        "passenger_count": [1] * n, "trip_distance": [3.0] * n,
        "PULocationID": [43] * n, "DOLocationID": [151] * n})


def test_donnees_suspectes_sous_le_seuil_corrompu():
    raw = raw_month()
    bad_raw, bad_clean = apply_scenario("suspect", raw, clean(raw))
    v = validate_current_data(bad_raw)
    assert not v.is_critical and v.impossible_share >= config.SUSPECT_SHARE
    event = build_event(v, drift_score=0.75, num_drifted_columns=3)
    assert event.payload["reason"] == "distribution_drift" and event.payload["suspect"] is True
    assert (bad_clean.trip_distance == 9999).any()  # les distances impossibles iraient jusqu'à l'entraînement
    # des données réelles propres ne sont pas suspectes
    assert build_event(validate_current_data(raw), 0.75, 3).payload["suspect"] is False


def test_challenger_pire_sur_donnees_suspectes_alerte_l_humain(orch, notifier):
    orch.handle(ev(EventType.DRIFT_DETECTED, "agent_qualite", reason="distribution_drift", drift_score=0.75,
                   period="2023-03", suspect=True, impossible_share=0.025, check_id="k1",
                   data_file="incoming/green_tripdata_2023-03_recu-k1.parquet"))
    rid = orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]
    orch.handle(ev(EventType.COST_ESTIMATED, "cost", request_id=rid, exceeds_budget=False,
                   options={"standard": {"cost": 0.02}}, month_total=0.2, budget=20.0))
    retrain = orch.bus.last(EventType.RETRAIN_REQUESTED).payload
    assert retrain["data_file"] == "incoming/green_tripdata_2023-03_recu-k1.parquet"

    orch.handle(ev(EventType.PROMOTION_BLOCKED, "agent_retrain", request_id=rid, challenger_version="5",
                   challenger_rmse=6.5, production_version="4", production_rmse=4.9, improvement=-0.33,
                   min_improvement=0.02))
    alert = notifier.latest()[0]["text"]
    assert "Challenger v5 refusé" in alert and "v4, qui reste en production" in alert
    assert "vérification humaine requise" in alert
    assert orch.log.latest()[0]["action"] == "suspect_data_alert"


def test_challenger_un_peu_moins_bon_sur_donnees_saines_sans_alerte(orch, notifier):
    orch.handle(ev(EventType.DRIFT_DETECTED, "agent_qualite", reason="distribution_drift", drift_score=0.75))
    rid = orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]
    orch.handle(ev(EventType.COST_ESTIMATED, "cost", request_id=rid, exceeds_budget=False,
                   options={"standard": {"cost": 0.02}}, month_total=0.2, budget=20.0))
    orch.handle(ev(EventType.PROMOTION_BLOCKED, "agent_retrain", request_id=rid, challenger_version="5",
                   challenger_rmse=5.0, production_version="4", production_rmse=4.9, improvement=-0.02,
                   min_improvement=0.02))
    assert "Promotion bloquée" in notifier.latest()[0]["text"]


# ---------------------------------------------------------------- scénario 2 : panne du service
OK = {"api_joignable": True, "erreurs_pct": 0.0, "sondes_echec_pct": 0.0, "latence_p95_s": 0.02, "modele_charge": True}
ERRORS = {**OK, "erreurs_pct": 45.0}
CHANGE = {"de": 2, "vers": 3, "at": 100, "origine": "externe (nouvelle promotion)", "sain_avant": True}


def test_version_saine_au_calme_le_rollback_attend_le_redemarrage():
    # la v3 a d'abord fonctionné : les erreurs au pic de trafic ne la désignent pas tout de suite
    assert guards.bad_model({**CHANGE, "sain_apres": True}, ERRORS, Signals(), now=400) is None
    assert guards.restart_did_not_help(CHANGE, [], ERRORS, Signals(), now=400) is None
    restarted = [{"action": "restart_api", "success": True}]
    reason = guards.restart_did_not_help(CHANGE, restarted, ERRORS, Signals(), now=400)
    assert "redémarrage n'a pas suffi" in reason and "v3" in reason and "v2" in reason
    assert guards.restart_did_not_help(CHANGE, restarted, OK, Signals(), now=400) is None
    assert guards.restart_did_not_help(CHANGE, restarted, ERRORS, Signals(mlflow_ok=False), now=400) is None
    assert guards.restart_did_not_help({**CHANGE, "origine": "rollback_par_l_agent"}, restarted, ERRORS,
                                       Signals(), now=400) is None


def test_modele_defaillant_plante_seulement_sur_les_longs_trajets():
    from scripts.faulty_model import LongTripBug

    class Model:
        def predict(self, X):
            return [10.0 for _ in X]

    model = LongTripBug(Model())
    assert model.predict([{"PU_DO": "43_151", "trip_distance": 1.8}]) == [10.0]  # sonde de Fiabilité
    with pytest.raises(ValueError):
        model.predict([{"PU_DO": "1_2", "trip_distance": 12.0}])


# ---------------------------------------------------------------- interface : suivi case par case
T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=3)


def write(log, seconds, agent, act, /, **details):
    with log.engine.begin() as conn:
        conn.execute(insert(action_log).values(timestamp=T0 + timedelta(seconds=seconds), agent=agent,
                                               action=act, model="taxi-duration", details=details))


class FakeStatus:
    def containers(self):
        return {}

    def services(self):
        return []

    def production(self):
        return {"version": "3", "rmse": 5.2, "created": T0}

    def month_cost(self):
        return {"month": "2026-11", "total": 0.2, "training": 0.1, "serving": 0.1, "budget": 20.0, "share": 0.01}


def board_for(log, notifier, orch=None):
    runner = ScenarioRunner(log, enabled=True, actions={k: (lambda run: "ok") for k in
                                                        ("drift", "suspect", "flaky-model")})
    return StoryBoard(runner, log, notifier, orch, FakeStatus(), sleep=lambda s: None)


def test_scenario_derive_suivi_case_par_case(log, notifier, monkeypatch):
    monkeypatch.setattr("common.budget.set_budget", lambda value, client=None: value)
    board = board_for(log, notifier)
    run = board.start("derive", {"month": "3"}, wait=True)
    assert run.facts["budget"] == 20.0 and run.check_id and run.facts["production"] == "3"
    states = [p["state"] for p in board.progress(run)]
    assert states == [DONE, "current", "pending", "pending", "pending", "pending"]

    c = run.check_id
    write(log, 1, "agent_qualite", "drift_detected", check_id=c, drift_score=0.75, drift_threshold=0.5,
          prediction_drift=0.3, period="2023-03", explanation="Les trajets sont plus longs.")
    write(log, 2, "orchestrator", "drift_received", request_id="r1", check_id=c, period="2023-03")
    write(log, 3, "orchestrator", "retrain_decision", request_id="r1", rule="auto", estimated_cost=0.02, budget=20.0)
    write(log, 9, "agent_retrain", "challenger_registered", request_id="r1", version="4", rmse=5.0,
          period=["2023-03"])
    write(log, 10, "agent_retrain", "model_promoted", request_id="r1", challenger_version="4", challenger_rmse=5.0,
          production_version="3", production_rmse=5.3, improvement=0.057, min_improvement=0.02)
    notifier.notify("✅ Modèle v4 promu : RMSE 5.00 contre 5.30 pour v3.", "r1")

    progress = board.progress(run)
    assert [p["state"] for p in progress] == [DONE] * 6 and board.finished(run)
    assert "plus longs" in progress[2]["detail"] and "automatique" in progress[2]["detail"]
    assert run.facts["rollback"] == {"target": "3", "current": "4"}  # bouton « Revenir à v3 »

    client = create_app(notifier, FakeBus(), log, status=FakeStatus(), runner=board.runner,
                        board=board).test_client()
    page = client.get(f"/storyboard/{run.id}").get_data(as_text=True)
    assert "Un seul message, déjà résolu" in page and "Revenir à v3" in page and "Scénario 1" in page


def test_scenario_promotion_bloquee_affichee_en_variante(log, notifier, monkeypatch):
    monkeypatch.setattr("common.budget.set_budget", lambda value, client=None: value)
    board = board_for(log, notifier)
    run = board.start("derive", {"month": "4"}, wait=True)
    c = run.check_id
    write(log, 1, "agent_qualite", "drift_detected", check_id=c, drift_score=0.75, period="2023-04")
    write(log, 2, "orchestrator", "drift_received", request_id="r2", check_id=c)
    write(log, 3, "orchestrator", "retrain_decision", request_id="r2", rule="auto")
    write(log, 9, "agent_retrain", "challenger_registered", request_id="r2", version="4", rmse=5.4)
    write(log, 10, "agent_retrain", "promotion_blocked", request_id="r2", challenger_version="4",
          challenger_rmse=5.4, production_version="3", production_rmse=5.3, improvement=-0.02, min_improvement=0.02)
    assert board.progress(run)[4]["state"] == VARIANT


def test_scenario_chat_lance_la_phrase_avec_un_budget_serre(log, notifier, monkeypatch):
    budgets = []
    monkeypatch.setattr("common.budget.set_budget", lambda value, client=None: budgets.append(value) or value)
    orch = Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules(RULES),
                        chat=chat.ChatParser(FakeLLM()))
    board = board_for(log, notifier, orch)
    monkeypatch.setattr(board, "month_figures", lambda: (0.2, 0.02))
    run = board.start("chat", {"who": "ines"}, wait=True)
    assert budgets == [pytest.approx(0.23)]  # un mois passe (0.22), trois mois non (0.26)
    assert orch.bus.last(EventType.COST_ESTIMATE).payload["months"] == 3
    progress = board.progress(run)
    assert [p["state"] for p in progress][:3] == [DONE, DONE, "current"]
    assert "les 3 derniers mois" in progress[1]["detail"] and "ines" in progress[1]["detail"]


def test_pages_storyboard_chat_et_donnees(log, notifier, tmp_path, monkeypatch):
    orch = Orchestrator(bus=FakeBus(), log=log, notifier=notifier, rules=Rules(RULES),
                        chat=chat.ChatParser(FakeLLM()))
    client = create_app(notifier, FakeBus(), log, orch, status=FakeStatus(),
                        runner=ScenarioRunner(log, enabled=True)).test_client()
    page = client.get("/storyboard").get_data(as_text=True)
    for title in ("Dérive des données", "Panne du service", "Commande par chat et budget", "Le mauvais challenger"):
        assert title in page
    assert client.get("/storyboard/inconnu").status_code == 404

    orch.on_chat("@agents retrain le modèle taxi sur les 3 derniers mois", "ines", request_id="c9")
    page = client.get("/chat").get_data(as_text=True)
    assert "retrain le modèle taxi" in page and "les 3 derniers mois" in page

    monkeypatch.setattr("pipeline.data.DATA_DIR", tmp_path)
    (tmp_path / "incoming").mkdir()
    bad_raw, _ = apply_scenario("suspect", raw_month(), clean(raw_month()))
    bad_raw.to_parquet(tmp_path / "incoming" / "f.parquet")
    page = client.get("/donnees?fichier=incoming/f.parquet").get_data(as_text=True)
    assert "Durées négatives" in page and "bad-row" in page
    assert "consultables" in client.get("/donnees?fichier=../.env").get_data(as_text=True)


def test_premier_essai_redemarrer_plutot_que_masquer_les_erreurs():
    reason = guards.restart_first(CHANGE, [], ERRORS, Signals(), now=400)
    assert "premier essai" in reason and "v3" in reason
    assert guards.restart_first(CHANGE, [{"action": "degraded_mode", "success": True}], ERRORS, Signals(), 400) is None
    assert guards.restart_first(CHANGE, [], OK, Signals(), now=400) is None
    assert guards.restart_first(None, [], ERRORS, Signals(), now=400) is None   # pas de promotion récente : le LLM choisit
    assert guards.restart_first(CHANGE, [], ERRORS, Signals(mlflow_ok=False), now=400) is None
