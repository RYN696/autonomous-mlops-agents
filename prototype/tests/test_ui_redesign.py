"""Nouvelle interface : « Qui a décidé ? », carte animée et graphiques du rapport de coûts."""
from datetime import datetime, timezone

import pytest

from agents.cost.sql_store import SqlCostStore
from common.action_log import ActionLog
from ui import cost_charts, decisions
from ui.storyboard import actor_icon

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def log(tmp_path):
    url = f"sqlite:///{tmp_path / 'ui.db'}"
    SqlCostStore(url)
    return ActionLog(url)


def test_origine_des_decisions():
    assert decisions.origin("humain", "approval_answered", {}) == "humain"
    assert decisions.origin("agent_retrain", "promotion_blocked", {}) == "regle"
    assert decisions.origin("agent_fiabilite", "action:rollback_model", {"by": "regle:mauvais_modele"}) == "regle"
    assert decisions.origin("agent_fiabilite", "action:reload_model", {"by": "llm"}) == "llm"
    assert decisions.origin("orchestrator", "chat_understood", {"understood_by": "llm"}) == "llm"
    assert decisions.origin("cost", "cost_recorded", {}) is None          # une mesure, pas une décision


def test_qui_a_decide(log):
    log.record("agent_retrain", "promotion_blocked", "taxi", {"rmse": 6.1})
    log.record("agent_fiabilite", "action:restart_api", "taxi", {"by": "llm"})
    log.record("agent_fiabilite", "judge_rejected", "taxi", {})
    log.record("humain", "approval_answered", "taxi", {"approved": True, "by": "ismail"})
    log.record("cost", "cost_recorded", "taxi", {"kind": "serving"})
    d = decisions.who_decides(log.engine)
    counts = {p["key"]: p["count"] for p in d["parts"]}
    assert counts == {"regle": 1, "llm": 1, "humain": 1} and d["total"] == 3 and d["guards"] == 1
    assert next(p for p in d["parts"] if p["key"] == "humain")["latest"]["summary"].startswith("Validé par ismail")


def test_graphiques_du_mois(log):
    store = SqlCostStore(engine=log.engine)
    store.add("taxi", "2026-11", "training", 100, 0.05, 1)
    store.add("taxi", "2026-11", "serving", 10, 0.005)
    log.record("cost", "cost_recorded", "taxi", {"kind": "training", "cost": 0.05, "cpu_seconds": 100, "request_id": "r1"})
    log.record("cost", "cost_recorded", "taxi", {"kind": "training", "cost": 0.5, "cpu_seconds": 1000, "request_id": "r1"})
    log.record("cost", "cost_anomaly", "taxi", {"kind": "training", "cost": 0.5, "ratio": 10.0, "request_id": "r1"})
    g = cost_charts.charts(log.engine, budget=1.0, projection=0.3, now=datetime.now(timezone.utc))
    c = g["cumul"]
    assert c["points"][-1]["value"] == pytest.approx(0.055)
    assert c["top"] >= 1.0 and c["budget_y"] is not None and c["projection"]["value"] == 0.3
    # seul l'entraînement en pic est marqué, pas l'autre de la même demande
    assert [b["spike"] for b in g["trainings"]["bars"]] == [False, True]
    assert g["trainings"]["bars"][1]["ratio"] == 10.0


def test_borne_ronde_des_axes():
    assert cost_charts._nice(1.7) == 2 and cost_charts._nice(21) == 25 and cost_charts._nice(0.012) == 0.02


def test_icones_des_personnages():
    assert actor_icon("Agent Coûts")[1] == "a-cost" and actor_icon("inconnu")[1] == "a-other"


def test_rapports_qualite_et_retrain():
    from agents.quality.report import build_report as quality
    from agents.retrain.report import build_report as retrain

    q = quality("drift_detected", {"reason": "distribution_drift", "period": "2023-03", "row_count": 70000,
                                   "drift_score": 0.75, "drift_threshold": 0.5, "prediction_drift": 0.4,
                                   "column_drift_threshold": 0.1, "invalid_share": 0.01, "invalid_rows": 700,
                                   "num_drifted_columns": 3, "check_id": "c1", "explanation": "x"})
    assert q["title"] == "Dérive détectée · 2023-03" and q["level"] == "warn"
    assert [h["alert"] for h in q["headline"][1:]] == [True, True, False]
    assert quality("drift_detected", {"reason": "corrupted_input_data"})["level"] == "bad"

    r = retrain("blocked", {"challenger_version": 37, "challenger_rmse": 6.04, "production_version": "27",
                            "production_rmse": 5.33, "improvement": -0.132, "min_improvement": 0.02,
                            "period": ["2023-03"], "request_id": "r1", "training_seconds": 30,
                            "tuning_seconds": 330, "tuning_trials": 20, "rows_train": 70000,
                            "message": "⛔ refusé", "message_source": "template"})
    assert r["title"] == "Promotion bloquée · v37" and r["headline"][2]["value"] == "-13.2 %"
    assert r["headline"][3]["value"] == "6.0 min" and r["request_id"] == "r1"
    assert retrain("skipped", {"reason": "doublon", "period": ["2023-03"]})["level"] == "muted"


def test_page_rapports(tmp_path):
    from agents.orchestrator.notifier import Notifier
    from agents.orchestrator.web import create_app
    from agents.retrain.report import build_report as retrain
    from tests.test_orchestrator import FakeBus

    log = ActionLog(f"sqlite:///{tmp_path / 'r.db'}")
    log.record("agent_retrain", "retrain_report", "taxi", retrain("skipped", {"reason": "doublon", "period": ["2023-03"]}))
    log.record("agent_qualite", "quality_report", "taxi", {"title": "Données conformes · 2023-03", "level": "ok",
                                                           "verdict": "rien", "sections": [], "next": "-"})
    client = create_app(Notifier(f"sqlite:///{tmp_path / 'n.db'}"), FakeBus(), log).test_client()
    page = client.get("/rapports").get_data(as_text=True)
    assert "Réentraînement ignoré" in page and "Données conformes" not in page   # conformes masqués
    assert "Données conformes" in client.get("/rapports?tous=1").get_data(as_text=True)
