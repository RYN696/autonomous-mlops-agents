import numpy as np
import pytest
from mlflow import MlflowClient

from agents.retrain import explain
from agents.retrain.agent import RetrainAgent, default_months, parse_period
from agents.retrain.tuning import TUNED, tune
from common import config
from common.action_log import ActionLog
from common.events import Event, EventType
from tests.conftest import synthetic_month


class FakeBus:
    """Garde les événements publiés au lieu de les envoyer sur Redis."""

    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


def reader(corrupted=()):
    """Données synthétiques ; les mois de `corrupted` ont des durées mélangées (données cassées)."""
    def read(year, month):
        df = synthetic_month(year, month)
        if (year, month) in corrupted:
            df["duration"] = np.random.default_rng(0).permutation(df["duration"].values)
        return df
    return read


def request(period=None, **extra):
    return Event(type=EventType.RETRAIN_REQUESTED, source="orchestrateur", payload={"period": period, **extra})


@pytest.fixture
def make_agent(local_mlflow, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")  # les tests n'appellent jamais Grok
    def make(corrupted=(), cooldown=0, tuning=0):
        shown = []
        agent = RetrainAgent(bus=FakeBus(), log=ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}"),
                             read_data=reader(corrupted), reload=lambda: True,
                             list_months=lambda: [(2023, 1), (2023, 2), (2023, 3)], cooldown_minutes=cooldown,
                             tuning_trials=tuning, show=lambda text: shown.append(text) or True)
        agent.shown = shown  # messages affichés
        return agent
    return make


def aliases(version):
    return set(MlflowClient(config.MLFLOW_TRACKING_URI).get_model_version(config.MODEL_NAME, version).aliases)


def test_parse_period():
    assert parse_period("2023-02") == [(2023, 2)]
    assert parse_period({"start": "2022-11", "end": "2023-01"}) == [(2022, 11), (2022, 12), (2023, 1)]


def test_better_challenger_is_promoted_and_old_version_kept(make_agent):
    agent = make_agent(corrupted={(2023, 1)})
    first = agent.handle(request("2023-01"))  # aucun modèle en production : promu directement
    assert first["decision"] == "promoted"

    second = agent.handle(request("2023-02"))  # données propres : nettement meilleur
    assert second["decision"] == "promoted"
    assert second["challenger_rmse"] < second["production_rmse"]
    assert "production" in aliases(second["challenger_version"])
    assert "previous" in aliases(first["challenger_version"])  # pour un rollback
    assert [e.type for e in agent.bus.events][-2:] == [EventType.MODEL_REGISTERED, EventType.MODEL_PROMOTED]
    assert agent.log.latest()[0]["action"] == "model_promoted"


def test_worse_challenger_is_blocked_and_production_unchanged(make_agent):
    agent = make_agent(corrupted={(2023, 2)})
    first = agent.handle(request("2023-01"))
    second = agent.handle(request("2023-02"))  # entraîné sur des données cassées

    assert second["decision"] == "blocked"
    assert second["challenger_rmse"] > second["production_rmse"]
    assert "production" in aliases(first["challenger_version"])
    assert "production" not in aliases(second["challenger_version"])
    blocked = agent.bus.events[-1]
    assert blocked.type == EventType.PROMOTION_BLOCKED
    assert {"challenger_rmse", "production_rmse"} <= blocked.payload.keys()


def test_default_months():
    available = [(2023, 1), (2023, 2), (2023, 3)]
    assert default_months(available, 1) == [(2023, 2)]  # mars sert de test
    assert default_months(available, 2) == [(2023, 1), (2023, 2)]


def test_without_period_uses_latest_months(make_agent):
    agent = make_agent()
    agent.handle(request())
    registered = next(e for e in agent.bus.events if e.type == EventType.MODEL_REGISTERED)
    assert registered.payload["period"] == ["2023-02"]


def test_duplicates_and_cooldown_are_skipped(make_agent):
    agent = make_agent(cooldown=60)
    assert agent.handle(request("2023-01"))["decision"] == "promoted"

    assert agent.handle(request("2023-01"))["decision"] == "skipped"  # même période
    assert agent.handle(request("2023-02"))["decision"] == "skipped"  # moins de 60 min
    assert agent.log.latest()[0]["action"] == "retrain_skipped"

    validated = agent.handle(request("2023-02", approved_by="responsable"))  # validé par un humain
    assert validated["decision"] != "skipped"


# --- v4 : Optuna et explication par le LLM ---

PROMOTED = {"challenger_version": "3", "challenger_rmse": 5.1, "production_version": "2",
            "production_rmse": 5.43, "improvement": 0.061, "min_improvement": 0.02, "period": ["2023-03"]}


def test_tune_returns_tuned_params():
    params, seconds = tune(synthetic_month(2023, 1), n_trials=3)
    assert set(params) == set(TUNED)
    assert seconds > 0


def test_tuned_challenger_and_message_shown(make_agent):
    agent = make_agent(tuning=2)
    result = agent.handle(request("2023-01"))
    registered = next(e for e in agent.bus.events if e.type == EventType.MODEL_REGISTERED)
    assert set(registered.payload["params"]) == set(TUNED)

    run = MlflowClient(config.MLFLOW_TRACKING_URI).get_run(
        MlflowClient(config.MLFLOW_TRACKING_URI).get_model_version(config.MODEL_NAME, "1").run_id)
    assert run.data.params["max_depth"] == str(registered.payload["params"]["max_depth"])

    assert agent.shown == [result["message"]]
    assert result["message_source"] == "template"
    promoted = next(e for e in agent.bus.events if e.type == EventType.MODEL_PROMOTED)
    assert promoted.payload["message"] == result["message"]


def test_message_without_llm(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    text, source = explain.explain_decision("promoted", PROMOTED)
    assert source == "template"
    assert text.startswith("✅ v3 promue : 6.1 % d'erreur en moins")
    blocked = {**PROMOTED, "challenger_rmse": 5.46, "production_rmse": 5.51, "improvement": 0.009}
    assert explain.template_message("blocked", blocked).startswith("⛔ Challenger v3 refusé")


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


def fake_grok(monkeypatch, text):
    calls = []
    monkeypatch.setattr(config, "LLM_API_KEY", "cle-de-test")
    monkeypatch.setattr(explain.requests, "post", lambda url, **kw: calls.append(kw) or FakeResponse(text))
    return calls


def test_message_written_by_grok(monkeypatch):
    calls = fake_grok(monkeypatch, "✅ v3 promue : 6,1 % d'erreur en moins que v2 (RMSE 5,10 contre 5,43 min).")
    text, source = explain.explain_decision("promoted", PROMOTED)
    assert source == "llm" and text.startswith("✅ v3 promue")
    assert calls[0]["headers"]["Authorization"] == "Bearer cle-de-test"
    assert "Gain : 6.1 %" in calls[0]["json"]["messages"][1]["content"]


def test_invented_number_falls_back_to_template(monkeypatch):
    fake_grok(monkeypatch, "✅ v3 promue : 12 % d'erreur en moins.")  # 12 n'est pas dans les faits
    text, source = explain.explain_decision("promoted", PROMOTED)
    assert source == "template"
    assert text == explain.template_message("promoted", PROMOTED)


def test_grok_down_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "cle-de-test")

    def down(url, **kw):
        raise explain.requests.ConnectionError("pas de réseau")

    monkeypatch.setattr(explain.requests, "post", down)
    assert explain.explain_decision("promoted", PROMOTED)[1] == "template"

def test_months_and_received_file(make_agent):
    """Scénario 3 : « les 3 derniers mois » ; scénario 4 : entraînement sur le fichier reçu par Qualité."""
    agent = make_agent()
    agent.list_months = lambda: [(2023, 1), (2023, 2), (2023, 3), (2023, 4)]
    agent.handle(request(months=3, force=True))
    registered = [e for e in agent.bus.events if e.type == EventType.MODEL_REGISTERED][-1]
    assert registered.payload["period"] == ["2023-01", "2023-02", "2023-03"]

    received = []
    agent.read_received = lambda name: received.append(name) or synthetic_month(2023, 3)
    agent.handle(request("2023-03", data_file="incoming/recu-k1.parquet"))
    assert received == ["incoming/recu-k1.parquet"]
    started = [r for r in agent.log.latest() if r["action"] == "retrain_started"][0]
    assert started["details"]["data_file"] == "incoming/recu-k1.parquet"
