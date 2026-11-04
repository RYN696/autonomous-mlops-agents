"""Tests de l'écoute des événements et du stockage SQL de l'agent Coûts.

Utilisent le vrai common/events.py, common/config.py et agents/cost/agent.py ;
Redis, Postgres et Docker ne sont pas nécessaires.
"""
import logging
from datetime import datetime

import pytest

from agents.cost.agent import CostAgent
from agents.cost.listener import make_handler, normalize_months, with_months
from agents.cost.store import InMemoryCostStore
from common import config
from common.events import Event, EventType, Severity

NOW = datetime(2026, 10, 18, 12, 0)


class FakeBus:
    """Même interface que common.events.EventBus.publish(event)."""

    def __init__(self):
        self.sent = []

    def publish(self, event):
        self.sent.append(event)

    def types(self):
        return [e.type for e in self.sent]


class FakeActionLog:
    """Même signature que common.action_log.ActionLog.record."""

    def __init__(self):
        self.rows = []

    def record(self, agent, action, model=None, details=None):
        self.rows.append(dict(agent=agent, action=action, model=model, details=details))


def make_agent():
    bus, alog, store = FakeBus(), FakeActionLog(), InMemoryCostStore()
    agent = CostAgent(store, bus, alog, clock=lambda: NOW)
    return agent, store, bus, alog


# ---------- période libre -> nombre de mois ----------
@pytest.mark.parametrize("value,expected", [
    (3, 3), ("3", 3), ("3 derniers mois", 3), ("3m", 3),
    ("1 an", 12), ("2 years", 24),
    (None, None), (0, None), ("récemment", None), (True, None),
])
def test_normalize_months(value, expected):
    assert normalize_months(value) == expected


def test_with_months_reads_period():
    ev = Event(type=EventType.COST_ESTIMATE, source="orchestrateur",
               payload={"period": "3 derniers mois", "requested_by": "ds"})
    out = with_months(ev)
    assert out.payload["months"] == 3 and out.payload["requested_by"] == "ds"


def test_months_key_wins_over_period():
    ev = Event(type=EventType.COST_ESTIMATE, source="o", payload={"months": 2, "period": "6"})
    assert with_months(ev).payload["months"] == 2


def test_with_months_does_not_modify_the_original_event():
    ev = Event(type=EventType.COST_ESTIMATE, source="o", payload={"period": "3"})
    with_months(ev)
    assert "months" not in ev.payload


# ---------- de bout en bout, avec les vrais réglages du socle ----------
def test_estimate_under_budget_publishes_only_cost_estimated():
    agent, store, bus, alog = make_agent()
    make_handler(agent)(Event(type=EventType.COST_ESTIMATE, source="orchestrateur",
                              payload={"period": "3 derniers mois", "request_id": "r1"}))
    assert bus.types() == [EventType.COST_ESTIMATED]
    # 3 mois x 600 s x 0,0005 = 0,90 sur un budget de 20
    assert bus.sent[0].payload["options"]["standard"]["cost"] == pytest.approx(0.9)
    assert bus.sent[0].payload["request_id"] == "r1"
    assert bus.sent[0].model == config.MODEL_NAME


def test_estimate_over_budget_publishes_budget_exceeded():
    agent, store, bus, alog = make_agent()
    store.add(config.MODEL_NAME, "2026-10", "serving", 0, 19.5)
    make_handler(agent)(Event(type=EventType.COST_ESTIMATE, source="orchestrateur",
                              payload={"period": "3 derniers mois", "request_id": "r2"}))
    assert bus.types() == [EventType.COST_ESTIMATED, EventType.BUDGET_EXCEEDED]
    exceeded = bus.sent[1]
    assert exceeded.payload["request_id"] == "r2"
    assert exceeded.payload["over_by"] == pytest.approx(0.4)  # 19,5 + 0,9 - 20
    assert [r["action"] for r in alog.rows] == ["cost_estimated", "budget_exceeded"]


def test_training_completed_event_is_charged():
    agent, store, bus, alog = make_agent()
    make_handler(agent)(Event(type=EventType.TRAINING_COMPLETED, source="agent_retrain",
                              payload={"cpu_seconds": 1000, "months": 2}))
    assert store.month_total("2026-10") == pytest.approx(0.5)
    assert store.training_history(config.MODEL_NAME) == [(1000.0, 2)]
    assert alog.rows[-1]["action"] == "cost_recorded"


def test_own_events_are_not_reprocessed():
    agent, store, bus, alog = make_agent()
    make_handler(agent)(Event(type=EventType.BUDGET_EXCEEDED, source="agent_cost", payload={}))
    assert bus.sent == [] and alog.rows == []


def test_handler_survives_a_bad_event(caplog):
    agent, store, bus, alog = make_agent()
    handler = make_handler(agent)
    bad = Event(type=EventType.TRAINING_COMPLETED, source="agent_retrain", payload={})
    with caplog.at_level(logging.ERROR, logger="agent_couts"):
        handler(bad)  # ne doit pas lever
    assert "événement ignoré" in caplog.text
    handler(Event(type=EventType.TRAINING_COMPLETED, source="agent_retrain",
                  payload={"cpu_seconds": 200, "months": 1}))
    assert store.month_total("2026-10") == pytest.approx(0.1)


def test_published_events_survive_json_round_trip():
    agent, store, bus, alog = make_agent()
    agent.recommend_server_size("taxi-duration", 0.5, "L")
    back = Event.from_json(bus.sent[0].to_json())
    assert back.type == EventType.COST_RECOMMENDATION
    assert back.payload["action"] == "downsize"


def test_budget_exceeded_severity_is_currently_info():
    """Documente le comportement actuel : l'agent ne fixe pas la gravité.
    Si l'équipe décide d'utiliser WARNING, changer ce test avec agent._publish."""
    agent, store, bus, alog = make_agent()
    store.add(config.MODEL_NAME, "2026-10", "serving", 0, 19.5)
    make_handler(agent)(Event(type=EventType.COST_ESTIMATE, source="o", payload={"months": 3}))
    assert bus.sent[1].type == EventType.BUDGET_EXCEEDED
    assert bus.sent[1].severity == Severity.INFO


# ---------- stockage SQL (nécessite sqlalchemy ; SQLite en mémoire) ----------
def test_sql_store_matches_in_memory_behaviour():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine
    from agents.cost.sql_store import SqlCostStore

    engine = create_engine("sqlite://")
    sql, mem = SqlCostStore(engine=engine), InMemoryCostStore()
    for s in (sql, mem):
        s.add("taxi", "2026-10", "training", 100, 1.0, 1)
        s.add("taxi", "2026-10", "serving", 50, 0.5)
        s.add("fraud", "2026-10", "training", 10, 0.1, 1)
        s.add("taxi", "2026-09", "training", 999, 9.9, 1)
    assert sql.month_total("2026-10") == pytest.approx(mem.month_total("2026-10"))
    assert sql.by_model("2026-10") == pytest.approx(mem.by_model("2026-10"))
    assert sorted(sql.training_history("taxi")) == sorted(mem.training_history("taxi"))
    assert sql.month_total("2030-01") == 0.0
