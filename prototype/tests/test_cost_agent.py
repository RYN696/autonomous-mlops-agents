"""Tests de l'agent Coûts. Aucun Docker, Redis ni Postgres nécessaire."""
from datetime import datetime, timedelta

import pytest

from agents.cost import calculator, pricing, recommender
from agents.cost.agent import CostAgent
from agents.cost.collector import FakeCollector
from agents.cost.settings import CostConfig
from agents.cost.store import InMemoryCostStore
from common.events import Event, EventType

NOW = datetime(2026, 10, 12, 14, 0)


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def types(self):
        return [event.type.value for event in self.events]


class FakeLog:
    def __init__(self):
        self.rows = []

    def record(self, agent, action, model=None, details=None):
        self.rows.append((agent, action, details))


def make_agent(budget=10.0, collector=None):
    cfg = CostConfig(cost_per_cpu_second=0.01, spot_discount=0.6,
                     monthly_budget=budget,
                     default_cpu_seconds_per_month_of_data=100.0)
    bus, log, store = FakeBus(), FakeLog(), InMemoryCostStore()
    agent = CostAgent(store, bus, log, cfg, collector=collector, clock=lambda: NOW)
    return agent, store, bus, log, cfg


def send(agent, event_type, payload):
    model = payload.get("model", "taxi")
    agent.on_event(Event(
        type=EventType(event_type),
        source="test",
        model=model,
        payload={key: value for key, value in payload.items() if key != "model"},
    ))


# ---------- calculator ----------
def test_cost_is_cpu_seconds_times_tariff():
    assert calculator.cost(200, 0.01) == pytest.approx(2.0)


def test_cost_rejects_negative_values():
    with pytest.raises(ValueError):
        calculator.cost(-1, 0.01)
    with pytest.raises(ValueError):
        calculator.cost(1, -0.01)


def test_month_key():
    assert calculator.month_key(datetime(2026, 3, 5)) == "2026-03"


def test_would_exceed_true_when_total_plus_estimate_over_budget():
    assert calculator.would_exceed(8.0, 2.5, 10.0) is True


def test_would_exceed_false_at_exact_budget():
    assert calculator.would_exceed(8.0, 2.0, 10.0) is False


def test_estimate_uses_history_scaled_by_months():
    # 300 s pour 3 mois = 100 s/mois ; on demande 6 mois -> 600 s
    assert calculator.estimate_cpu_seconds([(300, 3)], 6, 999) == pytest.approx(600)


def test_estimate_falls_back_to_default_without_history():
    assert calculator.estimate_cpu_seconds([], 3, 50.0) == pytest.approx(150)


def test_estimate_defaults_to_one_month_when_period_missing():
    assert calculator.estimate_cpu_seconds([], None, 50.0) == pytest.approx(50)


# ---------- pricing ----------
def test_spot_cheaper_than_standard():
    opts = pricing.price_options(100, CostConfig(cost_per_cpu_second=0.01, spot_discount=0.6))
    assert opts["standard"]["cost"] == pytest.approx(1.0)
    assert opts["spot"]["cost"] == pytest.approx(0.4)
    assert opts["spot"]["cost"] < opts["standard"]["cost"]
    assert opts["spot"]["saving"] == pytest.approx(0.6)


# ---------- recommender ----------
def test_size_recommendation_downsizes_when_cpu_underused():
    rec = recommender.recommend_size(0.5, "L", CostConfig())
    assert rec["action"] == "downsize"
    assert rec["recommended"] == "S"
    assert rec["monthly_saving"] > 0


def test_size_recommendation_keeps_when_right_sized():
    rec = recommender.recommend_size(1.4, "M", CostConfig())  # 1.4*1.3 = 1.82 <= 2
    assert rec["action"] == "keep"
    assert rec["recommended"] == "M"


def test_size_recommendation_upsizes_when_overloaded():
    rec = recommender.recommend_size(1.9, "S", CostConfig())
    assert rec["action"] == "upsize"
    assert rec["monthly_saving"] < 0


def test_idle_detection_after_threshold():
    cfg = CostConfig(idle_minutes=30)
    assert recommender.is_idle(NOW - timedelta(minutes=31), NOW, cfg) is True
    assert recommender.is_idle(NOW - timedelta(minutes=5), NOW, cfg) is False
    assert recommender.is_idle(None, NOW, cfg) is True


def test_scale_to_zero_saving():
    rec = recommender.scale_to_zero_saving(10, "M", CostConfig())
    assert rec["estimated_saving"] == pytest.approx(1.0)  # 10 h x 0.10


# ---------- store ----------
def test_month_total_accumulates_per_model():
    s = InMemoryCostStore()
    s.add("taxi", "2026-10", "training", 100, 1.0, 1)
    s.add("taxi", "2026-10", "serving", 50, 0.5)
    s.add("fraud", "2026-10", "training", 10, 0.1, 1)
    s.add("taxi", "2026-09", "training", 999, 9.9, 1)
    assert s.month_total("2026-10") == pytest.approx(1.6)
    assert s.by_model("2026-10") == {"taxi": pytest.approx(1.5), "fraud": pytest.approx(0.1)}
    assert s.training_history("taxi") == [(100, 1), (999, 1)]


# ---------- agent ----------
def test_cost_estimate_under_budget_does_not_publish_budget_exceeded():
    agent, store, bus, log, cfg = make_agent(budget=10.0)
    send(agent, "cost_estimate", {"model": "taxi", "months": 3, "requested_by": "ds"})
    assert bus.types() == ["cost_estimated"]
    answer = bus.events[0].payload
    assert answer["exceeds_budget"] is False
    assert answer["options"]["spot"]["cost"] < answer["options"]["standard"]["cost"]


def test_budget_exceeded_event_published():
    agent, store, bus, log, cfg = make_agent(budget=10.0)
    store.add("taxi", "2026-10", "training", 900, 9.0, 3)  # déjà 9 dépensés
    send(agent, "cost_estimate", {"model": "taxi", "months": 3, "request_id": "r1"})
    # historique 900 s / 3 mois = 300 s/mois -> 3 mois = 900 s -> 9.0 d'estimation
    assert bus.types() == ["cost_estimated", "budget_exceeded"]
    exceeded = bus.events[1].payload
    assert exceeded["request_id"] == "r1"
    assert exceeded["over_by"] == pytest.approx(8.0)
    assert ("cost", "budget_exceeded") in [(a, b) for a, b, _ in log.rows]


def test_estimate_at_exact_budget_is_not_exceeded():
    agent, store, bus, log, cfg = make_agent(budget=10.0)
    store.add("taxi", "2026-10", "training", 0, 9.0)  # total 9.0
    # défaut 100 s/mois x 1 mois x 0.01 = 1.0 -> total 10.0 = budget
    send(agent, "cost_estimate", {"model": "taxi", "months": 1})
    assert "budget_exceeded" not in bus.types()


def test_other_months_do_not_count_toward_budget():
    agent, store, bus, log, cfg = make_agent(budget=10.0)
    store.add("taxi", "2026-09", "serving", 5000, 50.0)  # mois précédent
    send(agent, "cost_estimate", {"model": "taxi", "months": 1})
    assert "budget_exceeded" not in bus.types()


def test_training_completed_is_recorded_with_cost():
    agent, store, bus, log, cfg = make_agent()
    send(agent, "training_completed", {"model": "taxi", "cpu_seconds": 200, "months": 3})
    assert store.month_total("2026-10") == pytest.approx(2.0)
    assert log.rows[-1][1] == "cost_recorded"


def test_serving_sample_charges_cpu_delta_only():
    collector = FakeCollector({"api": 100.0})
    agent, store, bus, log, cfg = make_agent(collector=collector)
    assert agent.record_serving_sample("taxi", "api") == 0.0  # 1er appel
    collector.values["api"] = 160.0
    assert agent.record_serving_sample("taxi", "api") == pytest.approx(0.6)
    collector.values["api"] = 10.0  # conteneur redémarré : pas de coût négatif
    assert agent.record_serving_sample("taxi", "api") == 0.0
    assert store.month_total("2026-10") == pytest.approx(0.6)


def test_serving_sample_requires_a_collector():
    agent, *_ = make_agent(collector=None)
    with pytest.raises(RuntimeError):
        agent.record_serving_sample("taxi", "api")


def test_recommendations_are_published_and_logged():
    agent, store, bus, log, cfg = make_agent()
    rec = agent.recommend_server_size("taxi", 0.4, "L")
    assert rec["action"] == "downsize"
    assert bus.types() == ["cost_recommendation"]
    assert log.rows[-1][1] == "cost_recommendation"


def test_idle_check_publishes_only_when_idle():
    agent, store, bus, log, cfg = make_agent()
    assert agent.check_idle("taxi", NOW - timedelta(minutes=1), "M", 2) is None
    assert bus.events == []
    rec = agent.check_idle("taxi", NOW - timedelta(hours=3), "M", 2)
    assert rec["action"] == "scale_to_zero"
    assert bus.types() == ["cost_recommendation"]


def test_unknown_events_are_ignored():
    agent, store, bus, log, cfg = make_agent()
    send(agent, "drift_detected", {"model": "taxi"})
    assert bus.events == [] and log.rows == []


from agents.cost.service import CostService, user_requests


# ---------- service ----------
class Clock:
    """Même heure pour le service (secondes) et l'agent (datetime)."""

    def __init__(self, start=NOW.timestamp()):
        self.t = start

    def seconds(self):
        return self.t

    def datetime(self):
        return datetime.fromtimestamp(self.t)


def make_service(cpu=0.0, requests=0.0, idle_minutes=30.0, size="S"):
    clock, collector, usage = Clock(), FakeCollector({"api": cpu}), {"count": requests}
    cfg = CostConfig(cost_per_cpu_second=0.01, monthly_budget=10.0, idle_minutes=idle_minutes,
                     size_headroom=0.3, hours_per_month=730.0)
    bus, log = FakeBus(), FakeLog()
    agent = CostAgent(InMemoryCostStore(), bus, log, cfg, collector=collector, clock=clock.datetime)
    service = CostService(agent=agent, bus=bus, interval_seconds=60, time_func=clock.seconds,
                          api_usage=lambda: usage["count"], model="taxi", container="api", server_size=size)
    return service, clock, collector, usage, bus, log


def recommendations(bus):
    return [e.payload["action"] for e in bus.events if e.type == EventType.COST_RECOMMENDATION]


def test_run_tick_enregistre_le_cout_de_l_api_servie():
    service, clock, collector, _, _, log = make_service(cpu=100.0)
    service.run_tick()  # 1er tour : base de comparaison
    clock.t += 60
    collector.values["api"] = 130.0
    service.run_tick()

    serving = [d for _, action, d in log.rows if action == "cost_recorded" and d["kind"] == "serving"]
    assert serving == [{"kind": "serving", "container": "api", "cpu_seconds": 30.0, "cost": pytest.approx(0.3)}]


def test_scale_to_zero_une_seule_fois_par_periode_d_inactivite():
    service, clock, _, usage, bus, _ = make_service(idle_minutes=30)
    for _ in range(40):  # 40 min sans requête utilisateur
        service.run_tick()
        clock.t += 60
    assert recommendations(bus) == ["scale_to_zero"]

    usage["count"] += 5  # le trafic revient...
    service.run_tick()
    for _ in range(31):  # ...puis 31 min de silence : nouvelle recommandation
        clock.t += 60
        service.run_tick()
    assert recommendations(bus) == ["scale_to_zero", "scale_to_zero"]


def test_api_active_pas_de_scale_to_zero():
    service, clock, _, usage, bus, _ = make_service(idle_minutes=30)
    for _ in range(40):
        usage["count"] += 1
        service.run_tick()
        clock.t += 60
    assert "scale_to_zero" not in recommendations(bus)


def test_taille_du_serveur_recommandee_quand_elle_change():
    service, clock, collector, usage, bus, _ = make_service(cpu=0.0, size="S")
    service.run_tick()
    for _ in range(3):  # 1,5 vCPU utilisé en continu sur un serveur S (1 vCPU)
        usage["count"] += 1
        clock.t += 60
        collector.values["api"] += 90.0
        service.run_tick()

    assert recommendations(bus) == ["upsize"]  # une seule fois, pas à chaque tour
    rec = [e.payload for e in bus.events if e.type == EventType.COST_RECOMMENDATION][0]
    assert rec["recommended"] == "M"


def test_une_tache_en_erreur_n_empeche_pas_les_autres():
    service, clock, collector, _, bus, _ = make_service(idle_minutes=0)

    def broken(container):
        raise RuntimeError("Docker injoignable")

    collector.cpu_seconds = broken
    service.run_tick()
    assert recommendations(bus) == ["scale_to_zero"]


def test_user_requests_lit_le_compteur_sans_les_sondes():
    class Http:
        def get(self, url, timeout):
            class Resp:
                text = ("# TYPE api_user_requests_total counter\napi_user_requests_total 12.0\n"
                        '# TYPE api_requests_total counter\napi_requests_total{endpoint="/predict",status="200"} 99.0\n')

                def raise_for_status(self):
                    pass
            return Resp()

    assert user_requests("http://api", Http()) == 12.0


def test_cost_service_start_n_est_pas_bloque_par_l_ecoute():
    import threading

    service, *_ = make_service()
    release = threading.Event()

    class BlockingBus:
        def listen(self, handler, types=None):
            release.wait(5)  # comme Redis : ne rend jamais la main

    service.bus = BlockingBus()
    service.interval_seconds = 0.1
    service.start()  # ne doit pas bloquer
    assert service._running is True
    assert service._thread is not None and service._thread.is_alive()
    assert service._listener.is_alive()

    service.stop()
    release.set()
    assert service._running is False
    assert not service._thread.is_alive()
