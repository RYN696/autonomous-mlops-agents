"""Détection de pic de coût (agents/cost/anomaly.py) : règle déterministe, alerte seulement."""
from datetime import datetime, timezone

import pytest

from agents.cost import anomaly
from agents.cost import report as rp
from agents.cost.settings import CostConfig
from agents.cost.sql_store import SqlCostStore
from common.action_log import ActionLog
from common.events import EventType
from tests.test_cost_agent import make_agent, make_service, send
from tests.test_orchestrator import ev  # noqa: F401  (fixtures orch / notifier ci-dessous)
from tests.test_orchestrator import notifier, orch  # noqa: F401

CFG = CostConfig(cost_per_cpu_second=0.01, spike_factor=3.0, spike_window=10,
                 spike_min_history=3, spike_min_vcpus=0.1)


# ---------- règle ----------
def test_pas_de_jugement_sans_historique():
    assert anomaly.spike([10, 10], 1000, CFG) is None


def test_pic_au_dela_du_facteur():
    found = anomaly.spike([10, 12, 9, 11], 40, CFG)
    assert found["ratio"] == pytest.approx(3.8) and found["typical"] == pytest.approx(10.5)


def test_hausse_normale_n_est_pas_un_pic():
    assert anomaly.spike([10, 12, 9, 11], 25, CFG) is None


def test_la_mediane_ignore_un_pic_passe():
    """Un ancien pic ne relève pas le seuil (une moyenne l'aurait fait)."""
    assert anomaly.spike([10, 10, 500, 10, 10], 40, CFG) is not None


def test_entrainement_compare_par_mois_de_donnees():
    history = [(100, 1), (300, 3), (110, 1)]          # ~100 s par mois de données
    assert anomaly.training_spike(history, 300, 3, CFG) is None   # 3 mois = 3 fois 1 mois
    found = anomaly.training_spike(history, 1200, 3, CFG)          # 400 s par mois : pic
    assert found["kind"] == "training" and found["ratio"] == pytest.approx(4.0)
    assert found["extra_cpu_seconds"] == pytest.approx(900)
    assert found["extra_cost"] == pytest.approx(9.0)
    assert "Optuna" in found["hint"]


def test_entrainement_sans_periode_pas_jugé():
    assert anomaly.training_spike([(100, 1)] * 5, 5000, None, CFG) is None


def test_api_bruit_de_fond_ignore():
    """0,005 -> 0,05 vCPU : x10, mais sous le plancher : ce n'est pas un coût qui compte."""
    assert anomaly.serving_spike([0.005] * 5, 0.05, CFG) is None
    assert anomaly.serving_spike([0.005] * 5, 0.2, CFG)["kind"] == "serving"


# ---------- agent ----------
def test_agent_publie_un_pic_d_entrainement():
    agent, store, bus, log, _ = make_agent()
    for _ in range(3):
        send(agent, "training_completed", {"cpu_seconds": 100, "months": 1})
    assert EventType.COST_ANOMALY not in [e.type for e in bus.events]

    send(agent, "training_completed", {"cpu_seconds": 900, "months": 1, "version": 34, "request_id": "r1"})
    alert = [e for e in bus.events if e.type == EventType.COST_ANOMALY][-1]
    assert alert.payload["ratio"] == pytest.approx(9.0)
    assert (alert.payload["version"], alert.payload["request_id"]) == (34, "r1")
    assert log.rows[-1][1] == "cost_anomaly"
    assert len(store.training_history("taxi")) == 4   # le coût est enregistré quand même


def test_agent_un_seul_signal_par_episode_de_charge():
    agent, _, bus, log, _ = make_agent()
    loads = [0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 0.2, 1.2]
    results = [agent.check_serving_spike("taxi", v) for v in loads]
    assert [r is not None for r in results] == [False] * 4 + [True, False, False, True]
    assert sum(e.type == EventType.COST_ANOMALY for e in bus.events) == 2


def test_service_signale_un_pic_de_charge_de_l_api():
    service, clock, collector, usage, bus, _ = make_service()
    for _ in range(5):                              # ~0,2 vCPU par tour de 60 s
        collector.values["api"] += 12; usage["count"] += 1
        service.run_tick(); clock.t += 60
    collector.values["api"] += 120; usage["count"] += 1   # 2 vCPU : x10
    service.run_tick()
    alert = [e for e in bus.events if e.type == EventType.COST_ANOMALY]
    assert len(alert) == 1 and alert[0].payload["kind"] == "serving"


# ---------- orchestrateur et rapport ----------
def test_orchestrateur_alerte_sans_agir(orch, notifier):  # noqa: F811
    orch.handle(ev(EventType.COST_ANOMALY, "agent_cost", kind="training", ratio=5.9, value=588.0,
                   typical=100.0, extra_cost=0.24, hint="vérifier Optuna", version=33))
    msg = notifier.latest()[0]
    assert "Pic de coût" in msg["text"] and "5.9×" in msg["text"] and "Aucune action automatique" in msg["text"]
    assert EventType.RETRAIN_REQUESTED not in orch.bus.types()


def test_le_rapport_mentionne_les_pics(tmp_path):
    url = f"sqlite:///{tmp_path / 'c.db'}"
    SqlCostStore(url)
    log = ActionLog(url)
    log.record("cost", "cost_anomaly", "taxi-duration", {"kind": "training", "ratio": 5.9})
    now = datetime.now(timezone.utc)
    f = rp.figures(log.engine, now)
    assert f["anomalies"] == 1
    assert any("Pics de coût ce mois : 1" in line and "5.9 fois" in line for line in rp.findings(f))
