"""Règles déterministes autour du LLM de l'agent Fiabilité."""
from agents.reliability import guards
from agents.reliability.signals import Signals

OK = {"api_joignable": True, "erreurs_pct": 0.0, "sondes_echec_pct": 0.0, "latence_p95_s": 0.015,
      "sondes_p95_s": 0.02, "modele_charge": True}


def test_healthy_measures_have_no_anomaly():
    assert guards.anomalies(OK, Signals()) == []


def test_latency_breach_only_above_the_objective():
    assert guards.breaches({**OK, "latence_p95_s": 0.9}) == []
    assert guards.breaches({**OK, "latence_p95_s": 2.4}) == ["latence_p95_s 2.4 > 1.0"]


def test_errors_and_signals_are_anomalies():
    assert guards.breaches({**OK, "erreurs_pct": 30.0})
    assert guards.breaches({**OK, "api_joignable": False}) == ["api_injoignable"]
    assert guards.signal_breaches(Signals(mlflow_ok=False)) == ["mlflow_ok = false"]
    assert guards.signal_breaches(Signals(container_status="exited")) == ["conteneur exited"]


def test_worse_needs_a_real_degradation():
    before = {**OK, "erreurs_pct": 100.0, "sondes_echec_pct": 100.0}
    assert not guards.worse(before, before)
    assert guards.worse({**OK, "erreurs_pct": 30.0}, {**OK, "erreurs_pct": 100.0})
    assert guards.worse(OK, {**OK, "api_joignable": False})


def test_bad_model_rule():
    change = {"de": 14, "vers": 21, "at": 100, "origine": "externe (nouvelle promotion)", "sain_avant": True}
    broken = {**OK, "erreurs_pct": 62.5, "sondes_echec_pct": 100.0}
    assert "v21" in guards.bad_model(change, broken, Signals(), now=130)
    assert guards.bad_model(change, OK, Signals(), now=130) is None
    assert guards.bad_model({**change, "origine": "rollback_par_l_agent"}, broken, Signals(), 130) is None
    assert guards.bad_model({**change, "sain_avant": False}, broken, Signals(), 130) is None
    assert guards.bad_model(change, broken, Signals(mlflow_ok=False), 130) is None
    assert guards.bad_model(change, broken, Signals(), now=100 + 3600) is None


def test_cause_contradicted_by_signals_becomes_unknown():
    assert guards.checked_cause("api_arretee", OK, Signals()) == "inconnue"
    assert guards.checked_cause("api_arretee", {**OK, "api_joignable": False},
                                Signals(container_status="exited")) == "api_arretee"
    assert guards.checked_cause("mlflow_indisponible", OK, Signals()) == "inconnue"
    assert guards.checked_cause("mlflow_indisponible", OK, Signals(mlflow_ok=False)) == "mlflow_indisponible"
    assert guards.checked_cause("mauvais_modele", OK, Signals()) == "mauvais_modele"


def test_model_cause_becomes_mlflow_when_mlflow_is_down():
    down = Signals(mlflow_ok=False)
    assert guards.checked_cause("mauvais_modele", OK, down) == "mlflow_indisponible"
    assert guards.checked_cause("postgres_indisponible", OK, Signals(mlflow_ok=False, postgres_ok=False)) \
        == "postgres_indisponible"


def test_hard_rules_checked_by_code():
    assert "MLflow" in guards.forbidden("rollback_model", Signals(mlflow_ok=False))
    assert guards.forbidden("degraded_mode", Signals(mlflow_ok=False)) is None
    assert "boucle" in guards.forbidden("restart_api", Signals(restarts_recent=3))
    assert guards.forbidden("restart_api", Signals(restarts_recent=2)) is None


def test_wrong_cause_while_mlflow_is_down_becomes_mlflow():
    assert guards.checked_cause("api_arretee", OK, Signals(mlflow_ok=False)) == "mlflow_indisponible"
