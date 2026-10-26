import pandas as pd

from agents.quality.checker import build_event, publish_quality_result
from agents.quality.validation import validate_current_data
from common import config
from common.events import EventType, Severity


# ---------- Outils de test ----------

def make_df(n=100):
    t = pd.Timestamp("2023-02-01 10:00")
    return pd.DataFrame({
        "lpep_pickup_datetime": [t] * n,
        "lpep_dropoff_datetime": [t + pd.Timedelta(minutes=10)] * n,
        "passenger_count": [1] * n,
        "trip_distance": [2.0] * n,
        "PULocationID": [43] * n,
        "DOLocationID": [151] * n,
    })


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, e):
        self.events.append(e)


class FakeLog:
    def __init__(self):
        self.records = []

    def record(self, **kw):
        self.records.append(kw)


# ---------- Tests de validation ----------

def test_clean_data():
    r = validate_current_data(make_df())
    assert r.invalid_rows == 0 and not r.is_critical


def test_negative_duration():
    df = make_df()
    df.loc[0, "lpep_dropoff_datetime"] = df.loc[0, "lpep_pickup_datetime"] - pd.Timedelta(minutes=5)
    assert "invalid_duration" in validate_current_data(df).reasons


def test_too_many_passengers():
    df = make_df()
    df.loc[0, "passenger_count"] = 20
    assert "invalid_passenger_count" in validate_current_data(df).reasons


def test_negative_distance():
    df = make_df()
    df.loc[0, "trip_distance"] = -1
    assert "invalid_distance" in validate_current_data(df).reasons


def test_missing_column():
    r = validate_current_data(make_df().drop(columns=["trip_distance"]))
    assert "missing_required_columns" in r.reasons and r.is_critical


def test_mostly_corrupted_is_critical():
    df = make_df()
    df.loc[:9, "trip_distance"] = -1   # 10 % de lignes invalides
    assert validate_current_data(df).is_critical


# ---------- Tests de décision et publication ----------

def test_quality_ok():
    e = build_event(validate_current_data(make_df()), 0.0, 0)
    assert e.type == EventType.QUALITY_OK and e.severity == Severity.INFO


def test_drift_warning():
    e = build_event(validate_current_data(make_df()), config.DRIFT_THRESHOLD + 0.1, 2)
    assert e.type == EventType.DRIFT_DETECTED
    assert e.severity == Severity.WARNING
    assert e.payload["reason"] == "distribution_drift"


def test_corrupted_is_critical():
    df = make_df()
    df.loc[:9, "trip_distance"] = -1
    e = build_event(validate_current_data(df), 0.0, 0)
    assert e.severity == Severity.CRITICAL
    assert e.payload["reason"] == "corrupted_input_data"


def test_publish_logs_and_publishes():
    bus, log = FakeBus(), FakeLog()
    e = build_event(validate_current_data(make_df()), 0.9, 3)
    publish_quality_result(e, bus, log)
    assert bus.events[0].type == EventType.DRIFT_DETECTED
    assert log.records[0]["action"] == "drift_detected"

import numpy as np
from agents.quality.drift import calculate_drift


class FakeModel:
    def predict(self, feats):
        return [f["trip_distance"] * 5 for f in feats]


def ride_df(n, dist, pu, do):
    return pd.DataFrame({
        "PULocationID": [str(x) for x in pu],
        "DOLocationID": [str(x) for x in do],
        "trip_distance": dist,
    })

def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(0)
    n = 3000
    a = ride_df(n, rng.uniform(1, 5, n), rng.choice([43, 74], n), rng.choice([151, 166], n))
    b = ride_df(n, rng.uniform(1, 5, n), rng.choice([43, 74], n), rng.choice([151, 166], n))
    r = calculate_drift(a, b, FakeModel())
    assert r["num_drifted_columns"] == 0
    assert r["prediction_drift"] < config.COLUMN_DRIFT_THRESHOLD


def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(1)
    n = 3000
    a = ride_df(n, rng.uniform(1, 5, n), rng.choice([43, 74], n), rng.choice([151, 166], n))
    b = ride_df(n, rng.uniform(15, 30, n), rng.choice([10, 20], n), rng.choice([30, 40], n))
    r = calculate_drift(a, b, FakeModel())
    assert r["prediction_drift"] >= config.COLUMN_DRIFT_THRESHOLD
    assert r["drift_score"] >= config.DRIFT_THRESHOLD


def test_prediction_drift_alone_triggers_event():
    e = build_event(validate_current_data(make_df()), 0.25, 1, prediction_drift=0.56)
    assert e.type == EventType.DRIFT_DETECTED
    assert e.payload["reason"] == "distribution_drift"

from agents.quality import explain


def test_explanation_falls_back_without_llm():
    e = build_event(validate_current_data(make_df()), 0.0, 0)
    out = explain.add_explanation(e, use_llm=False)
    assert out.payload["explanation_source"] == "template"
    assert out.type == e.type and out.severity == e.severity


def test_llm_text_is_used_but_decision_unchanged(monkeypatch):
    monkeypatch.setattr(explain, "ask_llm", lambda *a, **k: "Test explanation.")
    e = build_event(validate_current_data(make_df()), 0.9, 3)
    out = explain.add_explanation(e)
    assert out.payload["explanation"] == "Test explanation."
    assert out.payload["explanation_source"] == "llm"
    assert out.type == EventType.DRIFT_DETECTED


def test_llm_down_does_not_crash(monkeypatch):
    monkeypatch.setattr(explain, "ask_llm", lambda *a, **k: None)
    e = build_event(validate_current_data(make_df()), 0.9, 3)
    assert explain.add_explanation(e).payload["explanation_source"] == "template"

def test_findings_flag_the_real_cause():
    e = build_event(validate_current_data(make_df()), 0.25, 1, prediction_drift=0.56)
    f = explain.build_findings(e.payload)
    assert "ABOVE" in f[0] and "below" in f[1] and "below" in f[2]


def test_invented_number_is_rejected(monkeypatch):
    monkeypatch.setattr(explain, "ask_llm", lambda *a, **k: "Drift is 0.99.")
    e = build_event(validate_current_data(make_df()), 0.25, 1, prediction_drift=0.56)
    out = explain.add_explanation(e)
    assert out.payload["explanation_source"] == "template"
    assert out.payload["explanation_rejected"] is True


def test_valid_numbers_are_accepted(monkeypatch):
    monkeypatch.setattr(explain, "ask_llm", lambda *a, **k: "Prediction drift is 0.56, above 0.10.")
    e = build_event(validate_current_data(make_df()), 0.25, 1, prediction_drift=0.56)
    assert explain.add_explanation(e).payload["explanation_source"] == "llm"    

def test_ask_llm_builds_request_and_parses_reply(monkeypatch):
    sent = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": " Prediction drift is 0.56. "}}

    def fake_post(url, json=None, timeout=None):
        sent["body"] = json
        return FakeResp()

    monkeypatch.setattr(explain.requests, "post", fake_post)
    e = build_event(validate_current_data(make_df()), 0.25, 1, prediction_drift=0.56)
    assert explain.ask_llm("drift_detected", "warning", e.payload) == "Prediction drift is 0.56."
    assert "FINDINGS" in sent["body"]["messages"][1]["content"]