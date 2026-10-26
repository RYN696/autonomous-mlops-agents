from __future__ import annotations
from common import config

import os

import requests
import re

OLLAMA_URL = os.getenv("QUALITY_OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("QUALITY_LLM_MODEL", "llama3.1:8b")
TIMEOUT = float(os.getenv("QUALITY_LLM_TIMEOUT", "120"))

SYSTEM = (
    "You are the data-quality agent of an MLOps pipeline predicting NYC taxi trip duration. "
    "You receive FINDINGS already computed by the rules. "
    "Explain the decision to a human in at most 2 short sentences. "
    "Present as the cause ONLY the findings marked ABOVE; say the others are normal. "
    "Use ONLY numbers that appear in the findings. "
    "Never say a value exceeds a level unless its finding says ABOVE. "
    "Say that the distribution of the model's predictions changed; "
    "never claim the model is inaccurate or broken, we have no measure of its real error. "
    "Write in plain lowercase words, do not copy the words ABOVE or below literally."
)


def build_findings(p: dict) -> list[str]:
    """Constats tranchés par du code, pas par le LLM."""
    pd_, pthr = p.get("prediction_drift", 0.0), p.get("column_drift_threshold", 0.1)
    ds, dthr = p.get("drift_score", 0.0), p.get("drift_threshold", 0.5)
    inv, cthr = p.get("invalid_share", 0.0), config.CRITICAL_INVALID_SHARE
    flag = lambda v, t: "ABOVE" if v >= t else "below"
    return [
        f"Prediction drift = {pd_:.2f} (alert level {pthr:.2f}): {flag(pd_, pthr)}",
        f"Drifted columns = {p.get('num_drifted_columns', 0)} of 4, share {ds:.2f} "
        f"(alert level {dthr:.2f}): {flag(ds, dthr)}",
        f"Invalid rows = {inv * 100:.1f}% (critical level {cthr * 100:.0f}%): {flag(inv, cthr)}",
    ]


def numbers_ok(text: str, findings: list[str]) -> bool:
    """Tous les nombres cités doivent exister dans les constats."""
    nums = lambda s: {float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)}
    return nums(text) <= nums(" ".join(findings))


def template_text(event_type: str, severity: str, p: dict) -> str:
    """Phrase automatique, sans LLM (plan B)."""
    return (f"{event_type} ({severity}). drift_score={p.get('drift_score')}, "
            f"prediction_drift={p.get('prediction_drift')}, "
            f"invalid_share={p.get('invalid_share')}.")


def ask_llm(event_type: str, severity: str, payload: dict) -> str | None:
    findings = build_findings(payload)
    body = {
        "model": MODEL,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 150},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Decision: {event_type}, severity {severity}.\nFINDINGS:\n- "
                        + "\n- ".join(findings)},
        ],
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        return text or None
    except Exception:
        return None


def add_explanation(event, use_llm: bool = True):
    etype, sev = event.type.value, event.severity.value
    text = ask_llm(etype, sev, event.payload) if use_llm else None
    raw, rejected = text, False
    if text and not numbers_ok(text, build_findings(event.payload)):
        text, rejected = None, True
    if text:
        text = text[0].upper() + text[1:]    
    payload = dict(event.payload)
    payload["explanation"] = text or template_text(etype, sev, event.payload)
    payload["explanation_source"] = "llm" if text else "template"
    if text:
        payload["llm_model"] = MODEL
    if rejected:
        payload["explanation_rejected"] = True
        payload["rejected_text"] = raw
    return event.model_copy(update={"payload": payload})