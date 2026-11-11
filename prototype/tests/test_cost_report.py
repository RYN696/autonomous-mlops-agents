"""Rapport de coûts : chiffres calculés par le code, explication du LLM vérifiée."""
import time
from datetime import datetime, timezone

import pytest

from agents.cost import report as rp
from agents.cost.sql_store import SqlCostStore
from agents.orchestrator.notifier import Notifier
from agents.orchestrator.web import create_app
from common.action_log import ActionLog
from common.llm import LLMError
from tests.test_orchestrator import FakeBus

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)  # 10,5 jours écoulés sur 30


class FakeLLM:
    model = "llama3.2:3b"

    def __init__(self, text=None, error=False):
        self.text, self.error = text, error
        self.prompts = []

    def chat_json(self, system, user, schema=None):
        self.prompts.append(user)
        if self.error:
            raise LLMError("Ollama injoignable")
        return {"explication": self.text}


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'costs.db'}"
    store, log = SqlCostStore(url), ActionLog(url)
    store.add("taxi-duration", "2026-11", "training", 300.0, 0.15, months=1)
    store.add("taxi-duration", "2026-11", "training", 60.0, 0.03, months=1)
    store.add("taxi-duration", "2026-11", "serving", 20.0, 0.01)
    store.add("taxi-duration", "2026-10", "training", 100.0, 0.05, months=1)
    log.record("cost", "cost_estimated", "taxi-duration", {"budget": 0.1})
    log.record("cost", "cost_recorded", "taxi-duration",
               {"kind": "training", "cost": 0.15, "cpu_seconds": 300.0, "request_id": "req-1"})
    log.record("cost", "budget_exceeded", "taxi-duration", {"over_by": 0.08})
    return log


def test_chiffres_du_mois(db):
    f = rp.figures(db.engine, NOW)
    assert (f["month"], f["trainings"], f["budget"]) == ("2026-11", 2, 0.1)
    assert f["total"] == pytest.approx(0.19) and f["training"] == pytest.approx(0.18)
    assert f["previous_total"] == pytest.approx(0.05)
    assert f["projection"] == pytest.approx(0.19 / 10.5 * 30)
    assert f["top_trainings"][0]["request_id"] == "req-1"


def test_le_code_tranche_le_budget(db):
    lines = rp.findings(rp.figures(db.engine, NOW))
    assert "budget DÉPASSÉ de 0.0900 €" in lines[0]
    assert any("AU-DESSUS du budget" in line for line in lines)
    assert any("relever le budget" in line for line in lines)


def test_explication_du_llm_gardee_si_les_nombres_sont_exacts(db):
    f = rp.figures(db.engine, NOW)
    llm = FakeLLM("Le mois a coûté 0,1900 € pour un budget de 0.1000 € : il est dépassé de 0.0900 €.")
    out = rp.explain(f, llm)
    assert out["source"] == "llm" and out["explanation"].startswith("Le mois")
    assert "CONSTATS" in llm.prompts[0]


def test_nombre_invente_par_le_llm_ecarte(db):
    f = rp.figures(db.engine, NOW)
    out = rp.explain(f, FakeLLM("Le mois a coûté 12.5 €."))
    assert out["source"] == "template" and out["rejected_text"] == "Le mois a coûté 12.5 €."
    assert "dépassé de 0.0900 €" in out["explanation"]


def test_llm_injoignable_texte_du_code(db):
    out = rp.explain(rp.figures(db.engine, NOW), FakeLLM(error=True))
    assert out["source"] == "template" and "injoignable" in out["llm_error"]


def test_mois_vide_sans_projection_le_premier_jour(tmp_path):
    log = ActionLog(f"sqlite:///{tmp_path / 'vide.db'}")
    SqlCostStore(f"sqlite:///{tmp_path / 'vide.db'}")
    f = rp.figures(log.engine, datetime(2026, 11, 1, 8, tzinfo=timezone.utc))
    assert f["total"] == 0 and f["projection"] is None and f["avg_training"] is None
    assert "le budget TIENT" in rp.findings(f)[0]


def test_page_couts_genere_le_rapport_et_le_signale(db, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "OllamaClient", lambda *a, **k: FakeLLM("Le budget de 0.1000 € est dépassé."))
    notifier = Notifier(f"sqlite:///{tmp_path / 'n.db'}")
    client = create_app(notifier, FakeBus(), db).test_client()

    assert "Aucun rapport" in client.get("/couts").get_data(as_text=True)
    client.post("/couts/rapport")
    for _ in range(50):
        if notifier.latest():  # écrit après le journal : le rapport est complet
            break
        time.sleep(0.1)

    page = client.get("/couts").get_data(as_text=True)
    assert "Le budget de 0.1000 € est dépassé." in page and "expliqué par llama3.2:3b" in page
    assert notifier.latest()[0]["text"].startswith("📊 Rapport de coûts")
