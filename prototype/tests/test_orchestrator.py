import pytest

from agents.orchestrator.agent import AGENT, Orchestrator
from agents.orchestrator.notifier import Notifier
from agents.orchestrator.rules import APPROVAL, AUTO, FORBIDDEN, Rules
from common.action_log import ActionLog
from common.events import Event, EventType, Severity
from agents.orchestrator.web import create_app 


# ---------- règles ----------
def test_rules_lit_le_fichier_autonomy():
    rules = Rules()
    assert rules.can("retrain") == AUTO
    assert rules.can("retrain_over_budget") == APPROVAL
    assert rules.can("promote_worse_model") == FORBIDDEN


def test_action_inconnue_demande_validation():
    assert Rules({}).can("supprimer_la_prod") == APPROVAL


# ---------- notifications ----------
@pytest.fixture
def notifier(tmp_path):
    return Notifier(f"sqlite:///{tmp_path / 'notif.db'}")


def test_notify_puis_latest(notifier):
    notifier.notify("Modèle v3 promu")
    last = notifier.latest()[0]
    assert last["kind"] == "info"
    assert last["text"] == "Modèle v3 promu"


def test_une_question_ne_recoit_qu_une_reponse(notifier):
    notifier.ask("On réentraîne malgré le budget ?", "req-1")
    assert notifier.latest()[0]["status"] == "pending"

    assert notifier.answer("req-1", True, "ismail") is True
    assert notifier.answer("req-1", False, "omar") is False  # double clic ignoré
    assert notifier.answer("inconnue", True, "ismail") is False

    last = notifier.latest()[0]
    assert (last["status"], last["answered_by"]) == ("approved", "ismail")


# ---------- décisions de l'orchestrateur ----------
RULES = {"retrain": "auto", "retrain_over_budget": "approval"}


class FakeBus:
    """Garde les événements publiés au lieu de les envoyer sur Redis."""

    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]

    def last(self, type_):
        return [e for e in self.events if e.type == type_][-1]


def ev(type_, source="test", severity=Severity.INFO, **payload):
    return Event(type=type_, source=source, severity=severity, payload=payload)


def drift(reason="distribution_drift", **extra):
    return ev(EventType.DRIFT_DETECTED, "agent_qualite", Severity.WARNING,
              reason=reason, drift_score=0.7, **extra)


def cost_answer(request_id, exceeds=False):
    return ev(EventType.COST_ESTIMATED, "agent_cost", request_id=request_id, exceeds_budget=exceeds,
              options={"standard": {"cost": 0.3}}, month_total=19.9, budget=20.0)


@pytest.fixture
def orch(tmp_path, notifier):
    return Orchestrator(bus=FakeBus(), log=ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}"),
                        notifier=notifier, rules=Rules(RULES))


def ask_cost(orch):
    """Envoie une dérive et renvoie le request_id de la demande de coût."""
    orch.handle(drift())
    return orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]


def test_derive_dans_le_budget_lance_le_retrain(orch):
    request_id = ask_cost(orch)
    orch.handle(cost_answer(request_id))

    retrain = orch.bus.last(EventType.RETRAIN_REQUESTED)
    assert retrain.payload["requested_by"] == AGENT
    assert retrain.payload["approved_by"] is None
    assert "period" not in retrain.payload  # Retrain choisit alors les derniers mois de data/


def test_retrain_sur_le_mois_de_la_derive(orch):
    orch.handle(drift(period="2023-02"))
    orch.handle(cost_answer(orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]))

    assert orch.bus.last(EventType.RETRAIN_REQUESTED).payload["period"] == "2023-02"


def test_derive_hors_budget_attend_la_validation(orch, notifier):
    request_id = ask_cost(orch)
    orch.handle(cost_answer(request_id, exceeds=True))

    assert EventType.RETRAIN_REQUESTED not in orch.bus.types()
    assert EventType.APPROVAL_REQUESTED in orch.bus.types()
    assert notifier.latest()[0]["status"] == "pending"

    orch.handle(ev(EventType.APPROVAL_GIVEN, "page_web", request_id=request_id,
                   approved=True, approved_by="ismail"))
    assert orch.bus.last(EventType.RETRAIN_REQUESTED).payload["approved_by"] == "ismail"


def test_refus_humain_aucun_retrain(orch, notifier):
    request_id = ask_cost(orch)
    orch.handle(cost_answer(request_id, exceeds=True))
    orch.handle(ev(EventType.APPROVAL_GIVEN, "page_web", request_id=request_id,
                   approved=False, approved_by="omar"))

    assert EventType.RETRAIN_REQUESTED not in orch.bus.types()
    assert "refusé par omar" in notifier.latest()[0]["text"]


def test_donnees_corrompues_pas_de_retrain(orch):
    orch.handle(drift(reason="corrupted_input_data", invalid_share=0.1))

    assert orch.bus.types() == [EventType.INCIDENT_OPENED]  # ni coût, ni retrain


def test_model_registered_devient_training_completed(orch):
    orch.handle(ev(EventType.MODEL_REGISTERED, "agent_retrain", version="3", rmse=5.1,
                   training_seconds=42.0, period=["2023-02"]))

    done = orch.bus.last(EventType.TRAINING_COMPLETED)
    assert done.payload["cpu_seconds"] == 42.0
    assert done.payload["months"] == 1


def test_une_seule_demande_a_la_fois(orch):
    request_id = ask_cost(orch)
    orch.handle(cost_answer(request_id))
    orch.handle(drift())  # retrain en cours : ignorée
    assert orch.bus.types().count(EventType.COST_ESTIMATE) == 1

    orch.handle(ev(EventType.PROMOTION_BLOCKED, "agent_retrain", challenger_version="3", challenger_rmse=6.0,
                   production_version="2", production_rmse=6.05, improvement=0.008, min_improvement=0.02))
    orch.handle(drift())  # retrain terminé : nouvelle demande
    assert orch.bus.types().count(EventType.COST_ESTIMATE) == 2


def test_estimation_inconnue_ignoree(orch):
    orch.handle(cost_answer("pas-a-nous"))
    assert orch.bus.events == []


# ---------- page web ----------
def test_page_web_valider_publie_approval_given(tmp_path, notifier):
    bus = FakeBus()
    log = ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}")
    client = create_app(notifier, bus, log).test_client()
    notifier.ask("On réentraîne malgré le budget ?", "req-1")

    assert "On réentraîne malgré le budget ?" in client.get("/").get_data(as_text=True)

    client.post("/answer", data={"request_id": "req-1", "choice": "yes", "who": "ismail"})
    client.post("/answer", data={"request_id": "req-1", "choice": "no", "who": "omar"})  # trop tard

    assert bus.types() == [EventType.APPROVAL_GIVEN]
    assert bus.events[0].payload == {"request_id": "req-1", "approved": True, "approved_by": "ismail"}


# ---------- LangGraph ----------
def test_fil_en_pause_puis_oublie(orch):
    request_id = ask_cost(orch)
    assert orch.waiting_for(request_id) == EventType.COST_ESTIMATED

    orch.handle(cost_answer(request_id, exceeds=True))
    assert orch.waiting_for(request_id) == EventType.APPROVAL_GIVEN

    orch.handle(ev(EventType.APPROVAL_GIVEN, "page_web", request_id=request_id,
                   approved=True, approved_by="ismail"))
    assert orch.pending == set()
    assert orch.workflow.get_state(orch.thread(request_id)).values == {}  # fil effacé du checkpointer


def test_cout_renvoye_pendant_l_attente_humaine_ignore(orch):
    request_id = ask_cost(orch)
    orch.handle(cost_answer(request_id, exceeds=True))
    orch.handle(cost_answer(request_id))  # doublon : on attend un humain, pas un coût

    assert EventType.RETRAIN_REQUESTED not in orch.bus.types()
    assert orch.waiting_for(request_id) == EventType.APPROVAL_GIVEN


def test_regle_forbidden_aucun_retrain(tmp_path, notifier):
    orch = Orchestrator(bus=FakeBus(), log=ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}"),
                        notifier=notifier, rules=Rules({"retrain": "forbidden"}))
    orch.handle(cost_answer(ask_cost(orch)))

    assert EventType.RETRAIN_REQUESTED not in orch.bus.types()
    assert "interdit" in notifier.latest()[0]["text"]
    assert orch.pending == set()


def test_evenement_mal_forme_n_arrete_pas_l_ecoute(orch):
    request_id = ask_cost(orch)
    orch.safe_handle(ev(EventType.COST_ESTIMATED, "agent_cost", request_id=request_id))  # sans options
    assert orch.pending == set()  # la demande cassée ne bloque pas les suivantes

    orch.safe_handle(ev(EventType.MODEL_REGISTERED, "agent_retrain", version="4", training_seconds=1.0))
    assert orch.bus.last(EventType.TRAINING_COMPLETED).payload["version"] == "4"
    orch.handle(drift())
    assert orch.bus.types().count(EventType.COST_ESTIMATE) == 2


def test_page_graph_affiche_les_deux_graphes(tmp_path, notifier, orch):
    client = create_app(notifier, FakeBus(), orch.log, orch).test_client()
    html = client.get("/graph").get_data(as_text=True)

    assert "wait_human" in html and "drift_detected" in html


# ---------- coordination avec Fiabilité et Coûts ----------
def incident(type_, incident_id="inc-1"):
    return ev(type_, "agent_fiabilite", Severity.CRITICAL, incident_id=incident_id,
              cause="api_arretee", component="api", level="critical")


def test_pas_de_retrain_pendant_un_incident(orch, notifier):
    orch.handle(incident(EventType.INCIDENT_OPENED))
    orch.handle(drift())
    assert EventType.COST_ESTIMATE not in orch.bus.types()
    assert "pendant un incident" in notifier.latest()[0]["text"]

    orch.handle(incident(EventType.INCIDENT_RESOLVED))
    orch.handle(drift())
    assert EventType.COST_ESTIMATE in orch.bus.types()


def test_incident_jamais_clos_ne_bloque_pas_indefiniment(tmp_path, notifier):
    now = [1000.0]
    orch = Orchestrator(bus=FakeBus(), log=ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}"),
                        notifier=notifier, rules=Rules(RULES), clock=lambda: now[0])
    orch.handle(incident(EventType.INCIDENT_OPENED))
    now[0] += 31 * 60  # INCIDENT_HOLD_MINUTES = 30 par défaut
    orch.handle(drift())
    assert EventType.COST_ESTIMATE in orch.bus.types()


def test_alertes_de_l_agent_couts_relayees(orch, notifier):
    orch.handle(ev(EventType.BUDGET_EXCEEDED, "agent_cost", over_by=0.29,
                   suggested_action="Réduire la période à 1 mois"))
    assert "Réduire la période à 1 mois" in notifier.latest()[0]["text"]

    orch.handle(ev(EventType.COST_RECOMMENDATION, "agent_cost", model="taxi-duration",
                   recommendation="scale_to_zero", monthly_saving=3.2))
    assert "scale_to_zero" in notifier.latest()[0]["text"]
