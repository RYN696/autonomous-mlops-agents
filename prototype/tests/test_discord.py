"""Intégration Discord : boutons, gestes humains et relais des messages (sans réseau)."""
from types import SimpleNamespace

import discord
import pytest

from agents.orchestrator import discord_bot as db
from agents.orchestrator.agent import Orchestrator
from agents.orchestrator.human import HumanActions
from agents.orchestrator.notifier import Message, Notifier
from agents.orchestrator.rules import Rules
from common.action_log import ActionLog
from common.events import EventType, Severity
from tests.test_orchestrator import RULES, FakeBus, drift, ev


class FakeRelay:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def post(self, message):
        if self.fail:
            raise RuntimeError("Discord injoignable")
        self.messages.append(message)


class FakeRegistry:
    """Le strict nécessaire de MlflowClient."""

    def __init__(self, production="4"):
        self.production = production
        self.tags = {}

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=self.production)

    def set_registered_model_alias(self, name, alias, version):
        self.production = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


class FakeHttp:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None)


@pytest.fixture
def relay_notifier(tmp_path):
    return Notifier(f"sqlite:///{tmp_path / 'notif.db'}", relay=FakeRelay())


@pytest.fixture
def log(tmp_path):
    return ActionLog(url=f"sqlite:///{tmp_path / 'log.db'}")


@pytest.fixture
def human(relay_notifier, log):
    return HumanActions(relay_notifier, FakeBus(), log, registry=FakeRegistry(), http=FakeHttp())


def actions_of(log):
    return [(r["agent"], r["action"]) for r in reversed(log.latest(50))]


# ---------- notifier -> relais ----------
def test_chaque_message_part_aussi_vers_le_relais(relay_notifier):
    relay_notifier.notify("Incident", "inc-1", level="alert", actions=("report", "dashboard"))
    relay_notifier.ask("On réentraîne ?", "req-1")

    assert relay_notifier.relay.messages == [
        Message("Incident", "inc-1", "alert", ("report", "dashboard")),
        Message("On réentraîne ?", "req-1", "question", ("approve", "refuse", "report"))]


def test_discord_en_panne_ne_bloque_pas_les_messages(tmp_path):
    notifier = Notifier(f"sqlite:///{tmp_path / 'n.db'}", relay=FakeRelay(fail=True))
    notifier.notify("Modèle promu")
    assert notifier.latest()[0]["text"] == "Modèle promu"


# ---------- boutons ----------
def test_boutons_d_une_question():
    specs = db.button_specs(("approve", "refuse", "report"), "req-1", ui_url="http://pc:8090")
    assert [(s.label, s.custom_id or s.url) for s in specs] == [
        ("Valider", "mlops:approve:req-1"), ("Refuser", "mlops:refuse:req-1"),
        ("Rapport", "http://pc:8090/parcours/req-1")]


def test_boutons_promotion_alerte_et_donnees():
    rollback = db.button_specs(("report", "rollback:3:4"), "req-1")
    assert rollback[1].label == "Revenir à v3" and rollback[1].custom_id == "mlops:rollback:3:4:req-1"

    alert = db.button_specs(("data", "ignore", "dashboard"), "req-2", grafana_url="http://pc:3000")
    assert [s.label for s in alert] == ["Voir les données", "Ignorer", "Tableau de bord"]
    assert alert[2].url == "http://pc:3000"


def test_sans_parcours_pas_de_lien_rapport():
    assert db.button_specs(("report", "approve"), None) == []


# ---------- clics ----------
def test_valider_sur_discord_publie_approval_given(human, log):
    human.notifier.ask("On réentraîne ?", "req-1")

    reply = db.dispatch(human, "mlops:approve:req-1", "ismail (Discord)")

    assert reply.consumed == ("approve", "refuse") and "Validé" in reply.text
    event = human.bus.last(EventType.APPROVAL_GIVEN)
    assert event.source == "discord"
    assert event.payload == {"request_id": "req-1", "approved": True, "approved_by": "ismail (Discord)"}
    assert ("humain", "approval_answered") in actions_of(log)


def test_second_clic_ou_reponse_sur_la_page_web(human):
    human.notifier.ask("On réentraîne ?", "req-1")
    human.answer("req-1", False, "omar")  # page web

    reply = db.dispatch(human, "mlops:approve:req-1", "ismail (Discord)")

    assert "déjà une réponse" in reply.text
    assert human.bus.types() == [EventType.APPROVAL_GIVEN]  # un seul approval_given (celui d'omar)


def test_revenir_demande_une_confirmation(human):
    reply = db.dispatch(human, "mlops:rollback:3:4:req-1", "ismail (Discord)")

    assert reply.confirm.custom_id == "mlops:rollback-ok:3:4:req-1"
    assert human.registry.production == "4"  # rien n'a changé avant la confirmation


def test_revenir_confirme_remet_la_version_et_recharge_l_api(human, log):
    reply = db.dispatch(human, "mlops:rollback-ok:3:4:req-1", "ismail (Discord)")

    assert human.registry.production == "3"
    assert human.registry.tags[("4", "rolled_back")] == "true"
    assert human.http.calls[0].endswith("/reload")
    assert reply.consumed == ("rollback",)
    assert ("humain", "human_rollback") in actions_of(log)
    sent = human.notifier.relay.messages[-1]
    assert sent.key == "req-1" and "v3" in sent.text


def test_bouton_revenir_perime_ne_change_rien(human, log):
    human.registry.production = "5"  # un autre modèle a été promu depuis

    reply = db.dispatch(human, "mlops:rollback-ok:3:4:req-1", "ismail (Discord)")

    assert "v5" in reply.text and reply.consumed == ("rollback",)
    assert human.registry.production == "5"
    assert ("humain", "human_rollback") not in actions_of(log)


def test_ignorer_une_alerte_est_journalise(human, log):
    reply = db.dispatch(human, "mlops:ignore:req-1", "ismail (Discord)")
    assert reply.consumed == ("ignore",)
    assert ("humain", "alert_ignored") in actions_of(log)


def test_bouton_d_un_autre_bot_ignore(human):
    assert db.dispatch(human, "autre:approve:req-1", "x") is None


# ---------- messages de l'orchestrateur ----------
@pytest.fixture
def orch(tmp_path, relay_notifier, log):
    return Orchestrator(bus=FakeBus(), log=log, notifier=relay_notifier, rules=Rules(RULES))


def test_promotion_propose_de_revenir_a_l_ancienne_version(orch):
    orch.handle(ev(EventType.MODEL_PROMOTED, "agent_retrain", request_id="req-1", challenger_version=4,
                   challenger_rmse=5.1, production_version=3, production_rmse=5.4, improvement=0.05))
    m = orch.notifier.relay.messages[-1]
    assert (m.key, m.level, m.actions) == ("req-1", "success", ("report", "rollback:3:4"))


def test_donnees_corrompues_alerte_avec_voir_et_ignorer(orch):
    orch.handle(drift(reason="corrupted_input_data", invalid_share=0.1))
    m = orch.notifier.relay.messages[-1]
    assert m.level == "alert" and m.actions == ("data", "ignore", "dashboard")


def test_incident_dans_le_salon_des_alertes(orch):
    orch.handle(ev(EventType.INCIDENT_OPENED, "agent_fiabilite", Severity.CRITICAL,
                   incident_id="inc-1", cause="api_arretee", component="api", level="critique"))
    m = orch.notifier.relay.messages[-1]
    assert (m.key, m.level, m.actions) == ("inc-1", "alert", ("report", "dashboard"))


def test_le_message_de_retrain_donne_la_cause(orch):
    orch.handle(drift(period="2023-02", explanation="Les distances ont augmenté de 50 %."))
    request_id = orch.bus.last(EventType.COST_ESTIMATE).payload["request_id"]
    orch.handle(ev(EventType.COST_ESTIMATED, "agent_cost", request_id=request_id, exceeds_budget=False,
                   options={"standard": {"cost": 0.01}}, month_total=0.0, budget=20.0))
    text = orch.notifier.relay.messages[-1].text
    assert "sur 2023-02" in text and "distances ont augmenté" in text


# ---------- relais Discord (sans réseau) ----------
class FakeDiscordMessage:
    def __init__(self, **kwargs):
        self.sent = kwargs
        self.edits = []
        self.flags = SimpleNamespace(ephemeral=False)

    async def edit(self, **kwargs):
        self.edits.append(kwargs)


class FakeChannel:
    def __init__(self, reject_links=False):
        self.messages = []
        self.reject_links = reject_links

    async def send(self, **kwargs):
        view = kwargs.get("view")
        if self.reject_links and view and any(b.url for b in view.children):
            raise discord.HTTPException(SimpleNamespace(status=400, reason="Bad Request"), "Invalid Form Body")
        self.messages.append(FakeDiscordMessage(**kwargs))
        return self.messages[-1]


@pytest.fixture
def relay(human):
    r = db.DiscordRelay(human, token="x", alerts_channel="1", agents_channel="2", mention="@here")
    channels = {1: FakeChannel(), 2: FakeChannel()}

    async def ready():
        return None

    async def channel(channel_id):
        return channels[channel_id]

    r.client.wait_until_ready = ready
    r.channel = channel
    r.channels = channels
    yield r
    r.loop.close()


def run(relay, coro):
    return relay.loop.run_until_complete(coro)


def test_un_parcours_un_seul_message_complete(relay):
    run(relay, relay._post(Message("🔄 Réentraînement lancé.", "req-1", "info", ("report",))))
    run(relay, relay._post(Message("✅ v4 promu.", "req-1", "success", ("report", "rollback:3:4"))))

    agents = relay.channels[2].messages
    assert len(agents) == 1 and relay.channels[1].messages == []
    edit = agents[0].edits[-1]
    assert "Réentraînement lancé" in edit["embed"].description and "v4 promu" in edit["embed"].description
    assert [b.label for b in edit["view"].children] == ["Rapport", "Revenir à v3"]


def test_alerte_dans_son_salon_avec_mention(relay):
    run(relay, relay._post(Message("🚨 Incident", "inc-1", "alert", ("report", "dashboard"))))
    sent = relay.channels[1].messages[0].sent
    assert sent["content"] == "@here" and sent["embed"].color.value == db.COLORS["alert"]


def test_boutons_retires_apres_usage(relay):
    run(relay, relay._post(Message("On réentraîne ?", "req-1", "question", ("approve", "refuse", "report"))))
    run(relay, relay._consume(db.Reply("Validé", "req-1", ("approve", "refuse")), None))

    view = relay.channels[2].messages[0].edits[-1]["view"]
    assert [b.label for b in view.children] == ["Rapport"]


def test_liens_refuses_par_discord_ecrits_dans_le_texte(relay):
    relay.channels[2].reject_links = True
    run(relay, relay._post(Message("Réentraînement lancé.", "req-1", "info", ("report",))))

    sent = relay.channels[2].messages[0].sent
    assert "view" not in sent and "[Rapport](http" in sent["embed"].description


def test_description_trop_longue_garde_les_dernieres_lignes():
    lines = [f"ligne {i} " + "x" * 100 for i in range(100)]
    e = db.render("req-1", lines, "info")
    assert len(e["description"]) <= db.MAX_DESCRIPTION + 2
    assert e["description"].startswith("…") and e["description"].endswith(lines[-1])
