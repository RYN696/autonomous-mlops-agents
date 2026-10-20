"""Agent Fiabilité piloté par LLM, avec un LLM agent et un LLM juge scriptés (sans Ollama)."""
import json

from agents.reliability.actions import ActionResult, SystemSnapshot
from agents.reliability.catalog import Action
from agents.reliability.llm_agent import LLMReliabilityAgent
from agents.reliability.metrics import Snapshot
from agents.reliability.probes import ProbeResult
from agents.reliability.prompts import (ACTION_PROMPT, ACTION_SCHEMA, DETECTION_PROMPT, DETECTION_SCHEMA,
                                        VERIFICATION_PROMPT, VERIFICATION_SCHEMA)
from agents.reliability.signals import Signals
from common.events import EventType
from common.llm import LLMError

FIN = {"pensee": "rien à faire", "fin": True}


class ScriptedLLM:
    """Renvoie les réponses prévues dans l'ordre, puis {"fin": true}. Garde les messages reçus."""

    def __init__(self, model, answers=()):
        self.model, self.answers, self.received, self.schemas, self.down = model, list(answers), [], [], False

    def chat(self, messages, schema=None):
        self.received.append(messages)
        self.schemas.append(schema)
        if self.down:
            raise LLMError("Ollama arrêté")
        return self.answers.pop(0) if self.answers else FIN

    def chat_json(self, system, user, schema=None):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], schema=schema)


class Api:
    def __init__(self):
        self.requests = self.errors = 0.0
        self.reachable, self.version = True, 4

    def snapshot(self):
        if not self.reachable:
            return Snapshot(timestamp=0, reachable=False, error="connection refused")
        return Snapshot(timestamp=0, reachable=True, requests=self.requests, errors=self.errors,
                        buckets={0.05: self.requests, float("inf"): self.requests},
                        model_loaded=True, model_version=self.version, health_ok=True)


class FakeActions:
    def __init__(self, fail=()):
        self.fail, self.executed, self.restored, self.degraded_exits = set(fail), [], 0, 0

    def execute(self, action):
        self.executed.append(action)
        ok = action not in self.fail
        return ActionResult(action.value, ok, "ok" if ok else "échec simulé")

    def take_snapshot(self):
        return SystemSnapshot(0, "4", {"mlops-api": True}, complete=True)

    def restore(self, snap):
        self.restored += 1
        return ActionResult("restore_snapshot", True, "alias production remis sur la version 4")

    def exit_degraded_mode(self):
        self.degraded_exits += 1
        return ActionResult("exit_degraded_mode", True, "ok")

    def api_logs(self, tail=30):
        return ('  File "/app/pipeline/api/app.py", line 97, in predict\n'
                "ValueError: matmul: dimension mismatch\n"
                '172.22.0.1 - - "POST /predict HTTP/1.1" 500 -\n')


class FakeCollector:
    def __init__(self):
        self.signals = Signals()

    def observe(self, window, now):
        pass

    def collect(self, window, now=None):
        return self.signals


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]


class FakeLog:
    def __init__(self):
        self.records = []

    def record(self, agent, action, model=None, details=None):
        self.records.append({"agent": agent, "action": action, "details": details})

    def latest(self, limit=20):
        return list(reversed(self.records))[:limit]


class FakeStore:
    def __init__(self):
        self.rows = []

    def record(self, now, **values):
        self.rows.append(values)


class Harness:
    def __init__(self, agent_answers=(), judge_answers=(), fail=()):
        self.api, self.t = Api(), 0
        self.llm = ScriptedLLM("agent-test", agent_answers)
        self.judge_llm = ScriptedLLM("juge-test", judge_answers)
        from agents.judge.judge import Judge
        self.actions, self.bus, self.log, self.store = FakeActions(fail), FakeBus(), FakeLog(), FakeStore()
        self.collector = FakeCollector()
        self.agent = LLMReliabilityAgent(
            llm=self.llm, judge=Judge(self.judge_llm, min_score=6), bus=self.bus, action_log=self.log,
            actions=self.actions, collector=self.collector, fetch=self.api.snapshot,
            probe=lambda: ProbeResult(10, 0, 0.02), metrics_store=self.store,
            autonomy={"restart_api": "auto", "rollback_model": "auto"})
        self.agent.run_once(now=0)          # 1re lecture : référence, pas d'appel au LLM

    def cycle(self, ok=20, errors=0):
        self.t += 15
        self.api.requests += ok + errors
        self.api.errors += errors
        self.agent.run_once(now=self.t)


    def contexts(self, phase=None):
        """Contexte (JSON) envoyé au LLM à chaque appel, éventuellement filtré par phase."""
        return [json.loads(m[1]["content"]) for m in self.llm.received if phase is None or phase_of(m) == phase]

    def phases(self):
        return [phase_of(m) for m in self.llm.received]


def phase_of(messages):
    system = messages[0]["content"]
    return ("detection" if system == DETECTION_PROMPT else "action" if system == ACTION_PROMPT
            else "verification" if system == VERIFICATION_PROMPT else "?")


def incident(cause="mauvais_modele", gravite="elevee"):
    return {"pensee": "erreurs", "incident": True, "cause": cause, "composant": "modele",
            "gravite": gravite, "justification": "100 % d'erreurs"}


def choose(choice, **extra):
    return {"pensee": "agir", "choix": choice, "justification": "x", **extra}


def verdict(v, **extra):
    return {"pensee": "vérif", "verdict": v, **extra}


NO_INCIDENT = {"pensee": "normal", "incident": False}
APPROVE = {"approved": True, "score": 8, "critique": "cohérent"}
REJECT = {"approved": False, "score": 2, "critique": "MLflow est en panne : rollback interdit"}
POSTMORTEM = {"score": 9, "critique": "rapide"}


# ------------------------------------------------------------------ phase 1 : détection
def test_first_cycle_only_measures_then_only_the_detection_question_is_asked():
    h = Harness([NO_INCIDENT])
    assert h.llm.received == []                         # 1re lecture = référence
    h.cycle(ok=18, errors=2)
    assert h.phases() == ["detection"]
    ctx = h.contexts()[-1]
    assert ctx["mesures_actuelles"]["erreurs_pct"] == 10.0
    assert ctx["signaux"]["mlflow_ok"] is True
    assert h.bus.events == []                           # le LLM a dit "pas d'incident"


def test_detected_incident_is_opened_and_an_action_is_chosen_in_the_same_cycle():
    h = Harness([incident(), choose("rollback_model")], [APPROVE])
    h.cycle(ok=0, errors=20)
    assert h.phases() == ["detection", "action"]
    opened = h.bus.events[0].payload
    assert h.bus.events[0].type == EventType.INCIDENT_OPENED
    assert opened["cause"] == "mauvais_modele" and opened["level"] == "critical"
    assert h.actions.executed == [Action.ROLLBACK_MODEL]
    assert h.agent.incident.actions[0]["judge_score"] == 8
    assert any(r["action"] == "action:rollback_model" for r in h.log.records)


def test_accents_and_case_from_small_models_are_tolerated():
    h = Harness([incident(cause="Mauvais Modèle", gravite="Élevée"), choose("Rollback_Model")], [APPROVE])
    h.cycle(ok=0, errors=20)
    assert h.agent.incident.cause == "mauvais_modele" and h.agent.incident.level == "elevee"
    assert h.actions.executed == [Action.ROLLBACK_MODEL]


def test_llm_can_ask_for_the_logs_before_deciding():
    h = Harness([{"pensee": "cause ?", "incident": False, "besoin_des_journaux": True}, incident()])
    h.cycle(ok=10, errors=10)
    detections = h.contexts("detection")
    assert len(detections) == 2
    assert detections[1]["journaux_api"]["derniere_erreur"] == "ValueError: matmul: dimension mismatch"
    assert h.agent.incident is not None


def test_version_change_is_given_as_a_fact_with_its_origin():
    h = Harness()
    h.cycle()
    h.api.version = 5
    h.cycle(ok=0, errors=20)
    change = h.contexts("detection")[-1]["dernier_changement_de_version_du_modele"]
    assert change["de"] == 4 and change["vers"] == 5 and change["origine"].startswith("externe")


def test_rollback_by_the_agent_is_flagged_as_such():
    h = Harness([incident(), choose("rollback_model")], [APPROVE])
    h.cycle(ok=0, errors=20)
    h.api.version = 3                                   # effet du rollback
    h.agent.incident = None                             # (incident clos entre-temps)
    h.cycle()
    change = h.contexts("detection")[-1]["dernier_changement_de_version_du_modele"]
    assert change["origine"] == "rollback_par_l_agent"


# ------------------------------------------------------------------ phase 3 : vérification
def test_after_an_action_the_llm_is_asked_if_it_is_repaired_with_only_post_action_measures():
    h = Harness([incident(), choose("rollback_model"), verdict("attendre")], [APPROVE])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=20)
    assert h.phases() == ["detection", "action", "verification"]   # "attendre" : pas de nouvelle action
    ctx = h.contexts("verification")[-1]
    assert ctx["avant_l_action"]["erreurs_pct"] == 100.0
    assert [m["erreurs_pct"] for m in ctx["mesures_depuis_la_derniere_action"]] == [0.0]
    assert ctx["derniere_action"]["action"] == "rollback_model"


def test_repaired_verdict_closes_the_incident_exits_degraded_mode_and_gets_a_postmortem():
    h = Harness([incident(), choose("degraded_mode"), verdict("attendre"),
                 verdict("repare", resume="réparé")], [APPROVE, POSTMORTEM])
    h.cycle(ok=0, errors=20)
    assert h.agent.incident.degraded
    h.cycle(ok=20)
    h.cycle(ok=20)
    assert h.agent.incident is None
    assert h.actions.degraded_exits == 1
    resolved = [e for e in h.bus.events if e.type == EventType.INCIDENT_RESOLVED][0].payload
    assert resolved["judge"]["score"] == 9 and resolved["resume"] == "réparé"
    assert h.phases()[-1] == "verification"             # aucune action ajoutée après la réparation


def test_not_repaired_verdict_leads_to_a_new_action_question_showing_what_failed():
    h = Harness([incident(), choose("reload_model"), verdict("attendre"),
                 verdict("pas_repare"), choose("rollback_model")], [APPROVE, APPROVE])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    tried = h.contexts("action")[-1]["actions_deja_tentees"]
    assert tried[0]["action"] == "reload_model" and tried[0]["execution_technique"] == "ok"
    assert tried[0]["erreurs_pct_avant"] == 100.0 and tried[0]["erreurs_pct_maintenant"] == 100.0
    assert h.actions.executed == [Action.RELOAD_MODEL, Action.ROLLBACK_MODEL]


def test_worse_verdict_restores_the_snapshot_then_asks_for_another_action():
    h = Harness([incident(), choose("restart_api"), verdict("aggrave"), choose("attendre")], [APPROVE])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    assert h.actions.restored == 1
    assert h.phases()[-2:] == ["verification", "action"]


def test_no_verification_before_any_measure_after_the_action():
    h = Harness([incident(), choose("rollback_model")], [APPROVE])
    h.cycle(ok=0, errors=20)
    h.agent.now = h.t
    h.agent._llm_turn()                                 # même instant : rien n'a été mesuré depuis l'action
    assert h.phases() == ["detection", "action"]


def test_verification_can_change_the_severity():
    h = Harness([incident(gravite="moyenne"), choose("rollback_model"), verdict("attendre", gravite="elevee")],
                [APPROVE])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    assert h.agent.incident.level == "elevee"


# ------------------------------------------------------------------ phase 2 : action et juge
def test_judge_rejection_is_told_to_the_llm_which_proposes_something_else():
    h = Harness([incident(), choose("rollback_model"), choose("degraded_mode")], [REJECT, APPROVE])
    h.cycle(ok=0, errors=20)
    assert h.actions.executed == [Action.DEGRADED_MODE]
    second_call = h.llm.received[-1]
    feedback = json.loads(second_call[-1]["content"])
    assert feedback["resultat"]["refusee_par_le_juge"] and "MLflow" in feedback["resultat"]["critique"]
    assert h.agent.incident.rejections[0]["action"] == "rollback_model"


def test_proposals_per_cycle_are_bounded():
    h = Harness([incident()] + [choose("rollback_model")] * 5, [REJECT] * 5)
    h.cycle(ok=0, errors=20)
    assert h.phases().count("action") == 2              # LLM_MAX_PROPOSALS
    assert h.actions.executed == []


def test_rejections_are_shown_in_the_next_action_question():
    h = Harness([incident(), choose("rollback_model"), choose("attendre"), choose("attendre")], [REJECT])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    refus = h.contexts("action")[-1]["refus_du_juge"]
    assert refus[0]["action"] == "rollback_model" and "MLflow" in refus[0]["critique_du_juge"]


def test_invalid_choice_is_reported_to_the_llm_not_crashing_the_agent():
    h = Harness([incident(), choose("redemarrer_tout"), choose("attendre")])
    h.cycle(ok=0, errors=20)
    feedback = json.loads(h.llm.received[-1][-1]["content"])
    assert "action inconnue" in feedback["resultat"]["erreur"]
    assert h.actions.executed == []


def test_the_judge_receives_signals_and_autonomy_policy():
    h = Harness([incident(), choose("restart_api")], [APPROVE])
    h.collector.signals = Signals(mlflow_ok=False)
    h.cycle(ok=0, errors=20)
    content = h.judge_llm.received[0][1]["content"]
    judge_user = json.loads(content[content.index("{"):])  # un rappel des signaux précède le JSON
    assert judge_user["contexte"]["signaux"]["mlflow_ok"] is False
    assert "signaux.mlflow_ok = False" in content           # rappelés une 2e fois, en tête du message
    assert judge_user["contexte"]["politique_autonomie"]["restart_api"] == "auto"


def test_llm_notifies_a_human_then_checks_whether_the_service_comes_back():
    h = Harness([incident(cause="postgres_indisponible"),
                 choose("prevenir_humain", message="Postgres en panne", intervention_requise=True),
                 verdict("repare", resume="Postgres revenu")], [POSTMORTEM])
    h.cycle(ok=0, errors=20)
    approval = [e for e in h.bus.events if e.type == EventType.APPROVAL_REQUESTED][0].payload
    assert approval["reason"] == "Postgres en panne" and approval["intervention_requise"] is True
    h.cycle(ok=20)
    assert h.phases()[-1] == "verification" and h.agent.incident is None


def test_wait_choice_does_nothing():
    h = Harness([incident(), choose("attendre")])
    h.cycle(ok=0, errors=20)
    assert h.actions.executed == [] and h.agent.incident is not None


# ------------------------------------------------------------------ robustesse et traces
def test_without_llm_the_agent_takes_no_decision_and_warns_a_human_once():
    h = Harness()
    h.llm.down = True
    h.cycle(ok=0, errors=20)
    h.cycle(ok=0, errors=20)
    approvals = [e for e in h.bus.events if e.type == EventType.APPROVAL_REQUESTED]
    assert len(approvals) == 1 and "LLM indisponible" in approvals[0].payload["reason"]
    assert h.actions.executed == [] and h.agent.incident is None
    h.llm.down = False
    h.cycle()
    assert any(r["action"] == "llm_recovered" for r in h.log.records)


def test_recently_closed_incident_is_shown_to_avoid_reopening():
    h = Harness([incident(), choose("rollback_model"), verdict("repare", resume="ok")], [APPROVE, POSTMORTEM])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=20)
    h.cycle(ok=20)
    closed = h.contexts("detection")[-1]["dernier_incident_clos"]
    assert closed["cause"] == "mauvais_modele"


def test_llm_answers_are_traced_during_incidents_only_and_every_cycle_is_stored():
    h = Harness([NO_INCIDENT, incident(), choose("attendre")])
    h.cycle()
    assert not [r for r in h.log.records if r["action"] == "llm_decision"]   # journal non inondé
    h.cycle(ok=0, errors=20)
    phases = [r["details"]["phase"] for r in h.log.records if r["action"] == "llm_decision"]
    assert phases == ["detection", "action"]
    assert len(h.store.rows) == 2


def test_last_python_error_is_extracted_from_the_logs():
    h = Harness()
    assert h.agent.tool_lire_journaux_api()["derniere_erreur"] == "ValueError: matmul: dimension mismatch"


def test_each_phase_sends_ollama_its_own_structured_output_schema():
    """Un schéma JSON (pas seulement "format": "json") empêche un petit modèle de recopier
    le contexte reçu au lieu de répondre : il contraint le décodage aux clés attendues."""
    h = Harness([incident(), choose("rollback_model"), verdict("attendre")], [APPROVE])
    h.cycle(ok=0, errors=20)
    h.cycle(ok=20)
    assert h.llm.schemas == [DETECTION_SCHEMA, ACTION_SCHEMA, VERIFICATION_SCHEMA]
    assert DETECTION_SCHEMA["additionalProperties"] is False
    assert ACTION_SCHEMA["additionalProperties"] is False
    assert VERIFICATION_SCHEMA["additionalProperties"] is False
