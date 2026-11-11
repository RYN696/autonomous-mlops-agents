from agents.judge.judge import Judge, load_rubric
from common.llm import LLMError


class FakeLLM:
    model = "juge-test"

    def __init__(self, answer=None, error=False):
        self.answer, self.error, self.calls = answer, error, []

    def chat_json(self, system, user, schema=None):
        self.calls.append((system, user, schema))
        if self.error:
            raise LLMError("Ollama injoignable")
        return self.answer


PROPOSAL = {"action": "restart_api", "justification": "conteneur arrêté"}


def test_safety_rules_live_in_the_rubric_and_reach_the_judge():
    rubric = load_rubric("reliability")
    assert any("MLflow" in r for r in rubric["hard_rules"])
    llm = FakeLLM({"approved": True, "score": 8, "critique": "ok", "regles_enfreintes": [False] * 6})
    Judge(llm).review("reliability", PROPOSAL, {"signaux": {"mlflow_ok": True}})
    system, user, schema = llm.calls[0]
    assert "MLflow" in system and "Postgres" in system   # règles transmises au LLM juge
    assert "mlflow_ok" in user                          # contexte transmis
    assert "1. " in system and "6. " in system          # règles numérotées (une entrée par règle)
    assert schema["properties"]["regles_enfreintes"]["minItems"] == 6


def test_approval_needs_the_minimum_score():
    assert Judge(FakeLLM({"approved": True, "score": 8}), min_score=6).review("reliability", PROPOSAL, {}).approved
    assert not Judge(FakeLLM({"approved": True, "score": 4}), min_score=6).review("reliability", PROPOSAL, {}).approved
    assert not Judge(FakeLLM({"approved": False, "score": 9})).review("reliability", PROPOSAL, {}).approved


def test_unavailable_or_invalid_judge_means_rejection():
    assert not Judge(FakeLLM(error=True)).review("reliability", PROPOSAL, {}).approved
    assert not Judge(FakeLLM({"approved": "oui", "score": 9})).review("reliability", PROPOSAL, {}).approved
    assert not Judge(FakeLLM({"approved": True})).review("reliability", PROPOSAL, {}).approved


def test_approval_is_rejected_when_the_judge_flags_a_rule_violation_itself():
    """Le juge (petit modèle) peut signaler qu'une règle absolue est enfreinte tout en approuvant
    quand même, de manière incohérente (ex. hors-sujet, règle MLflow oubliée dans le score global).
    Une réponse qui se contredit ainsi est traitée comme invalide : rejet, comme documenté."""
    answer = {"regles_enfreintes": [True] + [False] * 5, "approved": True, "score": 9, "critique": "ok"}
    assert not Judge(FakeLLM(answer)).review("reliability", PROPOSAL, {}).approved


def test_no_rule_flagged_lets_the_judges_approval_stand():
    answer = {"regles_enfreintes": [False] * 6, "approved": True, "score": 8, "critique": "ok"}
    assert Judge(FakeLLM(answer)).review("reliability", PROPOSAL, {}).approved


def test_postmortem_evaluation():
    verdict = Judge(FakeLLM({"score": 7, "critique": "rapide"})).evaluate("reliability", {"actions": []})
    assert verdict.score == 7 and verdict.critique == "rapide"
