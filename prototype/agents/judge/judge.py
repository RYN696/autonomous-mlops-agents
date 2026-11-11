"""LLM as a judge, partagé par tous les agents (grille par agent dans rubrics/).

  - review()   : AVANT une action, décider si elle peut être exécutée (bloquant) ;
  - evaluate() : APRÈS un incident, noter la qualité de la résolution.

Les règles de sécurité sont écrites dans la grille et appliquées par le LLM juge ;
celles qu'un agent sait vérifier seul sur ses mesures sont aussi appliquées en code
avant le juge (ex. agents/reliability/guards.py). Le juge est un modèle d'une autre famille que celui qu'il juge,
pour ne pas valider ses propres biais. Une réponse invalide ou absente vaut un rejet."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from common import config
from common.llm import LLMClient, LLMError

RUBRICS_DIR = Path(__file__).resolve().parent / "rubrics"


@dataclass(frozen=True)
class Verdict:
    approved: bool
    score: int          # 0 à 10
    critique: str
    by: str             # nom du modèle juge


def load_rubric(agent: str) -> dict:
    with open(RUBRICS_DIR / f"{agent}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bullets(items: list) -> str:
    return "\n".join(f"- {i['nom']} : {i['question']}" if isinstance(i, dict) else f"- {i}" for i in items)


def _numbered(items: list) -> str:
    return "\n".join(f"{n}. {i['nom']} : {i['question']}" if isinstance(i, dict) else f"{n}. {i}"
                     for n, i in enumerate(items, 1))


def _signals_recap(context: dict) -> str:
    """Les règles absolues citent des champs précis (ex. signaux.mlflow_ok) noyés dans un contexte
    plus large : un petit modèle juge les retrouve bien dans un contexte minimal, mais les
    rate souvent une fois le contexte complet (mesures, historique, politique...). On les
    rappelle donc une seconde fois, en tête du message, juste avant le JSON complet."""
    signaux = context.get("signaux")
    if not isinstance(signaux, dict) or not signaux:
        return ""
    pairs = ", ".join(f"signaux.{k} = {v}" for k, v in signaux.items())
    return f"RAPPEL avant de répondre, relis ces valeurs : {pairs}.\n"


def _review_schema(n_hard_rules: int) -> dict:
    """Un petit modèle juge peut ignorer une règle absolue en l'évaluant seulement de manière
    globale/littéraire. On force donc une case true/false PAR règle, numérotée dans le même ordre
    que le texte : le juge doit se confronter à chacune individuellement avant de décider, au lieu
    de les retenir toutes en même temps dans un jugement d'ensemble."""
    return {
        "type": "object",
        "properties": {
            "regles_enfreintes": {"type": "array", "items": {"type": "boolean"},
                                  "minItems": n_hard_rules, "maxItems": n_hard_rules},
            "approved": {"type": "boolean"},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "critique": {"type": "string", "maxLength": 300},
        },
        "required": ["regles_enfreintes", "approved", "score", "critique"],
        "additionalProperties": False,
    }


class Judge:
    def __init__(self, llm: LLMClient, min_score: int | None = None):
        self.llm = llm
        self.min_score = config.JUDGE_MIN_SCORE if min_score is None else min_score

    def review(self, agent: str, proposal: dict, context: dict) -> Verdict:
        """proposal = {"action", "justification"} ; context = tout ce que l'agent sait."""
        rubric = load_rubric(agent)
        hard_rules = rubric.get("hard_rules", [])
        examples = rubric.get("hard_rule_examples", [])
        system = (
            f"{rubric['role'].strip()}\n"
            f"Critères d'évaluation :\n{_bullets(rubric['criteria'])}\n"
            f"Règles absolues, numérotées (une seule violation = rejet) :\n{_numbered(hard_rules)}\n"
            "Pour CHAQUE règle numérotée ci-dessus, dans l'ordre, relis la valeur exacte citée dans "
            "le contexte (ex. signaux.mlflow_ok, signaux.restarts_recent) : une règle qui mentionne "
            "un nombre ou un seuil ne s'applique QUE si ce nombre est vraiment atteint dans le "
            "contexte, jamais parce que l'action a déjà été proposée ou tentée une ou deux fois. "
            "\"regles_enfreintes\" doit avoir une entrée par règle, dans le même ordre : true si "
            "CETTE proposition enfreint cette règle précise, false sinon. Si au moins une entrée "
            "est true, \"approved\" DOIT être false, quel que soit le reste de ton analyse.\n"
            + (f"EXEMPLES de vérification (le format exact est imposé séparément) :\n{_bullets(examples)}\n"
               if examples else "")
            + 'Réponds UNIQUEMENT en JSON : {"regles_enfreintes": [true ou false, ...], '
              '"approved": true ou false, "score": entier de 0 à 10, '
              '"critique": "2 phrases maximum, en français"}'
        )
        user = (_signals_recap(context)
                + json.dumps({"proposition": proposal, "contexte": context}, ensure_ascii=False, default=str))
        try:
            answer = self.llm.chat_json(system, user, schema=_review_schema(len(hard_rules)))
            score = max(0, min(10, int(answer["score"])))
            approved = answer["approved"] is True and score >= self.min_score
            violations = answer.get("regles_enfreintes") or []
            if approved and any(v is True for v in violations):
                # Le juge a lui-même signalé une règle enfreinte mais approuve quand même : réponse
                # incohérente, traitée comme invalide (donc rejetée), pas une décision qu'on écrase.
                approved = False
            critique = str(answer.get("critique", "")).strip()
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            return Verdict(False, 0, f"juge indisponible ou réponse invalide ({exc})", self.llm.model)
        return Verdict(approved, score, critique, self.llm.model)

    def evaluate(self, agent: str, report: dict) -> Verdict:
        """Note la résolution d'un incident terminé (non bloquant : sert au post-mortem)."""
        rubric = load_rubric(agent)
        system = (
            f"{rubric['role'].strip()}\n"
            "L'incident est terminé. Note la qualité de sa résolution selon ces critères :\n"
            f"{_bullets(rubric.get('postmortem_criteria', rubric['criteria']))}\n"
            'Réponds UNIQUEMENT en JSON : {"score": entier de 0 à 10, '
            '"critique": "3 phrases maximum, en français : ce qui a bien marché, ce qui est à améliorer"}'
        )
        try:
            answer = self.llm.chat_json(system, json.dumps(report, ensure_ascii=False, default=str))
            score = max(0, min(10, int(answer["score"])))
            return Verdict(True, score, str(answer.get("critique", "")).strip(), self.llm.model)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            return Verdict(False, 0, f"évaluation impossible ({exc})", self.llm.model)
