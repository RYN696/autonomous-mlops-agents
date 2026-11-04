"""Commandes en langage naturel (canal #mlops-agents de l'interface) -> action précise.

    « @agents retrain le modèle taxi sur les 3 derniers mois »
    -> {action: retrain, modèle: taxi-duration, mois: 3}

Le LLM (Ollama, CHAT_LLM_MODEL) ne fait que COMPRENDRE la phrase. Le code vérifie
tout ce qu'il renvoie (action connue, modèle connu, nombre de mois borné) ; si le LLM
ne répond pas ou répond n'importe quoi, des règles simples (mots-clés) prennent le
relais. La décision de lancer reste au graphe de l'orchestrateur (coût, règles
d'autonomie, validation humaine) : une phrase ne contourne jamais le budget.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from common import config
from common.llm import LLMError, OllamaClient

RETRAIN, UNKNOWN = "retrain", "inconnue"

SYSTEM = (
    "Tu traduis une demande d'une équipe MLOps en JSON. Actions possibles : "
    "\"retrain\" (réentraîner un modèle) ou \"inconnue\" (tout le reste). "
    "\"modele\" : le modèle cité (le modèle des taxis s'appelle \"taxi\"), ou \"\" s'il n'est pas cité. "
    "\"mois\" : le nombre de mois de données demandés (entier), ou 0 s'il n'est pas précisé. "
    "Réponds uniquement avec le JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": [RETRAIN, UNKNOWN]},
                   "modele": {"type": "string"}, "mois": {"type": "integer"}},
    "required": ["action", "modele", "mois"],
}
NUMBERS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
           "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "douze": 12}
RETRAIN_WORDS = ("retrain", "reentrain", "re-entrain", "entraine", "entrainer", "retrainer")


@dataclass
class Command:
    action: str
    model: str | None
    months: int | None
    understood_by: str            # "llm" ou "règles" (LLM indisponible ou réponse rejetée)
    problem: str | None = None    # pourquoi la commande ne peut pas être lancée

    @property
    def ok(self) -> bool:
        return self.action == RETRAIN and self.problem is None

    @property
    def period_label(self) -> str:
        n = self.months or config.DEFAULT_TRAIN_MONTHS
        return "le dernier mois" if n == 1 else f"les {n} derniers mois"

    def as_dict(self) -> dict:
        return {**asdict(self), "period_label": self.period_label}


def plain(text: str) -> str:
    """minuscules, sans accents : « Réentraîne » -> « reentraine »."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def months_in(text: str) -> int | None:
    t = plain(text)
    m = re.search(r"(\d+|" + "|".join(NUMBERS) + r")\s+(?:derniers?\s+)?mois", t)
    if m:
        word = m.group(1)
        return int(word) if word.isdigit() else NUMBERS[word]
    if re.search(r"(dernier|ce)\s+mois", t):
        return 1
    return None


def model_in(text: str) -> str | None:
    """Modèle cité : "taxi" -> le nom MLflow ; un autre modèle -> son nom (refusé ensuite) ; rien -> None."""
    t = plain(text)
    if "taxi" in t or config.MODEL_NAME in t:
        return config.MODEL_NAME
    m = re.search(r"modele\s+([a-z0-9_-]+)", t)
    if m and m.group(1) not in ("de", "du", "le", "la", "sur"):
        return m.group(1)
    return None


def parse_rules(text: str) -> Command:
    action = RETRAIN if any(w in plain(text) for w in RETRAIN_WORDS) else UNKNOWN
    return check(Command(action, model_in(text), months_in(text), "règles"))


def check(cmd: Command) -> Command:
    """Ce que le code impose, quelle que soit la source : un seul modèle connu, une période bornée."""
    if cmd.action != RETRAIN:
        cmd.problem = "je sais seulement lancer un réentraînement (ex. « retrain le modèle taxi sur les 3 derniers mois »)"
        return cmd
    if cmd.model is None:
        cmd.model = config.MODEL_NAME          # un seul modèle dans le projet : c'est celui-là
    if cmd.model != config.MODEL_NAME:
        cmd.problem = f"modèle « {cmd.model} » inconnu : le seul modèle suivi est {config.MODEL_NAME} (taxi)"
    elif cmd.months is not None and not 1 <= cmd.months <= config.CHAT_MAX_MONTHS:
        cmd.problem = f"période de {cmd.months} mois refusée : entre 1 et {config.CHAT_MAX_MONTHS} mois"
    return cmd


def from_llm(answer: dict, text: str) -> Command | None:
    """Réponse du LLM vérifiée ; None si elle contredit la phrase (on passe alors aux règles)."""
    action = answer.get("action")
    if action not in (RETRAIN, UNKNOWN):
        return None
    raw_model = plain(str(answer.get("modele") or "")).strip()
    model = None if not raw_model else config.MODEL_NAME if "taxi" in raw_model or raw_model == config.MODEL_NAME \
        else raw_model
    try:
        months = int(answer.get("mois") or 0) or None
    except (TypeError, ValueError):
        return None
    said = months_in(text)
    if said is not None and months != said:   # un nombre écrit dans la phrase prime sur le LLM
        return None
    return check(Command(action, model, months, "llm"))


class ChatParser:
    def __init__(self, llm=None):
        self.llm = llm

    def client(self):
        if self.llm is None:
            self.llm = OllamaClient(config.CHAT_LLM_MODEL, timeout=config.CHAT_LLM_TIMEOUT_SECONDS)
        return self.llm

    def parse(self, text: str) -> Command:
        try:
            answer = self.client().chat_json(SYSTEM, text, schema=SCHEMA)
        except LLMError:
            return parse_rules(text)
        return from_llm(answer, text) or parse_rules(text)
