"""Catalogue fermé des actions de l'agent Fiabilité.

Ce sont les seules actions que l'agent sait exécuter : le LLM choisit parmi
elles, le juge les évalue. La description et le risque sont donnés aux deux LLM
comme information (aucune règle de code ne s'en sert pour décider)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    RESTART_API = "restart_api"
    RECREATE_API = "recreate_api"
    RELOAD_MODEL = "reload_model"
    ROLLBACK_MODEL = "rollback_model"
    DEGRADED_MODE = "degraded_mode"
    RESTART_REDIS = "restart_redis"
    RESTART_MLFLOW = "restart_mlflow"


@dataclass(frozen=True)
class ActionSpec:
    description: str  # texte lisible, réutilisé dans le prompt du LLM et du juge
    risk: str         # "low" ou "high"
    reversible: bool


CATALOG: dict[Action, ActionSpec] = {
    Action.RESTART_API: ActionSpec("Redémarrer le conteneur de l'API", "low", True),
    Action.RECREATE_API: ActionSpec("Recréer le conteneur de l'API à partir de son image", "low", True),
    Action.RELOAD_MODEL: ActionSpec("Recharger le modèle en production depuis MLflow (/reload)", "low", True),
    Action.ROLLBACK_MODEL: ActionSpec("Remettre en production la dernière version saine du modèle", "high", True),
    Action.DEGRADED_MODE: ActionSpec("Servir un modèle de secours simple en attendant la réparation", "low", True),
    Action.RESTART_REDIS: ActionSpec("Redémarrer Redis (bus d'événements)", "low", True),
    Action.RESTART_MLFLOW: ActionSpec("Redémarrer MLflow (registre des modèles)", "high", True),
}

LOW_RISK_ACTIONS = frozenset(a for a, spec in CATALOG.items() if spec.risk == "low")