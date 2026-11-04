"""Écoute des événements pour l'agent Coûts.

CostAgent.on_event(event) reçoit déjà un Event du socle. Ce fichier ajoute
deux choses que l'agent ne fait pas lui-même :
  - convertir une période libre ("3 derniers mois") en nombre de mois ;
  - isoler les erreurs : un événement mal formé n'arrête pas l'écoute.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

log = logging.getLogger("agent_couts")


def normalize_months(value: Any) -> Optional[int]:
    """Extrait un nombre de mois d'une période libre.

    3 -> 3 ; '3' -> 3 ; '3 derniers mois' -> 3 ; '3m' -> 3 ;
    '1 an' -> 12 ; '2 years' -> 24 ; None, 0 ou texte sans chiffre -> None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value >= 1 else None
    if isinstance(value, str):
        found = re.search(r"\d+", value)
        if not found:
            return None
        number = int(found.group())
        if number <= 0:
            return None
        if re.search(r"\b(an|ans|year|years)\b", value.lower()):
            number *= 12
        return number
    return None


def with_months(event: Any) -> Any:
    """Copie de l'événement dont le payload contient 'months'.

    'months' est pris tel quel s'il existe, sinon déduit de 'period'.
    L'événement d'origine n'est pas modifié.
    """
    payload = dict(event.payload)
    payload["months"] = normalize_months(payload.get("months") or payload.get("period"))
    return event.model_copy(update={"payload": payload})


def make_handler(agent: Any) -> Callable[[Any], None]:
    """Handler pour EventBus.listen(handler, types=...)."""

    def handler(event: Any) -> None:
        try:
            agent.on_event(with_months(event))
        except Exception:  # noqa: BLE001
            log.exception("événement ignoré : %s", getattr(event.type, "value", event.type))

    return handler
