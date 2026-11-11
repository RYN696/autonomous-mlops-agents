"""Gestes de l'humain, les mêmes depuis la page web et depuis Discord.

- answer   : Valider / Refuser une question (publie approval_given) ;
- rollback : « Revenir à vN » après une promotion (alias production dans MLflow, puis rechargement de l'API) ;
- ignore   : « Ignorer » une alerte (rien n'est fait, mais c'est écrit au journal).

Chaque geste est écrit dans le journal d'actions sous l'agent "humain", avec le
parcours (request_id) : il apparaît dans la chronologie de l'interface.
"""
from __future__ import annotations

from typing import Any

import requests

from common import config
from common.events import Event, EventType

AGENT = "humain"


class HumanActionError(Exception):
    """Geste impossible (production déjà changée, MLflow injoignable...) : message pour l'humain.
    stale=True : le bouton est périmé (la situation a changé), inutile de le laisser affiché."""

    def __init__(self, message: str, stale: bool = False):
        super().__init__(message)
        self.stale = stale


class HumanActions:
    def __init__(self, notifier, bus, log, registry=None, http=None):
        self.notifier = notifier
        self.bus = bus
        self.log = log
        self._registry = registry
        self.http = http or requests

    @property
    def registry(self):
        if self._registry is None:
            from mlflow.tracking import MlflowClient
            self._registry = MlflowClient()
        return self._registry

    def answer(self, request_id: str, approved: bool, who: str, source: str = "page_web") -> bool:
        """False si la question n'existe pas, a déjà une réponse ou a expiré."""
        if not self.notifier.answer(request_id, approved, who):
            return False
        self.bus.publish(Event(type=EventType.APPROVAL_GIVEN, source=source, payload={
            "request_id": request_id, "approved": approved, "approved_by": who}))
        self.record("approval_answered", {"request_id": request_id, "approved": approved, "by": who, "via": source})
        return True

    def production_version(self) -> str:
        try:
            return str(self.registry.get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS).version)
        except Exception as exc:
            raise HumanActionError(f"registre MLflow injoignable ({type(exc).__name__})") from exc

    def rollback(self, target: str, current: str, request_id: str | None, who: str, source: str) -> str:
        """Remet `target` en production, seulement si `current` y est toujours (sinon le bouton est périmé)."""
        serving = self.production_version()
        if serving != str(current):
            raise HumanActionError(f"la production est maintenant sur v{serving} (et non v{current}) : rien n'a été changé",
                                   stale=True)
        try:
            self.registry.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, str(target))
            # comme l'agent Fiabilité : cette version ne sera plus choisie pour un rollback
            self.registry.set_model_version_tag(config.MODEL_NAME, str(current), "rolled_back", "true")
        except Exception as exc:
            raise HumanActionError(f"MLflow a refusé le changement ({type(exc).__name__}: {exc})") from exc
        try:
            resp = self.http.post(f"{config.API_URL}/reload", timeout=config.ACTION_HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            reload = "API rechargée"
        except Exception as exc:  # l'alias est déjà changé : l'API le prendra à son prochain rechargement
            reload = f"rechargement de l'API en échec ({type(exc).__name__}), elle prendra v{target} à son prochain redémarrage"
        self.record("human_rollback", {"request_id": request_id, "from_version": str(current),
                                       "to_version": str(target), "by": who, "via": source, "reload": reload})
        self.notifier.notify(f"↩️ {who} a remis v{target} en production à la place de v{current} ({reload}).",
                             request_id, level="warning", actions=("report",))
        return f"v{target} est de nouveau en production ({reload})."

    def ignore(self, request_id: str | None, who: str, source: str) -> str:
        self.record("alert_ignored", {"request_id": request_id, "by": who, "via": source})
        self.notifier.notify(f"🙈 Alerte ignorée par {who} : aucune action.", request_id)
        return "Alerte ignorée : c'est noté dans le journal."

    def record(self, action: str, details: dict[str, Any]) -> None:
        self.log.record(AGENT, action, config.MODEL_NAME, details)
