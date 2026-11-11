"""Mode service de l'agent qualité (lancé par docker compose).

Contrôle le mois surveillé toutes les QUALITY_CHECK_MINUTES, et tout de suite
quand l'agent Retrain met un nouveau modèle en production (model_promoted) :
la dérive se mesure toujours par rapport au modèle servi.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from common.events import Event, EventType

log = logging.getLogger("agent_qualite")


def watch(check: Callable[[], None], bus, interval_seconds: float,
          stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    wake = threading.Event()

    def on_promoted(event: Event) -> None:
        log.info("Nouveau modèle en production (v%s) : contrôle immédiat",
                 event.payload.get("challenger_version"))
        wake.set()

    threading.Thread(target=bus.listen, args=(on_promoted,),
                     kwargs={"types": [EventType.MODEL_PROMOTED]}, daemon=True).start()
    while True:
        try:
            check()
        except Exception:  # un contrôle raté ne doit pas arrêter le service
            log.exception("Contrôle qualité en échec")
        if stop.is_set():
            return
        wake.wait(interval_seconds)
        wake.clear()
