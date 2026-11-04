"""Point d'entrée du service de l'agent Coûts avec boucle d'arrière-plan.

Lancer depuis le dossier prototype/ :
    python -m agents.cost.service
"""
from __future__ import annotations

import logging
import threading
import time

from common.action_log import ActionLog
from common.events import EventBus, EventType

from .agent import CostAgent
from .listener import make_handler
from .sql_store import SqlCostStore

log = logging.getLogger("agent_couts")

# L'agent n'écoute que ces deux types via l'EventBus.
LISTENED = (EventType.COST_ESTIMATE, EventType.TRAINING_COMPLETED)


class CostService:
    """Gère l'écoute des événements et l'exécution périodique des tâches de fond."""

    def __init__(
        self,
        agent: CostAgent,
        bus: EventBus,
        interval_seconds: float = 60.0,
        time_func=time.time,
    ):
        self.agent = agent
        self.bus = bus
        self.interval_seconds = interval_seconds
        self.time_func = time_func
        self._running = False
        self._thread: threading.Thread | None = None

    def run_tick(self) -> None:
        """Exécute un passage individuel des 3 tâches périodiques (découplé du temps)."""
        now = self.time_func()

        # 1. Enregistrement des métriques de l'API (Serving)
        try:
            if hasattr(self.agent, "record_serving_sample"):
                self.agent.record_serving_sample()
        except Exception as err:
            log.error("Erreur lors de record_serving_sample: %s", err)

        # 2. Vérification d'inactivité (Scale-to-zero)
        try:
            if hasattr(self.agent, "check_idle"):
                self.agent.check_idle(now=now)
        except Exception as err:
            log.error("Erreur lors de check_idle: %s", err)

        # 3. Recommandation de taille de serveur (vCPU/RAM)
        try:
            if hasattr(self.agent, "recommend_server_size"):
                self.agent.recommend_server_size()
        except Exception as err:
            log.error("Erreur lors de recommend_server_size: %s", err)

    def _periodic_loop(self) -> None:
        """Boucle d'arrière-plan exécutée dans un thread dédié."""
        while self._running:
            self.run_tick()

            # Attente réactive découpée par intervalles de 0.5s pour réagir à stop()
            elapsed = 0.0
            step = 0.5
            while self._running and elapsed < self.interval_seconds:
                time.sleep(step)
                elapsed += step

    def start(self) -> None:
        """Démarre l'écoute des événements et la boucle périodique."""
        self._running = True

        # S'abonner aux événements du bus
        self.bus.listen(make_handler(self.agent), types=LISTENED)

        # Démarrer le thread d'arrière-plan
        self._thread = threading.Thread(target=self._periodic_loop, daemon=True)
        self._thread.start()
        log.info(
            "CostService démarré (budget mensuel : %s €, intervalle : %ss)",
            self.agent.config.monthly_budget,
            self.interval_seconds,
        )

    def stop(self) -> None:
        """Arrête proprement la boucle périodique."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("CostService arrêté.")


def build_agent(
    bus: EventBus | None = None,
    action_log: ActionLog | None = None,
    store: SqlCostStore | None = None,
) -> CostAgent:
    return CostAgent(
        store=store or SqlCostStore(),
        bus=bus or EventBus(),
        action_log=action_log or ActionLog(),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    bus = EventBus()
    agent = build_agent(bus=bus)

    service = CostService(agent=agent, bus=bus, interval_seconds=60.0)
    service.start()

    # Garde le processus principal en vie
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Interruption détectée, arrêt du service...")
        service.stop()


if __name__ == "__main__":
    main()