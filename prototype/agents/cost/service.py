"""Point d'entrée du service de l'agent Coûts avec boucle d'arrière-plan.

Lancer depuis le dossier prototype/ :
    python -m agents.cost          (ou python -m agents.cost.service)

Deux fils tournent en même temps :
  - l'écoute Redis (cost_estimate, training_completed) ;
  - la boucle de fond, toutes les COST_TICK_SECONDS :
      1. coût de l'API servie (CPU du conteneur lu dans Docker) ;
      2. inactivité de l'API (requêtes utilisateurs, sondes de Fiabilité exclues)
         -> recommandation scale-to-zero, une fois par période d'inactivité ;
      3. taille du serveur d'après le CPU mesuré -> recommandation si elle change.
Les recommandations ne sont envoyées qu'une fois : l'humain les voit sur la page
de l'orchestrateur, inutile de les répéter à chaque tour.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families

from common import config
from common.action_log import ActionLog
from common.events import EventBus, EventType

from . import recommender
from .agent import CostAgent
from .collector import DockerStatsCollector
from .listener import make_handler
from .sql_store import SqlCostStore

log = logging.getLogger("agent_couts")

# L'agent n'écoute que ces deux types via l'EventBus.
LISTENED = (EventType.COST_ESTIMATE, EventType.TRAINING_COMPLETED)


def user_requests(api_url: str = config.API_URL, http=requests) -> Optional[float]:
    """Compteur cumulé des requêtes utilisateurs de l'API, ou None si elle ne répond pas."""
    try:
        resp = http.get(f"{api_url}/metrics", timeout=config.HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return sum(s.value for family in text_string_to_metric_families(resp.text)
               for s in family.samples if s.name == "api_user_requests_total")


class CostService:
    """Gère l'écoute des événements et l'exécution périodique des tâches de fond."""

    def __init__(
        self,
        agent: CostAgent,
        bus: EventBus,
        interval_seconds: float = config.COST_TICK_SECONDS,
        time_func: Callable[[], float] = time.time,
        api_usage: Callable[[], Optional[float]] = user_requests,
        model: str = config.MODEL_NAME,
        container: str = config.API_CONTAINER_NAME,
        server_size: str = config.COST_SERVER_SIZE,
    ):
        self.agent = agent
        self.bus = bus
        self.interval_seconds = interval_seconds
        self.time_func = time_func
        self.api_usage = api_usage
        self.model = model
        self.container = container
        self.server_size = server_size
        self._running = False
        self._thread: threading.Thread | None = None
        self._listener: threading.Thread | None = None
        self._last_tick: float | None = None
        self._last_requests: float | None = None
        self._last_request_at = time_func()  # historique inconnu au démarrage : on part de maintenant
        self._idle_reported = False
        self._last_size: tuple[str, str] | None = None  # (action, taille) de la dernière recommandation

    def run_tick(self) -> None:
        """Exécute un passage des 3 tâches périodiques. Une tâche en erreur n'empêche pas les autres."""
        now = self.time_func()
        elapsed = now - self._last_tick if self._last_tick is not None else None
        self._last_tick = now

        used_vcpus = None
        try:
            used_vcpus = self.record_serving(elapsed)
        except Exception as err:
            log.error("Erreur lors de record_serving_sample: %s", err)
        try:
            self.check_idle(now)
        except Exception as err:
            log.error("Erreur lors de check_idle: %s", err)
        try:
            if used_vcpus is not None:
                self.check_size(used_vcpus)
        except Exception as err:
            log.error("Erreur lors de recommend_server_size: %s", err)

    # 1. Coût de l'API servie ; renvoie le CPU utilisé (vCPU) depuis le tour précédent
    def record_serving(self, elapsed: float | None) -> float | None:
        if self.agent.collector is None:
            return None
        self.agent.record_serving_sample(self.model, self.container)
        delta = self.agent.last_cpu_delta
        if delta is None or not elapsed:
            return None  # 1er tour ou conteneur redémarré : pas encore de mesure
        return delta / elapsed

    # 2. Inactivité -> scale-to-zero, une seule fois par période d'inactivité
    def check_idle(self, now: float) -> None:
        count = self.api_usage()
        if count is None:
            return  # API injoignable : c'est l'affaire de l'agent Fiabilité
        if self._last_requests is not None and count != self._last_requests:
            self._last_request_at = now  # nouvelles requêtes (ou compteur remis à zéro)
            self._idle_reported = False
        self._last_requests = count
        if self._idle_reported:
            return
        idle_hours = (now - self._last_request_at) / 3600
        rec = self.agent.check_idle(self.model, datetime.fromtimestamp(self._last_request_at),
                                    self.server_size, round(idle_hours, 3))
        self._idle_reported = rec is not None

    # 3. Taille du serveur, seulement quand la recommandation change
    def check_size(self, used_vcpus: float) -> None:
        rec = recommender.recommend_size(used_vcpus, self.server_size, self.agent.config)
        key = (rec["action"], rec["recommended"])
        if key != self._last_size and rec["action"] != "keep":
            self.agent.recommend_server_size(self.model, used_vcpus, self.server_size)
        self._last_size = key

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
        """Démarre la boucle périodique puis l'écoute des événements (non bloquant)."""
        self._running = True

        # Démarrer le thread d'arrière-plan
        self._thread = threading.Thread(target=self._periodic_loop, daemon=True)
        self._thread.start()

        # S'abonner aux événements du bus : listen() bloque, il a donc son propre thread
        self._listener = threading.Thread(target=self.bus.listen, args=(make_handler(self.agent),),
                                          kwargs={"types": LISTENED}, daemon=True)
        self._listener.start()
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
    try:
        collector = DockerStatsCollector()
    except Exception as err:  # pas de socket Docker : pas de coût de l'API servie
        log.warning("Docker injoignable (%s) : coût de l'API servie désactivé", err)
        collector = None
    return CostAgent(
        store=store or SqlCostStore(),
        bus=bus or EventBus(),
        action_log=action_log or ActionLog(),
        collector=collector,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    bus = EventBus()
    agent = build_agent(bus=bus)

    service = CostService(agent=agent, bus=bus)
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
