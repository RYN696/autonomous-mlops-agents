"""Point d'entrée de l'agent Coûts.

Lancer depuis le dossier prototype/ (Redis et Postgres démarrés) :
    python -m agents.cost.service
"""
from __future__ import annotations

import logging

from common.action_log import ActionLog
from common.events import EventBus, EventType

from .agent import CostAgent
from .listener import make_handler
from .sql_store import SqlCostStore

log = logging.getLogger("agent_couts")

# L'agent n'écoute que ces deux types. Il publie cost_estimated,
# budget_exceeded et cost_recommendation, qu'il n'écoute pas : pas de boucle.
LISTENED = (EventType.COST_ESTIMATE, EventType.TRAINING_COMPLETED)


def build_agent(bus: EventBus | None = None, action_log: ActionLog | None = None,
                store: SqlCostStore | None = None) -> CostAgent:
    return CostAgent(
        store=store or SqlCostStore(),
        bus=bus or EventBus(),
        action_log=action_log or ActionLog(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    bus = EventBus()
    agent = build_agent(bus=bus)
    log.info("agent Coûts à l'écoute (budget mensuel : %s)", agent.config.monthly_budget)
    bus.listen(make_handler(agent), types=LISTENED)


if __name__ == "__main__":
    main()
