"""Lancement : python -m agents.orchestrator (depuis prototype/).

Trois choses tournent en même temps :
- l'orchestrateur écoute Redis (dans un fil à part) ;
- le bot Discord, si DISCORD_BOT_TOKEN est renseigné (dans un fil à part) ;
- la page web répond sur http://localhost:8080 (graphes LangGraph sur /graph)."""
import threading

from agents.orchestrator import discord_bot
from agents.orchestrator.agent import Orchestrator
from agents.orchestrator.human import HumanActions
from agents.orchestrator.notifier import Notifier
from agents.orchestrator.web import create_app
from common import config
from common.action_log import ActionLog
from common.events import EventBus

notifier = Notifier()
log = ActionLog()
human = HumanActions(notifier, EventBus(), log)  # Valider, Refuser, Revenir à vN, Ignorer
relay = discord_bot.from_config(human)
if relay is not None:
    notifier.relay = relay
    relay.start()
orchestrator = Orchestrator(bus=EventBus(), log=log, notifier=notifier)
orchestrator.expire_orphans()
threading.Thread(target=orchestrator.run, daemon=True).start()

create_app(notifier, EventBus(), log, orchestrator, human=human, discord=relay).run(host="0.0.0.0", port=config.ORCHESTRATOR_WEB_PORT)