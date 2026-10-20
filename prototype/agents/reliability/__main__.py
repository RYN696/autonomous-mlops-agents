"""Lancement : python -m agents.reliability (depuis prototype/).

L'agent est piloté par un LLM local (Ollama) ; un second LLM, d'une autre famille,
joue le rôle de juge. Modèles réglables dans .env (LLM_AGENT_MODEL, LLM_JUDGE_MODEL)."""
import logging

from agents.judge.judge import Judge
from agents.reliability.llm_agent import LLMReliabilityAgent
from common.llm import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

LLMReliabilityAgent(llm=get_client("agent"), judge=Judge(get_client("judge"))).run_forever()
