"""Les 5 agents tels que l'interface les présente : nom dans le journal d'actions,
service docker compose, rôle et événements (voir common/events.py)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentInfo:
    key: str                 # dans l'URL : /agents/<key>
    name: str
    log_name: str            # colonne agent du journal d'actions
    service: str             # service docker compose
    color: str               # classe CSS
    role: str
    listens: tuple[str, ...]
    publishes: tuple[str, ...]


AGENTS: tuple[AgentInfo, ...] = (
    AgentInfo("quality", "Qualité", "agent_qualite", "quality", "a-quality",
              "Valide les données du mois et mesure la dérive par rapport au modèle en production.",
              ("model_promoted",), ("drift_detected", "quality_ok")),
    AgentInfo("orchestrator", "Orchestrateur", "orchestrator", "orchestrator", "a-orchestrator",
              "Graphe LangGraph : décide de la suite selon rules/autonomy.yaml et demande l'avis d'un humain.",
              ("drift_detected", "cost_estimated", "approval_given", "model_registered", "model_promoted",
               "promotion_blocked", "action_logged", "incident_opened", "incident_resolved",
               "approval_requested", "budget_exceeded", "cost_recommendation", "cost_anomaly"),
              ("cost_estimate", "retrain_requested", "approval_requested", "training_completed",
               "incident_opened")),
    AgentInfo("cost", "Coûts", "cost", "cost", "a-cost",
              "Estime le coût d'un entraînement, suit le budget du mois et le coût de l'API servie, signale les pics.",
              ("cost_estimate", "training_completed"),
              ("cost_estimated", "budget_exceeded", "cost_recommendation", "cost_anomaly")),
    AgentInfo("retrain", "Retrain", "agent_retrain", "retrain", "a-retrain",
              "Entraîne un challenger, le compare à la production, le promeut ou le bloque.",
              ("retrain_requested",),
              ("model_registered", "model_promoted", "promotion_blocked", "action_logged", "incident_opened")),
    AgentInfo("reliability", "Fiabilité", "agent_fiabilite", "reliability", "a-reliability",
              "Surveille l'API, MLflow et Redis ; un LLM propose, un juge et des règles valident, puis il répare.",
              (), ("incident_opened", "incident_resolved", "approval_requested")),
)

BY_KEY = {a.key: a for a in AGENTS}
BY_LOG_NAME = {a.log_name: a for a in AGENTS}
HUMAN = AgentInfo("human", "Humain", "humain", "", "a-human",
                  "Valide ou refuse, revient à une version ou ignore une alerte (page web ou Discord).", (), ())
DEMO = AgentInfo("demo", "Banc de test", "demo", "", "a-demo", "Scénarios lancés depuis la page Scénarios.", (), ())


def agent_of(log_name: str) -> AgentInfo:
    if log_name == DEMO.log_name:
        return DEMO
    if log_name == HUMAN.log_name:
        return HUMAN
    return BY_LOG_NAME.get(log_name) or AgentInfo(log_name, log_name, log_name, "", "a-other", "", (), ())
