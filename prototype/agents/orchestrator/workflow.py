"""Graphe LangGraph d'une demande de réentraînement : un fil (thread) par request_id.

    check_data ─┬─ données corrompues / demande déjà en cours / incident en cours ─► fin
                └─► ask_cost ─► wait_cost ─► decide ─┬─ auto ──────────────────────────► retrain
                                                     ├─ approval ─► ask_human ─► wait_human ─┬─► retrain
                                                     │                                       └─► refused
                                                     └─ forbidden ─► forbidden

wait_cost et wait_human appellent interrupt() : le fil est mis en pause dans le
checkpointer, puis repris par Command(resume=payload) quand cost_estimated ou
approval_given arrive. Ces nœuds ne font rien d'autre, car LangGraph rejoue
le nœud en entier à la reprise.

La décision reste déterministe (rules/autonomy.yaml) : aucun LLM dans ce graphe.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.orchestrator.rules import APPROVAL, AUTO
from common import config
from common.events import EventType, Severity

if TYPE_CHECKING:
    from agents.orchestrator.agent import Orchestrator

AGENT = "orchestrator"

# Nœud en pause -> événement qui le réveille
WAITING_FOR = {"wait_cost": EventType.COST_ESTIMATED, "wait_human": EventType.APPROVAL_GIVEN}


class RetrainState(TypedDict, total=False):
    request_id: str
    drift: dict[str, Any]     # payload de drift_detected
    cost: dict[str, Any]      # payload de cost_estimated
    decision: dict[str, Any]  # action, règle, coût
    answer: dict[str, Any]    # payload de approval_given
    outcome: str              # corrupted, ignored, retrain_requested, refused, forbidden


def build_retrain_graph(orch: "Orchestrator", checkpointer):
    def check_data(state: RetrainState) -> RetrainState:
        p = state["drift"]
        if p.get("reason") == "corrupted_input_data":
            # On n'entraîne jamais sur des données cassées
            details = {"cause": "corrupted_input_data", "invalid_share": p.get("invalid_share"),
                       "invalid_reasons": p.get("invalid_reasons")}
            orch.publish(EventType.INCIDENT_OPENED, Severity.CRITICAL, details)
            orch.record("retrain_refused_corrupted_data", details)
            orch.notifier.notify(
                f"🚨 Données corrompues ({p.get('invalid_share', 0):.0%} de lignes invalides) : "
                "pas de réentraînement. Il faut vérifier la source des données.")
            return {"outcome": "corrupted"}
        if orch.busy():
            orch.record("drift_ignored", {"reason": "une demande est déjà en cours"})
            return {"outcome": "ignored"}
        if orch.incident_in_progress():
            orch.record("drift_ignored", {"reason": "incident en cours", "incidents": list(orch.open_incidents)})
            orch.notifier.notify("⏸️ Dérive détectée pendant un incident : pas de réentraînement pour "
                                 "l'instant. L'agent Qualité la signalera de nouveau au prochain contrôle.")
            return {"outcome": "ignored"}
        return {}

    def ask_cost(state: RetrainState) -> RetrainState:
        orch.publish(EventType.COST_ESTIMATE, Severity.INFO, {
            "request_id": state["request_id"], "requested_by": AGENT, "months": config.DEFAULT_TRAIN_MONTHS})
        orch.record("cost_requested", {"request_id": state["request_id"],
                                       "drift_score": state["drift"].get("drift_score")})
        return {}

    def wait_cost(state: RetrainState) -> RetrainState:
        return {"cost": interrupt({"waiting_for": EventType.COST_ESTIMATED.value})}

    def decide(state: RetrainState) -> RetrainState:
        p = state["cost"]
        action = "retrain_over_budget" if p.get("exceeds_budget") else "retrain"
        details = {"request_id": state["request_id"], "action": action, "rule": orch.rules.can(action),
                   "estimated_cost": p["options"]["standard"]["cost"],
                   "month_total": p.get("month_total"), "budget": p.get("budget")}
        orch.record("retrain_decision", details)
        return {"decision": details}

    def ask_human(state: RetrainState) -> RetrainState:
        d = state["decision"]
        orch.publish(EventType.APPROVAL_REQUESTED, Severity.WARNING, d)
        orch.notifier.ask(
            f"Dérive détectée. Réentraîner coûte {d['estimated_cost']:.2f} € ; ce mois : "
            f"{d.get('month_total') or 0:.2f} € sur un budget de {d.get('budget') or 0:.2f} €. On réentraîne ?",
            state["request_id"])
        return {}

    def wait_human(state: RetrainState) -> RetrainState:
        return {"answer": interrupt({"waiting_for": EventType.APPROVAL_GIVEN.value})}

    def retrain(state: RetrainState) -> RetrainState:
        orch.request_retrain(state["request_id"], approved_by=state.get("answer", {}).get("approved_by"),
                             period=state["drift"].get("period"))
        return {"outcome": "retrain_requested"}

    def refused(state: RetrainState) -> RetrainState:
        who = state["answer"].get("approved_by", "?")
        orch.record("retrain_refused", {"request_id": state["request_id"], "refused_by": who})
        orch.notifier.notify(f"Réentraînement refusé par {who}. La production ne change pas.")
        return {"outcome": "refused"}

    def forbidden(state: RetrainState) -> RetrainState:
        orch.notifier.notify(f"⛔ Réentraînement interdit par les règles ({state['decision']['action']}).")
        return {"outcome": "forbidden"}

    def after_check(state: RetrainState) -> str:
        return END if state.get("outcome") else "ask_cost"

    def after_decide(state: RetrainState) -> str:
        rule = state["decision"]["rule"]
        return "retrain" if rule == AUTO else "ask_human" if rule == APPROVAL else "forbidden"

    def after_human(state: RetrainState) -> str:
        return "retrain" if state["answer"].get("approved") else "refused"

    g = StateGraph(RetrainState)
    for node in (check_data, ask_cost, wait_cost, decide, ask_human, wait_human, retrain, refused, forbidden):
        g.add_node(node.__name__, node)
    g.add_edge(START, "check_data")
    g.add_conditional_edges("check_data", after_check, ["ask_cost", END])
    g.add_edge("ask_cost", "wait_cost")
    g.add_edge("wait_cost", "decide")
    g.add_conditional_edges("decide", after_decide, ["retrain", "ask_human", "forbidden"])
    g.add_edge("ask_human", "wait_human")
    g.add_conditional_edges("wait_human", after_human, ["retrain", "refused"])
    for node in ("retrain", "refused", "forbidden"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)
