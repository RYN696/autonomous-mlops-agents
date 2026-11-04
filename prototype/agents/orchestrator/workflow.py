"""Graphe LangGraph d'une demande de réentraînement : un fil (thread) par request_id.

Une demande naît d'une dérive (drift_detected) ou d'une commande chat (state["command"]).

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


def eur(x: float | None) -> str:
    """Montants souvent minuscules (tarifs fictifs) : 4 décimales sous 1 €."""
    x = x or 0.0
    return f"{x:.2f} €" if abs(x) >= 1 else f"{x:.4f} €"


class RetrainState(TypedDict, total=False):
    request_id: str
    drift: dict[str, Any]     # payload de drift_detected
    command: dict[str, Any]   # demande en chat : text, by, months, model, period_label
    cost: dict[str, Any]      # payload de cost_estimated
    decision: dict[str, Any]  # action, règle, coût
    answer: dict[str, Any]    # payload de approval_given
    outcome: str              # corrupted, ignored, retrain_requested, refused, forbidden


def build_retrain_graph(orch: "Orchestrator", checkpointer):
    def check_data(state: RetrainState) -> RetrainState:
        p, request_id, command = state.get("drift") or {}, state["request_id"], state.get("command")
        if not command:
            # Traçabilité : relie la demande au contrôle de l'agent Qualité (check_id)
            orch.record("drift_received", {"request_id": request_id, "check_id": p.get("check_id"),
                                           "reason": p.get("reason"), "drift_score": p.get("drift_score"),
                                           "period": p.get("period"), "suspect": p.get("suspect"),
                                           "impossible_share": p.get("impossible_share")})
        if p.get("reason") == "corrupted_input_data":
            # On n'entraîne jamais sur des données cassées
            details = {"request_id": request_id, "cause": "corrupted_input_data",
                       "invalid_share": p.get("invalid_share"), "invalid_reasons": p.get("invalid_reasons")}
            orch.publish(EventType.INCIDENT_OPENED, Severity.CRITICAL, details)
            orch.record("retrain_refused_corrupted_data", details)
            orch.notifier.notify(
                f"🚨 Données corrompues ({p.get('invalid_share', 0):.0%} de lignes invalides) : "
                "pas de réentraînement. Il faut vérifier la source des données.", request_id,
                level="alert", actions=("data", "ignore", "dashboard"))
            return {"outcome": "corrupted"}
        if orch.busy():
            orch.record("drift_ignored", {"request_id": request_id, "reason": "une demande est déjà en cours"})
            if command:  # un humain attend une réponse
                orch.notifier.notify("⏳ Une demande de réentraînement est déjà en cours : réessayez quand elle "
                                     "sera terminée.", request_id, level="warning")
            return {"outcome": "ignored"}
        if orch.incident_in_progress():
            orch.record("drift_ignored", {"request_id": request_id, "reason": "incident en cours",
                                          "incidents": list(orch.open_incidents)})
            orch.notifier.notify(
                "⏸️ Incident en cours : pas de réentraînement pour l'instant. Relancez la demande une fois "
                "l'incident résolu." if command else
                "⏸️ Dérive détectée pendant un incident : pas de réentraînement pour l'instant. "
                "L'agent Qualité la signalera de nouveau au prochain contrôle.", request_id, level="warning")
            return {"outcome": "ignored"}
        return {}

    def ask_cost(state: RetrainState) -> RetrainState:
        command = state.get("command") or {}
        months = command.get("months") or config.DEFAULT_TRAIN_MONTHS
        orch.publish(EventType.COST_ESTIMATE, Severity.INFO, {
            "request_id": state["request_id"], "requested_by": AGENT, "months": months})
        orch.record("cost_requested", {"request_id": state["request_id"], "months": months,
                                       "drift_score": (state.get("drift") or {}).get("drift_score")})
        return {}

    def wait_cost(state: RetrainState) -> RetrainState:
        return {"cost": interrupt({"waiting_for": EventType.COST_ESTIMATED.value})}

    def decide(state: RetrainState) -> RetrainState:
        p = state["cost"]
        action = "retrain_over_budget" if p.get("exceeds_budget") else "retrain"
        estimate = p["options"]["standard"]["cost"]
        details = {"request_id": state["request_id"], "action": action, "rule": orch.rules.can(action),
                   "estimated_cost": estimate, "month_total": p.get("month_total"), "budget": p.get("budget"),
                   "over_by": round((p.get("month_total") or 0) + estimate - (p.get("budget") or 0), 6)
                   if p.get("exceeds_budget") else 0.0}
        orch.record("retrain_decision", details)
        return {"decision": details}

    def ask_human(state: RetrainState) -> RetrainState:
        d = state["decision"]
        orch.publish(EventType.APPROVAL_REQUESTED, Severity.WARNING, d)
        command, drift = state.get("command"), state.get("drift") or {}
        if command:
            what = (f"Réentraîner le modèle taxi — demandé par {command['by']} — période : "
                    f"{command['period_label']} — raison : demande en chat « {command['text'][:160]} »")
        else:
            on = f" sur {drift['period']}" if drift.get("period") else ""
            what = f"Réentraîner le modèle taxi — demandé par l'agent Qualité — raison : dérive détectée{on}"
        over = (f" : au-dessus du budget de {eur(d['over_by'])}" if d.get("over_by") else "")
        orch.notifier.ask(
            f"{what}. Coût estimé {eur(d['estimated_cost'])}{over} (déjà dépensé ce mois : "
            f"{eur(d.get('month_total'))}, budget {eur(d.get('budget'))}). On réentraîne ?",
            state["request_id"])
        return {}

    def wait_human(state: RetrainState) -> RetrainState:
        return {"answer": interrupt({"waiting_for": EventType.APPROVAL_GIVEN.value})}

    def retrain(state: RetrainState) -> RetrainState:
        drift = state.get("drift") or {}
        orch.request_retrain(state["request_id"], approved_by=state.get("answer", {}).get("approved_by"),
                             period=drift.get("period"), cause=drift.get("explanation"),
                             command=state.get("command"), drift=drift)
        return {"outcome": "retrain_requested"}

    def refused(state: RetrainState) -> RetrainState:
        who = state["answer"].get("approved_by", "?")
        orch.record("retrain_refused", {"request_id": state["request_id"], "refused_by": who})
        orch.notifier.notify(f"Réentraînement refusé par {who}. La production ne change pas.", state["request_id"])
        return {"outcome": "refused"}

    def forbidden(state: RetrainState) -> RetrainState:
        orch.notifier.notify(f"⛔ Réentraînement interdit par les règles ({state['decision']['action']}).",
                             state["request_id"], level="warning")
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
