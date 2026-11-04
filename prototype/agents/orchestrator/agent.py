"""Orchestrateur : le chef d'orchestre des 4 agents, construit avec LangGraph.

Il n'entraîne, ne répare et ne calcule rien lui-même. Chaque événement reçu
passe dans un graphe routeur (un nœud par type d'événement). La demande de
réentraînement, qui attend le coût puis parfois un humain, est un second
graphe avec pause/reprise (voir workflow.py). Les décisions suivent
rules/autonomy.yaml et l'humain est prévenu sur la page web (notifier).

Écoute : drift_detected, cost_estimated, approval_given, model_registered,
         model_promoted, promotion_blocked, action_logged, incident_opened,
         incident_resolved, approval_requested, budget_exceeded,
         cost_recommendation
Publie : cost_estimate, retrain_requested, approval_requested,
         training_completed, incident_opened (données corrompues)
"""
from __future__ import annotations

import time
import traceback
import uuid
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.orchestrator.rules import Rules
from agents.orchestrator.workflow import AGENT, WAITING_FOR, build_retrain_graph
from common import config
from common.events import Event, EventType, Severity

RETRAIN = "agent_retrain"  # nom de source de l'agent Retrain
RELIABILITY = "agent_fiabilite"

LISTENED = [
    EventType.DRIFT_DETECTED, EventType.COST_ESTIMATED, EventType.APPROVAL_GIVEN,
    EventType.MODEL_REGISTERED, EventType.MODEL_PROMOTED, EventType.PROMOTION_BLOCKED,
    EventType.ACTION_LOGGED, EventType.INCIDENT_OPENED, EventType.INCIDENT_RESOLVED,
    EventType.APPROVAL_REQUESTED, EventType.BUDGET_EXCEEDED, EventType.COST_RECOMMENDATION,
]


class RouterState(TypedDict):
    event: Event


class Orchestrator:
    def __init__(self, bus, log, notifier, rules: Rules | None = None,
                 clock: Callable[[], float] = time.time, checkpointer=None):
        self.bus = bus
        self.log = log
        self.notifier = notifier
        self.rules = rules or Rules()
        self.clock = clock
        self.checkpointer = checkpointer or InMemorySaver()
        self.workflow = build_retrain_graph(self, self.checkpointer)
        self.router = self.build_router()
        self.pending: set[str] = set()  # request_id des fils en pause (attente du coût ou d'un humain)
        self.retrain_since: float | None = None  # heure de la dernière demande de retrain
        self.open_incidents: dict[str, float] = {}  # incidents Fiabilité en cours -> heure d'ouverture

    def build_router(self):
        handlers = {
            EventType.DRIFT_DETECTED: self.on_drift,
            EventType.COST_ESTIMATED: self.on_cost_estimated,
            EventType.APPROVAL_GIVEN: self.on_approval,
            EventType.MODEL_REGISTERED: self.on_model_registered,
            EventType.MODEL_PROMOTED: self.on_model_promoted,
            EventType.PROMOTION_BLOCKED: self.on_promotion_blocked,
            EventType.ACTION_LOGGED: self.on_action_logged,
            EventType.INCIDENT_OPENED: self.on_incident_opened,
            EventType.INCIDENT_RESOLVED: self.on_incident_resolved,
            EventType.APPROVAL_REQUESTED: self.on_approval_requested,
            EventType.BUDGET_EXCEEDED: self.on_budget_exceeded,
            EventType.COST_RECOMMENDATION: self.on_cost_recommendation,
        }
        g = StateGraph(RouterState)
        for type_, handler in handlers.items():
            g.add_node(type_.value, lambda state, h=handler: h(state["event"]) or {})
            g.add_edge(type_.value, END)
        g.add_conditional_edges(START, lambda state: state["event"].type.value, [t.value for t in handlers])
        return g.compile()

    def expire_orphans(self) -> None:
        """Au démarrage : les fils en pause d'avant le redémarrage sont perdus (checkpointer en
        mémoire). Leurs questions ne doivent pas rester sur la page avec des boutons sans effet."""
        for request_id in self.notifier.expire_pending():
            if request_id in self.pending:
                continue
            self.record("request_expired", {"request_id": request_id,
                                            "reason": "orchestrateur redémarré : la demande en pause a été perdue"})
            self.notifier.notify("⌛ Question expirée : l'orchestrateur a redémarré. "
                                 "L'agent Qualité signalera de nouveau la dérive au prochain contrôle.", request_id)

    def run(self) -> None:
        print("Orchestrateur (LangGraph) : en écoute…")
        self.bus.listen(self.safe_handle, types=LISTENED)

    def safe_handle(self, event: Event) -> None:
        """Un événement mal formé ne doit pas arrêter l'écoute."""
        try:
            self.handle(event)
        except Exception:
            print(f"Orchestrateur : erreur sur {event.type.value}\n{traceback.format_exc()}")

    def handle(self, event: Event) -> None:
        if event.source == AGENT:
            return  # nos propres événements reviennent par Redis : on les ignore
        if event.type in LISTENED:
            self.router.invoke({"event": event})

    # ------------------------------------------------ fils du workflow de réentraînement
    def thread(self, request_id: str) -> dict:
        return {"configurable": {"thread_id": request_id}}

    def waiting_for(self, request_id: str) -> EventType | None:
        """Événement attendu par le fil, ou None s'il n'est pas en pause."""
        if request_id not in self.pending:
            return None
        paused = self.workflow.get_state(self.thread(request_id)).next
        return WAITING_FOR.get(paused[0]) if paused else None

    def advance(self, request_id: str, value: Any) -> None:
        """Démarre ou reprend un fil ; l'oublie quand il est terminé ou en erreur
        (sinon une demande cassée bloquerait toutes les dérives suivantes)."""
        try:
            self.workflow.invoke(value, self.thread(request_id))
            paused = bool(self.workflow.get_state(self.thread(request_id)).next)
        except Exception:
            paused = False
            self.record("request_failed", {"request_id": request_id})
            raise
        finally:
            if paused:
                self.pending.add(request_id)
            else:
                self.pending.discard(request_id)
                self.checkpointer.delete_thread(request_id)

    # ------------------------------------------------ dérive -> coût -> retrain
    def on_drift(self, event: Event) -> None:
        request_id = uuid.uuid4().hex[:8]
        self.advance(request_id, {"request_id": request_id, "drift": event.payload})

    def on_cost_estimated(self, event: Event) -> None:
        request_id = event.payload.get("request_id")
        if self.waiting_for(request_id) == EventType.COST_ESTIMATED:  # sinon : demandée par quelqu'un d'autre
            self.advance(request_id, Command(resume=event.payload))

    def on_approval(self, event: Event) -> None:
        request_id = event.payload.get("request_id")
        if self.waiting_for(request_id) == EventType.APPROVAL_GIVEN:
            self.advance(request_id, Command(resume=event.payload))

    def request_retrain(self, request_id: str, approved_by: str | None, period: str | None = None) -> None:
        """period : mois où Qualité a vu la dérive ; sans lui, Retrain prend les derniers mois de data/."""
        self.retrain_since = self.clock()
        payload = {"request_id": request_id, "requested_by": AGENT, "approved_by": approved_by}
        if period:
            payload["period"] = period
        self.publish(EventType.RETRAIN_REQUESTED, Severity.INFO, payload)
        self.record("retrain_requested", payload)
        by = f", validé par {approved_by}" if approved_by else ""
        self.notifier.notify(f"🔄 Dérive détectée : réentraînement lancé{by}.", request_id)

    def busy(self) -> bool:
        """Une demande attend déjà (coût ou humain), ou un retrain tourne encore."""
        if self.pending:
            return True
        if self.retrain_since is None:
            return False
        return self.clock() - self.retrain_since < config.RETRAIN_TIMEOUT_MINUTES * 60

    def incident_in_progress(self) -> bool:
        """Un incident Fiabilité est ouvert (et pas oublié depuis INCIDENT_HOLD_MINUTES) :
        on ne réentraîne pas pendant que l'API ou MLflow est en panne."""
        limit = self.clock() - config.INCIDENT_HOLD_MINUTES * 60
        self.open_incidents = {i: t for i, t in self.open_incidents.items() if t > limit}
        return bool(self.open_incidents)

    def mermaid(self) -> dict[str, str]:
        """Les deux graphes au format Mermaid, pour la page web."""
        return {"Routeur d'événements": self.router.get_graph().draw_mermaid(),
                "Demande de réentraînement": self.workflow.get_graph().draw_mermaid()}

    # ------------------------------------------------ résultats de Retrain
    def on_model_registered(self, event: Event) -> None:
        # Traduction pour l'agent Coûts, qui attend training_completed
        p = event.payload
        self.publish(EventType.TRAINING_COMPLETED, Severity.INFO, {
            "cpu_seconds": p.get("training_seconds", 0.0), "months": len(p.get("period") or []) or None,
            "version": p.get("version"), "request_id": p.get("request_id")})

    def on_model_promoted(self, event: Event) -> None:
        self.retrain_since = None
        p = event.payload
        if p.get("production_rmse") is None:
            text = f"✅ Premier modèle v{p['challenger_version']} mis en production (RMSE {p['challenger_rmse']:.2f})."
        else:
            text = (f"✅ Modèle v{p['challenger_version']} promu : RMSE {p['challenger_rmse']:.2f} contre "
                    f"{p['production_rmse']:.2f} pour v{p['production_version']} ({p['improvement']:.1%} d'erreur en moins).")
        self.notifier.notify(text, p.get("request_id"))

    def on_promotion_blocked(self, event: Event) -> None:
        self.retrain_since = None
        p = event.payload
        self.notifier.notify(
            f"⛔ Promotion bloquée : v{p['challenger_version']} a un RMSE de {p['challenger_rmse']:.2f} contre "
            f"{p['production_rmse']:.2f} pour la production v{p['production_version']} "
            f"(gain {p['improvement']:.1%}, minimum {p['min_improvement']:.0%}). La production ne change pas.",
            p.get("request_id"))

    def on_action_logged(self, event: Event) -> None:
        p = event.payload
        if p.get("action") == "retrain_skipped":
            self.retrain_since = None
            self.notifier.notify(f"Réentraînement ignoré par Retrain : {p.get('reason')}.", p.get("request_id"))

    # ------------------------------------------------ incidents (Fiabilité, Retrain)
    def on_incident_opened(self, event: Event) -> None:
        p = event.payload
        if event.source == RETRAIN:
            self.retrain_since = None
            self.notifier.notify(f"🚨 L'entraînement a échoué : {p.get('error')}. La production ne change pas.",
                                 p.get("request_id"))
            return
        if event.source == RELIABILITY:
            self.open_incidents[p.get("incident_id")] = self.clock()
        self.notifier.notify(
            f"🚨 Incident {p.get('incident_id')} : {p.get('cause')} sur {p.get('component')} "
            f"(niveau {p.get('level')}). L'agent Fiabilité s'en occupe.", p.get("incident_id"))

    def on_incident_resolved(self, event: Event) -> None:
        p = event.payload
        self.open_incidents.pop(p.get("incident_id"), None)
        actions = ", ".join(a["action"] for a in p.get("actions", [])) or "aucune"
        self.notifier.notify(
            f"✅ Incident {p.get('incident_id')} résolu en {p.get('duration_seconds', 0):.0f} s "
            f"(actions : {actions}).", p.get("incident_id"))

    def on_approval_requested(self, event: Event) -> None:
        p = event.payload
        self.notifier.notify(
            f"🙋 L'agent {event.source} demande une intervention humaine : {p.get('reason')} "
            f"(incident {p.get('incident_id')}).", p.get("incident_id"))

    # ------------------------------------------------ agent Coûts
    def on_budget_exceeded(self, event: Event) -> None:
        p = event.payload
        self.notifier.notify(
            f"💶 Budget dépassé de {p.get('over_by', 0):.4f} € si on réentraîne. "
            f"Suggestion de l'agent Coûts : {p.get('suggested_action')}.", p.get("request_id"))

    def on_cost_recommendation(self, event: Event) -> None:
        p = event.payload
        detail = ", ".join(f"{k} : {v}" for k, v in p.items() if k != "model")
        self.notifier.notify(f"💡 Recommandation de l'agent Coûts ({detail}). Aucune action automatique.")

    # ------------------------------------------------ outils
    def publish(self, type_: EventType, severity: Severity, payload: dict[str, Any]) -> None:
        self.bus.publish(Event(type=type_, source=AGENT, severity=severity, payload=payload))

    def record(self, action: str, details: dict[str, Any]) -> None:
        self.log.record(AGENT, action, config.MODEL_NAME, details)
