"""Agent Fiabilité piloté par LLM, encadré par des règles déterministes (guards.py).

À chaque cycle :
  1. le code MESURE (sondes, /metrics, signaux Docker/MLflow/Redis/Postgres) ;
  2. le LLM agent répond à UNE question ciblée selon la phase :
       détection (y a-t-il un incident ?), action (que faire ?), vérification (est-ce réparé ?) ;
  3. chaque action passe par le LLM juge, qui applique les règles de sécurité de
     sa grille ; un snapshot est pris avant l'action pour pouvoir l'annuler.

Le code fait la plomberie (capteurs, exécution des actions, traces) et applique les
règles déterministes de guards.py : il écarte une décision du LLM contredite par les
mesures, et décide lui-même du rollback quand une promotion externe a cassé le service.
Si le LLM est indisponible, l'agent ne décide rien : il prévient un humain et
réessaie au cycle suivant."""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass, field

from agents.reliability import guards
from agents.judge.judge import Verdict
from agents.reliability.actions import ReliabilityActions
from agents.reliability.catalog import CATALOG, Action
from agents.reliability.metrics import ErrorBudget, Snapshot, WindowStats, compute_window, fetch_snapshot
from agents.reliability.probes import ProbeResult, run_probes
from agents.reliability.prompts import (ACTION_PROMPT, ACTION_SCHEMA, CAUSES, COMPONENTS, DETECTION_PROMPT,
                                        DETECTION_SCHEMA, LEVELS, VERIFICATION_PROMPT, VERIFICATION_SCHEMA)
from agents.reliability.signals import SignalCollector, Signals
from common import config
from common.events import Event, EventType, Severity
from common.llm import LLMClient, LLMError

AGENT_NAME = "agent_fiabilite"
SEVERITY_OF = {"faible": Severity.INFO, "moyenne": Severity.WARNING, "elevee": Severity.CRITICAL}
RESTART_ACTIONS = {Action.RESTART_API, Action.RECREATE_API}
WAIT_WORDS = {"attendre", "fin", "aucun", "aucune", "rien", "none", "stop", "aucune_action"}
ERROR_LINE = re.compile(r"\b(\w+(?:Error|Exception): .*)$", re.M)  # ex. "ValueError: matmul: ..."

log = logging.getLogger(AGENT_NAME)


def normalize(text) -> str:
    """'Élevée ' -> 'elevee' (les petits modèles varient sur les accents et la casse)."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower().replace(" ", "_")


@dataclass
class IncidentState:
    id: str
    opened_at: float
    cause: str
    component: str
    level: str                                         # faible | moyenne | elevee
    justification: str
    actions: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    degraded: bool = False
    human_notified: bool = False
    snapshot: object = None                            # état avant la dernière action
    last_action_at: float | None = None
    since_last_action: list = field(default_factory=list)   # mesures prises APRÈS la dernière action
    before_last_action: dict | None = None                  # mesures juste avant la dernière action
    rejections: list = field(default_factory=list)           # refus du juge (montrés au LLM)
    api_restart_deadline: float | None = None                # API en cours de redémarrage jusqu'à...

    def note(self, now: float, event: str, **details) -> None:
        self.timeline.append({"t": round(now - self.opened_at, 1), "event": event, **details})


class LLMReliabilityAgent:
    def __init__(self, llm: LLMClient, judge, bus=None, action_log=None, actions=None, collector=None,
                 fetch=fetch_snapshot, probe=run_probes, metrics_store=None, autonomy: dict | None = None):
        if bus is None:
            from common.events import EventBus
            bus = EventBus()
        if action_log is None:
            from common.action_log import ActionLog
            action_log = ActionLog()
        if metrics_store is None:
            from agents.reliability.store import MetricsStore
            metrics_store = MetricsStore()
        if autonomy is None:
            import yaml
            from pathlib import Path
            with open(Path(__file__).resolve().parents[2] / "rules" / "autonomy.yaml", encoding="utf-8") as f:
                autonomy = yaml.safe_load(f) or {}
        self.llm, self.judge = llm, judge
        self.bus, self.action_log, self.metrics_store = bus, action_log, metrics_store
        self.actions = actions or ReliabilityActions()
        self.collector = collector or SignalCollector()
        self.fetch, self.probe = fetch, probe
        self.autonomy = autonomy                       # politique transmise au juge (c'est lui qui l'applique)
        self.budget = ErrorBudget()
        self.prev: Snapshot | None = None
        self.trend: deque[dict] = deque(maxlen=config.LLM_TREND_CYCLES)
        self.incident: IncidentState | None = None
        self.llm_down = False
        self.now = 0.0
        self.window: WindowStats | None = None
        self.probes: ProbeResult | None = None
        self.signals: Signals | None = None
        self.version_seen: int | None = None
        self.version_change: dict | None = None    # dernier changement de version observé (fait mesuré)
        self.agent_changed_version = False         # la prochaine nouvelle version vient d'un rollback de l'agent
        self.last_closed: dict | None = None

    # ------------------------------------------------------------------ boucle
    def run_forever(self) -> None:
        log.info("Agent Fiabilité (LLM %s, juge %s) démarré", self.llm.model, self.judge.llm.model)
        while True:
            started = time.time()
            try:
                self.run_once(started)
            except Exception as exc:  # l'agent ne doit jamais s'arrêter
                log.exception("Erreur inattendue dans le cycle")
                self._record("agent_error", {"error": repr(exc)})
            time.sleep(max(0.0, config.RELIABILITY_POLL_SECONDS - (time.time() - started)))

    def run_once(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now
        self._measure()
        if self.window is None:          # premier cycle : pas encore de fenêtre de comparaison
            return
        self._llm_turn()
        self._store_metrics()

    # ---------------------------------------------------------- 1. mesurer (code)
    def _measure(self) -> None:
        self.probes = self.probe()       # avant /metrics : les sondes comptent dans la fenêtre
        snap = self.fetch()
        if self.prev is None:
            self.prev = snap
            self.window = None
            return
        self.window = compute_window(self.prev, snap)
        if snap.reachable:               # base de comparaison = dernière lecture réussie
            self.prev = snap
        self.budget.add(self.window, self.now)
        self.collector.observe(self.window, self.now)
        self.signals = self.collector.collect(self.window, self.now)
        self._track_version()
        self.trend.append(self._measures())
        inc = self.incident
        if inc and inc.last_action_at is not None and self.now > inc.last_action_at:
            if inc.api_restart_deadline is not None:
                back = self.window.reachable and self.window.model_loaded and self.probes.failure_rate < 1.0
                if not back and self.now < inc.api_restart_deadline:
                    return               # API encore en train de redémarrer : rien à vérifier
                inc.api_restart_deadline = None
            inc.since_last_action.append(self._short(self.trend[-1]))

    def _rebase(self) -> None:
        """Nouvelle base de comparaison juste après une action : sinon la fenêtre suivante
        commence au début du cycle, ~20 s AVANT l'action (temps de réponse du LLM), et mêle
        les erreurs d'avant à celles d'après (vu en direct : rollback réussi jugé "aggravé")."""
        try:
            snap = self.fetch()
        except Exception:
            return
        if snap.reachable:
            self.prev = snap

    @staticmethod
    def _short(m: dict) -> dict:
        return {"api_joignable": m["api_joignable"], "erreurs_pct": m["erreurs_pct"],
                "sondes_echec_pct": m["sondes_echec_pct"], "latence_p95_s": m["latence_p95_s"]}

    def _track_version(self) -> None:
        v = self.window.model_version if self.window.reachable else 0
        if not v:                       # API injoignable ou sans modèle : pas une nouvelle version
            return
        if self.version_seen is not None and v != self.version_seen:
            origin = "rollback_par_l_agent" if self.agent_changed_version else "externe (nouvelle promotion)"
            # trend[-1] = mesures du cycle précédent, avant la nouvelle version
            healthy_before = bool(self.trend) and guards.healthy(self.trend[-1])
            self.version_change = {"de": self.version_seen, "vers": v, "at": self.now, "origine": origin,
                                   "sain_avant": healthy_before}
            self.agent_changed_version = False
        self.version_seen = v

    def _measures(self) -> dict:
        w, p = self.window, self.probes
        return {
            "api_joignable": w.reachable,
            "requetes": int(w.requests),
            "erreurs_pct": round(w.error_rate * 100, 1),
            "latence_p95_s": None if w.p95_latency is None else round(w.p95_latency, 3),
            "sondes_echec_pct": round(p.failure_rate * 100, 1),
            "sondes_p95_s": None if p.p95_latency is None else round(p.p95_latency, 3),
            "modele_charge": w.model_loaded,
            "version_modele": w.model_version,
        }

    def _version_change_view(self) -> dict | None:
        c = self.version_change
        if c is None or self.now - c["at"] > 1800:
            return None
        return {"de": c["de"], "vers": c["vers"], "il_y_a_s": int(self.now - c["at"]), "origine": c["origine"]}

    def _last_closed_view(self) -> dict | None:
        c = self.last_closed
        if c is None or self.now - c["at"] > 600:
            return None
        return {"id": c["id"], "cause": c["cause"], "clos_il_y_a_s": int(self.now - c["at"]), "resume": c["resume"]}

    def _incident_view(self) -> dict | None:
        inc = self.incident
        if inc is None:
            return None
        view = {"id": inc.id, "ouvert_depuis_s": int(self.now - inc.opened_at), "cause": inc.cause,
                "composant": inc.component, "gravite": inc.level, "mode_degrade_actif": inc.degraded,
                "humain_prevenu": inc.human_notified,
                "actions_deja_tentees": [a["action"] for a in inc.actions],
                "actions": [{"action": a["action"], "execution_technique": "ok" if a["success"] else "echec",
                             "erreurs_pct_avant_l_action": a.get("erreurs_pct_avant"),
                             "erreurs_pct_maintenant": self.trend[-1]["erreurs_pct"],
                             "sondes_echec_pct_maintenant": self.trend[-1]["sondes_echec_pct"]}
                            for a in inc.actions[-5:]]}
        if inc.last_action_at is not None:
            view["derniere_action_il_y_a_s"] = int(self.now - inc.last_action_at)
            view["mesures_depuis_la_derniere_action"] = inc.since_last_action[-4:]
        return view

    # ---------------------------------------------------------- 2. décider (LLM, par phases)
    def _llm_turn(self) -> None:
        """Choisit QUELLE question poser au LLM ; c'est toujours le LLM qui répond et décide."""
        if self.incident is None:
            if not self._phase_detection():
                return
        inc = self.incident
        if inc.last_action_at is not None:
            if not inc.since_last_action:          # rien de mesuré depuis l'action : rien à vérifier
                return
            verdict = self._phase_verification()
            if verdict == "aggrave" and inc.snapshot is not None:
                self.tool_restaurer_snapshot(justification="le LLM juge que l'action a aggravé la situation")
            elif verdict not in ("pas_repare", "aggrave"):   # repare (déjà clos), attendre, ou LLM indisponible
                return                     # "aggrave" sans snapshot (humain prévenu) : on agit quand même
        self._phase_action()

    def _ask(self, phase: str, system: str, context: dict, history: list | None = None,
             schema: dict | None = None) -> dict | None:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
                    *(history or [])]
        try:
            answer = self.llm.chat(messages, schema=schema)
        except LLMError as exc:
            self._on_llm_down(exc)
            return None
        self._on_llm_up()
        if self.incident or self._yes(answer.get("incident")):   # pas de trace toutes les 15 s si tout va bien
            self._record("llm_decision", {"incident_id": self.incident.id if self.incident else None,
                                          "phase": phase, "reponse": answer})
        log.info("LLM [%s] : %s", phase, json.dumps(answer, ensure_ascii=False))
        if self.incident:
            self.incident.note(self.now, f"llm_{phase}", **{k: v for k, v in answer.items() if k == "pensee"})
        return answer

    @staticmethod
    def _yes(value) -> bool:
        return value is True or normalize(value) in ("true", "oui", "1", "yes")

    # -- phase 1 : détection
    def _detection_context(self) -> dict:
        trend = list(self.trend)
        n = len(trend)
        return {
            "mesures_actuelles": trend[-1],
            "tendance": [{"il_y_a_s": int((n - 1 - i) * config.RELIABILITY_POLL_SECONDS), **m}
                         for i, m in enumerate(trend[:-1])],
            "budget_erreurs_consomme_pct": round(self.budget.consumed() * 100, 1),
            "dernier_changement_de_version_du_modele": self._version_change_view(),
            "signaux": asdict(self.signals) if self.signals else None,
            "anomalies_mesurees": guards.anomalies(trend[-1], self.signals),   # calculées par le code
            "dernier_incident_clos": self._last_closed_view(),
        }

    def _bad_model_rule(self) -> str | None:
        return guards.bad_model(self.version_change, self.trend[-1], self.signals, self.now)

    def _phase_detection(self) -> bool:
        context = self._detection_context()
        answer = self._ask("detection", DETECTION_PROMPT, context, schema=DETECTION_SCHEMA)
        if answer is None:
            return False
        if not self._yes(answer.get("incident")) and self._yes(answer.get("besoin_des_journaux")):
            context["journaux_api"] = self.tool_lire_journaux_api()
            answer = self._ask("detection", DETECTION_PROMPT, context, schema=DETECTION_SCHEMA)
            if answer is None:
                return False
        rule = self._bad_model_rule()
        if rule:                                   # règle déterministe : le modèle promu est en cause
            if normalize(answer.get("cause", "")) != "mauvais_modele":
                log.info("Règle mauvais_modele : cause '%s' du LLM remplacée (%s)", answer.get("cause"), rule)
            answer = {**answer, "incident": True, "cause": "mauvais_modele", "composant": "modele",
                      "gravite": answer.get("gravite") or "elevee", "justification": rule}
        if not self._yes(answer.get("incident")):
            return False
        found = guards.anomalies(self.trend[-1], self.signals)
        if not rule and not found:                 # aucune mesure ni aucun signal hors objectif
            log.info("Incident proposé par le LLM écarté : aucune mesure hors objectif (%s)",
                     answer.get("justification"))
            self._record("incident_rejected_by_rule", {"reponse": answer, "mesures": self.trend[-1]})
            return False
        result = self.tool_ouvrir_incident(cause=answer.get("cause", "inconnue"),
                                           composant=answer.get("composant", "inconnu"),
                                           gravite=answer.get("gravite", "moyenne"),
                                           justification=answer.get("justification") or answer.get("pensee", ""))
        if "erreur" in result:                     # gravité invalide... : on ouvre en "moyenne"
            result = self.tool_ouvrir_incident(cause=answer.get("cause", "inconnue"),
                                               composant=answer.get("composant", "inconnu"), gravite="moyenne",
                                               justification=answer.get("justification", ""))
        return self.incident is not None

    # -- phase 2 : action
    def _action_context(self) -> dict:
        inc, now_m = self.incident, self.trend[-1]
        return {
            "incident": {"cause": inc.cause, "composant": inc.component, "gravite": inc.level,
                         "justification": inc.justification, "ouvert_depuis_s": int(self.now - inc.opened_at),
                         "mode_degrade_actif": inc.degraded, "humain_prevenu": inc.human_notified},
            "mesures_actuelles": now_m,
            "tendance_recente": list(self.trend)[-4:-1],
            "dernier_changement_de_version_du_modele": self._version_change_view(),
            "signaux": asdict(self.signals) if self.signals else None,
            "actions_deja_tentees": [{"action": a["action"], "execution_technique": "ok" if a["success"] else "echec",
                                      "detail": a.get("detail"), "erreurs_pct_avant": a.get("erreurs_pct_avant"),
                                      "erreurs_pct_maintenant": now_m["erreurs_pct"]} for a in inc.actions],
            "refus_du_juge": inc.rejections[-4:],
        }

    def _phase_action(self) -> None:
        rule = self._bad_model_rule()
        tried = [a["action"] for a in self.incident.actions]
        if rule and Action.ROLLBACK_MODEL.value not in tried:
            # Décision critique prise par une règle, pas par un LLM (ni l'agent, ni le juge)
            if self.autonomy.get(Action.ROLLBACK_MODEL.value) == "auto":
                self.tool_executer_action(action=Action.ROLLBACK_MODEL.value, justification=rule,
                                          regle="mauvais_modele")
            elif not self.incident.human_notified:
                self.tool_prevenir_humain(message=f"rollback nécessaire, validation humaine requise : {rule}")
            return
        if self.signals is not None and self.signals.mlflow_ok is False:
            self._mlflow_down_rule(tried)
            return
        context, history = self._action_context(), []
        for _ in range(config.LLM_MAX_PROPOSALS):
            answer = self._ask("action", ACTION_PROMPT, context, history, schema=ACTION_SCHEMA)
            if answer is None:
                return
            choice = normalize(answer.get("choix") or answer.get("action") or "attendre")
            justification = str(answer.get("justification") or answer.get("pensee") or "")
            if choice in WAIT_WORDS:
                return
            if choice in ("prevenir_humain", "humain", "prevenir"):
                self.tool_prevenir_humain(message=answer.get("message") or justification,
                                          intervention_requise=answer.get("intervention_requise", True))
                return
            if choice == "restaurer_snapshot":
                self.tool_restaurer_snapshot(justification=justification)
                return
            result = self.tool_executer_action(action=choice, justification=justification)
            if "refusee_par_le_juge" not in result and "erreur" not in result:
                return
            # refus du juge ou choix invalide : on le dit au LLM et il propose autre chose
            history += [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                        {"role": "user", "content": json.dumps({"resultat": result, "consigne":
                         "Propose un autre choix (ou prevenir_humain / attendre)."}, ensure_ascii=False)}]

    # -- phase 3 : vérification
    def _verification_context(self) -> dict:
        inc = self.incident
        last = inc.actions[-1] if inc.actions else None
        return {
            "incident": {"cause": inc.cause, "gravite": inc.level, "mode_degrade_actif": inc.degraded,
                         "humain_prevenu": inc.human_notified},
            "derniere_action": None if last is None else {
                "action": last["action"], "execution_technique": "ok" if last["success"] else "echec",
                "detail": last.get("detail")},
            "avant_l_action": inc.before_last_action,
            "mesures_depuis_la_derniere_action": inc.since_last_action[-6:],
            "derniere_action_il_y_a_s": int(self.now - inc.last_action_at),
        }

    def _phase_verification(self) -> str | None:
        answer = self._ask("verification", VERIFICATION_PROMPT, self._verification_context(),
                           schema=VERIFICATION_SCHEMA)
        if answer is None:
            return None
        llm_verdict = normalize(answer.get("verdict") or "attendre")
        verdict = self._check_verdict(llm_verdict)
        level = normalize(answer.get("gravite") or "")
        if level in LEVELS and level != self.incident.level:
            self.tool_changer_gravite(gravite=level, justification=answer.get("pensee", ""))
        if verdict in ("repare", "reparee", "resolu"):
            resume = answer.get("resume") or answer.get("pensee", "")
            if llm_verdict != verdict:             # le LLM n'a pas vu la réparation : résumé factuel
                last = self.incident.actions[-1]["action"] if self.incident.actions else "intervention humaine"
                resume = f"service revenu sous l'objectif depuis 2 mesures après {last}"
            self.tool_clore_incident(resume=resume)
            return "repare"
        if verdict in ("pas_repare", "non_repare", "aggrave"):
            return verdict if verdict != "non_repare" else "pas_repare"
        return "attendre"

    def _mlflow_down_rule(self, tried: list[str]) -> None:
        """MLflow injoignable : aucune action sur l'API ou le modèle n'est possible (elle perdrait
        son modèle). Protéger les utilisateurs si besoin (mode dégradé), puis prévenir un humain."""
        inc = self.incident
        users_hit = guards.breaches(self.trend[-1]) != []
        if users_hit and not inc.degraded and Action.DEGRADED_MODE.value not in tried                 and self.autonomy.get(Action.DEGRADED_MODE.value) == "auto":
            self.tool_executer_action(action=Action.DEGRADED_MODE.value, regle="mlflow_indisponible",
                                      justification="MLflow injoignable : modèle de secours pour les utilisateurs")
        if not inc.human_notified:
            self.tool_prevenir_humain(message="MLflow injoignable : l'API ne peut plus charger son modèle. "
                                              "Relancer MLflow (restart_mlflow demande une validation humaine).")

    def _check_verdict(self, verdict: str) -> str:
        """Écarte un verdict contredit par les mesures (règles de guards.py)."""
        inc = self.incident
        since = inc.since_last_action
        checked = verdict
        repaired = len(since) >= 2 and all(guards.healthy(m) for m in since[-2:])
        if repaired:                               # 2 mesures saines (sondes = vrai modèle) : réparé
            checked = "repare"
        elif verdict in ("repare", "reparee", "resolu"):     # moins de 2 mesures saines : pas encore
            checked = "attendre" if since and guards.healthy(since[-1]) else "pas_repare"
        elif verdict == "aggrave" and not guards.worse(inc.before_last_action, since[-1] if since else None):
            checked = "pas_repare"
        if checked != verdict:
            log.info("Verdict '%s' du LLM corrigé en '%s' d'après les mesures", verdict, checked)
            self._record("verdict_corrected", {"incident_id": inc.id, "llm": verdict, "retenu": checked,
                                               "avant": inc.before_last_action, "depuis": since[-2:]})
        return checked

    def _on_llm_down(self, exc: Exception) -> None:
        log.error("LLM indisponible : %s", exc)
        if self.llm_down:
            return
        self.llm_down = True
        reason = "LLM indisponible : l'agent Fiabilité ne prend plus de décision (surveillance suspendue)"
        self._record("llm_unavailable", {"error": str(exc)})
        self._publish(EventType.APPROVAL_REQUESTED, Severity.CRITICAL,
                      {"incident_id": self.incident.id if self.incident else None, "reason": reason,
                       "intervention_requise": True})

    def _on_llm_up(self) -> None:
        if self.llm_down:
            self.llm_down = False
            self._record("llm_recovered", {})
            log.info("LLM de nouveau disponible")

    # ---------------------------------------------------------- 3. outils (plomberie)
    def tool_lire_signaux(self, **_) -> dict:
        self.signals = self.collector.collect(self.window, self.now)
        return {"signaux": asdict(self.signals)}

    def tool_lire_journaux_api(self, **_) -> dict:
        try:
            logs = self.actions.api_logs()
            errors = ERROR_LINE.findall(logs)
            return {"derniere_erreur": errors[-1].strip() if errors else None, "journaux": logs[-1500:]}
        except Exception as exc:
            return {"erreur": f"journaux indisponibles ({exc})"}

    def tool_historique_incidents(self, **_) -> dict:
        try:
            rows = self.action_log.latest(50)
        except Exception as exc:
            return {"erreur": f"historique indisponible ({exc})"}
        incidents = [{"id": d.get("incident_id"), "cause": d.get("cause"), "duree_s": d.get("duration_seconds"),
                      "actions": [a.get("action") for a in d.get("actions", [])], "resume": d.get("resume")}
                     for r in rows if r.get("agent") == AGENT_NAME and r.get("action") == "incident_resolved"
                     for d in [r.get("details") or {}]]
        return {"incidents_recents": incidents[:5]}

    def tool_ouvrir_incident(self, cause="inconnue", composant="inconnu", gravite="moyenne",
                             justification="", **_) -> dict:
        if self.incident:
            return {"erreur": f"l'incident {self.incident.id} est déjà ouvert"}
        level = normalize(gravite)
        if level not in LEVELS:
            return {"erreur": f"gravité invalide '{gravite}', choisir parmi {LEVELS}"}
        cause, component = normalize(cause), normalize(composant)
        checked = guards.checked_cause(cause, self.trend[-1], self.signals)
        if checked != cause:
            log.info("Cause '%s' du LLM contredite par les signaux : '%s' retenue", cause, checked)
            cause = checked
        inc = IncidentState(id=f"inc-{int(self.now)}", opened_at=self.now,
                            cause=cause if cause in CAUSES else "inconnue",
                            component=component if component in COMPONENTS else "inconnu",
                            level=level, justification=str(justification))
        inc.note(self.now, "opened", cause=inc.cause, gravite=level, justification=inc.justification)
        self.incident = inc
        payload = {"incident_id": inc.id, "cause": inc.cause, "component": inc.component,
                   "level": SEVERITY_OF[level].value, "gravite": level, "evidence": [inc.justification],
                   **self.trend[-1]}
        log.warning("Incident %s ouvert par le LLM : %s (%s) - %s", inc.id, inc.cause, level, inc.justification)
        self._publish(EventType.INCIDENT_OPENED, SEVERITY_OF[level], payload)
        self._record("incident_opened", payload)
        return {"ok": True, "incident_id": inc.id}

    def tool_changer_gravite(self, gravite="", justification="", **_) -> dict:
        if not self.incident:
            return {"erreur": "aucun incident ouvert"}
        level = normalize(gravite)
        if level not in LEVELS:
            return {"erreur": f"gravité invalide '{gravite}', choisir parmi {LEVELS}"}
        old, self.incident.level = self.incident.level, level
        self.incident.note(self.now, "gravite_changee", avant=old, apres=level, justification=justification)
        self._record("severity_changed", {"incident_id": self.incident.id, "avant": old, "apres": level,
                                          "justification": justification})
        return {"ok": True, "gravite": level}

    def tool_executer_action(self, action="", justification="", regle: str | None = None, **_) -> dict:
        """regle : nom de la règle déterministe qui impose l'action (pas de juge LLM dans ce cas)."""
        inc = self.incident
        if not inc:
            return {"erreur": "ouvre d'abord un incident"}
        try:
            act = Action(normalize(action))
        except ValueError:
            return {"erreur": f"action inconnue '{action}'", "actions_disponibles": [a.value for a in CATALOG]}

        refusal = None if regle else guards.forbidden(act.value, self.signals)
        if regle:
            verdict = Verdict(True, 10, f"règle déterministe {regle} : {justification}", "regle")
        elif refusal:                              # règle absolue vérifiable par le code : pas besoin du juge
            verdict = Verdict(False, 0, refusal, "regle")
        else:
            verdict = self.judge.review("reliability", {"action": act.value, "justification": justification}, {
                "incident": self._incident_view(), "mesures": self.trend[-1],
                "dernier_changement_de_version_du_modele": self._version_change_view(),
                "signaux": asdict(self.signals) if self.signals else None,
                "politique_autonomie": self.autonomy,
                "action_risque": CATALOG[act].risk, "action_description": CATALOG[act].description,
            })
        judge_info = {"note": verdict.score, "critique": verdict.critique, "juge": verdict.by}
        if not verdict.approved:
            inc.note(self.now, "action_refusee", action=act.value, **judge_info)
            inc.rejections.append({"action": act.value, "critique_du_juge": verdict.critique})
            self._record("judge_rejected", {"incident_id": inc.id, "action": act.value,
                                            "justification": justification, **judge_info})
            log.info("%s : REJET de %s (%d/10) %s", "Règle" if verdict.by == "regle" else "Juge",
                     act.value, verdict.score, verdict.critique)
            return {"refusee_par_le_juge": True, **judge_info}

        snap = self.actions.take_snapshot()
        result = self.actions.execute(act)
        entry = {"action": act.value, "success": result.success, "detail": result.detail,
                 "t": round(self.now - inc.opened_at, 1), "by": f"regle:{regle}" if regle else "llm",
                 "justification": justification,
                 "judge_score": verdict.score, "erreurs_pct_avant": self.trend[-1]["erreurs_pct"]}
        inc.actions.append(entry)
        inc.snapshot, inc.last_action_at, inc.since_last_action = snap, self.now, []
        inc.before_last_action = self._short(self.trend[-1])
        if result.success and act == Action.ROLLBACK_MODEL:
            self.agent_changed_version = True
        if result.success and act in RESTART_ACTIONS:
            inc.api_restart_deadline = self.now + config.RELIABILITY_RESTART_GRACE_SECONDS
        self._rebase()
        inc.note(self.now, "action", **entry)
        self._record(f"action:{act.value}", {"incident_id": inc.id, **entry})
        log.info("Action %s (%s) : %s", act.value,
                 f"règle {regle}" if regle else f"juge {verdict.score}/10", result.detail)

        out = {"succes": result.success, "detail": result.detail, "juge": judge_info,
               "snapshot_disponible": bool(getattr(snap, "complete", False))}
        if result.success and act == Action.DEGRADED_MODE:
            inc.degraded = True
        if result.success and act in RESTART_ACTIONS and inc.degraded:
            inc.degraded = False
            out["attention"] = "le redémarrage a désactivé le mode dégradé de l'API"
        return out

    def tool_restaurer_snapshot(self, justification="", **_) -> dict:
        inc = self.incident
        if not inc or inc.snapshot is None:
            return {"erreur": "aucun snapshot à restaurer"}
        result = self.actions.restore(inc.snapshot)
        self._rebase()
        inc.snapshot = None
        self.agent_changed_version = True
        inc.note(self.now, "snapshot_restaure", success=result.success, detail=result.detail)
        self._record("snapshot_restored", {"incident_id": inc.id, "success": result.success,
                                           "detail": result.detail, "justification": justification})
        return {"succes": result.success, "detail": result.detail}

    def tool_prevenir_humain(self, message="", intervention_requise=True, **_) -> dict:
        inc = self.incident
        needed = intervention_requise is True or normalize(intervention_requise) in ("true", "oui", "1")
        details = {"incident_id": inc.id if inc else None, "reason": str(message),
                   "intervention_requise": needed, "level": SEVERITY_OF[inc.level].value if inc else None}
        if inc:
            inc.human_notified = True
            if inc.last_action_at is None:         # on vérifiera ensuite si le service revient
                inc.last_action_at, inc.since_last_action = self.now, []
                inc.before_last_action = self._short(self.trend[-1])
            inc.note(self.now, "humain_prevenu", message=message, intervention_requise=needed)
        log.warning("Humain prévenu : %s", message)
        self._publish(EventType.APPROVAL_REQUESTED,
                      SEVERITY_OF[inc.level] if inc else Severity.WARNING, details)
        self._record("human_notified", details)
        return {"ok": True}

    def tool_clore_incident(self, resume="", **_) -> dict:
        inc = self.incident
        if not inc:
            return {"erreur": "aucun incident ouvert"}
        if inc.degraded:
            r = self.actions.exit_degraded_mode()
            inc.note(self.now, "mode_degrade_desactive", success=r.success)
        inc.note(self.now, "resolved", resume=resume)
        report = {
            "incident_id": inc.id, "cause": inc.cause, "component": inc.component,
            "level": SEVERITY_OF[inc.level].value, "gravite": inc.level, "justification": inc.justification,
            "resume": str(resume), "actions": inc.actions, "degraded": inc.degraded,
            "human_notified": inc.human_notified, "duration_seconds": round(self.now - inc.opened_at, 1),
            "timeline": inc.timeline, "decided_by": self.llm.model,
        }
        verdict = self.judge.evaluate("reliability", report)
        report["judge"] = {"score": verdict.score, "critique": verdict.critique, "by": verdict.by,
                           "ok": verdict.approved}
        log.info("Incident %s clos par le LLM en %.0f s (note du juge %d/10)",
                 inc.id, report["duration_seconds"], verdict.score)
        self._publish(EventType.INCIDENT_RESOLVED, Severity.INFO, report)
        self._record("incident_resolved", report)
        self.last_closed = {"id": inc.id, "cause": inc.cause, "at": self.now, "resume": str(resume)}
        self.incident = None
        return {"ok": True, "duree_s": report["duration_seconds"], "note_du_juge": verdict.score}

    # ------------------------------------------------------------------ traces
    def _store_metrics(self) -> None:
        w, inc = self.window, self.incident
        try:
            self.metrics_store.record(
                self.now, reachable=w.reachable, requests=w.requests, error_rate=w.error_rate,
                p95_latency=w.p95_latency, probe_failure_rate=self.probes.failure_rate,
                budget_consumed=self.budget.consumed(), degraded=bool(inc and inc.degraded),
                incident_level=SEVERITY_OF[inc.level].value if inc else None)
        except Exception as exc:
            log.error("Écriture des mesures impossible : %s", exc)

    def _publish(self, type_: EventType, severity: Severity, payload: dict) -> None:
        try:
            self.bus.publish(Event(type=type_, source=AGENT_NAME, severity=severity, payload=payload))
        except Exception as exc:
            log.error("Publication %s impossible : %s", type_.value, exc)

    def _record(self, action: str, details: dict) -> None:
        try:
            self.action_log.record(AGENT_NAME, action, model=config.MODEL_NAME,
                                   details=json.loads(json.dumps(details, default=str)))
        except Exception as exc:
            log.error("Écriture du journal impossible : %s", exc)
