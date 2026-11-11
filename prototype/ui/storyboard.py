"""Les 4 scénarios du storyboard (Design Thinking), joués pour de vrai dans la pipeline.

Chaque scénario est une bande dessinée de 4 à 6 cases. « Lancer » prépare la situation
(budget du mois, données reçues, version du modèle, trafic, phrase en chat) avec les
mêmes outils que la page Scénarios de test, puis ce sont les agents qui réagissent seuls.
La page du scénario suit leur réaction case par case, à partir du journal d'actions :
chaque case passe à « fait » avec ce qui s'est vraiment passé (versions, RMSE, coûts...).
Si les agents prennent un autre chemin que l'histoire (ex. le challenger n'est pas meilleur),
la case l'affiche en « variante » : on ne maquille jamais le résultat.

  1. Dérive des données       : dérive -> retrain automatique -> promotion -> un seul message
  2. Panne du service         : erreurs au pic de trafic -> redémarrage sans effet -> rollback -> rapport
  3. Commande par chat        : phrase -> action comprise -> budget dépassé -> validation -> retrain tracé
  4. Le mauvais challenger    : données suspectes -> challenger moins bon -> promotion bloquée -> humain prévenu
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import or_, select

from common import budget as live_budget
from common import config
from common.action_log import action_log
from ui import traces as tr
from ui.scenarios import AGENT as DEMO_AGENT

LARGE_BUDGET = 20.0          # budget « normal » des scénarios 1 et 4 : le retrain est automatique
DONE, CURRENT, PENDING, VARIANT, FAILED = "done", "current", "pending", "variant", "failed"


@dataclass(frozen=True)
class Case:
    clock: str        # « LUNDI · 09:00 »
    actor: str        # qui agit dans la case
    title: str
    speech: str       # texte du storyboard (ce qu'on raconte)
    check: Callable[["Ctx"], tuple[str, str] | None]  # (état, ce qui s'est vraiment passé) ou None : pas encore


# Personnage de chaque case : icône et couleur (celle de l'agent, ou celle des humains)
ACTORS = {
    "API": ("🚕", "a-other"), "Agent Qualité": ("🔍", "a-quality"), "Orchestrateur": ("🧭", "a-orchestrator"),
    "Agent Retrain": ("🏋", "a-retrain"), "Agent Fiabilité": ("🛟", "a-reliability"), "Agent Coûts": ("💶", "a-cost"),
    "Data scientist": ("🧑‍💻", "a-human"), "Ingénieur ML": ("🧑‍🔧", "a-human"), "Responsable": ("🧑‍💼", "a-human"),
}


def actor_icon(actor: str) -> tuple[str, str]:
    return ACTORS.get(actor, ("•", "a-other"))


@dataclass(frozen=True)
class Story:
    key: str
    number: int
    title: str
    summary: str
    objectives: tuple[str, ...]
    cases: tuple[Case, ...]
    params: tuple[str, ...] = ()   # month, text, who


@dataclass
class StoryRun:
    id: str
    story: Story
    params: dict[str, Any]
    started: datetime
    status: str = "running"        # running, done, error
    notes: list[tuple[datetime, str]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None    # demande (request_id) ou incident (incident_id)
    check_id: str | None = None    # contrôle Qualité lancé
    error: str | None = None

    def note(self, text: str) -> None:
        self.notes.append((datetime.now(timezone.utc), text))


@dataclass
class Ctx:
    """Ce que les cases regardent : le parcours, les lignes du contrôle Qualité, les messages."""
    run: StoryRun
    steps: list[tr.Step]
    board: "StoryBoard"

    def first(self, action: str, agent: str | None = None, **where) -> tr.Step | None:
        for s in self.steps:
            if s.action == action and (agent is None or s.agent.log_name == agent) \
                    and all(s.details.get(k) == v for k, v in where.items()):
                return s
        return None

    def messages(self) -> list[tr.Step]:
        return [s for s in self.steps if s.is_message]


# ---------------------------------------------------------------- outils de texte
def pct(x: Any) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "?"


def num(x: Any, fmt: str = ".2f") -> str:
    return format(x, fmt) if isinstance(x, (int, float)) else "?"


def eur(x: Any) -> str:
    if not isinstance(x, (int, float)):
        return "?"
    return f"{x:.2f} €" if abs(x) >= 1 else f"{x:.4f} €"


def hhmmss(s: tr.Step, c: "Ctx | None" = None) -> str:
    """Temps écoulé depuis le lancement du scénario (l'heure du conteneur est en UTC, pas celle du lecteur)."""
    start = c.run.started if c is not None else s.ts
    secs = max(0, int((s.ts - start).total_seconds()))
    return f"+{secs} s" if secs < 60 else f"+{secs // 60} min {secs % 60:02d} s"


# ---------------------------------------------------------------- cases communes
def drift_seen(c: Ctx):
    s = c.first("drift_detected", "agent_qualite")
    if s:
        d = s.details
        if d.get("reason") == "corrupted_input_data":
            return VARIANT, f"données jugées corrompues ({pct(d.get('invalid_share'))} de lignes invalides) : pas de retrain"
        return DONE, (f"score de dérive {num(d.get('drift_score'))} (seuil {d.get('drift_threshold')}), "
                      f"dérive des prédictions {num(d.get('prediction_drift'))} sur {d.get('period')}")
    if c.first("quality_ok", "agent_qualite"):
        return VARIANT, "contrôle normal : aucune dérive au-dessus du seuil"
    return None


def challenger_trained(c: Ctx):
    s = c.first("challenger_registered", "agent_retrain")
    if s:
        d = s.details
        tuned = " après optimisation Optuna" if d.get("params") else ""
        started = c.first("retrain_started", "agent_retrain")
        received = (started.details.get("data_file") if started else None) or d.get("data_file")
        on = f" sur le fichier reçu {received.split('/')[-1]}" if received else \
            f" sur {', '.join(d.get('period') or [])}"
        return DONE, f"v{d.get('version')} entraînée{on}{tuned}, enregistrée dans MLflow comme challenger (RMSE {num(d.get('rmse'), '.3f')})"
    for action, label in (("retrain_skipped", "ignoré par Retrain"), ("retrain_failed", "entraînement en échec"),
                          ("retrain_refused", "refusé par un humain"), ("drift_ignored", "mis de côté")):
        s = c.first(action)
        if s:
            return VARIANT, f"{label} : {s.details.get('reason') or s.details.get('error') or s.summary}"
    if c.first("retrain_started", "agent_retrain"):
        return CURRENT, "entraînement en cours (Optuna puis XGBoost, quelques minutes)…"
    return None


def comparison(d: dict) -> str:
    return (f"v{d.get('challenger_version')} : RMSE {num(d.get('challenger_rmse'), '.3f')} · "
            f"v{d.get('production_version')} en production : RMSE {num(d.get('production_rmse'), '.3f')} "
            f"({gain(d.get('improvement'))}, il en faut {pct(d.get('min_improvement'))} de moins)")


def gain(x: Any) -> str:
    if not isinstance(x, (int, float)):
        return "?"
    return f"{x:.1%} d'erreur en moins" if x >= 0 else f"{-x:.1%} d'erreur en plus"


# ---------------------------------------------------------------- scénario 1
def s1_production(c: Ctx):
    v = c.run.facts.get("production")
    return DONE, f"l'API sert la v{v} du modèle taxi" if v else "l'API sert le modèle de production"


def s1_decision(c: Ctx):
    s = c.first("retrain_decision")
    if not s:
        return None
    d = s.details
    drift = c.first("drift_detected", "agent_qualite")
    why = (drift.details.get("explanation") or "").strip() if drift else ""
    why = f" Explication : « {why[:260]} »" if why else ""
    if d.get("rule") == "auto":
        return DONE, (f"règle d'autonomie : réentraîner = automatique (coût {eur(d.get('estimated_cost'))}, "
                      f"dans le budget de {eur(d.get('budget'))}).{why}")
    return VARIANT, f"règle : {d.get('rule')} (budget dépassé) : une validation humaine est demandée.{why}"


def s1_promoted(c: Ctx):
    s = c.first("model_promoted", "agent_retrain")
    if s:
        d = s.details
        if d.get("production_version") is not None:  # bouton « Revenir à vN » de la dernière case
            c.run.facts["rollback"] = {"target": d["production_version"], "current": d["challenger_version"]}
        return DONE, comparison(d) + " → promu, l'API le charge sans coupure"
    s = c.first("promotion_blocked", "agent_retrain")
    if s:
        return VARIANT, comparison(s.details) + " → pas assez meilleur : promotion bloquée, la production ne change pas"
    return None


def s1_message(c: Ctx):
    msgs = [m for m in c.messages() if "promu" in m.summary or "Promotion bloquée" in m.summary]
    if not msgs:
        return None
    m = msgs[-1]
    state = DONE if "promu" in m.summary else VARIANT
    return state, f"message envoyé {hhmmss(m, c)} (interface et Discord) : {m.summary}"


# ---------------------------------------------------------------- scénario 2
def s2_traffic(c: Ctx):
    if not c.run.facts.get("traffic_at"):
        v = c.run.facts.get("flaky_version")
        return (CURRENT, f"v{v} promue : saine au calme (les sondes passent), attente avant le pic de trafic…") \
            if v else None
    t = c.board.runner.traffic
    live = f" : {t.ok} réponses, {t.errors} erreurs" if t.running else ""
    return DONE, (f"pic de trafic simulé ({c.run.facts.get('rps')} req/s, trajets jusqu'à 15 km) sur la "
                  f"v{c.run.facts.get('flaky_version')}{live}")


def s2_incident(c: Ctx):
    s = c.first("incident_opened", "agent_fiabilite")
    if not s:
        return None
    d = s.details
    why = (d.get("evidence") or [""])[0]
    return DONE, (f"incident {d.get('incident_id')} ouvert : cause « {d.get('cause')} » sur {d.get('component')}, "
                  f"gravité {d.get('gravite') or d.get('level')} — {num(d.get('erreurs_pct'), '.1f')} % d'erreurs, "
                  f"latence p95 {num(d.get('latence_p95_s'), '.2f')} s. {why}".strip())


def s2_restart(c: Ctx):
    restart = next((s for s in c.steps if s.action in ("action:restart_api", "action:recreate_api")), None)
    rollback = c.first("action:rollback_model")
    if restart is None:
        if rollback is not None:
            return VARIANT, "pas de redémarrage : l'agent est allé directement au rollback"
        return None
    after = "les erreurs sont toujours là" if rollback else "vérification en cours…"
    return (DONE if rollback else CURRENT), (f"conteneur de l'API redémarré {hhmmss(restart, c)} (action autorisée "
                                             f"sans validation) ; {after}")


def s2_rollback(c: Ctx):
    s = c.first("action:rollback_model")
    if not s:
        return None
    d = s.details
    by = "règle déterministe" if str(d.get("by", "")).startswith("regle") else "décision du LLM validée par le juge"
    state = DONE if d.get("success") else FAILED
    return state, f"{d.get('detail')} ({by}) : {d.get('justification') or ''}".strip()


def s2_report(c: Ctx):
    s = c.first("incident_resolved", "agent_fiabilite")
    if not s:
        return None
    d = s.details
    actions = ", ".join(a.get("action", "?") for a in d.get("actions") or []) or "aucune"
    return DONE, (f"incident résolu en {num(d.get('duration_seconds'), '.0f')} s (actions : {actions}). "
                  f"{d.get('resume') or ''}".strip())


# ---------------------------------------------------------------- scénario 3
def s3_chat(c: Ctx):
    s = c.first("chat_received")
    return (DONE, f"« {s.details.get('text')} » — {s.details.get('by')}") if s else None


def s3_understood(c: Ctx):
    s = c.first("chat_understood")
    if not s:
        return None
    d = s.details
    if d.get("problem"):
        return FAILED, f"commande non lancée : {d['problem']}"
    by = "le LLM" if d.get("understood_by") == "llm" else "les règles (LLM indisponible)"
    return DONE, (f"action : réentraîner · modèle : {d.get('model')} · période : {d.get('period_label')} · "
                  f"demandé par : {d.get('by')} — compris par {by}")


def s3_budget(c: Ctx):
    s = c.first("cost_estimated", "cost")
    if not s:
        return None
    d = s.details
    est = (d.get("options") or {}).get("standard", {}).get("cost")
    text = (f"estimation {eur(est)} pour {d.get('months')} mois ; déjà dépensé ce mois-ci {eur(d.get('month_total'))} ; "
            f"budget {eur(d.get('budget'))}")
    if d.get("exceeds_budget"):
        over = (d.get("month_total") or 0) + (est or 0) - (d.get("budget") or 0)
        return DONE, f"{text} → dépassement de {eur(over)} : les règles demandent l'accord d'un responsable"
    return VARIANT, f"{text} → dans le budget : pas besoin de validation"


def s3_question(c: Ctx):
    q = next((s for s in c.steps if s.is_message and s.action == "question"), None)
    if q is None:
        return None
    status = q.details.get("status")
    if status == "pending":
        return CURRENT, "question envoyée au responsable (interface et Discord) : en attente de sa décision"
    if status == "approved":
        return DONE, f"approuvé par {q.details.get('answered_by')}"
    if status == "refused":
        return VARIANT, f"refusé par {q.details.get('answered_by')} : rien n'est lancé"
    return VARIANT, f"question {status}"


def s3_launched(c: Ctx):
    s = c.first("retrain_requested")
    if not s:
        return None
    marks = [("chat_received", "demande reçue (chat)"), ("budget_exceeded", "budget dépassé"),
             ("approval_answered", "approuvé"), ("retrain_requested", "retrain lancé"),
             ("retrain_started", "entraînement démarré"), ("model_promoted", "promu"),
             ("promotion_blocked", "promotion bloquée")]
    seen = [f"{hhmmss(st, c)} {label}" for action, label in marks if (st := c.first(action))]
    started = c.first("retrain_started", "agent_retrain")
    period = f" sur {', '.join(started.details.get('period') or [])}" if started else ""
    return DONE, f"journal d'actions{period} : " + " · ".join(seen)


# ---------------------------------------------------------------- scénario 4
def s4_suspect(c: Ctx):
    s = c.first("drift_detected", "agent_qualite")
    if not s:
        return drift_seen(c)
    d = s.details
    if d.get("data_file"):  # bouton « Voir les données » de la dernière case
        c.run.facts["data_file"] = d["data_file"]
    if d.get("reason") == "corrupted_input_data":
        return VARIANT, f"jugées corrompues ({pct(d.get('invalid_share'))} de lignes invalides) : aucun retrain"
    return DONE, (f"fichier reçu : {pct(d.get('impossible_share'))} de valeurs impossibles (durées négatives, "
                  f"distances de 9999 km) ; forte dérive : score {num(d.get('drift_score'))}, prédictions "
                  f"{num(d.get('prediction_drift'))}" + (" — données marquées suspectes" if d.get("suspect") else ""))


def s4_blocked(c: Ctx):
    s = c.first("promotion_blocked", "agent_retrain")
    if s:
        return DONE, comparison(s.details) + " → un modèle moins bon n'est jamais promu"
    s = c.first("model_promoted", "agent_retrain")
    if s:
        return VARIANT, comparison(s.details) + " → promu malgré tout"
    return None


def s4_alert(c: Ctx):
    s = c.first("suspect_data_alert")
    if not s:
        return None
    ignored = c.first("alert_ignored")
    who = f" — ignorée par {ignored.details.get('by')}" if ignored else ""
    return DONE, (f"alerte envoyée {hhmmss(s, c)} (#mlops-alertes) : challenger v{s.details.get('challenger_version')} "
                  f"refusé, données de {s.details.get('period')} suspectes, vérification humaine requise{who}")


STORIES: tuple[Story, ...] = (
    Story("derive", 1, "Dérive des données",
          "Les habitudes de trajet changent. Les agents détectent la dérive, réentraînent, vérifient et "
          "préviennent l'équipe, sans intervention humaine.",
          ("Obj. 1 · Qualité", "Obj. 3 · Moins de manuel", "Obj. 5 · Réaction"), (
              Case("LUNDI · 09:00", "API", "L'API prédit, mais le monde a changé",
                   "Notre API estime la durée des trajets. Depuis quelques jours, les trajets ont changé : ils sont "
                   "plus longs. Le modèle en production ne le sait pas encore.", s1_production),
              Case("09:05 · AGENT QUALITÉ", "Agent Qualité", "La dérive dépasse le seuil",
                   "L'agent Qualité compare les nouvelles données à celles de l'entraînement, avec Evidently. Le "
                   "score de dérive passe au-dessus du seuil : il publie un événement.", drift_seen),
              Case("09:06 · ORCHESTRATEUR", "Orchestrateur", "Comprendre avant d'agir",
                   "L'orchestrateur reçoit l'événement ; le LLM explique la dérive en langage simple. Il vérifie "
                   "les règles d'autonomie : réentraîner est autorisé sans validation humaine.", s1_decision),
              Case("09:07 · AGENT RETRAIN", "Agent Retrain", "Un challenger est entraîné",
                   "L'agent Retrain lance un entraînement XGBoost sur les données récentes. Le challenger est "
                   "enregistré dans MLflow ; rien ne change encore en production.", challenger_trained),
              Case("09:15 · COMPARAISON", "Agent Retrain", "Le challenger fait mieux, il est promu",
                   "On compare les deux modèles sur les mêmes données récentes. Le challenger fait moins "
                   "d'erreurs : il est promu et l'API le charge sans coupure.", s1_promoted),
              Case("09:20 · ÉQUIPE", "Data scientist", "Un seul message, déjà résolu",
                   "La data scientist reçoit un seul message : ce qui s'est passé, pourquoi, ce que les agents "
                   "ont fait. Elle peut lire le rapport, ou revenir à l'ancienne version en un clic.", s1_message),
          ), params=("month",)),
    Story("panne", 2, "Panne du service",
          "Un vendredi soir, l'API se met à renvoyer des erreurs. L'agent Fiabilité tente un redémarrage, puis "
          "revient à la dernière version saine.",
          ("Obj. 2 · Fiabilité", "Obj. 5 · Réaction"), (
              Case("VENDREDI · 19:00", "API", "Pic de trafic, les erreurs montent",
                   "Heure de pointe : l'API commence à renvoyer des erreurs 500. L'équipe est déjà partie en "
                   "week-end.", s2_traffic),
              Case("19:02 · AGENT FIABILITÉ", "Agent Fiabilité", "Le seuil d'erreurs est franchi",
                   "L'agent Fiabilité surveille la latence, le taux d'erreurs et les ressources. Les indicateurs "
                   "passent au rouge : il ouvre un incident et décide d'agir.", s2_incident),
              Case("19:03 · SELF-HEALING", "Agent Fiabilité", "Premier essai : redémarrer",
                   "Première réponse, la plus simple : redémarrer le conteneur de l'API, une action autorisée sans "
                   "validation. Mais au bout d'une minute, les erreurs sont toujours là.", s2_restart),
              Case("19:06 · ROLLBACK", "Agent Fiabilité", "Retour à la dernière version saine",
                   "Le problème vient donc de la nouvelle version du modèle. L'agent consulte le registre MLflow et "
                   "revient à la dernière version qui fonctionnait. Les erreurs retombent.", s2_rollback),
              Case("SAMEDI · 10:00", "Ingénieur ML", "Le rapport d'incident attend l'équipe",
                   "Le lendemain, l'ingénieur ML trouve un rapport d'incident déjà rédigé : la chronologie, les "
                   "actions faites et la version à corriger.", s2_report),
          )),
    Story("chat", 3, "Commande par chat et budget",
          "Une data scientist demande un réentraînement en une phrase. L'agent Coûts voit que le budget serait "
          "dépassé et demande l'accord d'un responsable.",
          ("Obj. 3 · Moins de manuel", "Obj. 4 · Coûts"), (
              Case("MARDI · 14:00", "Data scientist", "Une demande en langage naturel",
                   "Plus besoin d'ouvrir un notebook ni de lancer un script : elle écrit sa demande dans le canal "
                   "des agents.", s3_chat),
              Case("14:00 · ORCHESTRATEUR", "Orchestrateur", "La phrase devient une action précise",
                   "L'orchestrateur demande au LLM de comprendre la phrase : réentraîner, quel modèle, quelle "
                   "période. Avant de lancer quoi que ce soit, il consulte l'agent Coûts.", s3_understood),
              Case("14:01 · AGENT COÛTS", "Agent Coûts", "Le budget du mois serait dépassé",
                   "L'agent Coûts estime le prix de l'entraînement. Ajouté à ce qui a déjà été dépensé ce mois-ci, "
                   "il dépasse la limite : il faut l'accord d'un responsable.", s3_budget),
              Case("14:02 · VALIDATION", "Responsable", "Le responsable décide en un clic",
                   "Le responsable reçoit une demande claire : qui demande, pourquoi, combien cela coûte et de "
                   "combien le budget serait dépassé. Il clique sur Approuver ou Refuser.", s3_question),
              Case("14:05 · EXÉCUTION", "Agent Retrain", "Approuvé, lancé, tracé",
                   "L'entraînement démarre aussitôt, et chaque étape est écrite dans le journal d'actions : qui a "
                   "demandé, qui a validé, combien cela a coûté.", s3_launched),
          ), params=("text", "who")),
    Story("challenger", 4, "Le mauvais challenger",
          "Des données corrompues déclenchent un réentraînement. Le nouveau modèle est moins bon : il est bloqué "
          "et l'équipe est prévenue.",
          ("Obj. 1 · Qualité", "Garde-fou"), (
              Case("JEUDI · 03:00", "Agent Qualité", "Des données suspectes arrivent",
                   "En pleine nuit, un fichier de trajets arrive avec des valeurs aberrantes : durées négatives, "
                   "distances impossibles. L'agent Qualité voit une forte dérive et la signale.", s4_suspect),
              Case("03:05 · AGENT RETRAIN", "Agent Retrain", "Un challenger est entraîné quand même",
                   "Comme dans le premier scénario, l'orchestrateur demande un réentraînement. L'agent Retrain "
                   "produit une nouvelle version à partir de ces données. Rien n'est encore décidé.",
                   challenger_trained),
              Case("03:12 · COMPARAISON", "Agent Retrain", "Moins bon : promotion bloquée",
                   "La comparaison est sans appel : la nouvelle version fait plus d'erreurs que celle en production. "
                   "Un modèle moins bon n'est jamais promu.", s4_blocked),
              Case("JEUDI · 08:30", "Data scientist", "L'humain reprend la main",
                   "Au matin, l'équipe trouve une alerte : le challenger a été refusé, et les données de la nuit "
                   "semblent corrompues. Les agents protègent la production et laissent l'humain enquêter.",
                   s4_alert),
          ), params=("month",)),
)
BY_KEY = {s.key: s for s in STORIES}


class StoryBoard:
    """Lance les scénarios du storyboard et suit leur avancement."""

    def __init__(self, runner, log, notifier, orchestrator=None, status=None,
                 sleep: Callable[[float], None] = time.sleep, wait_healthy_seconds: float = 40.0,
                 incident_timeout_seconds: float = 900.0):
        self.runner = runner          # ui.scenarios.ScenarioRunner : contrôles Qualité, pannes, trafic
        self.log = log
        self.notifier = notifier
        self.orchestrator = orchestrator
        self.status = status
        self.sleep = sleep
        self.wait_healthy_seconds = wait_healthy_seconds
        self.incident_timeout_seconds = incident_timeout_seconds
        self.runs: list[StoryRun] = []
        self.lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.runner.enabled

    def get(self, run_id: str) -> StoryRun | None:
        return next((r for r in self.runs if r.id == run_id), None)

    def busy(self, key: str) -> bool:
        return any(r.story.key == key and r.status == "running" and not self.finished(r) for r in self.runs)

    # ------------------------------------------------ lancement
    def start(self, key: str, params: dict[str, Any] | None = None, wait: bool = False) -> StoryRun:
        if not self.enabled:
            raise PermissionError("scénarios désactivés (DEMO_CONTROLS=1 pour les activer)")
        story = BY_KEY[key]
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        if key == "chat" and self.orchestrator is None:
            raise RuntimeError("l'orchestrateur n'est pas branché sur cette page")
        with self.lock:
            if self.busy(key):
                raise RuntimeError("ce scénario est déjà en cours")
            run = StoryRun(uuid.uuid4().hex[:8], story, params, datetime.now(timezone.utc))
            self.runs.insert(0, run)
            del self.runs[20:]
        self.record("story_started", run)
        thread = threading.Thread(target=self._play, args=(run,), daemon=True)
        thread.start()
        if wait:
            thread.join()
        return run

    def _play(self, run: StoryRun) -> None:
        try:
            production = self.status.production() if self.status else None
            run.facts["production"] = production["version"] if production else None
            getattr(self, f"_play_{run.story.key}")(run)
        except BaseException as exc:  # SystemExit d'inject_failure compris
            run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
            run.note(f"échec du lancement : {run.error}")
            self.record("story_failed", run)

    def set_budget(self, run: StoryRun, value: float, why: str) -> None:
        value = live_budget.set_budget(value)
        run.facts["budget"] = value
        run.note(f"budget du mois réglé à {eur(value)} ({why})")
        self.log.record(DEMO_AGENT, "budget_changed", config.MODEL_NAME,
                        {"budget": value, "reason": why, "story_run": run.id})

    def _quality(self, run: StoryRun, scenario: str) -> None:
        month = int(run.params.get("month") or 3)
        r = self.runner.start(scenario, {"month": month})
        run.check_id = r.check_id
        run.note(f"contrôle Qualité de 2023-{month:02d} lancé (scénario « {scenario} », contrôle {r.check_id})")

    def _play_derive(self, run: StoryRun) -> None:
        self.set_budget(run, LARGE_BUDGET, "budget large : le réentraînement est automatique")
        self._quality(run, "drift")

    def _play_challenger(self, run: StoryRun) -> None:
        self.set_budget(run, LARGE_BUDGET, "budget large : le réentraînement est automatique")
        self._quality(run, "suspect")

    def _play_chat(self, run: StoryRun) -> None:
        spent, one_month = self.month_figures()
        # Un mois de données tient dans le budget, trois mois non : l'histoire du scénario 3
        self.set_budget(run, spent + 1.5 * one_month,
                        f"déjà dépensé {eur(spent)}, un mois d'entraînement estimé à {eur(one_month)}")
        run.trace_id = uuid.uuid4().hex[:8]
        text = run.params.get("text") or "@agents retrain le modèle taxi sur les 3 derniers mois"
        who = run.params.get("who") or "data scientist"
        run.note(f"message posté dans #mlops-agents par {who}")
        self.orchestrator.on_chat(text, who, request_id=run.trace_id)

    def _play_panne(self, run: StoryRun) -> None:
        r = self.runner.start("flaky-model", wait=True)
        if r.status != "ok":
            raise RuntimeError(r.output or "promotion du modèle défaillant impossible")
        run.facts["flaky_version"] = (r.output.split("Version ")[1].split()[0]
                                      if "Version " in r.output else "?")
        run.note(f"nouvelle version v{run.facts['flaky_version']} promue (bug sur les trajets de plus de 8 km)")
        self.sleep(self.wait_healthy_seconds)   # l'agent Fiabilité la voit saine au calme
        rps = float(run.params.get("rps") or 8)
        run.facts.update(rps=rps, traffic_at=datetime.now(timezone.utc).isoformat())
        self.runner.start_traffic(rps)
        run.note(f"pic de trafic : {rps:g} requêtes/s")
        traffic_at = datetime.now(timezone.utc)
        deadline = time.monotonic() + self.incident_timeout_seconds
        while time.monotonic() < deadline:
            run.trace_id = run.trace_id or tr.incident_after(self.log.engine, traffic_at)
            if run.trace_id and self.ctx(run).first("incident_resolved"):
                break
            self.sleep(5)
        self.runner.stop_traffic()
        run.note("fin du pic de trafic")

    def month_figures(self) -> tuple[float, float]:
        """(déjà dépensé ce mois, coût estimé d'un mois d'entraînement), comme le calcule l'agent Coûts."""
        from agents.cost import calculator, pricing
        from agents.cost.settings import CostConfig
        from agents.cost.sql_store import SqlCostStore
        store, cfg = SqlCostStore(engine=self.log.engine), CostConfig()
        cpu = calculator.estimate_cpu_seconds(store.training_history(config.MODEL_NAME), 1,
                                              cfg.default_cpu_seconds_per_month_of_data)
        one_month = pricing.price_options(cpu, cfg)["standard"]["cost"]
        return store.month_total(calculator.month_key(datetime.now())), one_month

    def record(self, action: str, run: StoryRun) -> None:
        details = {"story_run": run.id, "story": run.story.key, "title": run.story.title, **run.params}
        if run.error:
            details["error"] = run.error
        try:
            self.log.record(DEMO_AGENT, action, config.MODEL_NAME, details)
        except Exception:
            pass

    # ------------------------------------------------ suivi
    def trace_id(self, run: StoryRun) -> str | None:
        if run.trace_id is None and run.check_id:
            run.trace_id = tr.trace_of_check(self.log.engine, run.check_id)
        return run.trace_id

    def ctx(self, run: StoryRun) -> Ctx:
        steps: list[tr.Step] = []
        trace_id = self.trace_id(run)
        if trace_id:
            t = tr.load_trace(self.log.engine, trace_id, self.notifier)
            steps = t.steps if t else []
        if run.check_id and not any(s.action in ("drift_detected", "quality_ok") for s in steps):
            details = action_log.c.details
            with self.log.engine.connect() as conn:
                rows = [dict(r._mapping) for r in conn.execute(select(action_log).where(
                    action_log.c.agent == "agent_qualite", details["check_id"].as_string() == run.check_id))]
            steps = [tr._step(r) for r in rows] + steps
        return Ctx(run, steps, self)

    def progress(self, run: StoryRun) -> list[dict]:
        """Une entrée par case : état (done, variant, failed, current, pending) et ce qui s'est passé."""
        ctx = self.ctx(run)
        out, blocked = [], False
        for i, case in enumerate(run.story.cases):
            result = None if blocked else case.check(ctx)
            if result is None:
                state = PENDING if blocked or run.status == "error" else CURRENT
                out.append({"n": i + 1, "case": case, "state": state, "detail": ""})
                blocked = True
                continue
            state, detail = result
            out.append({"n": i + 1, "case": case, "state": state, "detail": detail})
            if state in (CURRENT, FAILED):
                blocked = True
        return out

    def finished(self, run: StoryRun) -> bool:
        if run.status == "error":
            return True
        last = self.progress(run)[-1]
        return last["state"] in (DONE, VARIANT, FAILED)

    def related_rows(self, run: StoryRun) -> list[dict]:
        """Lignes du journal du scénario (banc de test compris), pour l'encart « journal »."""
        ids = [x for x in (run.trace_id, run.check_id) if x]
        if not ids:
            return []
        d = action_log.c.details
        conds = [d["request_id"].as_string().in_(ids), d["incident_id"].as_string().in_(ids),
                 d["check_id"].as_string().in_(ids), d["story_run"].as_string() == run.id]
        with self.log.engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(
                select(action_log).where(or_(*conds)).order_by(action_log.c.id.desc()).limit(40))]
        return rows
