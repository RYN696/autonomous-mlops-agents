"""Règles déterministes de l'agent Fiabilité : des garde-fous en code autour du LLM.

Le LLM décide, mais un petit modèle (3-4B) se trompe sur des faits que le code sait
vérifier seul. Testé en direct :
  - il a ouvert un incident "surcharge" pour une latence de 0,015 s ("bien au-dessus
    de l'objectif de 1 s") ;
  - il a jugé "aggravé" un service qui redémarrait encore, et relancé l'API en boucle ;
  - il a diagnostiqué une "surcharge" juste après une promotion de modèle, et le juge a
    refusé le rollback.
Ces règles ne s'appliquent qu'aux mesures (seuils de config.py) : elles écartent une
décision contredite par les faits, ou prennent la décision critique (rollback) quand
les faits suffisent, comme le demande la politique du projet."""
from __future__ import annotations

from agents.reliability.signals import Signals
from common import config


def error_threshold_pct() -> float:
    """Part d'échecs tolérée par le SLO, en % (99 % de réussite -> 1 %)."""
    return round((1 - config.SLO_TARGET) * 100, 3)


def breaches(m: dict) -> list[str]:
    """Mesures hors objectif (m = mesures d'un cycle, complètes ou abrégées)."""
    found, limit, max_latency = [], error_threshold_pct(), config.MAX_LATENCY_SECONDS
    if not m.get("api_joignable", True):
        return ["api_injoignable"]
    if m.get("erreurs_pct", 0) > limit:
        found.append(f"erreurs_pct {m['erreurs_pct']} > {limit}")
    if m.get("sondes_echec_pct", 0) > limit:
        found.append(f"sondes_echec_pct {m['sondes_echec_pct']} > {limit}")
    for key in ("latence_p95_s", "sondes_p95_s"):
        if m.get(key) is not None and m[key] > max_latency:
            found.append(f"{key} {m[key]} > {max_latency}")
    if m.get("modele_charge") is False:
        found.append("modele_non_charge")
    return found


def signal_breaches(s: Signals | None) -> list[str]:
    if s is None:
        return []
    found = []
    if s.container_status != "running":
        found.append(f"conteneur {s.container_status}")
    if s.restarts_recent >= 3:
        found.append(f"{s.restarts_recent} redémarrages récents")
    for name in ("mlflow_ok", "redis_ok", "postgres_ok"):
        if getattr(s, name) is False:
            found.append(f"{name} = false")
    for name in ("memory_percent", "cpu_percent"):
        value = getattr(s, name)
        if value is not None and value >= config.RELIABILITY_SATURATION_PCT:
            found.append(f"{name} {value} >= {config.RELIABILITY_SATURATION_PCT}")
    return found


def anomalies(m: dict, s: Signals | None) -> list[str]:
    """Tout ce qui est objectivement anormal : vide = rien ne justifie un incident."""
    return breaches(m) + signal_breaches(s)


def healthy(m: dict) -> bool:
    return not breaches(m)


def worse(before: dict | None, after: dict | None) -> bool:
    """La situation est-elle vraiment pire qu'avant l'action ? (100 % -> 100 % ne l'est pas)"""
    if not before or not after:
        return False
    if before["api_joignable"] and not after["api_joignable"]:
        return True
    margin = config.RELIABILITY_WORSE_MARGIN_PCT
    return (after["erreurs_pct"] > before["erreurs_pct"] + margin
            or after["sondes_echec_pct"] > before["sondes_echec_pct"] + margin)


def bad_model(change: dict | None, m: dict, s: Signals | None, now: float) -> str | None:
    """Promotion externe récente, service sain juste avant, erreurs depuis : le modèle est en
    cause et la bonne action est le rollback. Renvoie la raison, ou None si la règle ne s'applique pas.
    Si la nouvelle version a d'abord fonctionné (sain_apres), les erreurs viennent peut-être d'ailleurs
    (pic de trafic...) : la règle ne s'applique pas, voir restart_did_not_help."""
    if not change or not change["origine"].startswith("externe") or not change.get("sain_avant"):
        return None
    if change.get("sain_apres"):
        return None
    if now - change["at"] > config.RELIABILITY_BAD_MODEL_WINDOW_SECONDS:
        return None
    if not m["api_joignable"] or not (m["erreurs_pct"] > error_threshold_pct()
                                      or m["sondes_echec_pct"] > error_threshold_pct()):
        return None
    if s is not None and not s.mlflow_ok:       # sans MLflow, un rollback ferait perdre le modèle
        return None
    return (f"erreurs depuis la promotion externe de la v{change['vers']} "
            f"(v{change['de']} saine avant) : {m['erreurs_pct']} % d'erreurs, "
            f"{m['sondes_echec_pct']} % d'échecs aux sondes")


def restart_did_not_help(change: dict | None, actions: list[dict], m: dict, s: Signals | None,
                         now: float) -> str | None:
    """Le redémarrage (la réponse la plus simple) a été fait et les erreurs restent, alors qu'une
    nouvelle version du modèle a été promue récemment : c'est elle qui est en cause -> rollback vers la
    dernière version saine. Renvoie la raison, ou None si la règle ne s'applique pas."""
    if not change or not change["origine"].startswith("externe"):
        return None
    if now - change["at"] > config.RELIABILITY_SUSPECT_MODEL_HOURS * 3600:
        return None
    if not any(a["action"] in RESTARTS and a.get("success") for a in actions):
        return None
    if not m["api_joignable"] or not (m["erreurs_pct"] > error_threshold_pct()
                                      or m["sondes_echec_pct"] > error_threshold_pct()):
        return None
    if s is not None and not s.mlflow_ok:       # sans MLflow, un rollback ferait perdre le modèle
        return None
    return (f"le redémarrage n'a pas suffi : {m['erreurs_pct']} % d'erreurs avec la v{change['vers']}, "
            f"promue il y a {int((now - change['at']) // 60)} min (la v{change['de']} fonctionnait) : "
            "retour à la dernière version saine")


def restart_first(change: dict | None, actions: list[dict], m: dict, s: Signals | None, now: float) -> str | None:
    """Erreurs 500 avec l'API joignable, après une promotion récente, et rien encore tenté : la première
    réponse est la plus simple, redémarrer (sans risque, autorisée seule). Pas le mode dégradé : il
    masquerait un modèle cassé, l'incident se fermerait et la version défaillante resterait en production.
    Si le redémarrage ne suffit pas, restart_did_not_help désigne le modèle (rollback)."""
    if not change or not change["origine"].startswith("externe") or not change.get("sain_avant") or actions:
        return None
    if now - change["at"] > config.RELIABILITY_SUSPECT_MODEL_HOURS * 3600:
        return None
    if not m["api_joignable"] or m["erreurs_pct"] <= error_threshold_pct():
        return None
    if s is not None and (not s.mlflow_ok or s.restarts_recent >= 3):
        return None
    return (f"{m['erreurs_pct']} % d'erreurs avec l'API joignable, v{change['vers']} promue il y a "
            f"{int((now - change['at']) // 60)} min : premier essai, redémarrer l'API")


DEPENDENCY_OF = {"mlflow_indisponible": "mlflow_ok", "redis_indisponible": "redis_ok",
                 "postgres_indisponible": "postgres_ok"}


STOPPED = {"exited", "dead"}


def checked_cause(cause: str, m: dict, s: Signals | None) -> str:
    """Cause du LLM contredite par les signaux -> "inconnue" (vu en direct : "api_arretee"
    pour une API qui tournait et répondait, avec 30 % d'erreurs)."""
    if s is None:
        return cause
    if s.container_status in STOPPED:
        return "api_arretee"  # vu en direct (agent en conteneur) : "surcharge" pour une API arrêtée
    if cause == "api_arretee" and s.container_status == "running" and m.get("api_joignable"):
        cause = "inconnue"
    if cause in DEPENDENCY_OF and getattr(s, DEPENDENCY_OF[cause]) is True:
        cause = "inconnue"
    return mlflow_cause(cause, s)


MODEL_CAUSES = {"mauvais_modele", "modele_absent", "inconnue"}


def mlflow_cause(cause: str, s: Signals | None) -> str:
    """MLflow injoignable : une cause "modèle" vient en fait de MLflow (vu en direct :
    "mauvais_modele" alors que l'API ne pouvait plus charger son modèle)."""
    if s is not None and s.mlflow_ok is False and cause in MODEL_CAUSES:
        return "mlflow_indisponible"
    return cause


NEEDS_MLFLOW = {"restart_api", "recreate_api", "reload_model", "rollback_model"}
RESTARTS = {"restart_api", "recreate_api"}


def forbidden(action: str, s: Signals | None) -> str | None:
    """Règles absolues de la grille du juge que le code sait vérifier seul (règles 1 et 6) :
    appliquées avant le juge LLM, qui peut se tromper sur un fait (vu en direct)."""
    if s is None:
        return None
    if action in NEEDS_MLFLOW and s.mlflow_ok is False:
        return f"{action} interdit : MLflow injoignable, l'API perdrait son modèle"
    if action in RESTARTS and s.restarts_recent >= 3:
        return f"{action} interdit : {s.restarts_recent} redémarrages récents (boucle de crash)"
    return None
