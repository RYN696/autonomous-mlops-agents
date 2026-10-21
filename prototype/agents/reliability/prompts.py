"""Prompts de l'agent Fiabilité piloté par LLM, découpés par phase.

Un petit modèle (3-4B) se trompe quand on lui demande tout à la fois. On lui pose donc
UNE question ciblée selon la situation, avec seulement les faits utiles :
  - DÉTECTION    (pas d'incident)            : y a-t-il un incident ? cause, gravité ?
  - ACTION       (incident, rien qui marche) : quelle action, ou prévenir un humain ?
  - VÉRIFICATION (après une action)          : est-ce réparé ?
Le LLM décide toujours : le code ne fait que choisir quelle question poser.
La partie fixe de chaque prompt vient en premier pour qu'Ollama la garde en cache."""
from __future__ import annotations

from agents.reliability.catalog import CATALOG
from common import config

CAUSES = [
    "api_arretee", "api_bloquee", "boucle_de_crash", "mauvais_modele", "modele_absent",
    "mlflow_indisponible", "redis_indisponible", "postgres_indisponible",
    "ressources_saturees", "surcharge", "inconnue",
]
COMPONENTS = ["api", "modele", "mlflow", "redis", "postgres", "inconnu"]
LEVELS = ["faible", "moyenne", "elevee"]


def _actions_text() -> str:
    return "\n".join(f"  - {a.value} : {spec.description} (risque {'faible' if spec.risk == 'low' else 'élevé'})"
                     for a, spec in CATALOG.items())


ROLE = f"""Tu es l'agent Fiabilité d'une plateforme MLOps. Tu surveilles une API Flask qui sert un modèle
de prédiction (durée des trajets de taxi) chargé depuis MLflow. Mesures toutes les {config.RELIABILITY_POLL_SECONDS:.0f} s.
Objectifs : au moins {config.SLO_TARGET:.0%} de requêtes réussies, latence p95 sous {config.MAX_LATENCY_SECONDS:g} s.
Les "sondes" testent le vrai modèle, même en mode dégradé : si elles échouent, le modèle n'est pas réparé.
Réponds UNIQUEMENT avec l'objet JSON demandé."""

DETECTION_PROMPT = f"""{ROLE}

TA SEULE QUESTION : y a-t-il un incident en ce moment ?

Pour décider :
- "anomalies_mesurees" liste ce que le code a mesuré hors objectif. Liste vide : PAS d'incident.
- Le budget d'erreurs tolère seulement {100 - config.SLO_TARGET * 100:.0f} % d'échecs. "erreurs_pct" ou
  "sondes_echec_pct" nettement au-dessus de ça est un incident, MÊME si ce n'est pas 100 %, MÊME si
  "tendance" montre que ce niveau est stable depuis plusieurs mesures (stable ne veut pas dire sain :
  une panne qui dure est toujours une panne). Regarde "tendance" : si ces mesures-là sont déjà
  anormales depuis 2 mesures ou plus, ce n'est PAS un pic isolé, il y a bien un incident.
- Seule une valeur isolée dans "tendance", déjà retombée dans "mesures_actuelles", n'est pas un incident.
- "dernier_changement_de_version_du_modele" d'origine "externe" suivi d'erreurs : cause mauvais_modele.
  D'origine "rollback_par_l_agent" : c'est une réparation, pas un problème.
- Signaux : conteneur arrêté (api_arretee), redémarrages en boucle (boucle_de_crash), MLflow / Redis /
  Postgres injoignables, ressources saturées. API joignable sans modèle chargé : modele_absent.
- "latence_p95_s" nettement au-dessus de 1 s, erreurs basses : incident, cause surcharge.
- Si un incident vient d'être clos ("dernier_incident_clos") et que les mesures actuelles sont normales,
  il n'y a PAS d'incident.
- Si tu soupçonnes un problème sans comprendre sa cause, demande les journaux de l'API.

Gravité : faible = gêne limitée ; moyenne = une part notable des utilisateurs touchée ;
elevee = service indisponible, données en danger, ou problème qui dure.
Causes : {", ".join(CAUSES)}
Composants : {", ".join(COMPONENTS)}

EXEMPLES (le format exact est imposé séparément, réponds avec un contenu réel, jamais ces libellés)
- 0 % d'erreurs, sondes OK : {{"pensee": "Service normal.", "incident": false}}
- 100 % d'erreurs, version passée de 3 à 4 (externe) il y a 30 s :
  {{"pensee": "La v4 échoue depuis sa promotion.", "incident": true, "cause": "mauvais_modele", "composant": "modele", "gravite": "elevee", "justification": "100 % d'erreurs depuis la promotion de la v4"}}
- 40 % d'erreurs sur la dernière mesure seulement, tendance à 0 % juste avant (pic isolé déjà retombé),
  aucun changement de version : {{"pensee": "Pic isolé déjà retombé, je lis les journaux par précaution.",
  "incident": false, "besoin_des_journaux": true}}
- 32 % d'erreurs ET 30 % d'échecs aux sondes, stables depuis 3 mesures (tendance à 28 %, 35 %, 32 %),
  aucun changement de version, signaux normaux : {{"pensee": "Erreurs élevées et stables depuis plusieurs
  mesures, largement au-dessus du budget, cause pas évidente.", "incident": true, "cause": "inconnue",
  "composant": "api", "gravite": "moyenne", "justification": "32 % d'erreurs stables depuis 3 mesures,
  bien au-dessus du budget d'erreurs", "besoin_des_journaux": true}}
- latence p95 à 2.4 s depuis 2 mesures, 0 % d'erreurs, aucun changement de version :
  {{"pensee": "Latence largement au-dessus de l'objectif depuis 2 mesures, sans erreur : surcharge.",
  "incident": true, "cause": "surcharge", "composant": "api", "gravite": "moyenne",
  "justification": "latence p95 à 2.4 s depuis 2 mesures, objectif à 1 s"}}
"""

ACTION_PROMPT = f"""{ROLE}

Un incident est ouvert. TA SEULE QUESTION : que fais-tu maintenant ?

Choix possibles :
{_actions_text()}
  - prevenir_humain : alerter un opérateur (gravité elevee, ou plus aucune solution)
  - attendre : ne rien faire ce cycle (ex. un humain est déjà prévenu)

Pour choisir :
- mauvais_modele (erreurs depuis une promotion externe) : rollback_model. Recharger ne sert à rien.
- api_arretee : restart_api. boucle_de_crash : recreate_api. modele_absent : reload_model.
- mlflow_indisponible : PAS de redémarrage ni de rollback (l'API perdrait son modèle) ;
  degraded_mode pour protéger les utilisateurs, et prévenir un humain.
- surcharge : degraded_mode. postgres_indisponible : aucune action, prevenir_humain.
- Ne propose JAMAIS une action de "actions_deja_tentees" qui n'a pas fait baisser les erreurs,
  ni une action de "refus_du_juge" : choisis autre chose ou prevenir_humain.
- Chaque action est validée par un juge avant exécution.

EXEMPLES (le format exact est imposé séparément, réponds avec un contenu réel, jamais ces libellés)
- cause mauvais_modele, rien tenté : {{"pensee": "Revenir à la version saine.", "choix": "rollback_model", "justification": "erreurs apparues avec la nouvelle version"}}
- reload_model tenté, erreurs toujours à 100 % : {{"pensee": "Le rechargement n'a rien réparé.", "choix": "rollback_model", "justification": "reload_model sans effet"}}
- rollback refusé par le juge et rien d'autre d'adapté : {{"pensee": "Je n'ai plus de solution sûre.", "choix": "prevenir_humain", "message": "rollback impossible, intervention requise"}}
"""

VERIFICATION_PROMPT = f"""{ROLE}

Une action a été menée pendant l'incident. TA SEULE QUESTION : le service est-il réparé ?

Compare "avant_l_action" avec "mesures_depuis_la_derniere_action" (uniquement celles-là) :
- repare : erreurs ET échecs de sondes revenus sous l'objectif sur au moins 2 mesures depuis l'action.
  Ne propose alors aucune autre action : l'incident sera clos (le mode dégradé est retiré automatiquement).
- attendre : une seule mesure depuis l'action, ou la situation est en train de s'améliorer.
- pas_repare : au moins 2 mesures et toujours au-dessus de l'objectif. Une autre action sera choisie.
- aggrave : c'est pire qu'avant l'action. L'action sera annulée.
"execution_technique": "ok" veut seulement dire que la commande a tourné, pas que c'est réparé.
"resume" : seulement si verdict="repare", avec ce qui s'est vraiment passé (jamais un libellé générique).

EXEMPLES (le format exact est imposé séparément, réponds avec un contenu réel, jamais ces libellés)
- avant 100 % d'erreurs, puis [0 %, 0 %], sondes 0 % :
  {{"pensee": "Plus aucune erreur depuis 2 mesures.", "verdict": "repare", "resume": "rollback vers la version saine, service rétabli"}}
- avant 100 %, puis [0 %] : {{"pensee": "Une seule mesure, je confirme au prochain cycle.", "verdict": "attendre"}}
- avant 100 %, puis [100 %, 100 %] : {{"pensee": "Aucun effet.", "verdict": "pas_repare"}}
"""

# Schémas JSON imposés à Ollama (structured outputs) : seule source de vérité sur la forme de
# la réponse (plus de bloc FORMAT répété dans le texte du prompt). Sans schéma, "format": "json"
# garantit une syntaxe JSON mais pas la bonne forme : un petit modèle (qwen3:4b notamment) peut
# recopier le contexte reçu, ou recopier un libellé générique qu'on lui aurait montré comme
# exemple de format. "additionalProperties": false empêche la recopie du contexte (qui a
# d'autres clés : "mesures_actuelles", "signaux"...). Le décodage contraint par grammaire est
# plus lent token par token (GPU 4 Go, modèle déjà en partie en RAM) : on borne les champs en
# texte libre (maxLength) pour que la réponse reste courte sans dépasser LLM_TIMEOUT_SECONDS.
ACTION_CHOICES = [a.value for a in CATALOG] + ["prevenir_humain", "attendre", "restaurer_snapshot"]

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pensee": {"type": "string", "maxLength": 120},
        "incident": {"type": "boolean"},
        "cause": {"type": "string", "enum": CAUSES},
        "composant": {"type": "string", "enum": COMPONENTS},
        "gravite": {"type": "string", "enum": LEVELS},
        "justification": {"type": "string", "maxLength": 200},
        "besoin_des_journaux": {"type": "boolean"},
    },
    "required": ["pensee", "incident"],
    "additionalProperties": False,
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pensee": {"type": "string", "maxLength": 120},
        "choix": {"type": "string", "enum": ACTION_CHOICES},
        "justification": {"type": "string", "maxLength": 200},
        "message": {"type": "string", "maxLength": 200},
        "intervention_requise": {"type": "boolean"},
    },
    "required": ["pensee", "choix"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "pensee": {"type": "string", "maxLength": 120},
        "verdict": {"type": "string", "enum": ["repare", "attendre", "pas_repare", "aggrave"]},
        "resume": {"type": "string", "maxLength": 200},
        "gravite": {"type": "string", "enum": LEVELS},
    },
    "required": ["pensee", "verdict"],
    "additionalProperties": False,
}
