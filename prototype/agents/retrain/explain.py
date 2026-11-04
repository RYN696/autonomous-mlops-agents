"""Message court qui explique la décision de l'agent Retrain à l'équipe.

Avec une clé LLM_API_KEY, Grok (API xAI) rédige 2 ou 3 phrases à partir des chiffres.
Sans clé, si l'appel échoue, ou si le texte cite un nombre absent des faits,
on garde un message écrit par le code : l'agent n'est jamais bloqué par le LLM.
Le LLM explique, il ne décide pas : la décision est déjà prise quand on l'appelle.
"""
from __future__ import annotations

import re

import requests

from common import config

SYSTEM = (
    "Tu rédiges les messages d'une équipe MLOps étudiante. "
    "On te donne les FAITS d'une décision déjà prise par l'agent Retrain sur le modèle taxi-duration "
    "(durée des trajets de taxi à New York). "
    "Écris en français 2 ou 3 phrases courtes, sans markdown : ce qui a été décidé, les chiffres clés et pourquoi. "
    "Commence par ✅ si le modèle est promu, par ⛔ si la promotion est bloquée. "
    "Recopie les nombres exactement comme dans les FAITS et n'en utilise aucun autre. "
    "Ne propose aucune autre décision."
)


def facts(decision: str, r: dict) -> list[str]:
    """Les chiffres de la décision, déjà formatés : le LLM ne calcule rien."""
    lines = [
        f"Décision : {'promu en production' if decision == 'promoted' else 'promotion bloquée'}",
        f"Challenger : v{r['challenger_version']}, RMSE {r['challenger_rmse']:.2f} min",
    ]
    if r.get("production_version") is None:
        lines.append("Production : aucun modèle avant celui-ci")
    else:
        lines.append(f"Production : v{r['production_version']}, RMSE {r['production_rmse']:.2f} min")
        lines.append(f"Gain : {r['improvement'] * 100:.1f} % d'erreur en moins "
                     f"(minimum exigé : {r['min_improvement'] * 100:.0f} %)")
    if r.get("period"):
        lines.append(f"Entraîné sur : {', '.join(r['period'])}")
    params = r.get("params") or {}
    if params:
        lines.append(f"Réglages choisis par Optuna : profondeur {params['max_depth']}, "
                     f"learning rate {params['learning_rate']:.3f}, {params['n_estimators']} arbres")
    return lines


def numbers(text: str) -> set[float]:
    """Nombres d'un texte ; « 6,1 » et « 6.1 » comptent pareil."""
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def numbers_ok(text: str, fact_lines: list[str]) -> bool:
    """Garde-fou : le texte ne cite que des nombres présents dans les faits."""
    return numbers(text) <= numbers(" ".join(fact_lines))


def template_message(decision: str, r: dict) -> str:
    """Message sans LLM, toujours disponible."""
    period = ", ".join(r.get("period") or [])
    if decision == "promoted" and r.get("production_version") is None:
        return (f"✅ taxi-duration v{r['challenger_version']} mise en production "
                f"(premier modèle, RMSE {r['challenger_rmse']:.2f} min).")
    if decision == "promoted":
        return (f"✅ v{r['challenger_version']} promue : {r['improvement'] * 100:.1f} % d'erreur en moins "
                f"que v{r['production_version']} (RMSE {r['challenger_rmse']:.2f} contre "
                f"{r['production_rmse']:.2f} min), entraînée sur {period}.")
    return (f"⛔ Challenger v{r['challenger_version']} refusé : RMSE {r['challenger_rmse']:.2f} contre "
            f"{r['production_rmse']:.2f} min pour v{r['production_version']} "
            f"({r['improvement'] * 100:.1f} % de gain, il faut au moins {r['min_improvement'] * 100:.0f} %). "
            f"La v{r['production_version']} reste en production.")


def llm_message(decision: str, r: dict) -> str | None:
    """Message rédigé par Grok, ou None si ce n'est pas possible."""
    if not config.LLM_API_KEY:
        return None
    body = {
        "model": config.LLM_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "FAITS :\n- " + "\n- ".join(facts(decision, r))},
        ],
    }
    try:
        response = requests.post(
            config.LLM_API_URL,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json=body,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"LLM indisponible, message par défaut : {exc}")
        return None
    return text or None


def explain_decision(decision: str, result: dict) -> tuple[str, str]:
    """Renvoie (message, source), avec source = "llm" ou "template"."""
    text = llm_message(decision, result)
    if text and numbers_ok(text, facts(decision, result)):
        return text, "llm"
    if text:
        print(f"Texte du LLM refusé (nombre inventé) : {text}")
    return template_message(decision, result), "template"