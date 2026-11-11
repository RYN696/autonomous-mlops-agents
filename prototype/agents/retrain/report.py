"""Rapport de fin de mission de l'agent Retrain.

Écrit par le code avec les faits de l'entraînement (aucun nombre inventé possible) : la demande,
les données, l'optimisation Optuna, la comparaison avec la production, la décision et la suite.
Le message de explain.py (LLM vérifié, ou phrase du code) sert de résumé.
Journalisé sous l'action « retrain_report », avec le request_id pour le relier au parcours.
"""
from __future__ import annotations

from typing import Any

from common import config


def _minutes(seconds: float | None) -> str:
    s = float(seconds or 0)
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def _who(f: dict) -> str:
    who = f"demandé par {f.get('requested_by') or '?'}"
    return who + (f", validé par {f['approved_by']}" if f.get("approved_by") else " (automatique)")


def build_report(decision: str, f: dict[str, Any]) -> dict[str, Any]:
    """decision : promoted, blocked, skipped ou failed ; f : les faits de l'entraînement."""
    period = ", ".join(f.get("period") or []) or "?"
    base = {"agent": "retrain", "request_id": f.get("request_id"), "period": f.get("period")}
    if decision == "skipped":
        return {**base, "title": f"Réentraînement ignoré · {period}", "verdict": f.get("reason", ""),
                "level": "muted", "headline": [], "sections": [{"title": "Demande", "lines": [_who(f)]}],
                "next": "Rien n'a été entraîné : la production ne change pas.",
                "summary": None, "summary_source": "template"}
    if decision == "failed":
        return {**base, "title": f"Entraînement en échec · {period}", "verdict": "erreur pendant l'entraînement",
                "level": "bad", "headline": [], "sections": [
                    {"title": "Demande", "lines": [_who(f)]}, {"title": "Erreur", "lines": [str(f.get("error"))]}],
                "next": "Un incident a été ouvert ; la production ne change pas.",
                "summary": None, "summary_source": "template"}

    promoted = decision == "promoted"
    prod_rmse, gain = f.get("production_rmse"), f.get("improvement")
    params = f.get("params") or {}
    cpu = float(f.get("training_seconds") or 0) + float(f.get("tuning_seconds") or 0)
    headline = [
        {"label": "Challenger", "value": f"v{f.get('challenger_version')}", "hint": f"RMSE {f['challenger_rmse']:.2f} min"},
        {"label": "Production", "value": f"v{f['production_version']}" if f.get("production_version") else "—",
         "hint": f"RMSE {prod_rmse:.2f} min" if prod_rmse is not None else "aucun modèle avant"},
        {"label": "Gain", "value": f"{gain * 100:+.1f} %" if gain is not None else "—",
         "hint": f"minimum {f.get('min_improvement', config.MIN_IMPROVEMENT) * 100:.0f} %",
         "alert": gain is not None and not promoted},
        {"label": "Calcul", "value": _minutes(cpu), "hint": f"≈ {cpu * config.COST_PER_CPU_SECOND:.4f} €"},
    ]
    data = [_who(f), f"Entraîné sur : {period}"
            + (f" ({f['rows_train']:,} trajets)".replace(",", " ") if f.get("rows_train") else "")]
    if f.get("test_month"):
        data.append(f"Testé sur : {f['test_month']}"
                    + (f" ({f['rows_test']:,} trajets, les mêmes pour les deux modèles)".replace(",", " ")
                       if f.get("rows_test") else ""))
    if f.get("data_file"):
        data.append(f"Fichier reçu utilisé : {f['data_file']}")
    tuning = [f"{f.get('tuning_trials', 0)} essais Optuna en {_minutes(f.get('tuning_seconds'))}, "
              f"entraînement final en {_minutes(f.get('training_seconds'))}"]
    if params:
        tuning.append(f"Réglages retenus : profondeur {params.get('max_depth')}, "
                      f"learning rate {float(params.get('learning_rate', 0)):.3f}, {params.get('n_estimators')} arbres")
    rule = f"Règle : promotion si l'erreur baisse d'au moins {f.get('min_improvement', config.MIN_IMPROVEMENT) * 100:.0f} %"
    if promoted:
        verdict = "promu en production"
        decided = [rule, f"v{f['challenger_version']} est maintenant servie par l'API"
                   + ("" if f.get("api_reloaded", True) else " (rechargement de l'API à vérifier)")]
        nxt = f"Retour possible à v{f['production_version']} en un clic." if f.get("production_version") else \
            "Premier modèle en production."
    else:
        verdict = "promotion bloquée"
        decided = [rule, f"v{f.get('production_version')} reste en production"]
        nxt = "Le challenger reste dans MLflow (tag decision=blocked) pour analyse ; aucune action sur l'API."
    return {
        **base, "title": f"{'Modèle promu' if promoted else 'Promotion bloquée'} · v{f.get('challenger_version')}",
        "verdict": verdict, "level": "ok" if promoted else "warn", "headline": headline,
        "sections": [{"title": "Demande et données", "lines": data}, {"title": "Optimisation", "lines": tuning},
                     {"title": "Décision", "lines": decided}],
        "next": nxt, "summary": f.get("message"), "summary_source": f.get("message_source", "template"),
    }
