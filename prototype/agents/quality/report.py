"""Rapport de fin de contrôle de l'agent Qualité.

Écrit par le code à partir des mesures du contrôle (aucun nombre inventé possible) :
chiffres clés, données, dérive, cause probable, décision et suite. L'explication du LLM,
déjà vérifiée par explain.py (nombres présents dans les constats), est jointe en résumé.
Journalisé sous l'action « quality_report », avec le check_id pour le relier au parcours.
"""
from __future__ import annotations

from typing import Any

from common import config

REASONS = {"invalid_duration": "durées invalides", "invalid_distance": "distances invalides",
           "invalid_passenger_count": "nombre de passagers invalide",
           "missing_required_columns": "colonnes obligatoires manquantes"}


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f} %"


def _median(name: str, unit: str, m: dict | None) -> str | None:
    if not m:
        return None
    change = f" ({m['change_pct']:+d} %)" if m.get("change_pct") is not None else ""
    return f"{name} : {m['ref']} → {m['cur']} {unit}{change}"


def build_report(event_type: str, p: dict[str, Any]) -> dict[str, Any]:
    reason = p.get("reason")
    if reason == "corrupted_input_data":
        title, verdict, level = "Données corrompues", "données refusées : pas de réentraînement", "bad"
        nxt = "L'orchestrateur refuse de réentraîner sur ces données ; un humain doit corriger la source."
    elif p.get("suspect"):
        title, verdict, level = "Données suspectes", "dérive avec des valeurs impossibles : à faire vérifier", "warn"
        nxt = ("Le réentraînement peut partir, mais le fichier reçu est gardé pour vérification : "
               "Retrain comparera le challenger à la production avant toute promotion.")
    elif event_type == "drift_detected":
        title, verdict, level = "Dérive détectée", "le modèle ne voit plus les mêmes données", "warn"
        nxt = "L'orchestrateur demande le coût, puis un réentraînement selon les règles d'autonomie."
    else:
        title, verdict, level = "Données conformes", "rien à faire", "ok"
        nxt = "Aucune action : prochain contrôle dans {:g} min.".format(config.QUALITY_CHECK_MINUTES)

    ds, dthr = p.get("drift_score", 0.0), p.get("drift_threshold", config.DRIFT_THRESHOLD)
    pdr, pthr = p.get("prediction_drift", 0.0), p.get("column_drift_threshold", config.COLUMN_DRIFT_THRESHOLD)
    inv = p.get("invalid_share", 0.0)
    headline = [
        {"label": "Lignes contrôlées", "value": f"{p.get('row_count', 0):,}".replace(",", " ")},
        {"label": "Dérive des colonnes", "value": f"{ds:.2f}", "hint": f"seuil {dthr:.2f}", "alert": ds >= dthr},
        {"label": "Dérive des prédictions", "value": f"{pdr:.2f}", "hint": f"seuil {pthr:.2f}", "alert": pdr >= pthr},
        {"label": "Lignes invalides", "value": pct(inv), "hint": f"seuil {pct(config.CRITICAL_INVALID_SHARE, 0)}",
         "alert": inv >= config.CRITICAL_INVALID_SHARE},
    ]
    data = [f"{p.get('invalid_rows', 0)} lignes invalides sur {p.get('row_count', 0)} ({pct(inv)})"]
    if p.get("invalid_reasons"):
        data.append("Problèmes : " + ", ".join(REASONS.get(r, r) for r in p["invalid_reasons"]))
    if p.get("impossible_share"):
        data.append(f"Valeurs physiquement impossibles : {pct(p['impossible_share'], 2)} "
                    f"(alerte à partir de {pct(config.SUSPECT_SHARE, 1)})")
    if p.get("data_file"):
        data.append(f"Fichier reçu gardé : {p['data_file']}")
    drift = [f"{p.get('num_drifted_columns', 0)} colonne(s) sur 4 ont changé de distribution"]
    rc = p.get("root_cause") or {}
    cause = [line for line in (
        _median("Durée prédite médiane", "min", rc.get("prediction_median_min")),
        _median("Distance médiane", "miles", rc.get("distance_median_miles")),
    ) if line]
    zones = rc.get("top_pickup_zone_shifts") or []
    if zones:
        z = max(zones, key=lambda s: abs(s["cur_share_pct"] - s["ref_share_pct"]))
        cause.append(f"Zone de départ la plus changée : {z['zone']} ({z['ref_share_pct']} % → {z['cur_share_pct']} % des trajets)")
    sections = [{"title": "Données", "lines": data}, {"title": "Dérive", "lines": drift}]
    if cause:
        sections.append({"title": "Cause probable", "lines": cause})
    return {
        "agent": "quality", "title": f"{title} · {p.get('period', '?')}", "verdict": verdict, "level": level,
        "headline": headline, "sections": sections, "next": nxt,
        # l'explication du LLM Qualité (en anglais) reste dans le détail : les chiffres clés suffisent
        "summary": None, "llm_note": p.get("explanation") if p.get("explanation_source") == "llm" else None,
        "llm_model": p.get("llm_model"),
        "period": p.get("period"), "check_id": p.get("check_id"),
    }
