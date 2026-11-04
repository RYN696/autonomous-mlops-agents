"""Détection de pic de coût : règle déterministe, sans LLM.

Un pic, c'est une mesure qui vaut au moins `factor` fois la médiane des dernières
mesures du même genre (la médiane, pas la moyenne : un pic passé ne relève pas le seuil).
Deux mesures comparables :
  - entraînement : CPU par mois de données (3 mois coûtent normalement 3 fois 1 mois,
    ce n'est pas un pic) ;
  - API servie : vCPU moyen sur un tour de la boucle de fond.
Pas assez d'historique -> pas de jugement. L'agent alerte seulement : il ne coupe rien.
"""
from __future__ import annotations

from statistics import median
from typing import Optional, Sequence

from .settings import CostConfig

# Causes à vérifier, données par le code (pas par un LLM) selon le genre de mesure
HINTS = {
    "training": "la période demandée et le nombre d'essais Optuna (RETRAIN_TUNING_TRIALS)",
    "serving": ("qui appelle l'API (trafic inhabituel, client en boucle) ; si cette charge est normale, "
                "la recommandation de taille de serveur"),
}


def spike(history: Sequence[float], value: float, config: CostConfig,
          floor: float = 0.0) -> Optional[dict]:
    """Renvoie le détail du pic, ou None. `history` : mesures passées, de la plus ancienne
    à la plus récente (la mesure jugée n'en fait pas partie)."""
    recent = [h for h in history if h >= 0][-config.spike_window:]
    if len(recent) < config.spike_min_history or value < floor:
        return None
    typical = median(recent)
    threshold = max(typical * config.spike_factor, floor)
    if typical <= 0 or value < threshold:
        return None
    return {"value": round(value, 4), "typical": round(typical, 4),
            "ratio": round(value / typical, 1), "factor": config.spike_factor,
            "compared_with": len(recent)}


def training_spike(history: Sequence[tuple[float, int]], cpu_seconds: float, months: Optional[int],
                   config: CostConfig) -> Optional[dict]:
    """Entraînement : CPU par mois de données comparé aux entraînements passés du modèle."""
    if not months or months <= 0:
        return None
    per_month = [c / m for c, m in history if m]
    found = spike(per_month, cpu_seconds / months, config)
    if found is None:
        return None
    extra = cpu_seconds - found["typical"] * months  # CPU en trop par rapport à d'habitude
    return {**found, "kind": "training", "unit": "s de CPU par mois de données",
            "extra_cpu_seconds": round(extra, 1),
            "extra_cost": round(extra * config.cost_per_cpu_second, 6),
            "hint": HINTS["training"]}


def serving_spike(history: Sequence[float], used_vcpus: float, config: CostConfig) -> Optional[dict]:
    """API servie : vCPU moyen du dernier tour comparé aux tours précédents."""
    found = spike(history, used_vcpus, config, floor=config.spike_min_vcpus)
    if found is None:
        return None
    return {**found, "kind": "serving", "unit": "vCPU", "hint": HINTS["serving"]}
