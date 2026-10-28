from __future__ import annotations

import pandas as pd


def _median_change(ref: pd.Series, cur: pd.Series) -> dict:
    """Médiane avant / après et variation en %."""
    r, c = float(ref.median()), float(cur.median())
    pct = int(round((c - r) / r * 100)) if r else None
    return {"ref": round(r, 1), "cur": round(c, 1), "change_pct": pct}


def _top_zone_shifts(ref: pd.Series, cur: pd.Series, k: int = 3) -> list[dict]:
    """Les k zones dont la part de trajets a le plus changé."""
    rs = ref.value_counts(normalize=True)
    cs = cur.value_counts(normalize=True)
    zones = rs.index.union(cs.index)
    diff = cs.reindex(zones, fill_value=0) - rs.reindex(zones, fill_value=0)
    order = diff.abs().sort_values(ascending=False).index[:k]
    return [
        {
            "zone": str(z),
            "ref_share_pct": round(float(rs.get(z, 0)) * 100, 1),
            "cur_share_pct": round(float(cs.get(z, 0)) * 100, 1),
        }
        for z in order
    ]


def analyze_shift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Compare deux frames issus de drift.build_frame
    (colonnes : prediction, trip_distance, PU, DO)."""
    return {
        "prediction_median_min": _median_change(ref["prediction"], cur["prediction"]),
        "distance_median_miles": _median_change(ref["trip_distance"], cur["trip_distance"]),
        "top_pickup_zone_shifts": _top_zone_shifts(ref["PU"], cur["PU"]),
        "top_dropoff_zone_shifts": _top_zone_shifts(ref["DO"], cur["DO"]),
    }