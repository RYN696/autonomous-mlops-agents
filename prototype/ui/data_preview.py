"""Aperçu d'un fichier de données reçu (bouton « Voir les données » d'une alerte).

Seuls les fichiers de data/incoming/ (déposés par l'agent Qualité) sont consultables :
on montre combien de valeurs sont impossibles et quelques lignes, aberrantes en tête.
"""
from __future__ import annotations

import pandas as pd

from common import config
from pipeline.data import data_file

COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "trip_distance", "PULocationID", "DOLocationID"]


def preview(relative: str, sample: int = 10) -> dict:
    if not relative.startswith("incoming/"):
        raise ValueError("seuls les fichiers reçus (data/incoming/) sont consultables")
    df = pd.read_parquet(data_file(relative), columns=COLUMNS)
    duration = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    negative = duration < 0
    far = (df.trip_distance < 0) | (df.trip_distance > config.MAX_TRIP_DISTANCE)
    bad = negative | far
    picks = pd.concat([df[negative].head(3), df[far].head(3), df[~bad].head(sample - 6)]).sort_index()
    rows = [{"duration": round(duration[i], 1), "distance": round(float(df.trip_distance[i]), 1),
             "pu": df.PULocationID[i], "do": df.DOLocationID[i], "bad": bool(bad[i])} for i in picks.index]
    n = len(df)
    return {"name": relative.split("/")[-1], "rows": n, "negative": int(negative.sum()), "far": int(far.sum()),
            "share": float(bad.sum() / n) if n else 0.0, "max_distance": config.MAX_TRIP_DISTANCE, "sample": rows}
