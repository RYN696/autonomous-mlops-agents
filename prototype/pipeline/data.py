"""Chargement des données NYC green taxi, avec un cache local dans prototype/data/."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"

CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance"]
TARGET = "duration"


def download(url: str, path: Path, attempts: int = 5, reader: Callable[[str], pd.DataFrame] = pd.read_parquet,
             sleep: Callable[[float], None] = time.sleep) -> None:
    """Téléchargement avec nouvelles tentatives : le DNS de Docker Desktop échoue parfois
    quelques secondes (vu au premier lancement d'un clone). Fichier écrit en une fois."""
    for attempt in range(1, attempts + 1):
        try:
            df = reader(url)
            break
        except OSError as exc:  # URLError, erreurs réseau
            if attempt == attempts:
                raise
            wait = 3 * 2 ** (attempt - 1)
            print(f"Téléchargement impossible ({exc}) : nouvel essai dans {wait} s ({attempt}/{attempts})")
            sleep(wait)
    tmp = path.with_suffix(".part")
    df.to_parquet(tmp)
    tmp.replace(path)  # pas de fichier à moitié écrit si le conteneur s'arrête


def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Lit un mois de données (téléchargé une seule fois, puis lu depuis le disque)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"green_tripdata_{year}-{month:02d}.parquet"
    if not path.exists():
        print(f"Téléchargement de {path.name}…")
        download(URL.format(year=year, month=month), path)
    df = pd.read_parquet(path)

    df[TARGET] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df


def to_features(df: pd.DataFrame) -> list[dict]:
    """Même transformation à l'entraînement et dans l'API."""
    records = df[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    return [prepare_features(r) for r in records]


def prepare_features(ride: dict) -> dict:
    return {
        "PU_DO": f"{ride['PULocationID']}_{ride['DOLocationID']}",
        "trip_distance": float(ride["trip_distance"]),
    }