"""Premier démarrage : met un modèle en production s'il n'y en a pas encore.

Lancé par le service `init` de docker-compose.yml (une fois, puis il s'arrête) :
un coéquipier qui vient de cloner le dépôt n'a rien à lancer à la main.

    python -m pipeline.bootstrap        # ne fait rien si l'alias production existe déjà
"""
from __future__ import annotations

import time

import requests
from mlflow import MlflowClient

from common import config
from pipeline.train import has_production_model, train

FIRST_PERIOD = (2023, 1)  # mois du premier modèle (validé sur le mois suivant)


def wait_for_mlflow(timeout: float = 180) -> MlflowClient:
    deadline = time.monotonic() + timeout
    while True:
        try:
            requests.get(f"{config.MLFLOW_TRACKING_URI}/health", timeout=5).raise_for_status()
            return MlflowClient(config.MLFLOW_TRACKING_URI)
        except requests.RequestException:
            if time.monotonic() > deadline:
                raise SystemExit(f"MLflow injoignable sur {config.MLFLOW_TRACKING_URI}")
            time.sleep(3)


def reload_api(timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.post(f"{config.API_URL}/reload", timeout=60).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def main() -> None:
    client = wait_for_mlflow()
    if has_production_model(client):
        print("Un modèle est déjà en production : rien à faire.")
        return
    print("Aucun modèle en production : entraînement du premier modèle "
          f"({FIRST_PERIOD[0]}-{FIRST_PERIOD[1]:02d}, téléchargement des données la première fois)…")
    result = train(*FIRST_PERIOD, promote=True)
    print(f"Modèle v{result['version']} en production (RMSE {result['rmse']:.3f}).")
    print("API rechargée." if reload_api() else "API pas encore prête : elle chargera le modèle à son démarrage.")


if __name__ == "__main__":
    main()
