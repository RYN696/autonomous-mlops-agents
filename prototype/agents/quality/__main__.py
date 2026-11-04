"""Point d'entrée de l'agent qualité.

Usage (depuis prototype/) :
    python -m agents.quality --year 2023 --month 2                        # cas normal
    python -m agents.quality --year 2023 --month 2 --scenario drift       # dérive fabriquée
    python -m agents.quality --year 2023 --month 2 --scenario corrupted   # données corrompues
    python -m agents.quality --year 2023 --month 3 --scenario suspect     # fichier reçu avec valeurs aberrantes
    ajouter --dry-run pour ne rien publier (ni Redis, ni Postgres)
    python -m agents.quality --year 2023 --month 2 --watch                # service (docker compose)
"""
from __future__ import annotations

import argparse
import uuid

import mlflow
import numpy as np
import pandas as pd

from agents.quality.checker import build_event, publish_quality_result
from agents.quality.drift import calculate_drift
from agents.quality.validation import validate_current_data
from common import config
from pipeline.data import DATA_DIR, clean as clean_rides, read_dataframe
from agents.quality.explain import add_explanation
from agents.quality.rootcause import analyze_shift


def load_raw(year: int, month: int) -> pd.DataFrame:
    """Parquet brut, non nettoyé (read_dataframe supprime les durées invalides)."""
    read_dataframe(year, month)  # garantit que le fichier est téléchargé
    return pd.read_parquet(DATA_DIR / f"green_tripdata_{year}-{month:02d}.parquet")


def apply_scenario(scenario: str, raw: pd.DataFrame, clean: pd.DataFrame):
    """Fabrique des données anormales pour la démonstration."""
    raw, clean = raw.copy(), clean.copy()
    if scenario == "drift":
        raw["trip_distance"] *= 1.5
        clean["trip_distance"] *= 1.5
    elif scenario == "corrupted":
        n = int(len(raw) * 0.10)  # 10 % de lignes corrompues
        raw.loc[raw.index[:n], "lpep_dropoff_datetime"] = (
            raw.loc[raw.index[:n], "lpep_pickup_datetime"] - pd.Timedelta(minutes=30)
        )
    elif scenario == "suspect":
        # Fichier reçu en pleine nuit : distances incohérentes sur 40 % des trajets, quelques distances
        # impossibles (9999) et durées négatives. Sous le seuil "corrompu" : Qualité voit surtout une forte dérive.
        rng = np.random.default_rng(7)
        idx = raw.index[rng.random(len(raw)) < 0.40]
        raw.loc[idx, "trip_distance"] = rng.uniform(0.3, 30, len(idx)).round(1)
        raw.loc[raw.index[rng.random(len(raw)) < 0.01], "trip_distance"] = 9999.0
        neg = raw.index[rng.random(len(raw)) < 0.015]
        raw.loc[neg, "lpep_dropoff_datetime"] = raw.loc[neg, "lpep_pickup_datetime"] - pd.Timedelta(minutes=5)
        clean = clean_rides(raw)
    return raw, clean


def save_received(raw: pd.DataFrame, year: int, month: int, check_id: str) -> str:
    """Le fichier reçu est gardé dans data/incoming/ : Retrain entraînera dessus, l'humain pourra le vérifier."""
    folder = DATA_DIR / "incoming"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"green_tripdata_{year}-{month:02d}_recu-{check_id}.parquet"
    raw.to_parquet(folder / name)
    return f"incoming/{name}"


def run_check(model, reference, clean, raw):
    validation = validate_current_data(raw)
    drift = calculate_drift(reference, clean, model)
    event = build_event(validation, drift["drift_score"], drift["num_drifted_columns"],
                        prediction_drift=drift["prediction_drift"])
    payload = dict(event.payload)
    payload["root_cause"] = analyze_shift(*drift["frames"])
    return event.model_copy(update={"payload": payload})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Agent qualité : validation + dérive.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--ref-year", type=int, default=2023)
    p.add_argument("--ref-month", type=int, default=1)
    p.add_argument("--scenario", choices=["normal", "drift", "corrupted", "suspect"], default="normal")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="désactive l'explication par LLM")
    p.add_argument("--check-id", help="identifiant du contrôle (donné par la page Scénarios de l'interface)")
    p.add_argument("--watch", action="store_true",
                   help="service : contrôle toutes les QUALITY_CHECK_MINUTES et à chaque model_promoted")
    args = p.parse_args(argv)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if args.watch:
        import logging

        from agents.quality.watch import watch
        from common.events import EventBus
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
        print(f"Agent Qualité : contrôle de {args.year}-{args.month:02d} toutes les "
              f"{config.QUALITY_CHECK_MINUTES:g} min et à chaque model_promoted")
        watch(lambda: check_once(args), EventBus(), config.QUALITY_CHECK_MINUTES * 60)
    else:
        check_once(args)


def check_once(args) -> None:
    # Rechargé à chaque contrôle : le modèle de production peut avoir changé
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.PRODUCTION_ALIAS}")

    reference = read_dataframe(args.ref_year, args.ref_month)
    clean = read_dataframe(args.year, args.month)
    raw = load_raw(args.year, args.month)
    raw, clean = apply_scenario(args.scenario, raw, clean)

    event = run_check(model, reference, clean, raw)
    # Mois contrôlé : l'orchestrateur le donne à Retrain comme période d'entraînement
    event.payload["period"] = f"{args.year}-{args.month:02d}"
    event.payload["check_id"] = args.check_id or uuid.uuid4().hex[:8]  # traçabilité : repris par l'orchestrateur
    if args.scenario == "suspect" and not args.dry_run:
        event.payload["data_file"] = save_received(raw, args.year, args.month, event.payload["check_id"])
    event = add_explanation(event, use_llm=not args.no_llm)

    if args.dry_run:
        print(event.model_dump_json(indent=2))
        return

    from common.action_log import ActionLog
    from common.events import EventBus
    publish_quality_result(event, EventBus(), ActionLog())
    print(f"Publié : {event.type.value} ({event.severity.value}) reason={event.payload.get('reason')} "
          f"check_id={event.payload['check_id']}")


if __name__ == "__main__":
    main()
