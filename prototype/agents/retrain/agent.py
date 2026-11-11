"""Agent Retrain : entraîne un challenger, le compare à la production, puis le promeut ou le bloque.

Écoute : retrain_requested  (payload : period ou months, requested_by, approved_by, force, data_file)
Publie : model_registered, puis model_promoted ou promotion_blocked
         (incident_opened si l'entraînement échoue)
Les versions ne sont jamais supprimées : l'ancienne production garde l'alias
"previous" pour permettre un rollback.

Sans "period", l'agent prend les "months" derniers mois présents dans data/ (1 par défaut).
"data_file" (fichier reçu dans data/, ex. incoming/...) remplace les mois de la période pour
l'entraînement ; le mois de test reste le mois réel suivant.
Une demande est ignorée si la même période a déjà été entraînée, ou si le
dernier entraînement date de moins de RETRAIN_COOLDOWN_MINUTES (sauf demande
validée par un humain). "force": true passe outre ces deux règles.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from pathlib import Path

import mlflow
import pandas as pd
from redis import event
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import root_mean_squared_error

from common import config
from agents.retrain.explain import explain_decision
from agents.retrain.notify import show_message
from agents.retrain.report import build_report
from agents.retrain.tuning import tune
from common.action_log import ActionLog
from common.events import Event, EventBus, EventType, Severity
from pipeline import train as training
from pipeline.data import DATA_DIR, TARGET, read_dataframe, read_file, to_features

AGENT = "agent_retrain"


def parse_period(period: str | dict) -> list[tuple[int, int]]:
    """"2023-02" ou {"start": "2023-01", "end": "2023-03"} -> [(2023, 1), (2023, 2), (2023, 3)]."""
    if isinstance(period, str):
        start = end = period
    else:
        start, end = period["start"], period.get("end") or period["start"]
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    if (year, month) > (end_year, end_month):
        raise ValueError(f"période invalide : {start} est après {end}")
    months = [(year, month)]
    while months[-1] != (end_year, end_month):
        months.append(training.next_month(*months[-1]))
    return months


def available_months() -> list[tuple[int, int]]:
    """Mois déjà présents dans prototype/data/, du plus ancien au plus récent."""
    months = []
    for path in DATA_DIR.glob("green_tripdata_*.parquet"):
        year, month = path.stem.split("_")[-1].split("-")
        months.append((int(year), int(month)))
    return sorted(months)


def default_months(available: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Le dernier mois disponible sert de test : on entraîne sur les n mois d'avant."""
    if len(available) < 2:
        raise ValueError("pas assez de mois dans data/ pour choisir une période")
    return available[max(0, len(available) - 1 - n):-1]


def reload_api() -> bool:
    """Demande à l'API de charger le nouveau modèle en production."""
    try:
        return requests.post(f"{config.API_URL}/reload", timeout=30).ok
    except requests.RequestException:
        return False


class RetrainAgent:
    def __init__(
        self,
        bus: EventBus | None = None,
        log: ActionLog | None = None,
        read_data: Callable[[int, int], pd.DataFrame] = read_dataframe,
        read_received: Callable[[str], pd.DataFrame] = read_file,
        reload: Callable[[], bool] = reload_api,
        list_months: Callable[[], list[tuple[int, int]]] = available_months,
        cooldown_minutes: float | None = None,
                tuning_trials: int | None = None,
        explain: Callable[[str, dict], tuple[str, str]] = explain_decision,
        show: Callable[[str], bool] = show_message,
    ):
        self.bus = bus or EventBus()
        self.log = log or ActionLog()
        self.read_data = read_data
        self.read_received = read_received
        self.reload = reload
        self.list_months = list_months
        self.cooldown_minutes = config.RETRAIN_COOLDOWN_MINUTES if cooldown_minutes is None else cooldown_minutes
        self.tuning_trials = config.RETRAIN_TUNING_TRIALS if tuning_trials is None else tuning_trials
        self.explain = explain
        self.show = show

    def run(self) -> None:
        print("Agent Retrain : en écoute de retrain_requested…")
        self.bus.listen(self.handle, types=[EventType.RETRAIN_REQUESTED])

    def handle(self, event: Event) -> dict[str, Any]:
        who = {
            "requested_by": event.payload.get("requested_by", event.source),
            "approved_by": event.payload.get("approved_by"),
        }
        if event.payload.get("request_id"):  # traçabilité : relie l'entraînement à la demande de l'orchestrateur
            who["request_id"] = event.payload["request_id"]
        try:
            if event.payload.get("period"):
                months = parse_period(event.payload["period"])
            else:
                n = int(event.payload.get("months") or config.DEFAULT_TRAIN_MONTHS)
                months = default_months(self.list_months(), n)
            period = [f"{y}-{m:02d}" for y, m in months]
            received = event.payload.get("data_file")
            train_period = f"{period[0]}..{period[-1]}" + (f" ({Path(received).name})" if received else "")

            reason = self.skip_reason(train_period, event.payload)
            if reason:
                skipped = {"action": "retrain_skipped", "reason": reason, "period": period, **who}
                self.log.record(AGENT, "retrain_skipped", event.model, skipped)
                self.publish(EventType.ACTION_LOGGED, Severity.INFO, skipped)
                self.report("skipped", skipped)
                return {"decision": "skipped", "reason": reason}

            started = {**who, "period": period}
            if received:
                started["data_file"] = received
            self.log.record(AGENT, "retrain_started", event.model, started)

            if received:  # fichier de nouvelles données (contrôlé par Qualité) à la place du mois téléchargé
                df_train = self.read_received(received)
            else:
                df_train = pd.concat([self.read_data(y, m) for y, m in months], ignore_index=True)
            df_test = self.read_data(*training.next_month(*months[-1]))
            # Optuna ne voit que df_train : le mois de test reste réservé à la comparaison
            params, tuning_seconds = tune(df_train, self.tuning_trials) if self.tuning_trials > 0 else ({}, 0.0)
            challenger = training.fit_and_register(
                df_train,
                df_test,
                tags={"role": "challenger", "train_period": train_period,
                      "tuning_trials": str(self.tuning_trials)},
                params=params,
            )
        except Exception as exc:  # l'agent doit continuer à écouter même si un entraînement échoue
            self.log.record(AGENT, "retrain_failed", event.model, {**who, "error": str(exc)})
            self.publish(EventType.INCIDENT_OPENED, Severity.CRITICAL, {**who, "error": str(exc)})
            self.report("failed", {**who, "error": str(exc), "period": locals().get("period")})
            return {"decision": "failed", "error": str(exc)}

        registered = {
            "version": challenger["version"],
            "rmse": challenger["rmse"],
            "training_seconds": challenger["training_seconds"],
            "tuning_seconds": tuning_seconds,
            "params": params,
            "period": period,
            **who,
        }
        self.log.record(AGENT, "challenger_registered", event.model, registered)
        self.publish(EventType.MODEL_REGISTERED, Severity.INFO, registered)

        result = self.compare_and_decide(challenger, df_test, {**who, "period": period, "params": params})
        test_y, test_m = training.next_month(*months[-1])
        self.report(result["decision"], {**result, "training_seconds": challenger["training_seconds"],
                                         "tuning_seconds": tuning_seconds, "tuning_trials": self.tuning_trials,
                                         "rows_train": len(df_train), "rows_test": len(df_test),
                                         "test_month": f"{test_y}-{test_m:02d}", "data_file": received})
        return result

    def compare_and_decide(self, challenger: dict, df_test: pd.DataFrame, who: dict) -> dict[str, Any]:
        client = MlflowClient(config.MLFLOW_TRACKING_URI)
        production = self.production_version(client)

        result = {
            "challenger_version": challenger["version"],
            "challenger_rmse": challenger["rmse"],
            "production_version": production,
            "production_rmse": None,
            "improvement": None,
            "min_improvement": config.MIN_IMPROVEMENT,
            **who,
        }
        if production is not None:
            # Même jeu de test pour les deux modèles
            model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{production}")
            production_rmse = root_mean_squared_error(df_test[TARGET].values, model.predict(to_features(df_test)))
            result["production_rmse"] = production_rmse
            result["improvement"] = (production_rmse - challenger["rmse"]) / production_rmse

        better = production is None or result["improvement"] > config.MIN_IMPROVEMENT
        if better:
            if production is not None:
                client.set_registered_model_alias(config.MODEL_NAME, config.PREVIOUS_ALIAS, production)
            client.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, challenger["version"])
            client.set_model_version_tag(config.MODEL_NAME, challenger["version"], "decision", "promoted")
            result["api_reloaded"] = self.reload()
            self.tell_humans("promoted", result)
            self.log.record(AGENT, "model_promoted", config.MODEL_NAME, result)
            self.publish(EventType.MODEL_PROMOTED, Severity.INFO, result)
            return {"decision": "promoted", **result}

        client.set_model_version_tag(config.MODEL_NAME, challenger["version"], "decision", "blocked")
        result["reason"] = f"le challenger ne réduit pas l'erreur d'au moins {config.MIN_IMPROVEMENT:.0%}"
        self.tell_humans("blocked", result)
        self.log.record(AGENT, "promotion_blocked", config.MODEL_NAME, result)
        self.publish(EventType.PROMOTION_BLOCKED, Severity.WARNING, result)
        return {"decision": "blocked", **result}
    def report(self, decision: str, facts: dict) -> None:
        """Rapport de fin de mission (écrit par le code) : un rapport raté ne doit pas casser l'agent."""
        try:
            self.log.record(AGENT, "retrain_report", config.MODEL_NAME, build_report(decision, facts))
        except Exception as exc:
            print(f"Rapport non écrit : {exc}")

    def tell_humans(self, decision: str, result: dict) -> None:
        """Ajoute au résultat un message lisible (LLM ou code), puis l'affiche."""
        result["message"], result["message_source"] = self.explain(decision, result)
        self.show(result["message"])
    def skip_reason(self, train_period: str, payload: dict) -> str | None:
        """Raison d'ignorer la demande (doublon ou pause), ou None pour entraîner."""
        if payload.get("force"):
            return None
        runs = self.previous_challengers()
        if runs.empty:
            return None
        if (runs["tags.train_period"] == train_period).any():
            return f"un challenger a déjà été entraîné sur {train_period}"
        if not payload.get("approved_by"):
            minutes = (datetime.now(timezone.utc) - runs["start_time"].iloc[0]).total_seconds() / 60
            if minutes < self.cooldown_minutes:
                return f"dernier entraînement il y a {minutes:.0f} min, pause de {self.cooldown_minutes:.0f} min"
        return None

    @staticmethod
    def previous_challengers() -> pd.DataFrame:
        """Entraînements déjà faits par l'agent, du plus récent au plus ancien (lus dans MLflow)."""
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        try:
            return mlflow.search_runs(
                experiment_names=[config.EXPERIMENT_NAME],
                filter_string="tags.role = 'challenger'",
                order_by=["start_time DESC"],
            )
        except MlflowException:
            return pd.DataFrame()

    @staticmethod
    def production_version(client: MlflowClient) -> str | None:
        try:
            return client.get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS).version
        except MlflowException:
            return None

    def publish(self, type_: EventType, severity: Severity, payload: dict) -> None:
        event = Event(type=type_, source=AGENT, severity=severity, payload=payload)
        self.bus.publish(event)
        print(f"[{severity.value}] {type_.value} {payload}")