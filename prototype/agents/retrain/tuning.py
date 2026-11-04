"""Optimisation automatique des hyperparamètres du challenger avec Optuna.

On découpe les données d'entraînement en deux (80 % pour apprendre, 20 % pour
évaluer chaque essai). Le mois de test n'est jamais utilisé ici : il reste
réservé à la comparaison équitable avec le modèle en production.
"""
from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

from pipeline.data import TARGET, to_features
from pipeline.train import PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNED = ["max_depth", "learning_rate", "n_estimators", "min_child_weight", "reg_alpha", "reg_lambda"]


def suggest(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 20, 200),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
    }


def tune(df_train: pd.DataFrame, n_trials: int, seed: int = 42) -> tuple[dict, float]:
    """Cherche les meilleurs réglages XGBoost. Renvoie (réglages, durée en secondes)."""
    start = time.perf_counter()
    mask = np.random.default_rng(seed).random(len(df_train)) < 0.8
    fit_df, val_df = df_train[mask], df_train[~mask]
    x_fit, y_fit = to_features(fit_df), fit_df[TARGET].values
    x_val, y_val = to_features(val_df), val_df[TARGET].values

    def objective(trial: optuna.Trial) -> float:
        model = make_pipeline(DictVectorizer(), xgb.XGBRegressor(**{**PARAMS, **suggest(trial)}))
        model.fit(x_fit, y_fit)
        return root_mean_squared_error(y_val, model.predict(x_val))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: PARAMS[k] for k in TUNED})  # premier essai : les réglages actuels
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, time.perf_counter() - start