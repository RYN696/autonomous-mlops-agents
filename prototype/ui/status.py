"""État en direct de la pipeline : conteneurs, services, modèle servi, budget.

Chaque lecture échoue en silence (valeur "inconnu") : la page doit s'afficher
même si Docker, MLflow ou l'API sont en panne, c'est justement là qu'on la regarde.
Les résultats sont gardés quelques secondes pour ne pas interroger les services
à chaque rafraîchissement de la page.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests
from sqlalchemy import func, select, text

from common import config
from common.docker_project import compose_project

TIMEOUT = 2.0
# Modèles Ollama utilisés par les agents (Qualité : explication ; Fiabilité : décision et juge)
OLLAMA_URL = os.getenv("OLLAMA_URL", config.OLLAMA_URL)
NEEDED_MODELS = (config.LLM_AGENT_MODEL, config.LLM_JUDGE_MODEL)


def cached(seconds: float):
    def wrap(fn: Callable[[], Any]) -> Callable[[], Any]:
        state: dict[str, Any] = {"at": 0.0, "value": None}

        def inner() -> Any:
            if time.monotonic() - state["at"] > seconds:
                state["value"], state["at"] = fn(), time.monotonic()
            return state["value"]
        return inner
    return wrap


class PipelineStatus:
    def __init__(self, engine, docker_client=None, http=requests):
        self.engine = engine
        self._docker = docker_client
        self.http = http
        self.containers = cached(5)(self._containers)
        self.services = cached(5)(self._services)
        self.production = cached(30)(self._production)

    # --- conteneurs docker compose : service -> état ---
    def _containers(self) -> dict[str, dict]:
        try:
            if self._docker is None:
                import docker
                self._docker = docker.from_env()
            found = self._docker.containers.list(
                all=True, filters={"label": f"com.docker.compose.project={compose_project(self._docker)}"})
        except Exception:
            return {}
        out = {}
        for c in found:
            state = c.attrs.get("State", {})
            out[c.labels.get("com.docker.compose.service", c.name)] = {
                "status": c.status, "name": c.name, "started_at": state.get("StartedAt"),
                "restarts": c.attrs.get("RestartCount", 0)}
        return out

    # --- services : santé ---
    def _http_ok(self, url: str) -> bool:
        try:
            return self.http.get(url, timeout=TIMEOUT).status_code == 200
        except Exception:
            return False

    def _services(self) -> list[dict]:
        api: dict = {}
        try:
            api = self.http.get(f"{config.API_URL}/health", timeout=TIMEOUT).json()
        except Exception:
            pass
        try:
            import redis
            redis_ok = bool(redis.Redis.from_url(config.REDIS_URL, socket_timeout=TIMEOUT).ping())
        except Exception:
            redis_ok = False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("select 1"))
            db_ok = True
        except Exception:
            db_ok = False
        api_detail = (f"modèle v{api.get('model_version')}" + (" — mode dégradé" if api.get("degraded") else "")
                      if api else "injoignable")
        if api and api.get("model_version") is None:
            api_detail = "pas encore de modèle"
        return [
            {"name": "API de prédiction", "ok": api.get("status") == "ok", "detail": api_detail,
             "warn": bool(api.get("degraded"))},
            {"name": "MLflow", "ok": self._http_ok(f"{config.MLFLOW_TRACKING_URI}/health"), "detail": "registre"},
            {"name": "Redis", "ok": redis_ok, "detail": "bus d'événements"},
            {"name": "Postgres", "ok": db_ok, "detail": "journal d'actions"},
            self._ollama(),
        ]

    def _ollama(self) -> dict:
        """LLM local des agents : sans lui, Fiabilité suspend sa surveillance."""
        try:
            tags = self.http.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT).json()
        except Exception:
            return {"name": "Ollama", "ok": False, "warn": False,
                    "detail": "injoignable : lancer Ollama sur le PC", "fix": "ollama serve"}
        have = {m["name"] for m in tags.get("models", [])}
        missing = [m for m in NEEDED_MODELS if m not in have and f"{m}:latest" not in have]
        if missing:
            return {"name": "Ollama", "ok": True, "warn": True, "detail": "modèle(s) manquant(s)",
                    "fix": " ; ".join(f"ollama pull {m}" for m in missing)}
        return {"name": "Ollama", "ok": True, "warn": False, "detail": "LLM des agents"}

    # --- modèle en production (registre MLflow) ---
    def _production(self) -> dict | None:
        base = f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow"
        try:
            mv = self.http.get(f"{base}/registered-models/alias", timeout=TIMEOUT,
                               params={"name": config.MODEL_NAME, "alias": config.PRODUCTION_ALIAS}
                               ).json()["model_version"]
            run = self.http.get(f"{base}/runs/get", params={"run_id": mv["run_id"]}, timeout=TIMEOUT).json()
            metrics = {m["key"]: m["value"] for m in run["run"]["data"].get("metrics", [])}
        except Exception:
            return None
        return {"version": mv["version"], "rmse": metrics.get("rmse"),
                "created": datetime.fromtimestamp(int(mv["creation_timestamp"]) / 1000, timezone.utc)}

    # --- coûts du mois (table de l'agent Coûts) ---
    def month_cost(self) -> dict | None:
        from agents.cost.sql_store import model_costs
        from common.action_log import action_log
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        try:
            with self.engine.connect() as conn:
                rows = dict(conn.execute(select(model_costs.c.kind, func.sum(model_costs.c.cost))
                                         .where(model_costs.c.month == month).group_by(model_costs.c.kind)).all())
                last = conn.execute(select(action_log.c.details).where(action_log.c.action == "cost_estimated")
                                    .order_by(action_log.c.id.desc()).limit(1)).scalar()
        except Exception:
            return None
        # Budget de l'agent Coûts (il peut différer de celui de ce conteneur) : lu dans sa dernière estimation
        budget = (last or {}).get("budget", config.MONTHLY_BUDGET)
        total = float(sum(rows.values()))
        return {"month": month, "total": total, "training": float(rows.get("training", 0.0)),
                "serving": float(rows.get("serving", 0.0)), "budget": budget,
                "share": total / budget if budget else 0.0}
