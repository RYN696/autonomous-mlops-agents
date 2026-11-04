"""Scénarios de test lancés depuis l'interface (page /scenarios), sans terminal.

- Données (agent Qualité) : un contrôle avec dérive fabriquée, données corrompues ou
  normal, exécuté DANS le conteneur quality (docker exec), comme
  `docker compose exec quality python -m agents.quality ...`.
- Pannes (agent Fiabilité) : les fonctions de scripts/inject_failure.py.
- Trafic : des utilisateurs simulés envoient des prédictions à l'API.

Chaque lancement tourne dans un thread, est gardé dans l'historique de la page et
écrit dans le journal d'actions (agent "demo") pour la traçabilité.
Désactivé si DEMO_CONTROLS n'est pas à 1 (c'est un banc de test, pas un outil de prod).
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from common import config
from common.docker_project import compose_project

ENABLED = os.getenv("DEMO_CONTROLS", "0") == "1"
AGENT = "demo"


@dataclass(frozen=True)
class Scenario:
    key: str
    group: str
    title: str
    description: str
    expected: str
    target: str            # agent qui doit réagir (clé de ui.catalog)
    with_month: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("drift", "Données", "Dérive des données",
             "L'agent Qualité contrôle le mois choisi avec des distances de trajet gonflées de 50 %.",
             "Orchestrateur → coût → validation humaine (budget de démo) → Retrain optimise (Optuna) puis "
             "entraîne sur ce mois, en quelques minutes → promotion ou promotion bloquée, expliquée.",
             "quality", with_month=True),
    Scenario("corrupted", "Données", "Données corrompues",
             "10 % des trajets du mois ont une arrivée avant le départ.",
             "Incident « données corrompues », aucun réentraînement.", "quality", with_month=True),
    Scenario("normal", "Données", "Contrôle normal",
             "Contrôle du mois sans rien modifier.",
             "quality_ok : rien ne se passe, c'est le cas nominal.", "quality", with_month=True),
    Scenario("crash", "Pannes", "Crash de l'API",
             "Arrête le conteneur de l'API.",
             "Incident api_arretee → Fiabilité redémarre l'API → incident clos.", "reliability"),
    Scenario("errors", "Pannes", "Erreurs 500",
             "30 % des prédictions renvoient une erreur 500.",
             "Incident → le LLM enquête, le juge valide → redémarrage → incident clos.", "reliability"),
    Scenario("latency", "Pannes", "Latence",
             "Chaque prédiction prend 2 s de plus.",
             "Incident surcharge → mode dégradé → humain prévenu ; « Remettre en état » pour clore.",
             "reliability"),
    Scenario("bad-model", "Pannes", "Mauvais modèle en production",
             "Promeut un modèle qui plante à chaque prédiction.",
             "Règle déterministe : rollback vers la version précédente → incident clos.", "reliability"),
    Scenario("mlflow-down", "Pannes", "MLflow en panne",
             "Arrête MLflow puis redémarre l'API, qui ne peut plus charger son modèle.",
             "Mode dégradé, pas de redémarrage ni de rollback → humain prévenu ; « Remettre en état » pour clore.",
             "reliability"),
    Scenario("recover", "Remise en état", "Remettre en état",
             "Relance MLflow et l'API si besoin, et retire les pannes injectées (le geste de l'humain).",
             "Les incidents en attente se ferment au cycle suivant de Fiabilité.", "reliability"),
)
BY_KEY = {s.key: s for s in SCENARIOS}


@dataclass
class Run:
    id: str
    scenario: Scenario
    params: dict[str, Any]
    started: datetime
    status: str = "running"  # running, ok, error
    output: str = ""
    finished: datetime | None = None
    check_id: str | None = None  # contrôle Qualité lancé (pour retrouver le parcours)


@dataclass
class Traffic:
    running: bool = False
    rps: float = 5.0
    ok: int = 0
    errors: int = 0
    fallback: int = 0
    since: datetime | None = None
    stop: threading.Event = field(default_factory=threading.Event)


class ScenarioRunner:
    def __init__(self, log=None, docker_client=None, enabled: bool = ENABLED,
                 actions: dict[str, Callable[[Run], str]] | None = None):
        self.log = log
        self._docker = docker_client
        self.enabled = enabled
        self.runs: list[Run] = []
        self.traffic = Traffic()
        self.lock = threading.Lock()
        self.actions = actions or {
            "drift": self._quality, "corrupted": self._quality, "normal": self._quality,
            "crash": self._inject("crash"), "errors": self._inject("errors"),
            "latency": self._inject("latency"), "bad-model": self._inject("bad_model"),
            "mlflow-down": self._inject("mlflow_down"), "recover": self._inject("recover"),
        }

    # ------------------------------------------------ lancement
    def busy(self, key: str) -> bool:
        return any(r.scenario.key == key and r.status == "running" for r in self.runs)

    def start(self, key: str, params: dict[str, Any] | None = None, wait: bool = False) -> Run:
        if not self.enabled:
            raise PermissionError("scénarios désactivés (DEMO_CONTROLS=1 pour les activer)")
        scenario = BY_KEY[key]
        params = params or {}
        if scenario.with_month:
            params["month"] = max(1, min(12, int(params.get("month") or 2)))
        with self.lock:
            if self.busy(key):
                raise RuntimeError("ce scénario est déjà en cours")
            run = Run(uuid.uuid4().hex[:8], scenario, params, datetime.now(timezone.utc))
            if scenario.target == "quality":
                run.check_id = uuid.uuid4().hex[:8]
            self.runs.insert(0, run)
            del self.runs[30:]
        self._record("scenario_started", run)
        thread = threading.Thread(target=self._execute, args=(run,), daemon=True)
        thread.start()
        if wait:
            thread.join()
        return run

    def _execute(self, run: Run) -> None:
        try:
            run.output = self.actions[run.scenario.key](run) or "terminé"
            run.status = "ok"
        except BaseException as exc:  # SystemExit d'inject_failure compris
            run.output = (run.output + "\n" if run.output else "") + f"{type(exc).__name__}: {exc}"
            run.status = "error"
        run.finished = datetime.now(timezone.utc)
        self._record("scenario_finished", run)

    def _record(self, action: str, run: Run) -> None:
        if self.log is None:
            return
        details = {"run_id": run.id, "scenario": run.scenario.key, "title": run.scenario.title,
                   **run.params}
        if run.check_id:
            details["check_id"] = run.check_id
        if action == "scenario_finished":
            details.update(status=run.status, output=run.output[-1500:],
                           duration_seconds=round((run.finished - run.started).total_seconds(), 1))
        try:
            self.log.record(AGENT, action, config.MODEL_NAME, details)
        except Exception:
            pass

    # ------------------------------------------------ implémentations
    @property
    def docker(self):
        if self._docker is None:
            import docker
            self._docker = docker.from_env()
        return self._docker

    def _quality(self, run: Run) -> str:
        """Contrôle Qualité dans son propre conteneur (données, modèle et LLM y sont configurés)."""
        found = self.docker.containers.list(filters={
            "label": [f"com.docker.compose.project={compose_project(self.docker)}",
                      "com.docker.compose.service=quality"]})
        if not found:
            raise RuntimeError("conteneur quality introuvable : lancer `docker compose up -d`")
        cmd = ["python", "-m", "agents.quality", "--year", "2023", "--month", str(run.params["month"]),
               "--scenario", run.scenario.key, "--check-id", run.check_id]
        result = found[0].exec_run(cmd, environment={"PYTHONIOENCODING": "utf-8"})
        out = result.output.decode("utf-8", errors="replace") if result.output else ""
        lines = [l for l in out.splitlines() if l.strip() and "Downloading artifacts" not in l]
        run.output = "\n".join(lines[-8:])
        if result.exit_code != 0:
            raise RuntimeError(f"le contrôle a échoué (code {result.exit_code})")
        return run.output

    @staticmethod
    def _inject(name: str) -> Callable[[Run], str]:
        def action(run: Run) -> str:
            from scripts import inject_failure
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn = getattr(inject_failure, name)
                if name == "errors":
                    fn(0.3)
                elif name == "latency":
                    fn(2.0)
                else:
                    fn()
            run.output = buf.getvalue().strip()
            return run.output
        return action

    # ------------------------------------------------ trafic simulé
    def start_traffic(self, rps: float = 5.0) -> None:
        if not self.enabled:
            raise PermissionError("scénarios désactivés (DEMO_CONTROLS=1 pour les activer)")
        if self.traffic.running:
            return
        self.traffic = Traffic(running=True, rps=max(0.5, min(20.0, rps)), since=datetime.now(timezone.utc))
        self._record_traffic("traffic_started")
        threading.Thread(target=self._traffic_loop, args=(self.traffic,), daemon=True).start()

    def stop_traffic(self) -> None:
        if self.traffic.running:
            self.traffic.stop.set()
            self.traffic.running = False
            self._record_traffic("traffic_stopped")

    def _record_traffic(self, action: str) -> None:
        if self.log is not None:
            t = self.traffic
            with contextlib.suppress(Exception):
                self.log.record(AGENT, action, config.MODEL_NAME,
                                {"rps": t.rps, "ok": t.ok, "errors": t.errors, "fallback": t.fallback})

    @staticmethod
    def _traffic_loop(t: Traffic) -> None:
        while not t.stop.is_set():
            ride = {"PULocationID": random.randint(1, 265), "DOLocationID": random.randint(1, 265),
                    "trip_distance": round(random.uniform(0.5, 15), 1)}
            try:
                resp = requests.post(f"{config.API_URL}/predict", json=ride, timeout=config.HTTP_TIMEOUT_SECONDS)
                if resp.status_code >= 500:
                    t.errors += 1
                else:
                    t.ok += 1
                    t.fallback += bool(resp.json().get("degraded", False))
            except requests.RequestException:
                t.errors += 1
            t.stop.wait(1 / t.rps)
