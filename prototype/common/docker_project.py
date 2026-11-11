"""Nom du projet docker compose dont font partie les conteneurs de la pipeline.

Il dépend du dossier où chacun a cloné le dépôt (ou de `-p`) : on ne l'écrit pas en
dur. Ordre : variable COMPOSE_PROJECT_NAME, sinon l'étiquette compose du conteneur
courant (un agent en conteneur se reconnaît par son hostname), sinon "prototype".
"""
from __future__ import annotations

import os
import socket
from functools import lru_cache

DEFAULT = "prototype"
LABEL = "com.docker.compose.project"


@lru_cache(maxsize=1)
def compose_project(client=None) -> str:
    if os.getenv("COMPOSE_PROJECT_NAME"):
        return os.environ["COMPOSE_PROJECT_NAME"]
    try:
        if client is None:
            import docker
            client = docker.from_env()
        return client.containers.get(socket.gethostname()).labels.get(LABEL, DEFAULT)
    except Exception:  # hors conteneur, ou pas d'accès à Docker
        return DEFAULT
