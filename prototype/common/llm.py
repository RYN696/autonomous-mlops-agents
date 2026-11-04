"""Client LLM commun (Ollama, 100 % local : gratuit, sans quota, hors ligne).

Un seul point d'appel au LLM pour tout le projet : on envoie une conversation,
on reçoit un objet JSON (dict), ou une LLMError."""
from __future__ import annotations

import json
from typing import Protocol

import requests

from common import config


THINKING_FAMILIES = ("qwen3", "deepseek-r1", "magistral", "phi4-reasoning")


class LLMError(Exception):
    """LLM injoignable, trop lent, ou réponse qui n'est pas un objet JSON."""


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict], schema: dict | None = None) -> dict: ...

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict: ...


class OllamaClient:
    def __init__(self, model: str, url: str | None = None, timeout: float | None = None,
                 keep_alive: str | None = None, session=None):
        self.model = model
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.timeout = config.LLM_TIMEOUT_SECONDS if timeout is None else timeout
        self.keep_alive = keep_alive or config.LLM_KEEP_ALIVE
        self.http = session or requests

    def chat(self, messages: list[dict], schema: dict | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            # schéma JSON (structured outputs) si fourni : contraint le décodage aux clés
            # attendues, pas seulement à du JSON valide (sinon un petit modèle peut recopier
            # le contexte reçu, qui est aussi du JSON valide).
            "format": schema if schema is not None else "json",
            "stream": False,
            "keep_alive": self.keep_alive,    # modèle gardé en mémoire GPU entre deux cycles
            "options": {"temperature": 0, "num_ctx": config.LLM_NUM_CTX},  # décisions reproductibles
        }
        if self.thinking_model() and config.LLM_THINK in ("off", "on"):
            payload["think"] = config.LLM_THINK == "on"   # qwen3 réfléchit longuement par défaut : trop lent
        try:
            resp = self.http.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            answer = json.loads(resp.json()["message"]["content"])
        except Exception as exc:
            raise LLMError(f"{self.model} : {type(exc).__name__}: {exc}") from exc
        if not isinstance(answer, dict):
            raise LLMError(f"{self.model} : réponse JSON inattendue ({type(answer).__name__})")
        return answer

    def thinking_model(self) -> bool:
        """Modèles à mode "réflexion" (on ne l'envoie pas aux autres : Ollama refuserait)."""
        return self.model.lower().startswith(THINKING_FAMILIES)

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], schema=schema)


def get_client(role: str) -> OllamaClient:
    """role = "agent" (décide) ou "judge" (contrôle, autre famille de modèle)."""
    return OllamaClient(config.LLM_JUDGE_MODEL if role == "judge" else config.LLM_AGENT_MODEL)
