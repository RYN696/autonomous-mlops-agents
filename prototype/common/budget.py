"""Budget du mois, réglable en direct par un responsable (page Scénarios de l'interface).

Il est gardé dans Redis (clé KEY) : l'agent Coûts le relit à chaque estimation, l'interface
l'affiche. Sans valeur dans Redis (ou Redis injoignable), chacun garde son budget de
configuration (MONTHLY_BUDGET). Chaque changement est écrit au journal par l'interface.
"""
from __future__ import annotations

import redis

from common import config

KEY = "mlops:monthly_budget"


def _client(client=None):
    return client or redis.Redis.from_url(config.REDIS_URL, socket_timeout=2, socket_connect_timeout=2)


def current(client=None) -> float | None:
    """Budget réglé en direct, ou None s'il n'y en a pas (ou si Redis ne répond pas)."""
    try:
        raw = _client(client).get(KEY)
        return float(raw) if raw is not None else None
    except (redis.RedisError, ValueError):
        return None


def get(default: float, client=None) -> float:
    value = current(client)
    return default if value is None else value


def set_budget(value: float, client=None) -> float:
    value = round(max(0.0, float(value)), 6)
    _client(client).set(KEY, value)
    return value


def clear(client=None) -> None:
    _client(client).delete(KEY)
