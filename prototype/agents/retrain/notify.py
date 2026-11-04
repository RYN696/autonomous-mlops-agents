"""Affiche le message de l'agent Retrain (pas d'envoi externe, pas de Telegram).

Le message est visible :
- dans la console (`docker compose logs retrain`) ;
- dans l'événement model_promoted ou promotion_blocked (champ "message") ;
- dans le journal d'actions (Postgres, donc Grafana).
"""
from __future__ import annotations


def show_message(text: str) -> bool:
    line = "─" * 60
    print(f"\n{line}\n{text}\n{line}\n", flush=True)
    return True