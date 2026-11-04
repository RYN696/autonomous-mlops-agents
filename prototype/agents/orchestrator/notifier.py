"""Messages pour l'humain, affichés sur la page web de l'orchestrateur (et sur Discord si configuré).

Ils sont rangés dans la table notifications de Postgres :
- kind="info"     : simple message (modèle promu, incident...) ;
- kind="question" : demande de validation, status pending -> approved / refused.

Chaque message peut aussi partir vers un relais (le bot Discord, voir discord_bot.py) avec :
- level   : info, success, warning, alert (salon des alertes) ou question ;
- actions : boutons proposés, ex. "report", "dashboard", "data", "ignore", "rollback:<cible>:<actuelle>"."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, Text,
                        create_engine, insert, select, update)

from common import config

metadata = MetaData()

notifications = Table(
    "notifications",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("text", Text, nullable=False),
    Column("request_id", String(64)),
    Column("status", String(16)),
    Column("answered_by", String(64)),
)


LEVELS = ("info", "success", "warning", "alert", "question")


@dataclass(frozen=True)
class Message:
    text: str
    key: str | None = None        # parcours : request_id ou incident_id
    level: str = "info"
    actions: tuple[str, ...] = ()


class Relay(Protocol):
    def post(self, message: Message) -> None: ...


class Notifier:
    def __init__(self, url: str = config.DATABASE_URL, relay: Relay | None = None):
        self.engine = create_engine(url)
        metadata.create_all(self.engine)
        self.relay = relay

    def _relay(self, message: Message) -> None:
        if self.relay is None:
            return
        try:
            self.relay.post(message)
        except Exception as exc:  # Discord en panne ne doit jamais bloquer un agent
            print(f"[notification] relais en échec : {type(exc).__name__}: {exc}")

    def _insert(self, **values) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(notifications).values(timestamp=datetime.now(timezone.utc), **values))

    def notify(self, text: str, request_id: str | None = None, level: str = "info",
               actions: tuple[str, ...] = ()) -> None:
        """request_id : parcours du message (demande de réentraînement ou incident), pour la traçabilité."""
        print(f"[notification] {text}")
        self._insert(kind="info", text=text, request_id=request_id)
        self._relay(Message(text, request_id, level, tuple(actions)))

    def ask(self, text: str, request_id: str) -> None:
        """Question avec les boutons Valider / Refuser (page web et Discord)."""
        print(f"[question {request_id}] {text}")
        self._insert(kind="question", text=text, request_id=request_id, status="pending")
        self._relay(Message(text, request_id, "question", ("approve", "refuse", "report")))

    def answer(self, request_id: str, approved: bool, who: str) -> bool:
        """Enregistre la réponse. False si la question n'existe pas ou a déjà une réponse."""
        with self.engine.begin() as conn:
            result = conn.execute(
                update(notifications)
                .where(notifications.c.request_id == request_id, notifications.c.status == "pending")
                .values(status="approved" if approved else "refused", answered_by=who))
        return result.rowcount == 1

    def latest(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(notifications).order_by(notifications.c.id.desc()).limit(limit))
            return [dict(r._mapping) for r in rows]

    def for_request(self, request_id: str) -> list[dict[str, Any]]:
        """Messages et questions d'une demande, du plus ancien au plus récent."""
        with self.engine.connect() as conn:
            rows = conn.execute(select(notifications).where(notifications.c.request_id == request_id)
                                .order_by(notifications.c.id))
            return [dict(r._mapping) for r in rows]

    def expire_pending(self) -> list[str]:
        """Questions encore ouvertes -> "expired" ; renvoie leurs request_id."""
        with self.engine.begin() as conn:
            ids = [r[0] for r in conn.execute(select(notifications.c.request_id)
                                              .where(notifications.c.status == "pending"))]
            conn.execute(update(notifications).where(notifications.c.status == "pending").values(status="expired"))
        return ids

    def pending(self) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(notifications).where(notifications.c.status == "pending")
                                .order_by(notifications.c.id.desc()))
            return [dict(r._mapping) for r in rows]