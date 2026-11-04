"""Stockage des coûts avec SQLAlchemy (même pile que common/action_log.py)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, func, insert, select)

metadata = MetaData()

model_costs = Table(
    "model_costs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("model", String(128), nullable=False),
    Column("month", String(7), nullable=False),      # ex. '2026-10'
    Column("kind", String(16), nullable=False),      # 'training' ou 'serving'
    Column("cpu_seconds", Float, nullable=False),
    Column("months", Integer),                        # mois de données (entraînement)
    Column("cost", Float, nullable=False),
)


class SqlCostStore:
    """Implémente l'interface CostStore. `url` par défaut : config.DATABASE_URL."""

    def __init__(self, url: Optional[str] = None, engine=None) -> None:
        if engine is None:
            if url is None:
                from common import config  # noqa: PLC0415
                url = config.DATABASE_URL
            engine = create_engine(url)
        self.engine = engine
        metadata.create_all(self.engine)

    def add(self, model: str, month: str, kind: str, cpu_seconds: float,
            cost: float, months: Optional[int] = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(model_costs).values(
                ts=datetime.now(timezone.utc), model=model, month=month, kind=kind,
                cpu_seconds=cpu_seconds, months=months, cost=cost))

    def month_total(self, month: str) -> float:
        with self.engine.connect() as conn:
            total = conn.execute(
                select(func.coalesce(func.sum(model_costs.c.cost), 0.0))
                .where(model_costs.c.month == month)).scalar_one()
        return float(total)

    def by_model(self, month: str) -> Dict[str, float]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(model_costs.c.model, func.sum(model_costs.c.cost))
                .where(model_costs.c.month == month)
                .group_by(model_costs.c.model))
            return {m: float(c) for m, c in rows}

    def training_history(self, model: str) -> List[Tuple[float, int]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(model_costs.c.cpu_seconds, model_costs.c.months)
                .where(model_costs.c.model == model,
                       model_costs.c.kind == "training",
                       model_costs.c.months.is_not(None),
                       model_costs.c.months > 0))
            return [(float(c), int(m)) for c, m in rows]
