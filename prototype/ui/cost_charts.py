"""Graphiques du rapport de coûts : séries calculées ici, dessinées en SVG par costs.html.

Trois graphiques, chacun avec un seul rôle :
  - cumul : dépense cumulée du mois jour par jour, avec la ligne du budget et la projection ;
  - jours : dépense de chaque jour, entraînement et API servie empilés ;
  - entrainements : coût de chaque entraînement du mois, les pics (cost_anomaly) marqués.
Les chiffres viennent des mêmes tables que le rapport (model_costs, action_log) : aucun LLM.
"""
from __future__ import annotations

import calendar
import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from agents.cost.sql_store import model_costs
from common.action_log import action_log

PAD = {"l": 52, "r": 16, "t": 16, "b": 28}
WIDE = (880, 260)    # dépense cumulée : toute la largeur
SMALL = (560, 260)    # les deux graphiques côte à côte


def _nice(top: float) -> float:
    """Borne haute « ronde » de l'axe (1, 2, 2.5, 5 × 10^n)."""
    if top <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(top))
    for m in (1, 2, 2.5, 5, 10):
        if top <= m * p:
            return m * p
    return 10 * p


def _ticks(top: float) -> list[float]:
    """Graduations rondes : 5 intervalles pour 2.5, 5, 10 (pas de 0.5, 1, 2), 4 pour 1 et 2."""
    mant = top / 10 ** math.floor(math.log10(top))
    n = 5 if round(mant, 6) in (2.5, 5.0, 10.0) else 4
    return [top * i / n for i in range(n + 1)]


class Frame:
    """Repère d'un graphique : largeur, hauteur, conversions valeur -> pixel."""

    def __init__(self, size: tuple[int, int]):
        self.w, self.h = size
        self.inner = self.w - PAD["l"] - PAD["r"]

    def x(self, i: float, n: int) -> float:
        return PAD["l"] + self.inner * (i / max(n, 1))

    def y(self, v: float, top: float) -> float:
        return PAD["t"] + (self.h - PAD["t"] - PAD["b"]) * (1 - v / top)

    def axis(self, top: float) -> dict:
        return {"w": self.w, "h": self.h, "top": top, "base": round(self.y(0, top), 1),
                "ticks": [{"v": t, "y": round(self.y(t, top), 1)} for t in _ticks(top)]}


def charts(engine, budget: float, projection: float | None, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    month = f"{now.year:04d}-{now.month:02d}"
    days = calendar.monthrange(now.year, now.month)[1]
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        rows = conn.execute(select(model_costs.c.ts, model_costs.c.kind, model_costs.c.cost)
                            .where(model_costs.c.month == month)).all()
        recorded = conn.execute(select(action_log.c.timestamp, action_log.c.details).where(
            action_log.c.agent == "cost", action_log.c.action == "cost_recorded",
            action_log.c.timestamp >= start).order_by(action_log.c.id)).all()
        spikes = conn.execute(select(action_log.c.details).where(
            action_log.c.agent == "cost", action_log.c.action == "cost_anomaly",
            action_log.c.timestamp >= start)).scalars().all()

    # --- dépense par jour, par poste ---
    per_day = {d: {"training": 0.0, "serving": 0.0} for d in range(1, days + 1)}
    for ts, kind, cost in rows:
        day = (ts.day if ts else now.day)
        per_day[day]["training" if kind == "training" else "serving"] += float(cost)
    today = now.day

    # 1. cumul + budget + projection
    cum, total, points = [], 0.0, []
    for d in range(1, today + 1):
        total += per_day[d]["training"] + per_day[d]["serving"]
        cum.append(total)
    f = Frame(WIDE)
    top = _nice(max([budget or 0, projection or 0, total, 0.01]) * 1.05)
    for i, v in enumerate(cum):
        points.append({"x": round(f.x(i + 1, days), 1), "y": round(f.y(v, top), 1), "day": i + 1, "value": v})
    line = " ".join(f"{p['x']},{p['y']}" for p in [{"x": f.x(0, days), "y": f.y(0, top)}] + points)
    cumul = {
        **f.axis(top), "points": points, "line": line, "budget": budget,
        "budget_y": round(f.y(budget, top), 1) if budget else None,
        "projection": None, "x_labels": [{"day": d, "x": round(f.x(d, days), 1)} for d in (1, 10, 20, days)],
    }
    if projection is not None and points:
        last = points[-1]
        cumul["projection"] = {"x1": last["x"], "y1": last["y"], "x2": round(f.x(days, days), 1),
                               "y2": round(f.y(projection, top), 1), "value": projection}

    # 2. barres empilées par jour
    f = Frame(SMALL)
    day_top = _nice(max([v["training"] + v["serving"] for v in per_day.values()] + [0.001]) * 1.05)
    slot = f.inner / days
    bars = []
    for d in range(1, days + 1):
        tr, sv = per_day[d]["training"], per_day[d]["serving"]
        x = PAD["l"] + slot * (d - 1) + slot * 0.18
        y_sv = f.y(sv, day_top)
        y_tr = f.y(sv + tr, day_top)
        bars.append({"day": d, "x": round(x, 1), "w": round(slot * 0.64, 1), "training": tr, "serving": sv,
                     "y_sv": round(y_sv, 1), "h_sv": round(f.y(0, day_top) - y_sv, 1),
                     "y_tr": round(y_tr, 1), "h_tr": round(max(y_sv - y_tr - (2 if sv and tr else 0), 0), 1),
                     "future": d > today})
    daily = {**f.axis(day_top), "bars": bars,
             "x_labels": [{"day": d, "x": round(PAD["l"] + slot * (d - 0.5), 1)} for d in (1, 10, 20, days)]}

    # 3. chaque entraînement, pics marqués
    # un pic se rattache à SON entraînement : même demande et même coût (une demande peut en compter plusieurs)
    def key(d: dict) -> tuple:
        return d.get("request_id"), round(float(d.get("cost") or 0), 6)
    spike_ratio = {key(s): s.get("ratio") for s in spikes if s and s.get("kind") == "training"}
    runs = [(ts, d) for ts, d in recorded if (d or {}).get("kind") == "training"]
    t_top = _nice(max([float(d.get("cost") or 0) for _, d in runs] + [0.001]) * 1.1)
    f = Frame(SMALL)
    n = max(len(runs), 1)
    slot = f.inner / n
    trainings = []
    for i, (ts, d) in enumerate(runs):
        cost = float(d.get("cost") or 0)
        y = f.y(cost, t_top)
        rid = d.get("request_id")
        trainings.append({"x": round(PAD["l"] + slot * i + slot * 0.15, 1), "w": round(max(slot * 0.7, 2), 1),
                          "y": round(y, 1), "h": round(f.y(0, t_top) - y, 1), "cost": cost,
                          "cpu": float(d.get("cpu_seconds") or 0), "request_id": rid, "ts": ts,
                          "spike": key(d) in spike_ratio, "ratio": spike_ratio.get(key(d))})
    per_training = {**f.axis(t_top), "bars": trainings, "spikes": len(spikes)}

    return {"pad": PAD, "cumul": cumul, "daily": daily, "trainings": per_training}
