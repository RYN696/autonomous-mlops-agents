"""Rapport de coûts du mois, expliqué par un LLM.

Les chiffres et les verdicts (budget tenu ou non, projection de fin de mois) sont
calculés ici, par du code, à partir de Postgres :
- model_costs : coût de chaque entraînement et de l'API servie ;
- action_log  : entraînements par demande (request_id), dépassements, recommandations.

Le LLM (Ollama) ne fait que les expliquer à l'équipe en quelques phrases. Son texte
est refusé s'il cite un nombre absent des constats ; on garde alors un texte écrit
par le code. Il ne décide rien : le budget reste une règle déterministe (AGENTS.md).
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from agents.cost.sql_store import model_costs
from common import config
from common.action_log import action_log
from common.llm import LLMError, OllamaClient

AGENT = "cost"

SYSTEM = (
    "Tu es l'agent Coûts d'une pipeline MLOps (prédiction de la durée des trajets de taxi à New York). "
    "Tu reçois des CONSTATS déjà calculés par le code. Explique-les à l'équipe en français, en 3 à 5 phrases "
    "courtes : combien a coûté le mois et où part l'argent (entraînement ou API), si le budget tient, "
    "et le conseil le plus utile. "
    "Cite chaque nombre exactement comme il est écrit dans les constats et n'utilise aucun autre nombre. "
    "Ne fais aucun calcul. Ne dis jamais que le budget est dépassé si les constats disent qu'il tient, "
    "et inversement. N'invente aucune cause qui n'est pas dans les constats."
)
SCHEMA = {"type": "object", "properties": {"explication": {"type": "string"}}, "required": ["explication"]}


def euros(x: float) -> str:
    return f"{x:.4f}"


def month_key(now: datetime) -> str:
    return now.strftime("%Y-%m")


def previous_month(month: str) -> str:
    year, mo = map(int, month.split("-"))
    return f"{year - 1}-12" if mo == 1 else f"{year}-{mo - 1:02d}"


def current_budget(conn) -> float:
    """Budget de l'agent Coûts : celui de sa dernière estimation (il peut différer de ce conteneur)."""
    last = conn.execute(select(action_log.c.details).where(action_log.c.agent == AGENT,
                                                           action_log.c.action == "cost_estimated")
                        .order_by(action_log.c.id.desc()).limit(1)).scalar()
    return float((last or {}).get("budget", config.MONTHLY_BUDGET))


def figures(engine, now: datetime | None = None) -> dict[str, Any]:
    """Tous les chiffres du rapport, calculés sans LLM."""
    now = now or datetime.now(timezone.utc)
    month = month_key(now)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        by_kind = dict(conn.execute(select(model_costs.c.kind, func.sum(model_costs.c.cost))
                                    .where(model_costs.c.month == month).group_by(model_costs.c.kind)).all())
        trainings, training_cpu = conn.execute(
            select(func.count(), func.coalesce(func.sum(model_costs.c.cpu_seconds), 0.0))
            .where(model_costs.c.month == month, model_costs.c.kind == "training")).one()
        previous = conn.execute(select(func.coalesce(func.sum(model_costs.c.cost), 0.0))
                                .where(model_costs.c.month == previous_month(month))).scalar_one()
        recorded = conn.execute(select(action_log.c.details).where(
            action_log.c.agent == AGENT, action_log.c.action == "cost_recorded",
            action_log.c.timestamp >= start)).scalars().all()
        exceeded = conn.execute(select(func.count()).where(
            action_log.c.agent == AGENT, action_log.c.action == "budget_exceeded",
            action_log.c.timestamp >= start)).scalar_one()
        recommendation = conn.execute(select(action_log.c.details).where(
            action_log.c.agent == AGENT, action_log.c.action == "cost_recommendation",
            action_log.c.timestamp >= start).order_by(action_log.c.id.desc()).limit(1)).scalar()
        budget = current_budget(conn)

    training, serving = float(by_kind.get("training", 0.0)), float(by_kind.get("serving", 0.0))
    total = training + serving
    days = calendar.monthrange(now.year, now.month)[1]
    elapsed = (now - start).total_seconds() / 86400
    # projection linéaire : seulement après une journée, sinon elle ne veut rien dire
    projection = total / elapsed * days if elapsed >= 1 else None
    top = sorted((d for d in recorded if (d or {}).get("kind") == "training"),
                 key=lambda d: d.get("cost") or 0.0, reverse=True)[:3]
    return {
        "month": month, "generated_at": now.isoformat(), "total": total, "training": training,
        "serving": serving, "budget": budget, "share": total / budget if budget else None,
        "remaining": budget - total, "trainings": int(trainings), "training_cpu_seconds": float(training_cpu),
        "avg_training": training / trainings if trainings else None, "previous_month": previous_month(month),
        "previous_total": float(previous), "days_elapsed": int(elapsed), "days_in_month": days,
        "projection": projection, "budget_exceeded_alerts": int(exceeded),
        "top_trainings": [{"request_id": d.get("request_id"), "cost": d.get("cost"),
                           "cpu_seconds": d.get("cpu_seconds")} for d in top],
        "recommendation": recommendation,
    }


def findings(f: dict[str, Any]) -> list[str]:
    """Constats tranchés par le code ; le LLM ne fait que les reformuler."""
    within = f["total"] <= f["budget"]
    lines = [
        f"Mois {f['month']} : {euros(f['total'])} € dépensés sur un budget de {euros(f['budget'])} € : "
        + ("le budget TIENT" if within else f"budget DÉPASSÉ de {euros(f['total'] - f['budget'])} €"),
        f"Entraînement : {euros(f['training'])} € pour {f['trainings']} entraînement(s), "
        f"{f['training_cpu_seconds']:.0f} s de CPU",
        f"API servie : {euros(f['serving'])} €",
        "Poste principal : " + ("l'entraînement" if f["training"] >= f["serving"] else "l'API servie"),
    ]
    if f["avg_training"] is not None:
        lines.append(f"Coût moyen d'un entraînement : {euros(f['avg_training'])} €")
    if f["projection"] is not None:
        over = f["projection"] > f["budget"]
        lines.append(f"Projection de fin de mois (rythme des {f['days_elapsed']} premiers jours) : "
                     f"{euros(f['projection'])} € : " + ("AU-DESSUS du budget" if over else "dans le budget"))
    lines.append(f"Mois précédent ({f['previous_month']}) : {euros(f['previous_total'])} €")
    if f["budget_exceeded_alerts"]:
        lines.append(f"Alertes de dépassement de budget ce mois : {f['budget_exceeded_alerts']} "
                     "(chaque réentraînement demande une validation humaine)")
    rec = f.get("recommendation") or {}
    if rec.get("action") == "scale_to_zero":
        lines.append("Conseil calculé : arrêter l'API quand elle ne sert pas (scale-to-zero)")
    elif rec.get("action"):
        lines.append(f"Conseil calculé : serveur {rec.get('current')} vers {rec.get('recommended')}")
    elif not within:
        lines.append("Conseil calculé : relever le budget ou espacer les réentraînements")
    return lines


def _numbers(text: str) -> set[float]:
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)       # 0,1775 écrit à la française
    text = re.sub(r"(\d)[  ](\d{3})\b", r"\1\2", text)  # 1 800
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def numbers_ok(text: str, lines: list[str]) -> bool:
    """Tous les nombres cités par le LLM doivent exister dans les constats."""
    return _numbers(text) <= _numbers(" ".join(lines))


def template_text(f: dict[str, Any]) -> str:
    """Explication écrite par le code (sans LLM, ou si le LLM s'est trompé)."""
    main = "l'entraînement" if f["training"] >= f["serving"] else "l'API servie"
    text = (f"En {f['month']}, la pipeline a coûté {euros(f['total'])} € ({euros(f['training'])} € d'entraînement "
            f"pour {f['trainings']} entraînement(s), {euros(f['serving'])} € d'API) ; le poste principal est {main}. ")
    if f["total"] <= f["budget"]:
        text += f"Le budget de {euros(f['budget'])} € tient."
    else:
        text += (f"Le budget de {euros(f['budget'])} € est dépassé de {euros(f['total'] - f['budget'])} € : "
                 "chaque réentraînement demande donc une validation humaine.")
    if f["projection"] is not None:
        text += f" Au rythme actuel, le mois finirait à {euros(f['projection'])} €."
    return text


def explain(f: dict[str, Any], client=None) -> dict[str, Any]:
    lines = findings(f)
    client = client or OllamaClient(config.COST_LLM_MODEL, timeout=config.COST_LLM_TIMEOUT_SECONDS)
    result: dict[str, Any] = {"findings": lines, "source": "template", "llm_model": client.model}
    try:
        answer = client.chat_json(SYSTEM, "CONSTATS :\n- " + "\n- ".join(lines), schema=SCHEMA)
        text = str(answer.get("explication") or "").strip()
    except LLMError as exc:
        result.update(explanation=template_text(f), llm_error=str(exc))
        return result
    if text and numbers_ok(text, lines):
        result.update(explanation=text, source="llm")
    else:  # nombre inventé ou réponse vide : on ne montre pas un texte faux
        result.update(explanation=template_text(f), rejected_text=text)
    return result


def make_report(engine, client=None, now: datetime | None = None) -> dict[str, Any]:
    f = figures(engine, now)
    return {**f, **explain(f, client)}


def headline(r: dict[str, Any]) -> str:
    """Message court pour l'humain (page web et Discord)."""
    share = f" ({r['share']:.0%} du budget)" if r.get("share") is not None else ""
    by = f"expliqué par {r['llm_model']}" if r.get("source") == "llm" else "texte du code : LLM indisponible ou refusé"
    return (f"📊 Rapport de coûts {r['month']} : {euros(r['total'])} € sur {euros(r['budget'])} €{share}.\n"
            f"> {r['explanation']}\n({by})")
