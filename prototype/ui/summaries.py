"""Une phrase lisible par ligne du journal d'actions (le JSON complet reste affiché à côté)."""
from __future__ import annotations

from typing import Any, Callable

RULES = {"auto": "automatique", "approval": "validation humaine", "forbidden": "interdit"}


def _pct(x: Any) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "?"


def _num(x: Any, fmt: str = ".2f") -> str:
    return format(x, fmt) if isinstance(x, (int, float)) else "?"


def _period(d: dict) -> str:
    p = d.get("period")
    if isinstance(p, list):
        return p[0] if len(p) == 1 else f"{p[0]} → {p[-1]}"
    return str(p) if p else ""


def _with_period(text: str, d: dict) -> str:
    return f"{text} ({_period(d)})" if _period(d) else text


def _drift(d: dict) -> str:
    if d.get("reason") == "corrupted_input_data":
        return f"Données corrompues : {_pct(d.get('invalid_share'))} de lignes invalides"
    return _with_period(f"Dérive : score {_num(d.get('drift_score'))} (seuil {d.get('drift_threshold')}), "
                        f"dérive des prédictions {_num(d.get('prediction_drift'))}", d)


def _cost_estimated(d: dict) -> str:
    opts = d.get("options") or {}
    std, spot = (opts.get("standard") or {}).get("cost"), (opts.get("spot") or {}).get("cost")
    over = " — dépasse le budget" if d.get("exceeds_budget") else ""
    return (f"Estimation : {_num(std, '.4f')} € (spot {_num(spot, '.4f')} €), mois {_num(d.get('month_total'), '.4f')} € "
            f"sur {d.get('budget')} €{over}")


def _cost_recorded(d: dict) -> str:
    what = "Entraînement facturé" if d.get("kind") == "training" else "API servie"
    return f"{what} : {_num(d.get('cpu_seconds'), '.1f')} s de CPU = {_num(d.get('cost'), '.4f')} €"


def _recommendation(d: dict) -> str:
    if d.get("action") == "scale_to_zero":
        return (f"Scale-to-zero conseillé : API inactive depuis {_num(d.get('idle_hours'), '.1f')} h, "
                f"économie {_num(d.get('estimated_saving'), '.4f')} €")
    return (f"Taille du serveur : {d.get('action')} {d.get('current')} → {d.get('recommended')} "
            f"({_num(d.get('monthly_saving'), '.2f')} €/mois)")


def _promoted(d: dict) -> str:
    if d.get("production_rmse") is None:
        return f"Premier modèle v{d.get('challenger_version')} en production (RMSE {_num(d.get('challenger_rmse'), '.3f')})"
    return (f"v{d.get('challenger_version')} promu : RMSE {_num(d.get('challenger_rmse'), '.3f')} contre "
            f"{_num(d.get('production_rmse'), '.3f')} (gain {_pct(d.get('improvement'))})")


def _blocked(d: dict) -> str:
    return (f"Promotion bloquée : v{d.get('challenger_version')} RMSE {_num(d.get('challenger_rmse'), '.3f')} contre "
            f"{_num(d.get('production_rmse'), '.3f')} pour v{d.get('production_version')} "
            f"(gain {_pct(d.get('improvement'))}, minimum {_pct(d.get('min_improvement'))})")


def _llm(d: dict) -> str:
    r = d.get("reponse") or {}
    phase = d.get("phase", "?")
    choice = r.get("choix") or r.get("verdict") or (r.get("cause") if r.get("incident") else None)
    head = f"LLM, {phase}" + (f" → {choice}" if choice else "")
    return f"{head} : {r.get('pensee') or r.get('resume') or ''}".strip(" :")


def _action(d: dict) -> str:
    ok = "réussie" if d.get("success") else "échec"
    by = f", décidée par {d['by']}" if d.get("by") else ""
    return f"Action {d.get('action')} {ok}{by} : {d.get('detail', '')}"


def _resolved(d: dict) -> str:
    judge = d.get("judge") or {}
    note = f", note du juge {judge['score']}/10" if judge.get("score") is not None else ""
    return f"Incident clos en {_num(d.get('duration_seconds'), '.0f')} s{note} : {d.get('resume', '')}"


SUMMARIES: dict[str, Callable[[dict], str]] = {
    # Qualité
    "drift_detected": _drift,
    "quality_ok": lambda d: _with_period(f"Données conformes ({d.get('row_count')} lignes, "
                                         f"dérive {_num(d.get('drift_score'))})", d),
    # Banc de test (page Scénarios)
    "scenario_started": lambda d: (f"Scénario lancé : {d.get('title')}"
                                   + (f" (2023-{int(d['month']):02d})" if d.get("month") else "")),
    "scenario_finished": lambda d: (f"Scénario {d.get('title')} : {'terminé' if d.get('status') == 'ok' else 'en échec'} "
                                    f"en {_num(d.get('duration_seconds'), '.0f')} s"),
    "traffic_started": lambda d: f"Trafic simulé démarré ({d.get('rps')} requêtes/s)",
    "traffic_stopped": lambda d: (f"Trafic simulé arrêté : {d.get('ok')} réponses, {d.get('errors')} erreurs, "
                                  f"{d.get('fallback')} par le modèle de secours"),
    # Orchestrateur
    "drift_received": lambda d: _with_period(f"Dérive reçue de l'agent Qualité ({d.get('reason')})", d),
    "drift_ignored": lambda d: f"Dérive mise de côté : {d.get('reason')}",
    "retrain_refused_corrupted_data": lambda d: f"Réentraînement refusé : données corrompues "
                                                f"({_pct(d.get('invalid_share'))} de lignes invalides)",
    "cost_requested": lambda d: "Coût demandé à l'agent Coûts",
    "retrain_decision": lambda d: (f"Règle « {d.get('action')} » : {RULES.get(d.get('rule'), d.get('rule'))} "
                                   f"(coût estimé {_num(d.get('estimated_cost'), '.4f')} €)"),
    "retrain_requested": lambda d: _with_period("Réentraînement demandé à l'agent Retrain"
                                                + (f", validé par {d['approved_by']}" if d.get("approved_by") else ""), d),
    "retrain_refused": lambda d: f"Réentraînement refusé par {d.get('refused_by')}",
    "request_expired": lambda d: f"Demande expirée : {d.get('reason')}",
    "request_failed": lambda d: "La demande a échoué (voir les journaux de l'orchestrateur)",
    # Coûts
    "cost_estimated": _cost_estimated,
    "budget_exceeded": lambda d: (f"Budget dépassé de {_num(d.get('over_by'), '.4f')} € — "
                                  f"suggestion : {d.get('suggested_action')}"),
    "cost_recorded": _cost_recorded,
    "cost_recommendation": _recommendation,
    # Retrain
    "retrain_started": lambda d: _with_period("Entraînement du challenger", d),
    "retrain_skipped": lambda d: f"Réentraînement ignoré : {d.get('reason')}",
    "retrain_failed": lambda d: f"Entraînement en échec : {d.get('error')}",
    "challenger_registered": lambda d: (f"Challenger v{d.get('version')} enregistré : RMSE {_num(d.get('rmse'), '.3f')} "
                                        f"en {_num(d.get('training_seconds'), '.0f')} s"),
    "model_promoted": _promoted,
    "promotion_blocked": _blocked,
    # Fiabilité
    "incident_opened": lambda d: (f"Incident {d.get('cause')} sur {d.get('component')} (niveau {d.get('level')})"
                                  + (f" : {d['evidence'][0]}" if d.get("evidence") else "")),
    "llm_decision": _llm,
    "judge_rejected": lambda d: (f"Action {d.get('action')} refusée par {d.get('juge', 'le juge')} "
                                 f"({d.get('note')}/10) : {d.get('critique', '')}"),
    "verdict_corrected": lambda d: f"Verdict du LLM corrigé par les mesures : {d.get('llm')} → {d.get('retenu')}",
    "incident_rejected_by_rule": lambda d: "Incident proposé par le LLM écarté : aucune anomalie mesurée",
    "snapshot_restored": lambda d: f"Retour à l'état d'avant l'action : {d.get('detail')}",
    "severity_changed": lambda d: f"Gravité {d.get('avant')} → {d.get('apres')}",
    "human_notified": lambda d: f"Humain prévenu : {d.get('reason')}",
    "incident_resolved": _resolved,
    "llm_unavailable": lambda d: f"LLM injoignable : {d.get('error')}",
    "llm_recovered": lambda d: "LLM de nouveau joignable",
    "agent_error": lambda d: f"Erreur interne : {d.get('error')}",
}


def summarize(action: str, details: dict | None) -> str:
    d = details or {}
    fn = SUMMARIES.get(action) or (_action if action.startswith("action:") else None)
    if fn:
        try:
            return fn(d)
        except Exception:  # un détail inattendu ne doit pas casser la page
            pass
    scalars = [f"{k} : {v}" for k, v in d.items() if isinstance(v, (str, int, float, bool)) and v != ""][:3]
    return ", ".join(scalars) or action
