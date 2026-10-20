# Intégration de l'agent Coûts

`agent.py` est déjà branché sur le socle (Event, EventBus, ActionLog). Il reste :

- `listener.py` : convertit `period` ("3 derniers mois") en `months`, isole les erreurs.
- `service.py` : point d'entrée. Depuis `prototype/` : `python -m agents.cost.service`.
- `sql_store.py` : stockage Postgres (table `model_costs`).

## Contrat d'événements
- Entrée `cost_estimate` : `months` ou `period`, `request_id?`, `requested_by?`.
- Entrée `training_completed` : `cpu_seconds`, `months?`.
- Sorties : `cost_estimated`, `budget_exceeded` (avec `over_by`), `cost_recommendation`.

## À confirmer avec l'équipe
- `period` ou `months` ; `request_id` repris par l'Orchestrateur ;
- qui publie `training_completed` ;
- gravité de `budget_exceeded` (INFO aujourd'hui) ;
- tarif en double dans `common/config.py` (CPU_HOUR_PRICE vs COST_PER_CPU_SECOND).
