# Prototype : système multi-agents MLOps autonome

Socle commun du prototype. Il lance la pipeline taxi NYC (entraînement, registre
MLflow, API de prédiction) et l'infrastructure dont les 5 agents ont besoin
(bus d'événements Redis, journal d'actions Postgres, tableau de bord Grafana).

## Prérequis

- Git
- Python 3.11 ou plus récent
- Docker Desktop (démarré)

## Installation (une seule fois)

Depuis la racine du dépôt, sur la branche `prototype` :

```bash
cd prototype
python -m venv .venv
# Windows (PowerShell) :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # macOS / Linux : cp .env.example .env
```

## Lancer le prototype

```bash
docker compose up -d --build                     # MLflow, Postgres, Redis, Grafana, API
python -m pipeline.train --year 2023 --month 1   # premier modèle, mis en production
curl -X POST http://localhost:9696/reload        # l'API charge le modèle
```

Le premier entraînement télécharge deux mois de données taxi dans `data/`
(une seule fois, elles sont ensuite lues depuis le disque).

Tester une prédiction :

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"PULocationID\": 43, \"DOLocationID\": 151, \"trip_distance\": 1.8}"
```

Arrêter : `docker compose down` (ajouter `-v` pour tout effacer, y compris les modèles).

## Services

| Service | Adresse | Rôle |
| --- | --- | --- |
| MLflow | http://localhost:5000 | Suivi des entraînements et registre des modèles |
| API | http://localhost:9696 | `/predict`, `/health`, `/reload` |
| Grafana | http://localhost:3000 | Tableau de bord (source Postgres déjà configurée) |
| Postgres | localhost:5432 (mlops / mlops) | Journal d'actions et métriques |
| Redis | localhost:6379 | Bus d'événements entre agents |
| Interface | http://localhost:8080 | Vue d'ensemble, parcours, agents, journal, validations humaines, graphes LangGraph (port hôte : `ORCHESTRATOR_HOST_PORT` si 8080 est pris) |

`docker compose up -d --build` lance aussi les 5 agents, qui ne se parlent que par Redis :

| Agent | Service | Rôle dans la boucle |
| --- | --- | --- |
| Qualité | `quality` | Contrôle 2023-02 toutes les 10 min et après chaque promotion ; publie `drift_detected` avec le mois contrôlé |
| Orchestrateur | `orchestrator` | Graphe LangGraph : coût → règle → validation humaine → `retrain_requested` ; pas de retrain pendant un incident |
| Coûts | `cost` | Estime le coût, signale le dépassement de budget, enregistre le coût de chaque entraînement |
| Retrain | `retrain` | Entraîne sur le mois de la dérive, promeut ou bloque le challenger |
| Fiabilité | `reliability` | Surveille l'API, MLflow, Redis ; redémarre / recrée les conteneurs (socket Docker monté) |

### Interface et traçabilité

L'interface (servie par l'orchestrateur, code dans `ui/`) se met à jour toute seule toutes les 5 s :

- **Vue d'ensemble** : modèle en production, coût du mois, incidents ouverts, questions à valider,
  carte des agents (état du conteneur, dernière action) et services.
- **Parcours** : chaque dérive ou incident suivi de bout en bout. Les agents se passent un identifiant
  (`check_id` du contrôle Qualité, puis `request_id` de l'orchestrateur repris par Coûts et Retrain,
  ou `incident_id` de Fiabilité), ce qui permet d'afficher la chronologie complète, agent par agent,
  avec le détail JSON de chaque étape et les messages envoyés à l'humain.
- **Agents** : rôle, événements écoutés et publiés, dernières actions filtrables.
- **Journal d'actions** : tout le journal, filtrable par agent, action ou texte.

Qualité et Fiabilité utilisent Ollama, lancé **sur le PC** (`ollama serve`, modèles
`qwen2.5:3b`, `llama3.2:3b`, `llama3.1:8b`) : les conteneurs le joignent par
`host.docker.internal`. Une démo se déclenche depuis les conteneurs, par exemple :

```bash
docker compose exec quality python -m agents.quality --year 2023 --month 2 --scenario drift
python -m scripts.inject_failure crash      # depuis le PC : l'agent Fiabilité répare
```

## Tests

```bash
pytest
```

Les tests utilisent des données synthétiques et un MLflow local : ils ne
téléchargent rien et ne demandent pas Docker.

## Organisation du code

```
prototype/
  common/          # partagé par tous les agents
    config.py      # seuils, budget, adresses (modifiables par variables d'environnement)
    events.py      # contrat d'événements + bus Redis
    action_log.py  # journal d'actions
  pipeline/
    data.py        # chargement des données taxi + préparation des variables
    train.py       # entraînement + enregistrement dans MLflow
    api/app.py     # API de prédiction
  agents/
    cost/          # estimation, suivi des coûts et recommandations de budget
  rules/autonomy.yaml  # ce que les agents font seuls / avec validation
  tests/
```

L'agent coûts écoute `cost_estimate` et `training_completed`, publie des
événements typés définis dans `common/events.py` et journalise ses décisions
dans `common/action_log.py`. Ses tarifs, seuils et budget sont configurables
dans `common/config.py`.

## Règles de travail

- Une branche par agent, créée depuis `prototype`, qui revient par pull request.
- Ne pas modifier les dossiers du cours (01 à 07) : tout le code va dans `prototype/`.
- Toute modification de `common/events.py` se valide en équipe : elle engage les 5 agents.
- Ne jamais commiter `.env`, `data/` ni `models/`.
- Une dépendance ajoutée à `requirements.txt` doit aussi aller dans `requirements-docker.txt` (dépendances de l'image Docker, avec `xgboost-cpu`).