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
`host.docker.internal`. Pour jouer des scénarios, voir la section suivante.

## Tester la pipeline avec l'interface

Tout se fait depuis l'interface : la page **Scénarios de test** a un bouton par
dérive ou panne, et chaque lancement mène au parcours qu'il a déclenché. L'interface
se rafraîchit toute seule toutes les 5 s.

### 1. Démarrer

```bash
# Windows (PowerShell) : .venv\Scripts\activate     macOS / Linux : source .venv/bin/activate
ollama serve                 # si Ollama ne tourne pas déjà (agents Qualité et Fiabilité)
docker compose up -d --build
```

Ouvrir http://localhost:8080. Si le port 8080 est déjà pris (par exemple par Oracle),
lancer avec un autre port et ouvrir http://localhost:8090 :

```bash
# PowerShell
$env:ORCHESTRATOR_HOST_PORT=8090; docker compose up -d --build
# macOS / Linux
ORCHESTRATOR_HOST_PORT=8090 docker compose up -d --build
```

Vérifier sur la carte que les 5 agents sont « en marche », puis écrire son prénom
dans **Signer en tant que** (menu de gauche) : il signe les validations.

### 2. Scénarios

Menu **Scénarios de test** : choisir le mois (pour les scénarios de données), cliquer
**Lancer**, puis suivre le lien **voir le parcours** dans la colonne **Lancements**.
Quand une question apparaît (dérive avec le budget de démo), la valider ou la refuser
sur la vue d'ensemble ou directement dans le parcours. Le bouton **Démarrer le trafic**
simule des utilisateurs, utile pendant une panne. Chaque lancement est écrit dans le
journal d'actions (acteur « Banc de test »). La page est active grâce à
`DEMO_CONTROLS: "1"` dans le service `orchestrator`.

Ce que chaque scénario doit montrer (avec la commande équivalente, si on préfère le terminal) :

| Scénario | Commande équivalente | Ce qu'on doit voir dans l'interface |
| --- | --- | --- |
| Dérive des données | `docker compose exec quality python -m agents.quality --year 2023 --month 3 --scenario drift` | Parcours « Dérive 2023-03 » et une question dans **À valider**. Cliquer **Valider** : Retrain entraîne (~1 min), puis promotion ou promotion bloquée |
| Refus humain | même commande avec `--month 4`, puis **Refuser** | Parcours « Refusé par un humain », la production ne change pas |
| Données corrompues | `docker compose exec quality python -m agents.quality --year 2023 --month 2 --scenario corrupted` | Parcours « Données corrompues », aucun réentraînement |
| Crash de l'API | `python -m scripts.inject_failure crash` | Incident `api_arretee`, Fiabilité redémarre l'API, puis « Résolu en … s » |
| Erreurs 500 | `python -m scripts.inject_failure errors` | Incident, décisions du LLM et du juge, action de Fiabilité étape par étape |
| Latence | `python -m scripts.inject_failure latency` | Même déroulé, cause latence |
| Mauvais modèle | `python -m scripts.inject_failure bad-model` | Rollback automatique vers la version précédente (règle déterministe) |
| MLflow en panne | `python -m scripts.inject_failure mlflow-down` | Mode dégradé et demande d'intervention humaine (relancer MLflow demande une validation) |
| Dérive pendant une panne | `crash`, puis tout de suite la dérive | Parcours de dérive « Mise de côté : incident en cours » |

Annuler toutes les pannes : bouton **Remettre en état** (ou `python -m scripts.inject_failure recover`).

Changer de mois à chaque dérive (3, 4, 5…) : Retrain ignore un mois déjà entraîné
(anti-doublon, la liste des mois indique « déjà entraîné »), et le parcours finit alors
en « Ignoré par Retrain ». La première
fois, les données du mois sont téléchargées : c'est un peu plus long.

### 3. Où lire la traçabilité

- **Parcours** : cliquer sur un parcours pour sa chronologie (heure, temps écoulé, agent,
  ce qu'il a fait). **détails** affiche le JSON brut de l'étape.
- **Agents** : toutes les actions d'un agent, filtrables par type (ex. `incident_opened`).
- **Journal d'actions** : recherche par identifiant de parcours, cause ou version.
- **Graphes LangGraph** : le circuit de décision de l'orchestrateur et les demandes en pause.

### À savoir

- Le budget de l'agent Coûts vaut `0.01` € dans `docker-compose.yml` (démo du dépassement) :
  chaque dérive demande donc une validation. Pour un réentraînement automatique, mettre
  `MONTHLY_BUDGET: "20"` puis `docker compose up -d cost`.
- Un challenger n'est promu que s'il réduit l'erreur d'au moins 2 % (`MIN_IMPROVEMENT`) :
  la promotion bloquée est le résultat le plus fréquent.
- Si l'orchestrateur redémarre, les questions en attente expirent : relancer la dérive.

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