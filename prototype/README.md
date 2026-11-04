# Prototype : système multi-agents MLOps autonome

Pipeline taxi NYC (entraînement, registre MLflow, API de prédiction) surveillée par
5 agents (Qualité, Orchestrateur LangGraph, Coûts, Retrain, Fiabilité) qui se parlent
par Redis, avec une interface web pour suivre chaque décision et lancer des tests.

## Démarrage rapide (après un `git pull`)

Prérequis : **Git**, **Docker Desktop** (démarré) et **[Ollama](https://ollama.com)**
(les LLM des agents tournent en local). Python n'est nécessaire que pour lancer les tests.

```bash
# 1. Récupérer le code (tant que la PR n'est pas fusionnée, la branche est orchestrator-langgraph)
git fetch origin
git checkout orchestrator-langgraph     # ensuite : git checkout prototype && git pull
cd prototype

# 2. Les modèles LLM des agents (une seule fois, ~4 Go)
ollama pull qwen2.5:3b       # agent Fiabilité : décide
ollama pull llama3.2:3b      # juge de l'agent Fiabilité, et explication de l'agent Qualité

# 3. Tout lancer
copy .env.example .env       # macOS / Linux : cp .env.example .env
docker compose up -d --build
```

Ouvrir **http://localhost:8080**. Si le port 8080 est déjà pris (Oracle, par exemple),
mettre `ORCHESTRATOR_HOST_PORT=8090` dans `.env`, relancer `docker compose up -d`
et ouvrir http://localhost:8090.

Au premier lancement, le service `init` télécharge les données taxi (Internet requis)
et entraîne un premier modèle : compter 3 à 5 minutes, l'interface l'indique. Ensuite,
tout se teste depuis la page **Scénarios de test** (voir plus bas). Si Ollama ou un
modèle manque, un bandeau de la vue d'ensemble donne la commande à lancer.

Arrêter : `docker compose down` (ajouter `-v` pour tout effacer, y compris les modèles
et le journal d'actions).

## Pour développer (Python)

```bash
cd prototype
python -m venv .venv
# Windows (PowerShell) :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate
pip install -r requirements.txt
pytest                        # avant chaque pull request
```

Tester une prédiction à la main :

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"PULocationID\": 43, \"DOLocationID\": 151, \"trip_distance\": 1.8}"
```

## Services

| Service | Adresse | Rôle |
| --- | --- | --- |
| MLflow | http://localhost:5000 | Suivi des entraînements et registre des modèles |
| API | http://localhost:9696 | `/predict`, `/health`, `/reload` |
| Grafana | http://localhost:3000 | Tableau de bord (source Postgres déjà configurée) |
| Postgres | localhost:5432 (mlops / mlops) | Journal d'actions et métriques |
| Redis | localhost:6379 | Bus d'événements entre agents |
| Interface | http://localhost:8080 | Vue d'ensemble, parcours, agents, journal, validations humaines, graphes LangGraph (port hôte : `ORCHESTRATOR_HOST_PORT` si 8080 est pris) |
| Discord (facultatif) | vos salons | Mêmes messages que l'interface, avec boutons (voir « Alertes et boutons sur Discord ») |

`docker compose up -d --build` lance aussi les 5 agents, qui ne se parlent que par Redis :

| Agent | Service | Rôle dans la boucle |
| --- | --- | --- |
| Qualité | `quality` | Contrôle 2023-02 toutes les 10 min et après chaque promotion ; publie `drift_detected` avec le mois contrôlé |
| Orchestrateur | `orchestrator` | Graphe LangGraph : coût → règle → validation humaine → `retrain_requested` ; pas de retrain pendant un incident |
| Coûts | `cost` | Estime le coût, signale le dépassement de budget, enregistre le coût de chaque entraînement, signale les pics de coût |
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
- **Rapport de coûts** : dépense du mois (entraînement, API), budget, projection de fin de mois.
  Le bouton « Générer le rapport du mois » fait expliquer ces chiffres par un LLM
  (`COST_LLM_MODEL`, `llama3.2:3b` par défaut). Le code calcule tous les chiffres et les verdicts ;
  un texte du LLM qui cite un nombre absent des constats est écarté et remplacé par un texte du code.
  Le rapport est écrit au journal (`cost_report`) et envoyé sur Discord s'il est configuré.

### Alertes et boutons sur Discord (facultatif)

Si un bot Discord est configuré, chaque message de l'interface part aussi sur Discord :

| Salon | Variable | Messages |
| --- | --- | --- |
| `#mlops-alertes` | `DISCORD_ALERTS_CHANNEL_ID` | incidents, demandes d'intervention, données corrompues, budget dépassé, entraînement en échec |
| `#mlops-agents` | `DISCORD_AGENTS_CHANNEL_ID` | réentraînements, promotions, promotions bloquées, questions à valider |

Un seul salon renseigné : tout y va. Un parcours (dérive ou incident) tient en **un seul
message**, complété étape par étape (cause → action → résultat). Boutons :

| Bouton | Où | Effet |
| --- | --- | --- |
| **Valider / Refuser** | question de budget | comme sur la page web (publie `approval_given`) |
| **Revenir à vN** | modèle promu | après confirmation, remet l'ancienne version en production (alias MLflow + `/reload` de l'API) |
| **Ignorer** | données corrompues | aucune action, mais noté au journal |
| **Rapport**, **Voir les données** | partout | ouvre le parcours dans l'interface |
| **Tableau de bord** | incidents, données | ouvre Grafana |

Chaque clic est écrit au journal sous l'agent « Humain », avec le nom Discord de la
personne : il apparaît dans la chronologie du parcours. Les boutons utilisés disparaissent.

Mise en place (une fois, ~5 min) :

1. https://discord.com/developers/applications > **New Application** > onglet **Bot** >
   **Reset Token** : copier le jeton (aucun « Privileged Gateway Intent » n'est nécessaire).
2. Onglet **OAuth2 > URL Generator** : cocher `bot`, puis les permissions *View Channels*,
   *Send Messages*, *Embed Links* ; ouvrir l'URL générée pour inviter le bot sur votre serveur.
3. Dans Discord : *Paramètres > Avancés > Mode développeur*, puis clic droit sur chaque salon >
   **Copier l'identifiant**.
4. Dans `.env` : `DISCORD_BOT_TOKEN`, `DISCORD_ALERTS_CHANNEL_ID`, `DISCORD_AGENTS_CHANNEL_ID`
   (et `DISCORD_ALERT_MENTION=@here` pour être prévenu), puis `docker compose up -d orchestrator`.

La pastille **Discord** de la vue d'ensemble indique si le bot est connecté. Le bot se
connecte vers Discord (aucune URL publique nécessaire) ; les liens des boutons pointent vers
`localhost`, ils s'ouvrent donc sur le PC qui fait tourner la pipeline. Le jeton reste dans
`.env`, qui n'est pas versionné : ne jamais le mettre dans un commit.

Qualité et Fiabilité utilisent Ollama, lancé **sur le PC** (`ollama serve`, modèles
`qwen2.5:3b` et `llama3.2:3b`) : les conteneurs le joignent par
`host.docker.internal`. Pour jouer des scénarios, voir la section suivante.

## Storyboard : les 4 scénarios joués pour de vrai

Page **Storyboard** de l'interface : les 4 scénarios du Design Thinking, chacun avec ses
cases. **Lancer** prépare la situation, puis ce sont les agents qui réagissent seuls ; chaque
case se coche avec ce qui s'est vraiment passé (versions, RMSE, coûts, heures). Si les agents
prennent un autre chemin que l'histoire (ex. challenger pas assez meilleur), la case passe en
**variante** : le résultat n'est jamais maquillé.

| Scénario | Ce que « Lancer » prépare | Ce que font les agents |
| --- | --- | --- |
| 1. Dérive des données | budget large (20 €), dérive fabriquée sur le mois choisi | Qualité → orchestrateur (règle : retrain automatique) → Retrain → promotion si le challenger est meilleur d'au moins 2 % → un seul message, avec **Revenir à vN** |
| 2. Panne du service | promeut une version qui plante sur les trajets de plus de 8 km (saine au calme), puis un pic de trafic | Fiabilité ouvre un incident → redémarre l'API → erreurs toujours là → règle : rollback vers la dernière version saine → rapport d'incident |
| 3. Commande par chat et budget | budget serré (un mois d'entraînement passe, trois non), puis la phrase « @agents retrain le modèle taxi sur les 3 derniers mois » | le LLM comprend la phrase, le code la vérifie → Coûts : budget dépassé → question **Valider / Refuser** au responsable → retrain tracé |
| 4. Le mauvais challenger | budget large, fichier reçu avec des valeurs aberrantes (`data/incoming/`) | Qualité : forte dérive, données suspectes (sous le seuil « corrompu ») → Retrain entraîne dessus → challenger moins bon → promotion bloquée → alerte « vérification humaine requise » avec **Voir les données** / **Ignorer** |

Nouveautés derrière ces scénarios :

- **Chat #mlops-agents** (`/chat`) : une phrase devient une action (LLM `CHAT_LLM_MODEL`,
  `llama3.2:3b` ; des règles par mots-clés prennent le relais sans Ollama). Le code vérifie
  le modèle et la période (1 à `CHAT_MAX_MONTHS` mois) ; la demande suit ensuite le même
  graphe qu'une dérive (coût, règles d'autonomie, validation humaine).
- **Budget du mois réglable en direct** (page Storyboard) : gardé dans Redis
  (`common/budget.py`), relu par l'agent Coûts à chaque estimation. Sans réglage, chaque
  agent garde `MONTHLY_BUDGET`.
- **Données suspectes** : part de valeurs impossibles ≥ `SUSPECT_SHARE` (0,5 %) mais sous
  `CRITICAL_INVALID_SHARE`. Un challenger *pire* que la production, entraîné sur ces données,
  déclenche l'alerte `suspect_data_alert`.
- **Fiabilité** : une version qui a d'abord fonctionné n'est plus désignée tout de suite par la
  règle « mauvais modèle » ; si un redémarrage ne règle rien et qu'une version a été promue
  depuis moins de `RELIABILITY_SUSPECT_MODEL_HOURS` h, la règle `restart_did_not_help` fait le
  rollback. Une action suivie de `RELIABILITY_NOT_REPAIRED_CYCLES` mesures en erreur est jugée
  « pas réparée » sans attendre le LLM.
- Mêmes gestes humains depuis la page que sur Discord : **Revenir à vN**, **Ignorer**.

## Tester la pipeline avec l'interface

Tout se fait depuis l'interface : la page **Scénarios de test** a un bouton par
dérive ou panne, et chaque lancement mène au parcours qu'il a déclenché. L'interface
se rafraîchit toute seule toutes les 5 s.

### 1. Démarrer

Voir **Démarrage rapide** en haut de ce fichier. Vérifier sur la carte de la vue
d'ensemble que les 5 agents sont « en marche », puis écrire son prénom dans
**Signer en tant que** (menu de gauche) : il signe les validations.

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
| Dérive des données | `docker compose exec quality python -m agents.quality --year 2023 --month 3 --scenario drift` | Parcours « Dérive 2023-03 » et une question dans **À valider**. Cliquer **Valider** : Retrain optimise les réglages (Optuna) puis entraîne (quelques minutes), puis promotion ou promotion bloquée, avec un message qui l'explique |
| Refus humain | même commande avec `--month 4`, puis **Refuser** | Parcours « Refusé par un humain », la production ne change pas |
| Données corrompues | `docker compose exec quality python -m agents.quality --year 2023 --month 2 --scenario corrupted` | Parcours « Données corrompues », aucun réentraînement |
| Données suspectes | même commande avec `--month 3 --scenario suspect` | Retrain sur le fichier reçu, promotion bloquée, alerte « données suspectes » |
| Crash de l'API | `python -m scripts.inject_failure crash` | Incident `api_arretee`, Fiabilité redémarre l'API, puis « Résolu en … s » |
| Erreurs 500 | `python -m scripts.inject_failure errors` | Incident, décisions du LLM et du juge, action de Fiabilité étape par étape |
| Latence | `python -m scripts.inject_failure latency` | Même déroulé, cause latence |
| Mauvais modèle | `python -m scripts.inject_failure bad-model` | Rollback automatique vers la version précédente (règle déterministe) |
| Modèle défaillant sous la charge | `python -m scripts.inject_failure flaky-model`, puis **Démarrer le trafic** | Saine au calme ; avec du trafic : incident, redémarrage sans effet, puis rollback (règle `restart_did_not_help`) |
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
  Sur Discord, leurs anciens boutons répondent alors que la question a expiré.

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

**Pic de coût** (`agents/cost/anomaly.py`, événement `cost_anomaly`) : règle déterministe, sans LLM.
Une mesure est un pic si elle vaut au moins `COST_SPIKE_FACTOR` (3) fois la médiane des
`COST_SPIKE_WINDOW` (10) mesures précédentes, avec au moins `COST_SPIKE_MIN_HISTORY` (3) mesures.
- entraînement : CPU **par mois de données** (3 mois ne sont pas un pic face à 1 mois) ;
- API servie : vCPU moyen d'un tour de boucle, au-dessus de `COST_SPIKE_MIN_VCPUS` (0,1) pour
  ignorer le bruit de fond ; une seule alerte par épisode.
L'orchestrateur prévient l'humain (page web, Discord) avec la cause à vérifier ; aucune action
automatique. Les pics du mois apparaissent dans le rapport de coûts.

## Règles de travail

- Une branche par agent, créée depuis `prototype`, qui revient par pull request.
- Ne pas modifier les dossiers du cours (01 à 07) : tout le code va dans `prototype/`.
- Toute modification de `common/events.py` se valide en équipe : elle engage les 5 agents.
- Ne jamais commiter `.env`, `data/` ni `models/`.
- Une dépendance ajoutée à `requirements.txt` doit aussi aller dans `requirements-docker.txt` (dépendances de l'image Docker, avec `xgboost-cpu`).