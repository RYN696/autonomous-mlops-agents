"""Configuration partagée par tous les agents.
Chaque valeur peut être changée par une variable d'environnement (voir .env.example)."""
import os

# Services
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://mlops:mlops@localhost:5432/mlops")
API_URL = os.getenv("API_URL", "http://localhost:9696")

# Modèle
MODEL_NAME = os.getenv("MODEL_NAME", "taxi-duration")
PRODUCTION_ALIAS = "production"
PREVIOUS_ALIAS = "previous"  # version remplacée, pour un rollback
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "nyc-taxi-experiment")

# Bus d'événements
EVENTS_CHANNEL = os.getenv("EVENTS_CHANNEL", "mlops.events")

# Seuils (à ajuster par l'équipe)
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.02"))  # 2 % d'erreur en moins
RETRAIN_COOLDOWN_MINUTES = float(os.getenv("RETRAIN_COOLDOWN_MINUTES", "60"))  # pause entre deux entraînements
DEFAULT_TRAIN_MONTHS = int(os.getenv("DEFAULT_TRAIN_MONTHS", "1"))  # mois d'entraînement si aucune période
MAX_ERROR_RATE = float(os.getenv("MAX_ERROR_RATE", "0.05"))
MAX_LATENCY_SECONDS = float(os.getenv("MAX_LATENCY_SECONDS", "1.0"))

# Coûts (tarif fictif pour le prototype)
CPU_HOUR_PRICE = float(os.getenv("CPU_HOUR_PRICE", "0.05"))
MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET", "20.0"))


# Qualité des données (agent qualité)
MAX_TRIP_MINUTES = float(os.getenv("MAX_TRIP_MINUTES", "60"))
MAX_PASSENGERS = int(os.getenv("MAX_PASSENGERS", "8"))
MAX_TRIP_DISTANCE = float(os.getenv("MAX_TRIP_DISTANCE", "100"))
CRITICAL_INVALID_SHARE = float(os.getenv("CRITICAL_INVALID_SHARE", "0.05"))

COLUMN_DRIFT_THRESHOLD = float(os.getenv("COLUMN_DRIFT_THRESHOLD", "0.1"))


# --- Fiabilité ---
MONITORED_ENDPOINT = os.getenv("MONITORED_ENDPOINT", "/predict")  # route dont on surveille le SLO
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "3"))  # au-delà, l'API est jugée injoignable
MIN_REQUESTS = int(os.getenv("MIN_REQUESTS", "10"))  # en dessous, le taux d'erreurs n'est pas significatif
SLO_TARGET = float(os.getenv("SLO_TARGET", "0.99"))  # 99 % de requêtes réussies
SLO_WINDOW_MINUTES = float(os.getenv("SLO_WINDOW_MINUTES", "60"))
API_CONTAINER_NAME = os.getenv("API_CONTAINER_NAME", "mlops-api")
API_SERVICE_NAME = os.getenv("API_SERVICE_NAME", "api")  # nom du service dans docker-compose.yml
REDIS_CONTAINER_NAME = os.getenv("REDIS_CONTAINER_NAME", "mlops-redis")
MLFLOW_CONTAINER_NAME = os.getenv("MLFLOW_CONTAINER_NAME", "mlops-mlflow")
# Seuls conteneurs que l'agent a le droit de toucher (jamais Postgres)
ALLOWED_CONTAINERS = (API_CONTAINER_NAME, REDIS_CONTAINER_NAME, MLFLOW_CONTAINER_NAME)
ACTION_HTTP_TIMEOUT_SECONDS = float(os.getenv("ACTION_HTTP_TIMEOUT_SECONDS", "30"))  # /reload peut prendre plusieurs secondes
# Le client MLflow réessaie par défaut jusqu'à 7 fois (backoff x2, jusqu'à 120 s de délai) : pendant
# une panne MLflow, un simple snapshot avant action bloquerait l'agent plusieurs minutes. Ici, un
# échec rapide suffit (le code traite déjà l'absence de MLflow comme un fait, pas une erreur fatale).
RELIABILITY_MLFLOW_HTTP_MAX_RETRIES = int(os.getenv("RELIABILITY_MLFLOW_HTTP_MAX_RETRIES", "1"))
# Entier : MLflow fait int(valeur) sur cette variable, "3.0" lèverait une ValueError.
RELIABILITY_MLFLOW_HTTP_TIMEOUT_SECONDS = int(os.getenv("RELIABILITY_MLFLOW_HTTP_TIMEOUT_SECONDS", "3"))
# Modèle de secours du mode dégradé : durée = base + minutes_par_mile x distance
FALLBACK_BASE_MINUTES = float(os.getenv("FALLBACK_BASE_MINUTES", "4.0"))
FALLBACK_MINUTES_PER_MILE = float(os.getenv("FALLBACK_MINUTES_PER_MILE", "3.5"))
# Boucle de l'agent
RELIABILITY_POLL_SECONDS = float(os.getenv("RELIABILITY_POLL_SECONDS", "15"))
PROBE_COUNT = int(os.getenv("PROBE_COUNT", "10"))  # sondes par cycle (>= MIN_REQUESTS : mesure fiable sans trafic)
PROBE_MAX_CONSECUTIVE_FAILURES = int(os.getenv("PROBE_MAX_CONSECUTIVE_FAILURES", "2"))  # au-delà, API jugée HS : on arrête les sondes
FAULT_INJECTION = os.getenv("FAULT_INJECTION", "0") == "1"  # route /fault de démo, désactivée par défaut

COST_PER_CPU_SECOND = float(os.getenv("COST_PER_CPU_SECOND", "0.0005"))
COST_SPOT_DISCOUNT = float(os.getenv("COST_SPOT_DISCOUNT", "0.6"))
DEFAULT_CPU_SECONDS_PER_MONTH_OF_DATA = float(
    os.getenv("DEFAULT_CPU_SECONDS_PER_MONTH_OF_DATA", "600.0")
)
COST_IDLE_MINUTES = float(os.getenv("COST_IDLE_MINUTES", "30.0"))
COST_SIZE_HEADROOM = float(os.getenv("COST_SIZE_HEADROOM", "0.3"))
COST_HOURS_PER_MONTH = float(os.getenv("COST_HOURS_PER_MONTH", "730.0"))


# --- LLM (Ollama, 100 % local) : ils prennent toutes les décisions de l'agent Fiabilité ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_AGENT_MODEL = os.getenv("LLM_AGENT_MODEL", "qwen2.5:3b")    # décide (appelé à chaque cycle)
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "llama3.2:3b")   # juge : autre famille de modèle
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))  # 1er chargement d'un modèle : ~30-45 s
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")             # garder le modèle en mémoire GPU
LLM_MAX_PROPOSALS = int(os.getenv("LLM_MAX_PROPOSALS", "2"))    # propositions d'action max par cycle (si refus du juge)
LLM_THINK = os.getenv("LLM_THINK", "off")    # "off" : coupe le mode réflexion (qwen3...) trop lent ; "on" ; "auto" : défaut du modèle
LLM_TREND_CYCLES = int(os.getenv("LLM_TREND_CYCLES", "8"))      # mesures passées montrées au LLM
JUDGE_MIN_SCORE = int(os.getenv("JUDGE_MIN_SCORE", "6"))        # note minimale (sur 10) pour approuver
