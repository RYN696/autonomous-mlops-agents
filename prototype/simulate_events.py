import os
import sys
from pathlib import Path

# Configuration de la BDD SQLite locale
os.environ["DATABASE_URL"] = "sqlite:///cost_demo.db"

PROTOTYPE_DIR = Path(__file__).resolve().parents[0]
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

import time
from datetime import datetime, timedelta
from agents.cost.service import build_agent, CostService, make_handler, LISTENED
from agents.cost.sql_store import SqlCostStore
from common.events import Event, EventType

class InMemoryEventBus:
    def __init__(self):
        self.listeners = []

    def listen(self, handler, types=None):
        self.listeners.append((handler, types))

    def publish(self, event):
        for handler, types in self.listeners:
            if types is None or event.type in types:
                handler(event)

def run_dynamic_simulation():
    print("🚀 Démarrage de la simulation dynamique de l'Agent Coûts...")
    
    bus = InMemoryEventBus()
    store = SqlCostStore()
    agent = build_agent(store=store, bus=bus)
    
    # Écoute sur le bus en mémoire
    bus.listen(make_handler(agent), types=LISTENED)

    model = "taxi"

    try:
        # --- Objectif 1 : Training Completed ---
        print("\n[Événement 1/4] Transmission de TRAINING_COMPLETED...")
        bus.publish(Event(
            type=EventType.TRAINING_COMPLETED,
            source="trainer",
            model=model,
            payload={"cpu_seconds": 1500, "months": 3}
        ))
        time.sleep(2)

        # --- Objectifs 3 & 4 : Cost Estimate avec dépassement & Spot ---
        print("\n[Événement 2/4] Transmission de COST_ESTIMATE (Dépassement de budget)...")
        bus.publish(Event(
            type=EventType.COST_ESTIMATE,
            source="orchestrator",
            model=model,
            payload={"months": 12, "request_id": "req-999"}
        ))
        time.sleep(2)

        # --- Objectif 2 : Scale-to-Zero ---
        print("\n[Événement 3/4] Exécution de check_idle (Scale-to-Zero)...")
        last_req = datetime.now() - timedelta(hours=2)
        agent.check_idle(model=model, last_request_at=last_req, size="M", idle_hours=2.0)
        time.sleep(2)

        # --- Objectif 5 : Dimensionnement vCPU ---
        print("\n[Événement 4/4] Exécution de recommend_server_size (Right-sizing)...")
        agent.recommend_server_size(model=model, used_vcpus=0.3, current_size="L")
        time.sleep(2)

        print("\n✅ Simulation exécutée sans aucune erreur.")

    except Exception as e:
        print(f"❌ Erreur lors de la simulation : {e}")

if __name__ == "__main__":
    run_dynamic_simulation()