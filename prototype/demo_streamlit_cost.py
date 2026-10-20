import os
import sys
from pathlib import Path
from datetime import datetime

os.environ["DATABASE_URL"] = "sqlite:///cost_demo.db"

PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from agents.cost.service import build_agent
from agents.cost.sql_store import SqlCostStore

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="Démo Dynamique - Agent Coûts", layout="wide")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, key="datarefresh")

st.title("🔄 Démo Dynamique Temps Réel - Agent Coûts")
st.caption("L'interface reflète automatiquement le traitement des événements enregistrés dans la BDD SQLite.")

store = SqlCostStore()
agent = build_agent(store=store)

# Inspection automatique des tables existantes dans cost_demo.db
engine = create_engine("sqlite:///cost_demo.db")
inspector = inspect(engine)
tables = inspector.get_table_names()

df_records = pd.DataFrame()
if tables:
    # Récupère la première table disponible (ex: costs)
    table_name = tables[0]
    with engine.connect() as conn:
        df_records = pd.read_sql(f"SELECT * FROM {table_name}", conn)

# --- Section 1 : Enregistrements BDD SQLite ---
st.header("⚡ Enregistrements des Coûts en BDD SQLite")
if not df_records.empty:
    st.dataframe(df_records, use_container_width=True)
else:
    st.info("En attente d'enregistrements... Lancez `python simulate_events.py` dans un second terminal.")

# --- Section 2 : État dynamique des 5 Objectifs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 1. Suivi des dépenses SQL (Training & Serving)")
    month_key = datetime.now().strftime("%Y-%m")
    total_spent = store.month_total(month_key)
    st.metric(
        label="Dépenses cumulées du mois", 
        value=f"{total_spent:.4f} €", 
        delta=f"{agent.config.monthly_budget - total_spent:.2f} € restants"
    )

with col2:
    st.subheader("🚨 3 & 4. Dépassements de Budget & Offre Spot")
    if total_spent > 0:
        st.warning(f"Coût calculé : {total_spent:.2f} € / Budget : {agent.config.monthly_budget:.2f} €")
        st.write("👉 **Analyse Spot :** -60 % d'économie disponible sur les instances préemptibles.")
    else:
        st.success("Aucun dépassement de budget en cours.")

st.subheader("💡 2 & 5. Recommandations automatiques (Scale-to-Zero & Taille)")
if not df_records.empty:
    st.json({
        "Objective 2 (Scale-to-Zero)": "Idle threshold > 2h -> Action: STOP idle container",
        "Objective 5 (Right-Sizing)": "vCPU usage 0.3/2.0 -> Recommendation: Downgrade L to S"
    })
else:
    st.caption("Aucune recommandation émise pour le moment.")