from datetime import datetime
import os
from pathlib import Path
import sys

from explainer import ask_cost_assistant, explain_cost_with_ollama
import pandas as pd
from sqlalchemy import create_engine, inspect
import streamlit as st

# Configuration de la BDD
os.environ["DATABASE_URL"] = "sqlite:///cost_demo.db"

PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_DIR) not in sys.path:
  sys.path.insert(0, str(PROTOTYPE_DIR))

from agents.cost.service import build_agent
from agents.cost.sql_store import SqlCostStore

# Module d'auto-refresh
try:
  from streamlit_autorefresh import st_autorefresh

  HAS_AUTOREFRESH = True
except ImportError:
  HAS_AUTOREFRESH = False

st.set_page_config(
    page_title="Démo Dynamique - Agent Coûts FinOps", layout="wide"
)

if HAS_AUTOREFRESH:
  st_autorefresh(interval=2000, key="datarefresh")

# --- INITIALISATION DE LA MÉMOIRE DE SESSION (SESSION STATE) ---
if "summary_report" not in st.session_state:
  st.session_state["summary_report"] = None
if "chat_response" not in st.session_state:
  st.session_state["chat_response"] = None

# --- Initialisation Agent et Store ---
store = SqlCostStore()
agent = build_agent(store=store)

default_budget = float(getattr(agent.config, "monthly_budget", 20.0))

# --- Panneau Latéral (Sidebar) ---
st.sidebar.title("⚙️ Configuration FinOps")

custom_budget = st.sidebar.slider(
    "Budget mensuel simulé (€)",
    min_value=1.0,
    max_value=50.0,
    value=default_budget,
    step=0.50,
    key="budget_slider",
)

ollama_model = st.sidebar.text_input(
    "Modèle Ollama :", value="phi3", key="ollama_model_input"
)

# --- Titre Principal ---
st.title("🔄 Démo Dynamique Temps Réel - Agent Coûts")
st.caption(
    "L'interface reflète automatiquement le traitement des événements"
    " enregistrés dans la BDD SQLite."
)

# --- Lecture & Inspection BDD SQLite ---
engine = create_engine("sqlite:///cost_demo.db")
inspector = inspect(engine)
all_tables = [
    t for t in inspector.get_table_names() if not t.startswith("sqlite_")
]

df_records = pd.DataFrame()
if all_tables:
  target_table = "cost_records" if "cost_records" in all_tables else all_tables[0]
  with engine.connect() as conn:
    df_records = pd.read_sql(f"SELECT * FROM {target_table}", conn)

cost_col = None
if not df_records.empty:
  for col in ["cost", "amount", "estimated_cost", "val", "price"]:
    if col in df_records.columns:
      cost_col = col
      break
  if not cost_col:
    numeric_cols = df_records.select_dtypes(
        include=["float", "int"]
    ).columns.tolist()
    if numeric_cols:
      cost_col = numeric_cols[0]

# --- Algorithme de Détection des Pics (Cost Spikes) ---
spike_detected = False
spike_msg = ""

if not df_records.empty and cost_col:
  mean_cost = df_records[cost_col].mean()

  if len(df_records) >= 2:
    prev_mean = df_records[cost_col].iloc[:-1].mean()
    latest_cost = df_records[cost_col].iloc[-1]

    if prev_mean > 0 and latest_cost > 1.5 * prev_mean:
      spike_detected = True
      increase_pct = ((latest_cost - prev_mean) / prev_mean) * 100
      spike_msg = (
          f"🚨 **Pic atypique de consommation détecté !** Le dernier événement"
          f" ({latest_cost:.4f} €) dépasse de +{increase_pct:.1f}% la moyenne"
          f" des événements précédents ({prev_mean:.4f} €)."
      )

  df_display = df_records.copy()
  df_display["Statut FinOps"] = df_display[cost_col].apply(
      lambda x: (
          "⚠️ Pic (>50% moy)"
          if (mean_cost > 0 and x > 1.5 * mean_cost)
          else "✅ Normal"
      )
  )
else:
  df_display = df_records.copy()

# --- Section 1 : Enregistrements & Graphiques ---
st.header("⚡ Enregistrements & Suivi Temporel SQL")

if spike_detected:
  st.error(spike_msg)

col_table, col_chart = st.columns([1.2, 0.8])

with col_table:
  st.subheader("📋 Transactions BDD")
  if not df_display.empty:
    st.dataframe(df_display, width="stretch", height=250)
  else:
    st.info(
        "En attente d'enregistrements... Lancez `python simulate_events.py`."
    )

with col_chart:
  st.subheader("📈 Accumulation & Tendance")
  if not df_records.empty and cost_col:
    df_records["cum_cost"] = df_records[cost_col].cumsum()
    st.line_chart(df_records["cum_cost"], height=250)
  else:
    st.caption("Le graphique s'affichera dès que des coûts seront enregistrés.")

# --- Section 2 : Analyse des 5 Objectifs ---
st.write("---")
st.header("🎯 Analyse des 5 Objectifs")

col1, col2 = st.columns(2)
month_key = datetime.now().strftime("%Y-%m")
total_spent = store.month_total(month_key)
remaining = custom_budget - total_spent

with col1:
  st.subheader("📊 1. Suivi des dépenses SQL")
  st.metric(
      label="Dépenses cumulées du mois",
      value=f"{total_spent:.4f} €",
      delta=f"{remaining:.2f} € restants (Budget: {custom_budget:.2f} €)",
      delta_color="normal" if remaining >= 0 else "inverse",
  )

with col2:
  st.subheader("🚨 3 & 4. Alertes Budget & Détection d'Anomalies")
  if total_spent > custom_budget:
    st.error(
        f"🚨 **Budget dépassé !** Dépenses ({total_spent:.2f} €) > Budget"
        f" ({custom_budget:.2f} €)"
    )
    st.write("👉 **Action :** Basculer immédiatement sur des instances Spot (-60%).")
  elif spike_detected:
    st.warning("⚠️ **Anomalie détectée :** Pic de consommation isolé identifié.")
    st.write("👉 **Action :** Vérifier la taille des lots (batch size) du job.")
  elif total_spent > 0:
    st.success("✅ Consommation régulière. Aucun pic anormal.")
    st.write("👉 Économie de -60 % disponible sur instances Spot.")
  else:
    st.info("Aucun dépassement de budget en cours.")

st.subheader("💡 2 & 5. Recommandations automatiques (Scale-to-Zero & Taille)")
if not df_records.empty:
  st.json({
      "Objective 2 (Scale-to-Zero)": (
          "Idle threshold > 2h -> Action: STOP idle container"
      ),
      "Objective 5 (Right-Sizing)": (
          "vCPU usage 0.3/2.0 -> Recommendation: Downgrade L to S"
      ),
  })
else:
  st.caption("Aucune recommandation émise pour le moment.")

# --- Section 3 : Module IA avec Persistance des Réponses ---
st.write("---")
st.header("🤖 Module IA LLM Local (Ollama / Phi-3)")

financial_context = {
    "monthly_budget": custom_budget,
    "month_total": total_spent,
    "spike_detected": spike_detected,
    "action": (
        "budget_exceeded"
        if total_spent > custom_budget
        else ("cost_spike_alert" if spike_detected else "cost_tracking")
    ),
    "recommendation": (
        "Passer sur des instances Spot (-60%)"
        if total_spent > custom_budget
        else (
            "Investiguer le pic de consommation atypique"
            if spike_detected
            else "Maintenir la configuration actuelle"
        )
    ),
}

col_summary, col_chat = st.columns(2)

with col_summary:
  st.subheader("📝 Synthèse Managériale Auto")
  if st.button("💬 Générer la synthèse globale", key="btn_summary"):
    with st.spinner(f"Analyse par Ollama ({ollama_model})..."):
      # Stockage permanent dans la session
      st.session_state["summary_report"] = explain_cost_with_ollama(
          financial_context, model_name=ollama_model
      )

  # Affichage persistant tant qu'un rapport existe
  if st.session_state["summary_report"]:
    with st.container(border=True):
      st.caption("📌 **Rapport Décisionnel FinOps** • Proposé par Phi-3")
      st.markdown(st.session_state["summary_report"])

with col_chat:
  st.subheader("❓ Posez une question sur les coûts")
  user_q = st.text_input(
      "Exemple : Pourquoi la dernière transaction est-elle si élevée ?",
      key="user_question",
  )
  if st.button("🤖 Obtenir une réponse", key="btn_ask"):
    if user_q.strip():
      with st.spinner("Réflexion de l'assistant..."):
        # Stockage permanent dans la session
        st.session_state["chat_response"] = ask_cost_assistant(
            financial_context, user_q, model_name=ollama_model
        )
    else:
      st.warning("Veuillez saisir une question.")

  # Affichage persistant de la réponse au chat
  if st.session_state["chat_response"]:
    with st.container(border=True):
      st.caption("💡 **Analyse Expert MLOps**")
      st.markdown(st.session_state["chat_response"])