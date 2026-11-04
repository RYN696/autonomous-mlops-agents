import json
import requests


def _call_ollama(
    prompt: str,
    model_name: str = "phi3",
    endpoint: str = "http://localhost:11434/api/generate",
) -> str:
  """Envoie un prompt à l'API locale d'Ollama et retourne la réponse générée."""
  try:
    response = requests.post(
        endpoint,
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()
  except Exception as e:
    return (
        f"❌ Erreur lors de la communication avec Ollama ({model_name}) :"
        f" {str(e)}"
    )


def explain_cost_with_ollama(
    financial_context: dict, model_name: str = "phi3"
) -> str:
  """Génère une synthèse exécutive FinOps structurée en Markdown."""
  prompt = f"""
Tu es un expert FinOps & MLOps Senior. Rédige une synthèse décisionnelle claire, structurée et professionnelle pour le Management / CFO.

DONNÉES DU SYSTÈME :
- Budget mensuel alloué : {financial_context.get('monthly_budget')} €
- Dépenses cumulées actuelles : {financial_context.get('month_total', 0):.2f} €
- Pic de consommation atypique : {"DÉTECTÉ 🚨" if financial_context.get('spike_detected') else "Aucun"}
- Statut opérationnel : {financial_context.get('action')}
- Recommandation technique : {financial_context.get('recommendation')}

RÉDIGE TA RÉPONSE EN RESPECTANT EXACTEMENT CE FORMAT MARKDOWN :

### 📊 Synthèse Exécutive
[Une analyse concise de 2 phrases sur l'état global du budget et de la consommation]

### 🚨 Diagnostic & Analyse d'Anomalie
[Explication technique FinOps/MLOps : impact des jobs d'entraînement, ressources CPU/GPU ou volumes de données]

### 🎯 Plan d'Action Recommandé
- **Action Immédiate :** [Action corrective directe]
- **Optimisation Continue :** [Recommandation type instances Spot (-60%), Scale-to-Zero ou Right-sizing]
"""
  return _call_ollama(prompt, model_name=model_name)


def ask_cost_assistant(
    financial_context: dict, user_question: str, model_name: str = "phi3"
) -> str:
  """Répond aux questions interactives de l'utilisateur sur la gestion des coûts."""
  prompt = f"""
Tu es un assistant expert MLOps et FinOps. Réponds de manière concise, précise et professionnelle à la question de l'utilisateur.

CONTEXTE FINOPS :
- Budget : {financial_context.get('monthly_budget')} €
- Total dépense : {financial_context.get('month_total', 0):.2f} €
- Pic atypique : {"Oui (Dépense > 50% de la moyenne)" if financial_context.get('spike_detected') else "Non"}

QUESTION DE L'UTILISATEUR :
"{user_question}"

RÉPONSE DÉTAILLÉE :
"""
  return _call_ollama(prompt, model_name=model_name)