import json
import urllib.request


def _call_ollama(prompt: str, model_name: str = "phi3") -> str:
  """Fonction utilitaire privée pour appeler l'API HTTP d'Ollama local."""
  url = "http://localhost:11434/api/generate"
  payload = {"model": model_name, "prompt": prompt, "stream": False}

  try:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
      res_data = json.loads(response.read().decode("utf-8"))
      return res_data.get("response", "Aucune réponse générée.")
  except Exception as e:
    return f"⚠️ Impossible de contacter Ollama (http://localhost:11434). Vérifiez qu'Ollama tourne (`ollama serve`). Erreur : {e}"


def explain_cost_with_ollama(data: dict, model_name: str = "phi3") -> str:
  """Génère une synthèse managériale globale."""
  prompt = f"""
    En tant qu'expert MLOps, rédige un court résumé exécutif (2 à 3 phrases) pour le management basé sur ces métriques financières :
    - Budget mensuel : {data.get('monthly_budget', 20.0)} €
    - Dépenses cumulées : {data.get('month_total', 0.0)} €
    - Action : {data.get('action', 'budget_check')}
    - Recommandation : {data.get('recommendation', 'Aucune')}

    Sois concis, professionnel et donne une perspective claire sur l'impact budgétaire en français.
    """
  return _call_ollama(prompt, model_name=model_name)


def ask_cost_assistant(
    data: dict, user_query: str, model_name: str = "phi3"
) -> str:
  """Répond à une question spécifique posée par l'utilisateur."""
  prompt = f"""
    Tu es un assistant FinOps MLOps. Voici le contexte financier actuel :
    - Budget mensuel alloué : {data.get('monthly_budget', 20.0)} €
    - Dépenses cumulées actuelles : {data.get('month_total', 0.0)} €
    - Statut du budget : {data.get('action', 'N/A')}
    - Recommandation en cours : {data.get('recommendation', 'N/A')}

    Réponds de façon synthétique, professionnelle et précise en français à la question suivante :
    "{user_query}"
    """
  return _call_ollama(prompt, model_name=model_name)