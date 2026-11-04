"""Lancement : python -m agents.cost (depuis prototype/).

Écoute cost_estimate et training_completed sur Redis, et fait tourner la
boucle de fond (coût de l'API servie, scale-to-zero, taille du serveur) :
voir service.py. Les coûts sont gardés dans Postgres (table model_costs)."""
from agents.cost.service import main

main()
