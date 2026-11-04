"""Modèle défaillant pour la démonstration (scénario « Panne du service »).

C'est le vrai modèle de production, avec un bug : il plante sur les longs trajets.
Au calme, les sondes de l'agent Fiabilité (trajet court) passent : la version paraît
saine. Au pic de trafic, les longs trajets arrivent et l'API renvoie des erreurs 500.
Un redémarrage n'y change rien : seul un retour à la version précédente répare.

Ce module doit rester importable par l'API (le modèle enregistré dans MLflow y fait référence).
"""
from __future__ import annotations


class LongTripBug:
    def __init__(self, model, max_distance: float = 8.0):
        self.model = model
        self.max_distance = max_distance

    def predict(self, X):
        long_trips = [x for x in X if float(x.get("trip_distance", 0)) > self.max_distance]
        if long_trips:
            raise ValueError(f"bug de la version : trajet de {long_trips[0]['trip_distance']} km non géré "
                             f"(> {self.max_distance} km)")
        return self.model.predict(X)
