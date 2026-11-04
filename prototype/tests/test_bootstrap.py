"""Premier démarrage (service init) : un modèle seulement s'il n'y en a pas."""
from pipeline import bootstrap


def test_rien_a_faire_si_un_modele_est_en_production(monkeypatch):
    trained = []
    monkeypatch.setattr(bootstrap, "wait_for_mlflow", lambda: object())
    monkeypatch.setattr(bootstrap, "has_production_model", lambda client: True)
    monkeypatch.setattr(bootstrap, "train", lambda *a, **k: trained.append(a))
    bootstrap.main()
    assert trained == []


def test_premier_modele_entraine_puis_api_rechargee(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "wait_for_mlflow", lambda: object())
    monkeypatch.setattr(bootstrap, "has_production_model", lambda client: False)
    monkeypatch.setattr(bootstrap, "train", lambda *a, **k: calls.append(("train", a, k)) or
                        {"version": "1", "rmse": 5.3})
    monkeypatch.setattr(bootstrap, "reload_api", lambda: calls.append(("reload",)) or True)
    bootstrap.main()
    assert calls == [("train", (2023, 1), {"promote": True}), ("reload",)]
