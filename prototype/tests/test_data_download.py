"""Téléchargement des données : résiste aux coupures DNS passagères."""
from urllib.error import URLError

import pandas as pd
import pytest

from pipeline.data import download


def test_reessaie_apres_une_erreur_reseau(tmp_path):
    calls, waits = [], []

    def reader(url):
        calls.append(url)
        if len(calls) < 3:
            raise URLError("No address associated with hostname")
        return pd.DataFrame({"a": [1, 2]})

    download("https://exemple/mois.parquet", tmp_path / "m.parquet", reader=reader, sleep=waits.append)
    assert len(calls) == 3 and waits == [3, 6]
    assert pd.read_parquet(tmp_path / "m.parquet")["a"].tolist() == [1, 2]
    assert not (tmp_path / "m.part").exists()


def test_abandonne_apres_plusieurs_essais(tmp_path):
    def reader(url):
        raise URLError("pas de réseau")

    with pytest.raises(URLError):
        download("https://exemple/mois.parquet", tmp_path / "m.parquet", attempts=2, reader=reader, sleep=lambda s: None)
    assert not (tmp_path / "m.parquet").exists()
