"""Uji skema fixture 10 flow nyata (dibuat oleh scripts/make_sample_fixture.py)."""
from pathlib import Path

import pandas as pd
import pytest

from qos.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_real_10.csv"
COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Flow IAT Mean", "Flow IAT Std", "cluster",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIXTURE)


def test_skema_dan_jumlah(df):
    assert list(df.columns) == COLUMNS
    assert len(df) == 10
    assert not df.isna().any().any()


def test_mencakup_noise_dan_banyak_klaster(df):
    assert (df["cluster"] == -1).any()
    assert df.loc[df["cluster"] != -1, "cluster"].nunique() >= 4


def test_memenuhi_filter_sesi_final(df):
    ms = load_config(use_user=False)["data"]["min_session"]
    assert (df["Total Fwd Packets"] >= ms["fwd"]).all()
    assert (df["Total Bwd Packets"] >= ms["bwd"]).all()
    assert (df["Flow IAT Mean"] > 0).all() and (df["Flow IAT Std"] >= 0).all()
