"""Uji tingkat ITU per metrik dan per flow."""
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.profiling.labeling import flow_levels, level_delay, level_jitter, level_loss

TH = load_config(use_user=False)["thresholds"]


def test_level_delay_batas():
    d = pd.Series([0.0, 150.0, 150.01, 400.0, 400.01, np.nan])
    out = level_delay(d, TH)
    assert out.iloc[:5].tolist() == [1, 1, 2, 2, 3]
    assert np.isnan(out.iloc[5])


def test_level_jitter_dan_loss_batas():
    assert level_jitter(pd.Series([50.0, 50.01]), TH).tolist() == [1, 3]
    assert level_loss(pd.Series([0.0, 1.0, 1.01, 3.0, 3.01]), TH).tolist() == [1, 1, 2, 2, 3]
    assert np.isnan(level_loss(pd.Series([np.nan]), TH).iloc[0])


def test_flow_levels_total_max():
    lv = flow_levels(pd.Series([100.0, 200.0, 500.0]), pd.Series([1.0, 1.0, 60.0]), pd.Series([0.0, 5.0, 0.0]), TH)
    assert lv["total"].tolist() == [1, 3, 3]
    assert lv.loc[1, "delay"] == 2 and lv.loc[1, "loss"] == 3


def test_flow_levels_nan_menjalar():
    lv = flow_levels(pd.Series([np.nan]), pd.Series([1.0]), pd.Series([0.0]), TH)
    assert np.isnan(lv.loc[0, "total"])


# Tabel 2.3 naskah (L_delay, L_jitter, L_loss, T, label)
TABEL_2_3 = [
    (1, 1, 1, 1, "QoS Baik"),
    (2, 1, 1, 2, "QoS Cukup - Delay Meningkat"),
    (3, 1, 3, 3, "QoS Buruk - Delay dan Packet Loss Melebihi Batas"),
    (1, 3, 1, 3, "QoS Buruk - Jitter Melebihi Batas"),
    (1, 1, 2, 2, "QoS Cukup - Packet Loss Meningkat"),
]


@pytest.mark.parametrize("d,j,l,t,label", TABEL_2_3)
def test_tabel_2_3_naskah(d, j, l, t, label):
    from qos.profiling.labeling import label_cluster

    levels = {"delay": d, "jitter": j, "loss": l}
    assert max(levels.values()) == t
    assert label_cluster(levels)[0] == label


@pytest.mark.parametrize("d,j,l", [(1, 1, 1), (2, 1, 1), (3, 1, 3), (1, 3, 1), (1, 1, 2)])
def test_noise_tabel_2_3_tanpa_tingkat(d, j, l):
    from qos.profiling.labeling import label_cluster

    label = label_cluster({"delay": d, "jitter": j, "loss": l}, noise=True)[0]
    assert label.startswith("Anomali")
    assert "QoS" not in label and "Meningkat" not in label and "Melebihi" not in label


def test_profil_lama_tanpa_loss_good_pakai_dua_level():
    old = {k: v for k, v in TH.items() if k != "loss_good_pct"}
    assert level_loss(pd.Series([0.0, old["loss_max_pct"], old["loss_max_pct"] + 0.01]), old).tolist() == [1, 1, 3]
