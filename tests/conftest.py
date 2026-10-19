"""Fixture bersama: config terisolasi dan model sintetis aktif (dipakai uji M8)."""
import copy

import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.model.train import save_artifacts, train


def synthetic_flows(seed: int = 0, n: int = 80) -> pd.DataFrame:
    """Tiga kelompok flow sintetis (kolom mentah iat_mean_us, iat_std_us, fwd, bwd)."""
    rng = np.random.default_rng(seed)
    parts = []
    for mean_us, std_us, pkts, ratio in ((1000.0, 500.0, 100, 0.9), (10_000.0, 8_000.0, 300, 1.0), (400_000.0, 90_000.0, 800, 0.95)):
        fwd = rng.normal(pkts, pkts * 0.02, n).round()
        parts.append(
            pd.DataFrame(
                {
                    "iat_mean_us": rng.normal(mean_us, mean_us * 0.03, n),
                    "iat_std_us": rng.normal(std_us, std_us * 0.03, n),
                    "fwd_packets_count": fwd,
                    "bwd_packets_count": (fwd * rng.normal(ratio, 0.01, n)).round(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def isolated_cfg(root) -> dict:
    """Config default dengan folder artefak/laporan/log di `root` dan grid kecil."""
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"].update(
        {"artifacts_dir": str(root / "artifacts"), "reports_dir": str(root / "reports"), "logs_dir": str(root / "logs")}
    )
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
    return cfg


@pytest.fixture
def iso_cfg(tmp_path):
    """Config terisolasi tanpa model."""
    return isolated_cfg(tmp_path)


@pytest.fixture
def active_cfg(iso_cfg):
    """Config terisolasi dengan satu model sintetis yang aktif; `iso_cfg['_active']` = nama versi."""
    out = save_artifacts(train(synthetic_flows(), iso_cfg), iso_cfg, activate=True)
    iso_cfg["_active"] = out.name
    return iso_cfg
