"""Uji pemuat konfigurasi."""
import logging

import pytest
import yaml

from qos.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    deep_merge,
    get_key,
    load_config,
    resolve_path,
    setup_logging,
    validate_config,
)


def test_default_config_termuat():
    cfg = load_config(use_user=False)
    assert cfg["thresholds"] == {
        "delay_good_ms": 150.0,
        "delay_max_ms": 400.0,
        "jitter_max_ms": 50.0,
        "loss_max_pct": 0.1,
    }
    assert cfg["hdbscan"]["min_cluster_size"] == 5
    assert cfg["hdbscan"]["min_samples"] == 2
    assert cfg["data"]["voip_filter"]["sip_ports"] == [5060, 5061]
    assert cfg["features"]["cluster_columns"] == [
        "iat_mean_us",
        "iat_std_us",
        "fai_pct",
    ]
    assert cfg["features"]["log_transform"] == ["iat_mean_us", "iat_std_us"]  # fai_pct tidak di-log
    assert set(cfg["data"]["column_map"]) == {"cic", "darknet", "bccc"}


def test_deep_merge_tidak_mengubah_base():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    merged = deep_merge(base, {"a": {"y": 9}, "b": [3]})
    assert merged == {"a": {"x": 1, "y": 9}, "b": [3]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}


def test_get_key():
    cfg = {"a": {"b": {"c": 3}}}
    assert get_key(cfg, "a.b.c") == 3
    with pytest.raises(KeyError):
        get_key(cfg, "a.z")


def test_override_pengguna_tergabung(tmp_path):
    user = tmp_path / "user_config.yaml"
    user.write_text(yaml.safe_dump({"thresholds": {"delay_good_ms": 100.0}}), encoding="utf-8")
    cfg = load_config(user_path=user)
    assert cfg["thresholds"]["delay_good_ms"] == 100.0
    assert cfg["thresholds"]["delay_max_ms"] == 400.0


def test_user_config_otomatis_di_folder_yang_sama(tmp_path):
    base = tmp_path / "default_config.yaml"
    base.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "user_config.yaml").write_text(
        yaml.safe_dump({"monitoring": {"window_sec": 10}}), encoding="utf-8"
    )
    assert load_config(base)["monitoring"]["window_sec"] == 10
    assert load_config(base, use_user=False)["monitoring"]["window_sec"] == 5


def test_kunci_wajib_hilang(tmp_path):
    cfg = load_config(use_user=False)
    del cfg["thresholds"]["jitter_max_ms"]
    with pytest.raises(ConfigError, match="thresholds.jitter_max_ms"):
        validate_config(cfg)
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="thresholds.jitter_max_ms"):
        load_config(broken, use_user=False)


def test_berkas_tidak_ada(tmp_path):
    with pytest.raises(ConfigError, match="tidak ditemukan"):
        load_config(tmp_path / "tidak_ada.yaml")


def test_resolve_path(tmp_path):
    assert resolve_path(tmp_path) == tmp_path
    assert resolve_path("config/default_config.yaml") == DEFAULT_CONFIG_PATH


def test_setup_logging_idempoten(tmp_path):
    cfg = load_config(use_user=False)
    cfg["paths"]["logs_dir"] = str(tmp_path / "logs")
    logger = logging.getLogger("qos")
    before = list(logger.handlers)
    try:
        log_path = setup_logging(cfg)
        setup_logging(cfg)
        added = [h for h in logger.handlers if h not in before]
        assert len(added) == 1
        logger.info("uji")
        added[0].flush()
        assert "uji" in log_path.read_text(encoding="utf-8")
    finally:
        for h in [h for h in logger.handlers if h not in before]:
            logger.removeHandler(h)
            h.close()
