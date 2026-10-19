"""Pemuat konfigurasi YAML dan penyiapan logging."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"
USER_CONFIG_NAME = "user_config.yaml"
LOG_FILE_NAME = "app.log"

REQUIRED_KEYS: tuple[str, ...] = (
    "data.raw_dir",
    "data.csv_sep",
    "data.time_unit",
    "data.column_map",
    "data.min_session.fwd",
    "data.min_session.bwd",
    "data.exclude_traffic_types",
    "data.max_train_rows",
    "data.random_state",
    "data.voip_filter.mode",
    "data.voip_filter.label_value",
    "data.voip_filter.protocol",
    "data.voip_filter.sip_ports",
    "data.voip_filter.rtp_port_ranges",
    "data.voip_filter.min_flows_warning",
    "data.voip_filter.fallback_udp_all",
    "features.cluster_columns",
    "features.fai_clip_negative",
    "features.log_transform",
    "features.delay_column",
    "features.jitter_column",
    "evaluation.silhouette_bands",
    "evaluation.noise_ideal",
    "evaluation.noise_too_strict",
    "evaluation.target_silhouette",
    "evaluation.pca_sample_size",
    "eda.sensitivity_n",
    "eda.sensitivity_exclude",
    "eda.duration_cap_s",
    "eda.duration_cap_tol_ms",
    "hdbscan.min_cluster_size",
    "hdbscan.min_samples",
    "hdbscan.cluster_selection_method",
    "hdbscan.grid_search.enabled",
    "hdbscan.grid_search.min_cluster_size",
    "hdbscan.grid_search.min_samples",
    "hdbscan.grid_search.max_noise_ratio",
    "hdbscan.grid_search.silhouette_sample_size",
    "retrain.log_source.exclude_insufficient",
    "retrain.log_source.exclude_anomaly",
    "retrain.log_source.per_flow",
    "retrain.log_source.exclude_imputed",
    "retrain.log_source.feature_mode",
    "retrain.log_source.min_rows",
    "profiling.jitter_percentile",
    "profiling.max_outliers",
    "thresholds.delay_good_ms",
    "thresholds.delay_max_ms",
    "thresholds.jitter_max_ms",
    "thresholds.loss_max_pct",
    "monitoring.capture_backend",
    "monitoring.bpf_filter",
    "monitoring.live_poll_sec",
    "monitoring.window_sec",
    "monitoring.flow_idle_timeout_sec",
    "monitoring.feature_mode",
    "monitoring.min_session_live.fwd",
    "monitoring.min_session_live.bwd",
    "monitoring.min_packets_per_window",
    "monitoring.refresh_sec",
    "monitoring.replay_speed",
    "monitoring.kritis_ratio",
    "paths.artifacts_dir",
    "paths.reports_dir",
    "paths.logs_dir",
    "paths.recommendations_file",
)


class ConfigError(ValueError):
    """Konfigurasi tidak valid atau tidak lengkap."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gabungkan `override` ke salinan `base` secara rekursif (list ditimpa utuh)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_key(cfg: dict[str, Any], dotted: str) -> Any:
    """Ambil nilai dengan kunci bertitik, mis. `thresholds.delay_good_ms`."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def validate_config(cfg: dict[str, Any]) -> None:
    """Pastikan semua kunci wajib tersedia; jika tidak, lempar ConfigError."""
    missing = []
    for dotted in REQUIRED_KEYS:
        try:
            get_key(cfg, dotted)
        except KeyError:
            missing.append(dotted)
    if missing:
        raise ConfigError("Kunci konfigurasi wajib tidak ditemukan: " + ", ".join(missing))


def _read_yaml(path: Path) -> dict[str, Any]:
    """Baca satu berkas YAML sebagai dict."""
    if not path.is_file():
        raise ConfigError(f"Berkas konfigurasi tidak ditemukan: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ConfigError(f"Isi konfigurasi harus berupa mapping: {path}")
    return content


def load_config(
    path: str | Path | None = None,
    user_path: str | Path | None = None,
    use_user: bool = True,
) -> dict[str, Any]:
    """Muat konfigurasi default lalu timpa dengan konfigurasi pengguna bila ada.

    Tanpa `user_path`, berkas `user_config.yaml` di folder yang sama dipakai jika ada.
    """
    base_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    cfg = _read_yaml(base_path)
    if use_user:
        override_path = Path(user_path) if user_path is not None else base_path.parent / USER_CONFIG_NAME
        if user_path is not None or override_path.is_file():
            cfg = deep_merge(cfg, _read_yaml(override_path))
    validate_config(cfg)
    return cfg


def resolve_path(value: str | Path) -> Path:
    """Ubah path relatif menjadi absolut terhadap akar proyek."""
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logging(cfg: dict[str, Any], level: int = logging.INFO) -> Path:
    """Arahkan logger `qos` ke `<logs_dir>/app.log`; aman dipanggil berulang."""
    log_dir = resolve_path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE_NAME
    logger = logging.getLogger("qos")
    logger.setLevel(level)
    already = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers
    )
    if not already:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return log_path
