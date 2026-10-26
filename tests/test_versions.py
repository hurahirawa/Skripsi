"""Uji versi artefak klasterisasi murni: aktivasi, penolakan versi lama (pelabelan QoS), dan Perbarui Deskripsi."""
import json
import time

import pytest

from conftest import synthetic_flows
from qos.model.train import save_artifacts, train
from qos.model.versions import (
    LEGACY_MESSAGE,
    VersionError,
    activate_version,
    active_version,
    description_changes,
    is_complete,
    is_legacy,
    list_versions,
    redescribe_profile,
    refresh_descriptions,
    validate_thresholds,
)


def _legacy_version(cfg, name="20200101_000000"):
    """Folder versi dengan format lama: tanpa penanda `schema`, ada label_map QoS."""
    from qos.config import resolve_path

    folder = resolve_path(cfg["paths"]["artifacts_dir"]) / name
    folder.mkdir(parents=True)
    for f in ("hdbscan_model.joblib", "scaler.joblib", "pca.joblib"):
        (folder / f).write_bytes(b"x")
    profile = {"model_version": name, "evaluation": {"n_clusters": 2, "silhouette": 0.5, "chi": 10.0, "noise_ratio": 0.1},
               "clusters": [], "label_map": {"0": "QoS Baik"}}
    (folder / "cluster_profile.json").write_text(json.dumps(profile), "utf-8")
    (folder / "train_meta.json").write_text(json.dumps({"timestamp": "2020-01-01T00:00:00"}), "utf-8")
    return folder


def test_validate_thresholds():
    ok = {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 50, "loss_good_pct": 1, "loss_max_pct": 3}
    assert validate_thresholds(ok)["delay_max_ms"] == 400.0
    with pytest.raises(VersionError, match="tidak lengkap"):
        validate_thresholds({"delay_good_ms": 1})
    with pytest.raises(VersionError, match="delay_good_ms"):
        validate_thresholds({**ok, "delay_good_ms": 500})
    with pytest.raises(VersionError, match="loss_good_pct"):
        validate_thresholds({**ok, "loss_good_pct": 5})
    with pytest.raises(VersionError, match="negatif"):
        validate_thresholds({**ok, "jitter_max_ms": -1})


def test_versi_lama_ditolak_saat_aktivasi_dengan_pesan_jelas(active_cfg):
    folder = _legacy_version(active_cfg)
    assert is_legacy(folder) and not is_complete(folder)
    before = active_version(active_cfg)
    with pytest.raises(VersionError, match="pelabelan QoS") as exc:
        activate_version(active_cfg, folder.name)
    assert str(exc.value) == LEGACY_MESSAGE.format(v=folder.name)
    assert active_version(active_cfg) == before  # current.txt tidak berubah


def test_list_versions_menandai_kompatibilitas(active_cfg):
    legacy = _legacy_version(active_cfg)
    table = list_versions(active_cfg).set_index("versi")
    assert bool(table.loc[active_cfg["_active"], "kompatibel"]) and bool(table.loc[active_cfg["_active"], "aktif"])
    assert not bool(table.loc[legacy.name, "kompatibel"]) and not bool(table.loc[legacy.name, "aktif"])
    assert {"dbcv", "silhouette", "klaster", "noise_ratio"} <= set(table.columns)


def test_aktivasi_versi_baru_dan_tolak_saat_monitoring(active_cfg):
    time.sleep(1.1)
    new = save_artifacts(train(synthetic_flows(1), active_cfg), active_cfg, activate=False)
    with pytest.raises(VersionError, match="monitoring"):
        activate_version(active_cfg, new.name, monitoring_active=True)
    assert active_version(active_cfg) == active_cfg["_active"]
    activate_version(active_cfg, new.name)
    assert active_version(active_cfg) == new.name
    with pytest.raises(VersionError, match="tidak ditemukan"):
        activate_version(active_cfg, "tidak_ada")


def test_versi_tidak_lengkap_ditolak(active_cfg):
    from qos.config import resolve_path

    folder = resolve_path(active_cfg["paths"]["artifacts_dir"]) / active_cfg["_active"]
    time.sleep(1.1)
    new = save_artifacts(train(synthetic_flows(1), active_cfg), active_cfg, activate=False)
    (new / "assignment.json").unlink()
    assert is_complete(folder) and not is_complete(new)
    with pytest.raises(VersionError, match="assignment.json"):
        activate_version(active_cfg, new.name)


def test_perbarui_deskripsi_tanpa_refit_membuat_versi_baru(active_cfg):
    from qos.config import resolve_path

    root = resolve_path(active_cfg["paths"]["artifacts_dir"])
    src = root / active_cfg["_active"]
    old = json.loads((src / "cluster_profile.json").read_text("utf-8"))
    new_th = {**old["thresholds"], "delay_good_ms": 1.0, "delay_max_ms": 2.0}
    time.sleep(1.1)
    out = refresh_descriptions(active_cfg, src.name, new_th)
    assert out.name != src.name and active_version(active_cfg) == src.name  # aktif tidak berubah
    new = json.loads((out / "cluster_profile.json").read_text("utf-8"))
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert meta["deskripsi_dari"] == src.name and new["thresholds"]["delay_max_ms"] == 2.0
    for name in ("hdbscan_model.joblib", "scaler.joblib", "assignment.json", "kmeans_n.joblib"):
        assert (out / name).read_bytes() == (src / name).read_bytes()  # tanpa refit: model & centroid identik
    for a, b in zip(old["clusters"], new["clusters"]):
        assert a["statistik"] == b["statistik"] and a["centroid_asli"] == b["centroid_asli"]
    assert any("delay" in c["metrik_melewati_ambang"] for c in new["clusters"] if c["id_klaster"] != -1)
    assert description_changes(old, new)["berubah"].any()
    assert is_complete(out)


def test_redescribe_profile_murni_dan_tolak_ambang_tak_valid(active_cfg):
    from qos.config import resolve_path

    src = resolve_path(active_cfg["paths"]["artifacts_dir"]) / active_cfg["_active"]
    old = json.loads((src / "cluster_profile.json").read_text("utf-8"))
    same = redescribe_profile(old, old["thresholds"])
    assert not description_changes(old, same)["berubah"].any()
    with pytest.raises(VersionError):
        redescribe_profile(old, {**old["thresholds"], "delay_good_ms": 999})
