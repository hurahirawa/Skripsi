"""Uji saran berdasarkan karakteristik klaster (PRD FR-REC): pemicu = metrik klaster melewati ambang ITU / di luar klaster."""
import copy

import pandas as pd
import pytest

from qos.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from qos.monitor.calls import INSUFFICIENT, SIGNALING, STATUS_CLUSTERED, STATUS_OUTSIDE
from qos.recommend.rules import (
    RulesError,
    call_recommendations,
    cluster_metric_map,
    cluster_recommendations,
    load_rules,
    recommend_metrics,
    recommend_outside,
    recommendations_enabled,
    session_recommendations,
)

RULES = load_rules(PROJECT_ROOT / "config" / "recommendations.yaml")
SARAN = {k: RULES[k]["saran"] for k in RULES}
METRIC_MAP = {1: [], 2: ["delay", "loss"], 3: ["jitter"]}


def test_aturan_sesuai_tabel_prd():
    assert RULES["delay"]["saran"] == [
        "Perlebar bandwidth jalur VoIP", "Prioritas antrean DiffServ (EF)", "Evaluasi beban router dan jalur routing"
    ]
    assert RULES["jitter"]["saran"][0] == "Priority queuing untuk RTP"
    assert RULES["di_luar_klaster"]["kategori"] == "Investigasi Kasus Tunggal"
    assert all(RULES[k]["kategori"] == "Perbaikan Kapasitas/Konfigurasi" for k in ("delay", "jitter", "loss"))
    assert "anomali" not in RULES  # tidak ada lagi konsep Anomali


def test_tanpa_metrik_melewati_ambang_tidak_ada_saran():
    assert recommend_metrics([], RULES, "Klaster 1") is None


def test_satu_dan_beberapa_metrik_menggabungkan_saran_berurutan():
    rec = recommend_metrics(["jitter"], RULES, "Klaster 3")
    assert rec.triggers == (RULES["jitter"]["pemicu"],) and list(rec.suggestions) == SARAN["jitter"]
    rec2 = recommend_metrics(["loss", "delay"], RULES, "Klaster 2")  # urutan tetap delay, jitter, loss
    assert rec2.triggers == (RULES["delay"]["pemicu"], RULES["loss"]["pemicu"])
    assert list(rec2.suggestions) == SARAN["delay"] + SARAN["loss"]
    assert list(recommend_metrics(["delay", "jitter", "loss"], RULES, "k").suggestions) == SARAN["delay"] + SARAN["jitter"] + SARAN["loss"]


def test_saran_duplikat_dibuang():
    rules = {k: dict(v) for k, v in RULES.items()}
    rules["jitter"]["saran"] = [rules["delay"]["saran"][0], "Saran unik jitter"]
    rec = recommend_metrics(["delay", "jitter"], rules, "k")
    assert list(rec.suggestions).count(rules["delay"]["saran"][0]) == 1


def test_di_luar_klaster_saran_investigasi_kasus_tunggal():
    rec = recommend_outside(RULES)
    assert rec.key == "Di luar klaster" and rec.category == "Investigasi Kasus Tunggal"
    assert list(rec.suggestions) == SARAN["di_luar_klaster"]


def test_cluster_metric_map_dan_cluster_recommendations_dari_profil():
    profile = [
        {"id_klaster": -1, "nama": "Noise", "jumlah_anggota": 3, "metrik_melewati_ambang": ["delay"]},
        {"id_klaster": 1, "nama": "Klaster 1", "jumlah_anggota": 5, "metrik_melewati_ambang": []},
        {"id_klaster": 2, "nama": "Klaster 2", "jumlah_anggota": 7, "metrik_melewati_ambang": ["loss"],
         "deskripsi": {"ringkas": "Packet Loss >3 %"}},
    ]
    assert cluster_metric_map(profile) == {-1: ["delay"], 1: [], 2: ["loss"]}
    recs = cluster_recommendations(profile, RULES)
    assert [r["id_klaster"] for r in recs] == [-1, 2]  # klaster tanpa metrik melewati ambang tidak disarankan
    assert recs[0]["saran"] == SARAN["di_luar_klaster"] and recs[1]["saran"] == SARAN["loss"] and recs[1]["jumlah_anggota"] == 7
    assert recs[1]["deskripsi"] == "Packet Loss >3 %"


def test_session_recommendations_status_terakhir_dan_urutan():
    rows = pd.DataFrame(
        {
            "flow_id": ["a", "a", "b", "c", "d", "e", "f", "g"],
            "status": [STATUS_OUTSIDE, STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_OUTSIDE, INSUFFICIENT, SIGNALING,
                       STATUS_CLUSTERED],
            "klaster_terdekat": [1, 3, 3, 1, 2, pd.NA, pd.NA, 2],
        }
    )
    recs = session_recommendations(rows, METRIC_MAP, RULES)
    # a: terakhir Klaster 3 (jitter), b: Klaster 3 (jitter), c: Klaster 1 (tanpa metrik), d: di luar klaster, g: Klaster 2 (delay, loss)
    assert [(r["kunci"], r["jumlah_flow"]) for r in recs] == [("Klaster 3", 2), ("Di luar klaster", 1), ("Klaster 2", 1)]
    assert recs[0]["saran"] == SARAN["jitter"] and recs[1]["saran"] == SARAN["di_luar_klaster"]
    assert recs[2]["saran"] == SARAN["delay"] + SARAN["loss"]
    assert session_recommendations(pd.DataFrame(columns=["flow_id", "status", "klaster_terdekat"]), METRIC_MAP, RULES) == []


def test_session_recommendations_pulih_tidak_lagi_di_luar_klaster():
    rows = pd.DataFrame({"flow_id": ["a", "a"], "status": [STATUS_OUTSIDE, STATUS_CLUSTERED], "klaster_terdekat": [2, 1]})
    assert session_recommendations(rows, METRIC_MAP, RULES) == []  # status terakhir: Klaster 1 tanpa metrik melewati ambang


def test_call_recommendations_per_panggilan():
    calls = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c", "d"],
            "status_akhir": [STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_OUTSIDE, INSUFFICIENT],
            "klaster_akhir": [1, 2, 3, pd.NA],
        }
    )
    recs = call_recommendations(calls, METRIC_MAP, RULES)
    assert [(r["flow_id"], r["kunci"]) for r in recs] == [("b", "Klaster 2"), ("c", "Di luar klaster")]
    assert recs[1]["kategori"] == "Investigasi Kasus Tunggal"


def test_load_rules_validasi(tmp_path):
    with pytest.raises(RulesError, match="tidak ditemukan"):
        load_rules(tmp_path / "tidak_ada.yaml")
    bad = tmp_path / "r.yaml"
    bad.write_text("rules:\n  delay: {pemicu: x, kategori: y, saran: [a]}\n", encoding="utf-8")
    with pytest.raises(RulesError, match="jitter"):
        load_rules(bad)
    bad.write_text(
        "rules:\n" + "".join(f"  {k}: {{pemicu: x, kategori: y, saran: teks}}\n" for k in ("delay", "jitter", "loss", "di_luar_klaster")),
        encoding="utf-8",
    )
    with pytest.raises(RulesError, match="daftar"):
        load_rules(bad)


def test_recommendations_enabled_dari_config():
    cfg = load_config(use_user=False)
    assert cfg["recommendations"]["enabled"] is True and recommendations_enabled(cfg)
    off = copy.deepcopy(cfg)
    off["recommendations"]["enabled"] = False
    assert not recommendations_enabled(off)
    assert recommendations_enabled({})  # tanpa blok config: default aktif


def test_default_capture_backend_scapy_dan_bpf_tanpa_trafik_di_luar_distribusi():
    cfg = load_config(DEFAULT_CONFIG_PATH, use_user=False)
    assert cfg["monitoring"]["capture_backend"] == "scapy"
    bpf = cfg["monitoring"]["bpf_filter"]
    assert bpf.startswith("udp")
    for port in (443, 53, 123, 5353, 1900):
        assert f"not port {port}" in bpf
    assert cfg["paths"]["recommendations_file"] == "config/recommendations.yaml"
