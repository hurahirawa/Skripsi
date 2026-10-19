"""Uji rekomendasi rule-based (PRD FR-REC) dan konfigurasi default M7."""
import itertools

import pandas as pd
import pytest

from qos.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from qos.profiling.labeling import ACTIONS, label_cluster
from qos.recommend.rules import (
    RulesError,
    cluster_recommendations,
    load_rules,
    metrics_from_label,
    recommend,
    session_recommendations,
)

RULES = load_rules(PROJECT_ROOT / "config" / "recommendations.yaml")
SARAN = {k: RULES[k]["saran"] for k in RULES}


def test_aturan_sesuai_tabel_prd():
    assert RULES["delay"]["pemicu"] == "Delay Melebihi Batas"
    assert RULES["delay"]["saran"] == [
        "Perlebar bandwidth jalur VoIP", "Prioritas antrean DiffServ (EF)", "Evaluasi beban router dan jalur routing"
    ]
    assert RULES["jitter"]["saran"][0] == "Priority queuing untuk RTP"
    assert RULES["loss"]["pemicu"] == "Packet Loss Melebihi Batas"
    assert RULES["anomali"]["kategori"] == "Investigasi Kasus Tunggal"
    assert all(RULES[k]["kategori"] == "Perbaikan Kapasitas/Konfigurasi" for k in ("delay", "jitter", "loss"))


def test_kategori_aturan_sama_dengan_kategori_tindakan_label():
    assert RULES["delay"]["kategori"] == ACTIONS["QoS Buruk"]
    assert RULES["anomali"]["kategori"] == ACTIONS["Anomali"]


@pytest.mark.parametrize("label", ["QoS Baik", "QoS Cukup - Delay Meningkat", "Data Belum Cukup", ""])
def test_tidak_ada_rekomendasi_untuk_baik_cukup_dan_data_belum_cukup(label):
    assert recommend(label, RULES) is None


def test_satu_metrik_penyebab():
    rec = recommend("QoS Buruk - Jitter Melebihi Batas", RULES)
    assert rec.triggers == ("Jitter Melebihi Batas",)
    assert list(rec.suggestions) == SARAN["jitter"]
    assert rec.category == "Perbaikan Kapasitas/Konfigurasi"


def test_beberapa_metrik_menggabungkan_saran_berurutan():
    rec = recommend("QoS Buruk - Delay dan Packet Loss Melebihi Batas", RULES)
    assert rec.triggers == ("Delay Melebihi Batas", "Packet Loss Melebihi Batas")
    assert list(rec.suggestions) == SARAN["delay"] + SARAN["loss"]
    rec3 = recommend("QoS Buruk - Delay, Jitter dan Packet Loss Melebihi Batas", RULES)
    assert list(rec3.suggestions) == SARAN["delay"] + SARAN["jitter"] + SARAN["loss"]


def test_anomali_memakai_saran_anomali_apa_pun_metriknya():
    for label in ("Anomali", "Anomali - Delay Ekstrem", "Anomali - Delay, Jitter dan Packet Loss Ekstrem"):
        rec = recommend(label, RULES)
        assert list(rec.suggestions) == SARAN["anomali"] and rec.category == "Investigasi Kasus Tunggal"


def test_saran_duplikat_dibuang():
    rules = {k: dict(v) for k, v in RULES.items()}
    rules["jitter"]["saran"] = [rules["delay"]["saran"][0], "Saran unik jitter"]
    rec = recommend("QoS Buruk - Delay dan Jitter Melebihi Batas", rules)
    assert list(rec.suggestions).count(rules["delay"]["saran"][0]) == 1


def test_metrics_from_label_konsisten_dengan_label_cluster():
    """Semua kombinasi level: metrik yang di-parse = metrik berlevel 3 (Buruk) sesuai label_cluster."""
    for d, j, l in itertools.product((1, 2, 3), (1, 3), (1, 3)):
        levels = {"delay": d, "jitter": j, "loss": l}
        label, _ = label_cluster(levels)
        if max(levels.values()) == 3:
            assert metrics_from_label(label) == [m for m in ("delay", "jitter", "loss") if levels[m] == 3]
            assert recommend(label, RULES) is not None
        else:
            assert recommend(label, RULES) is None


def test_session_recommendations_label_terakhir_dan_urutan():
    rows = pd.DataFrame(
        {
            "flow_id": ["a", "a", "b", "c", "d", "e"],
            "label": ["QoS Baik", "QoS Buruk - Jitter Melebihi Batas", "QoS Buruk - Jitter Melebihi Batas",
                      "Anomali", "Data Belum Cukup", "QoS Buruk - Jitter Melebihi Batas"],
        }
    )
    recs = session_recommendations(rows, RULES)
    assert [(r["label"], r["jumlah_flow"]) for r in recs] == [
        ("QoS Buruk - Jitter Melebihi Batas", 3), ("Anomali", 1)
    ]
    assert recs[0]["saran"] == SARAN["jitter"]
    assert session_recommendations(pd.DataFrame(columns=["flow_id", "label"]), RULES) == []


def test_session_recommendations_label_terakhir_menentukan():
    rows = pd.DataFrame({"flow_id": ["a", "a"], "label": ["Anomali", "QoS Baik"]})
    assert session_recommendations(rows, RULES) == []  # pulih → tidak lagi bermasalah


def test_cluster_recommendations_dari_profil():
    profile = [
        {"id_klaster": -1, "jumlah_anggota": 3, "label": "Anomali - Delay Ekstrem"},
        {"id_klaster": 0, "jumlah_anggota": 5, "label": "QoS Baik"},
        {"id_klaster": 1, "jumlah_anggota": 7, "label": "QoS Buruk - Packet Loss Melebihi Batas"},
    ]
    recs = cluster_recommendations(profile, RULES)
    assert [r["id_klaster"] for r in recs] == [-1, 1]
    assert recs[1]["saran"] == SARAN["loss"] and recs[1]["jumlah_anggota"] == 7


def test_load_rules_validasi(tmp_path):
    with pytest.raises(RulesError, match="tidak ditemukan"):
        load_rules(tmp_path / "tidak_ada.yaml")
    bad = tmp_path / "r.yaml"
    bad.write_text("rules:\n  delay: {pemicu: x, kategori: y, saran: [a]}\n", encoding="utf-8")
    with pytest.raises(RulesError, match="jitter"):
        load_rules(bad)
    bad.write_text(
        "rules:\n" + "".join(f"  {k}: {{pemicu: x, kategori: y, saran: teks}}\n" for k in ("delay", "jitter", "loss", "anomali")),
        encoding="utf-8",
    )
    with pytest.raises(RulesError, match="daftar"):
        load_rules(bad)


def test_default_capture_backend_scapy_dan_bpf_tanpa_trafik_di_luar_distribusi():
    cfg = load_config(DEFAULT_CONFIG_PATH, use_user=False)
    assert cfg["monitoring"]["capture_backend"] == "scapy"
    bpf = cfg["monitoring"]["bpf_filter"]
    assert bpf.startswith("udp")
    for port in (443, 53, 123, 5353, 1900):
        assert f"not port {port}" in bpf
    assert cfg["paths"]["recommendations_file"] == "config/recommendations.yaml"


# ---- kolom metrik_penyebab (M8) -----------------------------------------------------------------
def test_recommend_membaca_kolom_metrik_penyebab_bila_ada():
    label = "QoS Buruk - Delay Melebihi Batas"
    rec = recommend(label, RULES, causes="jitter;loss")  # kolom didahulukan atas parser teks
    assert rec.triggers == ("Jitter Melebihi Batas", "Packet Loss Melebihi Batas")
    assert list(rec.suggestions) == SARAN["jitter"] + SARAN["loss"]


@pytest.mark.parametrize("causes", [None, "", float("nan"), "tidak_dikenal"])
def test_recommend_fallback_parser_teks_untuk_data_lama(causes):
    rec = recommend("QoS Buruk - Delay dan Jitter Melebihi Batas", RULES, causes=causes)
    assert rec.triggers == ("Delay Melebihi Batas", "Jitter Melebihi Batas")


def test_causes_from_column_urutan_dan_validasi():
    from qos.recommend.rules import causes_from_column

    assert causes_from_column("loss;delay") == ["delay", "loss"]  # urutan tetap
    assert causes_from_column("delay") == ["delay"]
    assert causes_from_column("") is None and causes_from_column(None) is None and causes_from_column(float("nan")) is None


def test_anomali_tetap_hanya_saran_anomali_walau_ada_kolom():
    rec = recommend("Anomali - Delay dan Packet Loss Ekstrem", RULES, causes="delay;loss")
    assert list(rec.suggestions) == SARAN["anomali"]


def test_session_recommendations_memakai_kolom_dan_fallback():
    rows = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c"],
            "label": ["QoS Buruk - Delay Melebihi Batas", "QoS Buruk - Delay Melebihi Batas", "QoS Baik"],
            "metrik_penyebab": ["jitter", float("nan"), ""],  # a: kolom; b: CSV lama → parser; c: baik
        }
    )
    recs = {tuple(r["pemicu"]): r["jumlah_flow"] for r in session_recommendations(rows, RULES)}
    assert recs == {("Jitter Melebihi Batas",): 1, ("Delay Melebihi Batas",): 1}
    no_col = session_recommendations(rows.drop(columns="metrik_penyebab"), RULES)
    assert [(r["pemicu"], r["jumlah_flow"]) for r in no_col] == [(["Delay Melebihi Batas"], 2)]


def test_cluster_recommendations_memakai_kolom_profil():
    profile = [{"id_klaster": 1, "jumlah_anggota": 4, "label": "QoS Buruk - Delay Melebihi Batas", "metrik_penyebab": "loss"}]
    assert cluster_recommendations(profile, RULES)[0]["pemicu"] == ["Packet Loss Melebihi Batas"]
