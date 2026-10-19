"""Uji ekspor laporan sesi (CSV dan PDF)."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from qos.config import PROJECT_ROOT, load_config
from qos.recommend.rules import load_rules
from qos.report.export import build_session_pdf, category_counts, distribution_chart, export_session, per_flow_table

RULES = load_rules(PROJECT_ROOT / "config" / "recommendations.yaml")
PROFILE = {
    "model_version": "20260101_000000",
    "evaluation": {"silhouette": 0.8866, "silhouette_label": "Strong Structure", "chi": 7554.68, "noise_ratio": 0.1377, "n_clusters": 2},
    "clusters": [
        {"id_klaster": -1, "jumlah_anggota": 3, "label": "Anomali - Delay Ekstrem",
         "statistik_pelabelan": {"mean_delay_ms": 900.0, "p999_jitter_ms": 50.0, "plr_pct": 3.0}},
        {"id_klaster": 0, "jumlah_anggota": 9, "label": "QoS Baik",
         "statistik_pelabelan": {"mean_delay_ms": 10.0, "p999_jitter_ms": 5.0, "plr_pct": 0.0}},
    ],
}


@pytest.fixture
def rows():
    return pd.DataFrame(
        {
            "flow_id": ["a", "a", "b", "c", "d"],
            "delay_ms": [10.0, 12.0, 900.0, 5.0, 1.0],
            "jitter_ms": [1.0, 2.0, 80.0, 1.0, 1.0],
            "fai_pct": [0.0, 0.0, 5.0, 0.0, 0.0],
            "label": ["QoS Baik", "QoS Buruk - Jitter Melebihi Batas", "Anomali - Delay Ekstrem", "QoS Baik", "Data Belum Cukup"],
        }
    )


def test_per_flow_table(rows):
    t = per_flow_table(rows).set_index("flow_id")
    assert t.loc["a", "jendela"] == 2 and t.loc["a", "label_terakhir"] == "QoS Buruk - Jitter Melebihi Batas"
    assert t.loc["a", "delay_ms"] == pytest.approx(11.0)
    assert list(per_flow_table(rows.iloc[0:0]).columns)[:3] == ["flow_id", "jendela", "label_terakhir"]


def test_category_counts_termasuk_data_belum_cukup(rows):
    c = category_counts(per_flow_table(rows))
    assert c == {"QoS Baik": 1, "QoS Cukup": 0, "QoS Buruk": 1, "Anomali": 1, "Data Belum Cukup": 1}


def test_distribution_chart_png(rows):
    png = distribution_chart(category_counts(per_flow_table(rows)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_pdf_valid_dan_tidak_kosong(rows):
    pdf = build_session_pdf({"id_sesi_pemantauan": "s1", "status_keseluruhan": "Kritis", "total_aliran": 4}, rows, RULES, PROFILE)
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF") and len(pdf) > 5000


def test_build_pdf_sesi_kosong_dan_tanpa_model():
    empty = pd.DataFrame(columns=["flow_id", "delay_ms", "jitter_ms", "fai_pct", "label"])
    pdf = build_session_pdf({"id_sesi_pemantauan": "kosong"}, empty, RULES, None)
    assert pdf[:5] == b"%PDF-"


def test_build_pdf_tidak_gagal_pada_banyak_flow():
    n = 120
    big = pd.DataFrame(
        {"flow_id": [f"10.0.0.{i}:1-10.0.1.{i}:2-17" for i in range(n)], "delay_ms": 1.0, "jitter_ms": 1.0,
         "fai_pct": 0.0, "label": "QoS Buruk - Delay, Jitter dan Packet Loss Melebihi Batas"}
    )
    assert build_session_pdf({"id_sesi_pemantauan": "besar"}, big, RULES, PROFILE)[:5] == b"%PDF-"


def test_export_session_menulis_csv_dan_pdf(tmp_path, rows):
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    window_csv = tmp_path / "jendela_x.csv"
    rows.to_csv(window_csv, index=False)
    summary = {"id_sesi_pemantauan": "x", "status_keseluruhan": "Kritis", "total_aliran": 4, "lokasi_laporan": str(window_csv)}
    out = export_session(summary, cfg, RULES, PROFILE)
    assert out["csv"].parent == tmp_path / "reports" / "exports"
    assert pd.read_csv(out["csv"]).equals(pd.read_csv(window_csv))
    assert out["pdf"].read_bytes()[:5] == b"%PDF-"
    with pytest.raises(FileNotFoundError):
        export_session({**summary, "lokasi_laporan": str(tmp_path / "hilang.csv")}, cfg, RULES, PROFILE)
