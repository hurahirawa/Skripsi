"""Uji ekspor laporan sesi (CSV dan PDF) untuk klasterisasi murni."""
import copy

import pandas as pd
import pytest

from qos.config import PROJECT_ROOT, load_config
from qos.monitor.calls import INSUFFICIENT, SIGNALING, STATUS_CLUSTERED, STATUS_OUTSIDE
from qos.recommend.rules import call_recommendations, cluster_metric_map, load_rules
from qos.report.export import build_session_pdf, distribution_chart, export_session, per_flow_table, state_counts

RULES = load_rules(PROJECT_ROOT / "config" / "recommendations.yaml")


def _desc(delay, jitter, fai):
    from qos.profiling.describe import describe

    return describe(delay, jitter, fai, load_config(use_user=False)["thresholds"])


def _cluster(cid, n, delay, jitter, fai, limit):
    stats = {m: {"mean": v, "median": v, "std": 1.0, "min": v, "max": v} for m, v in (("delay_ms", delay), ("jitter_ms", jitter), ("fai_pct", fai))}
    d = _desc(delay, jitter, fai)
    return {"id_klaster": cid, "nama": "Noise" if cid == -1 else f"Klaster {cid}", "jumlah_anggota": n, "statistik": stats,
            "batas_jarak": limit, "deskripsi": d, "metrik_melewati_ambang": d["melewati_ambang"]}


PROFILE = {
    "model_version": "20260101_000000",
    "evaluation": {"silhouette": 0.8866, "silhouette_label": "Strong Structure", "dbcv": 0.74, "chi": 7554.68, "noise_ratio": 0.1377, "n_clusters": 2},
    "clusters": [_cluster(-1, 3, 900.0, 50.0, 3.0, None), _cluster(1, 9, 10.0, 5.0, 0.0, 0.5), _cluster(2, 6, 500.0, 5.0, 0.0, 0.4)],
}


@pytest.fixture
def rows():
    return pd.DataFrame(
        {
            "flow_id": ["a", "a", "b", "c", "d", "s"],
            "delay_ms": [10.0, 12.0, 900.0, 5.0, 1.0, 1.0],
            "jitter_ms": [1.0, 2.0, 80.0, 1.0, 1.0, 1.0],
            "fai_pct": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            "status": [STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_OUTSIDE, STATUS_CLUSTERED, INSUFFICIENT, SIGNALING],
            "klaster_terdekat": [1, 2, 2, 1, pd.NA, pd.NA],
            "jarak_terdekat": [0.1, 0.2, 3.0, 0.1, float("nan"), float("nan")],
        }
    )


def test_per_flow_table(rows):
    t = per_flow_table(rows).set_index("flow_id")
    assert t.loc["a", "jendela"] == 2 and t.loc["a", "klaster_terakhir"] == "Klaster 2"
    assert t.loc["b", "klaster_terakhir"] == "Di luar klaster (terdekat: Klaster 2)"
    assert t.loc["d", "klaster_terakhir"] == INSUFFICIENT
    assert t.loc["a", "delay_ms"] == pytest.approx(11.0)
    assert list(per_flow_table(rows.iloc[0:0]).columns)[:3] == ["flow_id", "jendela", "status_terakhir"]


def test_state_counts_per_klaster_di_luar_dan_belum_cukup(rows):
    c = state_counts(rows, [1, 2])
    assert c == {"Klaster 1": 1, "Klaster 2": 1, "Di luar klaster": 1, INSUFFICIENT: 1}  # sinyal SIP tidak dihitung
    assert state_counts(rows.iloc[0:0], [1]) == {"Klaster 1": 0, "Di luar klaster": 0, INSUFFICIENT: 0}


def test_distribution_chart_png(rows):
    png = distribution_chart(state_counts(rows, [1, 2]))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_pdf_valid_dan_tidak_kosong(rows):
    summary = {"id_sesi_pemantauan": "s1", "status_sesi": "Selesai", "total_aliran": 4, "klaster_mayoritas": 1, "persen_di_luar_klaster": 25.0}
    pdf = build_session_pdf(summary, rows, RULES, PROFILE)
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF") and len(pdf) > 5000


def test_build_pdf_sesi_kosong_dan_tanpa_model():
    empty = pd.DataFrame(columns=["flow_id", "delay_ms", "jitter_ms", "fai_pct", "status", "klaster_terdekat", "jarak_terdekat"])
    pdf = build_session_pdf({"id_sesi_pemantauan": "kosong"}, empty, RULES, None)
    assert pdf[:5] == b"%PDF-"


def test_build_pdf_tanpa_saran_lebih_pendek_dan_tetap_valid(rows):
    with_recs = build_session_pdf({"id_sesi_pemantauan": "a"}, rows, RULES, PROFILE)
    without = build_session_pdf({"id_sesi_pemantauan": "a"}, rows, RULES, PROFILE, show_recommendations=False)
    assert without[:5] == b"%PDF-" and len(without) < len(with_recs)


def test_build_pdf_tidak_gagal_pada_banyak_flow():
    n = 120
    big = pd.DataFrame(
        {"flow_id": [f"10.0.0.{i}:1-10.0.1.{i}:2-17" for i in range(n)], "delay_ms": 1.0, "jitter_ms": 1.0,
         "fai_pct": 0.0, "status": STATUS_OUTSIDE, "klaster_terdekat": 2, "jarak_terdekat": 3.0}
    )
    assert build_session_pdf({"id_sesi_pemantauan": "besar"}, big, RULES, PROFILE)[:5] == b"%PDF-"


def test_export_session_menulis_csv_dan_pdf(tmp_path, rows):
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    window_csv = tmp_path / "jendela_x.csv"
    rows.to_csv(window_csv, index=False)
    summary = {"id_sesi_pemantauan": "x", "status_sesi": "Selesai", "total_aliran": 4, "lokasi_laporan": str(window_csv)}
    out = export_session(summary, cfg, RULES, PROFILE)
    assert out["csv"].parent == tmp_path / "reports" / "exports"
    assert pd.read_csv(out["csv"]).equals(pd.read_csv(window_csv))
    assert out["pdf"].read_bytes()[:5] == b"%PDF-"
    with pytest.raises(FileNotFoundError):
        export_session({**summary, "lokasi_laporan": str(tmp_path / "hilang.csv")}, cfg, RULES, PROFILE)


def test_export_mengikuti_config_recommendations_enabled(tmp_path, rows):
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    window_csv = tmp_path / "jendela_y.csv"
    rows.to_csv(window_csv, index=False)
    summary = {"id_sesi_pemantauan": "y", "lokasi_laporan": str(window_csv)}
    on = export_session(summary, cfg, RULES, PROFILE)["pdf"].stat().st_size
    cfg["recommendations"]["enabled"] = False
    off = export_session({**summary, "id_sesi_pemantauan": "z"}, cfg, RULES, PROFILE)["pdf"].stat().st_size
    assert off < on


def test_rekomendasi_dan_pdf_per_panggilan(rows):
    from qos.viz.plots import style_by_state

    calls = pd.DataFrame(
        {
            "flow_id": ["1.1.1.1:1-2.2.2.2:2-17", "3.3.3.3:1-4.4.4.4:2-17", "5.5.5.5:1-6.6.6.6:2-17"],
            "durasi_s": [30.0, 12.5, 4.0],
            "jumlah_jendela": [6, 3, 1],
            "status_akhir": [STATUS_CLUSTERED, STATUS_OUTSIDE, STATUS_CLUSTERED],
            "klaster_akhir": [1, 2, 2],
            "jarak_akhir": [0.1, 3.0, 0.2],
            "persen_di_luar_klaster": [0.0, 66.7, 0.0],
            "delay_ms": [10.0, 900.0, 500.0],
            "fai_pct": [0.0, 5.0, 0.0],
            "catatan": [float("nan")] * 3,
        }
    )
    recs = call_recommendations(calls, cluster_metric_map(PROFILE["clusters"]), RULES)
    assert [r["flow_id"] for r in recs] == ["3.3.3.3:1-4.4.4.4:2-17", "5.5.5.5:1-6.6.6.6:2-17"]  # di luar klaster dan Klaster 2 (delay)
    assert build_session_pdf({"id_sesi_pemantauan": "s"}, rows, RULES, PROFILE, calls).startswith(b"%PDF")
    html = style_by_state(pd.DataFrame({"klaster": ["Klaster 1", "Di luar klaster (terdekat: Klaster 2)"]}), "klaster").to_html()
    assert "#2E86AB33" in html and "#6B728033" in html
