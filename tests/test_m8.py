"""Uji M8: apply_overrides, kolom CSV jendela (feature_mode, diimputasi), dan versioning. Latih ulang unggahan: tests/test_dataset.py."""
import pandas as pd

from qos.model.dataset import FORMAT_PANGGILAN, load_upload
from qos.model.retrain import apply_overrides
from qos.monitor.engine import WINDOW_COLUMNS, InferenceEngine, load_artifacts
from qos.capture.flow_table import WindowFlow


# ---- latih ulang --------------------------------------------------------------------------------
def test_apply_overrides_tidak_mengubah_config_asli(iso_cfg):
    out = apply_overrides(iso_cfg, {"min_cluster_size": 9, "min_samples": 3, "grid_enabled": False,
                                    "thresholds": {"loss_max_pct": 1.0}, "selection": {"min_ari_mean": 0.5},
                                    "limit_percentile": 90, "limit_rule": "global"})
    assert (out["hdbscan"]["min_cluster_size"], out["hdbscan"]["min_samples"]) == (9, 3)
    assert out["hdbscan"]["grid_search"]["enabled"] is False and out["thresholds"]["loss_max_pct"] == 1.0
    assert out["hdbscan"]["selection"]["min_ari_mean"] == 0.5 and out["hdbscan"]["selection"]["max_noise_ratio"] == 0.20
    assert (out["assignment"]["limit_percentile"], out["assignment"]["limit_rule"]) == (90, "global")
    assert iso_cfg["thresholds"]["loss_max_pct"] == 3.0 and iso_cfg["hdbscan"]["min_cluster_size"] == 3
    assert iso_cfg["hdbscan"]["selection"]["min_ari_mean"] == 0.90 and iso_cfg["assignment"]["limit_rule"] == "max_cluster_global"


# ---- kolom feature_mode dan diimputasi (M8b) ----------------------------------------------------
def test_csv_jendela_punya_kolom_feature_mode_dan_diimputasi(active_cfg):
    eng = InferenceEngine(load_artifacts(active_cfg), active_cfg)
    flows = [
        WindowFlow("normal", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("nan", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, float("nan"), 8_000.0),
        WindowFlow("kurang", 0.0, 5.0, 17, 100, 2, 102, 1e6, float("nan"), 10.0),
    ]
    out = eng.predict(flows, "s").set_index("flow_id")
    assert out["feature_mode"].tolist() == [active_cfg["monitoring"]["feature_mode"]] * 3
    assert out["diimputasi"].tolist() == [False, True, False]  # baris tanpa prediksi tidak diimputasi
    assert out["diimputasi"].dtype == bool
    assert {"feature_mode", "diimputasi"} <= set(WINDOW_COLUMNS)


def test_csv_jendela_nyata_dari_sesi_bisa_dibaca_loader(active_cfg, tmp_path):
    """Round-trip: CSV hasil MonitoringSession memuat kolom baru dan lolos pengecualian bila mode cocok."""
    from qos.capture.demo import write_demo_pcap
    from qos.capture.sniffer import iter_pcap
    from qos.monitor.engine import MonitoringSession

    pcap = write_demo_pcap(tmp_path / "d.pcap", duration_sec=12.0)
    active_cfg["monitoring"]["feature_mode"] = "cumulative"
    s = MonitoringSession(active_cfg, InferenceEngine(load_artifacts(active_cfg), active_cfg), lambda stop: iter_pcap(pcap, 0, stop), "rt")
    s.start()
    s.wait(60)
    s.stop()
    rows = pd.read_csv(s.window_csv)
    assert set(rows["feature_mode"]) == {"cumulative"} and not rows["diimputasi"].any()
    if s.calls_csv.is_file():  # panggilan selesai di sesi demo → format panggilan terdeteksi dan terbaca
        active_cfg["data"]["min_session"] = {"fwd": 1, "bwd": 1}
        frame, rep = load_upload(s.calls_csv, active_cfg)
        assert rep.format == FORMAT_PANGGILAN and rep.error is None and len(frame) >= 1
