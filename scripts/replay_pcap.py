"""Replay PCAP melalui mesin inferensi (tanpa UI) dan tulis riwayat sesi.

Pemakaian:
  python scripts/replay_pcap.py --pcap data/sample_pcap/demo_voip.pcap [--speed 0] [--feature-mode window|cumulative]
  python scripts/replay_pcap.py --make-demo     # buat data/sample_pcap/demo_voip.pcap (SINTETIS)
"""
from __future__ import annotations

import argparse
import copy
import sys

import pandas as pd

from qos.capture.demo import write_demo_pcap
from qos.capture.sniffer import iter_pcap
from qos.config import load_config, resolve_path, setup_logging
from qos.monitor.engine import InferenceEngine, MonitoringSession, load_artifacts

DEMO_PCAP = "data/sample_pcap/demo_voip.pcap"


def run_once(cfg: dict, pcap: str, speed: float | None) -> tuple[MonitoringSession, pd.DataFrame]:
    """Jalankan satu sesi replay sampai selesai; kembalikan sesi dan seluruh baris jendela."""
    engine = InferenceEngine(load_artifacts(cfg), cfg)
    session = MonitoringSession(cfg, engine, lambda stop: iter_pcap(pcap, speed, stop), f"replay_{pcap.split('/')[-1]}")
    session.start()
    session.wait()
    session.stop()
    if session.error:
        raise SystemExit(f"Sesi gagal: {session.error}")
    return session, pd.read_csv(session.window_csv) if session.window_csv.exists() else pd.DataFrame()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--pcap", default=None)
    parser.add_argument("--speed", type=float, default=None, help="pengali kecepatan; 0 = secepat mungkin (default: config)")
    parser.add_argument("--feature-mode", choices=["window", "cumulative"], default=None)
    parser.add_argument("--compare", action="store_true", help="bandingkan feature_mode window vs cumulative")
    parser.add_argument("--make-demo", action="store_true", help="buat PCAP sintetis untuk demo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.make_demo:
        path = write_demo_pcap(resolve_path(DEMO_PCAP))
        print(f"PCAP sintetis ditulis: {path}")
        return
    if not args.pcap:
        parser.error("--pcap wajib (atau pakai --make-demo)")
    pcap = str(resolve_path(args.pcap))
    speed = cfg["monitoring"]["replay_speed"] if args.speed is None else args.speed

    modes = ["window", "cumulative"] if args.compare else [args.feature_mode or cfg["monitoring"]["feature_mode"]]
    for mode in modes:
        c = copy.deepcopy(cfg)
        c["monitoring"]["feature_mode"] = mode
        session, rows = run_once(c, pcap, speed)
        print(f"\n== feature_mode = {mode} ==")
        print(f"Sesi: {session.session_id} · baris jendela: {len(rows)} · flow: {rows['flow_id'].nunique() if len(rows) else 0}")
        print(f"Kartu: {session.card_counts()}")
        if len(rows):
            print(rows.groupby(["klaster_terdekat", "status"], dropna=False).size().rename("jumlah_jendela").to_string())
        print(f"CSV jendela: {session.window_csv}\nRingkasan: {session.summary_path}")
        print(pd.read_csv(session.summary_path).T.to_string(header=False))


if __name__ == "__main__":
    main()
