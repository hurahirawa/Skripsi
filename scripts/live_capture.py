"""Live capture melalui mesin inferensi (tanpa UI): cek prasyarat, daftar interface, atau jalankan sesi singkat.

Pemakaian:
  python scripts/live_capture.py --check [--backend scapy]
  python scripts/live_capture.py --list-interfaces
  python scripts/live_capture.py --interface "Wi-Fi" --backend scapy --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from qos.capture import sniffer
from qos.config import load_config, setup_logging
from qos.monitor.engine import InferenceEngine, ModelNotFoundError, MonitoringSession, load_artifacts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=list(sniffer.LIVE_BACKENDS), default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--feature-mode", choices=["window", "cumulative"], default=None)
    parser.add_argument("--check", action="store_true", help="hanya periksa prasyarat")
    parser.add_argument("--list-interfaces", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.feature_mode:
        cfg["monitoring"]["feature_mode"] = args.feature_mode
    backend = args.backend or (
        cfg["monitoring"]["capture_backend"] if cfg["monitoring"]["capture_backend"] in sniffer.LIVE_BACKENDS else "pyshark"
    )

    report = sniffer.check_live_environment(cfg, backend)
    print(f"Backend: {backend} · admin: {sniffer.is_admin()}")
    for note in report.notes:
        print(f"  ✓ {note}")
    for problem in report.problems:
        print(f"  ✗ {problem}")
    if args.list_interfaces:
        for name in sniffer.list_interfaces(cfg):
            print(f"  - {name}")
        return 0
    if args.check:
        return 0 if report.ok else 2
    if not report.ok:
        print("Berhenti: prasyarat live capture belum terpenuhi.")
        return 2
    if not args.interface:
        print("Berhenti: --interface wajib (lihat --list-interfaces).")
        return 2

    try:
        art = load_artifacts(cfg)
    except ModelNotFoundError as exc:
        print(f"Berhenti: {exc}")
        return 2
    session = MonitoringSession(
        cfg, InferenceEngine(art, cfg),
        lambda stop: sniffer.open_source(cfg, backend, stop=stop, interface=args.interface),
        f"{backend}_{args.interface}",
    )
    session.start()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and session.running:
        time.sleep(0.5)
    summary = session.stop()
    if session.error:
        print(f"Sesi gagal: {session.error}")
        return 1
    print(f"Sesi: {session.session_id} · baris jendela: {session.n_windows_rows}")
    print(f"Kartu: {session.card_counts()}")
    if session.window_csv.exists():
        rows = pd.read_csv(session.window_csv)
        print(rows.groupby("label").size().rename("jumlah_jendela").to_string())
    print(f"CSV jendela: {session.window_csv}\nRingkasan: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
