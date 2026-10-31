"""Buat data_latih.csv untuk versi artefak yang sudah ada TANPA refit (dari training_sample.csv).

Pemakaian: python scripts/backfill_data_latih.py [--version 20261027_010848] [--sumber Darknet.CSV]
Default versi = model aktif. Model tidak diubah; train_meta.json diberi catatan data_latih.
"""
from __future__ import annotations

import argparse
import sys

from qos.config import load_config, setup_logging
from qos.model.dataset import DatasetError, backfill_data_latih
from qos.model.versions import active_version


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None, help="default: versi aktif")
    parser.add_argument("--sumber", default="Darknet.CSV", help="nama berkas asal data latih (kolom sumber)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    version = args.version or active_version(cfg)
    if not version:
        raise SystemExit("Tidak ada versi aktif; sebutkan --version")
    try:
        path = backfill_data_latih(cfg, version, args.sumber)
    except DatasetError as exc:
        raise SystemExit(str(exc))
    print(f"Dibuat: {path}")


if __name__ == "__main__":
    main()
