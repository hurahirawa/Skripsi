"""CLI pelatihan HDBSCAN (PRD TR-6).

Pemakaian: python scripts/train.py --config config/default_config.yaml [--source csv|log] [--data path] [--no-grid] [--activate]

Sumber `log` melatih dari log jendela monitoring (reports/riwayat/jendela_*.csv) sesuai opsi retrain.log_source.
"""
from __future__ import annotations

import argparse
import sys

from qos.config import load_config, resolve_path, setup_logging
from qos.model.retrain import RetrainError, retrain
from qos.model.train import load_training_frame, save_artifacts, train
from qos.model.versions import activate_version


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="berkas CSV atau folder (default: data.raw_dir)")
    parser.add_argument("--source", choices=["csv", "log"], default="csv", help="sumber data latih (default: csv)")
    parser.add_argument("--no-grid", action="store_true", help="lewati grid search; pakai parameter tetap config")
    parser.add_argument("--activate", action="store_true", help="jadikan versi ini aktif (current.txt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.source == "log":
        try:
            res = retrain(cfg, "log", params={"grid_enabled": False} if args.no_grid else None)
        except RetrainError as exc:
            raise SystemExit(f"Latih ulang dari log ditolak: {exc}")
        if args.activate:
            activate_version(cfg, res["version"])
        ev = res["evaluation"]
        print(f"Artefak: {res['path']} (aktif: {args.activate})")
        print(f"Parameter: {res['params']}")
        print(
            f"Silhouette={ev['silhouette']:.4f} ({ev['silhouette_label']}) CHI={ev['chi']:.2f} "
            f"Noise={ev['noise_ratio']:.2%} Klaster={ev['n_clusters']}"
        )
        return
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    flows, counts = load_training_frame(source, cfg)
    result = train(flows, cfg, use_grid=False if args.no_grid else None, counts=counts)
    out = save_artifacts(result, cfg, activate=True if args.activate else None)

    ev = result["evaluation"]
    print(f"Artefak: {out}")
    print(f"Parameter: {result['params']}")
    print(
        f"Silhouette={ev['silhouette']:.4f} ({ev['silhouette_label']}) CHI={ev['chi']:.2f} "
        f"Noise={ev['noise_ratio']:.2%} Klaster={ev['n_clusters']}"
    )


if __name__ == "__main__":
    main()
