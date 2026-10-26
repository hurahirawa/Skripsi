"""CLI pelatihan klasterisasi murni: HDBSCAN (aturan seleksi final) + pembanding K-Means (PRD TR-6).

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


def print_eval(ev: dict) -> None:
    """Ringkasan evaluasi HDBSCAN (tanpa noise) di terminal."""
    print(
        f"Klaster={ev['n_clusters']} Noise={ev['noise_ratio']:.2%} Silhouette={ev['silhouette']:.4f} ({ev['silhouette_label']}) "
        f"DBI={ev['davies_bouldin']:.4f} CHI={ev['chi']:.2f} DBCV={ev['dbcv']:.4f} ARI={ev['ari_mean']:.3f}±{ev['ari_sd']:.3f}"
    )


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
        print(f"Artefak: {res['path']} (aktif: {args.activate})")
        print(f"Parameter: {res['params']}")
        print_eval(res["evaluation"])
        return
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    flows, counts = load_training_frame(source, cfg)
    result = train(flows, cfg, use_grid=False if args.no_grid else None, counts=counts)
    out = save_artifacts(result, cfg, activate=True if args.activate else None)

    print(f"Artefak: {out}")
    print(f"Parameter: {result['params']}")
    print_eval(result["evaluation"])
    for key in ("hdbscan_assigned", "kmeans_n", "kmeans_opt"):
        c = result["comparison"][key]
        print(f"  {c['nama']}: Silhouette={c['silhouette']:.4f} DBI={c['davies_bouldin']:.4f} DBCV={c['dbcv']:.4f} ARI={c['ari_mean']:.3f}")


if __name__ == "__main__":
    main()
