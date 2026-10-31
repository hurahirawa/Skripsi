"""CLI pelatihan klasterisasi murni: HDBSCAN (aturan seleksi final) + pembanding K-Means (PRD TR-6).

Pelatihan awal: python scripts/train.py --config config/default_config.yaml [--data path] [--no-grid] [--activate]
Latih ulang dari unggahan (fungsi yang sama dengan halaman Pengaturan):
  python scripts/train.py --upload panggilan_x.csv Darknet2.csv --mode tambah|ganti [--activate]
Mode tambah = data_latih.csv model aktif + data baru; ganti = hanya data baru.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qos.config import load_config, resolve_path, setup_logging
from qos.model.retrain import RetrainError, retrain_upload
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
    parser.add_argument("--upload", nargs="+", default=None, metavar="CSV", help="latih ulang dari berkas CSV (panggilan_*.csv atau flow darknet/cic/bccc)")
    parser.add_argument("--mode", choices=["tambah", "ganti"], default=None, help="dengan --upload: tambah ke data latih aktif, atau ganti")
    parser.add_argument("--no-grid", action="store_true", help="lewati grid search; pakai parameter tetap config")
    parser.add_argument("--activate", action="store_true", help="jadikan versi ini aktif (current.txt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.upload:
        if args.mode is None:
            raise SystemExit("--upload butuh --mode tambah|ganti")
        files = [Path(f) for f in args.upload]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise SystemExit(f"Berkas tidak ditemukan: {missing}")
        try:
            res = retrain_upload(cfg, [(f, f.name) for f in files], args.mode, params={"grid_enabled": False} if args.no_grid else None)
        except RetrainError as exc:
            raise SystemExit(f"Latih ulang ditolak: {exc}")
        if args.activate:
            activate_version(cfg, res["version"])
        d = res["data"]
        for rep in d.files:
            print(f"  {rep.name} [{rep.format}]: " + " -> ".join(f"{label} {n}" for label, n in rep.stages))
        print(f"Data: lama {d.n_old} + baru {d.n_new} = {d.n_total} baris (mode {args.mode})")
        for msg in [*d.warnings, *res["peringatan"]]:
            print(f"PERINGATAN: {msg}")
        print(f"Artefak: {res['path']} (aktif: {args.activate})")
        print(f"Parameter: {res['params']}")
        print_eval(res["evaluation"])
        if res["pemetaan"] is not None:
            print(res["pemetaan"][["klaster_lama", "klaster_baru", "jarak"]].to_string(index=False))
        return
    if args.mode:
        raise SystemExit("--mode hanya dipakai bersama --upload")
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    flows, counts = load_training_frame(source, cfg)
    result = train(flows, cfg, use_grid=False if args.no_grid else None, counts=counts, source={"jenis": "csv", "lokasi": str(source), "sumber": source.name})
    out = save_artifacts(result, cfg, activate=True if args.activate else None)

    print(f"Artefak: {out}")
    print(f"Parameter: {result['params']}")
    print_eval(result["evaluation"])
    for key in ("hdbscan_assigned", "kmeans_n", "kmeans_opt"):
        c = result["comparison"][key]
        print(f"  {c['nama']}: Silhouette={c['silhouette']:.4f} DBI={c['davies_bouldin']:.4f} DBCV={c['dbcv']:.4f} ARI={c['ari_mean']:.3f}")


if __name__ == "__main__":
    main()
