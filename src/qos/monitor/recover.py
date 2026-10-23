"""Pemulihan sesi terputus: CSV jendela tanpa ringkasan (aplikasi dimatikan paksa) diberi ringkasan abnormal."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from qos.config import resolve_path
from qos.monitor.calls import CALL_COLUMNS, INSUFFICIENT, SIGNALING, count_final_labels
from qos.monitor.engine import STATUS_ABNORMAL, SUMMARY_COLUMNS, TIME_FORMAT

logger = logging.getLogger(__name__)
WINDOW_PREFIX, CALLS_PREFIX, SUMMARY_PREFIX = "jendela_", "panggilan_", "riwayat_pemantauan_"


def recover_orphan_sessions(cfg: dict[str, Any], active_ids: Iterable[str] = ()) -> list[Path]:
    """Tulis ringkasan "Dihentikan tidak normal" untuk tiap CSV jendela yang belum punya ringkasan.

    Dilewati: sesi di `active_ids` dan berkas yang diubah < monitoring.orphan_after_sec (mungkin masih berjalan).
    Skor model tidak diketahui (kosong). Kembalikan path ringkasan yang ditulis.
    """
    folder = resolve_path(cfg["paths"]["reports_dir"]) / "riwayat"
    if not folder.is_dir():
        return []
    known = set(active_ids)
    for p in folder.glob(f"{SUMMARY_PREFIX}*.csv"):
        try:
            known.update(pd.read_csv(p, usecols=["id_sesi_pemantauan"])["id_sesi_pemantauan"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
    stale = time.time() - float(cfg["monitoring"]["orphan_after_sec"])
    written = []
    for window_csv in sorted(folder.glob(f"{WINDOW_PREFIX}*.csv")):
        sid = window_csv.stem[len(WINDOW_PREFIX):]
        if sid in known or window_csv.stat().st_mtime > stale:
            continue
        written.append(_write_recovered(cfg, folder, sid, window_csv))
    return written


def _write_recovered(cfg: dict[str, Any], folder: Path, sid: str, window_csv: Path) -> Path:
    rows = pd.read_csv(window_csv)
    ended = datetime.fromtimestamp(window_csv.stat().st_mtime)
    calls_csv = folder / f"{CALLS_PREFIX}{sid}.csv"
    calls = pd.read_csv(calls_csv) if calls_csv.is_file() else pd.DataFrame(columns=list(CALL_COLUMNS))
    final = count_final_labels(calls)
    scored = rows[rows["label"] != SIGNALING]
    row = dict.fromkeys(SUMMARY_COLUMNS)
    row.update(
        id_sesi_pemantauan=sid,
        waktu_mulai=str(rows["waktu_jendela"].iloc[0]) if len(rows) else ended.isoformat(timespec="seconds"),
        waktu_selesai=ended.isoformat(timespec="seconds"),
        total_aliran=scored["flow_id"].nunique(),
        status_keseluruhan=STATUS_ABNORMAL,
        lokasi_laporan=str(window_csv),
        jumlah_panggilan=len(calls),
        panggilan_baik=final["QoS Baik"], panggilan_cukup=final["QoS Cukup"], panggilan_buruk=final["QoS Buruk"],
        panggilan_anomali=final["Anomali"], panggilan_belum_cukup=final[INSUFFICIENT],
    )
    base = f"{SUMMARY_PREFIX}{ended.strftime(TIME_FORMAT)}"
    path, n = folder / f"{base}.csv", 1
    while path.exists():
        path, n = folder / f"{base}_{n}.csv", n + 1
    pd.DataFrame([row])[list(SUMMARY_COLUMNS)].to_csv(path, index=False)
    logger.warning("Sesi %s tanpa ringkasan (terputus); ringkasan abnormal ditulis ke %s", sid, path)
    return path
