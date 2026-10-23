"""Mesin inferensi real-time dan sesi pemantauan (PRD MON-5…MON-8, §10.3)."""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import hdbscan
import joblib
import numpy as np
import pandas as pd

from qos.capture.flow_table import FlowEnd, WindowAggregator, WindowFlow
from qos.config import resolve_path
from qos.features.extract import derive_metrics
from qos.features.preprocess import build_features, impute_median
from qos.monitor.calls import CALL_COLUMNS, INSUFFICIENT, CallTracker, count_final_labels
from qos.profiling.labeling import (
    ACTIONS,
    CATEGORY_ANOMALI,
    CATEGORY_BURUK,
    CAUSE_SEPARATOR,
    METRIC_NAMES,
    NOISE_ID,
    cause_metrics,
)

logger = logging.getLogger(__name__)

CURRENT_FILE = "current.txt"
WINDOW_COLUMNS: tuple[str, ...] = (
    "waktu_jendela", "id_sesi", "flow_id", "delay_ms", "jitter_ms", "fwd", "bwd", "fai_pct",
    "cluster_id", "membership_prob", "label", "kategori_tindakan", "metrik_penyebab", "pc1", "pc2",
    "feature_mode", "diimputasi",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "id_sesi_pemantauan", "waktu_mulai", "waktu_selesai", "total_aliran", "jumlah_klaster",
    "skor_silhouette", "skor_chi", "noise_ratio", "status_keseluruhan", "lokasi_laporan",
    "jumlah_panggilan", "panggilan_baik", "panggilan_cukup", "panggilan_buruk", "panggilan_anomali",
    "panggilan_belum_cukup",
)
STATUS_CRITICAL, STATUS_EMPTY = "Kritis", "Tidak ada data"
TIME_FORMAT = "%Y%m%d_%H%M%S"
US_PER_MS = 1000.0
SESSION_SUFFIX_LEN = 6
CARD_KEYS: tuple[str, ...] = ("QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali")


class ModelNotFoundError(RuntimeError):
    """Tidak ada model aktif atau artefak tidak lengkap (MON-8)."""


@dataclass
class Artifacts:
    """Artefak model aktif yang dimuat dari artifacts/<versi>/."""

    version: str
    model: Any
    scaler: Any
    pca: Any
    profile: dict[str, Any]
    meta: dict[str, Any]

    @property
    def labels(self) -> dict[int, str]:
        """cluster_id → label (lookup, tanpa hitung ulang)."""
        return {int(k): v for k, v in self.profile["label_map"].items()}

    @property
    def causes(self) -> dict[int, str]:
        """cluster_id → metrik_penyebab ("delay;loss"); untuk profil lama dihitung dari `tingkat`."""
        out = {}
        for c in self.profile["clusters"]:
            cid = int(c["id_klaster"])
            if "metrik_penyebab" in c:
                out[cid] = c["metrik_penyebab"]
            else:
                levels = {m: c["tingkat"][m] for m in METRIC_NAMES}
                out[cid] = CAUSE_SEPARATOR.join(cause_metrics(levels, noise=cid == NOISE_ID))
        return out

    @property
    def actions(self) -> dict[int, str]:
        """cluster_id → kategori tindakan."""
        return {int(c["id_klaster"]): c["kategori_tindakan"] for c in self.profile["clusters"]}


def load_artifacts(cfg: dict[str, Any], version: str | None = None) -> Artifacts:
    """Muat artefak versi `version`, atau versi aktif di current.txt. Error jelas jika tidak ada."""
    root = resolve_path(cfg["paths"]["artifacts_dir"])
    if version is None:
        current = root / CURRENT_FILE
        if not current.is_file():
            raise ModelNotFoundError(f"Tidak ada model aktif ({current} tidak ada). Latih model: python scripts/train.py")
        version = current.read_text("utf-8").strip()
    folder = root / version
    needed = ("hdbscan_model.joblib", "scaler.joblib", "pca.joblib", "cluster_profile.json", "train_meta.json")
    missing = [n for n in needed if not (folder / n).is_file()]
    if missing:
        raise ModelNotFoundError(f"Artefak versi {version} tidak lengkap, kurang: {missing}")
    return Artifacts(
        version=version,
        model=joblib.load(folder / "hdbscan_model.joblib"),
        scaler=joblib.load(folder / "scaler.joblib"),
        pca=joblib.load(folder / "pca.joblib"),
        profile=json.loads((folder / "cluster_profile.json").read_text("utf-8")),
        meta=json.loads((folder / "train_meta.json").read_text("utf-8")),
    )


class InferenceEngine:
    """Fitur jendela → scaler.transform → approximate_predict → lookup label (scaler TIDAK di-fit ulang)."""

    def __init__(self, art: Artifacts, cfg: dict[str, Any]) -> None:
        self.art, self.cfg = art, cfg
        self.labels, self.actions, self.causes = art.labels, art.actions, art.causes
        self.n_insufficient = 0  # baris "Data Belum Cukup"

    def predict(self, flows: list[WindowFlow], session_id: str) -> pd.DataFrame:
        """Prediksi klaster dan label tiap WindowFlow; kolom sesuai WINDOW_COLUMNS."""
        if not flows:
            return pd.DataFrame(columns=list(WINDOW_COLUMNS))
        feat, meta = self.cfg["features"], self.art.meta
        live = self.cfg["monitoring"]["min_session_live"]
        min_fwd, min_bwd = max(live["fwd"] or 0, 1), live["bwd"] or 0  # fwd = 0 membuat FAI tak terdefinisi
        enough = np.array([f.fwd >= min_fwd and f.bwd >= min_bwd for f in flows])
        self.n_insufficient += int((~enough).sum())

        fwd = np.array([f.fwd for f in flows], dtype=float)
        bwd = np.array([f.bwd for f in flows], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fai = np.where(fwd > 0, (fwd - bwd) / fwd * 100, np.nan)
        if meta["fai_clip_negative"]:
            fai = np.where(np.isnan(fai), fai, np.maximum(fai, 0))
        # baris "Data Belum Cukup": nilai mentah dicatat, tanpa prediksi (cluster/prob/PC kosong)
        out = pd.DataFrame(
            {
                "waktu_jendela": [datetime.fromtimestamp(f.window_start).isoformat(timespec="seconds") for f in flows],
                "id_sesi": session_id,
                "flow_id": [f.flow_id for f in flows],
                "delay_ms": [f.iat_mean_us / US_PER_MS for f in flows],
                "jitter_ms": [f.iat_std_us / US_PER_MS for f in flows],
                "fwd": fwd.astype(int),
                "bwd": bwd.astype(int),
                "fai_pct": fai,
                "cluster_id": pd.array([pd.NA] * len(flows), dtype="Int64"),
                "membership_prob": np.nan,
                "label": INSUFFICIENT,
                "kategori_tindakan": "-",
                "metrik_penyebab": "",
                "pc1": np.nan,
                "pc2": np.nan,
                "feature_mode": self.cfg["monitoring"]["feature_mode"],
                "diimputasi": False,  # baris tanpa prediksi tidak diimputasi
            }
        )
        if not enough.any():
            return out[list(WINDOW_COLUMNS)]

        # MON-5: nilai kosong → median artefak latih (hanya untuk baris yang diprediksi)
        raw = pd.DataFrame(
            {
                feat["delay_column"]: [f.iat_mean_us for f, ok in zip(flows, enough) if ok],
                feat["jitter_column"]: [f.iat_std_us for f, ok in zip(flows, enough) if ok],
                "fwd_packets_count": fwd[enough],
                "bwd_packets_count": bwd[enough],
            }
        )
        base = list(meta["median_imputasi"])
        imputed = raw[[feat["delay_column"], feat["jitter_column"]]].isna().any(axis=1).to_numpy()
        raw[base] = impute_median(raw[base], meta["median_imputasi"])
        d = derive_metrics(raw, feat["delay_column"], feat["jitter_column"], meta["fai_clip_negative"])
        X = self.art.scaler.transform(build_features(d, meta["cluster_columns"], meta["log_transform"]))
        ids, prob = hdbscan.approximate_predict(self.art.model, np.asarray(X, dtype=float))
        pcs = self.art.pca.transform(X)
        ids = np.asarray(ids, dtype=int)

        idx = out.index[enough]
        out.loc[idx, "delay_ms"] = d["delay_ms"].to_numpy()  # nilai setelah imputasi median
        out.loc[idx, "jitter_ms"] = d["jitter_ms"].to_numpy()
        out.loc[idx, "fai_pct"] = d["fai_pct"].to_numpy()
        out.loc[idx, "cluster_id"] = pd.array(ids, dtype="Int64")
        out.loc[idx, "membership_prob"] = np.asarray(prob, dtype=float)
        out.loc[idx, "label"] = [self.labels.get(int(i), self.labels.get(NOISE_ID, CATEGORY_ANOMALI)) for i in ids]
        out.loc[idx, "kategori_tindakan"] = [self.actions.get(int(i), ACTIONS[CATEGORY_ANOMALI]) for i in ids]
        out.loc[idx, "metrik_penyebab"] = [self.causes.get(int(i), self.causes.get(NOISE_ID, "")) for i in ids]
        out.loc[idx, "pc1"] = pcs[:, 0]
        out.loc[idx, "pc2"] = pcs[:, 1]
        out.loc[idx, "diimputasi"] = imputed
        return out[list(WINDOW_COLUMNS)]


def card_category(label: str) -> str:
    """Kategori kartu dari label: QoS Baik / Cukup / Buruk / Anomali, atau "Data Belum Cukup"."""
    if label == INSUFFICIENT:
        return INSUFFICIENT
    for key in CARD_KEYS:
        if label.startswith(key):
            return key
    return CATEGORY_ANOMALI


def overall_status(last_labels: dict[str, str], kritis_ratio: float) -> str:
    """"Kritis" jika proporsi flow (QoS Buruk + Anomali) > kritis_ratio; selain itu label mayoritas."""
    if not last_labels:
        return STATUS_EMPTY
    scored = {k: v for k, v in last_labels.items() if v != INSUFFICIENT}  # "Data Belum Cukup" tidak dihitung
    if not scored:
        return INSUFFICIENT
    bad = sum(card_category(v) in (CATEGORY_BURUK, CATEGORY_ANOMALI) for v in scored.values())
    if bad / len(scored) > kritis_ratio:
        return STATUS_CRITICAL
    return Counter(scored.values()).most_common(1)[0][0]


def safe_name(text: str) -> str:
    """Nama aman untuk berkas."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "sesi"


class MonitoringSession:
    """Satu sesi pemantauan: worker thread → queue.Queue → konsumen (UI/CLI); riwayat CSV real-time."""

    def __init__(
        self,
        cfg: dict[str, Any],
        engine: InferenceEngine,
        source: Iterable | Callable[[threading.Event], Iterable],
        interface: str,
    ) -> None:
        self.cfg, self.engine, self.interface = cfg, engine, interface
        self._source = source
        self.stop_event = threading.Event()
        self.results: queue.Queue[pd.DataFrame] = queue.Queue()
        self.error: str | None = None
        self.started_at = datetime.now()
        reports = resolve_path(cfg["paths"]["reports_dir"]) / "riwayat"
        reports.mkdir(parents=True, exist_ok=True)
        self.history_dir = reports
        suffix = uuid.uuid4().hex[:SESSION_SUFFIX_LEN]  # mencegah bentrok ID untuk sesi pada detik yang sama
        self.session_id = f"{safe_name(interface)}_{self.started_at.strftime(TIME_FORMAT)}_{suffix}"
        self.window_csv = reports / f"jendela_{self.session_id}.csv"
        self.calls_csv = reports / f"panggilan_{self.session_id}.csv"
        self.call_results: queue.Queue[pd.DataFrame] = queue.Queue()
        self._tracker = CallTracker(cfg["monitoring"]["feature_mode"])
        self.n_calls = 0
        self.summary_path: Path | None = None
        self.n_windows_rows = 0
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._last_label: dict[str, str] = {}
        self._finished = False

    # ---- kontrol ----
    def start(self) -> None:
        """Mulai worker thread."""
        if self._thread is not None:
            raise RuntimeError("Sesi sudah dimulai")
        self._thread = threading.Thread(target=self._run, name=f"monitor-{self.session_id}", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        """True selama worker masih berjalan."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        """Tunggu worker selesai (mis. replay habis); True jika sudah selesai."""
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.running

    def stop(self) -> Path:
        """Hentikan worker, tulis ringkasan sesi, dan kembalikan path ringkasan (idempoten)."""
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self.summary_path is None:
                self.summary_path = self._write_summary()
        return self.summary_path

    # ---- data untuk UI ----
    def card_counts(self) -> dict[str, int]:
        """Hitung flow unik berdasarkan label terakhir: total, 4 kategori, dan "Data Belum Cukup"."""
        with self._lock:
            labels = list(self._last_label.values())
        counts = Counter(card_category(v) for v in labels)
        return {
            "Total Aliran Trafik": len(labels),
            **{k: counts.get(k, 0) for k in CARD_KEYS},
            INSUFFICIENT: counts.get(INSUFFICIENT, 0),
        }

    def drain(self) -> list[pd.DataFrame]:
        """Ambil semua hasil jendela yang menunggu di queue."""
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def drain_calls(self) -> list[pd.DataFrame]:
        """Ambil semua baris panggilan selesai yang menunggu di queue."""
        out = []
        while True:
            try:
                out.append(self.call_results.get_nowait())
            except queue.Empty:
                return out

    # ---- internal ----
    def _run(self) -> None:
        mon = self.cfg["monitoring"]
        agg = WindowAggregator(
            mon["window_sec"], mon["feature_mode"], mon["min_packets_per_window"], mon["flow_idle_timeout_sec"],
            mon["call_end_idle_sec"],
        )
        try:
            source = self._source(self.stop_event) if callable(self._source) else self._source
            try:
                for pkt in source:
                    if self.stop_event.is_set():
                        break
                    # sumber live mengeluarkan None sebagai detak saat idle → majukan jam dinding
                    done = agg.advance(time.time()) if pkt is None else agg.add(pkt)
                    if done:
                        self._emit(done)
                    self._emit_calls(agg.pop_ended())
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()  # hentikan sniffer live
            self._emit(agg.flush())
            self._emit_calls(agg.pop_ended() + agg.end_all())  # akhir sumber / Stop: semua flow tersisa selesai
        except Exception as exc:  # error capture ditampilkan di UI (MON-8)
            logger.exception("Sesi %s gagal", self.session_id)
            self.error = f"{type(exc).__name__}: {exc}"

    def _emit(self, flows: list[WindowFlow]) -> None:
        df = self.engine.predict(flows, self.session_id)
        if df.empty:
            return
        df.to_csv(self.window_csv, mode="a", header=not self.window_csv.exists(), index=False)  # MON-7
        with self._lock:
            self._last_label.update(dict(zip(df["flow_id"], df["label"])))
            self.n_windows_rows += len(df)
            self._tracker.update(df)
        self.results.put(df)

    def _emit_calls(self, ends: list[FlowEnd]) -> None:
        if not ends:
            return
        with self._lock:
            calls = self._tracker.finish(ends)
            if calls.empty:
                return
            calls.to_csv(self.calls_csv, mode="a", header=not self.calls_csv.exists(), index=False)
            self.n_calls += len(calls)
        self.call_results.put(calls)

    def _write_summary(self) -> Path:
        ended = datetime.now()
        ev = self.engine.art.profile["evaluation"]
        calls = pd.read_csv(self.calls_csv) if self.calls_csv.is_file() else pd.DataFrame(columns=list(CALL_COLUMNS))
        final = count_final_labels(calls)
        with self._lock:
            status = overall_status(self._last_label, self.cfg["monitoring"]["kritis_ratio"])
            total = len(self._last_label)
        row = {
            "id_sesi_pemantauan": self.session_id,
            "waktu_mulai": self.started_at.isoformat(timespec="seconds"),
            "waktu_selesai": ended.isoformat(timespec="seconds"),
            "total_aliran": total,
            "jumlah_klaster": ev["n_clusters"],
            "skor_silhouette": ev["silhouette"],
            "skor_chi": ev["chi"],
            "noise_ratio": ev["noise_ratio"],
            "status_keseluruhan": status,
            "lokasi_laporan": str(self.window_csv),
            "jumlah_panggilan": len(calls),
            "panggilan_baik": final["QoS Baik"],
            "panggilan_cukup": final["QoS Cukup"],
            "panggilan_buruk": final["QoS Buruk"],
            "panggilan_anomali": final["Anomali"],
            "panggilan_belum_cukup": final[INSUFFICIENT],
        }
        path = self.history_dir / f"riwayat_pemantauan_{ended.strftime(TIME_FORMAT)}.csv"
        n = 1
        while path.exists():
            path = self.history_dir / f"riwayat_pemantauan_{ended.strftime(TIME_FORMAT)}_{n}.csv"
            n += 1
        pd.DataFrame([row])[list(SUMMARY_COLUMNS)].to_csv(path, index=False)
        logger.info("Ringkasan sesi %s ditulis ke %s", self.session_id, path)
        return path
