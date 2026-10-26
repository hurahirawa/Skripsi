"""Mesin inferensi real-time dan sesi pemantauan (PRD MON-5…MON-8, §10.3)."""
from __future__ import annotations

import atexit
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd

from qos.capture.flow_table import FlowEnd, WindowAggregator, WindowFlow, flow_ports
from qos.config import resolve_path
from qos.features.extract import derive_metrics
from qos.features.preprocess import build_features, impute_median
from qos.model.assignment import NOISE_ID, assign, format_distances, parse_distances
from qos.model.versions import LEGACY_MESSAGE, REQUIRED_FILES, is_legacy
from qos.monitor.calls import (
    CALL_COLUMNS,
    INSUFFICIENT,
    OUTSIDE_KEY,
    SIGNALING,
    STATUS_CLUSTERED,
    STATUS_OUTSIDE,
    CallTracker,
    cluster_key,
    summarize_calls,
)

logger = logging.getLogger(__name__)

CURRENT_FILE = "current.txt"
WINDOW_COLUMNS: tuple[str, ...] = (
    "waktu_jendela", "id_sesi", "flow_id", "delay_ms", "jitter_ms", "fwd", "bwd", "fai_pct",
    "status", "klaster_terdekat", "jarak_terdekat", "di_luar_klaster", "jarak_ke_klaster", "klaster_kmeans5",
    "pc1", "pc2", "feature_mode", "diimputasi",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "id_sesi_pemantauan", "waktu_mulai", "waktu_selesai", "total_aliran", "jumlah_klaster",
    "skor_silhouette", "skor_dbcv", "skor_chi", "noise_ratio", "status_sesi", "lokasi_laporan",
    "jumlah_panggilan", "panggilan_per_klaster", "panggilan_di_luar_klaster", "persen_di_luar_klaster",
    "klaster_mayoritas", "panggilan_belum_cukup",
)
STATUS_DONE, STATUS_EMPTY = "Selesai", "Tidak ada data"
STATUS_ABNORMAL = "Dihentikan tidak normal"
TIME_FORMAT = "%Y%m%d_%H%M%S"
US_PER_MS = 1000.0
SESSION_SUFFIX_LEN = 6


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
    assignment: dict[str, Any]

    @property
    def centroids(self) -> dict[int, np.ndarray]:
        """Centroid HDBSCAN (ruang ter-skala) per nomor klaster 1..k."""
        return {int(k): np.asarray(v, dtype=float) for k, v in self.assignment["hdbscan"]["centroid"].items()}

    @property
    def limits(self) -> dict[int, float]:
        """Batas jarak "di luar klaster" per klaster (aturan tersimpan di assignment.json)."""
        return {int(k): float(v) for k, v in self.assignment["hdbscan"]["batas"].items()}

    @property
    def kmeans_centroids(self) -> dict[int, np.ndarray]:
        """Centroid K-Means pembanding (k = jumlah klaster HDBSCAN)."""
        return {int(k): np.asarray(v, dtype=float) for k, v in self.assignment["kmeans_n"]["centroid"].items()}

    @property
    def cluster_ids(self) -> list[int]:
        """Nomor klaster HDBSCAN (1..k)."""
        return sorted(self.centroids)

    @property
    def descriptions(self) -> dict[int, str]:
        """Nomor klaster → deskripsi ringkas posisi terhadap ambang ITU (deskriptif, bukan label)."""
        return {int(c["id_klaster"]): c["deskripsi"]["ringkas"] for c in self.profile["clusters"] if int(c["id_klaster"]) != NOISE_ID}


def load_artifacts(cfg: dict[str, Any], version: str | None = None) -> Artifacts:
    """Muat artefak versi `version`, atau versi aktif di current.txt. Error jelas jika tidak ada."""
    root = resolve_path(cfg["paths"]["artifacts_dir"])
    if version is None:
        current = root / CURRENT_FILE
        if not current.is_file():
            raise ModelNotFoundError(f"Tidak ada model aktif ({current} tidak ada). Latih model: python scripts/train.py")
        version = current.read_text("utf-8").strip()
    folder = root / version
    if folder.is_dir() and is_legacy(folder):
        raise ModelNotFoundError(LEGACY_MESSAGE.format(v=version))
    missing = [n for n in REQUIRED_FILES if not (folder / n).is_file()]
    if missing:
        raise ModelNotFoundError(f"Artefak versi {version} tidak lengkap, kurang: {missing}")
    return Artifacts(
        version=version,
        model=joblib.load(folder / "hdbscan_model.joblib"),
        scaler=joblib.load(folder / "scaler.joblib"),
        pca=joblib.load(folder / "pca.joblib"),
        profile=json.loads((folder / "cluster_profile.json").read_text("utf-8")),
        meta=json.loads((folder / "train_meta.json").read_text("utf-8")),
        assignment=json.loads((folder / "assignment.json").read_text("utf-8")),
    )


class InferenceEngine:
    """Fitur jendela → scaler.transform → jarak Euclidean ke centroid → klaster terdekat (scaler TIDAK di-fit ulang)."""

    def __init__(self, art: Artifacts, cfg: dict[str, Any]) -> None:
        self.art, self.cfg = art, cfg
        self.centroids, self.limits, self.km_centroids = art.centroids, art.limits, art.kmeans_centroids
        self.n_insufficient = 0  # baris "Data Belum Cukup"
        self.signaling_ports = {int(p) for p in self.cfg["monitoring"].get("signaling_ports") or []}

    def predict(self, flows: list[WindowFlow], session_id: str) -> pd.DataFrame:
        """Klaster terdekat tiap WindowFlow; kolom sesuai WINDOW_COLUMNS."""
        if not flows:
            return pd.DataFrame(columns=list(WINDOW_COLUMNS))
        feat, meta = self.cfg["features"], self.art.meta
        live = self.cfg["monitoring"]["min_session_live"]
        min_fwd, min_bwd = max(live["fwd"] or 0, 1), live["bwd"] or 0  # fwd = 0 membuat FAI tak terdefinisi
        signaling = np.array([bool(self.signaling_ports.intersection(flow_ports(f.flow_id))) for f in flows])
        enough = np.array([f.fwd >= min_fwd and f.bwd >= min_bwd for f in flows]) & ~signaling
        self.n_insufficient += int((~enough & ~signaling).sum())

        fwd = np.array([f.fwd for f in flows], dtype=float)
        bwd = np.array([f.bwd for f in flows], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fai = np.where(fwd > 0, (fwd - bwd) / fwd * 100, np.nan)
        if meta["fai_clip_negative"]:
            fai = np.where(np.isnan(fai), fai, np.maximum(fai, 0))
        # baris "Data Belum Cukup": nilai mentah dicatat, tanpa penugasan (klaster/jarak/PC kosong)
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
                "status": INSUFFICIENT,
                "klaster_terdekat": pd.array([pd.NA] * len(flows), dtype="Int64"),
                "jarak_terdekat": np.nan,
                "di_luar_klaster": pd.array([pd.NA] * len(flows), dtype="boolean"),
                "jarak_ke_klaster": "",
                "klaster_kmeans5": pd.array([pd.NA] * len(flows), dtype="Int64"),
                "pc1": np.nan,
                "pc2": np.nan,
                "feature_mode": self.cfg["monitoring"]["feature_mode"],
                "diimputasi": False,  # baris tanpa penugasan tidak diimputasi
            }
        )
        out.loc[signaling, "status"] = SIGNALING  # flow sinyal: nilai mentah dicatat, tanpa penugasan
        if not enough.any():
            return out[list(WINDOW_COLUMNS)]

        # MON-5: nilai kosong → median artefak latih (hanya untuk baris yang ditugaskan)
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
        X = np.asarray(self.art.scaler.transform(build_features(d, meta["cluster_columns"], meta["log_transform"])), dtype=float)
        a = assign(X, self.centroids, self.limits)
        km = assign(X, self.km_centroids)
        pcs = self.art.pca.transform(X)

        idx = out.index[enough]
        out.loc[idx, "delay_ms"] = d["delay_ms"].to_numpy()  # nilai setelah imputasi median
        out.loc[idx, "jitter_ms"] = d["jitter_ms"].to_numpy()
        out.loc[idx, "fai_pct"] = d["fai_pct"].to_numpy()
        out.loc[idx, "status"] = np.where(a.outside, STATUS_OUTSIDE, STATUS_CLUSTERED)
        out.loc[idx, "klaster_terdekat"] = pd.array(a.nearest, dtype="Int64")
        out.loc[idx, "jarak_terdekat"] = a.nearest_dist
        out.loc[idx, "di_luar_klaster"] = pd.array(a.outside, dtype="boolean")
        out.loc[idx, "jarak_ke_klaster"] = [format_distances(a.ids, row) for row in a.dist]
        out.loc[idx, "klaster_kmeans5"] = pd.array(km.nearest, dtype="Int64")
        out.loc[idx, "pc1"] = pcs[:, 0]
        out.loc[idx, "pc2"] = pcs[:, 1]
        out.loc[idx, "diimputasi"] = imputed
        return out[list(WINDOW_COLUMNS)]


def state_key(status: str, cluster: Any = None) -> str:
    """Kunci kartu/sparkline: "Klaster N", "Di luar klaster", "Data Belum Cukup", atau "Sinyal SIP"."""
    if status == STATUS_CLUSTERED:
        return cluster_key(cluster)
    if status == STATUS_OUTSIDE:
        return OUTSIDE_KEY
    return status


STATUS_WAITING = "Menunggu data"


def connection_status(active: list[dict[str, Any]]) -> dict[str, Any]:
    """Status koneksi saat ini dari panggilan aktif pada jendela terbaru (tanpa makna baik/buruk).

    Kelompok terbanyak menentukan banner: "Klaster N" (jarak = median jarak panggilan di kelompok itu) atau
    "Di luar klaster" (terdekat = klaster terdekat yang paling sering). Seri → klaster bernomor kecil; di luar klaster
    menang hanya bila lebih banyak. Tanpa penugasan: "Data Belum Cukup" bila ada panggilan aktif, selain itu "Menunggu data".
    """
    scored = [c for c in active if c["status"] in (STATUS_CLUSTERED, STATUS_OUTSIDE)]
    if not scored:
        key = INSUFFICIENT if active else STATUS_WAITING
        return {"teks": key, "kunci": key, "klaster": None, "jarak": None, "di_luar": False, "n_aktif": len(active), "n_kelompok": 0}
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in scored:
        groups.setdefault(state_key(c["status"], c["klaster"]), []).append(c)
    order = sorted((k for k in groups if k != OUTSIDE_KEY), key=lambda k: int(k.split()[-1])) + ([OUTSIDE_KEY] if OUTSIDE_KEY in groups else [])
    top = max(order, key=lambda k: len(groups[k]))  # max mengambil yang pertama bila seri: klaster bernomor kecil, di luar klaster terakhir
    members = groups[top]
    dist = float(np.median([m["jarak"] for m in members]))
    nearest = Counter(int(m["klaster"]) for m in members).most_common(1)[0][0] if top == OUTSIDE_KEY else int(members[0]["klaster"])
    text = f"{OUTSIDE_KEY} (terdekat: {cluster_key(nearest)})" if top == OUTSIDE_KEY else f"paling dekat ke {cluster_key(nearest)}"
    return {"teks": text, "kunci": top, "klaster": nearest, "jarak": dist, "di_luar": top == OUTSIDE_KEY,
            "n_aktif": len(active), "n_kelompok": len(members)}


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
        self._last_state: dict[str, dict[str, Any]] = {}  # flow_id → status/klaster terakhir
        self._signaling: set[str] = set()  # flow sinyal: tidak masuk kartu, status, panggilan
        self.n_packets = 0  # status capture: paket yang masuk ke aggregator
        self.last_packet_at: float | None = None  # jam dinding paket terakhir
        self.active_flows = 0
        self._active: dict[str, dict[str, Any]] = {}  # panggilan aktif: nilai terbaru + riwayat per jendela
        atexit.register(self._on_exit)  # aplikasi ditutup tanpa Stop → ringkasan "Dihentikan tidak normal"
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
        atexit.unregister(self._on_exit)
        return self.summary_path

    def _on_exit(self) -> None:
        """Hook atexit: sesi yang belum di-Stop ditutup dan diringkas dengan status abnormal."""
        if self._thread is None or self.summary_path is not None:
            return
        try:
            self.stop_event.set()
            self._thread.join(self.cfg["monitoring"]["exit_join_sec"])
            self._finalize_abnormal()
        except Exception:  # noqa: BLE001 — jangan ganggu proses keluar
            logger.exception("Gagal menulis ringkasan sesi %s saat keluar", self.session_id)

    def _finalize_abnormal(self) -> None:
        with self._lock:
            if self.summary_path is None:
                self.summary_path = self._write_summary(STATUS_ABNORMAL)

    # ---- status live untuk dashboard ----
    def active_calls(self) -> list[dict[str, Any]]:
        """Panggilan aktif (bukan Sinyal SIP): klaster terdekat, jarak, jarak ke semua centroid, dan riwayat per jendela."""
        with self._lock:
            out = [
                {
                    "flow_id": fid, "durasi_s": a["last_end"] - a["first_start"], "status": a["status"],
                    "kunci": state_key(a["status"], a["klaster"]), "klaster": a["klaster"], "jarak": a["jarak"],
                    "di_luar": a["status"] == STATUS_OUTSIDE, "jarak_ke_klaster": a["jarak_ke_klaster"],
                    "delay_ms": a["delay_ms"], "jitter_ms": a["jitter_ms"], "fai_pct": a["fai_pct"], "riwayat": list(a["history"]),
                }
                for fid, a in self._active.items()
            ]
        return sorted(out, key=lambda c: c["durasi_s"], reverse=True)

    def capture_stats(self) -> dict[str, Any]:
        """Status capture: paket tertangkap, flow aktif, dan umur paket terakhir (detik; None jika belum ada)."""
        with self._lock:
            last = self.last_packet_at
            n = self.n_packets
        return {
            "paket": n, "flow_aktif": self.active_flows,
            "paket_terakhir": None if last is None else datetime.fromtimestamp(last),
            "detik_sejak_paket": None if last is None else max(0.0, time.time() - last),
        }

    # ---- data untuk UI ----
    def card_counts(self) -> dict[str, int]:
        """Jumlah flow unik menurut status terakhir: total, tiap klaster, Di luar klaster, Data Belum Cukup, Sinyal SIP."""
        with self._lock:
            states = list(self._last_state.values())
            n_signaling = len(self._signaling)
        counts = Counter(state_key(s["status"], s["klaster"]) for s in states)
        return {
            "Total Aliran Trafik": len(states),
            **{cluster_key(c): counts.get(cluster_key(c), 0) for c in self.engine.art.cluster_ids},
            OUTSIDE_KEY: counts.get(OUTSIDE_KEY, 0),
            INSUFFICIENT: counts.get(INSUFFICIENT, 0),
            SIGNALING: n_signaling,
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
                    if pkt is not None:
                        with self._lock:
                            self.n_packets += 1
                            self.last_packet_at = time.time()
                    # sumber live mengeluarkan None sebagai detak saat idle → majukan jam dinding
                    done = agg.advance(time.time()) if pkt is None else agg.add(pkt)
                    if done:
                        self._emit(done)
                    self._emit_calls(agg.pop_ended())
                    self.active_flows = agg.active_flows
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()  # hentikan sniffer live
            self._emit(agg.flush())
            self._emit_calls(agg.pop_ended() + agg.end_all())  # akhir sumber / Stop: semua flow tersisa selesai
        except Exception as exc:  # error capture ditampilkan di UI (MON-8)
            logger.exception("Sesi %s gagal", self.session_id)
            self.error = f"{type(exc).__name__}: {exc}"
            self._finalize_abnormal()  # sesi terputus oleh error: tetap ada di riwayat

    def _emit(self, flows: list[WindowFlow]) -> None:
        df = self.engine.predict(flows, self.session_id)
        if df.empty:
            return
        df.to_csv(self.window_csv, mode="a", header=not self.window_csv.exists(), index=False)  # MON-7
        with self._lock:
            is_sig = df["status"] == SIGNALING
            self._signaling.update(df.loc[is_sig, "flow_id"])
            scored = df[~is_sig]
            for r in scored.to_dict("records"):
                self._last_state[r["flow_id"]] = {"status": r["status"], "klaster": r["klaster_terdekat"]}
            self.n_windows_rows += len(df)
            self._tracker.update(scored)
            self._update_active(flows, df)
        self.results.put(df)

    def _update_active(self, flows: list[WindowFlow], df: pd.DataFrame) -> None:
        spark = self.cfg["monitoring"]["sparkline_windows"]
        for f, row in zip(flows, df.to_dict("records")):
            if row["status"] == SIGNALING:
                continue
            a = self._active.setdefault(
                f.flow_id, {"first_start": f.window_start, "history": deque(maxlen=spark)}
            )
            cluster = None if pd.isna(row["klaster_terdekat"]) else int(row["klaster_terdekat"])
            a.update(last_end=f.window_end, status=row["status"], klaster=cluster,
                     jarak=None if pd.isna(row["jarak_terdekat"]) else float(row["jarak_terdekat"]),
                     jarak_ke_klaster=parse_distances(row["jarak_ke_klaster"]), delay_ms=row["delay_ms"],
                     jitter_ms=row["jitter_ms"], fai_pct=row["fai_pct"])
            a["history"].append(state_key(row["status"], cluster))

    def _emit_calls(self, ends: list[FlowEnd]) -> None:
        if not ends:
            return
        with self._lock:
            for end in ends:
                self._active.pop(end.flow_id, None)  # panggilan selesai → bukan aktif lagi
            calls = self._tracker.finish(ends)
            if calls.empty:
                return
            calls.to_csv(self.calls_csv, mode="a", header=not self.calls_csv.exists(), index=False)
            self.n_calls += len(calls)
        self.call_results.put(calls)

    def _write_summary(self, status_override: str | None = None) -> Path:
        ended = datetime.now()
        ev = self.engine.art.profile["evaluation"]
        calls = pd.read_csv(self.calls_csv) if self.calls_csv.is_file() else pd.DataFrame(columns=list(CALL_COLUMNS))
        with self._lock:
            total = len(self._last_state)
        row = {
            "id_sesi_pemantauan": self.session_id,
            "waktu_mulai": self.started_at.isoformat(timespec="seconds"),
            "waktu_selesai": ended.isoformat(timespec="seconds"),
            "total_aliran": total,
            "jumlah_klaster": ev["n_clusters"],
            "skor_silhouette": ev["silhouette"],
            "skor_dbcv": ev.get("dbcv"),
            "skor_chi": ev["chi"],
            "noise_ratio": ev["noise_ratio"],
            "status_sesi": status_override or (STATUS_DONE if total else STATUS_EMPTY),
            "lokasi_laporan": str(self.window_csv),
            **summarize_calls(calls),
        }
        path = self.history_dir / f"riwayat_pemantauan_{ended.strftime(TIME_FORMAT)}.csv"
        n = 1
        while path.exists():
            path = self.history_dir / f"riwayat_pemantauan_{ended.strftime(TIME_FORMAT)}_{n}.csv"
            n += 1
        pd.DataFrame([row])[list(SUMMARY_COLUMNS)].to_csv(path, index=False)
        logger.info("Ringkasan sesi %s ditulis ke %s", self.session_id, path)
        return path
