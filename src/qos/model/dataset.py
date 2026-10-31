"""Data latih per versi dan unggah CSV (mode Tambah / Ganti): deteksi format, filter per tahap, gabung, dedup.

Setiap versi artefak menyimpan `data_latih.csv` (SELURUH baris latih, kolom kanonik). Unggahan baru dapat berupa
`panggilan_<id_sesi>.csv` (hasil monitoring) atau CSV flow format darknet/cic/bccc (column_map config).
Modul ini tidak melatih model; pelatihan ada di qos.model.retrain.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from qos.config import resolve_path
from qos.data.loader import CSV_ENCODING, detect_format, detect_sep, filter_voip, load_csv
from qos.features.preprocess import exclude_types, session_mask
from qos.monitor.calls import INSUFFICIENT, SIGNALING

logger = logging.getLogger(__name__)

DATA_FILE = "data_latih.csv"
NUMERIC_COLUMNS: tuple[str, ...] = (
    "iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count", "duration_us", "protocol",
)
META_COLUMNS: tuple[str, ...] = ("sumber", "id_baris", "waktu_ditambahkan")
DATA_COLUMNS: tuple[str, ...] = (*NUMERIC_COLUMNS, *META_COLUMNS)
MODES: tuple[str, ...] = ("awal", "tambah", "ganti")
UPLOAD_MODES: tuple[str, ...] = ("tambah", "ganti")
FORMAT_PANGGILAN = "panggilan"
PANGGILAN_REQUIRED: tuple[str, ...] = (
    "flow_id", "durasi_s", "total_fwd", "total_bwd", "delay_ms", "jitter_ms", "status_akhir",
)
PANGGILAN_HINT: tuple[str, ...] = ("flow_id", "total_fwd", "total_bwd", "status_akhir")  # penanda format panggilan
US_PER_MS = 1000.0
US_PER_S = 1_000_000.0
DEDUP_DECIMALS = 6
BOOL_MAP = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
TIME_FORMAT_SECONDS = "seconds"


class DatasetError(RuntimeError):
    """Berkas unggahan tidak bisa dipakai (format salah, kolom kurang) atau data lama tidak tersedia."""


@dataclass
class FileReport:
    """Hasil pemeriksaan satu berkas: format, jumlah baris per tahap filter, dan baris akhir yang diterima."""

    name: str
    format: str | None = None
    stages: list[tuple[str, int]] = field(default_factory=list)
    accepted: int = 0
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def add(self, label: str, count: int) -> None:
        self.stages.append((label, int(count)))


@dataclass
class Prepared:
    """Hasil persiapan data latih (belum melatih): data gabungan, laporan per berkas, dan masalah yang menghalangi."""

    mode: str
    parent_version: str | None
    frame: pd.DataFrame            # data latih gabungan (kolom DATA_COLUMNS)
    new_rows: pd.DataFrame         # hanya baris baru yang lolos
    files: list[FileReport]
    n_old: int                     # baris lama yang dipakai (0 pada mode Ganti)
    n_old_dropped: int             # baris lama yang dibuang (mode Ganti)
    min_rows: int
    warn_rows: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return len(self.new_rows)

    @property
    def n_total(self) -> int:
        return len(self.frame)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(self.frame["sumber"].tolist())) if len(self.frame) else []

    def info(self) -> dict[str, Any]:
        """Ringkasan untuk train_meta.json (kunci `data_latih`)."""
        return data_info(self.mode, self.parent_version, self.n_old, self.n_new, self.sources, self.n_old_dropped)


def data_info(mode: str, parent: str | None, n_old: int, n_new: int, sources: list[str], n_old_dropped: int = 0) -> dict[str, Any]:
    """Catatan data latih untuk train_meta.json."""
    return {
        "mode": mode, "versi_induk": parent, "jumlah_baris_lama": int(n_old), "jumlah_baris_baru": int(n_new),
        "jumlah_total": int(n_old + n_new), "jumlah_baris_lama_dibuang": int(n_old_dropped), "file_sumber": list(sources),
    }


def now_text() -> str:
    return datetime.now().isoformat(timespec=TIME_FORMAT_SECONDS)


def ensure_canonical(df: pd.DataFrame, sumber: str = "-", prefix: str = "awal", stamp: str | None = None) -> pd.DataFrame:
    """Tabel dengan kolom kanonik DATA_COLUMNS; kolom hilang diisi NaN / `sumber` / id unik / waktu sekarang."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    out["sumber"] = df["sumber"] if "sumber" in df.columns else sumber
    if "id_baris" in df.columns:
        out["id_baris"] = df["id_baris"]
    else:
        tag = stamp or f"{prefix}-{uuid4().hex[:6]}"
        out["id_baris"] = [f"{tag}-{i:06d}" for i in range(1, len(df) + 1)]
    out["waktu_ditambahkan"] = df["waktu_ditambahkan"] if "waktu_ditambahkan" in df.columns else now_text()
    return out.reset_index(drop=True)


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """Kunci duplikat tiap baris (hash 6 kolom numerik, dibulatkan agar derau float tidak menggagalkan)."""
    return pd.Series(pd.util.hash_pandas_object(df[list(NUMERIC_COLUMNS)].round(DEDUP_DECIMALS), index=False).to_numpy(), index=df.index)


# ------------------------------------------------------------------------------------------ deteksi format
def _read_header(path: Path) -> list[str]:
    with path.open(encoding=CSV_ENCODING, errors="replace") as fh:
        line = fh.readline()
    sep = detect_sep(line)
    return [c.strip().strip('"') for c in line.rstrip("\r\n").split(sep)]


def detect_upload_format(path: str | Path, data_cfg: dict[str, Any]) -> str:
    """`panggilan` atau nama format di column_map (darknet/cic/bccc). Format lain → DatasetError dengan kolom yang kurang."""
    path = Path(path)
    try:
        columns = _read_header(path)
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Berkas tidak bisa dibaca: {exc}") from exc
    available = set(columns)
    if not columns or columns == [""]:
        raise DatasetError("Berkas kosong atau tanpa header.")
    if sum(h in available for h in PANGGILAN_HINT) >= 2:  # tampak seperti panggilan_<id_sesi>.csv
        missing = [c for c in PANGGILAN_REQUIRED if c not in available]
        if missing:
            raise DatasetError(f"Format panggilan tidak lengkap: kolom yang kurang {missing}.")
        return FORMAT_PANGGILAN
    try:
        return detect_format(columns, data_cfg["column_map"])
    except ValueError as exc:
        missing = [c for c in PANGGILAN_REQUIRED if c not in available]
        raise DatasetError(
            f"Format CSV tidak dikenali. panggilan: kurang {missing}; "
            + str(exc).removeprefix("Format CSV tidak dikenali. ")
        ) from exc


# ------------------------------------------------------------------------------------------- pembacaan
def _protocol_from_flow_id(flow_id: pd.Series) -> pd.Series:
    """Nomor protokol dari bagian akhir flow_id ("a:p-b:q-17" → 17); tidak terbaca → NaN."""
    return pd.to_numeric(flow_id.astype("string").str.rsplit("-", n=1).str[-1], errors="coerce")


def _load_panggilan(path: Path, rep: FileReport) -> pd.DataFrame:
    """panggilan_<id_sesi>.csv → tabel kanonik: satu baris per panggilan; delay/jitter ms dikembalikan ke µs."""
    raw = pd.read_csv(path, encoding=CSV_ENCODING, encoding_errors="replace")
    raw.columns = [c.strip() for c in raw.columns]
    rep.add("baris di berkas", len(raw))
    df = raw[raw["status_akhir"] != SIGNALING]
    rep.add("tanpa Sinyal SIP", len(df))
    df = df[df["status_akhir"] != INSUFFICIENT]
    rep.add("tanpa Data Belum Cukup", len(df))
    if "catatan" in df.columns:  # catatan terisi = mode window (nilai bukan kumulatif)
        df = df[df["catatan"].isna() | (df["catatan"].astype("string").str.strip() == "")]
    rep.add("tanpa mode window", len(df))
    if "diimputasi" in df.columns:
        df = df[df["diimputasi"].map(BOOL_MAP) != True]  # noqa: E712
        rep.add("tanpa baris diimputasi", len(df))
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    return pd.DataFrame(
        {
            "iat_mean_us": num("delay_ms") * US_PER_MS,
            "iat_std_us": num("jitter_ms") * US_PER_MS,
            "fwd_packets_count": num("total_fwd"),
            "bwd_packets_count": num("total_bwd"),
            "duration_us": num("durasi_s") * US_PER_S,
            "protocol": _protocol_from_flow_id(df["flow_id"]),
        }
    ).reset_index(drop=True)


def _load_flow_csv(path: Path, data_cfg: dict[str, Any], rep: FileReport) -> pd.DataFrame:
    """CSV flow (darknet/cic/bccc): filter VoIP lalu exclude_traffic_types, seperti pelatihan awal."""
    raw = load_csv(path, data_cfg)
    rep.add("baris di berkas", len(raw))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # peringatan min_flows_warning khusus pelatihan awal
            voip = filter_voip(raw, data_cfg["voip_filter"])
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    rep.add("setelah filter VoIP", len(voip))
    if "traffic_type" in voip.columns:
        kept = exclude_types(voip, data_cfg["exclude_traffic_types"])
        rep.add(f"tanpa {', '.join(data_cfg['exclude_traffic_types']) or '-'}", len(kept))
    else:
        kept = voip
        if data_cfg["exclude_traffic_types"]:
            rep.notes.append("Format ini tidak punya kolom jenis traffic; exclude_traffic_types dilewati.")
    return kept[list(NUMERIC_COLUMNS)].reset_index(drop=True)


def _clean(df: pd.DataFrame, cfg: dict[str, Any], rep: FileReport) -> pd.DataFrame:
    """Filter sesi minimum lalu buang NaN/inf/negatif pada kolom numerik."""
    ms = cfg["data"]["min_session"]
    df = df[session_mask(df, ms["fwd"], ms["bwd"])]
    rep.add("setelah filter sesi minimum", len(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df.notna().all(axis=1) & (df >= 0).all(axis=1)
    df = df[valid]
    rep.add("setelah buang NaN/inf/negatif", len(df))
    return df.reset_index(drop=True)


def load_upload(path: str | Path, cfg: dict[str, Any], name: str | None = None) -> tuple[pd.DataFrame, FileReport]:
    """Baca satu berkas unggahan sampai tahap bersih (belum dedup). Gagal → FileReport.error terisi dan tabel kosong."""
    path = Path(path)
    rep = FileReport(name=name or path.name)
    try:
        rep.format = detect_upload_format(path, cfg["data"])
        df = _load_panggilan(path, rep) if rep.format == FORMAT_PANGGILAN else _load_flow_csv(path, cfg["data"], rep)
        return _clean(df, cfg, rep), rep
    except (DatasetError, ValueError, KeyError, pd.errors.ParserError) as exc:
        rep.error = str(exc)
        return pd.DataFrame(columns=list(NUMERIC_COLUMNS)), rep


# ----------------------------------------------------------------------------------------- data versi
def data_path(cfg: dict[str, Any], version: str) -> Path:
    return resolve_path(cfg["paths"]["artifacts_dir"]) / version / DATA_FILE


def load_version_data(cfg: dict[str, Any], version: str) -> pd.DataFrame:
    """data_latih.csv sebuah versi. Tidak ada → DatasetError dengan petunjuk membuatnya."""
    path = data_path(cfg, version)
    if not path.is_file():
        raise DatasetError(
            f"Versi {version} belum punya {DATA_FILE}. Buat dulu tanpa refit: python scripts/backfill_data_latih.py --version {version}"
        )
    return pd.read_csv(path)


def prepare_dataset(
    cfg: dict[str, Any], uploads: list[tuple[str | Path, str]], mode: str, parent_version: str | None
) -> Prepared:
    """Gabungkan unggahan baru dengan data versi induk menurut `mode` (tambah | ganti). Tidak melatih dan tidak menulis berkas.

    uploads: daftar (path, nama asli). Duplikat dibuang di dalam data baru dan terhadap data lama (mode Tambah).
    Masalah yang menghalangi pelatihan dicatat di `problems` (berkas ditolak, data kurang dari retrain.min_rows).
    """
    if mode not in UPLOAD_MODES:
        raise DatasetError(f"Mode tidak dikenal: {mode!r} (pilihan: {list(UPLOAD_MODES)})")
    rcfg = cfg["retrain"]
    old, n_old_dropped = pd.DataFrame(columns=list(DATA_COLUMNS)), 0
    problems: list[str] = []
    if mode == "tambah":
        if parent_version is None:
            raise DatasetError("Mode Tambah butuh model aktif sebagai induk (tidak ada versi aktif).")
        old = load_version_data(cfg, parent_version)
    elif parent_version is not None and data_path(cfg, parent_version).is_file():
        n_old_dropped = len(pd.read_csv(data_path(cfg, parent_version), usecols=["id_baris"]))
    if not uploads:
        problems.append("Belum ada berkas CSV yang diunggah.")

    seen = set(_row_keys(old).tolist()) if len(old) else set()
    stamp, when = f"{datetime.now():%Y%m%d%H%M%S}-{uuid4().hex[:6]}", now_text()
    reports: list[FileReport] = []
    parts: list[pd.DataFrame] = []
    for path, name in uploads:
        df, rep = load_upload(path, cfg, name)
        if rep.error:
            problems.append(f"{rep.name}: {rep.error}")
        else:
            keys = _row_keys(df)
            keep = ~keys.duplicated() & ~keys.isin(seen)  # duplikat dalam berkas, antar-berkas, dan terhadap data lama
            seen.update(keys[keep].tolist())
            df = df[keep]
            rep.add("setelah buang duplikat", len(df))
            rep.accepted = len(df)
            parts.append(df.assign(sumber=rep.name))
            if not len(df):
                rep.notes.append("Tidak ada baris yang lolos dari berkas ini.")
        reports.append(rep)

    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*NUMERIC_COLUMNS, "sumber"])
    new = ensure_canonical(new.assign(waktu_ditambahkan=when), stamp=stamp) if len(new) else pd.DataFrame(columns=list(DATA_COLUMNS))
    frame = pd.concat([old[list(DATA_COLUMNS)], new], ignore_index=True) if len(old) else new.copy()
    if frame["id_baris"].duplicated().any():
        raise DatasetError("id_baris tidak unik pada data gabungan.")

    notes: list[str] = []
    if not problems and len(new) == 0:
        problems.append("Tidak ada baris baru yang lolos filter (semua berkas habis atau duplikat).")
    if len(frame) < rcfg["min_rows"]:
        problems.append(f"Data latih {len(frame)} baris < batas minimum retrain.min_rows ({rcfg['min_rows']}); latih ulang ditolak.")
    elif len(frame) < rcfg["warn_rows"]:
        notes.append(f"Data latih hanya {len(frame)} baris (< {rcfg['warn_rows']}); hasil klaster kurang stabil.")
    return Prepared(
        mode=mode, parent_version=parent_version, frame=frame, new_rows=new, files=reports, n_old=len(old),
        n_old_dropped=n_old_dropped, min_rows=rcfg["min_rows"], warn_rows=rcfg["warn_rows"], problems=problems, warnings=notes,
    )


def stage_table(prep: Prepared) -> pd.DataFrame:
    """Jumlah baris per tahap per berkas (satu baris per berkas; tahap yang tak berlaku kosong)."""
    labels = list(dict.fromkeys(label for rep in prep.files for label, _ in rep.stages))
    rows = []
    for rep in prep.files:
        row: dict[str, Any] = {"berkas": rep.name, "format": rep.format or "ditolak"}
        row.update({label: n for label, n in rep.stages})
        row["catatan"] = rep.error or "; ".join(rep.notes)
        rows.append(row)
    return pd.DataFrame(rows, columns=["berkas", "format", *labels, "catatan"])


# -------------------------------------------------------------------------------- data latih versi lama
def backfill_data_latih(cfg: dict[str, Any], version: str, sumber: str) -> Path:
    """Buat data_latih.csv untuk versi yang sudah ada TANPA refit, dari training_sample.csv (harus memuat seluruh baris latih).

    Berkas model tidak disentuh; train_meta.json hanya diberi catatan `data_latih` (mode awal). Versi yang sudah punya data_latih.csv ditolak.
    """
    folder = resolve_path(cfg["paths"]["artifacts_dir"]) / version
    meta_path, sample_path = folder / "train_meta.json", folder / "training_sample.csv"
    if not meta_path.is_file() or not sample_path.is_file():
        raise DatasetError(f"Versi {version} tidak ditemukan atau tidak punya training_sample.csv")
    if (folder / DATA_FILE).is_file():
        raise DatasetError(f"{DATA_FILE} sudah ada di versi {version}; tidak ditimpa.")
    meta = json.loads(meta_path.read_text("utf-8"))
    sample = pd.read_csv(sample_path)
    if len(sample) != int(meta["jumlah_baris_latih"]):
        raise DatasetError(
            f"training_sample.csv ({len(sample)} baris) bukan seluruh data latih ({meta['jumlah_baris_latih']} baris); backfill dibatalkan."
        )
    data = ensure_canonical(sample, sumber=sumber, prefix=f"awal-{version}")
    data.to_csv(folder / DATA_FILE, index=False)
    meta["data_latih"] = {**data_info("awal", None, 0, len(data), [sumber]), "dibuat_tanpa_refit": True}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    logger.info("data_latih.csv dibuat untuk %s (%d baris, sumber %s)", version, len(data), sumber)
    return folder / DATA_FILE
