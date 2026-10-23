"""Pemuatan CSV statistik aliran (CIC asli / BCCC), penamaan kolom internal, dan filter VoIP."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SEP_CANDIDATES: tuple[str, ...] = (",", ";", "\t")
TIME_FACTORS_TO_US: dict[str, float] = {"s": 1_000_000.0, "ms": 1_000.0, "us": 1.0}
# Nomor protokol IANA untuk sumber yang menulis protokol sebagai teks
PROTOCOL_NUMBERS: dict[str, int] = {"ICMP": 1, "TCP": 6, "UDP": 17}
TIME_COLUMNS: dict[str, str] = {"duration": "duration_us", "iat_mean": "iat_mean_us", "iat_std": "iat_std_us"}
OUTPUT_COLUMNS: tuple[str, ...] = (
    "duration_us",
    "iat_mean_us",
    "iat_std_us",
    "fwd_packets_count",
    "bwd_packets_count",
    "protocol",
    "src_port",
    "dst_port",
)
LABEL_COLUMNS: tuple[str, ...] = ("traffic_type", "app_category")
VOIP_MODES: tuple[str, ...] = ("label", "port")
CSV_ENCODING = "utf-8-sig"


def detect_sep(header_line: str) -> str:
    """Tebak pemisah kolom dari baris header (kandidat yang paling sering muncul)."""
    return max(SEP_CANDIDATES, key=header_line.count)


def detect_format(columns: list[str], column_map: dict[str, Any]) -> str:
    """Kembalikan nama format pertama di `column_map` yang semua kolom sumbernya tersedia."""
    available = {c.strip() for c in columns}
    problems = []
    for name, spec in column_map.items():
        missing = [src for src in spec["columns"].values() if src not in available]
        if not missing:
            return name
        problems.append(f"{name}: kurang {missing}")
    raise ValueError("Format CSV tidak dikenali. " + "; ".join(problems))


def normalize_protocol(series: pd.Series) -> pd.Series:
    """Ubah protokol menjadi nomor IANA; menerima angka (17) maupun teks ("UDP")."""
    numeric = pd.to_numeric(series, errors="coerce")
    names = series.astype("string").str.strip().str.upper().map(PROTOCOL_NUMBERS)
    return numeric.fillna(names.astype("float64"))


def load_csv(path: str | Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Muat satu CSV menjadi kolom internal; waktu dikonversi ke µs, nilai tak terbaca → NaN.

    Format yang punya `label_columns` (darknet) juga menghasilkan kolom teks traffic_type/app_category.
    """
    path = Path(path)
    sep = data_cfg["csv_sep"]
    if sep == "auto":
        with path.open(encoding=CSV_ENCODING, errors="replace") as fh:
            sep = detect_sep(fh.readline())
    header = pd.read_csv(path, sep=sep, nrows=0, encoding=CSV_ENCODING, encoding_errors="replace")
    fmt = detect_format(list(header.columns), data_cfg["column_map"])
    spec = data_cfg["column_map"][fmt]
    label_map = spec.get("label_columns", {})
    source_to_internal = {src: internal for internal, src in {**spec["columns"], **label_map}.items()}

    time_unit = data_cfg["time_unit"]
    if time_unit == "auto":
        time_unit = spec["time_unit"]
    if time_unit not in TIME_FACTORS_TO_US:
        raise ValueError(f"time_unit tidak dikenal: {time_unit!r} (pilihan: {sorted(TIME_FACTORS_TO_US)})")

    raw = pd.read_csv(
        path,
        sep=sep,
        usecols=lambda c: c.strip() in source_to_internal,
        dtype=str,
        encoding=CSV_ENCODING,
        encoding_errors="replace",
    )
    raw = raw.rename(columns=lambda c: source_to_internal[c.strip()])

    out = pd.DataFrame(index=raw.index)
    unparseable: dict[str, int] = {}
    for col in raw.columns:
        if col in label_map:
            continue
        values = normalize_protocol(raw[col]) if col == "protocol" else pd.to_numeric(raw[col], errors="coerce")
        filled = raw[col].notna() & (raw[col].str.strip() != "")
        unparseable[col] = int((filled & values.isna()).sum())
        if col in TIME_COLUMNS:
            out[TIME_COLUMNS[col]] = values * TIME_FACTORS_TO_US[time_unit]
        else:
            out[col] = values
    out = out[list(OUTPUT_COLUMNS)]
    for col in label_map:
        out[col] = raw[col].str.strip()

    logger.info("Memuat %s: format=%s, sep=%r, time_unit=%s, %d baris", path.name, fmt, sep, time_unit, len(out))
    bad = {k: v for k, v in unparseable.items() if v}
    if bad:
        logger.warning("Nilai tidak terbaca (dijadikan NaN) di %s: %s", path.name, bad)
    return out


def load_flows(source: str | Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Muat satu CSV atau semua `*.csv` dalam sebuah folder, lalu gabungkan."""
    source = Path(source)
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f"Tidak ada berkas CSV di {source}")
    return pd.concat([load_csv(f, data_cfg) for f in files], ignore_index=True)


def in_port_ranges(ports: pd.Series, ranges: list[list[int]]) -> pd.Series:
    """True jika port berada di salah satu rentang [lo, hi] (inklusif)."""
    mask = pd.Series(False, index=ports.index)
    for lo, hi in ranges:
        mask |= ports.between(lo, hi)
    return mask


def voip_port_masks(df: pd.DataFrame, voip_cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Kembalikan (mask_sip, mask_rtp): salah satu port cocok port SIP / rentang RTP."""
    sip = df["src_port"].isin(voip_cfg["sip_ports"]) | df["dst_port"].isin(voip_cfg["sip_ports"])
    ranges = voip_cfg["rtp_port_ranges"]
    rtp = in_port_ranges(df["src_port"], ranges) | in_port_ranges(df["dst_port"], ranges)
    return sip, rtp


def label_mask(df: pd.DataFrame, voip_cfg: dict[str, Any]) -> pd.Series:
    """True jika kategori aplikasi == label_value (tanpa peduli huruf besar/kecil dan spasi)."""
    if "app_category" not in df.columns:
        raise ValueError("Mode filter 'label' butuh kolom kategori aplikasi; format CSV ini tidak punya label_columns")
    target = str(voip_cfg["label_value"]).strip().casefold()
    return df["app_category"].astype("string").str.strip().str.casefold() == target


def filter_voip(df: pd.DataFrame, voip_cfg: dict[str, Any]) -> pd.DataFrame:
    """Saring flow VoIP. mode=label: kategori == label_value; mode=port: UDP DAN (port SIP ATAU RTP) — PRD §7.2."""
    mode = voip_cfg["mode"]
    if mode not in VOIP_MODES:
        raise ValueError(f"voip_filter.mode tidak dikenal: {mode!r} (pilihan: {list(VOIP_MODES)})")
    if mode == "label":
        mask = label_mask(df, voip_cfg).fillna(False).astype(bool)
    else:
        mask = df["protocol"] == voip_cfg["protocol"]
        if voip_cfg["fallback_udp_all"]:
            logger.warning("fallback_udp_all aktif: semua flow UDP dipakai tanpa filter port")
        else:
            sip, rtp = voip_port_masks(df, voip_cfg)
            mask &= sip | rtp
    result = df[mask].reset_index(drop=True)
    logger.info("Filter VoIP (%s): %d flow sebelum, %d flow sesudah", mode, len(df), len(result))
    if len(result) < voip_cfg["min_flows_warning"]:
        msg = (
            f"Jumlah flow VoIP setelah filter ({len(result)}) di bawah "
            f"min_flows_warning ({voip_cfg['min_flows_warning']})"
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    return result
