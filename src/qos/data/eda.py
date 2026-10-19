"""Ringkasan EDA: rincian flow kandidat VoIP dan statistik deskriptif fitur."""
from __future__ import annotations

from typing import Any

import pandas as pd

from qos.data.loader import in_port_ranges, voip_port_masks


def voip_breakdown(
    df: pd.DataFrame, voip_cfg: dict[str, Any], suspect_ports: list[int], top_n: int = 10
) -> dict[str, Any]:
    """Rinci flow UDP yang lolos filter port: SIP vs hanya-RTP, dan port lawan yang mencurigakan.

    "Port lawan" pada flow hanya-RTP = port yang berada di luar rentang RTP.
    """
    udp = df["protocol"] == voip_cfg["protocol"]
    sip, rtp = voip_port_masks(df, voip_cfg)
    sip_flows = udp & sip
    rtp_only = udp & rtp & ~sip
    suspect = df["src_port"].isin(suspect_ports) | df["dst_port"].isin(suspect_ports)

    ranges = voip_cfg["rtp_port_ranges"]
    sub = df[rtp_only]
    src_in = in_port_ranges(sub["src_port"], ranges)
    dst_in = in_port_ranges(sub["dst_port"], ranges)
    counter = sub["dst_port"].where(src_in & ~dst_in, sub["src_port"].where(dst_in & ~src_in))
    top = counter.dropna().astype(int).value_counts().head(top_n)

    return {
        "total": int(len(df)),
        "udp": int(udp.sum()),
        "voip": int((sip_flows | rtp_only).sum()),
        "sip": int(sip_flows.sum()),
        "sip_suspect": int((sip_flows & suspect).sum()),
        "rtp_only": int(rtp_only.sum()),
        "rtp_only_suspect": int((rtp_only & suspect).sum()),
        "rtp_only_both_in_range": int((src_in & dst_in).sum()),
        "suspect_by_port": {
            int(p): int((rtp_only & ((df["src_port"] == p) | (df["dst_port"] == p))).sum())
            for p in suspect_ports
        },
        "top_counter_ports": [(int(p), int(n)) for p, n in top.items()],
    }


def describe_features(df: pd.DataFrame, columns: list[str], percentiles: list[float]) -> pd.DataFrame:
    """Statistik deskriptif per fitur ditambah jumlah NaN, nol, dan negatif."""
    desc = df[columns].describe(percentiles=percentiles).T
    desc.insert(1, "nan", df[columns].isna().sum())
    desc.insert(2, "nol", (df[columns] == 0).sum())
    desc.insert(3, "negatif", (df[columns] < 0).sum())
    return desc


GROUP_COLUMN = "traffic_type"
NUMERIC_COLUMNS: tuple[str, ...] = (
    "duration_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count", "protocol", "src_port", "dst_port",
)


def voip_by_group(df: pd.DataFrame, voip: pd.DataFrame, group_col: str = GROUP_COLUMN) -> pd.DataFrame:
    """Jumlah flow total, flow VoIP, dan proporsi VoIP per kelompok (mis. Non-Tor/NonVPN/Tor/VPN)."""
    total = df[group_col].value_counts()
    kept = voip[group_col].value_counts()
    out = pd.DataFrame({"total": total, "voip": kept}).fillna(0).astype(int)
    out["voip_pct"] = (out["voip"] / out["total"] * 100).round(2)
    return out.sort_index()


def corrupt_counts(df: pd.DataFrame, columns: tuple[str, ...] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Nilai rusak per kolom: NaN (kosong/tak terbaca), ±inf, negatif, dan nol."""
    cols = [c for c in columns if c in df.columns]
    sub = df[cols]
    return pd.DataFrame(
        {
            "nan": sub.isna().sum(),
            "inf": sub.apply(lambda s: int((s.abs() == float("inf")).sum())),
            "negatif": (sub < 0).sum(),
            "nol": (sub == 0).sum(),
        }
    )


def describe_by_group(
    df: pd.DataFrame, columns: list[str], percentiles: list[float], group_col: str = GROUP_COLUMN
) -> dict[str, pd.DataFrame]:
    """`describe_features` per kelompok; nilai ±inf dijadikan NaN dulu (seperti tahap cleaning)."""
    clean = df.replace([float("inf"), float("-inf")], float("nan"))
    return {
        str(name): describe_features(sub, columns, percentiles)
        for name, sub in clean.groupby(group_col, sort=True)
    }


def packet_medians(df: pd.DataFrame, group_col: str = GROUP_COLUMN) -> pd.DataFrame:
    """Median paket per arah (fwd/bwd) per kelompok, plus jumlah flow fwd == 0 (dibuang di FAI)."""
    rows = {}
    for name, sub in [*df.groupby(group_col, sort=True), ("(semua)", df)]:
        rows[str(name)] = {
            "n": len(sub),
            "median_fwd": sub["fwd_packets_count"].median(),
            "median_bwd": sub["bwd_packets_count"].median(),
            "fwd_nol": int((sub["fwd_packets_count"] == 0).sum()),
            "bwd_nol": int((sub["bwd_packets_count"] == 0).sum()),
            "bwd_lebih_besar": int((sub["bwd_packets_count"] > sub["fwd_packets_count"]).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def feature_summary(df: pd.DataFrame, cap_us: float, tol_us: float, clip_negative: bool) -> dict[str, float]:
    """Median & p95 duration/iat_std/FAI, % iat_std = 0, dan % durasi menempel di batas `cap_us`."""
    from qos.features.preprocess import compute_fai

    fai = compute_fai(df["fwd_packets_count"], df["bwd_packets_count"], clip_negative)
    n = len(df)
    nan = float("nan")
    out: dict[str, float] = {"n": n}
    for name, s in (("duration_us", df["duration_us"]), ("iat_std_us", df["iat_std_us"]), ("fai_pct", fai)):
        out[f"{name}_median"] = s.median() if n else nan
        out[f"{name}_p95"] = s.quantile(0.95) if n else nan
    out["iat_nol_pct"] = (df["iat_std_us"] == 0).mean() * 100 if n else nan
    out["durasi_batas_pct"] = (df["duration_us"] >= cap_us - tol_us).mean() * 100 if n else nan
    return out


def sensitivity(
    voip: pd.DataFrame,
    ns: list[int],
    exclude: list[str],
    cap_us: float,
    tol_us: float,
    clip_negative: bool,
    group_col: str = GROUP_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Sensitivitas filter sesi (fwd ≥ N dan bwd ≥ N) untuk tiap N.

    Kembalian: `jumlah` (per traffic type + total + total tanpa `exclude`), `protokol`
    (per protokol), `statistik_semua` dan `statistik_tanpa` (ringkasan fitur, tanpa `exclude`).
    """
    from qos.features.preprocess import exclude_types, session_mask

    count_rows, proto_rows, stat_all, stat_wo = {}, {}, {}, {}
    for n in ns:
        sub = voip[session_mask(voip, n, n)]
        wo = exclude_types(sub, exclude, group_col)
        count_rows[n] = {**sub[group_col].value_counts().to_dict(), "total": len(sub), "total_tanpa": len(wo)}
        proto_rows[n] = sub["protocol"].value_counts().to_dict()
        stat_all[n] = feature_summary(sub, cap_us, tol_us, clip_negative)
        stat_wo[n] = feature_summary(wo, cap_us, tol_us, clip_negative)
    jumlah = pd.DataFrame.from_dict(count_rows, orient="index").fillna(0).astype(int)
    jumlah = jumlah[[c for c in jumlah.columns if c not in ("total", "total_tanpa")] + ["total", "total_tanpa"]]
    proto = pd.DataFrame.from_dict(proto_rows, orient="index").fillna(0).astype(int)
    proto.columns = [f"proto_{int(c)}" for c in proto.columns]
    return {
        "jumlah": jumlah.rename_axis("N"),
        "protokol": proto.sort_index(axis=1).rename_axis("N"),
        "statistik_semua": pd.DataFrame.from_dict(stat_all, orient="index").rename_axis("N"),
        "statistik_tanpa": pd.DataFrame.from_dict(stat_wo, orient="index").rename_axis("N"),
    }
