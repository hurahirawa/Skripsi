"""Ringkasan per panggilan (flow): akumulasi baris jendela → satu baris saat flow selesai (PRD MON-9, §10.3)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from qos.capture.flow_table import FlowEnd

INSUFFICIENT = "Data Belum Cukup"
SIGNALING = "Sinyal SIP"  # flow sinyal (monitoring.signaling_ports): bukan panggilan, tidak diprediksi
STATUS_CLUSTERED = "Terklaster"       # jarak ke salah satu centroid masih dalam batas klaster itu
STATUS_OUTSIDE = "Di luar klaster"    # jarak ke SETIAP centroid melebihi batas klaster tersebut
OUTSIDE_KEY = "Di luar klaster"
CALL_COLUMNS: tuple[str, ...] = (
    "flow_id", "waktu_mulai", "waktu_selesai", "durasi_s", "total_fwd", "total_bwd", "delay_ms", "jitter_ms",
    "fai_pct", "klaster_akhir", "jarak_akhir", "di_luar_klaster_akhir", "status_akhir", "jumlah_jendela",
    "persen_per_klaster", "persen_di_luar_klaster", "persen_belum_cukup", "klaster_terbanyak",
    "pernah_di_luar_klaster", "catatan",
)
WINDOW_MODE_NOTE = (
    "Mode window: nilai dan klaster_akhir berasal dari jendela terakhir, bukan kumulatif; "
    "ringkasan panggilan yang sah hanya pada feature_mode cumulative"
)
TIME_PRECISION = "seconds"


def cluster_key(cluster: int) -> str:
    """Kunci tampilan klaster: "Klaster N"."""
    return f"Klaster {int(cluster)}"


def format_percentages(counts: dict[int, int], total: int) -> str:
    """Persen per klaster sebagai "1:62.5;2:37.5" (titik desimal agar aman di CSV); kosong bila tidak ada."""
    if not total:
        return ""
    return ";".join(f"{c}:{counts[c] / total * 100:.1f}" for c in sorted(counts))


def parse_percentages(text: Any) -> dict[int, float]:
    """Kebalikan format_percentages; teks kosong/NaN → dict kosong."""
    if not isinstance(text, str) or not text.strip():
        return {}
    return {int(k): float(v) for k, v in (part.split(":") for part in text.split(";"))}


@dataclass
class _CallState:
    """Akumulator satu flow selama berjalan."""

    n_windows: int = 0
    clusters: Counter = field(default_factory=Counter)  # jendela Terklaster per klaster
    n_outside: int = 0
    n_insufficient: int = 0
    last: dict[str, Any] = field(default_factory=dict)


class CallTracker:
    """Kumpulkan baris jendela per flow; `finish` menghasilkan baris panggilan saat flow selesai."""

    def __init__(self, feature_mode: str) -> None:
        self.feature_mode = feature_mode
        self._state: dict[str, _CallState] = {}

    def update(self, windows: pd.DataFrame) -> None:
        """Tambahkan batch baris jendela (kolom WINDOW_COLUMNS)."""
        for row in windows.to_dict("records"):
            st = self._state.setdefault(row["flow_id"], _CallState())
            status = str(row["status"])
            st.n_windows += 1
            if status == STATUS_CLUSTERED:
                st.clusters[int(row["klaster_terdekat"])] += 1
            elif status == STATUS_OUTSIDE:
                st.n_outside += 1
            else:
                st.n_insufficient += 1
            st.last = row

    def finish(self, ends: list[FlowEnd]) -> pd.DataFrame:
        """Baris panggilan untuk flow yang selesai (flow tanpa baris jendela dilewati); state dibuang."""
        rows = []
        for end in ends:
            st = self._state.pop(end.flow_id, None)
            if st is None or not st.n_windows:
                continue
            rows.append(self._row(end, st))
        return pd.DataFrame(rows, columns=list(CALL_COLUMNS))

    def _row(self, end: FlowEnd, st: _CallState) -> dict[str, Any]:
        last = st.last
        top = max(st.clusters, key=lambda c: (st.clusters[c], -c)) if st.clusters else None
        return {
            "flow_id": end.flow_id,
            "waktu_mulai": datetime.fromtimestamp(end.first_ts).isoformat(timespec=TIME_PRECISION),
            "waktu_selesai": datetime.fromtimestamp(end.last_ts).isoformat(timespec=TIME_PRECISION),
            "durasi_s": end.last_ts - end.first_ts,
            "total_fwd": last["fwd"],
            "total_bwd": last["bwd"],
            "delay_ms": last["delay_ms"],
            "jitter_ms": last["jitter_ms"],
            "fai_pct": last["fai_pct"],
            "klaster_akhir": last["klaster_terdekat"],
            "jarak_akhir": last["jarak_terdekat"],
            "di_luar_klaster_akhir": last["di_luar_klaster"],
            "status_akhir": last["status"],
            "jumlah_jendela": st.n_windows,
            "persen_per_klaster": format_percentages(dict(st.clusters), st.n_windows),
            "persen_di_luar_klaster": st.n_outside / st.n_windows * 100,
            "persen_belum_cukup": st.n_insufficient / st.n_windows * 100,
            "klaster_terbanyak": top if top is not None else pd.NA,
            "pernah_di_luar_klaster": st.n_outside > 0,
            "catatan": WINDOW_MODE_NOTE if self.feature_mode == "window" else "",
        }


def summarize_calls(calls: pd.DataFrame) -> dict[str, Any]:
    """Ringkasan sesi dari tabel panggilan: jumlah per klaster, di luar klaster (jumlah dan %), klaster mayoritas.

    Panggilan berstatus akhir Terklaster dihitung per `klaster_akhir`; Di luar klaster dan Data Belum Cukup dihitung terpisah.
    % di luar klaster = dari panggilan yang dinilai (tanpa Data Belum Cukup).
    """
    status = calls["status_akhir"].astype(str) if len(calls) else pd.Series([], dtype=str)
    clustered = calls[status == STATUS_CLUSTERED]
    per = Counter(int(c) for c in clustered["klaster_akhir"].dropna()) if len(clustered) else Counter()
    n_out = int((status == STATUS_OUTSIDE).sum())
    n_ins = int((status == INSUFFICIENT).sum())
    scored = len(calls) - n_ins
    return {
        "jumlah_panggilan": len(calls),
        "panggilan_per_klaster": ";".join(f"{c}:{per[c]}" for c in sorted(per)),
        "panggilan_di_luar_klaster": n_out,
        "persen_di_luar_klaster": n_out / scored * 100 if scored else 0.0,
        "klaster_mayoritas": max(per, key=lambda c: (per[c], -c)) if per else pd.NA,
        "panggilan_belum_cukup": n_ins,
    }


def parse_counts(text: Any) -> dict[int, int]:
    """Teks "1:3;2:1" → {1: 3, 2: 1}; kosong/NaN → {}."""
    return {k: int(v) for k, v in parse_percentages(text).items()}


def call_title(flow_id: str) -> str:
    """Judul terbaca dari flow_id `ipA:portA-ipB:portB-proto` → `ipA ↔ ipB` (flow_id apa adanya jika tak sesuai pola)."""
    parts = flow_id.split("-")
    if len(parts) != 3 or any(":" not in p for p in parts[:2]):
        return flow_id
    return " ↔ ".join(p.rsplit(":", 1)[0] for p in parts[:2])


def call_state_text(status: str, cluster: Any) -> str:
    """Teks status akhir: "Klaster N", "Di luar klaster (terdekat: Klaster N)", atau status lain apa adanya."""
    if status == STATUS_CLUSTERED:
        return cluster_key(cluster)
    if status == STATUS_OUTSIDE:
        return f"{OUTSIDE_KEY} (terdekat: {cluster_key(cluster)})"
    return status


def call_toast_message(calls: pd.DataFrame) -> str:
    """Teks toast untuk panggilan yang baru selesai: satu panggilan → rinci; lebih → satu toast gabungan."""
    if len(calls) == 1:
        row = calls.iloc[0]
        text = call_state_text(str(row["status_akhir"]), row["klaster_akhir"])
        return f"Panggilan {call_title(row['flow_id'])} selesai: paling dekat ke {text}" if row["status_akhir"] == STATUS_CLUSTERED else (
            f"Panggilan {call_title(row['flow_id'])} selesai: {text}"
        )
    s = summarize_calls(calls)
    parts = [f"{n} di {cluster_key(c)}" for c, n in parse_counts(s["panggilan_per_klaster"]).items()]
    if s["panggilan_di_luar_klaster"]:
        parts.append(f"{s['panggilan_di_luar_klaster']} di luar klaster")
    if s["panggilan_belum_cukup"]:
        parts.append(f"{s['panggilan_belum_cukup']} {INSUFFICIENT.lower()}")
    return f"{len(calls)} panggilan selesai: {', '.join(parts)}"
