"""Ringkasan per panggilan (flow): akumulasi baris jendela → satu baris saat flow selesai (PRD MON-9, §10.3)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from qos.capture.flow_table import FlowEnd

INSUFFICIENT = "Data Belum Cukup"
CATEGORY_ORDER: tuple[str, ...] = ("QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali")  # urut keparahan naik
PERCENT_COLUMNS: tuple[str, ...] = (
    "persen_baik", "persen_cukup", "persen_buruk", "persen_anomali", "persen_belum_cukup",
)
CALL_COLUMNS: tuple[str, ...] = (
    "flow_id", "waktu_mulai", "waktu_selesai", "durasi_s", "total_fwd", "total_bwd", "delay_ms", "jitter_ms",
    "fai_pct", "cluster_id", "label_akhir", "kategori_tindakan", "metrik_penyebab", "jumlah_jendela",
    *PERCENT_COLUMNS, "label_terburuk", "catatan",
)
WINDOW_MODE_NOTE = (
    "Mode window: nilai dan label_akhir berasal dari jendela terakhir, bukan kumulatif; "
    "ringkasan panggilan yang sah hanya pada feature_mode cumulative"
)
TIME_PRECISION = "seconds"


def category_of(label: str) -> str:
    """Kategori label: salah satu CATEGORY_ORDER, INSUFFICIENT, atau Anomali untuk label tak dikenal."""
    if label == INSUFFICIENT:
        return INSUFFICIENT
    for key in CATEGORY_ORDER[:3]:
        if label.startswith(key):
            return key
    return "Anomali"


@dataclass
class _CallState:
    """Akumulator satu flow selama berjalan."""

    n_windows: int = 0
    categories: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)  # label lengkap (tanpa Data Belum Cukup)
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
            label = str(row["label"])
            st.n_windows += 1
            st.categories[category_of(label)] += 1
            if label != INSUFFICIENT:
                st.labels[label] += 1
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
        pct = {
            col: st.categories.get(cat, 0) / st.n_windows * 100
            for col, cat in zip(PERCENT_COLUMNS, (*CATEGORY_ORDER, INSUFFICIENT))
        }
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
            "cluster_id": last["cluster_id"],
            "label_akhir": last["label"],
            "kategori_tindakan": last["kategori_tindakan"],
            "metrik_penyebab": last["metrik_penyebab"],
            "jumlah_jendela": st.n_windows,
            **pct,
            "label_terburuk": worst_label(st.labels),
            "catatan": WINDOW_MODE_NOTE if self.feature_mode == "window" else "",
        }


def worst_label(labels: Counter) -> str:
    """Label paling parah (kategori tertinggi; seri → paling sering); INSUFFICIENT jika tidak ada jendela terprediksi."""
    if not labels:
        return INSUFFICIENT
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return max(labels, key=lambda lab: (rank[category_of(lab)], labels[lab]))


def count_final_labels(calls: pd.DataFrame) -> dict[str, int]:
    """Jumlah panggilan per kategori label_akhir (termasuk Data Belum Cukup)."""
    counts = Counter(category_of(str(v)) for v in calls["label_akhir"]) if len(calls) else Counter()
    return {k: counts.get(k, 0) for k in (*CATEGORY_ORDER, INSUFFICIENT)}


def call_title(flow_id: str) -> str:
    """Judul terbaca dari flow_id `ipA:portA-ipB:portB-proto` → `ipA ↔ ipB` (flow_id apa adanya jika tak sesuai pola)."""
    parts = flow_id.split("-")
    if len(parts) != 3 or any(":" not in p for p in parts[:2]):
        return flow_id
    return " ↔ ".join(p.rsplit(":", 1)[0] for p in parts[:2])
