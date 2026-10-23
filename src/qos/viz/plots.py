"""Grafik Plotly untuk dashboard (PRD FR-VIZ); fungsi murni tanpa state Streamlit."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qos.monitor.engine import INSUFFICIENT, card_category
from qos.profiling.labeling import loss_good_pct

LABEL_COLORS: dict[str, str] = {
    "QoS Baik": "#2563EB",
    "QoS Cukup": "#EAB308",
    "QoS Buruk": "#F97316",
    "Anomali": "#DC2626",
    INSUFFICIENT: "#9CA3AF",
}
NOISE_ID = -1
BACKGROUND_OPACITY = 0.15
LOG_PAD_DECADES = 0.3  # ruang di atas/bawah nilai terkecil dan terbesar pada sumbu log
# metrik box plot → (kunci statistik pelabelan, judul, satuan, nama penanda)
BOX_METRICS: dict[str, tuple[str, str, str, str]] = {
    "delay_ms": ("mean_delay_ms", "Delay", "ms", "Mean delay"),
    "jitter_ms": ("p999_jitter_ms", "Jitter", "ms", "P99,9 jitter"),
    "fai_pct": ("plr_pct", "FAI (loss)", "%", "PLR"),
}


def label_color(label: str) -> str:
    """Warna label sesuai kategorinya."""
    return LABEL_COLORS[card_category(label)]


ROW_TINT_ALPHA = "33"  # hex alpha (~20%) agar teks tetap terbaca di tema terang/gelap


def style_by_label(df: pd.DataFrame, label_column: str):
    """Styler tabel: warna latar tiap baris sesuai kategori label pada `label_column`."""
    return df.style.apply(
        lambda row: [f"background-color: {label_color(str(row[label_column]))}{ROW_TINT_ALPHA}"] * len(row), axis=1
    )


def cluster_name(cluster_id: int) -> str:
    """Nama klaster untuk sumbu: `Noise` untuk -1, selain itu `K<id>`."""
    return "Noise" if int(cluster_id) == NOISE_ID else f"K{int(cluster_id)}"


def pca_scatter(background: pd.DataFrame, labels: dict[int, str], live: pd.DataFrame | None = None) -> go.Figure:
    """Scatter PCA 2D: latar titik latih pudar + titik jendela live berwarna menurut label.

    `background` berkolom pc1, pc2, cluster_id; `live` berkolom pc1, pc2, label, flow_id, dll.
    Baris live tanpa PC (Data Belum Cukup) tidak digambar.
    """
    fig = go.Figure()
    bg = background.assign(
        category=[card_category(labels.get(int(c), "Anomali")) for c in background["cluster_id"]]
    )
    for cat, sub in bg.groupby("category"):
        fig.add_trace(
            go.Scatter(
                x=sub["pc1"], y=sub["pc2"], mode="markers", name=f"Latih · {cat}",
                marker={"color": LABEL_COLORS[cat], "size": 6, "opacity": BACKGROUND_OPACITY},
                hoverinfo="skip", legendgroup="latih",
            )
        )
    if live is not None and len(live):
        pts = live.dropna(subset=["pc1", "pc2"])
        for cat, sub in pts.assign(category=[card_category(v) for v in pts["label"]]).groupby("category"):
            fig.add_trace(
                go.Scatter(
                    x=sub["pc1"], y=sub["pc2"], mode="markers", name=f"Live · {cat}",
                    marker={"color": LABEL_COLORS[cat], "size": 10, "line": {"width": 1, "color": "white"}},
                    customdata=sub[["flow_id", "label", "delay_ms", "jitter_ms", "fai_pct", "membership_prob"]],
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>delay %{customdata[2]:.2f} ms · "
                        "jitter %{customdata[3]:.2f} ms · FAI %{customdata[4]:.2f}%<br>"
                        "prob %{customdata[5]:.2f}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2", margin={"t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18},  # di bawah judul sumbu agar tidak menimpa "PC1"
    )
    return fig


def labeling_bars(profile: list[dict[str, Any]], thresholds: dict[str, float]) -> go.Figure:
    """Bar chart statistik pelabelan per klaster (mean delay, P99,9 jitter, PLR) dengan garis ambang ITU."""
    names = [cluster_name(c["id_klaster"]) for c in profile]
    colors = [label_color(c["label"]) for c in profile]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Mean delay (ms)", "P99,9 jitter (ms)", "PLR (%)"))
    for col, (_, (key, *_)) in enumerate(BOX_METRICS.items(), start=1):
        fig.add_trace(
            go.Bar(
                x=names, y=[c["statistik_pelabelan"][key] for c in profile], marker_color=colors,
                text=[c["label"] for c in profile], textposition="none",  # label hanya muncul saat hover
                hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>", showlegend=False,
            ),
            row=1, col=col,
        )
    for value, name, col in (
        (thresholds["delay_good_ms"], "ambang delay baik", 1),
        (thresholds["delay_max_ms"], "ambang delay maks", 1),
        (thresholds["jitter_max_ms"], "ambang jitter", 2),
        (loss_good_pct(thresholds), "ambang loss baik", 3),
        (thresholds["loss_max_pct"], "ambang loss maks", 3),
    ):
        if name == "ambang loss baik" and value == thresholds["loss_max_pct"]:
            continue  # profil lama (2 level): satu garis saja
        fig.add_hline(y=value, line_dash="dash", line_color="#6B7280", row=1, col=col,
                      annotation_text=f"{name} {value:g}", annotation_font_size=10)
    fig.update_layout(margin={"t": 40, "b": 10})
    return fig


def boxplot_fig(profile: list[dict[str, Any]], metric: str, thresholds: dict[str, float], log: bool = False) -> go.Figure:
    """Box plot satu metrik per klaster dari statistik tersimpan (go.Box q1/median/q3/fence), plus penanda."""
    key, title, unit, marker_name = BOX_METRICS[metric]
    names = [cluster_name(c["id_klaster"]) for c in profile]
    boxes = [c["boxplot"][metric] for c in profile]
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=names, q1=[b["q1"] for b in boxes], median=[b["median"] for b in boxes], q3=[b["q3"] for b in boxes],
            lowerfence=[b["whisker_low"] for b in boxes], upperfence=[b["whisker_high"] for b in boxes],
            marker_color="#6B7280", name="Box", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=names, y=[c["statistik_pelabelan"][key] for c in profile], mode="markers", name=marker_name,
            marker={"symbol": "diamond", "size": 10, "color": [label_color(c["label"]) for c in profile],
                    "line": {"width": 1, "color": "black"}},
            text=[
                f"{cluster_name(c['id_klaster'])}<br>Q1 {b['q1']:.2f} · median {b['median']:.2f} · Q3 {b['q3']:.2f}"
                f"<br>whisker {b['whisker_low']:.2f}–{b['whisker_high']:.2f}<br>pencilan {len(b['outliers'])}"
                f"<br>{marker_name} {c['statistik_pelabelan'][key]:.2f}"
                for c, b in zip(profile, boxes)
            ],
            hoverinfo="text",
        )
    )
    values = {
        "delay_ms": [thresholds["delay_good_ms"], thresholds["delay_max_ms"]],
        "jitter_ms": [thresholds["jitter_max_ms"]],
        "fai_pct": sorted({loss_good_pct(thresholds), thresholds["loss_max_pct"]}),
    }[metric]
    for value in values:
        fig.add_hline(y=value, line_dash="dash", line_color="#6B7280", annotation_text=f"ambang {value:g} {unit}")
    fig.update_layout(title=f"{title} ({unit})", yaxis_title=unit, margin={"t": 40, "b": 10})
    if log:
        # sumbu log tidak bisa memuat 0 (mis. FAI 0); tentukan rentang dari nilai positif agar autorange tidak meledak
        pos = [v for b in boxes for k in ("q1", "median", "q3", "whisker_low", "whisker_high") if (v := b[k]) > 0]
        pos += [c["statistik_pelabelan"][key] for c in profile if c["statistik_pelabelan"][key] > 0]
        pos += [v for v in values if v > 0]
        if pos:
            fig.update_yaxes(type="log", range=[math.log10(min(pos)) - LOG_PAD_DECADES, math.log10(max(pos)) + LOG_PAD_DECADES])
        else:
            fig.update_yaxes(type="log")
    return fig
