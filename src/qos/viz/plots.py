"""Grafik Plotly untuk dashboard (PRD FR-VIZ); fungsi murni tanpa state Streamlit.

Warna klaster = palet kategorikal tanpa makna baik/buruk; abu-abu hanya untuk "Di luar klaster", noise, dan status tanpa penugasan.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qos.monitor.calls import INSUFFICIENT, OUTSIDE_KEY, SIGNALING

CLUSTER_PALETTE: tuple[str, ...] = (
    "#2E86AB", "#8E5BB5", "#2A9D8F", "#E9A23B", "#C2557A", "#6C8B3C", "#4C6EF5", "#B5651D", "#1F9EAD", "#9B6B9E",
)
OUTSIDE_COLOR = "#6B7280"     # Di luar klaster / noise
NEUTRAL_COLOR = "#D1D5DB"     # Data Belum Cukup / Sinyal SIP
NOISE_ID = -1
BACKGROUND_OPACITY = 0.15
LOG_PAD_DECADES = 0.3  # ruang di atas/bawah nilai terkecil dan terbesar pada sumbu log
ROW_TINT_ALPHA = "33"  # hex alpha (~20%) agar teks tetap terbaca di tema terang/gelap
# metrik box plot → (judul, satuan)
BOX_METRICS: dict[str, tuple[str, str]] = {
    "delay_ms": ("Delay", "ms"),
    "jitter_ms": ("Jitter", "ms"),
    "fai_pct": ("FAI (proksi loss)", "%"),
}


def cluster_color(cluster_id: int) -> str:
    """Warna klaster bernomor 1..k dari palet kategorikal (berulang bila k > panjang palet); noise → abu-abu."""
    if int(cluster_id) == NOISE_ID:
        return OUTSIDE_COLOR
    return CLUSTER_PALETTE[(int(cluster_id) - 1) % len(CLUSTER_PALETTE)]


def cluster_name(cluster_id: int) -> str:
    """Nama klaster: `Noise` untuk -1, selain itu `Klaster N`."""
    return "Noise" if int(cluster_id) == NOISE_ID else f"Klaster {int(cluster_id)}"


def state_color(text: str) -> str:
    """Warna dari teks status: "Klaster N", "Di luar klaster (terdekat: ...)", Data Belum Cukup, Sinyal SIP."""
    text = str(text)
    if text.startswith("Klaster "):
        return cluster_color(int(text.split()[1]))
    if text.startswith(OUTSIDE_KEY) or text == "Noise":
        return OUTSIDE_COLOR
    return NEUTRAL_COLOR


def style_by_state(df: pd.DataFrame, state_column: str):
    """Styler tabel: warna latar tiap baris sesuai klaster/status pada `state_column`."""
    return df.style.apply(lambda row: [f"background-color: {state_color(str(row[state_column]))}{ROW_TINT_ALPHA}"] * len(row), axis=1)


def pca_scatter(
    background: pd.DataFrame, live: pd.DataFrame | None = None, centroids: dict[int, tuple[float, float]] | None = None
) -> go.Figure:
    """Scatter PCA 2D: latar titik latih pudar per klaster, titik jendela live, dan centroid sebagai penanda besar.

    `background` berkolom pc1, pc2, cluster_id (-1 = noise); `live` berkolom pc1, pc2, status, klaster_terdekat, flow_id, dll.
    Titik live diwarnai menurut klaster terdekat; yang di luar klaster abu-abu (simbol X). Baris tanpa PC tidak digambar.
    """
    fig = go.Figure()
    for cid in sorted(background["cluster_id"].unique()):
        sub = background[background["cluster_id"] == cid]
        fig.add_trace(
            go.Scatter(
                x=sub["pc1"], y=sub["pc2"], mode="markers", name=f"Latih · {cluster_name(cid)}",
                marker={"color": cluster_color(cid), "size": 6, "opacity": BACKGROUND_OPACITY},
                hoverinfo="skip", legendgroup="latih",
            )
        )
    if live is not None and len(live):
        pts = live.dropna(subset=["pc1", "pc2"])
        outside = pts["status"] == OUTSIDE_KEY
        for cid, sub in pts[~outside].groupby("klaster_terdekat"):
            fig.add_trace(_live_trace(sub, f"Live · {cluster_name(cid)}", cluster_color(cid), "circle"))
        if outside.any():
            fig.add_trace(_live_trace(pts[outside], f"Live · {OUTSIDE_KEY}", OUTSIDE_COLOR, "x"))
    if centroids:
        ids = sorted(centroids)
        fig.add_trace(
            go.Scatter(
                x=[centroids[i][0] for i in ids], y=[centroids[i][1] for i in ids], mode="markers+text",
                name="Centroid", text=[f"K{i}" for i in ids], textposition="top center",
                marker={"symbol": "star", "size": 20, "color": [cluster_color(i) for i in ids], "line": {"width": 2, "color": "black"}},
                hovertext=[cluster_name(i) for i in ids], hoverinfo="text",
            )
        )
    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2", margin={"t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18},  # di bawah judul sumbu agar tidak menimpa "PC1"
    )
    return fig


def _live_trace(sub: pd.DataFrame, name: str, color: str, symbol: str) -> go.Scatter:
    return go.Scatter(
        x=sub["pc1"], y=sub["pc2"], mode="markers", name=name,
        marker={"color": color, "size": 10, "symbol": symbol, "line": {"width": 1, "color": "white"}},
        customdata=sub[["flow_id", "klaster_terdekat", "jarak_terdekat", "delay_ms", "jitter_ms", "fai_pct"]],
        hovertemplate=(
            "%{customdata[0]}<br>terdekat: Klaster %{customdata[1]} (jarak %{customdata[2]:.2f})<br>"
            "delay %{customdata[3]:.2f} ms · jitter %{customdata[4]:.2f} ms · FAI %{customdata[5]:.2f}%<extra></extra>"
        ),
    )


def median_bars(profile: list[dict[str, Any]], thresholds: dict[str, float]) -> go.Figure:
    """Bar chart median delay, jitter, dan FAI per klaster dengan garis ambang ITU sebagai referensi (bukan label)."""
    names = [cluster_name(c["id_klaster"]) for c in profile]
    colors = [cluster_color(c["id_klaster"]) for c in profile]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Median delay (ms)", "Median jitter (ms)", "Median FAI (%)"))
    for col, metric in enumerate(BOX_METRICS, start=1):
        fig.add_trace(
            go.Bar(
                x=names, y=[c["statistik"][metric]["median"] for c in profile], marker_color=colors,
                text=[c["deskripsi"]["ringkas"] for c in profile], textposition="none",
                hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>", showlegend=False,
            ),
            row=1, col=col,
        )
    for value, name, col in (
        (thresholds["delay_good_ms"], "ambang delay", 1), (thresholds["delay_max_ms"], "ambang delay", 1),
        (thresholds["jitter_max_ms"], "ambang jitter", 2),
        (thresholds["loss_good_pct"], "ambang loss", 3), (thresholds["loss_max_pct"], "ambang loss", 3),
    ):
        fig.add_hline(y=value, line_dash="dash", line_color="#6B7280", row=1, col=col,
                      annotation_text=f"{name} {value:g}", annotation_font_size=10)
    fig.update_layout(margin={"t": 40, "b": 10})
    return fig


def reference_lines(metric: str, thresholds: dict[str, float]) -> list[float]:
    """Nilai garis ambang ITU untuk satu metrik (hanya referensi)."""
    return {
        "delay_ms": [thresholds["delay_good_ms"], thresholds["delay_max_ms"]],
        "jitter_ms": [thresholds["jitter_max_ms"]],
        "fai_pct": sorted({thresholds["loss_good_pct"], thresholds["loss_max_pct"]}),
    }[metric]


def boxplot_fig(profile: list[dict[str, Any]], metric: str, thresholds: dict[str, float], log: bool = False) -> go.Figure:
    """Box plot satu metrik per klaster dari statistik tersimpan (q1/median/q3/fence) dan penanda mean; garis = ambang ITU."""
    title, unit = BOX_METRICS[metric]
    fig = go.Figure()
    boxes = [c["boxplot"][metric] for c in profile]
    for c, b in zip(profile, boxes):
        name = cluster_name(c["id_klaster"])
        fig.add_trace(
            go.Box(
                x=[name], q1=[b["q1"]], median=[b["median"]], q3=[b["q3"]], lowerfence=[b["whisker_low"]], upperfence=[b["whisker_high"]],
                mean=[c["statistik"][metric]["mean"]], marker_color=cluster_color(c["id_klaster"]), name=name, showlegend=False,
                hovertext=f"{name}<br>Q1 {b['q1']:.2f} · median {b['median']:.2f} · Q3 {b['q3']:.2f}"
                          f"<br>whisker {b['whisker_low']:.2f}–{b['whisker_high']:.2f}<br>pencilan {len(b['outliers'])}",
                hoverinfo="text",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[cluster_name(c["id_klaster"]) for c in profile], y=[c["statistik"][metric]["mean"] for c in profile],
            mode="markers", name="Mean", marker={"symbol": "diamond", "size": 9, "color": "white", "line": {"width": 1.5, "color": "black"}},
            hovertemplate="%{x}<br>mean %{y:.2f}<extra></extra>",
        )
    )
    values = reference_lines(metric, thresholds)
    for value in values:
        fig.add_hline(y=value, line_dash="dash", line_color="#6B7280", annotation_text=f"ambang {value:g} {unit}")
    fig.update_layout(title=f"{title} ({unit})", yaxis_title=unit, margin={"t": 40, "b": 10})
    if log:
        # sumbu log tidak bisa memuat 0 (mis. FAI 0); tentukan rentang dari nilai positif agar autorange tidak meledak
        pos = [v for b in boxes for k in ("q1", "median", "q3", "whisker_low", "whisker_high") if (v := b[k]) > 0]
        pos += [c["statistik"][metric]["mean"] for c in profile if c["statistik"][metric]["mean"] > 0]
        pos += [v for v in values if v > 0]
        if pos:
            fig.update_yaxes(type="log", range=[math.log10(min(pos)) - LOG_PAD_DECADES, math.log10(max(pos)) + LOG_PAD_DECADES])
        else:
            fig.update_yaxes(type="log")
    return fig


def distance_bars(dists: dict[int, float], limits: dict[int, float] | None = None, nearest: int | None = None, height: int = 150) -> go.Figure:
    """Bar kecil jarak ke semua centroid (satu bar per klaster); tanda | = batas "di luar klaster", klaster terdekat disorot."""
    ids = sorted(dists)
    fig = go.Figure(
        go.Bar(
            x=[dists[i] for i in ids], y=[cluster_name(i) for i in ids], orientation="h",
            marker={"color": [cluster_color(i) for i in ids], "line": {"width": [2 if i == nearest else 0 for i in ids], "color": "black"}},
            hovertemplate="%{y}: jarak %{x:.3f}<extra></extra>",
        )
    )
    if limits:
        fig.add_trace(
            go.Scatter(
                x=[limits[i] for i in ids if i in limits], y=[cluster_name(i) for i in ids if i in limits], mode="markers",
                marker={"symbol": "line-ns", "size": 14, "line": {"width": 2, "color": "black"}}, name="batas",
                hovertemplate="batas %{x:.3f}<extra></extra>",
            )
        )
    fig.update_layout(height=height, margin={"t": 5, "b": 5, "l": 5, "r": 5}, showlegend=False, yaxis={"autorange": "reversed"},
                      xaxis_title="jarak Euclidean (ter-skala)")
    return fig


def kmeans_curves_fig(sweep: pd.DataFrame, k_n: int, k_opt: int) -> go.Figure:
    """Elbow (inertia), Silhouette, DBI, dan CHI K-Means per k; garis putus = k pembanding (k_n) dan k optimal."""
    panels = (("inertia", "Elbow: inertia"), ("silhouette", "Silhouette (↑)"), ("davies_bouldin", "Davies-Bouldin (↓)"), ("chi", "Calinski-Harabasz (↑)"))
    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[1] for p in panels])
    for col, (key, _) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep[key], mode="lines+markers", marker_color="#2E86AB", showlegend=False), row=1, col=col)
        fig.add_vline(x=k_opt, line_dash="dash", line_color="#2A9D8F", row=1, col=col)
        if k_n != k_opt:
            fig.add_vline(x=k_n, line_dash="dot", line_color="#C2557A", row=1, col=col)
        fig.update_xaxes(title_text="k", row=1, col=col)
    fig.update_layout(margin={"t": 40, "b": 10}, height=300)
    return fig


def grid_heatmaps_fig(grid: pd.DataFrame, best: tuple[int, int] | None = None) -> go.Figure:
    """Heatmap grid HDBSCAN (min_cluster_size × min_samples): jumlah klaster, noise, Silhouette, DBCV, ARI; kotak = terpilih."""
    panels = (("n_clusters", "Jumlah klaster"), ("noise_ratio", "Noise ratio"), ("silhouette", "Silhouette"),
              ("dbcv", "DBCV"), ("ari_mean", "ARI bootstrap (mean)"), ("davies_bouldin", "Davies-Bouldin"))
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels], vertical_spacing=0.18)
    for i, (key, _) in enumerate(panels):
        r, c = divmod(i, 3)
        pivot = grid.pivot(index="min_cluster_size", columns="min_samples", values=key)
        fig.add_trace(
            go.Heatmap(z=pivot.to_numpy(dtype=float), x=[str(x) for x in pivot.columns], y=[str(y) for y in pivot.index],
                       colorscale="Viridis", showscale=False, text=np.round(pivot.to_numpy(dtype=float), 2), texttemplate="%{text}",
                       hovertemplate="mcs %{y}, ms %{x}: %{z:.3f}<extra></extra>"),
            row=r + 1, col=c + 1,
        )
        if best is not None and best[0] in pivot.index and best[1] in pivot.columns:
            x0, y0 = list(pivot.columns).index(best[1]), list(pivot.index).index(best[0])
            fig.add_shape(type="rect", x0=x0 - 0.5, x1=x0 + 0.5, y0=y0 - 0.5, y1=y0 + 0.5, line={"color": "#DC2626", "width": 3}, row=r + 1, col=c + 1)
        fig.update_xaxes(title_text="min_samples", row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="min_cluster_size", row=r + 1, col=c + 1)
    fig.update_layout(height=620, margin={"t": 40, "b": 10})
    return fig
