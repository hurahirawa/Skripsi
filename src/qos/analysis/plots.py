"""Grafik PNG analisis klasterisasi (matplotlib, backend Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qos.analysis.run import RunResult, pca_coords  # noqa: E402

NOISE_COLOR = "#9CA3AF"
HEATMAPS: tuple[tuple[str, str], ...] = (
    ("n_clusters", "Jumlah klaster"), ("noise_ratio", "Noise ratio"), ("silhouette", "Silhouette (tanpa noise)"),
    ("davies_bouldin", "Davies-Bouldin (tanpa noise)"), ("calinski_harabasz", "Calinski-Harabasz (tanpa noise)"), ("dbcv", "DBCV"),
)


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def hist_before_after(raw: pd.DataFrame, transformed: pd.DataFrame, path: Path, dpi: int, bins: int = 60) -> Path:
    """Histogram tiap fitur: nilai mentah (atas) dan setelah transformasi (bawah)."""
    cols = list(transformed.columns)
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 6), squeeze=False)
    for j, c in enumerate(cols):
        axes[0, j].hist(raw[c].dropna(), bins=bins, color="#2563EB")
        axes[0, j].set_title(f"{c} (mentah)")
        axes[0, j].set_yscale("log")
        axes[1, j].hist(transformed[c].dropna(), bins=bins, color="#F97316")
        axes[1, j].set_title(f"{c} (setelah transformasi)")
        axes[1, j].set_yscale("log")
    for ax in axes.ravel():
        ax.set_ylabel("jumlah flow (skala log)")
    return _save(fig, path, dpi)


def kmeans_curves(res: RunResult, path: Path, dpi: int) -> Path:
    """Elbow (inertia) dan tiga metrik K-Means terhadap k; garis putus = k optimal, titik siku ditandai."""
    sw = res.sweep
    panels = (("inertia", "Elbow: inertia"), ("silhouette", "Silhouette (maks lebih baik)"),
              ("davies_bouldin", "Davies-Bouldin (min lebih baik)"), ("calinski_harabasz", "Calinski-Harabasz (maks lebih baik)"))
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, (col, title) in zip(axes, panels):
        ax.plot(sw["k"], sw[col], marker="o", color="#2563EB")
        ax.axvline(res.k_opt, ls="--", color="#DC2626", label=f"k optimal = {res.k_opt}")
        if col == "inertia" and res.elbow_k is not None:
            ax.axvline(res.elbow_k, ls=":", color="#16A34A", label=f"siku (kneedle) = {res.elbow_k}")
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.legend(fontsize=8)
    return _save(fig, path, dpi)


def hdbscan_heatmaps(res: RunResult, path: Path, dpi: int) -> Path:
    """Heatmap grid min_cluster_size x min_samples untuk jumlah klaster, noise, dan lima metrik validitas."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title) in zip(axes.ravel(), HEATMAPS):
        pivot = res.grid.pivot(index="min_cluster_size", columns="min_samples", values=col)
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("min_samples")
        ax.set_ylabel("min_cluster_size")
        ax.set_title(title)
        for (i, j), v in np.ndenumerate(pivot.to_numpy(dtype=float)):
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}", ha="center", va="center", color="white", fontsize=7)
        best = (list(pivot.index).index(int(res.hdb_best["min_cluster_size"])), list(pivot.columns).index(int(res.hdb_best["min_samples"])))
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor="#DC2626", lw=2))
        fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, path, dpi)


def pca_scatter(res: RunResult, path: Path, dpi: int, max_points: int, random_state: int) -> Path:
    """PCA 2D berdampingan: HDBSCAN optimal, K-Means k optimal, dan K-Means k = klaster HDBSCAN."""
    prep = res.prep
    xy = pca_coords(prep.X, 2, random_state)
    idx = np.arange(len(xy))
    if len(idx) > max_points:
        idx = np.random.default_rng(random_state).choice(idx, size=max_points, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, key in zip(axes, ("hdbscan", "kmeans_opt", "kmeans_fair")):
        cfg = res.configs[key]
        labels = cfg.labels[idx]
        noise = labels == -1
        if noise.any():
            ax.scatter(xy[idx][noise, 0], xy[idx][noise, 1], s=6, c=NOISE_COLOR, label="Noise")
        cmap = plt.get_cmap("tab20")
        for c in sorted(set(labels[~noise])):
            m = labels == c
            ax.scatter(xy[idx][m, 0], xy[idx][m, 1], s=6, color=cmap((c - 1) % 20), label=f"Klaster {c}")
        ax.set_title(f"{cfg.name}\n{cfg.description}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if len(set(labels)) <= 10:
            ax.legend(fontsize=7, markerscale=2)
    return _save(fig, path, dpi)


def own_distance_plot(res: RunResult, path: Path, dpi: int) -> Path:
    """Boxplot jarak titik ke centroid klasternya sendiri, per klaster, untuk HDBSCAN dan K-Means k optimal."""
    X = res.prep.X
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, key in zip(axes, ("hdbscan", "kmeans_opt")):
        cfg = res.configs[key]
        data, names = [], []
        for c, centre in cfg.centroids.items():
            data.append(np.linalg.norm(X[cfg.labels == c] - centre, axis=1))
            names.append(str(c))
        ax.boxplot(data, tick_labels=names, showfliers=False)
        ax.set_title(f"{cfg.name}: jarak ke centroid sendiri")
        ax.set_xlabel("klaster")
        ax.set_ylabel("jarak Euclidean (ruang ter-skala)")
    return _save(fig, path, dpi)
