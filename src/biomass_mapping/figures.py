from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_coverage_plot(table: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(table["resolution_m"], table["coverage_percent"], marker="o")
    ax.set_xlabel("Résolution spatiale (m)")
    ax.set_ylabel("Cellules avec ≥ 1 footprint GEDI (%)")
    ax.set_title("Couverture spatiale GEDI selon la résolution")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    for _, r in table.iterrows():
        ax.annotate(
            f'{r["coverage_percent"]:.1f}%',
            (r["resolution_m"], r["coverage_percent"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_density_by_resolution(table: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        table["resolution_m"],
        table["mean_shots_per_covered_cell"],
        marker="o",
        label="Moyenne",
    )
    ax.plot(
        table["resolution_m"],
        table["median_shots_per_covered_cell"],
        marker="o",
        label="Médiane",
    )
    ax.set_xlabel("Résolution spatiale (m)")
    ax.set_ylabel("Footprints GEDI par cellule couverte")
    ax.set_title("Densité d'observations GEDI selon la résolution")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_agbd_hist(df: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    upper = df["agbd"].quantile(0.995)
    x = df.loc[df["agbd"] <= upper, "agbd"]
    ax.hist(x, bins=60)
    ax.set_xlabel("AGBD (Mg/ha)")
    ax.set_ylabel("Nombre de footprints")
    ax.set_title("Distribution de la biomasse GEDI")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_correlations(df: pd.DataFrame, target: str, features: list[str], path: str | Path):
    vals = []
    for f in features:
        pair = df[[target, f]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 10:
            continue
        vals.append((f, pair[target].corr(pair[f], method="spearman")))
    vals = sorted(vals, key=lambda z: abs(z[1]), reverse=True)[:20]
    names = [v[0] for v in vals][::-1]
    corr = [v[1] for v in vals][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.3 * len(names))))
    ax.barh(names, corr)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Corrélation de Spearman avec AGBD")
    ax.set_title("Relations monotones entre télédétection et biomasse")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
