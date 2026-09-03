#!/usr/bin/env python3
"""Stage 6: figures.

Genre conventions (Kobak 2024, Liang 2024, Geng & Trotta 2025):
no ranked-category bars with printed values; ranked words appear as annotated
scatter or in tables; every "observed exceeds expected" argument carries an
explicit dashed counterfactual line; frequency units are named on every axis.
"""
import glob
import json
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, FIGS, MARKERS, MARKERS_STRONG, CONTROL, PLACEBO, quarter_label
import figstyle

figstyle.apply()
C = figstyle.C
SEC = ["intro", "methods", "results", "discussion", "conclusions"]
SECLAB = {"intro": "Introduction", "methods": "Methods/Data", "results": "Results",
          "discussion": "Discussion", "conclusions": "Conclusions"}


def qmid(q):
    return 2015 + q / 4.0 + 0.125


# ------------------------------------------------------------------ figure 1
def fig1_rates():
    """Marker vs control occurrence rate, with the pre-2020 counterfactual."""
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    ab = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharex=True)

    for ax, df, title in ((axes[0], ab, "Abstracts"), (axes[1], ft, "Full text (body)")):
        for col, lab, colr in (("K_marker", "LLM marker basket (38 words)", C["vermillion"]),
                               ("K_control", "Neutral astronomy control", C["blue"]),
                               ("K_placebo", "Hedge / intensifier basket", C["orange"])):
            g = df.groupby("q").apply(
                lambda x: 1000 * x[col].sum() / x.L.sum(), include_groups=False)
            x = np.array([qmid(q) for q in g.index])
            y = g.values / g.values[:20].mean()          # relative to 2015-2019
            ax.plot(x, y, lw=1.3, color=colr, label=lab)
            # counterfactual: log-linear fit on the known-negative era, extended
            m = x < 2020
            cf = np.polyfit(x[m], np.log(y[m]), 1)
            ax.plot(x[~m], np.exp(np.polyval(cf, x[~m])), lw=1.0, ls=(0, (4, 2)),
                    color=colr, alpha=0.75)
        ax.text(2022.99, ax.get_ylim()[1] * 0.97, "ChatGPT", fontsize=7,
                color=C["grey"], va="top", rotation=90)
        ax.set_title(title)
        ax.set_xlabel("First-submission year")
    axes[0].set_ylabel("Occurrence rate per 1000 tokens\n(relative to 2015--2019)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    for ax in axes:
        ax.set_xlim(2015, 2026.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_rates.pdf"))
    plt.close(fig)
    print("fig1 done")


# ------------------------------------------------------------------ figure 2
def fig2_prevalence(tags=("fulltext_primary", "fulltext_unconstrained",
                          "fulltext_frozen_drift", "fulltext_tracked_drift")):
    """Posterior prevalence by quarter under the drift and shape assumptions."""
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    styles = {
        "fulltext_primary": ("Linear drift, monotone", C["vermillion"], "-", True),
        "fulltext_unconstrained": ("Linear drift, unconstrained", C["black"], "--", False),
        "fulltext_frozen_drift": ("Frozen background", C["orange"], ":", False),
        "fulltext_tracked_drift": ("Control-tracked background", C["blue"], "-.", False),
    }
    for tag in tags:
        p = os.path.join(DATA, f"pi_{tag}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        lab, colr, ls, band = styles[tag]
        qi = [int(q[:4]) * 4 + int(q[-1]) - 1 - 2015 * 4 for q in d.quarter]
        x = [qmid(q) for q in qi]
        ax.plot(x, 100 * d["mean"], color=colr, ls=ls, lw=1.3, label=lab)
        if band:
            ax.fill_between(x, 100 * d.lo, 100 * d.hi, color=colr, alpha=0.16, lw=0)
    ax.set_xlabel("First-submission quarter")
    ax.set_ylabel(r"Posterior prevalence $\pi_t$ (% of papers)")
    ax.set_xlim(2020, 2026.6)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_prevalence.pdf"))
    plt.close(fig)
    print("fig2 done")


# ------------------------------------------------------------------ figure 3
def fig3_marker_trajectories(phase="fulltext"):
    """Small multiples: does each marker fade while prevalence rises?"""
    p = os.path.join(DATA, f"marker_trajectories_{phase}.csv")
    if not os.path.exists(p):
        print("fig3 skipped (no trajectories)")
        return
    d = pd.read_csv(p)
    d = d[d.year >= 2018]
    show = (d[d.year >= 2024].groupby("marker").excess_ratio.mean()
            .sort_values(ascending=False).head(12).index.tolist())
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 4.6), sharex=True)
    for ax, w in zip(axes.ravel(), show):
        g = d[d.marker == w].groupby("year").agg(
            obs=("obs_rate", "mean"), bg=("bg_rate", "mean"))
        ax.plot(g.index, 1e4 * g.obs, color=C["vermillion"], lw=1.2)
        ax.plot(g.index, 1e4 * g.bg, color=C["grey"], lw=0.9, ls=(0, (4, 2)))
        ax.set_title(w, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([2018, 2021, 2024])
    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=7.5)
    for r in range(axes.shape[0]):
        axes[r, 0].set_ylabel("per 10,000 tokens", fontsize=7)
    handles = [Line2D([], [], color=C["vermillion"], lw=1.2, label="Observed"),
               Line2D([], [], color=C["grey"], lw=0.9, ls=(0, (4, 2)),
                      label="Pre-2020 extrapolation")]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIGS, "fig3_marker_trajectories.pdf"))
    plt.close(fig)
    print("fig3 done")


# ------------------------------------------------------------------ figure 6
def fig6_discovery():
    """Volcano: whole-vocabulary excess, with the tail annotated directly."""
    p = os.path.join(DATA, "discovered_markers.csv")
    if not os.path.exists(p):
        print("fig6 skipped")
        return
    d = pd.read_csv(p)
    # The selected words are named in the table, not on the plot: a dozen
    # labels crowd into one decade of document frequency and become illegible.
    sel = set()
    bpath = os.path.join(DATA, "expanded_basket.json")
    if os.path.exists(bpath):
        sel = set(json.load(open(bpath))["words"])

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    r = np.clip(d.disc_ratio, 0.2, 12)
    ax.scatter(d.base_df, r, s=1.4, c=C["grey"], alpha=0.22, lw=0,
               rasterized=True, label="All stable-background words")
    picked = d[d.word.isin(sel)]
    ax.scatter(picked.base_df, np.clip(picked.disc_ratio, 0.2, 12), s=9,
               facecolors="none", edgecolors=C["black"], lw=0.5,
               label="Selected (Table 2)", zorder=2)
    seed = d[d.is_seed]
    ax.scatter(seed.base_df, np.clip(seed.disc_ratio, 0.2, 12), s=9,
               c=C["vermillion"], lw=0, label="Frozen seed markers", zorder=3)
    ctrl = d[d.is_control]
    ax.scatter(ctrl.base_df, np.clip(ctrl.disc_ratio, 0.2, 12), s=9,
               c=C["blue"], lw=0, label="Neutral controls", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Document frequency, 2015--2019")
    ax.set_ylabel("Excess over extrapolated background\n(2023--2024)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right",
              handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_discovery.pdf"), dpi=400)
    plt.close(fig)
    print("fig6 done")


# ------------------------------------------------------------------ figure 7
def fig7_sections():
    """Section comparison as positions in a plane, not bar heights."""
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    if f"C_{SEC[0]}" not in ft.columns:
        print("fig7 skipped (no per-section control counts)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    ax = axes[0]
    for s, colr in zip(SEC, [C["vermillion"], C["blue"], C["green"],
                             C["orange"], C["purple"]]):
        g = ft.groupby("year").apply(
            lambda x: pd.Series({
                "mk": 1000 * x[f"K_{s}"].sum() / max(x[f"L_{s}"].sum(), 1),
                "ct": 1000 * x[f"C_{s}"].sum() / max(x[f"L_{s}"].sum(), 1)}),
            include_groups=False)
        ax.plot(g.index, g.mk / g.mk.loc[2015:2019].mean(), color=colr, lw=1.2,
                label=SECLAB[s])
        axes[1].plot(g.index, g.ct / g.ct.loc[2015:2019].mean(), color=colr, lw=1.2)
    ax.set_ylabel("Marker rate\n(relative to 2015--2019)")
    axes[1].set_ylabel("Control rate\n(relative to 2015--2019)")
    for a in axes:
        a.set_xlabel("First-submission year")
        a.axhline(1.0, color=C["black"], lw=0.6, ls=(0, (4, 2)))
    axes[1].set_ylim(0.7, 1.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_title("LLM markers, within section")
    axes[1].set_title("Neutral controls, within section")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_sections.pdf"))
    plt.close(fig)
    print("fig7 done")


# ------------------------------------------------------------------ figure 8
def fig8_disclosure():
    """Estimated prevalence against measured declaration, on one log axis."""
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    decl = ft.groupby("year").declared.agg(["mean", "sum", "size"])
    p = os.path.join(DATA, "pi_fulltext_primary.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(decl.index, 100 * decl["mean"], color=C["blue"], lw=1.3,
            marker="o", ms=2.5, label="Declared LLM use (full text)")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["year"] = [int(q[:4]) for q in d.quarter]
        g = d.groupby("year")["mean"].mean()
        lo = d.groupby("year")["lo"].mean()
        hi = d.groupby("year")["hi"].mean()
        ax.plot(g.index, 100 * g, color=C["vermillion"], lw=1.3,
                label=r"Posterior prevalence $\pi_t$")
        ax.fill_between(g.index, 100 * lo, 100 * hi, color=C["vermillion"],
                        alpha=0.16, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("First-submission year")
    ax.set_ylabel(r"Percentage of papers")
    ax.set_xlim(2019.5, 2026.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_disclosure.pdf"))
    plt.close(fig)
    print("fig8 done")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    which = sys.argv[1:] or ["1", "2", "3", "6", "7", "8"]
    fns = {"1": fig1_rates, "2": fig2_prevalence, "3": fig3_marker_trajectories,
           "6": fig6_discovery, "7": fig7_sections, "8": fig8_disclosure}
    for k in which:
        try:
            fns[k]()
        except Exception as e:
            print(f"fig{k} FAILED: {type(e).__name__}: {e}")
