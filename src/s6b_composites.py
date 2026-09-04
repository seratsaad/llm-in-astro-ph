#!/usr/bin/env python3
"""Stage 6b: the four composite figures of the paper.

The analysis produces more panels than a paper should carry, so they are
composed into four multi-panel figures, following the layout of the
biomedical full-text study this work builds on:

  fig1  measurement and the placebo failure   (2 x 2)
  fig3  prevalence, fading excess, disclosure (1 x 3)
  fig4  marker discovery                      (1 x 2)

Figure 2 is the model schematic pair, composed in LaTeX from the TikZ
sources rather than here.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, FIGS, MARKERS, CONTROL, PLACEBO, quarter_label
import figstyle

figstyle.apply()
C = figstyle.C
SEC = ["intro", "methods", "results", "discussion", "conclusions"]
SECLAB = {"intro": "Introduction", "methods": "Methods/Data", "results": "Results",
          "discussion": "Discussion", "conclusions": "Conclusions"}


def qmid(q):
    return 2015 + q / 4.0 + 0.125


# ------------------------------------------------------------------ figure 1
def fig1():
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    ab = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.9))

    for ax, df, title in ((axes[0, 0], ab, "Abstracts"),
                          (axes[0, 1], ft, "Full text (body)")):
        for col, lab, colr in (("K_marker", "LLM markers (38 words)", C["vermillion"]),
                               ("K_control", "Neutral controls", C["blue"]),
                               ("K_placebo", "Hedges / intensifiers", C["orange"])):
            g = df.groupby("q").apply(
                lambda x: 1000 * x[col].sum() / x.L.sum(), include_groups=False)
            x = np.array([qmid(q) for q in g.index])
            y = g.values / g.values[:20].mean()
            ax.plot(x, y, lw=1.3, color=colr, label=lab)
            m = x < 2020
            cf = np.polyfit(x[m], np.log(y[m]), 1)
            ax.plot(x[~m], np.exp(np.polyval(cf, x[~m])), lw=1.0, ls=(0, (4, 2)),
                    color=colr, alpha=0.75)
        ax.axvline(2022.92, color=C["grey"], lw=1.1, alpha=0.55, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(2015, 2026.6)
    axes[0, 0].set_ylabel("Rate per 1000 tokens\n(relative to 2015--2019)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left",
                      bbox_to_anchor=(0.06, 0.94))

    # (c) and (d): within-section, markers then controls
    for ax, pre, title in ((axes[1, 0], "K", "LLM markers, within section"),
                           (axes[1, 1], "C", "Neutral controls, within section")):
        for s, colr in zip(SEC, [C["vermillion"], C["blue"], C["green"],
                                 C["orange"], C["purple"]]):
            g = ft.groupby("year").apply(
                lambda x: 1000 * x[f"{pre}_{s}"].sum() / max(x[f"L_{s}"].sum(), 1),
                include_groups=False)
            ax.plot(g.index, g / g.loc[2015:2019].mean(), color=colr, lw=1.2,
                    label=SECLAB[s])
        ax.axvline(2022.92, color=C["grey"], lw=1.1, alpha=0.55, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("First-submission year")
    axes[1, 1].set_ylim(0.7, 1.6)
    axes[1, 0].set_ylabel("Rate within section\n(relative to 2015--2019)")
    axes[1, 0].legend(frameon=False, fontsize=7.5, loc="upper left",
                      bbox_to_anchor=(0.06, 0.94))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_composite.pdf"))
    plt.close(fig)
    print("fig1 composite done")




def prim_csv():
    n = "pi_fulltext_primary_nuts.csv"
    return n if os.path.exists(os.path.join(DATA, n)) else "pi_fulltext_primary.csv"

# ------------------------------------------------------------------ figure 3
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75))

    # (a) prevalence with the drift bracket
    ax = axes[0]
    import os as _os
    prim = ("fulltext_primary_nuts"
            if _os.path.exists(_os.path.join(DATA, "pi_fulltext_primary_nuts.csv"))
            else "fulltext_primary")
    styles = {
        prim: ("Linear drift (primary)", C["vermillion"], "-", True),
        "fulltext_unconstrained": ("Unconstrained", C["black"], "--", False),
        "fulltext_frozen_drift": ("Frozen background", C["orange"], ":", False),
        "fulltext_tracked_drift": ("Control-tracked", C["blue"], "-.", False),
    }
    for tag, (lab, colr, ls, band) in styles.items():
        p = os.path.join(DATA, f"pi_{tag}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        qi = [int(q[:4]) * 4 + int(q[-1]) - 1 - 2015 * 4 for q in d.quarter]
        x = [qmid(q) for q in qi]
        ax.plot(x, 100 * d["mean"], color=colr, ls=ls, lw=1.3, label=lab)
        if band:
            ax.fill_between(x, 100 * d.lo, 100 * d.hi, color=colr, alpha=0.16, lw=0)
    ax.set_xlabel("First-submission quarter")
    ax.set_ylabel(r"Prevalence $\pi_t$ (% of papers)")
    ax.set_xlim(2020, 2026.6)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.075, 0.925),
              labelspacing=0.28, handlelength=1.5, handletextpad=0.5,
              borderpad=0.2)

    # (b) the fading excess, plus four representative marker words
    ax = axes[1]
    lp = os.path.join(DATA, "laplace_fulltext_primary.json")
    if os.path.exists(lp):
        d = json.load(open(lp))["delta_by_quarter"]
        qs = sorted(d, key=lambda q: (int(q[:4]), int(q[-1])))
        x = [int(q[:4]) + (int(q[-1]) - 1) / 4.0 + 0.125 for q in qs]
        ax.plot(x, [d[q] for q in qs], color=C["black"], lw=1.6,
                label=r"Fitted excess $e^{\delta_t}$")
    tr = os.path.join(DATA, "marker_trajectories_fulltext.csv")
    if os.path.exists(tr):
        t = pd.read_csv(tr)
        t = t[t.year >= 2019]
        for w, colr in zip(["delve", "underscore", "notably", "intricate"],
                           [C["vermillion"], C["blue"], C["green"], C["orange"]]):
            g = t[t.marker == w].groupby("year").excess_ratio.mean()
            if len(g):
                ax.plot(g.index + 0.5, g.values, lw=1.0, color=colr,
                        alpha=0.85, label=w)
    ax.set_xlabel("First-submission year")
    ax.set_ylabel(r"Excess over background")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 60)
    from matplotlib.ticker import FuncFormatter, FixedLocator
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=1, bbox_to_anchor=(0.075, 0.925),
              labelspacing=0.28, handlelength=1.5, handletextpad=0.5,
              borderpad=0.2)

    # (c) the disclosure gap
    ax = axes[2]
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"),
                         columns=["year", "declared"])
    decl = ft.groupby("year").declared.mean()
    p = os.path.join(DATA, prim_csv())
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["year"] = d.quarter.str[:4].astype(int)
        g = d.groupby("year")["mean"].mean()
        lo = d.groupby("year")["lo"].mean()
        hi = d.groupby("year")["hi"].mean()
        ax.plot(g.index, 100 * g, color=C["vermillion"], lw=1.4,
                label=r"Estimated $\pi_t$")
        ax.fill_between(g.index, 100 * lo, 100 * hi, color=C["vermillion"],
                        alpha=0.16, lw=0)
    ax.plot(decl.index, 100 * decl.clip(lower=1e-4), color=C["blue"], lw=1.4,
            marker="o", ms=2.5, label="Declared use")
    ax.set_yscale("log")
    ax.set_xlim(2019.5, 2026.5)
    ax.set_ylim(0.05, 200)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.set_xlabel("First-submission year")
    ax.set_ylabel(r"Percentage of papers")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.075, 0.925),
              labelspacing=0.28, handlelength=1.5, handletextpad=0.5,
              borderpad=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_composite.pdf"))
    plt.close(fig)
    print("fig3 composite done")




# ---------------------------------------------- figure 3 split (YST note 10)
def fig3_split():
    """The three arguments of the old composite as standalone figures."""
    # (a) prevalence with the drift bracket
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    import os as _os
    prim = ("fulltext_primary_nuts"
            if _os.path.exists(_os.path.join(DATA, "pi_fulltext_primary_nuts.csv"))
            else "fulltext_primary")
    styles = {
        prim: ("Linear drift (primary)", C["vermillion"], "-", True),
        "fulltext_unconstrained": ("Unconstrained", C["black"], "--", False),
        "fulltext_frozen_drift": ("Frozen background", C["orange"], ":", False),
        "fulltext_tracked_drift": ("Control-tracked", C["blue"], "-.", False),
    }
    for tag, (lab, colr, ls, band) in styles.items():
        p = os.path.join(DATA, f"pi_{tag}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        qi = [int(q[:4]) * 4 + int(q[-1]) - 1 - 2015 * 4 for q in d.quarter]
        x = [qmid(q) for q in qi]
        ax.plot(x, 100 * d["mean"], color=colr, ls=ls, lw=1.3, label=lab)
        if band:
            ax.fill_between(x, 100 * d.lo, 100 * d.hi, color=colr,
                            alpha=0.16, lw=0)
    ax.set_xlabel("First-submission quarter")
    ax.set_ylabel(r"Prevalence $\pi_t$ (% of papers)")
    ax.set_xlim(2020, 2026.7)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.legend(frameon=False, fontsize=6.6, loc="upper left",
              bbox_to_anchor=(0.075, 0.925), labelspacing=0.3,
              handlelength=1.5, handletextpad=0.5, borderpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_prevalence.pdf"))
    plt.close(fig)

    # (b) the fading excess with four representative words
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    lp = os.path.join(DATA, "laplace_fulltext_primary.json")
    if os.path.exists(lp):
        d = json.load(open(lp))["delta_by_quarter"]
        qs = sorted(d, key=lambda q: (int(q[:4]), int(q[-1])))
        x = [int(q[:4]) + (int(q[-1]) - 1) / 4.0 + 0.125 for q in qs]
        ax.plot(x, [d[q] for q in qs], color=C["black"], lw=1.6,
                label=r"Fitted excess $e^{\delta_t}$")
    tr = os.path.join(DATA, "marker_trajectories_fulltext.csv")
    if os.path.exists(tr):
        t = pd.read_csv(tr)
        t = t[t.year >= 2019]
        for w, colr in zip(["delve", "underscore", "notably", "intricate"],
                           [C["vermillion"], C["blue"], C["green"], C["orange"]]):
            g = t[t.marker == w].groupby("year").excess_ratio.mean()
            if len(g):
                ax.plot(g.index + 0.5, g.values, lw=1.0, color=colr,
                        alpha=0.85, label=w)
    ax.set_xlabel("First-submission year")
    ax.set_ylabel("Excess over background")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 60)
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=6.6, loc="upper left",
              bbox_to_anchor=(0.075, 0.925), labelspacing=0.3,
              handlelength=1.5, handletextpad=0.5, borderpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_fading.pdf"))
    plt.close(fig)

    # (c) the disclosure gap
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"),
                         columns=["year", "declared"])
    decl = ft.groupby("year").declared.mean()
    p = os.path.join(DATA, prim_csv())
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["year"] = d.quarter.str[:4].astype(int)
        g = d.groupby("year")["mean"].mean()
        lo = d.groupby("year")["lo"].mean()
        hi = d.groupby("year")["hi"].mean()
        ax.plot(g.index, 100 * g, color=C["vermillion"], lw=1.4,
                label=r"Estimated $\pi_t$")
        ax.fill_between(g.index, 100 * lo, 100 * hi, color=C["vermillion"],
                        alpha=0.16, lw=0)
    ax.plot(decl.index, 100 * decl.clip(lower=1e-4), color=C["blue"], lw=1.4,
            marker="o", ms=2.5, label="Declared use")
    ax.set_yscale("log")
    ax.set_xlim(2019.5, 2026.7)
    ax.set_ylim(0.05, 200)
    from matplotlib.ticker import FuncFormatter as _FF
    ax.yaxis.set_major_formatter(_FF(lambda v, _: f"{v:g}"))
    ax.set_xlabel("First-submission year")
    ax.set_ylabel("Percentage of papers")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left",
              bbox_to_anchor=(0.075, 0.925), labelspacing=0.3,
              handlelength=1.5, handletextpad=0.5, borderpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_gap.pdf"))
    plt.close(fig)
    print("fig3 split done")


# ------------------------------------------------------------------ figure 4
def fig4():
    p = os.path.join(DATA, "discovered_markers.csv")
    if not os.path.exists(p):
        print("fig4 skipped")
        return
    d = pd.read_csv(p)
    sel = set()
    bp = os.path.join(DATA, "expanded_basket.json")
    if os.path.exists(bp):
        sel = set(json.load(open(bp))["words"])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))

    ax = axes[0]
    ax.scatter(d.base_df, np.clip(d.disc_ratio, 0.2, 12), s=1.4, c=C["grey"],
               alpha=0.22, lw=0, rasterized=True)
    pick = d[d.word.isin(sel)]
    ax.scatter(pick.base_df, np.clip(pick.disc_ratio, 0.2, 12), s=10,
               facecolors="none", edgecolors=C["black"], lw=0.5,
               label="Selected here (Basket 2)", zorder=2)
    ax.scatter(d[d.is_seed].base_df, np.clip(d[d.is_seed].disc_ratio, 0.2, 12),
               s=10, c=C["vermillion"], lw=0, label="Imported markers (Basket 1)", zorder=3)
    ax.scatter(d[d.is_control].base_df, np.clip(d[d.is_control].disc_ratio, 0.2, 12),
               s=10, c=C["blue"], lw=0, label="Neutral controls", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Document frequency, 2015--2019")
    ax.set_ylabel("Excess over background\n(discovery: 2023--2024)")
    ax.set_ylim(0.28, 34)
    ax.legend(frameon=False, fontsize=8.6,
              labelspacing=0.32, handlelength=1.4, handletextpad=0.5,
              borderpad=0.2, loc="upper right", bbox_to_anchor=(0.935, 0.925))

    # (b) held-out validation of the frozen selection
    ax = axes[1]
    ax.scatter(d.disc_ratio, d.val_ratio, s=1.4, c=C["grey"], alpha=0.22,
               lw=0, rasterized=True)
    ax.scatter(pick.disc_ratio, pick.val_ratio, s=11, facecolors="none",
               edgecolors=C["black"], lw=0.6, label="Selected here (Basket 2)", zorder=3)
    ax.scatter(d[d.is_control].disc_ratio, d[d.is_control].val_ratio, s=11,
               c=C["blue"], lw=0, label="Neutral controls", zorder=4)
    lim = [0.4, 12]
    ax.plot(lim, lim, color=C["black"], lw=0.7, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(0.3, 25)
    ax.set_xlabel("Discovery excess (2023--2024)")
    ax.set_ylabel("Held-out excess (2025--2026)")
    ax.legend(frameon=False, fontsize=8.6,
              labelspacing=0.32, handlelength=1.4, handletextpad=0.5,
              borderpad=0.2, loc="upper left", bbox_to_anchor=(0.075, 0.925))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_composite.pdf"), dpi=400)
    plt.close(fig)
    print("fig4 composite done")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig1()
    fig3()
    fig3_split()
    fig4()


# ------------------------------------------------------------ supplementary
def figS1():
    """Simulation-based calibration: recovered vs injected prevalence."""
    p = os.path.join(DATA, "calibration.json")
    if not os.path.exists(p):
        print("figS1 skipped")
        return
    rows = pd.DataFrame(json.load(open(p)))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))

    ax = axes[0]
    for d, colr, mk in zip(sorted(rows.true_delta.unique()),
                           [C["orange"], C["blue"], C["green"]], ["s", "o", "^"]):
        g = rows[rows.true_delta == d].sort_values("true_pi")
        ax.errorbar(100 * g.true_pi, 100 * g.pi_mean, yerr=100 * g.pi_sd,
                    color=colr, marker=mk, ms=3.5, lw=1.1, capsize=2,
                    label=rf"$e^{{\delta}}={d}$")
    ax.plot([0, 80], [0, 80], color=C["black"], lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel(r"Injected prevalence (%)")
    ax.set_ylabel(r"Recovered prevalence (%)")
    ax.legend(frameon=False, fontsize=7.2,
              labelspacing=0.28, handlelength=1.4, handletextpad=0.5,
              borderpad=0.2, loc="upper left", bbox_to_anchor=(0.075, 0.925))

    ax = axes[1]
    for d, colr, mk in zip(sorted(rows.true_delta.unique()),
                           [C["orange"], C["blue"], C["green"]], ["s", "o", "^"]):
        g = rows[rows.true_delta == d].sort_values("true_pi")
        ax.plot(100 * g.true_pi, 100 * (g.pi_mean - g.true_pi), color=colr,
                marker=mk, ms=3.5, lw=1.1, label=rf"$e^{{\delta}}={d}$")
    ax.axhline(0, color=C["black"], lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel(r"Injected prevalence (%)")
    ax.set_ylabel(r"Bias (percentage points)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figS1_calibration.pdf"))
    plt.close(fig)
    print("figS1 done")
