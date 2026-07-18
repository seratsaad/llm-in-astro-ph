#!/usr/bin/env python3
"""Astrobites-style figures for the LLM-in-astronomy analysis."""
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)

# PANTERA-letter visual style (serif fonts, warm palette, boxed inward ticks)
from pantera_style import C  # noqa: E402  (sets rcParams on import)

def footer(fig, txt=None):
    return  # source note lives in the figure caption, not on the figure

# ---------------------------------------------------------------- FIG 1
def fig1_markers_vs_control():
    ym = pd.read_csv(os.path.join(DATA, "year_markers.csv"))
    a = pd.read_csv(os.path.join(DATA, "alpha_series.csv"))
    # aggregate strong-basket per year (from alpha series -> yearly)
    yr = (a.assign(hitw=a.hit).groupby("year")
            .apply(lambda g: pd.Series({"rate": g.hit.sum()/g.total.sum()}))
            .reset_index())
    # control mean per year
    ctrl_words = ["observed","measured","obtained","presented","galaxy","stellar",
                  "sample","temperature","redshift","spectra"]
    ctrl = ym[ym.word.isin(ctrl_words)].groupby("year").freq.mean().reset_index()

    # end at the last full year: 2026 is partial, and the marker/control values are
    # per-abstract fractions (not counts), so a partial year is not comparable here.
    yr = yr[yr.year <= 2025]
    ctrl = ctrl[ctrl.year <= 2025]
    # Wilson 95% intervals on the yearly basket rate
    agg = a.groupby("year").agg(hit=("hit", "sum"), total=("total", "sum")).reset_index()
    agg = agg[agg.year <= 2025]
    import math
    def wilson(k, n, z=1.96):
        p = k/n; d = 1 + z*z/n
        c = (p + z*z/(2*n))/d
        h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
        return c-h, c+h
    lohi = [wilson(k, n) for k, n in zip(agg.hit, agg.total)]
    yerr = np.array([[r*100 - lo*100 for (lo, hi), r in zip(lohi, agg.hit/agg.total)],
                     [hi*100 - r*100 for (lo, hi), r in zip(lohi, agg.hit/agg.total)]])
    fig, ax = plt.subplots(figsize=(4.8, 2.85))
    ax.errorbar(yr.year, yr.rate*100, yerr=yerr, fmt="-o", color=C["vermillion"],
                lw=1.3, ms=3.4, capsize=2, elinewidth=0.9,
                label="LLM marker basket\n(delve, underscore, intricate, pivotal, ...)")
    ax.plot(ctrl.year, ctrl.freq*100, "-s", color=C["blue"], lw=1.3, ms=3.4,
            label="Neutral control words\n(observed, measured, galaxy, ...)")
    ax.set_ylim(0, 15)
    # counterfactual: linear extrapolation of the 2015-2021 trend (Kobak-style)
    pre = yr[yr.year <= 2021]
    coef = np.polyfit(pre.year, pre.rate*100, 1)
    xx = np.array([2022, 2025.4])
    ax.plot(xx, np.polyval(coef, xx), ls=(0, (3, 2)), lw=0.9, color=C["black"], zorder=2)
    ax.text(2025.0, np.polyval(coef, 2024.4) - 0.4, "pre-2022 trend", fontsize=8.5,
            color=C["black"], ha="right", va="top")
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 10.8, "ChatGPT", rotation=90, va="top", ha="right",
            fontsize=9.5, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("% of abstracts containing the word(s)")
    ax.legend(loc="center left", fontsize=9)
    ax.set_xticks(range(2015, 2026, 2))
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_markers_vs_control.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 2 (delve collapse)
def fig2_delve_collapse():
    """Kobak-style: aggregate quarterly rate on top; per-word small multiples below,
    each with its own y-range and a dashed pre-2022 counterfactual segment."""
    q = pd.read_csv(os.path.join(DATA, "quarter_markers.csv"))
    a = pd.read_csv(os.path.join(DATA, "alpha_series.csv"))
    ym = pd.read_csv(os.path.join(DATA, "year_markers.csv"))
    words = ["delve", "underscore", "leveraging", "pivotal"]

    fig = plt.figure(figsize=(6.5, 3.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.52, wspace=0.42)
    top = fig.add_subplot(gs[0, :])
    top.plot(a.yq, a.rate*100, color=C["black"], lw=1.3)
    top.axvline(2024.35, color=C["grey"], ls=":", lw=1.0)
    top.set_xlim(2022, 2026.6)
    top.set_ylabel("% with any marker", fontsize=9)
    top.text(0.015, 0.93, "any marker word", transform=top.transAxes,
             va="top", ha="left", fontsize=9.5, fontweight="bold")
    top.tick_params(labelsize=9)

    for i, w in enumerate(words):
        ax = fig.add_subplot(gs[1, i])
        d = q[(q.word == w) & (q.yq >= 2022)].sort_values("yq")
        # counterfactual: linear fit to the 2015-2021 yearly rates, extrapolated
        yy = ym[(ym.word == w) & (ym.year <= 2021)]
        coef = np.polyfit(yy.year, yy.freq*100, 1)
        xx = np.array([2022, 2026.5])
        ax.plot(xx, np.clip(np.polyval(coef, xx), 0, None), ls=(0, (3, 2)),
                lw=0.9, color=C["black"], zorder=2)
        ax.plot(d.yq, d.freq*100, "-o", ms=2.2, lw=1.1, color=C["blue"], zorder=3)
        ax.axvline(2024.35, color=C["grey"], ls=":", lw=0.8)
        ax.set_xlim(2022, 2026.6)
        ax.set_ylim(0, max(d.freq.max()*100*1.25, 0.1))
        ax.text(0.05, 0.94, w, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8.5)
        ax.set_xticks([2022, 2024, 2026])
        if i == 0:
            ax.set_ylabel("% of abstracts", fontsize=9)
    fig.savefig(os.path.join(FIGS, "fig2_delve_collapse.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 3 (data-driven discovery)
def fig3_discovery():
    """Kobak-style volcano: full vocabulary as a cloud, frequency ratio vs recent
    frequency on log axes, with the classified tail annotated directly."""
    from pantera_style import place_labels
    instruments = {"nircam","miri","nirspec","jwst","desi","xrism","mrs","ixpe",
                   "lhaaso","prism","kagra","noon","euclid","ligo","virgo"}
    ml_topic = {"interpretable","differentiable","normalizing","transformer","embeddings"}
    style = {"leveraging","aligns","offering","pivotal","intricate","highlighting",
             "examines","align","advancing","establishes","refining","incorporating",
             "investigates","showcasing","underscore","underscores","frameworks",
             "struggle","delve","nuanced","boasts","tapestry","meticulous","garner",
             "elucidate","multifaceted","comprehensively"}
    wf = pd.read_parquet(os.path.join(DATA, "word_docfreq.parquet"))
    wf = wf[(wf.rec_freq > 0) & (wf.ratio > 0)].copy()
    bg = pd.read_csv(os.path.join(DATA, "bigram_discovery.csv"))
    bg = bg[(bg.rec_freq > 0) & (bg.ratio > 0)].copy()
    topic_kw = ("jwst","desi","kagra","virgo","webb","spectroscopic","instrument",
                "habitable","worlds","noon","nircam","miri","nirspec","xrism",
                "language","transformer","simulation","cosmos","euclid","lrds",
                "little","red","dots")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.6))
    for ax in (axA, axB):
        ax.set_xscale("log"); ax.set_yscale("log")

    # ---- panel (a): single words ----
    axA.scatter(wf.rec_freq*100, wf.ratio, s=1.5, color="#C9C9C9", alpha=0.5,
                linewidths=0, zorder=1, rasterized=True)
    iconic = {"delve", "delves", "tapestry", "meticulous", "nuanced"}
    sel = wf[((wf.base_freq > 0.0005) | wf.word.isin(iconic)) & (wf.ratio >= 4)].copy()
    def kindw(w):
        if w in instruments: return "inst"
        if w in ml_topic: return "ml"
        if w in style: return "sty"
        return None
    sel["kind"] = sel.word.map(kindw)
    sel = sel[sel.kind.notna()]
    kcol = {"inst": C["blue"], "sty": C["vermillion"], "ml": C["green"]}
    for k, g in sel.groupby("kind"):
        axA.scatter(g.rec_freq*100, g.ratio, s=10, color=kcol[k], zorder=4, linewidths=0)
    axA.axhline(5, color=C["grey"], ls="--", lw=0.7, zorder=2)
    axA.set_xlim(3e-3, 30); axA.set_ylim(0.25, 40)
    axA.set_xlabel("frequency in 2024-2026 (% of abstracts)")
    axA.set_ylabel("frequency ratio, 2024-2026 / 2018-2021")
    axA.text(0.025, 0.97, "(a) single words", transform=axA.transAxes,
             va="top", ha="left", fontsize=9.5, fontweight="bold", zorder=7)
    show_a = ["nircam", "miri", "nirspec", "jwst", "desi", "xrism", "ixpe",
              "interpretable", "leveraging", "aligns", "offering", "pivotal",
              "intricate", "highlighting", "delve"]
    ann = sel[sel.word.isin(show_a)]
    others = sel[~sel.word.isin(show_a)]
    place_labels(axA, list(ann.rec_freq*100), list(ann.ratio), list(ann.word),
                 colors=[kcol[k] for k in ann.kind], fontsize=8.5,
                 obstacles=list(zip(others.rec_freq*100, others.ratio)))

    # ---- panel (b): two-word phrases ----
    axB.scatter(bg.rec_freq*100, bg.ratio, s=1.5, color="#C9C9C9", alpha=0.4,
                linewidths=0, zorder=1, rasterized=True)
    selb = bg[(bg.base_freq > 0.0005) & (bg.ratio >= 4)].copy()
    def kindb(t):
        return "inst" if any(k in t.split() for k in topic_kw) else "sty"
    selb["kind"] = selb.bigram.map(kindb)
    for k, g in selb.groupby("kind"):
        axB.scatter(g.rec_freq*100, g.ratio, s=10, color=kcol[k], zorder=4, linewidths=0)
    axB.axhline(5, color=C["grey"], ls="--", lw=0.7, zorder=2)
    axB.set_xlim(3e-3, 30); axB.set_ylim(0.25, 40)
    axB.set_xlabel("frequency in 2024-2026 (% of abstracts)")
    axB.text(0.025, 0.97, "(b) two-word phrases", transform=axB.transAxes,
             va="top", ha="left", fontsize=9.5, fontweight="bold", zorder=7)
    show_b = ["jwst observations", "the desi", "energy spectroscopic",
              "habitable worlds", "leveraging the", "align with", "offering a",
              "results highlight", "findings indicate"]
    annb = selb[selb.bigram.isin(show_b)]
    othb = selb[~selb.bigram.isin(show_b)]
    place_labels(axB, list(annb.rec_freq*100), list(annb.ratio), list(annb.bigram),
                 colors=[kcol[k] for k in annb.kind], fontsize=8.5,
                 obstacles=list(zip(othb.rec_freq*100, othb.ratio)))

    from matplotlib.lines import Line2D
    axA.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["blue"], ms=4,
               label="new instrument / survey / topic"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["vermillion"], ms=4,
               label="style word / phrase"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["green"], ms=4,
               label="ML method term")],
        loc="lower left", fontsize=9, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_discovery.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 4 (disclosure)
def fig4_disclosure():
    d = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
    yrs = [int(y) for y in d["years"]]
    ack = [d["ack"][str(y)] for y in yrs]
    full = [d["full"][str(y)] for y in yrs]
    tot = [d["total_astro"][str(y)] for y in yrs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.74))
    ax1.bar([y-0.2 for y in yrs], full, width=0.4, color=C["sky"], label="mentions in full text")
    ax1.bar([y+0.2 for y in yrs], ack, width=0.4, color=C["vermillion"],
            label="stated in acknowledgments")
    from pantera_style import no_minor_x
    no_minor_x(ax1)
    ax1.set_ylabel("number of papers per year"); ax1.set_xlabel("Year")
    ax1.legend(fontsize=9, loc="upper left"); ax1.set_xticks(range(2016,2027,2))
    # fraction
    frac_ack = [100*a/t for a,t in zip(ack,tot)]
    frac_full = [100*f/t for f,t in zip(full,tot)]
    ax2.plot(yrs, frac_full, "-o", color=C["sky"], lw=1.3, label="full text")
    ax2.plot(yrs, frac_ack, "-o", color=C["vermillion"], lw=1.3, label="acknowledgments")
    ax2.set_ylabel("% of astronomy papers"); ax2.set_xlabel("Year")
    ax2.legend(fontsize=9); ax2.set_xticks(range(2016,2027,2))
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    footer(fig); fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(os.path.join(FIGS, "fig4_disclosure.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 5 (the gap + fields)
def fig5_gap():
    from pantera_style import no_minor_y
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    d = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
    est = al["by_year"]["2025"]["excess"]*100
    disc_full = 100*d["full"]["2025"]/d["total_astro"]["2025"]
    disc_ack = 100*d["ack"]["2025"]/d["total_astro"]["2025"]

    # one log-axis dot plot: literature estimates (grey), astro-ph estimate
    # (accent, lower bound -> right arrow), astro-ph disclosure (open circles)
    rows = [
        ("Computer science (Liang et al.)",     17.5,     C["grey"], "lit"),
        ("Biomedicine (Kobak et al.)",          13.5,     C["grey"], "lit"),
        ("Mathematics / Nature (Liang et al.)",  6.3,     C["grey"], "lit"),
        ("astro-ph, marker-word lower bound",    est,     C["vermillion"], "lb"),
        ("astro-ph, any AI mention in text",     disc_full, C["blue"], "open"),
        ("astro-ph, stated in acknowledgments",  disc_ack,  C["blue"], "open"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ys = list(range(len(rows)))
    for yi, (lab, v, col, kind) in zip(ys, rows):
        ax.plot([0.1, v], [yi, yi], lw=0.7, color="#CCCCCC", zorder=1)
        if kind == "open":
            ax.plot(v, yi, "o", ms=5, mfc="white", mec=col, mew=1.1, zorder=3)
        else:
            ax.plot(v, yi, "o", ms=5, color=col, zorder=3)
        if kind == "lb":   # lower bound: arrow to the right
            ax.annotate("", xy=(v*1.9, yi), xytext=(v*1.12, yi),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(v*2.2 if kind == "lb" else v*1.25, yi,
                f"{v:.2f}%" if v < 1 else f"{v:.1f}%",
                va="center", fontsize=9, color="#555555")
    no_minor_y(ax)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(0.1, 60)
    ax.set_xlabel("% of papers with model-edited text (2025), log scale", labelpad=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_gap.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 6 (volume)
def fig6_volume():
    yc = pd.read_csv(os.path.join(DATA, "year_counts.csv"))
    yc = yc[yc.year <= 2025]  # drop partial 2026 for clean trend
    fig, ax = plt.subplots(figsize=(4.80, 2.76))
    ax.bar(yc.year, yc.n_abstracts, color=C["blue"])
    ax.axvline(2022.85, color=C["vermillion"], ls="--", lw=1.5)
    ax.text(2023.0, yc.n_abstracts.max()*0.96, "ChatGPT", color=C["vermillion"], fontsize=9)
    ax.set_xlabel("Year"); ax.set_ylabel("astro-ph submissions (sampled via arXiv API)")
    ax.set_xticks(range(2015,2026,2))
    footer(fig); fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(os.path.join(FIGS, "fig6_volume.png"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    fig1_markers_vs_control(); print("fig1 done")
    fig2_delve_collapse(); print("fig2 done")
    fig3_discovery(); print("fig3 done")
    fig4_disclosure(); print("fig4 done")
    fig5_gap(); print("fig5 done")
    fig6_volume(); print("fig6 done")
    print("ALL FIGURES WRITTEN to figs/")
