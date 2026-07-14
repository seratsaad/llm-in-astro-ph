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
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(yr.year, yr.rate*100, "-o", color=C["vermillion"], lw=2.8, ms=6,
            label="LLM marker basket\n(delve, underscore, intricate, pivotal, ...)")
    ax.plot(ctrl.year, ctrl.freq*100, "-s", color=C["blue"], lw=2.2, ms=5,
            label="Neutral control words\n(observed, measured, galaxy, ...)")
    ax.set_ylim(0, 15)
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 14.3, "ChatGPT", rotation=90, va="top", ha="right",
            fontsize=8, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("% of abstracts containing the word(s)")
    ax.legend(loc="center left", fontsize=9.5)
    ax.set_xticks(range(2015, 2026, 2))
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_markers_vs_control.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 2 (delve collapse)
def fig2_delve_collapse():
    q = pd.read_csv(os.path.join(DATA, "quarter_markers.csv"))
    a = pd.read_csv(os.path.join(DATA, "alpha_series.csv"))
    def s(w):
        d = q[(q.word==w)].sort_values("yq")
        return d.yq.values, d.freq.values*100
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios":[1, 1.25]})
    # TOP: aggregate footprint doesn't shrink
    top.plot(a.yq, a.rate*100, color=C["black"], lw=2.2, zorder=5)
    top.set_ylabel("% with ANY marker")
    # BOTTOM: individual words, zoomed
    for w, col, lab in [("delve", C["vermillion"], "delve  (went viral → abandoned)"),
                        ("underscore", C["green"], "underscore"),
                        ("leveraging", C["blue"], "leveraging"),
                        ("pivotal", C["purple"], "pivotal")]:
        x, y = s(w)
        bot.plot(x, y, "-o", ms=3.8, lw=1.9, color=col, label=lab)
    bot.set_ylabel("% of abstracts (individual words)")
    bot.set_xlabel("Year (quarterly)")
    for ax in (top, bot):
        ax.axvline(2024.35, color=C["grey"], ls=":", lw=1.4)
    bot.set_xlim(2022, 2026.6)
    bot.set_ylim(0, 2.0)
    # legend in the clear upper-left (all lines are low there in 2022-2023);
    # annotation above the data, connected by the existing dotted line
    bot.legend(loc="upper left", ncol=1, fontsize=8.6)
    bot.text(2025.15, 1.9, '"delve" outed as the\nChatGPT tell (mid-2024)',
             fontsize=8.4, color=C["grey"], ha="center", va="top", style="italic")
    footer(fig); fig.tight_layout(rect=[0,0.03,1,0.97])
    fig.savefig(os.path.join(FIGS, "fig2_delve_collapse.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 3 (data-driven discovery)
def fig3_discovery():
    df = pd.read_parquet(os.path.join(DATA, "word_docfreq.parquet"))
    df = df[df.base_freq > 0.0005].copy()
    # classify
    instruments = {"nircam","miri","nirspec","jwst","desi","xrism","mrs","ixpe",
                   "lhaaso","prism","kagra","noon","euclid","ligo","virgo"}
    ml_topic = {"interpretable","differentiable","normalizing","transformer","embeddings"}
    style = {"leveraging","aligns","offering","pivotal","intricate","highlighting",
             "examines","align","advancing","establishes","refining","incorporating",
             "investigates","showcasing","underscore","underscores","frameworks",
             "struggle","delve","nuanced","boasts","tapestry","meticulous","garner",
             "elucidate","multifaceted","comprehensively"}
    def kind(w):
        if w in instruments: return "instrument"
        if w in ml_topic: return "ml-method"
        if w in style: return "llm-style"
        return "other"
    df["kind"] = df.word.apply(kind)
    top = df.sort_values("ratio", ascending=False).head(28).iloc[::-1]
    colmap = {"instrument": C["blue"], "llm-style": C["vermillion"],
              "ml-method": C["green"], "other": C["grey"]}
    from pantera_style import no_minor_y
    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    ax.barh(range(len(top)), top.ratio, color=[colmap[k] for k in top.kind])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.word, fontsize=10)
    no_minor_y(ax)
    ax.set_xlim(0, top.ratio.max()*1.28)   # headroom for right-edge value labels
    for i,(_,r) in enumerate(top.iterrows()):
        ax.text(r.ratio+0.2, i, f"{r.base_freq*100:.2f}%→{r.rec_freq*100:.2f}%",
                va="center", fontsize=7.4, color="#444")
    ax.set_xlabel("Frequency ratio: 2024-2026 vs 2018-2021 baseline")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="new telescope/survey (real astronomy)"),
                       Patch(color=C["vermillion"], label="LLM stylistic word"),
                       Patch(color=C["green"], label="ML method term"),
                       Patch(color=C["grey"], label="other")],
              loc="lower right", fontsize=9)
    ax.margins(y=0.01)
    footer(fig); fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(os.path.join(FIGS, "fig3_discovery.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 4 (disclosure)
def fig4_disclosure():
    d = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
    yrs = [int(y) for y in d["years"]]
    ack = [d["ack"][str(y)] for y in yrs]
    full = [d["full"][str(y)] for y in yrs]
    tot = [d["total_astro"][str(y)] for y in yrs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    ax1.bar([y-0.2 for y in yrs], full, width=0.4, color=C["sky"], label="mentions in full text")
    ax1.bar([y+0.2 for y in yrs], ack, width=0.4, color=C["vermillion"], label="in acknowledgments (used it)")
    ax1.set_ylabel("number of papers per year"); ax1.set_xlabel("Year")
    ax1.legend(fontsize=9); ax1.set_xticks(range(2016,2027,2))
    ax1.annotate("exactly zero\nbefore 2023", xy=(2019, 2), xytext=(2016.5, 250),
                 fontsize=8.5, color=C["grey"], arrowprops=dict(arrowstyle="->", color=C["grey"]))
    ax1.annotate("2026 partial", xy=(2026, full[-1]), xytext=(2023.4, 650),
                 fontsize=8, color=C["grey"], arrowprops=dict(arrowstyle="->", color=C["grey"]))
    # fraction
    frac_ack = [100*a/t for a,t in zip(ack,tot)]
    frac_full = [100*f/t for f,t in zip(full,tot)]
    ax2.plot(yrs, frac_full, "-o", color=C["sky"], lw=2.4, label="full text")
    ax2.plot(yrs, frac_ack, "-o", color=C["vermillion"], lw=2.4, label="acknowledgments")
    ax2.set_ylabel("% of astronomy papers"); ax2.set_xlabel("Year")
    ax2.legend(fontsize=9); ax2.set_xticks(range(2016,2027,2))
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    footer(fig); fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(os.path.join(FIGS, "fig4_disclosure.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 5 (the gap + fields)
def fig5_gap():
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    d = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
    # astro-ph estimated lower bound (2025) vs disclosed (2025 full text & ack)
    est = al["by_year"]["2025"]["excess"]*100
    disc_full = 100*d["full"]["2025"]/d["total_astro"]["2025"]
    disc_ack = 100*d["ack"]["2025"]/d["total_astro"]["2025"]

    from pantera_style import no_minor_x, no_minor_y
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw={"width_ratios":[1,1.15]})
    # LEFT: the gap (2025 astronomy)
    labels = ["Estimated\nLLM-touched\n(lower bound)", "Disclosed\n(full text)", "Disclosed\n(acknowledg.)"]
    vals = [est, disc_full, disc_ack]
    cols = [C["vermillion"], C["sky"], C["blue"]]
    bars = axL.bar(labels, vals, color=cols, width=0.62)
    no_minor_x(axL)
    for b,v in zip(bars, vals):
        axL.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.1f}%" if v>=1 else f"{v:.2f}%",
                 ha="center", fontsize=11, fontweight="normal")
    axL.set_ylabel("% of 2025 astronomy papers")
    axL.set_ylim(0, max(vals)*1.25)
    axL.text(1.0, est*0.7, f"~{est/max(disc_full,0.01):.0f}x gap", ha="center",
             color=C["grey"], fontsize=10, fontweight="normal")

    # RIGHT: field comparison (literature) + astro-ph
    fields = ["Computer\nScience", "Biomed\n(PubMed)", "astro-ph\n(this work,\nlower bound)",
              "Math /\nNature"]
    fvals = [17.5, 12.5, est, 6.3]
    fcol = [C["grey"], C["grey"], C["vermillion"], C["grey"]]
    order = np.argsort(fvals)
    fields = [fields[i] for i in order]; fvals=[fvals[i] for i in order]; fcol=[fcol[i] for i in order]
    b2 = axR.barh(range(len(fields)), fvals, color=fcol)
    axR.set_yticks(range(len(fields))); axR.set_yticklabels(fields, fontsize=9.5)
    no_minor_y(axR)
    for i,v in enumerate(fvals):
        axR.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="normal")
    axR.set_xlabel("% of abstracts LLM-modified / LLM-touched")
    axR.set_xlim(0, 21)
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_gap.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- FIG 6 (volume)
def fig6_volume():
    yc = pd.read_csv(os.path.join(DATA, "year_counts.csv"))
    yc = yc[yc.year <= 2025]  # drop partial 2026 for clean trend
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
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
