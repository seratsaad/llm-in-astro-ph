#!/usr/bin/env python3
"""Figures for the C-series follow-up analyses (C6, C7). C2 figure added separately."""
import os, json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
from pantera_style import C  # PANTERA visual style (sets rcParams on import)

def footer(fig, txt=None):
    return  # source note lives in the figure caption, not on the figure

# ---- FIG 7: hype vs hedge ----
def fig7_hype_hedge():
    d = pd.read_csv(os.path.join(DATA, "c7_hype_hedge.csv"))
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    basket = {int(y): v["rate"]*100 for y, v in al["by_year"].items()}
    d["basket"] = d.year.map(basket)
    base = d[d.year.between(2015, 2019)]
    hbase, gbase = base.hype_per1k.mean(), base.hedge_per1k.mean()
    d["hype_idx"] = 100 * d.hype_per1k / hbase
    d["hedge_idx"] = 100 * d.hedge_per1k / gbase
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    ax.axhline(100, color=C["grey"], lw=1, ls=":")
    ax.plot(d.year, d.hype_idx, "-o", color=C["vermillion"], lw=2.8, ms=6,
            label="Hype words (unprecedented, remarkable, striking, ...)")
    ax.plot(d.year, d.hedge_idx, "-s", color=C["blue"], lw=2.4, ms=5,
            label="Hedging words (may, might, suggest, likely, ...)")
    ax.axvspan(2022.85, 2026.6, color=C["yellow"], alpha=0.16, zorder=0)
    ax.text(2024.3, 104, "ChatGPT era", color="#8a6d00", fontsize=9.5, style="italic", ha="center")
    ax.set_ylabel("frequency, indexed to 2015-19 = 100"); ax.set_xlabel("Year")
    ax.set_ylim(90, 128)
    ax.legend(loc="upper left", fontsize=9.4)
    ax.set_xticks(range(2015, 2027, 2))
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_hype_hedge.png"), bbox_inches="tight"); plt.close(fig)

# ---- FIG 8: detectability erosion (C6) ----
def fig8_detectability():
    s = pd.read_csv(os.path.join(DATA, "c6_cluster_series.csv"))
    ads = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
    yrs = [int(y) for y in ads["years"]]
    full = [ads["full"][str(y)] for y in yrs]
    tot = [ads["total_astro"][str(y)] for y in yrs]
    # annualize 2026 (partial ~ half year): scale full-text fraction is a rate so fine
    frac_full = [100*f/t for f, t in zip(full, tot)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 5.2))
    # LEFT: marker clusters quarterly
    axL.plot(s.yq, s.all_markers*100, color=C["black"], lw=3, label="all markers", zorder=5)
    axL.plot(s.yq, s.collapsing*100, "-o", ms=3.5, color=C["vermillion"], lw=1.9,
             label="early tells (delve, intricate, pivotal ...)")
    axL.plot(s.yq, s.rising*100, "-o", ms=3.5, color=C["green"], lw=1.9,
             label="late tells (underscore, leveraging, notably ...)")
    axL.axvspan(2025.75, 2026.5, color=C["grey"], alpha=0.10)
    axL.text(2025.55, axL.get_ylim()[1]*0.12, "2026:\nboth fall", fontsize=9, color=C["grey"])
    axL.set_xlabel("Year (quarterly)"); axL.set_ylabel("% of abstracts with marker(s)")
    axL.legend(loc="upper left", fontsize=8.8)
    axL.set_xlim(2022, 2026.6)
    # RIGHT: disclosure still rising -> contradiction
    axR.plot(yrs, frac_full, "-o", color=C["sky"], lw=2.6, ms=6)
    axR.axvspan(2025.5, 2026.5, color=C["grey"], alpha=0.10)
    axR.set_xlabel("Year"); axR.set_ylabel("% of astronomy papers disclosing LLM use (full text)")
    axR.set_xticks(range(2016, 2027, 2))
    axR.annotate("usage up while word-signal down\n= detection going underground",
                 xy=(2026, frac_full[-1]), xytext=(2019.4, frac_full[-1]*0.62),
                 fontsize=9.5, color=C["black"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["black"]))
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_detectability_erosion.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig7_hype_hedge(); print("fig7 done")
    fig8_detectability(); print("fig8 done")
