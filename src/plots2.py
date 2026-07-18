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
    fig, ax = plt.subplots(figsize=(4.8, 2.85))
    ax.axhline(100, color=C["grey"], lw=1, ls=":")
    ax.plot(d.year, d.hype_idx, "-o", color=C["vermillion"], lw=1.5, ms=3,
            label="Hype words (unprecedented, remarkable, striking, ...)")
    ax.plot(d.year, d.hedge_idx, "-s", color=C["blue"], lw=1.3, ms=3.4,
            label="Hedging words (may, might, suggest, likely, ...)")
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 126.5, "ChatGPT", rotation=90, va="top", ha="right", fontsize=9.5, color=C["grey"])
    ax.set_ylabel("frequency, indexed to 2015-19 = 100"); ax.set_xlabel("Year")
    ax.set_ylim(90, 128)
    ax.legend(loc="upper left", fontsize=9)
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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 2.77))
    # LEFT: marker clusters quarterly
    axL.plot(s.yq, s.all_markers*100, color=C["black"], lw=1.4, label="all markers", zorder=5)
    axL.plot(s.yq, s.collapsing*100, "-o", ms=2.8, color=C["vermillion"], lw=1.2,
             label="early tells")
    axL.plot(s.yq, s.rising*100, "-o", ms=2.8, color=C["green"], lw=1.2,
             label="late tells")
    axL.set_xlabel("Year (quarterly)"); axL.set_ylabel("% of abstracts with marker(s)")
    axL.legend(loc="upper left", fontsize=9)
    axL.set_xlim(2022, 2026.6)
    # RIGHT: disclosure still rising -> contradiction
    axR.plot(yrs, frac_full, "-o", color=C["sky"], lw=1.4, ms=3)
    axR.set_xlabel("Year"); axR.set_ylabel("% of astronomy papers disclosing LLM use (full text)")
    axR.set_xticks(range(2016, 2027, 2))
    footer(fig); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_detectability_erosion.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig7_hype_hedge(); print("fig7 done")
    fig8_detectability(); print("fig8 done")
