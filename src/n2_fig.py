#!/usr/bin/env python3
"""The avoidance figure for the Nature Astronomy draft.
(a) Quarterly incidence P(>=1 basket word) and conditional density P(>=2 | >=1)
    with Wilson bands. The density collapses after 2024Q1 while incidence holds.
(b) Tell half-lives: quarters from each word's peak excess to half of it.
    Publicized words decay within quarters; unpublicized ones had not halved by
    the end of 2025 (arrows).
Output: figs/fig_avoidance.png (also copied by hand into paper_nature/figs).
"""
import json, os
import pandas as pd
import matplotlib.pyplot as plt
from pantera_style import C, no_minor_y

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")

def main():
    d = pd.read_csv(os.path.join(DATA, "n2_density.csv"))
    d = d[d.q >= 2022]
    dec = json.load(open(os.path.join(DATA, "n2_avoidance.json")))["decay"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) incidence + conditional density
    axA.plot(d.q, d.p1 * 100, "-o", ms=3, lw=1.3, color=C["blue"], label="Incidence, P($\\geq$1 marker)")
    axA.fill_between(d.q, d.p1_lo * 100, d.p1_hi * 100, color=C["blue"], alpha=0.15, lw=0)
    axA.plot(d.q, d.cond * 100, "-s", ms=3, lw=1.3, color=C["vermillion"],
             label="Density, P($\\geq$2 $|$ $\\geq$1)")
    axA.fill_between(d.q, d.cond_lo * 100, d.cond_hi * 100, color=C["vermillion"], alpha=0.15, lw=0)
    axA.set_xlabel("Year (quarterly)")
    axA.set_ylabel("% of abstracts")
    axA.set_xlim(2022, 2026.0)
    axA.set_ylim(0, 22)
    axA.legend(loc="upper left", fontsize=8.5)
    axA.text(0.03, 0.60, "(a)", transform=axA.transAxes, fontsize=10)

    # (b) half-life lollipop
    order = ["delve", "meticulous", "nuanced", "intricate", "showcasing", "pivotal",
             "underscore", "leveraging", "highlighting", "offering"]
    ys = list(range(len(order)))
    NOTREACHED = 8.5
    for yi, w in enumerate(order):
        rec = dec.get(w)
        if rec is None: continue
        hl = rec["half_life_quarters"]
        if hl is not None:
            axB.plot([0, hl], [yi, yi], lw=0.8, color="#CCCCCC", zorder=1)
            axB.plot(hl, yi, "o", ms=4.5, color=C["vermillion"], zorder=3)
        else:
            axB.plot([0, NOTREACHED - 0.9], [yi, yi], lw=0.8, color="#CCCCCC", zorder=1)
            axB.annotate("", xy=(NOTREACHED, yi), xytext=(NOTREACHED - 0.9, yi),
                         arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.1))
    no_minor_y(axB)
    axB.set_yticks(ys); axB.set_yticklabels(order, fontsize=8.5, family="monospace")
    axB.invert_yaxis()
    axB.set_xlim(0, 9.6)
    axB.set_xticks([0, 2, 4, 6, 8])
    axB.set_xlabel("Quarters from peak to half of peak excess")
    axB.text(0.05, 0.08, "(b)", transform=axB.transAxes, fontsize=10)
    axB.text(8.3, 5.35, "Not reached\nby end-2025", fontsize=8, color=C["blue"],
             ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_avoidance.png"), bbox_inches="tight")
    print("wrote figs/fig_avoidance.png")

if __name__ == "__main__":
    main()
