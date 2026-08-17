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

    fig = plt.figure(figsize=(6.9, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.02], height_ratios=[1, 0.95],
                          hspace=0.42, wspace=0.32)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # (a) incidence + conditional density
    axA.plot(d.q, d.p1 * 100, "-o", ms=3, lw=1.3, color=C["blue"], label="Incidence, P($\\geq$1 marker)")
    axA.fill_between(d.q, d.p1_lo * 100, d.p1_hi * 100, color=C["blue"], alpha=0.15, lw=0)
    axA.plot(d.q, d.cond * 100, "-s", ms=3, lw=1.3, color=C["vermillion"],
             label="Density, P($\\geq$2 $|$ $\\geq$1)")
    axA.fill_between(d.q, d.cond_lo * 100, d.cond_hi * 100, color=C["vermillion"], alpha=0.15, lw=0)
    axA.set_xlabel("Year (quarterly)")
    axA.set_ylabel("% of abstracts")
    axA.set_xlim(2022, 2026.5)
    axA.set_ylim(0, 22)
    axA.legend(loc="upper left", fontsize=8.5)
    axA.text(0.96, 0.95, "(a)", transform=axA.transAxes, fontsize=10, ha="right", va="top")

    # (b) frequency-matched pairs as horizontal dumbbells, one row per pair.
    # Each named word (red) is joined to its matched unnamed word (blue); the
    # two decays overlap heavily, which is the null the text reports.
    p4 = json.load(open(os.path.join(DATA, "p4_freqmatched.json")))
    pairs = sorted(p4["C_matched"]["pairs"], key=lambda pr: pr["decay_named"])
    n = len(pairs)
    for i, pr in enumerate(pairs):
        xn, xu = pr["decay_named"], pr["decay_unnamed"]
        axB.plot([xn, xu], [i, i], "-", color="#CCCCCC", lw=1.0, zorder=1)
        axB.plot(xn, i, "o", ms=4.3, color=C["vermillion"], zorder=3)
        axB.plot(xu, i, "o", ms=4.3, color=C["blue"], zorder=3)
        axB.text(-7, i, pr["named"], fontsize=6.3, ha="right", va="center",
                 color=C["vermillion"])
        axB.text(147, i, pr["unnamed"], fontsize=6.3, ha="left", va="center",
                 color=C["blue"])
    axB.set_xlim(-72, 212)
    axB.set_ylim(-0.8, n + 1.1)
    axB.set_yticks([])
    axB.set_xticks([0, 50, 100])
    axB.spines["bottom"].set_bounds(0, 135)   # axis line under the data only
    axB.set_xlabel("Decay from peak by mid-2026 (%)", fontsize=8.5, x=0.42)
    axB.text(-38, n + 0.25, "Named, 2024", fontsize=7.6, ha="center",
             va="bottom", color=C["vermillion"])
    axB.text(178, n + 0.25, "Matched unnamed", fontsize=7.6, ha="center",
             va="bottom", color=C["blue"])
    axB.spines["left"].set_visible(False)
    axB.text(60, n + 0.55, "(b)", fontsize=10, ha="center", va="bottom")
    # (c) timing test: decay follows publicity, not model releases
    import numpy as np
    wq = pd.read_csv(os.path.join(DATA, "n2_words.csv"))
    RELEASES = [(2023.20, "GPT-4"), (2023.85, "GPT-4 Turbo"), (2024.17, "Claude 3"),
                (2024.37, "GPT-4o"), (2024.70, "o1"), (2025.60, "GPT-5")]
    PUB = (2024.25, 2024.50)   # documented: Liang 1 Apr, Graham 7 Apr, Kobak 11 Jun 2024
    show = [("delve", C["vermillion"], "-"), ("intricate", C["vermillion"], "--"),
            ("pivotal", C["vermillion"], ":"),
            ("leveraging", C["blue"], "-"), ("offering", C["blue"], "--")]
    axC.axvspan(*PUB, color=C["vermillion"], alpha=0.10, lw=0)
    axC.set_xlim(2022.5, 2026.5)
    axC.set_ylim(-0.15, 1.65)
    rel_texts = [axC.text(x, 1.62, name, rotation=90, ha="center", va="top",
                          fontsize=7, color="#777777") for x, name in RELEASES]
    fig.canvas.draw()
    inv = axC.transData.inverted()
    for (x, name), txt in zip(RELEASES, rel_texts):
        bb = txt.get_window_extent()
        label_bottom = inv.transform((bb.x0, bb.y0))[1] - 0.035
        axC.plot([x, x], [-0.15, label_bottom], color="#999999",
                 ls=(0, (2, 2)), lw=0.7, zorder=1)
    for w, col, ls in show:
        d = wq[wq.word == w].sort_values("q")
        pre = d[d.q < 2022]
        coef = np.polyfit(pre.q, pre.freq, 1)
        exc = d[d.q >= 2022.5].copy()
        exc["e"] = exc.freq - np.polyval(coef, exc.q)
        pk = exc.e.max()
        axC.plot(exc.q, exc.e / pk, ls, color=col, lw=1.3, label=w)
    axC.axhline(0, color="#CCCCCC", lw=0.6)
    axC.set_xlim(2022.5, 2026.5)
    axC.set_ylim(-0.15, 1.65)
    axC.set_xticks(range(2023, 2027))
    axC.set_xlabel("Year (quarterly)")
    axC.set_ylabel("Excess over trend, peak = 1")
    axC.legend(loc="upper left", fontsize=8, ncol=1, bbox_to_anchor=(0.005, 0.99))
    axC.text(2024.335, -0.05, "Publicity", fontsize=7.5, color=C["vermillion"],
             ha="center", va="bottom")
    axC.text(0.985, 0.06, "(c)", transform=axC.transAxes, fontsize=10, ha="right", va="bottom")
    fig.savefig(os.path.join(FIGS, "fig_avoidance.png"), bbox_inches="tight")
    print("wrote figs/fig_avoidance.png")

if __name__ == "__main__":
    main()
