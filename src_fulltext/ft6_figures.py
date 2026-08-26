#!/usr/bin/env python3
"""FT6 -- the figure set for the full-text paper."""
import gzip, json, os, sys, collections
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pantera_style import C

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")

wq = json.load(open(os.path.join(DATA, "ft_word_quarters.json")))
counts, wt, basket, control = wq["counts"], wq["body_words"], wq["basket"], wq["control"]

def qv(k):
    return (int(k.split(".")[0]) + int(k.split(".")[1]) / 100) if "." in k else int(k)

QK = sorted((k for k in counts if "." in k), key=qv)
YK = sorted((k for k in counts if "." not in k), key=int)

def series(words, keys):
    x = np.array([qv(k) for k in keys])
    y = np.array([sum(counts[k].get(w, 0) for w in words) / wt[k] * 1e4 for k in keys])
    return x, y

# ---------------------------------------------------------------- fig21 rise
def fig21():
    fig = plt.figure(figsize=(6.96, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.52, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    keys = YK + QK
    x, y = series(basket, keys)
    xc, yc = series(control, keys)
    ax.plot(x, y, "-o", ms=2.8, lw=1.3, color=C["vermillion"],
            label="LLM marker basket (38 words)")
    ax.plot(xc, yc / 100, "-", lw=1.1, color=C["blue"],
            label="Neutral control words ($\\times$1/100)")
    pre = (x >= 2012) & (x < 2022)
    coef = np.polyfit(x[pre], y[pre], 1)
    xx = np.array([2022, 2026.4])
    ax.plot(xx, np.polyval(coef, xx), ls=(0, (3, 2)), lw=0.9, color=C["black"])
    ax.text(2026.3, np.polyval(coef, 2025.6) - 0.06, "pre-2022 trend",
            fontsize=8, ha="right", va="top", color=C["black"])
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 1.12, "ChatGPT", rotation=90, va="top", ha="right",
            fontsize=8.5, color=C["grey"])
    ax.set_xlim(2011.6, 2026.7); ax.set_ylim(0, 1.25)
    ax.set_xticks(range(2012, 2027, 2))
    ax.set_ylabel("Marker tokens per\n10{,}000 body words")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlabel("Year (quarterly from 2015, yearly before)")

    show = ["leveraging", "underscore", "intricate", "delve"]
    for i, w in enumerate(show):
        axw = fig.add_subplot(gs[1, i])
        xw, yw = series([w], YK + QK)
        m = xw >= 2019
        axw.plot(xw[m], yw[m], "-o", ms=2.0, lw=1.0, color=C["blue"])
        prew = (xw >= 2012) & (xw < 2022)
        cw = np.polyfit(xw[prew], yw[prew], 1)
        xx = np.array([2019, 2026.4])
        axw.plot(xx, np.clip(np.polyval(cw, xx), 0, None), ls=(0, (3, 2)),
                 lw=0.8, color=C["black"])
        axw.axvline(2024.35, color=C["grey"], ls=":", lw=0.8)
        axw.set_xlim(2019, 2026.6); axw.set_ylim(0, max(yw[m]) * 1.2)
        axw.set_xticks([2020, 2023, 2026])
        axw.tick_params(labelsize=8)
        axw.text(0.06, 0.93, w, transform=axw.transAxes, va="top", fontsize=9)
        if i == 0:
            axw.set_ylabel("per 10k words", fontsize=8.5)
    fig.savefig(os.path.join(FIGS, "fig21_rise.png"), bbox_inches="tight")
    plt.close(fig); print("fig21 done")

# ------------------------------------------------- fig22 calibration ladder
def fig22():
    cal = json.load(open(os.path.join(DATA, "ft_calibration.json")))
    fig, ax = plt.subplots(figsize=(6.96, 2.7))
    rows = [
        ("2018--2021 baseline",              cal["D0"],       C["grey"]),
        ("Coding-only disclosures (n=29)",   cal["Dcode"],    C["blue"]),
        ("Full 2025 corpus",                 cal["D25"],      C["vermillion"]),
        ("Writing disclosures (n=186)",      cal["Dq"],       C["green"]),
        ("LLM-research papers (n=92)",       cal["Dresearch"], C["grey"]),
    ]
    ys = np.arange(len(rows))[::-1]
    for (lab, v, col), yy in zip(rows, ys):
        ax.plot([0, v], [yy, yy], "-", color="#DDDDDD", lw=1.0, zorder=1)
        ax.plot(v, yy, "o", ms=6.5, color=col, zorder=3)
        ax.text(v + 0.09, yy, f"{v:.2f}", va="center", fontsize=8.5, color="#555555")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 5.1)
    ax.set_xlabel("Marker tokens per 10{,}000 body words, full text")
    # alpha geometry
    ax.annotate("", xy=(cal["D25"], 1.55), xytext=(cal["D0"], 1.55),
                arrowprops=dict(arrowstyle="<->", color=C["vermillion"], lw=1.0))
    ax.text((cal["D0"] + cal["D25"]) / 2, 1.72, "excess", fontsize=8,
            color=C["vermillion"], ha="center")
    ax.annotate("", xy=(cal["Dq"], 0.45), xytext=(cal["D0"], 0.45),
                arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1.0))
    ax.text((cal["D0"] + cal["Dq"]) / 2, 0.62, "disclosed excess", fontsize=8,
            color=C["green"], ha="center")
    fig.savefig(os.path.join(FIGS, "fig22_calibration.png"), bbox_inches="tight")
    plt.close(fig); print("fig22 done")

# ---------------------------------------------------------- fig23 erasure
def fig23():
    wt5 = json.load(open(os.path.join(DATA, "ft_word_tests.json")))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.96, 3.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) zoomed turnover + concentration
    x, y = series(basket, QK)
    m = x >= 2022
    axA.plot(x[m], y[m], "-o", ms=3.2, lw=1.4, color=C["vermillion"],
             label="Basket density")
    # concentration from per-paper rows
    per = collections.defaultdict(lambda: [0, 0])
    with gzip.open(os.path.join(DATA, "ft_papers_2015plus.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["bw"] <= 500 or r["bt"] == 0: continue
            q = int(r["ym"][:4]) + (int(r["ym"][5:7]) - 1) // 3 * 0.25
            per[q][0] += r["bt"]; per[q][1] += 1
    qs = sorted(q for q in per if q >= 2022)
    conc = [per[q][0] / per[q][1] for q in qs]
    ax2 = axA.twinx()
    ax2.plot(qs, conc, "-s", ms=2.8, lw=1.2, color=C["blue"])
    ax2.set_ylabel("Marker tokens per\ncarrying paper", fontsize=8.5, color=C["blue"])
    ax2.tick_params(axis="y", labelcolor=C["blue"], labelsize=8)
    axA.axvspan(2024.25, 2024.50, color=C["vermillion"], alpha=0.10, lw=0)
    axA.text(2024.375, 0.15, "publicity", fontsize=7.5, ha="center",
             color=C["vermillion"])
    axA.set_xlabel("Year (quarterly)")
    axA.set_ylabel("Marker tokens per 10{,}000\nbody words", color=C["vermillion"])
    axA.tick_params(axis="y", labelcolor=C["vermillion"])
    axA.set_xlim(2022, 2026.6)
    axA.text(0.03, 0.95, "(a)", transform=axA.transAxes, fontsize=10, va="top")

    # (b) placebo strip: named changes vs central-threshold crossers
    pl = wt5["placebo"]
    named = list(pl["named_changes"].values())
    thr_keys = sorted(pl["sensitivity"], key=float)
    central = pl["sensitivity"][thr_keys[1]]
    cross = list(central["unnamed_words"].values())
    rng = np.random.default_rng(2)
    for i, (vals, col, xc) in enumerate([(named, C["vermillion"], 0),
                                         (cross, C["blue"], 1)]):
        jit = np.linspace(-0.15, 0.15, len(vals))
        axB.scatter(xc + jit, vals, s=28, color=col, alpha=0.9,
                    edgecolor="white", linewidth=0.6, zorder=3)
        mmean = np.mean(vals)
        axB.text(xc + (0.30 if xc else -0.30), mmean, f"{mmean:+.0f}%",
                 fontsize=8.5, color=col, fontweight="bold",
                 ha="left" if xc else "right", va="center")
    axB.axhline(0, color="#888888", lw=0.9, ls="--")
    axB.set_xlim(-0.7, 1.7)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["Named\nas a tell", "Matched,\nnever named"], fontsize=8.5)
    axB.set_ylabel("Change over the next\ntwo quarters (%)", fontsize=8.5)
    axB.text(0.5, 0.95, f"$p={central['p']:.3f}$", transform=axB.transAxes,
             fontsize=8.5, ha="center", va="top")
    axB.text(0.96, 0.05, "(b)", transform=axB.transAxes, fontsize=10, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig23_erasure.png"), bbox_inches="tight")
    plt.close(fig); print("fig23 done")

# ---------------------------------------------------------- fig25 equity
def fig25():
    cal = json.load(open(os.path.join(DATA, "ft_calibration.json")))
    NATIVE = {"USA", "United Kingdom", "Australia", "Canada", "Ireland", "New Zealand"}
    DISPLAY = {"Korea": "South Korea"}
    rows = sorted(cal["equity_2025"].items(), key=lambda kv: kv[1]["dens"])
    fig, ax = plt.subplots(figsize=(6.96, 3.6))
    ys = np.arange(len(rows))
    for yy, (c, v) in zip(ys, rows):
        col = C["blue"] if c in NATIVE else C["vermillion"]
        ax.plot([0, v["dens"]], [yy, yy], "-", color="#E5E5E5", lw=1.0, zorder=1)
        ax.plot(v["dens"], yy, "o", ms=5.5, color=col, zorder=3)
        ax.text(v["dens"] + 0.03, yy, f"  {v['n']:,}", va="center", fontsize=7,
                color="#999999")
    ax.set_yticks(ys)
    ax.set_yticklabels([DISPLAY.get(c, c) for c, _ in rows], fontsize=8.5)
    ax.set_xlabel("Marker tokens per 10{,}000 body words, 2025")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C["blue"],
               label='Native English', ms=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C["vermillion"],
               label='Non-native English', ms=8)], loc="lower right", fontsize=8.5)
    ax.set_xlim(0, 1.75)
    fig.savefig(os.path.join(FIGS, "fig25_equity.png"), bbox_inches="tight")
    plt.close(fig); print("fig25 done")

if __name__ == "__main__":
    fig21(); fig22(); fig23(); fig25()
