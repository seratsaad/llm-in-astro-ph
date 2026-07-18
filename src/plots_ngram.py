#!/usr/bin/env python3
"""Publication-style figures for the n-gram and co-occurrence analysis."""
import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pantera_style import C, no_minor_y

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
# sober tweaks on top of the shared style
mpl.rcParams.update({"axes.titleweight": "normal"})

def fig_ngram_viewer():
    w = pd.read_csv(os.path.join(DATA, "ngram_watch.csv"))
    show = [("underscores the", C["vermillion"]), ("highlighting the", C["orange"]),
            ("leveraging the", C["green"]), ("pivotal role", C["purple"]),
            ("delve into", C["black"]), ("wide range", C["blue"])]
    fig, ax = plt.subplots(figsize=(4.20, 2.64))
    for p, col in show:
        s = w[w.phrase == p].sort_values("year")
        ls = "--" if p == "wide range" else "-"
        ax.plot(s.year, s.freq, ls, color=col, lw=1.3, ms=3, marker="o",
                label=f'"{p}"' + ("  (control)" if p == "wide range" else ""))
    ax.axvline(2022.85, color=C["grey"], ls=":", lw=1)
    ax.set_ylim(-0.1, 2.8)
    ax.text(2022.7, 1.55, "ChatGPT", rotation=90, va="top",
            ha="right", fontsize=7.5, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("fraction of abstracts with the phrase (%)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.03, 0.40))
    ax.set_xticks(range(2015, 2026, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_ngram_viewer.png"), bbox_inches="tight"); plt.close(fig)

def fig_cooccur():
    import math
    d = np.load(os.path.join(DATA, "cooccur.npz"), allow_pickle=True)
    lift = d["lift"].astype(float); M = list(d["markers"])
    pairs = d["pairs"].astype(int); single = d["single"]; nd = int(d["rec_docs"])
    n = len(M)

    def pois_tail(x, mu):
        """P(X >= x) for X ~ Poisson(mu)."""
        if x == 0:
            return 1.0
        s, term = 0.0, math.exp(-mu)
        for k in range(x):
            s += term
            term *= mu / (k + 1)
        return max(0.0, 1 - s)

    # significance mask: keep cells where the joint count exceeds independence at p < 0.01
    sig = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mu = nd * (single[i]/nd) * (single[j]/nd)
            if pois_tail(int(pairs[i, j]), mu) < 0.01:
                sig[i, j] = True

    order = np.argsort(-single)
    lift = lift[np.ix_(order, order)]; sig = sig[np.ix_(order, order)]
    pairs = pairs[np.ix_(order, order)]; M = [M[i] for i in order]
    disp = np.where(sig, lift, np.nan)
    fig, ax = plt.subplots(figsize=(4.44, 3.72))
    ax.set_facecolor("#EDEDED")   # grey background = not significant / no excess
    im = ax.imshow(np.ma.masked_invalid(disp), cmap="YlOrBr",
                   norm=LogNorm(vmin=1, vmax=np.nanmax(disp)))
    ax.set_xticks(range(n)); ax.set_xticklabels(M, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(M, fontsize=8.5)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(disp[i, j]):
                v = disp[i, j]
                ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if v > 30 else "#333")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("co-occurrence lift (1 = independent)", fontsize=7.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cooccur.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_ngram_viewer(); print("ngram viewer done")
    fig_cooccur(); print("cooccur done")
