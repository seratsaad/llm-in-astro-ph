#!/usr/bin/env python3
"""Publication-style figures for the n-gram and co-occurrence analysis."""
import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pantera_style import C, no_minor_y

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
# sober tweaks on top of the shared style
mpl.rcParams.update({"axes.titleweight": "normal", "font.size": 11})

def fig_ngram_viewer():
    w = pd.read_csv(os.path.join(DATA, "ngram_watch.csv"))
    show = [("underscores the", C["vermillion"]), ("highlighting the", C["orange"]),
            ("leveraging the", C["green"]), ("pivotal role", C["purple"]),
            ("delve into", C["black"]), ("wide range", C["blue"])]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for p, col in show:
        s = w[w.phrase == p].sort_values("year")
        ls = "--" if p == "wide range" else "-"
        ax.plot(s.year, s.freq, ls, color=col, lw=1.6, ms=3, marker="o",
                label=f'"{p}"' + ("  (control)" if p == "wide range" else ""))
    ax.axvline(2022.85, color=C["grey"], ls=":", lw=1)
    ax.text(2022.7, ax.get_ylim()[1]*0.9, "ChatGPT", rotation=90, va="top",
            ha="right", fontsize=8, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("fraction of abstracts with the phrase (%)")
    ax.set_ylim(-0.1, 2.75)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.02, 0.62))
    ax.set_xticks(range(2015, 2026, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_ngram_viewer.png"), bbox_inches="tight"); plt.close(fig)

def fig_bigram_discovery():
    bd = pd.read_csv(os.path.join(DATA, "bigram_discovery.csv"))
    bd = bd[bd.base_freq > 0.0005].copy()
    topic_kw = ("jwst", "desi", "kagra", "virgo", "webb", "spectroscopic", "instrument",
                "habitable", "worlds", "noon", "nircam", "miri", "nirspec", "xrism",
                "language", "transformer", "simulation", "cosmos", "euclid", "lrds",
                "little", "red", "dots")
    def is_topic(bg):
        return any(k in bg.split() for k in topic_kw)
    bd["topic"] = bd.bigram.apply(is_topic)
    top = bd.sort_values("ratio", ascending=False).head(24).iloc[::-1]
    cols = [C["blue"] if t else C["vermillion"] for t in top.topic]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.barh(range(len(top)), top.ratio, color=cols, alpha=0.85, height=0.72)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.bigram, fontsize=8.5)
    no_minor_y(ax)
    ax.set_xlabel("frequency ratio, 2024-2026 vs 2018-2021")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="new topic (instrument / survey / object)"),
                       Patch(color=C["vermillion"], label="stylistic phrase")],
              loc="lower right", fontsize=8.5)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_bigram_discovery.png"), bbox_inches="tight"); plt.close(fig)

def fig_cooccur():
    d = np.load(os.path.join(DATA, "cooccur.npz"), allow_pickle=True)
    lift = d["lift"].astype(float); M = list(d["markers"])
    n = len(M)
    # order markers by total single frequency for a cleaner block structure
    single = d["single"]; order = np.argsort(-single)
    lift = lift[np.ix_(order, order)]; M = [M[i] for i in order]
    disp = lift.copy()
    np.fill_diagonal(disp, np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
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
    cb.set_label("co-occurrence lift (1 = independent)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cooccur.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_ngram_viewer(); print("ngram viewer done")
    fig_bigram_discovery(); print("bigram discovery done")
    fig_cooccur(); print("cooccur done")
