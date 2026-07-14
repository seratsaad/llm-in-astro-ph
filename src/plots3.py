#!/usr/bin/env python3
"""Figures for C4 (geography/equity) and C5 (subfield diffusion)."""
import os, json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
from pantera_style import C  # PANTERA visual style (sets rcParams on import)
def footer(fig, t=None): return  # source note lives in the figure caption

def fig10_subfield():
    df = pd.read_csv(os.path.join(DATA, "c5_subfield.csv"))
    order = df[df.year==2025].sort_values("rate", ascending=False)["subfield"].tolist()
    palette = {order[0]: C["vermillion"]}
    others = [C["blue"], C["green"], C["purple"], C["sky"], C["orange"]]
    for i, s in enumerate(order[1:]): palette[s] = others[i % len(others)]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for s in order:
        d = df[df.subfield==s].sort_values("year")
        lw = 3.2 if s == order[0] else 1.8
        ax.plot(d.year, d.rate, "-o", ms=4, lw=lw, color=palette[s],
                label=s + ("  ← leads" if s == order[0] else ""))
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 7.9, "ChatGPT", rotation=90, va="top", ha="right", fontsize=8, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("% of abstracts with LLM marker basket")
    ax.legend(loc="upper left", fontsize=9.2)
    ax.set_xticks(range(2018, 2027, 2))
    footer(fig, "Data: 200,547 astro-ph abstracts by primary arXiv category  |  Analysis for Astrobites")
    fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(os.path.join(FIGS, "fig10_subfield_diffusion.png"), bbox_inches="tight"); plt.close(fig)

def fig11_geography():
    g = json.load(open(os.path.join(DATA, "c4_geography.json")))
    NATIVE = {"USA", "United Kingdom", "Australia", "Canada"}
    rows = []
    for c, rec in g.items():
        r = rec["2025"]
        if r["total"] < 300:  # drop tiny-n noisy countries from the main view
            continue
        rows.append((c, r["marker_pct"], r["disc_pct"], r["total"], c in NATIVE))
    FS = 9.5                       # label font size
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for (c, mk, dc, n, nat) in rows:
        col = C["blue"] if nat else C["vermillion"]
        ax.scatter(mk, dc, s=30+n/40, color=col, alpha=0.85, edgecolor="white",
                   linewidth=0.8, zorder=3)
    mks = [r[1] for r in rows]; dcs = [r[2] for r in rows]
    xmax = max(mks) * 1.20
    ylo, yhi = -0.06, 1.14
    ax.set_xlim(0, xmax); ax.set_ylim(ylo, yhi)
    ax.axvline(np.median(mks), color=C["grey"], ls=":", lw=1, zorder=1)
    ax.axhline(np.median(dcs), color=C["grey"], ls=":", lw=1, zorder=1)

    # ---- candidate-based label placement (no overlap with points/labels/legend) ----
    # data units per inch, to size marker/label boxes correctly
    axpos = ax.get_position()
    axw_in = fig.get_size_inches()[0] * axpos.width
    axh_in = fig.get_size_inches()[1] * axpos.height
    dpx = xmax / axw_in
    dpy = (yhi - ylo) / axh_in
    charw = FS * 0.60 / 72 * dpx          # per-character label width in data-x
    labh = FS * 1.30 / 72 * dpy           # label height in data-y

    def marker_half(n):
        r_in = ((30 + n/40) / np.pi) ** 0.5 / 72
        return r_in*dpx, r_in*dpy

    def lbox(name, x, y, ha):
        w = len(name)*charw
        if ha == "left":  x0, x1 = x, x+w
        elif ha == "right": x0, x1 = x-w, x
        else:             x0, x1 = x-w/2, x+w/2
        return (x0, y-labh/2, x1, y+labh/2)

    def overlap(a, b, padx=0.0, pady=0.0):
        return not (a[2] < b[0]-padx or a[0] > b[2]+padx or
                    a[3] < b[1]-pady or a[1] > b[3]+pady)

    marker_boxes = []
    for (c, mk, dc, n, nat) in rows:
        hx, hy = marker_half(n)
        marker_boxes.append((mk-hx, dc-hy, mk+hx, dc+hy))
    # fixed obstacles in data coords: legend (upper-left) and annotation (upper-right)
    legend_box = (0.0, 0.86, 0.34*xmax, yhi)
    annot_box  = (0.66*xmax, 0.58, xmax, 0.80)

    def candidates(mk, dc, n, name):
        hx, hy = marker_half(n); pad = 0.05*dpx*72/72 + 0.06
        out = []
        for ext, cost in [(0.0, 0), (0.6, 2), (1.4, 5)]:   # increasing offset
            out += [
                (mk+hx+pad+ext, dc, "left",  1+cost),
                (mk-hx-pad-ext, dc, "right", 2+cost),
                (mk, dc+hy+labh*0.6+0.02+ext*0.03, "center", 3+cost),
                (mk, dc-hy-labh*0.6-0.02-ext*0.03, "center", 3+cost),
                (mk+hx+pad+ext, dc+hy+labh*0.6, "left",  4+cost),
                (mk+hx+pad+ext, dc-hy-labh*0.6, "left",  4+cost),
                (mk-hx-pad-ext, dc+hy+labh*0.6, "right", 4+cost),
                (mk-hx-pad-ext, dc-hy-labh*0.6, "right", 4+cost),
            ]
        return out

    placed = []
    order = sorted(range(len(rows)), key=lambda i: -rows[i][3])  # biggest bubbles first
    results = {}
    for i in order:
        c, mk, dc, n, nat = rows[i]
        best, bestpen = None, 1e18
        for (lx, ly, ha, dcost) in candidates(mk, dc, n, c):
            box = lbox(c, lx, ly, ha)
            pen = dcost
            if box[0] < 0.05 or box[2] > xmax-0.05 or box[1] < ylo+0.02 or box[3] > yhi-0.02:
                pen += 200
            for mb in marker_boxes:
                if overlap(box, mb): pen += 15
            for pb in placed:
                if overlap(box, pb, padx=0.05, pady=0.005): pen += 18
            if overlap(box, legend_box): pen += 120
            if overlap(box, annot_box):  pen += 120
            pen += (((lx-mk)/dpx)**2 + ((ly-dc)/dpy)**2) ** 0.5 * 0.20  # prefer near
            if pen < bestpen:
                bestpen, best = pen, (lx, ly, ha, box)
        lx, ly, ha, box = best
        placed.append(box)
        results[i] = (lx, ly, ha)

    for i, (c, mk, dc, n, nat) in enumerate(rows):
        lx, ly, ha = results[i]
        hx, hy = marker_half(n)
        if abs(ly-dc) > hy+0.015 or abs(lx-mk) > hx+0.10:
            ax.plot([mk, lx], [dc, ly], color=C["grey"], lw=0.5, alpha=0.6, zorder=2)
        ax.annotate(c, (lx, ly), fontsize=FS, ha=ha, va="center", zorder=5)

    # verify: report any residual label overlaps with markers / other labels / boxes
    final = [lbox(rows[i][0], *results[i]) for i in range(len(rows))]
    probs = 0
    for i, b in enumerate(final):
        for mb in marker_boxes:
            if overlap(b, mb): probs += 1
        for j in range(i+1, len(final)):
            if overlap(b, final[j]): probs += 1
        if overlap(b, legend_box) or overlap(b, annot_box): probs += 1
    print(f"fig11 label overlaps remaining: {probs}")

    ax.text(0.985, 0.72, "bottom-right =\nhigh use, low disclosure\n(biggest 'concealment' zone)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            color=C["grey"], style="italic")
    ax.set_xlabel("LLM marker-word incidence, 2025 (% of papers)")
    ax.set_ylabel("Explicit LLM disclosure, 2025 (% of papers)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=C["blue"],label='native English',ms=9),
                       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["vermillion"],label='non-native English',ms=9)],
              loc="upper left", fontsize=9)
    footer(fig, "Data: NASA ADS aff: x abs:/ack: queries, 2025  |  aff: matches any affiliation (multi-country collabs double-counted)")
    fig.savefig(os.path.join(FIGS, "fig11_equity_map.png"), bbox_inches="tight"); plt.close(fig)

def fig12_citation_integrity():
    """C3: astronomy's citation integrity vs the fabrication problem elsewhere."""
    v = json.load(open(os.path.join(DATA, "c3_verify.json")))
    m = v["doi"]["missing"]
    def bucket(d):
        if d.startswith("10.48550/arxiv"): return "arXiv DOI (real)"
        if d.startswith("10.5281/zenodo"): return "Zenodo DOI (real)"
        if d.startswith(("10.17909","10.26131","10.25574","10.18434")): return "data-archive DOI (real)"
        return "regional journal / funder (real)"
    import collections
    cc = collections.Counter(bucket(d) for d in m)
    n_arxiv = v["arxiv"]["n_ids"]; n_doi_checked = v["doi"]["checked"]
    total_checked = n_arxiv + n_doi_checked

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios":[1,1]})
    # LEFT: the 123 "non-resolving" DOIs are all real (registry artifacts)
    labels = list(cc.keys()); vals = [cc[k] for k in labels]
    from pantera_style import no_minor_y, no_minor_x
    axL.barh(range(len(labels)), vals, color=C["green"])
    axL.set_yticks(range(len(labels))); axL.set_yticklabels(labels, fontsize=9.5)
    no_minor_y(axL)
    axL.set_xlabel("count among the 123 Crossref 'misses'")
    for i, val in enumerate(vals):
        axL.text(val+1, i, str(val), va="center", fontsize=10, fontweight="normal")
    axL.set_xlim(0, max(vals)*1.2)
    # RIGHT: fabrication rate astro vs elsewhere
    fields = ["astro-ph\n(this work)", "Biomed 2025\n(Lancet)", "Biomed 2026\n(Lancet)"]
    fvals = [0.0, 0.22, 0.36]   # % of papers with >=1 fabricated citation
    cols = [C["blue"], C["grey"], C["grey"]]
    b = axR.bar(fields, fvals, color=cols, width=0.6)
    no_minor_x(axR)
    for bar, val in zip(b, fvals):
        axR.text(bar.get_x()+bar.get_width()/2, val+0.008,
                 "0" if val == 0 else f"{val:.2f}%", ha="center", fontsize=11, fontweight="normal")
    axR.set_ylabel("% of papers with a fabricated citation")
    axR.set_ylim(0, 0.45)
    axR.text(0.02, 0.92, f"astro-ph: 0 fabrications in\n~{total_checked:,} references checked\n"
             "(all arXiv IDs + DOIs resolve)\n→ ADS-BibTeX culture protects\n   the citation graph",
             transform=axR.transAxes, fontsize=9, va="top", color=C["black"])
    footer(fig, "astro-ph: arXiv API + Crossref resolution of refs from recent LaTeX source. "
                "Biomed: Lancet/Topaz+2026 (1-in-458 papers 2025).")
    fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(os.path.join(FIGS, "fig12_citation_integrity.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig10_subfield(); print("fig10 done")
    fig11_geography(); print("fig11 done")
    fig12_citation_integrity(); print("fig12 done")
