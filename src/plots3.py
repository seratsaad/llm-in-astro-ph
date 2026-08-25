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
    df = df[df.n > 0]          # drop the empty trailing year bin
    order = df[df.year==2025].sort_values("rate", ascending=False)["subfield"].tolist()
    palette = {order[0]: C["vermillion"]}
    others = [C["blue"], C["green"], C["purple"], C["sky"], C["orange"]]
    for i, s in enumerate(order[1:]): palette[s] = others[i % len(others)]
    fig, ax = plt.subplots(figsize=(6.96, 4.21))
    for s in order:
        d = df[df.subfield==s].sort_values("year")
        lw = 3.2 if s == order[0] else 1.8
        ax.plot(d.year, d.rate, "-o", ms=3.2, lw=lw, color=palette[s],
                label=s)
    ax.axvline(2022.85, color=C["grey"], ls="--", lw=1)
    ax.text(2022.7, 7.9, "ChatGPT", rotation=90, va="top", ha="right", fontsize=9.5, color=C["grey"])
    ax.set_xlabel("Year"); ax.set_ylabel("% of abstracts with LLM marker basket")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(range(2018, 2026, 2))
    footer(fig, "Data: 200,547 astro-ph abstracts by primary arXiv category  |  Analysis for Astrobites")
    fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(os.path.join(FIGS, "fig10_subfield_diffusion.png"), bbox_inches="tight"); plt.close(fig)

NATIVE = {"USA", "United Kingdom", "Australia", "Canada", "Ireland",
          "New Zealand"}
_COUNTRIES = ["USA", "United States", "China", "Germany", "United Kingdom",
    "UK", "Italy", "France", "Japan", "Spain", "India", "Australia", "Canada",
    "Netherlands", "Switzerland", "Brazil", "Korea", "Russia", "Iran",
    "Sweden", "Poland", "Austria", "Belgium", "Mexico", "Chile", "Portugal",
    "Israel", "Taiwan", "Denmark", "Norway", "Finland", "Turkey", "Greece",
    "Czech", "Hungary", "Argentina", "South Africa", "Ireland", "New Zealand",
    "Scotland", "England"]
_ALIAS = {"United States": "USA", "UK": "United Kingdom",
          "England": "United Kingdom", "Scotland": "United Kingdom"}


def _country_of(aff):
    import re
    for c in _COUNTRIES:
        if re.search(r"\b" + re.escape(c) + r"\b", aff, re.I):
            return _ALIAS.get(c, c)
    return None


def _firstauthor_rows(min_n=200):
    """Rows for the first-author equity figure: per-country marker incidence
    (x) against writing-disclosure rate (y), from the enlarged disclosure set
    joined to first-author affiliations. No error bars are used, so tiny
    disclosure numerators no longer dominate the panel."""
    import re
    p13 = json.load(open(os.path.join(DATA, "p13_firstauthor.json")))["country_rows"]
    enl = set(json.load(open(os.path.join(DATA, "p1c_enlarged.json")))["writing_ids_enlarged"])
    tot, disc = {}, {}
    for line in open(os.path.join(DATA, "p13_aff_cache.jsonl")):
        rec = json.loads(line)
        c = _country_of(rec.get("aff0") or "")
        if not c:
            continue
        tot[c] = tot.get(c, 0) + 1
        base = re.sub(r"v\d+$", "", rec["id"])
        if rec["id"] in enl or base in enl:
            disc[c] = disc.get(c, 0) + 1
    rows = []
    for c, r in p13.items():
        if r["n"] < min_n:
            continue
        n = tot.get(c, r["n"])
        rows.append((c, r["pct"], disc.get(c, 0) / n * 100, r["n"], c in NATIVE))
    return rows


def _equity_figure(rows, outfile, ylabel):
    FS = 9.5                       # label font size
    DISPLAY = {"Korea": "South Korea"}
    rows = [(DISPLAY.get(c, c), mk, dc, n, nat) for (c, mk, dc, n, nat) in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.14))
    for (c, mk, dc, n, nat) in rows:
        col = C["blue"] if nat else C["vermillion"]
        ax.scatter(mk, dc, s=30+n/40, color=col, alpha=0.85, edgecolor="white",
                   linewidth=0.8, zorder=3)
    mks = [r[1] for r in rows]; dcs = [r[2] for r in rows]
    xmin = max(min(mks) - 0.8, 0)
    xmax = max(mks) * 1.06
    ymax = max(dcs)
    yhi = ymax * 1.14
    ylo = -0.06 * yhi
    ax.set_xlim(xmin, xmax); ax.set_ylim(ylo, yhi)
    ax.axvline(np.median(mks), color=C["grey"], ls=":", lw=1, zorder=1)
    ax.axhline(np.median(dcs), color=C["grey"], ls=":", lw=1, zorder=1)

    # ---- candidate-based label placement (no overlap with points/labels/legend) ----
    # data units per inch, to size marker/label boxes correctly
    axpos = ax.get_position()
    axw_in = fig.get_size_inches()[0] * axpos.width
    axh_in = fig.get_size_inches()[1] * axpos.height
    dpx = (xmax - xmin) / axw_in
    dpy = (yhi - ylo) / axh_in
    charw = FS * 0.60 / 72 * dpx          # per-character label width in data-x
    labh = FS * 1.30 / 72 * dpy           # label height in data-y

    def marker_half(n, country=None):
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
        hx, hy = marker_half(n, c)
        marker_boxes.append((mk-hx, dc-hy, mk+hx, dc+hy))
    # fixed obstacles in data coords: legend (upper-left) and annotation (upper-right)
    legend_box = (xmin, 0.80*yhi, xmin + 0.34*(xmax-xmin), yhi)   # upper left
    annot_box  = (9e9, 9e9, 9e9, 9e9)   # no in-panel annotation any more

    def candidates(mk, dc, n, name):
        hx, hy = marker_half(n, name); pad = 0.05*dpx*72/72 + 0.06
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
            if box[0] < xmin+0.05 or box[2] > xmax-0.05 or box[1] < ylo+0.02 or box[3] > yhi-0.02:
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
        hx, hy = marker_half(n, c)
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

    ax.set_xlabel("LLM marker-word incidence, 2025 (% of papers)")
    ax.set_ylabel(ylabel)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=C["blue"],label='Native English',ms=9),
                       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["vermillion"],label='Non-native English',ms=9)],
              loc="upper left", fontsize=9)
    fig.savefig(os.path.join(FIGS, outfile), bbox_inches="tight"); plt.close(fig)


def fig11_firstauthor_equity():
    """Figure 4: first-author country marker rate vs writing-disclosure rate,
    no error bars (the disclosure numerators are small)."""
    _equity_figure(_firstauthor_rows(min_n=200), "fig11_equity_map.png",
                   "Stated model use, 2025 (% of papers)")


def fig11_geography():
    """Extended Data: any-affiliation version of the equity map."""
    g = json.load(open(os.path.join(DATA, "c4_geography.json")))
    rows = []
    for c, rec in g.items():
        r = rec["2025"]
        if r["total"] < 300:
            continue
        rows.append((c, r["marker_pct"], r["disc_pct"], r["total"], c in NATIVE))
    _equity_figure(rows, "fig11_equity_anyaff.png",
                   "Explicit LLM disclosure, 2025 (% of papers)")

def fig13_crossref_taxonomy():
    """ED: classification of the 126 Crossref misses (all real works)."""
    st = json.load(open(os.path.join(DATA, "c3_stats25.json")))
    cls = st["classification"]
    order = ["arXiv DataCite (real)", "Zenodo (real)",
             "data archive / regional / funder (real)",
             "other registry, resolves via handle (real)",
             "wrong identifier, real reference", "extraction artifact"]
    disp_name = {"arXiv DataCite (real)": "arXiv DataCite DOI\n(real)",
                 "Zenodo (real)": "Zenodo DOI (real)",
                 "data archive / regional / funder (real)": "data archive / regional\njournal / funder (real)",
                 "other registry, resolves via handle (real)": "other registry,\nresolves (real)",
                 "wrong identifier, real reference": "wrong identifier,\nreal reference",
                 "extraction artifact": "our extraction\nartifact"}
    labels = [disp_name[k] for k in order if cls.get(k)]
    vals = [cls[k] for k in order if cls.get(k)]
    from pantera_style import no_minor_y
    fig, ax = plt.subplots(figsize=(6.96, 3.77))
    ys = list(range(len(labels)))
    ax.barh(ys, vals, color=C["green"], height=0.62)
    for yi, v in zip(ys, vals):
        ax.text(v+2.5, yi, str(v), va="center", fontsize=9, color="#555555")
    no_minor_y(ax)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel(f"count among the {sum(vals)} Crossref 'misses'", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig13_crossref_taxonomy.png"), bbox_inches="tight")
    plt.close(fig)

def fig12_citation_integrity():
    """C3: identifier census of references + fabrication rates per population."""
    st = json.load(open(os.path.join(DATA, "c3_stats25.json")))
    r7 = json.load(open(os.path.join(DATA, "r7_summary.json")))["counts"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios":[1.1,1]})
    # LEFT: identifier census of the 4,964 references in the audited papers
    from pantera_style import no_minor_y, no_minor_x
    census = [("arXiv identifier", r7["with_arxiv"], C["green"]),
              ("DOI only", r7["with_doi_only"], C["blue"]),
              ("no identifier", r7["no_id"], C["vermillion"])]
    tot = sum(v for _, v, _ in census)
    ys = list(range(len(census)))
    for yi, (lab, v, col) in zip(ys, census):
        axL.barh(yi, v, color=col, height=0.58)
        axL.text(v + 130, yi, f"{v:,} ({v/tot*100:.0f}%)", va="center",
                 fontsize=9, color="#555555")
    no_minor_y(axL)
    axL.set_yticks(ys); axL.set_yticklabels([c[0] for c in census], fontsize=9)
    axL.invert_yaxis()
    axL.set_xlim(0, 8600)
    axL.set_xlabel(f"references in the 186 audited papers ({tot:,} total)", fontsize=9)

    # RIGHT: fabrication rate per paper, log scale. Astronomy upper limits
    # (identifier-bearing and unflagged control) set apart from biomedicine.
    # The single flagged detection is a sample of one with a huge Poisson
    # interval, so it is described in the text rather than plotted here.
    cats = ["with IDs\n(95% limit)", "no IDs\ncontrol\n(95% limit)",
            "2025", "2026"]
    xs = [0, 1.2, 3.0, 4.0]
    ul = st["per_paper_UL_pct"]
    axR.axvspan(-0.55, 1.75, color=C["blue"], alpha=0.05, lw=0)
    axR.plot(0, ul, marker="_", ms=9, color=C["blue"], mew=1.4)
    axR.annotate("", xy=(0, ul*0.55), xytext=(0, ul),
                 arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.0))
    # 0 detections in the 77 control papers: one-sided 95% limit 3.0/77
    ulc = 100 * 3.0 / 77
    axR.plot(1.2, ulc, marker="_", ms=9, color=C["blue"], mew=1.4)
    axR.annotate("", xy=(1.2, ulc*0.55), xytext=(1.2, ulc),
                 arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.0))
    axR.plot(3.0, 0.22, "o", ms=5, color=C["grey"])
    axR.plot(4.0, 0.36, "o", ms=5, color=C["grey"])
    axR.set_yscale("log")
    no_minor_x(axR)
    axR.set_xticks(xs); axR.set_xticklabels(cats, fontsize=7.5)
    axR.set_xlim(-0.55, 4.5); axR.set_ylim(0.008, 24)
    axR.text(0.6, 14, "astro-ph", ha="center", fontsize=8.5, color=C["blue"])
    axR.text(3.5, 14, "biomedicine", ha="center", fontsize=8.5, color=C["grey"])
    axR.set_ylabel("% of papers with a fabricated citation", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig12_citation_integrity.png"), bbox_inches="tight")

if __name__ == "__main__":
    fig10_subfield(); print("fig10 done")
    fig11_firstauthor_equity(); print("fig11 (first-author) done")
    fig11_geography(); print("fig11 (any-aff, ED) done")
    fig12_citation_integrity(); print("fig12 done")
