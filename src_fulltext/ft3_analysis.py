#!/usr/bin/env python3
"""
FT3 -- analysis + figure for the full-text extension.

Reads data/ft_series.json (half-yearly aggregates over the whole corpus) and
data/ft_papers_2015plus.jsonl.gz (per-paper rows, month precision), and makes
  (a) the two-surface series, each normalized to its own 2018-2021 baseline
  (b) the change from the 2025H1 peak to 2026H1 per series, with bootstrap
      intervals: abstract, body, body named subset, body unnamed subset, and
      the neutral control words
Output: data/ft_results.json, figs/fig14_fulltext.png
"""
import gzip, json, os, collections
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pantera_style import C

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")

def halfval(k):
    if k.endswith("H1"): return int(k[:4]) + 0.0
    if k.endswith("H2"): return int(k[:4]) + 0.5
    return int(k)

def main():
    agg = json.load(open(os.path.join(DATA, "ft_series.json")))
    keys = sorted((k for k in agg if "H" in k), key=halfval)

    # ---- series ----
    xs, abs_pct, body_dens, named_dens, unnamed_dens, ctrl_dens = [], [], [], [], [], []
    for k in keys:
        d = agg[k]
        if d["n_abs"] < 300:
            continue
        xs.append(halfval(k))
        abs_pct.append(d["abs_hit"] / d["n_abs"] * 100)
        body_dens.append(d["bt"] / d["body_words"] * 1e4)
        named_dens.append(d["bt_named"] / d["body_words"] * 1e4)
        unnamed_dens.append((d["bt"] - d["bt_named"]) / d["body_words"] * 1e4)
        ctrl_dens.append(d["bt_ctrl"] / d["body_words"] * 1e4)
    xs = np.array(xs)

    def base(series):
        m = (xs >= 2018) & (xs < 2022)
        return np.mean(np.array(series)[m])

    b_abs, b_body, b_ctrl = base(abs_pct), base(body_dens), base(ctrl_dens)
    b_named, b_unnamed = base(named_dens), base(unnamed_dens)

    # ---- per-paper rows for the peak-to-2026H1 bootstrap ----
    rows = {"2025H1": [], "2026H1": []}
    with gzip.open(os.path.join(DATA, "ft_papers_2015plus.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            y, m = int(r["ym"][:4]), int(r["ym"][5:7])
            per = f"{y}H1" if m <= 6 else f"{y}H2"
            if per in rows and r["bw"] > 500:
                rows[per].append(r)
    A, B = rows["2025H1"], rows["2026H1"]
    print(f"bootstrap rows: 2025H1 {len(A):,}  2026H1 {len(B):,}")

    def change(A, B, num, den):
        a = sum(num(r) for r in A) / sum(den(r) for r in A)
        b = sum(num(r) for r in B) / sum(den(r) for r in B)
        return b / a - 1

    series = {
        "abstract": (lambda r: (r["abs_hit"] or 0), lambda r: 1 if r["abs_hit"] is not None else 0),
        "body_all": (lambda r: r["bt"],            lambda r: r["bw"]),
        "body_named": (lambda r: r["btn"],         lambda r: r["bw"]),
        "body_unnamed": (lambda r: r["bt"] - r["btn"], lambda r: r["bw"]),
        "controls": (lambda r: r["btc"],           lambda r: r["bw"]),
    }
    rng = np.random.default_rng(11)
    res = {}
    NB = 1500
    ia_all = rng.integers(0, len(A), (NB, len(A)))
    ib_all = rng.integers(0, len(B), (NB, len(B)))
    for name, (num, den) in series.items():
        obs = change(A, B, num, den)
        boots = []
        numA = np.array([num(r) for r in A], float); denA = np.array([den(r) for r in A], float)
        numB = np.array([num(r) for r in B], float); denB = np.array([den(r) for r in B], float)
        for i in range(NB):
            a = numA[ia_all[i]].sum() / max(denA[ia_all[i]].sum(), 1)
            b = numB[ib_all[i]].sum() / max(denB[ib_all[i]].sum(), 1)
            boots.append(b / a - 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[name] = {"change_pct": obs * 100, "ci95": [lo * 100, hi * 100]}
        print(f"{name:14s} {obs*100:+7.1f}%  [{lo*100:+.1f}, {hi*100:+.1f}]")

    # differential with its own bootstrap
    diffs = []
    numA1 = np.array([(r["abs_hit"] or 0) for r in A], float)
    denA1 = np.array([1 if r["abs_hit"] is not None else 0 for r in A], float)
    numB1 = np.array([(r["abs_hit"] or 0) for r in B], float)
    denB1 = np.array([1 if r["abs_hit"] is not None else 0 for r in B], float)
    numA2 = np.array([r["bt"] for r in A], float); denA2 = np.array([r["bw"] for r in A], float)
    numB2 = np.array([r["bt"] for r in B], float); denB2 = np.array([r["bw"] for r in B], float)
    for i in range(NB):
        ia, ib = ia_all[i], ib_all[i]
        da = numB1[ib].sum() / max(denB1[ib].sum(), 1) / (numA1[ia].sum() / max(denA1[ia].sum(), 1)) - 1
        db = numB2[ib].sum() / denB2[ib].sum() / (numA2[ia].sum() / denA2[ia].sum()) - 1
        diffs.append(da - db)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res["differential_abs_minus_body"] = {
        "pp": (res["abstract"]["change_pct"] - res["body_all"]["change_pct"]),
        "ci95": [lo * 100, hi * 100],
        "p_ge_0": float((np.array(diffs) >= 0).mean())}
    print("differential:", res["differential_abs_minus_body"])

    json.dump({"baselines": {"abs_pct": b_abs, "body_per10k": b_body,
                             "named_per10k": b_named, "unnamed_per10k": b_unnamed,
                             "ctrl_per10k": b_ctrl},
               "series": {"x": xs.tolist(), "abs_pct": abs_pct,
                          "body_dens": body_dens, "named_dens": named_dens,
                          "unnamed_dens": unnamed_dens, "ctrl_dens": ctrl_dens},
               "peak_to_2026H1": res},
              open(os.path.join(DATA, "ft_results.json"), "w"), indent=1)

    # ------------------------------- figure -------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.96, 3.1),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    axA.plot(xs, np.array(abs_pct) / b_abs, "-o", ms=3.4, lw=1.4,
             color=C["blue"], label="Abstract incidence")
    axA.plot(xs, np.array(body_dens) / b_body, "-s", ms=3.2, lw=1.4,
             color=C["vermillion"], label="Body density")
    axA.plot(xs, np.array(ctrl_dens) / b_ctrl, "-", lw=1.1,
             color=C["grey"], label="Control words (body)")
    axA.axhline(1, color="#CCCCCC", lw=0.7)
    axA.axvline(2022.85, color=C["grey"], ls="--", lw=0.9)
    axA.text(2022.72, axA.get_ylim()[1] * 0.7, "ChatGPT", rotation=90,
             va="top", ha="right", fontsize=8, color=C["grey"])
    axA.set_xlabel("Year (half-yearly)")
    axA.set_ylabel("Excess over each surface's\n2018--2021 baseline")
    axA.legend(loc="upper left", fontsize=8)
    axA.text(0.97, 0.05, "(a)", transform=axA.transAxes, ha="right", fontsize=10)

    labels = ["Abstract", "Body,\nall words", "Body,\nnamed", "Body,\nunnamed",
              "Body,\ncontrols"]
    keys2 = ["abstract", "body_all", "body_named", "body_unnamed", "controls"]
    xpos = np.arange(len(keys2))
    for i, k in enumerate(keys2):
        v = res[k]
        col = C["blue"] if k == "abstract" else (C["grey"] if k == "controls" else C["vermillion"])
        axB.errorbar(i, v["change_pct"],
                     yerr=[[v["change_pct"] - v["ci95"][0]], [v["ci95"][1] - v["change_pct"]]],
                     fmt="o", ms=5, color=col, elinewidth=1.1, capsize=3)
    axB.axhline(0, color="#888888", lw=0.9, ls="--")
    axB.set_xticks(xpos); axB.set_xticklabels(labels, fontsize=7.5)
    axB.set_ylabel("Change from 2025H1\nto 2026H1 (%)")
    axB.text(0.97, 0.05, "(b)", transform=axB.transAxes, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig14_fulltext.png"), bbox_inches="tight")
    print("wrote figs/fig14_fulltext.png")

if __name__ == "__main__":
    main()
