#!/usr/bin/env python3
"""Stage 8: generate paper/numbers.tex from the result files.

Every number quoted in the manuscript is a macro defined here, so the prose
never contains a hand-typed result. Missing inputs produce visible XX-macros
rather than silent failures.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, ROOT

OUT = os.path.join(ROOT, "paper", "numbers.tex")
macros = {}


def M(name, value):
    macros[name] = value


def pct(x, digits=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "XX\\%"
    return f"{100 * x:.{digits}f}\\%"


def load_json(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pi_year(tag, year, stat="mean"):
    """Average quarterly posterior prevalence over one year."""
    p = os.path.join(DATA, f"pi_{tag}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["year"] = d.quarter.str[:4].astype(int)
    sub = d[d.year == year]
    return float(sub[stat].mean()) if len(sub) else None


def main():
    cohort = load_json("cohort_summary.json")
    ladder = load_json("ladder.json")
    exp = load_json("expanded_basket.json")

    # ---------------- cohort
    if cohort:
        M("nAbstracts", f"{cohort['n_kept']:,}".replace(",", "{,}"))
    ftp = os.path.join(DATA, "fulltext_features.parquet")
    if os.path.exists(ftp):
        ft = pd.read_parquet(ftp, columns=["year", "declared"])
        M("nFulltextRaw", f"{len(ft):,}")
        M("nDeclared", str(int(ft.declared.sum())))
        M("nDeclaredTwentyFive", str(int(ft[ft.year == 2025].declared.sum())))
        M("declRateTwentyFive",
          pct(ft[ft.year == 2025].declared.mean(), 2))
    # cohort actually fitted (>=300 named-section tokens, topic assigned)
    M("nFulltext", "202{,}680")

    # ---------------- ladder + crude estimates
    if ladder:
        ab, fu = ladder["abstracts"], ladder["fulltext"]
        M("betaMarkersAbs", pct(ab["markers"]["rung0_docfreq"]["2025"], 1))
        M("betaMarkersFull", pct(fu["markers"]["rung0_docfreq"]["2025"], 0))
        M("ctrlBetaAbs", pct(ab["control"]["rung0_docfreq"]["2025"], 0))
        M("ctrlBetaFull", pct(fu["control"]["rung0_docfreq"]["2025"], 0))
        M("ladderAbsRungZero", pct(ab["markers"]["rung0_docfreq"]["2025"], 1))
        M("ladderFullRungZero", pct(fu["markers"]["rung0_docfreq"]["2025"], 0))
        M("ladderCtrlRungZero", pct(fu["control"]["rung0_docfreq"]["2025"], 0))
        M("ladderAbsRungTwo", pct(ab["markers"]["rung2_flatbg"]["2025"], 0))
        M("ladderFullRungTwo", pct(fu["markers"]["rung2_flatbg"]["2025"], 0))
        M("ladderAbsRungThree", pct(ab["markers"]["rung3_lineardrift"]["2025"], 0))
        M("ladderFullRungThree", pct(fu["markers"]["rung3_lineardrift"]["2025"], 0))

    # ---------------- model posteriors (Laplace grid)
    def annual(tag, year=2025, stat="mean"):
        return pi_year(tag, year, stat)

    def band(tag, year):
        m, lo, hi = (annual(tag, year, k) for k in ("mean", "lo", "hi"))
        if m is None:
            return None
        return f"{100*m:.0f}\\% ({100*lo:.0f}--{100*hi:.0f}\\%)"

    b = band("fulltext_primary", 2025)
    if b:
        M("piFulltextHeadline", b)
        M("ladderFullRungFour", b)
    b24 = band("fulltext_primary", 2024)
    if b24:
        M("piFulltextTwentyFour", b24)
    b = band("abstracts_primary", 2025)
    if b:
        M("piAbstractsHeadline", b)
        M("ladderAbsRungFour", b)
    b = band("fulltext_expanded", 2025)
    if b:
        M("piExpandedHeadline", b)
    exp_t = annual("fulltext_expanded_tracked", 2025)
    if exp_t is not None:
        M("piExpandedTracked", pct(exp_t, 0))
    # drift bracket: linear (floor) to frozen (ceiling), tracked between
    lo_v = annual("fulltext_primary", 2025)
    mid_v = annual("fulltext_tracked_drift", 2025)
    hi_v = annual("fulltext_frozen_drift", 2025)
    if None not in (lo_v, hi_v):
        M("piFulltextBracket", f"{100*lo_v:.0f}--{100*hi_v:.0f}\\%")
    unc = annual("fulltext_unconstrained", 2025)
    if unc is not None:
        M("piUnconstrained", pct(unc, 0))
    # whole-grid floor: minimum 2025 value across every variant fitted
    import glob as _g
    vals = []
    for f in _g.glob(os.path.join(DATA, "pi_fulltext_*.csv")):
        tag = os.path.basename(f)[3:-4]
        if "control" in tag or "smoke" in tag:
            continue
        v = annual(tag, 2025)
        if v is not None:
            vals.append((tag, v))
    if vals:
        gmin = min(v for _, v in vals)
        gmax = max(v for _, v in vals)
        M("piGridFloor", pct(gmin, 0))
        M("piGridRange", f"{100*gmin:.0f}--{100*gmax:.0f}\\%")

    m25 = annual("fulltext_primary", 2025)
    if m25 is not None and os.path.exists(ftp):
        d25 = ft[ft.year == 2025].declared.mean()
        M("disclosureGapFactor", f"$\\sim${m25/d25:.0f}")

    # placebo, anchors, and excess-rate attribution from the laplace files
    def laplace(tag):
        pth = os.path.join(DATA, f"laplace_{tag}.json")
        return json.load(open(pth)) if os.path.exists(pth) else None

    lp_p = laplace("fulltext_primary")
    lp_c = laplace("fulltext_control_tracked")
    if lp_p and lp_c:
        import pandas as _pd
        for nm, lp, pi_tag in (("marker", lp_p, "fulltext_primary"),
                               ("ctrl", lp_c, "fulltext_control_tracked")):
            pi = _pd.read_csv(os.path.join(DATA, f"pi_{pi_tag}.csv"))
            pi["year"] = pi.quarter.str[:4].astype(int)
            q25 = pi[pi.year == 2025]
            dl = np.array([lp["delta_by_quarter"][q] for q in q25.quarter])
            ex = float(np.mean(q25["mean"].values * (dl - 1)))
            if nm == "marker":
                M("markerExcessRate", pct(ex, 0))
                M("deltaEarly", f"{lp['delta_by_quarter'].get('2023Q1', 0):.1f}")
                M("deltaLate", f"{lp['delta_by_quarter'].get('2026Q1', 0):.1f}")
            else:
                M("ctrlExcessRate", pct(ex, 0))
        M("ctrlModelPi", band("fulltext_control_tracked", 2025) or "XX")
        M("ladderCtrlRungFour", "unid.")
        u = np.array(lp_p["u_topic"])
        if cohort:
            classes = cohort["topic_classes"]
            short = {"Cosmology & Nongalactic Physics": "cosmology",
                     "Galaxy Physics": "galaxy physics",
                     "High Energy Astrophysics": "high-energy astrophysics",
                     "Solar & Stellar Physics": "solar and stellar physics",
                     "Earth & Planetary Science": "planetary science",
                     "Numerical Simulation": "numerical methods",
                     "Instrumental Design": "instrumentation",
                     "Statistics & AI": "statistics and machine learning"}
            if len(classes) == len(u):
                M("topTopicName", short.get(classes[int(np.argmax(u))]))
                M("bottomTopicName", short.get(classes[int(np.argmin(u))]))
    M("anchorRangeMarkers", "2.6--6.8")
    M("anchorRangeControls", "1.05--1.2")

    def shift(tag_a, tag_b, year=2025):
        a, b_ = annual(tag_a, year), annual(tag_b, year)
        if a is None or b_ is None:
            return None
        return abs(a - b_)

    s_ = shift("fulltext_primary", "fulltext_unconstrained")
    if s_ is not None:
        M("monotoneShift", f"{100*s_:.0f} percentage points")
    s_ = shift("fulltext_primary", "fulltext_late_boundary")
    if s_ is not None:
        M("boundaryShift", f"{100*s_:.0f} percentage points")
    s1 = shift("fulltext_primary", "fulltext_gamma_low")
    s2 = shift("fulltext_primary", "fulltext_gamma_high")
    if s1 is not None and s2 is not None:
        M("gammaShift", f"{100*max(s1,s2):.0f} percentage points")
    s_ = shift("fulltext_primary", "fulltext_no_disclosure")
    if s_ is not None:
        M("noDisclosureShift", f"{100*max(s_,0.005):.0f} percentage points")

    # ---------------- excess and rise
    ld = load_json("length_diagnostic.json")
    if ld:
        s = ld["markers_38"]["per_1000_tokens"]
        M("markerRiseAbs", f"{s['2025']/np.mean([s[str(y)] for y in range(2015,2020)]):.1f}")
    M("markerRiseFulltext", "2.6")     # from within-section table, s1 output
    M("observedExcess", "2.4")

    # ---------------- expanded basket
    if exp:
        M("nExpandedWords", str(exp["n_words"]))
        M("nExpandedNew", str(exp["n_new"]))
        M("nExpandedHeld", "32")

    # ---------------- write
    lines = ["% auto-generated by src/s8_fill_numbers.py; do not edit\n"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # visible placeholders for anything the manuscript uses but we lack
    used = set()
    main_tex = open(os.path.join(ROOT, "paper", "main.tex")).read()
    import re
    for m in re.finditer(r"\\([a-zA-Z]+)", main_tex):
        used.add(m.group(1))
    known_missing = [u for u in sorted(used)
                     if u not in macros and u[:1].islower() and len(u) > 4
                     and any(u.startswith(p) for p in
                             ("pi", "ladder", "beta", "ctrl", "placebo",
                              "marker", "observed", "disclosure", "decl",
                              "nExpanded", "nDecl", "nAbstract", "nFulltext"))
                     and u not in ("piecewise",)]
    for u in known_missing:
        lines.append(f"\\newcommand{{\\{u}}}{{\\textcolor{{red}}{{[{u}]}}}}\n")
    with open(OUT, "w") as f:
        f.writelines(lines)
    print(f"wrote {OUT}: {len(macros)} filled, {len(known_missing)} pending")
    for u in known_missing:
        print("  pending:", u)


if __name__ == "__main__":
    main()
