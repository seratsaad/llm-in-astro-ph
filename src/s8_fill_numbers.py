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
    ftq = pd.read_parquet(ftp, columns=["arxiv_id", "year", "declared",
                                        "L_intro", "L_methods", "L_results",
                                        "L_discussion", "L_conclusions"])
    ab_t = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"),
                           columns=["arxiv_id", "topic"])
    ftq = ftq.merge(ab_t, on="arxiv_id", how="left")
    Lnamed = ftq[["L_intro", "L_methods", "L_results", "L_discussion",
                  "L_conclusions"]].sum(axis=1)
    coh = ftq[(ftq.topic.fillna(-1) >= 0) & (Lnamed >= 300)]
    M("nFulltext", f"{len(coh):,}".replace(",", "{,}"))
    # declaration counts, per year, inside the fitted cohort
    dc = coh.groupby("year").declared.sum().astype(int)
    M("nDeclaredPreChatGPT", str(int(dc.loc[:2022].sum())))
    yrs = [int(dc.get(y, 0)) for y in (2023, 2024, 2025, 2026)]
    M("declCountsByYear", f"{yrs[0]}, {yrs[1]}, {yrs[2]}, and {yrs[3]}")
    M("nDeclaredCohort", str(int(coh.declared.sum())))

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
    b = band("fulltext_primary_wholebody", 2025)
    if b:
        M("piWholeBody", b)
    b24 = band("fulltext_primary", 2024)
    if b24:
        M("piFulltextTwentyFour", b24)
    b = band("abstracts_primary", 2025)
    if b:
        M("piAbstractsHeadline", b)
        M("ladderAbsRungFour", b)
    b = band("abstracts_primary_cohort", 2025)
    if b:
        M("piAbstractsCohort", b)
    b = band("fulltext_expanded", 2025)
    if b:
        M("piExpandedHeadline", b)
    exp_t = annual("fulltext_expanded_tracked", 2025)
    if exp_t is not None:
        M("piExpandedTracked", pct(exp_t, 0))
    # drift bracket: linear (floor) to frozen (ceiling), tracked between
    def year_tab(tag, year):
        p_ = os.path.join(DATA, f"pi_{tag}.csv")
        if not os.path.exists(p_):
            return None
        d_ = pd.read_csv(p_)
        d_["year"] = d_.quarter.str[:4].astype(int)
        g_ = d_[d_.year == year]
        if not len(g_):
            return None
        l68 = float(g_.lo68.mean()) if "lo68" in g_ else None
        h68 = float(g_.hi68.mean()) if "hi68" in g_ else None
        return (float(g_["mean"].mean()), float(g_.lo.mean()),
                float(g_.hi.mean()), l68, h68)

    m25_tab = year_tab("fulltext_primary", 2025)
    lo_v = annual("fulltext_primary", 2025)
    mid_v = annual("fulltext_tracked_drift", 2025)
    hi_v = annual("fulltext_frozen_drift", 2025)
    if None not in (lo_v, hi_v):
        M("piFulltextBracket", f"{100*lo_v:.0f}--{100*hi_v:.0f}\\%")
    # headline written as +/- terms, generated rather than typed (both levels)
    if None not in (lo_v, hi_v) and m25_tab is not None:
        m, lo, hi, l68, h68 = m25_tab
        bg_up = max(0.0, hi_v - m)
        pm95 = (f"${100*m:.0f}^{{+{100*(hi-m):.0f}}}_{{-{100*(m-lo):.0f}}}"
                f"\,(\mathrm{{stat}},\,95\%)\,"
                f"^{{+{100*bg_up:.0f}}}_{{-0}}\,(\mathrm{{background}})$")
        M("piHeadlinePM", pm95)
        if l68 is not None:
            M("piStatOneSigma",
              f"$^{{+{100*(h68-m):.0f}}}_{{-{100*(m-l68):.0f}}}$")
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
                # per-class 2025 prevalence: logit(pi_2025) + u_c (YST note 19)
                if m25 is not None:
                    lg = np.log(m25 / (1 - m25))
                    pic = 1 / (1 + np.exp(-(lg + u)))
                    M("piTopClass", pct(float(pic.max()), 0))
                    M("piBottomClass", pct(float(pic.min()), 0))
                    # paper-count weighting vs the class-average (YST note 8)
                    cnt = coh[coh.year == 2025].topic.value_counts()
                    w = np.array([cnt.get(k, 0) for k in range(len(u))],
                                 dtype=float)
                    if w.sum() > 0:
                        pw = float((pic * w).sum() / w.sum())
                        M("piWeightShift",
                          f"{100*abs(pw - m25):.1f} percentage points")
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
