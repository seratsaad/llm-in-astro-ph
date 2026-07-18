#!/usr/bin/env python3
"""
Analyze the C2 source-leak scan (works on partial or complete data).
- Per-year: papers scanned, % with recoverable source, Tier-1 hard-leak rate,
  Tier-2 prompt-residue rate (with pre-2023 baseline as the false-positive floor).
- Dumps every Tier-1 hit and every post-2022 Tier-2 hit for manual verification.
- Concealment factor: source-leak rate vs ADS disclosure rate.
"""
import json, os, collections
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
SCAN = os.path.join(DATA, "source_scan.jsonl")

def main():
    recs = [json.loads(l) for l in open(SCAN)]
    ok = [r for r in recs if r.get("ok")]
    by_year = collections.defaultdict(lambda: {"n": 0, "src": 0, "t1": 0, "t2": 0})
    t1_hits, t2_hits = [], []
    for r in ok:
        y = r["year"]; d = by_year[y]
        d["n"] += 1
        if r.get("has_source"): d["src"] += 1
        hits = r.get("hits", [])
        has_t1 = any(h["tier"] == 1 for h in hits)
        has_t2 = any(h["tier"] == 2 for h in hits)
        if has_t1: d["t1"] += 1
        if has_t2: d["t2"] += 1
        for h in hits:
            (t1_hits if h["tier"] == 1 else t2_hits).append(
                {"id": r["id"], "year": y, "pattern": h["pattern"],
                 "where": h["where"], "snippet": h["snippet"]})

    rows = []
    for y in sorted(by_year):
        d = by_year[y]
        rows.append({"year": y, "scanned": d["n"], "with_source": d["src"],
                     "src_pct": 100*d["src"]/d["n"] if d["n"] else 0,
                     "tier1": d["t1"], "tier1_pct": 100*d["t1"]/d["n"] if d["n"] else 0,
                     "tier2": d["t2"], "tier2_pct": 100*d["t2"]/d["n"] if d["n"] else 0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "source_leak_by_year.csv"), index=False)
    print("=== source-leak scan (partial ok) ===")
    print(df.to_string(index=False))
    print(f"\ntotal scanned={len(ok)}  Tier-1 hits={len(t1_hits)}  Tier-2 hits={len(t2_hits)}")

    # baseline false-positive floor from pre-2023
    pre = df[df.year <= 2022]
    if pre.scanned.sum():
        print(f"\nPre-2023 baseline: Tier-1 {pre.tier1.sum()}/{pre.scanned.sum()} "
              f"({100*pre.tier1.sum()/pre.scanned.sum():.2f}%), "
              f"Tier-2 {pre.tier2.sum()}/{pre.scanned.sum()} "
              f"({100*pre.tier2.sum()/pre.scanned.sum():.2f}%)")

    print("\n=== ALL TIER-1 HARD-LEAK HITS (verify each) ===")
    for h in t1_hits:
        print(f"  {h['id']} ({h['year']}) [{h['pattern']}] :: ...{h['snippet']}...")
    print("\n=== TIER-2 (post-2022 only) prompt-residue hits ===")
    for h in sorted(t2_hits, key=lambda x: -x["year"]):
        if h["year"] >= 2023:
            print(f"  {h['id']} ({h['year']}) [{h['pattern']}] :: ...{h['snippet']}...")

    # concealment factor vs ADS disclosure (recent pooled)
    try:
        ads = json.load(open(os.path.join(DATA, "ads_disclosure.json")))
        rec = df[df.year.between(2024, 2026)]
        leak = (rec.tier1.sum() + rec.tier2.sum())  # generous "any signal"
        leak_pct = 100*leak/rec.scanned.sum() if rec.scanned.sum() else 0
        yrs = [2024, 2025, 2026]
        ack = sum(ads["ack"][str(y)] for y in yrs); tot = sum(ads["total_astro"][str(y)] for y in yrs)
        print(f"\n2024-26 source-leak (any tier) ~ {leak_pct:.2f}% of scanned; "
              f"ADS ack-disclosure ~ {100*ack/tot:.2f}% of papers")
    except Exception as e:
        print("concealment calc skipped:", e)

def make_figure():
    """The 'detection ladder': how much LLM use each method reveals in astronomy.
    Each rung is a different lens; cruder/harder-evidence methods find far less."""
    import matplotlib as mpl, matplotlib.pyplot as plt
    from pantera_style import C as PC
    import json as _j
    df = pd.read_csv(os.path.join(DATA, "source_leak_by_year.csv"))
    rec = df[df.year.between(2024, 2026)]
    src_leak = 100*(rec.tier1.sum()+rec.tier2.sum())/rec.scanned.sum()
    ads = _j.load(open(os.path.join(DATA, "ads_disclosure.json")))
    yrs = [2024, 2025, 2026]
    ack = 100*sum(ads["ack"][str(y)] for y in yrs)/sum(ads["total_astro"][str(y)] for y in yrs)
    full = 100*sum(ads["full"][str(y)] for y in yrs)/sum(ads["total_astro"][str(y)] for y in yrs)
    al = _j.load(open(os.path.join(DATA, "alpha_summary.json")))
    alpha = al["by_year"]["2025"]["excess"]*100

    rungs = [
        ("Crude source-code leaks\n(\"as an AI language model\" in LaTeX)", src_leak, PC["vermillion"]),
        ("Explicit disclosure\n(acknowledgments, NASA ADS)", ack, PC["orange"]),
        ("Any AI mention in full text\n(NASA ADS)", full, PC["sky"]),
        ("Marker-word excess\n(this work, lower bound)", alpha, PC["blue"]),
        ("Biomedical benchmark\n(Kobak+2024, for scale)", 13.5, PC["grey"]),
    ]
    labels = [r[0] for r in rungs]; vals = [r[1] for r in rungs]; cols = [r[2] for r in rungs]
    from pantera_style import no_minor_y
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = list(range(len(rungs)))
    for yi, v, c in zip(y, vals, cols):
        ax.plot([0.05, v], [yi, yi], lw=0.7, color="#BBBBBB", zorder=1)
        ax.plot(v, yi, "o", ms=5, color=c, zorder=3)
        ax.text(v*1.25, yi, f"{v:.2f}%" if v < 1 else f"{v:.1f}%", va="center",
                fontsize=7, color="#555555")
    no_minor_y(ax)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.05, 40)
    ax.set_xlabel("% of astronomy papers (2024 to 2026), log scale", labelpad=4)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figs", "fig9_detection_ladder.png"),
                bbox_inches="tight")
    print(f"wrote figs/fig9_detection_ladder.png  (src_leak={src_leak:.2f}%, ack={ack:.2f}%, "
          f"full={full:.2f}%, alpha={alpha:.2f}%)")

if __name__ == "__main__":
    main()
    import sys
    if "--fig" in sys.argv:
        make_figure()
