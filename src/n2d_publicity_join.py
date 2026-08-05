#!/usr/bin/env python3
"""
N2d -- join the extended per-word decay measurements with the external
publicity measures (HN Algolia primary; GDELT where cached) and compute the
half-life vs publicity statistics for the scatter (referee point 4).

Reliability floor for a usable half-life: peak excess >= 0.10 pp or
peak_docs >= 15; words below the floor are reported but not used in the rank
statistics (their half-lives are quarter-noise).
Output: data/n2d_publicity_join.json
"""
import json, os, glob
import numpy as np, importlib.util

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location("n2b", os.path.join(HERE, "n2b_publicity.py"))
n2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n2b)

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def perm_p(x, y, rho, n_iter=20000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    cnt = 0
    for _ in range(n_iter):
        if spearman(x, rng.permutation(y)) <= rho + 1e-12:
            cnt += 1
    return cnt / n_iter

def main():
    decay = n2b.decay_all()
    hn = json.load(open(os.path.join(DATA, "n2c_hn_publicity.json")))
    gd = {}
    for fn in glob.glob(os.path.join(DATA, "n2b_gdelt_cache", "*_co.json")):
        w = os.path.basename(fn)[:-8]
        s = n2b.series_stats(json.load(open(fn)))
        if s:
            gd[w] = s
    rows = []
    for w, d in decay.items():
        if w not in hn:
            continue
        reliable = d["peak_excess_pct"] >= 0.10 or d["peak_docs"] >= 15
        rows.append({"word": w, **d, "reliable": reliable,
                     "hn_total": hn[w]["total"], "hn_peak_q": hn[w]["peak_q"],
                     "gdelt_co_mean": gd.get(w, {}).get("mean"),
                     "gdelt_peak_date": gd.get(w, {}).get("peak_date")})
    # rank stats on reliable words
    rel = [r for r in rows if r["reliable"]]
    meas = [r for r in rel if r["half_life_quarters"] is not None]
    cen = [r for r in rel if r["half_life_quarters"] is None]
    stats = {"n_words": len(rows), "n_reliable": len(rel),
             "n_measured": len(meas), "n_censored": len(cen)}
    if len(meas) >= 5:
        x = [np.log10(r["hn_total"] + 1) for r in meas]
        y = [r["half_life_quarters"] for r in meas]
        rho = spearman(x, y)
        stats["spearman_hl_vs_hn"] = rho
        stats["perm_p_neg"] = perm_p(x, y, rho)
    if meas and cen:
        a = [r["hn_total"] for r in meas]; b = [r["hn_total"] for r in cen]
        u = sum(1 for xx in a for yy in b if xx > yy) + 0.5 * sum(1 for xx in a for yy in b if xx == yy)
        stats["mw_frac_measured_gt_censored"] = u / (len(a) * len(b))
        stats["mw_U"] = u; stats["n_meas_cen"] = [len(a), len(b)]
    out = {"rows": rows, "stats": stats}
    json.dump(out, open(os.path.join(DATA, "n2d_publicity_join.json"), "w"),
              indent=1, default=float)
    rows.sort(key=lambda r: -r["hn_total"])
    print(f"{'word':14s} {'HN':>6s} {'half-life':>10s} {'peak pp':>8s} rel")
    for r in rows:
        hl = f"{r['half_life_quarters']:.2f}" if r["half_life_quarters"] else "cens"
        print(f"{r['word']:14s} {r['hn_total']:6d} {hl:>10s} "
              f"{r['peak_excess_pct']:8.3f} {'*' if r['reliable'] else ' '}")
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
