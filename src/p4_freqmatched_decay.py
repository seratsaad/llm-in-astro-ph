#!/usr/bin/env python3
"""
P4 (referee round 4, point 4) -- is the named/unnamed decay contrast just a
proxy for baseline frequency?

Objection: the words that decayed were largely alien to pre-2022 astronomy
(delve, showcasing, tapestry), while the words still rising are ordinary
participles with heavy organic use (leveraging, encompassing, offering). A
rare ornament is both more likely to be named as a tell and more likely to be
deletable without rewriting the sentence, so the split may be proxying
"removable ornament versus functional vocabulary".

Three tests, all on the same per-word decay measure:
  A. Is the confound real? Correlate pre-2022 document frequency with decay.
  B. Stratified contrast. Split words at the median pre-2022 frequency and
     run the named/unnamed rank test within each stratum.
  C. Matched pairs. For each named word take the unnamed word closest in
     log pre-2022 frequency, without reuse, and run a paired sign test.
  D. Regression. Decay on named status and log baseline frequency together,
     to see which carries the signal.
Output: data/p4_freqmatched.json
"""
import json, os, re, collections, itertools
import numpy as np
import importlib.util

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location("n2b", os.path.join(HERE, "n2b_publicity.py"))
n2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n2b)

# Floor for a usable decay measure. Looser than the main text's 0.20 pp so
# that the matched design has enough words to work with, with the main-text
# subset reported alongside.
FLOOR_PP = 0.05


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_rank_p(a, b):
    """One-sided exact permutation p that group a exceeds group b."""
    vals = list(a) + list(b)
    n = len(a)
    obs = sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)
    ge = tot = 0
    for idx in itertools.combinations(range(len(vals)), n):
        g = [vals[i] for i in idx]
        h = [vals[i] for i in range(len(vals)) if i not in idx]
        u = sum(1 for x in g for y in h if x > y) + 0.5 * sum(1 for x in g for y in h if x == y)
        tot += 1
        if u >= obs:
            ge += 1
    return obs, ge / tot


def main():
    # pre-2022 document frequency for every basket word
    pre = collections.Counter(); npre = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y < 2015 or y >= 2022:
            continue
        npre += 1
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        for w in n2b.WORDS:
            if w in toks:
                pre[w] += 1

    decay = n2b.decay_all()
    per = json.load(open(os.path.join(DATA, "n2f_publicity_sources.json")))["per_word"]

    rows = []
    for w, d in decay.items():
        if d["peak_excess_pct"] < FLOOR_PP:
            continue
        dec = (d["peak_excess_pct"] - d["end_excess_pct"]) / d["peak_excess_pct"] * 100
        f0 = pre[w] / npre * 100
        rows.append({"word": w, "named": len(per.get(w, [])) > 0,
                     "n_sources": len(per.get(w, [])),
                     "pre2022_pct": f0, "log_f0": np.log10(max(f0, 1e-4)),
                     "decay_pct": dec, "peak_pp": d["peak_excess_pct"],
                     "main_text_set": d["peak_excess_pct"] >= 0.20})
    rows.sort(key=lambda r: r["log_f0"])
    out = {"n_words": len(rows), "floor_pp": FLOOR_PP, "rows": rows}

    named = [r for r in rows if r["named"]]
    unnamed = [r for r in rows if not r["named"]]
    print(f"words: {len(rows)} ({len(named)} named, {len(unnamed)} unnamed)\n")
    print(f"{'word':14s} {'named':>5s} {'pre2022%':>9s} {'decay%':>7s}")
    for r in rows:
        print(f"{r['word']:14s} {str(r['named']):>5s} {r['pre2022_pct']:9.4f} {r['decay_pct']:7.0f}")

    # A. is the confound real
    rho_fd = spearman([r["log_f0"] for r in rows], [r["decay_pct"] for r in rows])
    rho_fn = spearman([r["log_f0"] for r in rows], [1.0 * r["named"] for r in rows])
    out["A_confound"] = {"spearman_logf0_vs_decay": rho_fd,
                         "spearman_logf0_vs_named": rho_fn}
    print(f"\nA. baseline frequency vs decay: rho = {rho_fd:+.2f}")
    print(f"   baseline frequency vs named:  rho = {rho_fn:+.2f}")

    # B. stratified rank test
    med = float(np.median([r["log_f0"] for r in rows]))
    strata = {}
    for lab, sel in (("rare", [r for r in rows if r["log_f0"] <= med]),
                     ("common", [r for r in rows if r["log_f0"] > med])):
        a = [r["decay_pct"] for r in sel if r["named"]]
        b = [r["decay_pct"] for r in sel if not r["named"]]
        if a and b:
            u, p = exact_rank_p(a, b)
            strata[lab] = {"n_named": len(a), "n_unnamed": len(b),
                           "mean_named": float(np.mean(a)), "mean_unnamed": float(np.mean(b)),
                           "U": u, "p_one_sided": p}
            print(f"B. {lab:6s} stratum: named {np.mean(a):+.0f}% (n={len(a)}) vs "
                  f"unnamed {np.mean(b):+.0f}% (n={len(b)}), p = {p:.3f}")
        else:
            strata[lab] = {"n_named": len(a), "n_unnamed": len(b), "note": "one group empty"}
            print(f"B. {lab:6s} stratum: cannot test, named {len(a)} unnamed {len(b)}")
    out["B_stratified"] = {"median_log_f0": med, "strata": strata}

    # C. matched pairs, nearest unnamed neighbour in log baseline frequency
    pool = list(unnamed)
    pairs = []
    for r in sorted(named, key=lambda r: -r["peak_pp"]):
        if not pool:
            break
        j = min(range(len(pool)), key=lambda i: abs(pool[i]["log_f0"] - r["log_f0"]))
        m = pool.pop(j)
        pairs.append({"named": r["word"], "unnamed": m["word"],
                      "f0_named": r["pre2022_pct"], "f0_unnamed": m["pre2022_pct"],
                      "decay_named": r["decay_pct"], "decay_unnamed": m["decay_pct"],
                      "named_higher": r["decay_pct"] > m["decay_pct"]})
    wins = sum(1 for p in pairs if p["named_higher"])
    # exact one-sided sign test
    from math import comb
    n = len(pairs)
    p_sign = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else None
    out["C_matched"] = {"n_pairs": n, "named_higher": wins, "p_sign": p_sign,
                        "pairs": pairs}
    print(f"\nC. matched pairs: named decayed more in {wins} of {n} pairs, sign p = {p_sign:.3f}")
    for p in pairs:
        print(f"   {p['named']:13s} ({p['f0_named']:.3f}%, {p['decay_named']:+.0f}%) vs "
              f"{p['unnamed']:13s} ({p['f0_unnamed']:.3f}%, {p['decay_unnamed']:+.0f}%)")

    # D. regression, decay on named and log baseline frequency
    X = np.column_stack([np.ones(len(rows)),
                         [1.0 * r["named"] for r in rows],
                         [r["log_f0"] for r in rows]])
    y = np.array([r["decay_pct"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(rows) - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    out["D_regression"] = {"coef_intercept": float(beta[0]), "coef_named": float(beta[1]),
                           "se_named": float(se[1]), "t_named": float(beta[1] / se[1]),
                           "coef_log_f0": float(beta[2]), "se_log_f0": float(se[2]),
                           "t_log_f0": float(beta[2] / se[2]), "dof": dof}
    print(f"\nD. decay = {beta[0]:.0f} + {beta[1]:.0f}(named) + {beta[2]:.0f}(log f0)")
    print(f"   named:  {beta[1]:+.0f} +- {se[1]:.0f}  (t = {beta[1]/se[1]:+.2f})")
    print(f"   log f0: {beta[2]:+.0f} +- {se[2]:.0f}  (t = {beta[2]/se[2]:+.2f})")

    json.dump(out, open(os.path.join(DATA, "p4_freqmatched.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
