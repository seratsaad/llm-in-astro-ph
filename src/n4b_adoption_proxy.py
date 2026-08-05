#!/usr/bin/env python3
"""
N4b (referee point 6) -- does the subfield decay depth of publicized markers
track an independent measure of that subfield's AI adoption?

Adoption proxy: fraction of a subfield's 2022-2025 papers cross-listed to a
machine-learning or AI category (cs.LG, cs.AI, cs.CL, cs.CV, stat.ML), computed
from the corpus itself. A model-mix or provider-side story gives no reason for
decay depth to follow this; author-side awareness does.
Output: data/n4b_adoption.json
"""
import json, os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
GROUPS = {
    "astro-ph.IM": "Instrum./Methods", "astro-ph.GA": "Galaxies",
    "astro-ph.CO": "Cosmology", "astro-ph.EP": "Earth/Planetary",
    "astro-ph.SR": "Solar/Stellar", "astro-ph.HE": "High-Energy",
}
ML = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"}

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def main():
    n = {g: 0 for g in GROUPS.values()}
    ml = {g: 0 for g in GROUPS.values()}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        g = GROUPS.get(r["primary_category"])
        if g is None:
            continue
        y = int(r["published"][:4])
        if y < 2022 or y > 2025:
            continue
        n[g] += 1
        if ML & set(r.get("categories", [])):
            ml[g] += 1

    grad = json.load(open(os.path.join(DATA, "n4_subfield_gradient.json")))
    rows = []
    for g in GROUPS.values():
        frac = ml[g] / n[g]
        decay = grad[g]["publicized"]["decay_frac"]
        rows.append({"group": g, "n_2022_2025": n[g], "ml_crosslist": ml[g],
                     "ml_frac": frac, "decay_frac": decay})
        print(f"{g:18s} ML cross-list {frac*100:5.2f}%  (n={n[g]:6d})   "
              f"pub decay {decay*100:.0f}%")
    x = [r["ml_frac"] for r in rows]
    y = [r["decay_frac"] for r in rows]
    rho = spearman(x, y)
    r_p = float(np.corrcoef(x, y)[0, 1])
    # exact permutation p for Spearman with n=6 (720 permutations)
    import itertools
    perms = 0; ge = 0
    for p in itertools.permutations(range(6)):
        perms += 1
        if spearman(x, [y[i] for i in p]) >= rho - 1e-12:
            ge += 1
    out = {"rows": rows, "spearman_rho": rho, "pearson_r": r_p,
           "perm_p_one_sided": ge / perms}
    json.dump(out, open(os.path.join(DATA, "n4b_adoption.json"), "w"), indent=2)
    print(f"\nSpearman rho = {rho:.2f}, Pearson r = {r_p:.2f}, "
          f"one-sided permutation p = {ge/perms:.3f} (n=6)")

if __name__ == "__main__":
    main()
