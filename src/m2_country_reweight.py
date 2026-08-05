#!/usr/bin/env python3
"""
M2 (independent referee) -- country-composition check of the calibration.
Disclosure propensity and marker incidence anti-correlate across countries
(Fig. 5), so the disclosed writing sample may under-represent the
high-marker-yield populations and understate q. We fetch first-author
affiliations for the 188 writing-disclosure papers from ADS, split
native-English against non-native affiliations (same grouping as the equity
analysis), compute q per stratum, reweight to the 2025 corpus composition,
and rerun the alpha posterior with the reweighted mixture.
Output: data/m2_country_reweight.json
"""
import json, os, re, time, urllib.parse, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
NATIVE = ["USA", "United States", "UK", "United Kingdom", "England", "Scotland",
          "Australia", "Canada", "Ireland", "New Zealand"]

def ads(q, fl, rows=50):
    url = BASE + "?" + urllib.parse.urlencode({"q": q, "fl": fl, "rows": rows})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["response"]["docs"]
        except Exception:
            time.sleep(2.0 * (a + 1))
    return []

def is_native(aff):
    return any(n.lower() in aff.lower() for n in NATIVE)

def main():
    cls = json.load(open(os.path.join(DATA, "n1b_classified.json")))
    writing = cls["writing_ids"]
    BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())
    hits = {}
    n25 = k25 = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        base = re.sub(r"v\d+$", "", r["id"])
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        if base in set(writing):
            hits[base] = int(bool(toks & BASKET))
        if r["published"][:4] == "2025":
            n25 += 1
            if toks & BASKET:
                k25 += 1

    # fetch first-author affiliation per calibration paper, batched
    strata = {}
    ids = sorted(writing)
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        q = "identifier:(" + " OR ".join(f'"arXiv:{x}"' for x in chunk) + ")"
        for d in ads(q, "identifier,aff"):
            pid = None
            for ident in d.get("identifier", []):
                m = re.match(r"arXiv:(\d{4}\.\d{4,5})$", ident)
                if m and m.group(1) in set(chunk):
                    pid = m.group(1)
            affs = d.get("aff") or []
            first = affs[0] if affs else ""
            if pid and first and first != "-":
                strata[pid] = "native" if is_native(first) else "nonnative"
        time.sleep(0.3)
    print(f"affiliations resolved for {len(strata)}/{len(writing)}", flush=True)

    counts = {}
    for pid, s in strata.items():
        if pid in hits:
            c = counts.setdefault(s, [0, 0])
            c[1] += 1; c[0] += hits[pid]
    for s, (k, n) in counts.items():
        print(f"  {s}: q = {k}/{n} = {k/n:.3f}", flush=True)

    # corpus 2025 native share from the equity analysis affiliation queries
    geo = json.load(open(os.path.join(DATA, "c4_geography.json")))
    nat = sum(v["2025"]["total"] for c, v in geo.items()
              if "2025" in v and any(x.lower() in c.lower() for x in NATIVE))
    tot = sum(v["2025"]["total"] for v in geo.values() if "2025" in v)
    w_native = nat / tot if tot else 0.35
    print(f"  corpus native-affiliation weight: {w_native:.2f}", flush=True)

    # reweighted alpha posterior: mix Beta posteriors of the strata
    rng = np.random.default_rng(7)
    kn, nn = counts.get("native", [0, 1])
    km, nm = counts.get("nonnative", [0, 1])
    NMC = 4000
    qs = w_native * rng.beta(kn + 1, nn - kn + 1, NMC) + \
         (1 - w_native) * rng.beta(km + 1, nm - km + 1, NMC)
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    f0s = rng.normal(al["base_rate"], 0.001, NMC).clip(1e-4, 0.1)
    agrid = np.linspace(0, 1, 501)
    post = np.zeros(501)
    for i, a in enumerate(agrid):
        ft = ((1 - a) * f0s + a * qs).clip(1e-6, 1 - 1e-6)
        ll = k25 * np.log(ft) + (n25 - k25) * np.log1p(-ft)
        m = ll.max()
        post[i] = m + np.log(np.mean(np.exp(ll - m)))
    post = np.exp(post - post.max()); post /= post.sum()
    cdf = np.cumsum(post)
    med = float(agrid[np.searchsorted(cdf, 0.5)])
    lo, hi = float(agrid[np.searchsorted(cdf, 0.16)]), float(agrid[np.searchsorted(cdf, 0.84)])
    out = {"strata_q": {s: {"k": c[0], "n": c[1]} for s, c in counts.items()},
           "resolved": len(strata), "w_native_corpus": w_native,
           "alpha_reweighted_median": med, "alpha_reweighted_68": [lo, hi],
           "kt_2025": k25, "nt_2025": n25}
    json.dump(out, open(os.path.join(DATA, "m2_country_reweight.json"), "w"), indent=2)
    print(f"reweighted alpha median {med:.2f} [{lo:.2f},{hi:.2f}]68")

if __name__ == "__main__":
    main()
