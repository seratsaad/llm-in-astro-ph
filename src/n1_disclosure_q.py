#!/usr/bin/env python3
"""
N1 -- disclosure-calibrated q.
Papers that state model use in their acknowledgments are self-labeled positives.
We fetch them from NASA ADS (ack: search, astronomy collection, 2023-2025), match
them to the astro-ph abstract corpus by arXiv identifier, and measure the fraction
whose abstract carries at least one basket word. That fraction is an empirical
estimate of q, the probability that a model-touched abstract keeps a basket word,
and it converts the lower bound alpha_lb = f_t - f0 into a point estimate
alpha = (f_t - f0) / (q - f0).

Output: data/n1_disclosure_q.json
"""
import json, os, re, time, urllib.parse, urllib.request

TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

LLM = ["ChatGPT", "GPT-4", "GPT-4o", "large language model", "large language models",
       "GitHub Copilot", "generative pre-trained transformer"]
BASKET = {"delve","delves","delving","underscore","underscores","underscoring",
    "intricate","intricacies","showcasing","showcase","showcases","showcased",
    "boasts","tapestry","pivotal","meticulous","meticulously","nuanced",
    "garner","garners","garnered","multifaceted","commendable","noteworthy",
    "myriad","plethora","testament","encompassing","seamless","seamlessly",
    "elucidate","elucidating","unravel","unraveling","unravelling",
    "realm","realms","leveraging"}

def ads_disclosed_ids(year):
    """arXiv ids of astronomy papers whose acknowledgments match the LLM terms."""
    q = 'ack:(' + " OR ".join(f'"{t}"' for t in LLM) + ') database:astronomy'
    ids = []
    start = 0
    while True:
        url = BASE + "?" + urllib.parse.urlencode(
            {"q": q, "fq": f"year:{year}", "fl": "identifier", "rows": 200, "start": start})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        for a in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.load(r)["response"]
                break
            except Exception:
                time.sleep(1.5 * (a + 1))
        docs = resp["docs"]
        for d in docs:
            for ident in d.get("identifier", []):
                m = re.match(r"arXiv:(\d{4}\.\d{4,5})$", ident)
                if m:
                    ids.append(m.group(1)); break
        start += len(docs)
        if start >= resp["numFound"] or not docs:
            break
        time.sleep(0.3)
    return ids, resp["numFound"]

def main():
    # corpus abstracts keyed by base arXiv id (2023-2025 only, to bound memory)
    corpus = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if 2023 <= y <= 2025:
            corpus[re.sub(r"v\d+$", "", r["id"])] = (y, r["abstract"])
    print(f"corpus 2023-2025: {len(corpus)} abstracts", flush=True)

    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    f0 = al["base_rate"]

    out = {"per_year": {}, "f0": f0}
    pooled_hit = 0; pooled_n = 0
    for year in (2023, 2024, 2025):
        ids, nfound = ads_disclosed_ids(year)
        matched = [i for i in ids if i in corpus]
        hits = sum(1 for i in matched if set(re.findall(r"[a-z]+", corpus[i][1].lower())) & BASKET)
        ft = al["by_year"][str(year)]["rate"]
        qhat = hits / len(matched) if matched else float("nan")
        alpha = (ft - f0) / (qhat - f0) if matched and qhat > f0 else float("nan")
        out["per_year"][year] = {"ads_found": nfound, "arxiv_ids": len(ids),
            "matched_astroph": len(matched), "basket_hits": hits, "q_hat": qhat,
            "f_t": ft, "alpha_point": alpha}
        pooled_hit += hits; pooled_n += len(matched)
        print(f"{year}: ADS {nfound}, arXiv ids {len(ids)}, matched {len(matched)}, "
              f"hits {hits}, q-hat {qhat:.3f}, alpha {alpha*100 if alpha==alpha else float('nan'):.1f}%", flush=True)
        time.sleep(0.5)

    q_pool = pooled_hit / pooled_n if pooled_n else float("nan")
    ft25 = al["by_year"]["2025"]["rate"]
    import math
    se_q = math.sqrt(q_pool * (1 - q_pool) / pooled_n) if pooled_n else float("nan")
    alpha25 = (ft25 - f0) / (q_pool - f0)
    # simple error propagation: dominated by q uncertainty
    dadq = -(ft25 - f0) / (q_pool - f0) ** 2
    se_alpha = abs(dadq) * se_q
    out["pooled"] = {"n": pooled_n, "hits": pooled_hit, "q_hat": q_pool, "se_q": se_q,
                     "alpha_2025_point": alpha25, "se_alpha": se_alpha}
    json.dump(out, open(os.path.join(DATA, "n1_disclosure_q.json"), "w"), indent=2)
    print(f"\npooled q-hat = {q_pool:.3f} +- {se_q:.3f}  (n={pooled_n})")
    print(f"alpha(2025) point estimate = {alpha25*100:.1f}% +- {se_alpha*100:.1f}%   (floor was 4.3%)")

if __name__ == "__main__":
    main()
