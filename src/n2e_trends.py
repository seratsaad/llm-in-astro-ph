#!/usr/bin/env python3
"""
N2e -- Google Trends attention spike per word (referee point 4, primary axis).
For each word we fetch worldwide weekly search interest 2023-01-01..2025-12-31
(each series is normalized to its own maximum, so levels are not comparable
across words) and compute the self-normalized spike ratio
    spike = mean(interest 2024-2025) / mean(interest 2023),
which is invariant to the per-word normalization. A word that became famous
after the AI-vocabulary discourse began shows spike > 1; a word nobody looked
up differently shows spike near 1. Also records the peak week.
Cached per word in data/n2e_trends_cache/. Output: data/n2e_trends.json
"""
import json, os, time, random

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA, "n2e_trends_cache")
os.makedirs(CACHE, exist_ok=True)

WORDS = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging offering
highlighting""".split()

def fetch(word, pt):
    fn = os.path.join(CACHE, word + ".json")
    if os.path.exists(fn):
        return json.load(open(fn))
    for a in range(5):
        try:
            pt.build_payload([word], timeframe="2023-01-01 2025-12-31")
            df = pt.interest_over_time()
            if df.empty:
                d = {"weeks": [], "vals": []}
            else:
                d = {"weeks": [str(i.date()) for i in df.index],
                     "vals": [int(v) for v in df[word]]}
            json.dump(d, open(fn, "w"))
            return d
        except Exception as e:
            wait = 20.0 * (a + 1) + random.uniform(0, 10)
            print(f"  retry {word} in {wait:.0f}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)
    return None

def main():
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=0)
    out = {}
    for w in WORDS:
        d = fetch(w, pt)
        if d is None or not d["vals"]:
            print(f"{w:14s} FAILED/EMPTY", flush=True)
            continue
        pre = [v for wk, v in zip(d["weeks"], d["vals"]) if wk < "2024-01-01"]
        post = [v for wk, v in zip(d["weeks"], d["vals"]) if wk >= "2024-01-01"]
        mpre = sum(pre) / len(pre) if pre else 0.0
        mpost = sum(post) / len(post) if post else 0.0
        imax = max(range(len(d["vals"])), key=lambda i: d["vals"][i])
        out[w] = {"spike": (mpost / mpre) if mpre > 0 else None,
                  "pre_mean": mpre, "post_mean": mpost,
                  "peak_week": d["weeks"][imax]}
        s = out[w]["spike"]
        print(f"{w:14s} spike {s if s is None else round(s,2)}  peak {out[w]['peak_week']}", flush=True)
        time.sleep(12 + random.uniform(0, 6))
    json.dump(out, open(os.path.join(DATA, "n2e_trends.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
