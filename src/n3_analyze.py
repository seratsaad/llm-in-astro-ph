#!/usr/bin/env python3
"""
N3 analysis -- injected vocabulary from version diffs.
For each paper with a pre-2022 v1 and a post-2022 revision, compare the v1
abstract with the latest abstract. Words present in the revised abstract but not
in the v1 form the injected vocabulary. Same paper, same science, same authors,
so topic is fully controlled. We measure:
  (a) the fraction of revised papers whose revision ADDED at least one basket
      word that the v1 lacked (against the reverse direction as a null),
  (b) the ranking of injected words across all pairs, style vs topic,
  (c) the same for removed words (avoidance in revisions).
Output: data/n3_results.json
"""
import json, os, re, collections

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())

def tok(s): return set(re.findall(r"[a-z]+", s.lower()))

def main():
    pairs = [json.loads(l) for l in open(os.path.join(DATA, "n3_pairs.jsonl"))]
    n = len(pairs)
    added_basket = removed_basket = changed = 0
    inj = collections.Counter(); rem = collections.Counter()
    for p in pairs:
        t1 = tok(p["abstract_v1"]); t2 = tok(p["abstract_latest"])
        if t1 == t2: continue
        changed += 1
        add = t2 - t1; drop = t1 - t2
        if add & BASKET: added_basket += 1
        if drop & BASKET: removed_basket += 1
        for w in add: inj[w] += 1
        for w in drop: rem[w] += 1
    out = {"pairs": n, "changed_abstract": changed,
           "added_basket_word": added_basket,
           "removed_basket_word": removed_basket,
           "top_injected": inj.most_common(40),
           "top_removed": rem.most_common(40),
           "injected_basket_detail": {w: c for w, c in inj.items() if w in BASKET},
           "removed_basket_detail": {w: c for w, c in rem.items() if w in BASKET}}
    json.dump(out, open(os.path.join(DATA, "n3_results.json"), "w"), indent=2)
    print(f"pairs {n}, changed {changed}, added-basket {added_basket}, removed-basket {removed_basket}")
    if changed:
        print(f"P(add basket | revised+changed) = {added_basket/changed:.3f}")
        print(f"P(remove basket | revised+changed) = {removed_basket/changed:.3f}")
    print("top injected:", inj.most_common(15))

if __name__ == "__main__":
    main()
