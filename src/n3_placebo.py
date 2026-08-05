#!/usr/bin/env python3
"""
N3 placebo (referee point 8) -- gain/loss asymmetry for frequency-matched
neutral word sets on the same changed abstract pairs.

Revisions tend to add text, so some add-over-remove asymmetry is expected for
any word set. For each of 1000 placebo sets we draw one neutral word per basket
word from its pre-2022 document-frequency rank neighborhood (basket words and
their morphological kin excluded), then run exactly the test run on the basket:
count papers whose revision added at least one set word the v1 lacked, and
papers whose revision removed one. The basket's net asymmetry is compared with
the placebo null.
Output: data/n3_placebo.json
"""
import json, os, re, collections, random

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASKET = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()
BASKET_SET = set(BASKET)
# morphological kin of basket stems, excluded from the placebo vocabulary
KIN_RE = re.compile(r"^(delv|underscor|intric|showcas|boast|tapestr|pivot|meticulous|"
                    r"nuanc|garner|multifacet|commendabl|noteworth|myriad|plethora|"
                    r"testament|encompass|seamless|elucidat|unravel|realm|leverag)")

def tok(s): return set(re.findall(r"[a-z]+", s.lower()))

def main():
    rng = random.Random(20260805)
    # pre-2022 document frequencies over the corpus
    df = collections.Counter()
    ndoc = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        if int(r["published"][:4]) >= 2022:
            continue
        ndoc += 1
        for w in tok(r["abstract"]):
            df[w] += 1
    print(f"pre-2022 docs: {ndoc}, vocab: {len(df)}", flush=True)

    # eligible placebo vocabulary: length>=4, df>=20, not basket or kin
    vocab = [(w, c) for w, c in df.items()
             if len(w) >= 4 and c >= 20 and w not in BASKET_SET and not KIN_RE.match(w)]
    vocab.sort(key=lambda wc: -wc[1])
    rank = {w: i for i, (w, c) in enumerate(vocab)}
    words_by_rank = [w for w, c in vocab]

    # rank neighborhoods for each basket word (nearest eligible frequency rank)
    neigh = {}
    for b in BASKET:
        cb = df.get(b, 20)
        # position where a word of count cb would sit
        lo, hi = 0, len(vocab)
        while lo < hi:
            mid = (lo + hi) // 2
            if vocab[mid][1] > cb: lo = mid + 1
            else: hi = mid
        i0, i1 = max(0, lo - 25), min(len(vocab), lo + 25)
        neigh[b] = words_by_rank[i0:i1]

    pairs = [json.loads(l) for l in open(os.path.join(DATA, "n3_pairs.jsonl"))]
    changed = []
    for p in pairs:
        t1 = tok(p["abstract_v1"]); t2 = tok(p["abstract_latest"])
        if t1 != t2:
            changed.append((t2 - t1, t1 - t2))
    print(f"changed pairs: {len(changed)}", flush=True)

    # overall addition asymmetry of revisions (context for the reader)
    tot_add = sum(len(a) for a, d in changed)
    tot_drop = sum(len(d) for a, d in changed)

    def run_set(ws):
        s = set(ws)
        added = sum(1 for a, d in changed if a & s)
        removed = sum(1 for a, d in changed if d & s)
        return added, removed

    b_added, b_removed = run_set(BASKET)
    NSET = 1000
    null_added, null_removed, null_net = [], [], []
    for k in range(NSET):
        ws = [rng.choice(neigh[b]) for b in BASKET]
        a, r = run_set(ws)
        null_added.append(a); null_removed.append(r); null_net.append(a - r)
    net = b_added - b_removed
    p_net = sum(1 for x in null_net if x >= net) / NSET
    p_added = sum(1 for x in null_added if x >= b_added) / NSET
    mean_a = sum(null_added) / NSET
    mean_r = sum(null_removed) / NSET
    out = {"changed_pairs": len(changed),
           "words_added_total": tot_add, "words_removed_total": tot_drop,
           "basket_added": b_added, "basket_removed": b_removed,
           "placebo_sets": NSET,
           "placebo_added_mean": mean_a, "placebo_removed_mean": mean_r,
           "placebo_net_mean": mean_a - mean_r,
           "p_net_ge_basket": p_net, "p_added_ge_basket": p_added,
           "null_net_percentiles": {q: sorted(null_net)[int(q / 100 * (NSET - 1))]
                                    for q in (50, 90, 95, 99)}}
    json.dump(out, open(os.path.join(DATA, "n3_placebo.json"), "w"), indent=2)
    print(f"revisions add {tot_add} vs remove {tot_drop} words overall "
          f"(ratio {tot_add/tot_drop:.2f})")
    print(f"basket: added {b_added}, removed {b_removed}, net {net}")
    print(f"placebo null: added {mean_a:.2f}, removed {mean_r:.2f}, "
          f"net {mean_a-mean_r:.2f}; P(net >= {net}) = {p_net:.4f}, "
          f"P(added >= {b_added}) = {p_added:.4f}")

if __name__ == "__main__":
    main()
