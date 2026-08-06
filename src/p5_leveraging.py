#!/usr/bin/env python3
"""
P5 (referee round 4, point 5) -- a dedicated treatment of "leveraging".

The word contributes about a quarter of the floor and has the most plausible
organic driver, since machine-learning method papers use it naturally. The
earlier bound relied on cs cross-listing, which is weak because most astronomy
papers that use machine-learning methods never cross-list.

Two better tests:
  A. Restrict to abstracts containing no machine-learning or method vocabulary
     at all. If the rise of "leveraging" persists among papers that never
     mention such methods, the rise is not carried by topic.
  B. Report the calibrated estimate with "leveraging" removed from the basket,
     recomputing both the floor and q on the same reduced word list, which is
     the configuration the earlier sensitivity table left out.
Output: data/p5_leveraging.json
"""
import json, os, re
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

FULL = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()

# Vocabulary that marks a methods or machine-learning paper. Deliberately
# broad, since the test is stronger the more aggressively we strip such papers.
ML = re.compile(
    r"\b(machine learning|deep learning|neural network|neural nets?|"
    r"convolutional|transformer|random forest|gradient boost|xgboost|"
    r"support vector|classifier|classification algorithm|training set|"
    r"training data|trained on|feature extraction|autoencoder|"
    r"gaussian process|bayesian inference|mcmc|emulator|surrogate model|"
    r"algorithm|pipeline|framework|architecture|dataset|data set|"
    r"simulation suite|inference|regression|clustering|"
    r"artificial intelligence|language model|foundation model|"
    r"supervised|unsupervised|self-supervised|reinforcement)\b", re.I)


def main():
    years, tok_cache = {}, {}
    per_word_pre = {w: 0 for w in FULL}
    npre = 0
    # counters: [all, non-ML] x [pre-2022, 2024-25]
    cnt = {("all", "pre"): 0, ("all", "post"): 0,
           ("noml", "pre"): 0, ("noml", "post"): 0}
    lev = {k: 0 for k in cnt}
    basket_any = {k: 0 for k in cnt}
    basket_noLev = {k: 0 for k in cnt}
    BS, BS_noLev = set(FULL), set(FULL) - {"leveraging"}

    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y < 2015 or y > 2025:
            continue
        ab = r["abstract"]
        toks = set(re.findall(r"[a-z]+", ab.lower()))
        if y < 2022:
            era = "pre"; npre += 1
            for w in FULL:
                if w in toks:
                    per_word_pre[w] += 1
        elif y >= 2024:
            era = "post"
        else:
            continue
        noml = ML.search(ab) is None
        for grp in (("all", era),) + ((("noml", era),) if noml else ()):
            cnt[grp] += 1
            if "leveraging" in toks: lev[grp] += 1
            if toks & BS: basket_any[grp] += 1
            if toks & BS_noLev: basket_noLev[grp] += 1

    def rate(d, k): return d[k] / cnt[k] * 100

    out = {"counts": {f"{a}_{b}": cnt[(a, b)] for a, b in cnt}}
    print("A. leveraging, all abstracts vs abstracts with no method vocabulary")
    for grp in ("all", "noml"):
        pre_r, post_r = rate(lev, (grp, "pre")), rate(lev, (grp, "post"))
        out[f"leveraging_{grp}"] = {"pre2022_pct": pre_r, "post2024_pct": post_r,
                                    "excess_pp": post_r - pre_r,
                                    "n_pre": cnt[(grp, "pre")], "n_post": cnt[(grp, "post")]}
        print(f"   {grp:5s}: pre-2022 {pre_r:.3f}%, 2024-25 {post_r:.3f}%, "
              f"excess {post_r-pre_r:+.3f} pp  (n_post {cnt[(grp,'post')]:,})")
    share = cnt[("noml", "post")] / cnt[("all", "post")] * 100
    print(f"   abstracts with no method vocabulary: {share:.0f} per cent of 2024-25")
    r_all = out["leveraging_all"]["excess_pp"]
    r_noml = out["leveraging_noml"]["excess_pp"]
    out["organic_share_bound"] = 1 - r_noml / r_all if r_all else None
    print(f"   excess retained after stripping method papers: "
          f"{r_noml/r_all*100:.0f} per cent of the full excess")

    print("\nB. floor with and without leveraging in the basket")
    for lab, d in (("full basket", basket_any), ("no leveraging", basket_noLev)):
        f0 = rate(d, ("all", "pre")); ft = rate(d, ("all", "post"))
        out[f"floor_{lab.replace(' ', '_')}"] = {"f0_pct": f0, "ft_pct": ft,
                                                 "floor_pp": ft - f0}
        print(f"   {lab:14s}: f0 {f0:.2f}%, f_t {ft:.2f}%, floor {ft-f0:.2f} pp")

    out["pre2022_doc_freq_pct"] = {w: per_word_pre[w] / npre * 100 for w in FULL}
    json.dump(out, open(os.path.join(DATA, "p5_leveraging.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
