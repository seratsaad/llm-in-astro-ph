#!/usr/bin/env python3
"""
C7 -- Does LLM adoption change the RHETORIC of astronomy abstracts?
Track promotional/hype vocabulary vs hedging vocabulary per year, and compare the
trend to the LLM marker basket. Everyone measures LLM 'style' words; the hype-vs-hedge
asymmetry (are claims getting louder / less cautious?) is a named gap in the literature.
"""
import json, os, re, collections
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
TOK = re.compile(r"[a-z][a-z-]*")

# Strong promotional adjectives/adverbs (deliberately NOT science-neutral words like
# 'significant'/'novel'/'first', which rise for legitimate reasons).
HYPE = {"unprecedented", "groundbreaking", "remarkable", "remarkably", "striking",
        "strikingly", "compelling", "intriguing", "intriguingly", "exciting",
        "surprising", "surprisingly", "dramatic", "dramatically", "revolutionary",
        "breakthrough", "cutting-edge", "state-of-the-art", "paradigm-shifting",
        "game-changing", "exceptional", "extraordinary", "spectacular", "profound"}
# Hedging / epistemic caution.
HEDGE = {"may", "might", "could", "suggest", "suggests", "suggesting", "indicate",
         "indicates", "appear", "appears", "seem", "seems", "likely", "possibly",
         "potentially", "presumably", "tentatively", "arguably", "perhaps",
         "plausibly", "conceivably"}

def main():
    y_tot = collections.Counter()
    y_hype_doc = collections.Counter(); y_hedge_doc = collections.Counter()
    y_hype_n = collections.Counter(); y_hedge_n = collections.Counter()
    y_words = collections.Counter()
    seen = set()
    for line in open(INP):
        r = json.loads(line); b = r["id"].split("v")[0]
        if b in seen: continue
        seen.add(b)
        pub = r["published"]
        if not pub or len(pub) < 7: continue
        y = int(pub[:4])
        if y < 2015 or y > 2026: continue
        toks = TOK.findall((r.get("abstract") or "").lower())
        tset = set(toks)
        y_tot[y] += 1; y_words[y] += len(toks)
        nh = sum(1 for t in toks if t in HYPE)
        ng = sum(1 for t in toks if t in HEDGE)
        y_hype_n[y] += nh; y_hedge_n[y] += ng
        if tset & HYPE: y_hype_doc[y] += 1
        if tset & HEDGE: y_hedge_doc[y] += 1

    rows = []
    for y in range(2015, 2027):
        if not y_tot[y]: continue
        rows.append({"year": y, "n": y_tot[y],
                     "hype_docpct": 100*y_hype_doc[y]/y_tot[y],
                     "hedge_docpct": 100*y_hedge_doc[y]/y_tot[y],
                     "hype_per1k": 1000*y_hype_n[y]/y_words[y],
                     "hedge_per1k": 1000*y_hedge_n[y]/y_words[y]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "c7_hype_hedge.csv"), index=False)
    print(df.to_string(index=False))
    # correlation of hype with marker basket
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    basket = {int(y): v["rate"]*100 for y, v in al["by_year"].items()}
    df["basket"] = df.year.map(basket)
    print("\ncorr(hype_per1k, marker_basket) =",
          round(df[["hype_per1k", "basket"]].corr().iloc[0, 1], 3))
    print("corr(hedge_per1k, marker_basket) =",
          round(df[["hedge_per1k", "basket"]].corr().iloc[0, 1], 3))
    # pre vs post ChatGPT
    pre = df[df.year.between(2018, 2021)]; post = df[df.year.between(2024, 2025)]
    print(f"\nhype_per1k: pre(2018-21)={pre.hype_per1k.mean():.3f}  post(2024-25)={post.hype_per1k.mean():.3f}  "
          f"(+{100*(post.hype_per1k.mean()/pre.hype_per1k.mean()-1):.0f}%)")
    print(f"hedge_per1k: pre={pre.hedge_per1k.mean():.3f}  post={post.hedge_per1k.mean():.3f}  "
          f"({100*(post.hedge_per1k.mean()/pre.hedge_per1k.mean()-1):+.0f}%)")

if __name__ == "__main__":
    main()
