#!/usr/bin/env python3
"""
Second round of referee checks (independent-referee points).
A: frequency-matched placebo basket (specificity of the estimator).
B: structural-break test -- step at 2022Q4 (LLM) vs smooth ramp (jargon), per word.
C: abstract-length control -- per-token marker rate and mean length by year.
D: unsupervised recovery -- do imported basket words rank in the ratio tail, and
   how many non-marker words clear the same threshold.
G: significance of the detrended tone correlation.
Writes data/referee_checks2.json.
"""
import json, os, re, math, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
ABS = os.path.join(DATA, "astroph_abstracts.jsonl")

BASKET = {"delve","delves","delving","underscore","underscores","underscoring",
    "intricate","intricacies","showcasing","showcase","showcases","showcased",
    "boasts","tapestry","pivotal","meticulous","meticulously","nuanced",
    "garner","garners","garnered","multifaceted","commendable","noteworthy",
    "myriad","plethora","testament","encompassing","seamless","seamlessly",
    "elucidate","elucidating","unravel","unraveling","unravelling",
    "realm","realms","leveraging"}
INSTR = {"nircam","miri","nirspec","jwst","desi","xrism","mrs","ixpe","lhaaso",
    "prism","kagra","noon","euclid","ligo","virgo","webb","spectroscopic"}
CARRIERS = ["leveraging","underscore","highlighting","pivotal"]   # top style carriers
INSTR_T = ["jwst","nircam","desi"]                                # instruments for contrast

def tok(s): return re.findall(r"[a-z]+", s.lower())

# ---------- single corpus pass: collect what A, B, C, D need ----------
def corpus_pass():
    # per-quarter counts for target words + basket-any; per-year length + basket-any; per-basket placebo membership
    targets = CARRIERS + INSTR_T
    q_tot = collections.Counter(); q_hit = {w: collections.Counter() for w in targets + ["_basket"]}
    y_len = collections.defaultdict(list); y_tot = collections.Counter()
    y_basket_docs = collections.Counter(); y_basket_tokens = collections.Counter(); y_tokens = collections.Counter()
    # placebo: build candidate pool + baskets after we know vocab; do membership in a second structure
    recs = []   # keep (year, quarter, tokenset) lightweight -> too big; instead accumulate needed counts now
    # placebo baskets need to be known before the pass; build from word_docfreq
    wf = pd.read_parquet(os.path.join(DATA, "word_docfreq.parquet"))
    pool = wf[(wf.base_freq > 0.0003) & (wf.base_freq < 0.005) & (wf.word.str.len() >= 4)]
    pool = pool[~pool.word.isin(BASKET | INSTR)]
    pool = pool[~pool.word.str.contains(r"[^a-z]")]
    cand = pool.word.tolist()
    rng = np.random.default_rng(7)
    K = 50
    placebos = [set(rng.choice(cand, size=38, replace=False)) for _ in range(K)]
    plac_hit_rec = np.zeros(K); plac_n_rec = 0
    plac_hit_base = np.zeros(K); plac_n_base = 0

    for line in open(ABS):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7]); qk = y + ((m - 1) // 3) / 4.0
        toks = tok(r["abstract"]); ts = set(toks); L = len(toks)
        basket_hit = bool(ts & BASKET)
        # quarterly (2018-2025) for targets + basket
        if 2018 <= y <= 2025:
            q_tot[qk] += 1
            if basket_hit: q_hit["_basket"][qk] += 1
            for w in targets:
                if w in ts: q_hit[w][qk] += 1
        # yearly length + per-token basket rate
        y_tot[y] += 1; y_len[y].append(L); y_tokens[y] += L
        if basket_hit: y_basket_docs[y] += 1
        y_basket_tokens[y] += sum(1 for t in toks if t in BASKET)
        # placebo membership for 2025 (recent) and 2018-2021 (base)
        if y == 2025:
            plac_n_rec += 1
            for k, pb in enumerate(placebos):
                if ts & pb: plac_hit_rec[k] += 1
        elif 2018 <= y <= 2021:
            plac_n_base += 1
            for k, pb in enumerate(placebos):
                if ts & pb: plac_hit_base[k] += 1
    return dict(q_tot=q_tot, q_hit=q_hit, y_tot=y_tot, y_len=y_len, y_tokens=y_tokens,
                y_basket_docs=y_basket_docs, y_basket_tokens=y_basket_tokens,
                plac_rec=plac_hit_rec/plac_n_rec*100, plac_base=plac_hit_base/plac_n_base*100)

def step_test(qx, rate):
    """Fit rate ~ a + b*qx  vs  rate ~ a + b*qx + c*step(qx>=2022.75). Return step size, F p-value."""
    qx = np.asarray(qx, float); y = np.asarray(rate, float)
    X1 = np.column_stack([np.ones_like(qx), qx])
    step = (qx >= 2022.75).astype(float)
    X2 = np.column_stack([np.ones_like(qx), qx, step])
    b1, r1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b2, r2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    rss1 = float(((y - X1 @ b1) ** 2).sum()); rss2 = float(((y - X2 @ b2) ** 2).sum())
    n = len(y); F = ((rss1 - rss2) / 1) / (rss2 / (n - 3))
    # F(1, n-3) survival
    from math import lgamma, log, exp
    def betacf(a, b, x):
        MAXIT=200; EPS=3e-7; FPMIN=1e-30
        qab=a+b; qap=a+1; qam=a-1; c=1; d=1-qab*x/qap
        if abs(d)<FPMIN: d=FPMIN
        d=1/d; h=d
        for mm in range(1,MAXIT):
            m2=2*mm
            aa=mm*(b-mm)*x/((qam+m2)*(a+m2)); d=1+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1/d; h*=d*c
            aa=-(a+mm)*(qab+mm)*x/((a+m2)*(qap+m2)); d=1+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1/d; de=d*c; h*=de
            if abs(de-1)<EPS: break
        return h
    def betai(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        bt=exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log(1-x))
        if x<(a+1)/(a+b+2): return bt*betacf(a,b,x)/a
        return 1-bt*betacf(b,a,1-x)/b
    d1, d2 = 1, n-3
    p = betai(d2/2, d1/2, d2/(d2+d1*F)) if F>0 else 1.0
    return {"step_pp": float(b2[2]*100), "F": float(F), "p": float(p),
            "rss_ratio": rss2/rss1}

def main():
    cp = corpus_pass()
    out = {}

    # ---- A: placebo ----
    alpha_plac = cp["plac_rec"] - cp["plac_base"]
    out["A_placebo"] = {"n_baskets": len(alpha_plac),
        "placebo_alpha_mean_pp": float(alpha_plac.mean()),
        "placebo_alpha_sd_pp": float(alpha_plac.std()),
        "placebo_alpha_max_pp": float(alpha_plac.max()),
        "observed_marker_alpha_pp": 4.3}

    # ---- B: structural break ----
    qs = sorted(cp["q_tot"])
    def series(w):
        return [100 * cp["q_hit"][w][q] / cp["q_tot"][q] for q in qs]
    out["B_stepvsramp"] = {}
    for w in ["_basket"] + CARRIERS + INSTR_T:
        out["B_stepvsramp"][w] = step_test(qs, series(w))

    # ---- C: length control ----
    yrs = sorted(cp["y_tot"])
    out["C_length"] = {"mean_len_by_year": {str(y): float(np.mean(cp["y_len"][y])) for y in yrs},
        "basket_per1000tok_by_year": {str(y): 1000 * cp["y_basket_tokens"][y] / cp["y_tokens"][y] for y in yrs},
        "basket_docpct_by_year": {str(y): 100 * cp["y_basket_docs"][y] / cp["y_tot"][y] for y in yrs}}

    # ---- D: unsupervised recovery ----
    wf = pd.read_parquet(os.path.join(DATA, "word_docfreq.parquet"))
    tail = wf[(wf.base_freq > 0.0005) & (wf.ratio >= 5)].sort_values("ratio", ascending=False)
    tailwords = set(tail.word)
    basket_in_wf = wf[wf.word.isin(BASKET)]
    basket_in_tail = sorted(set(tail.word) & BASKET)
    ranked = wf[wf.base_freq > 0.0005].sort_values("ratio", ascending=False).reset_index(drop=True)
    ranks = {w: int(ranked.index[ranked.word == w][0]) + 1 for w in BASKET if (ranked.word == w).any()}
    top50 = set(ranked.head(50).word)
    out["D_recovery"] = {"tail_size": int(len(tail)),
        "basket_words_in_tail": len(basket_in_tail),
        "basket_or_instr_in_tail": int(len(set(tail.word) & (BASKET | INSTR))),
        "non_marker_non_instr_in_tail": int(len(set(tail.word) - BASKET - INSTR)),
        "basket_in_top50": int(len(top50 & BASKET)),
        "instr_in_top50": int(len(top50 & INSTR)),
        "basket_word_ranks": ranks,
        "example_tail_style": basket_in_tail[:12]}

    # ---- G: tone significance ----
    rc = json.load(open(os.path.join(DATA, "referee_checks.json")))["R4_hype_detrend"]
    def r_to_p(r, n):
        if abs(r) >= 1: return 0.0
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # two-sided via incomplete beta
        from math import lgamma, log, exp
        df = n - 2; x = df / (df + t * t)
        def betacf(a,b,x):
            c=1; d=1-(a+b)*x/(a+1); d=1/(d if abs(d)>1e-30 else 1e-30); h=d
            for m in range(1,200):
                m2=2*m; aa=m*(b-m)*x/((a-1+m2)*(a+m2)); d=1/((1+aa*d) if abs(1+aa*d)>1e-30 else 1e-30); c=1+aa/c; h*=d*c
                aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+1+m2)); d=1/((1+aa*d) if abs(1+aa*d)>1e-30 else 1e-30); c=1+aa/c; de=d*c; h*=de
                if abs(de-1)<3e-7: break
            return h
        bt=exp(lgamma(a:=df/2)+0)  # placeholder
        # use regularized incomplete beta directly
        a=df/2; b=0.5
        btln=lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log(1-x)
        I = exp(btln)*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-exp(btln)*betacf(b,a,1-x)/b
        return float(I)
    out["G_tone"] = {"firstdiff_r": rc["hype_firstdiff_r"], "firstdiff_n": rc["n_years"]-1,
        "firstdiff_p": r_to_p(rc["hype_firstdiff_r"], rc["n_years"]-1),
        "detrended_r": rc["hype_detrended_r"],
        "detrended_p": r_to_p(rc["hype_detrended_r"], rc["n_years"])}

    json.dump(out, open(os.path.join(DATA, "referee_checks2.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
