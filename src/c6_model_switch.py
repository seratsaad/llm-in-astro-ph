#!/usr/bin/env python3
"""
C6 -- Is the 2026 dip in the LLM-marker basket a real DECLINE, or a MIGRATION to
different vocabulary (consistent with a shift in the dominant source model / author
awareness)?

Method: compute quarterly document-frequency trajectories for a broad set of LLM-style
words, cluster them by their RECENT trend (rising vs collapsing), and check whether the
basket dip is concentrated in the 'collapsing' cluster while the 'rising' cluster keeps
climbing. If so, the footprint is redistributing, not shrinking.

Caveat baked into the framing: we cannot prove which *model* without labeled generations;
the early-fame collapsing words (delve, tapestry, testament) are the GPT-3.5/4-era tells,
so a generation shift is *consistent* with the pattern -- but author adaptation to the
'delve' discourse produces the same signature. Both imply detection decays, footprint persists.
"""
import json, os, re, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
TOK = re.compile(r"[a-z]+")

MARKERS = ["delve", "delves", "delving", "tapestry", "testament", "boasts",
           "intricate", "intricacies", "meticulous", "meticulously", "realm",
           "realms", "underscore", "underscores", "underscoring", "showcasing",
           "showcase", "showcases", "pivotal", "leveraging", "nuanced", "notably",
           "comprehensive", "encompass", "encompasses", "encompassing", "myriad",
           "seamless", "seamlessly", "multifaceted", "highlighting", "offering",
           "aligns", "aligning", "garner", "elucidate", "unravel", "crucial",
           "harnessing", "advancing", "endeavor"]

QSTART, QEND = (2022, 1), (2026, 2)  # drop sparse 2026Q3

def qkey(pub):
    y = int(pub[:4]); m = int(pub[5:7]); return y, (m - 1) // 3 + 1

def qrange():
    y, q = QSTART; out = []
    while (y, q) <= QEND:
        out.append((y, q)); q += 1
        if q == 5: y += 1; q = 1
    return out

def main():
    qs = qrange()
    qidx = {qq: i for i, qq in enumerate(qs)}
    tot = np.zeros(len(qs))
    cnt = {w: np.zeros(len(qs)) for w in MARKERS}
    seen = set()
    for line in open(INP):
        r = json.loads(line); b = r["id"].split("v")[0]
        if b in seen: continue
        seen.add(b)
        pub = r["published"]
        if not pub or len(pub) < 7: continue
        yq = qkey(pub)
        if yq not in qidx: continue
        i = qidx[yq]
        toks = set(TOK.findall((r.get("abstract") or "").lower()))
        tot[i] += 1
        for w in MARKERS:
            if w in toks: cnt[w][i] += 1

    freq = {w: cnt[w] / tot for w in MARKERS}  # doc-frequency per quarter
    # classify each word by recent trend: mean(2025Q1..2026Q2) vs peak-quarter value,
    # AND slope over last 6 quarters.
    # Cluster by PEAK QUARTER: 'early tells' peaked by 2024Q4 (the GPT-3.5/4-era words
    # that went viral and were edited out); 'late tells' peaked 2025 or later.
    # Only classify words with a real signal (peak doc-freq above a floor).
    rows = []
    for w in MARKERS:
        f = freq[w]
        peak_i = int(np.argmax(f))
        if f[peak_i] < 0.001:      # too rare to trust -> drop from clusters
            cluster = "rare"
        else:
            cluster = "early" if qs[peak_i] <= (2024, 4) else "late"
        rows.append({"word": w, "peak_q": f"{qs[peak_i][0]}Q{qs[peak_i][1]}",
                     "peak_val": f[peak_i], "recent": f[len(qs)-4:].mean(),
                     "cluster": cluster})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(DATA, "c6_word_clusters.csv"), index=False)

    # aggregate incidence per cluster (sum of doc counts / total) per quarter
    def agg(words):
        s = np.zeros(len(qs))
        for w in words: s += cnt[w]
        return s / tot
    collapsing = cdf[cdf.cluster == "early"].word.tolist()
    rising = cdf[cdf.cluster == "late"].word.tolist()
    agg_all = agg(MARKERS); agg_col = agg(collapsing); agg_ris = agg(rising)
    series = pd.DataFrame({
        "yq": [y + (q - 1) / 4 for (y, q) in qs],
        "label": [f"{y}Q{q}" for (y, q) in qs],
        "total": tot, "all_markers": agg_all,
        "collapsing": agg_col, "rising": agg_ris})
    series.to_csv(os.path.join(DATA, "c6_cluster_series.csv"), index=False)

    print(f"EARLY tells (peaked <=2024Q4, n={len(collapsing)}):")
    print("  " + ", ".join(sorted(collapsing)))
    print(f"LATE tells (peaked >=2025Q1, n={len(rising)}):")
    print("  " + ", ".join(sorted(rising)))
    # how much of the 2025->2026 basket drop is from the collapsing cluster?
    i25 = qidx[(2025, 2)]; i26 = qidx[(2026, 2)]
    d_all = agg_all[i25] - agg_all[i26]
    d_col = agg_col[i25] - agg_col[i26]
    d_ris = agg_ris[i25] - agg_ris[i26]
    print(f"\n2025Q2 -> 2026Q2 change in incidence:")
    print(f"  all markers : {agg_all[i25]*100:.2f}% -> {agg_all[i26]*100:.2f}%  (Δ {d_all*100:+.2f}pp)")
    print(f"  collapsing  : {agg_col[i25]*100:.2f}% -> {agg_col[i26]*100:.2f}%  (Δ {d_col*100:+.2f}pp)")
    print(f"  rising      : {agg_ris[i25]*100:.2f}% -> {agg_ris[i26]*100:.2f}%  (Δ {d_ris*100:+.2f}pp)")
    if abs(d_all) > 1e-9:
        print(f"  -> collapsing cluster explains {100*d_col/d_all:.0f}% of the basket change; "
              f"rising cluster moved {d_ris*100:+.2f}pp")
    print("wrote c6_word_clusters.csv, c6_cluster_series.csv")

if __name__ == "__main__":
    main()
