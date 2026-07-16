# How much of the astronomy literature is written with a chatbot?

Code and data behind the Astrobites "Beyond" post measuring the footprint of large
language models (LLMs) in the astronomy literature. Everything is reproducible from
public arXiv and NASA ADS data.

The write-up is in [`POST.md`](POST.md) (and [`POST.pdf`](POST.pdf)); the figures are in
[`figs/`](figs/). A journal-style paper version is in [`paper/`](paper/)
([`paper/main.pdf`](paper/main.pdf)).

## What the analysis does

- Counts LLM "marker words" (delve, underscore, intricate, ...) in 200,547 astro-ph
  abstracts (2015–2026) and compares them to neutral control words.
- Discovers astro-ph's own excess vocabulary from the data (telescopes vs. style words).
- Estimates a lower bound on the fraction of LLM-touched abstracts.
- Measures explicit LLM disclosure via NASA ADS acknowledgments.
- Audits arXiv LaTeX source for pasted-chatbot leftovers.
- Checks cited arXiv IDs and DOIs for fabricated ("hallucinated") citations.
- Breaks the signal down by subfield and by author country, and tracks hype vs. hedging.

## Layout

```
src/    all analysis + plotting scripts
data/   derived data the plot scripts read (the 300 MB raw abstract dump is NOT committed)
figs/   output figures (PNG)
POST.md the post
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Some steps use NASA ADS. Get a free token at https://ui.adsabs.harvard.edu (Account →
API Token) and save it:

```bash
mkdir -p ~/.ads && printf '%s' 'YOUR_ADS_TOKEN' > ~/.ads/dev_key && chmod 600 ~/.ads/dev_key
```

## Reproduce the figures (fast path)

The small derived data is committed, so you can rebuild every figure without
re-harvesting:

```bash
python src/plots.py            # figures 1–6
python src/plots2.py           # figures 7–8
python src/plots3.py           # figures 10–12
python src/plots_ngram.py      # n-gram viewer, bigram discovery, co-occurrence
python src/analyze_source.py --fig   # figure 9 (detection ladder)
```

## Reproduce the data from scratch

Run in order. Steps that hit arXiv/ADS are rate-limited and polite; the abstract harvest
and the source scan each take a while.

```bash
# 1. Harvest all astro-ph abstracts via the arXiv API  (~300 MB, ~15 min)
python src/harvest_arxiv.py

# 2. Per-year LLM disclosure counts from NASA ADS  (needs token)
python src/harvest_ads.py

# 3. Tokenise abstracts -> marker/control frequencies, excess-word table
python src/process_abstracts.py

# 4. Lower-bound estimate of the LLM-touched fraction
python src/alpha_estimate.py

# 5. Follow-up cuts
python src/c5_subfield.py       # marker rate by subfield
python src/c4_geography.py      # marker + disclosure by country (needs token)
python src/c6_model_switch.py   # marker-vocabulary trajectories
python src/c7_hype_hedge.py     # promotional vs hedging vocabulary
python src/ngrams.py            # two-word phrase + co-occurrence analysis
python src/rigor_stats.py       # uncertainties, robustness variants, significance tests

# 6. Source-leak audit: download arXiv LaTeX source and scan it  (~1 hr, downloads)
python src/harvest_source.py
python src/analyze_source.py    # rates + detection-ladder figure

# 7. Hallucinated-citation audit
python src/c3_citations.py --collect
python src/c3_citations.py --verify
```

## Rebuild the post document

```bash
python src/build_pdf.py         # POST.md -> POST.html (then print to PDF)
node   src/build_docx.js        # POST.md -> POST.docx   (needs: npm install docx)
```

## Data sources

- arXiv API — https://info.arxiv.org/help/api/
- NASA Astrophysics Data System (ADS) — https://ui.adsabs.harvard.edu
- Some tools this work builds on, including the astro-ph knowledge graph, are shared by
  Yuan-Sen Ting at https://github.com/tingyuansen

Marker-word method follows Liang et al. (2024), Gray (2025), and Kobak et al. (2024).

## Note

No credentials are stored in this repository. The ADS token is read from `~/.ads/dev_key`
at runtime and is never committed.
