# lis501-frt-analysis

A cleaned Reddit thread corpus (2008–2019) plus utilities for LDA, visualization, and idea-relation analysis.

## Key data artifacts
- `corpus_threads_2008_2019_clean.jsonl` (25 GB): full cleaned thread documents (4,690,333 threads).
- `cached_corpus_2008_2019_clean.jsonl` (17 MB): small cached subset used for quick tests.
- `lda_2008_2019/`: full combined LDA outputs
  - `topics.json`: top words per topic
  - `doc_topics.jsonl`: topic distributions per thread
  - `corpus_threads_2008_2019_clean.jsonl`: small cached corpus copy
- `figures/2008_2019_full/`: visuals from the full corpus
  - `document_length_distribution.png` and `document_length_stats.json` (all 4.69M docs)
  - `topics_pyldavis.html` (pyLDAvis on a 5k sample for tractability)
- `figures/relations_2008_2019.png` + `figures/relations_2008_2019_summary.json`: cooccurrence/prevalence relation plots (friendship/tryst/arms-race/head-to-head) using topic-weight presence ≥0.10.

## Core scripts
- `scripts/make_visualizations.py`: streams the corpus to build the word-count histogram/stats; optionally fits a sampled LDA for pyLDAvis.
- `scripts/make_relation_plots.py`: computes topic cooccurrence PMI + yearly-prevalence correlation for selected pairs and plots relation quadrants.
- Other helpers: `run_lda.py`, `clean_corpus.py`, `merge_corpus.py`, `run_month_pipeline.sh`, etc., for end-to-end preprocessing/lda.

## Common commands
- Full-corpus length distribution (streamed) + sampled pyLDAvis:
  ```bash
  python3 scripts/make_visualizations.py \
    --corpus corpus_threads_2008_2019_clean.jsonl \
    --output-dir figures/2008_2019_full \
    --lda-max-docs 5000 \
    --clip-pct 99.5
  ```
  Outputs: `document_length_distribution.png`, `document_length_stats.json`, `topics_pyldavis.html`.

- Topic relation plot (defaults: threshold 0.10, pairs chosen to cover all four relation types):
  ```bash
  python3 scripts/make_relation_plots.py \
    --doc-topics lda_2008_2019/doc_topics.jsonl \
    --topics-json lda_2008_2019/topics.json \
    --output-dir figures
  ```
  Outputs: `relations_2008_2019.png`, `relations_2008_2019_summary.json`.

## Notes
- PyLDAvis is generated on a sampled subset (default 5k docs) to stay memory-friendly. Increase `--lda-max-docs` if you need a denser view.
- Relation types follow Chuang et al. (ACL 2017): friendship (corr+, PMI+), tryst (corr−, PMI+), arms-race (corr+, PMI−), head-to-head (corr−, PMI−). Presence is defined by topic weight ≥ `--presence-threshold` (default 0.10).
