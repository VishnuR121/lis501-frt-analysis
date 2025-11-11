# lis501-frt-analysis

## Project structure

```
.
├── years/
│   ├── 2008/
│   │   ├── comments/
│   │   │   ├── raw/       # Monthly Politosphere dumps (.bz2 + .jsonl)
│   │   │   ├── threads/   # Reconstructed thread trees
│   │   │   └── corpus/    # Thread-level documents (monthly + yearly)
│   │   └── lda/
│   │       ├── monthly/   # Per-month LDA outputs
│   │       └── yearly/    # Full-year topic models
│   ├── 2009/              # Same structure (placeholders for now)
│   ├── 2010/
│   ├── 2011/
│   └── 2012/
├── notebooks/             # Exploratory analysis (empty placeholder for now)
├── scripts/               # Data preparation + topic modeling helpers
└── src/                   # Future library / modeling code
```

## 1. Acquire the Reddit dumps

The Politosphere dataset is hosted on Zenodo. Grab any month you need (example: January 2008):

```bash
curl -L 'https://zenodo.org/records/5851729/files/comments_2008-01.bz2?download=1' \
  -o years/2008/comments/raw/comments_2008-01.bz2
bunzip2 -k years/2008/comments/raw/comments_2008-01.bz2
mv years/2008/comments/raw/comments_2008-01 \
   years/2008/comments/raw/comments_2008-01.jsonl
```

> Each line in `.jsonl` is a single comment, containing `link_id` (submission id) and `parent_id` (either the submission or another comment). Those two columns are all we need to rebuild the full conversation context.

## 2. Reconstruct full threads

`reconstruct_threads.py` ties the monthly dump back into hierarchical conversations, yielding one JSON record per submission (`link_id`):

```bash
python3 scripts/reconstruct_threads.py \
  years/2008/comments/raw/comments_2008-01.jsonl \
  years/2008/comments/threads/threads_2008-01.jsonl \
  --min-comments 5          # optional quality filter
  --max-threads 3           # optional debug limit, drop for full run
```

Output schema per line:

- `link_id`, `subreddit`
- `comment_count`, `root_count`, `created_utc_min/max`, `orphan_comments`
- `roots`: array of nested comment trees; each node carries `author`, `body`, `body_cleaned`, `net_votes` (upvotes minus downvotes), `controversiality`, timestamps, and its children.

During reconstruction we drop comments whose body is `[deleted]`, `[removed]`, or empty. Their children slide up in the tree so you only see living content while preserving reply chains. This structure lets us traverse a thread depth-first, isolate quoted spans, and compute discourse strategies analogous to the newspaper workflow from the paper.

### Quick thread previews

To inspect any reconstructed conversation without loading it into a notebook, use `view_thread.py`:

```bash
# Show the first thread in the file
python3 scripts/view_thread.py years/2008/comments/threads/threads_2008-01.jsonl --index 0

# Or target a specific submission id
python3 scripts/view_thread.py years/2008/comments/threads/threads_2008-01.jsonl --link-id t3_648iy

# Save to a text file instead of printing
python3 scripts/view_thread.py years/2008/comments/threads/threads_2008-01.jsonl \
  --link-id t3_648iy \
  --output years/2008/comments/threads/t3_648iy.txt
```

The script prints a readable tree with author, net votes, timestamps, and truncated bodies so you can quickly verify how the comments connect.

### Export every thread to text files

When you want a human-readable file per submission, run:

```bash
python3 scripts/export_threads_text.py \
  years/2008/comments/threads/threads_2008-01.jsonl \
  years/2008/comments/threads/text_dumps/threads_2008-01
```

Each thread becomes `.../text_dumps/000123_t3_abcd12.txt`. Use `--limit` for a quick smoke test or `--max-body-chars` to control truncation.

### Build thread-level documents for LDA

When you are ready for topic modeling, convert each reconstructed thread into a single document (concatenated comment text). Threads that do not meet the minimum size threshold are skipped:

```bash
python3 scripts/build_thread_corpus.py \
  years/2008/comments/threads/threads_2008-01.jsonl \
  years/2008/comments/corpus/corpus_threads_2008-01.jsonl \
  --min-comments 5        # default; adjust if needed
```

The resulting JSONL stores `link_id`, `subreddit`, `comment_count`, timestamps, and the aggregated `text` field ready for vectorization/LDA.

### Fit LDA topics

Install scikit-learn if you have not already (`python3 -m pip install --user scikit-learn`), then run:

```bash
python3 scripts/run_lda.py \
  years/2008/comments/corpus/corpus_threads_2008-01.jsonl \
  years/2008/lda/monthly/2008-01 \
  --num-topics 15 \
  --min-df 5 \
  --max-docs 2000 \
  --learning-method online \
  --batch-size 1024 \
  --max-iter 10
```

Outputs:

- `years/2008/lda/monthly/2008-01/topics.json`: top words + weights per topic.
- `years/2008/lda/monthly/2008-01/doc_topics.jsonl`: one line per thread with its topic distribution (useful for subreddit/month aggregations and visualizations).

### Scaling to yearly corpora

1. Build monthly corpora for every month in a year (see above), then concatenate them:
   ```bash
   cat years/2008/comments/corpus/corpus_threads_2008-*.jsonl \
     > years/2008/comments/corpus/corpus_threads_2008.jsonl
   ```
   (33,981 threads survived the ≥5 comment filter for 2008.)
2. Run a larger LDA job (example: 50 topics over the entire year):
   ```bash
   python3 scripts/run_lda.py \
     years/2008/comments/corpus/corpus_threads_2008.jsonl \
     years/2008/lda/yearly/2008 \
     --num-topics 50 \
     --min-df 50 \
     --max-features 20000 \
     --learning-method online \
     --batch-size 1024 \
     --max-iter 10
   ```
   This took ~8.5 minutes on the CLI host and produced yearly topics in `years/2008/lda/yearly/2008/`.

## 3. Next steps toward the paper reproduction

1. **Thread sampling:** Decide how many monthly dumps (2008–2019) you need and whether to filter by subreddits (`politics`, `politicstalk`, etc.).
2. **Quote/strategy extraction:** Adapt the paper’s quote matching heuristics to Reddit replies (e.g., `>` quotes, lexical overlap).
3. **Stance modelling:** Define stance labels (support/attack) per comment edge, and train models mirroring the paper’s approach.
4. **Politosphere integration:** Incorporate metadata (user ideology embeddings, subreddit info) from the upstream repo to enrich features.
5. **Evaluation:** Recreate the paper’s metrics (influence of strategies on cascade reach) using the Reddit conversation graphs built here.

Document any modelling experiments inside `notebooks/` or `src/` as they materialize.
