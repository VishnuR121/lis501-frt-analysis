# lis501-frt-analysis

Recreating *Fighting Fire with Fire: Modelling the Influence of Discourse Strategies in News* (Bosc et al., ACL 2017) with Reddit political discussions instead of news articles. The pipeline below ingests the Politosphere monthly Reddit dumps, rebuilds entire comment threads, and prepares them for downstream stance/quote strategy analyses.

## Project structure

```
.
├── data/                # Ignored by git; holds raw/interim/processed artifacts
├── notebooks/           # Exploratory analysis (empty placeholder for now)
├── scripts/             # Data preparation helpers
└── src/                 # Future library / modeling code
```

## 1. Acquire the Reddit dumps

The Politosphere dataset is hosted on Zenodo. Grab any month you need (example: January 2008):

```bash
curl -L 'https://zenodo.org/records/5851729/files/comments_2008-01.bz2?download=1' \
  -o data/raw/comments_2008-01.bz2
bunzip2 -k data/raw/comments_2008-01.bz2
mv data/raw/comments_2008-01 data/raw/comments_2008-01.jsonl
```

> Each line in `.jsonl` is a single comment, containing `link_id` (submission id) and `parent_id` (either the submission or another comment). Those two columns are all we need to rebuild the full conversation context.

## 2. Reconstruct full threads

`reconstruct_threads.py` ties the monthly dump back into hierarchical conversations, yielding one JSON record per submission (`link_id`):

```bash
python3 scripts/reconstruct_threads.py \
  data/raw/comments_2008-01.jsonl \
  data/interim/threads_2008-01.jsonl \
  --min-comments 5          # optional quality filter
  --max-threads 3           # optional debug limit, drop for full run
```

Output schema per line:

- `link_id`, `subreddit`
- `comment_count`, `root_count`, `created_utc_min/max`, `orphan_comments`
- `roots`: array of nested comment trees; each node carries `author`, `body`, `body_cleaned`, `score`, `controversiality`, timestamps, and its children.

This structure lets us traverse a thread depth-first, isolate quoted spans, and compute discourse strategies analogous to the newspaper workflow from the paper.

## 3. Next steps toward the paper reproduction

1. **Thread sampling:** Decide how many monthly dumps (2008–2019) you need and whether to filter by subreddits (`politics`, `politicstalk`, etc.).
2. **Quote/strategy extraction:** Adapt the paper’s quote matching heuristics to Reddit replies (e.g., `>` quotes, lexical overlap).
3. **Stance modelling:** Define stance labels (support/attack) per comment edge, and train models mirroring the paper’s approach.
4. **Politosphere integration:** Incorporate metadata (user ideology embeddings, subreddit info) from the upstream repo to enrich features.
5. **Evaluation:** Recreate the paper’s metrics (influence of strategies on cascade reach) using the Reddit conversation graphs built here.

Document any modelling experiments inside `notebooks/` or `src/` as they materialize.
