#!/usr/bin/env python3
"""
Train an LDA topic model on thread-level documents and export topic summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an LDA model to thread-level documents."
    )
    parser.add_argument(
        "corpus_path",
        type=Path,
        help="Path to corpus JSONL produced by build_thread_corpus.py.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to store topics and document-topic weights.",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of LDA topics.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20_000,
        help="Vocabulary size cap for the CountVectorizer.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Minimum document frequency for tokens (passed to CountVectorizer).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents loaded from the corpus (debugging).",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=15,
        help="Number of representative words to save per topic.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_corpus(
    path: Path,
    max_docs: int | None = None,
) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    meta: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "").strip()
            if not text:
                continue
            texts.append(text)
            meta.append(
                {
                    "link_id": record["link_id"],
                    "subreddit": record["subreddit"],
                    "comment_count": record["comment_count"],
                    "created_utc_min": record["created_utc_min"],
                    "created_utc_max": record["created_utc_max"],
                }
            )
            if max_docs and len(texts) >= max_docs:
                break
    if not texts:
        raise ValueError("No documents loaded from corpus.")
    return texts, meta


def train_lda(
    texts: List[str],
    *,
    num_topics: int,
    max_features: int,
    min_df: int,
    random_state: int,
) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
    )
    doc_term = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method="batch",
        max_iter=20,
        random_state=random_state,
    )
    lda.fit(doc_term)
    doc_topic = lda.transform(doc_term)
    return lda, vectorizer, doc_topic


def save_topics(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_words: int,
    output_path: Path,
) -> None:
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:top_words]
        topics.append(
            {
                "topic_id": topic_idx,
                "top_words": feature_names[top_indices].tolist(),
                "weights": topic[top_indices].tolist(),
            }
        )
    output_path.write_text(json.dumps({"topics": topics}, indent=2) + "\n", encoding="utf-8")


def save_doc_topics(
    doc_topic: np.ndarray,
    meta: List[dict],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for weights, info in zip(doc_topic, meta):
            record = {
                **info,
                "topic_distribution": weights.tolist(),
                "top_topic": int(np.argmax(weights)),
                "top_topic_score": float(np.max(weights)),
            }
            handle.write(json.dumps(record))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    if not args.corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")

    texts, meta = load_corpus(args.corpus_path, max_docs=args.max_docs)
    print(f"Loaded {len(texts):,} documents for LDA training.", flush=True)

    lda, vectorizer, doc_topic = train_lda(
        texts,
        num_topics=args.num_topics,
        max_features=args.max_features,
        min_df=args.min_df,
        random_state=args.random_state,
    )
    print("LDA training complete.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topics_path = args.output_dir / "topics.json"
    doc_topics_path = args.output_dir / "doc_topics.jsonl"

    save_topics(lda, vectorizer, args.top_words, topics_path)
    save_doc_topics(doc_topic, meta, doc_topics_path)
    print(f"Wrote topics to {topics_path}")
    print(f"Wrote document-topic weights to {doc_topics_path}")


if __name__ == "__main__":
    main()
