"""Measure search quality: for each test question, where does the right page land in the results?

Hit@k = share of questions whose right page is in the top k.
MRR (mean reciprocal rank) = average of 1/rank, so rank 1 scores 1.0, rank 2 scores 0.5, and so on.

    python scripts/eval_search.py [data/eval_queries.jsonl] [data/index]
"""

import json
import sys
from pathlib import Path

from fetcher.search import FastEmbedder, SearchIndex

K = 5
queries_file = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval_queries.jsonl")
index_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/index")

cases = [json.loads(line) for line in queries_file.read_text(encoding="utf-8").splitlines() if line]
index = SearchIndex.load(index_dir)
embedder = FastEmbedder()

ranks: list[int | None] = []
for case in cases:
    urls = [hit.chunk.url for hit in index.search(embedder.embed_query(case["query"]), k=K)]
    rank = urls.index(case["expected_url"]) + 1 if case["expected_url"] in urls else None
    ranks.append(rank)
    print(f"{rank or '-':>2}  {case['query']}")

n = len(cases)
print(f"\n{n} questions")
print(f"hit@1  {sum(r == 1 for r in ranks) / n:.2f}")
print(f"hit@{K}  {sum(r is not None for r in ranks) / n:.2f}")
print(f"MRR    {sum(1 / r for r in ranks if r) / n:.2f}")
