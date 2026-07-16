"""
bm25_local.py
==============
A small, dependency-free BM25 implementation (Robertson/Sparck-Jones IDF +
standard k1/b saturation), used ONLY as the sandbox-safe sparse retriever in
03b_/04b_/05_*_demo.py scripts — because fastembed's `Qdrant/bm25` model
still needs a one-time vocabulary download from huggingface.co, which this
sandbox cannot reach.

This is the REAL BM25 scoring formula (not a TF-IDF approximation) — just
implemented in plain Python/NumPy instead of via fastembed. On your machine,
03_embed_index.py uses Qdrant's native BM25 sparse vectors via fastembed
instead of this module; the ranking math is equivalent.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance
Framework: BM25 and Beyond."
"""

import re
import math
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Standard BM25 (Okapi) over a fixed corpus, scored via sparse dot product —
    structurally equivalent to a Qdrant sparse vector (token_id -> weight)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()       # document frequency per token
        self.n_docs: int = 0
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> None:
        self.n_docs = len(documents)
        for doc in documents:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            self.doc_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for token in tf:
                self.df[token] += 1
        self.avg_doc_len = sum(self.doc_lens) / max(self.n_docs, 1)
        # standard BM25 IDF with +1 smoothing to keep values non-negative
        for token, freq in self.df.items():
            self.idf[token] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def doc_vector(self, doc_index: int) -> dict[str, float]:
        """Sparse vector for an indexed document: token -> BM25 weight."""
        tf = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        vec = {}
        for token, freq in tf.items():
            idf = self.idf.get(token, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avg_doc_len, 1e-9))
            vec[token] = idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
        return vec

    def query_vector(self, query: str) -> dict[str, float]:
        """Sparse query vector — IDF-weighted term presence (standard BM25 query side)."""
        tokens = set(tokenize(query))
        return {t: self.idf.get(t, 0.0) for t in tokens if t in self.idf}

    def score(self, query: str, doc_index: int) -> float:
        """Direct BM25 score of `query` against document `doc_index` (sanity-check path)."""
        q_tokens = tokenize(query)
        tf = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for token in q_tokens:
            if token not in tf:
                continue
            idf = self.idf.get(token, 0.0)
            freq = tf[token]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avg_doc_len, 1e-9))
            score += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Brute-force BM25 ranking over all fitted documents (sandbox-scale: 300 docs, fine)."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """
    Standard Reciprocal Rank Fusion (RRF) — combines multiple ranked lists
    (e.g. dense top-K and sparse/BM25 top-K) into one fused ranking, the
    same algorithm Qdrant's hybrid query API uses internally.
    score(d) = sum over rankings of 1 / (k + rank(d))
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
