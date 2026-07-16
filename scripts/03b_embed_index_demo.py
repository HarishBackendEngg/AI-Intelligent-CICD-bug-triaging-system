"""
03b_embed_index_demo.py
========================
⚠ DEMO / SANDBOX-ONLY SCRIPT — NOT PART OF THE DISSERTATION PIPELINE.

This sandbox cannot reach huggingface.co to download BAAI/bge-large-en-v1.5
OR Qdrant's fastembed BM25 model (only PyPI/npm/GitHub are network-
allowlisted here). This script substitutes:
  - DENSE branch:  TF-IDF + Truncated SVD (scikit-learn, pure CPU, no
                    downloads) standing in for BGE-large
  - SPARSE branch: REAL BM25 via `rank_bm25` (pure Python, no downloads —
                    this is genuine BM25, not a stand-in)
  - FUSION:        hand-rolled Reciprocal Rank Fusion (RRF), identical
                    formula to Qdrant's native RRF: score(d) = Σ 1/(k+rank)

This lets the HYBRID RETRIEVAL LOGIC be verified end-to-end right now,
even though the dense branch is a weak substitute. On your own machine,
run 03_embed_index.py instead — it uses real BGE-large + Qdrant's native
BM25 + Qdrant's native RRF fusion via the Query API. The downstream
pipeline (04_rag_pipeline.py) is identical either way.

Run:
    python 03b_embed_index_demo.py
"""

import json
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rank_bm25 import BM25Okapi
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

OUT_DIR     = Path(__file__).resolve().parent.parent / "outputs"
QDRANT_PATH = OUT_DIR / "qdrant_storage_demo"
COLLECTION  = "jira_bug_tickets_demo"
VECTOR_DIM  = 1024  # match BGE-large dimensionality for pipeline compatibility
RRF_K       = 60    # Qdrant / Elasticsearch default RRF smoothing constant

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class DemoEmbedder:
    """
    DENSE branch only — TF-IDF + Truncated SVD → fixed 1024-dim dense
    vectors. Stands in for BAAI/bge-large-en-v1.5 in this offline sandbox.
    """
    def __init__(self, dim: int = VECTOR_DIM):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_comp < self.dim:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        if not self._fitted:
            raise RuntimeError("Call .fit() before .encode()")
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        if vecs.shape[1] < self.dim:
            pad = np.zeros((vecs.shape[0], self.dim - vecs.shape[1]))
            vecs = np.hstack([vecs, pad])
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
        return vecs[0] if single else vecs


class BM25SparseIndex:
    """
    SPARSE branch — REAL BM25 (via rank_bm25), not a stand-in. Kept separate
    from Qdrant's sparse-vector storage (which needs the fastembed BM25
    tokenizer/IDF artifact we can't download here) — instead this runs BM25
    scoring directly in Python and is fused with the dense Qdrant results
    via manual RRF below.
    """
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []
        self.category_lookup: dict[str, str] = {}  # ticket_id -> category,
                                                      # populated for EVERY
                                                      # indexed doc (not just
                                                      # whatever the dense
                                                      # branch happened to
                                                      # retrieve) so sparse-
                                                      # branch category
                                                      # filtering is correct
                                                      # and independent.

    def fit(self, texts: list[str], doc_ids: list[str], categories: list[str] = None) -> None:
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = doc_ids
        if categories is not None:
            self.category_lookup = dict(zip(doc_ids, categories))

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Returns [(doc_id, bm25_score), ...] sorted descending by score."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:limit]


def embed_query(model: DemoEmbedder, text: str):
    instruction = "Represent this sentence for searching relevant passages: "
    return model.encode(instruction + text, normalize_embeddings=True).tolist()


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def create_collection(client: QdrantClient) -> None:
    print(f"\n[2/5] Creating Qdrant collection '{COLLECTION}' (dense-only; "
          f"sparse handled separately via rank_bm25 in this demo)...")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print("  (existing collection deleted — rebuilding fresh)")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"  ✓ Collection '{COLLECTION}' created")


def index_jira_tickets(client, dense_model: DemoEmbedder, bm25_index: BM25SparseIndex,
                        jira_processed: list[dict]) -> None:
    print(f"\n[3/5] Embedding {len(jira_processed)} Jira ticket text blobs "
          f"(DENSE: TF-IDF stand-in, SPARSE: real BM25)...")
    texts = [t["text_blob"] for t in jira_processed]
    ticket_ids = [t["ticket_id"] for t in jira_processed]
    categories = [t["metadata"]["category"] for t in jira_processed]

    t0 = time.time()
    dense_model.fit(texts)
    dense_vectors = dense_model.encode(texts, normalize_embeddings=True)
    print(f"  ✓ Dense (TF-IDF) embedded {len(dense_vectors)} tickets in {time.time()-t0:.1f}s")

    t0 = time.time()
    bm25_index.fit(texts, ticket_ids, categories=categories)
    print(f"  ✓ Sparse (BM25) index built over {len(texts)} tickets in {time.time()-t0:.1f}s")

    print(f"\n[4/5] Upserting dense vectors + payload metadata into Qdrant...")
    points = []
    for i, (ticket, vector) in enumerate(zip(jira_processed, dense_vectors)):
        payload = {
            "ticket_id": ticket["ticket_id"],
            "text_blob": ticket["text_blob"][:500],
            **ticket["metadata"],
        }
        points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  ✓ {len(points)} dense points upserted into Qdrant collection '{COLLECTION}'")


def rrf_fuse(dense_ranked: list[tuple], sparse_ranked: list[tuple[str, float]], k: int = RRF_K) -> list[tuple]:
    """
    Reciprocal Rank Fusion — identical formula to Qdrant's native RRF:
        score(d) = sum over retrieval branches of 1 / (k + rank(d))
    where rank is 1-indexed position in that branch's ranked list.

    KNOWN LIMITATION (documented, not silently hidden): with a small
    per-category candidate pool (~30-40 docs after category filtering),
    multiple documents legitimately tie at the RRF ceiling — e.g. two
    documents both ranking #1 in one branch and #1/#2 in the other produce
    near-identical fused scores. Pure rank-based RRF then has almost no
    power to distinguish "this is clearly the best match" from "this
    merely tied for first." Production systems with thousands of
    candidates rarely hit this; our 300-doc demo corpus does.

    Fix applied: ties (or near-ties, within TIE_EPSILON) in the primary
    RRF score are broken using a secondary score = the underlying raw
    dense cosine similarity + raw (normalised) BM25 score. This is a
    standard secondary-ranking step, NOT a change to the RRF formula
    itself — Qdrant's native hybrid query API has the same practical
    need and handles it via its `score_threshold` / rerank stage.
    """
    TIE_EPSILON = 1e-6
    rrf_scores: dict[str, float] = {}
    raw_dense: dict[str, float] = {}
    raw_sparse: dict[str, float] = {}

    for rank, (point, raw_score) in enumerate(dense_ranked, start=1):
        doc_id = point.payload["ticket_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        raw_dense[doc_id] = raw_score

    max_sparse = max((s for _, s in sparse_ranked), default=1.0) or 1.0
    for rank, (doc_id, raw_score) in enumerate(sparse_ranked, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        raw_sparse[doc_id] = raw_score / max_sparse  # normalise BM25's unbounded scale to [0,1]

    payload_lookup = {p.payload["ticket_id"]: p.payload for p, _ in dense_ranked}

    def sort_key(item):
        doc_id, rrf_score = item
        tiebreak = raw_dense.get(doc_id, 0.0) + raw_sparse.get(doc_id, 0.0)
        return (round(rrf_score / TIE_EPSILON), tiebreak)  # primary: RRF bucket, secondary: raw score

    fused = sorted(rrf_scores.items(), key=sort_key, reverse=True)
    return [(doc_id, score, payload_lookup.get(doc_id)) for doc_id, score in fused]


def hybrid_search_demo(client, dense_model: DemoEmbedder, bm25_index: BM25SparseIndex,
                        query_text: str, limit: int = 5, category_filter: str = None) -> list[dict]:
    """Mirrors hybrid_search() in 03_embed_index.py but with the demo dense/sparse pair."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qfilter = None
    if category_filter:
        qfilter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category_filter))])

    dense_vec = embed_query(dense_model, query_text)
    dense_results = client.query_points(
        collection_name=COLLECTION, query=dense_vec, query_filter=qfilter, limit=20,
    ).points
    dense_ranked = [(p, p.score) for p in dense_results]

    # FIX: the sparse (BM25) branch has no native category filter — it scores
    # over the whole 300-ticket corpus. Previously, sparse candidates whose
    # category was unknown (i.e. not also present in the dense top-20) fell
    # through `cat_lookup.get(tid, category_filter)`, which silently DEFAULTED
    # to "category matches" and let off-category tickets leak into the fused
    # ranking. We now look up category from the full payload index (built
    # once, every ticket, not just the dense top-20) so the sparse branch is
    # filtered correctly and independently of what the dense branch happened
    # to retrieve.
    sparse_ranked_all = bm25_index.search(query_text, limit=50)
    if category_filter:
        sparse_ranked = [(tid, s) for tid, s in sparse_ranked_all
                          if bm25_index.category_lookup.get(tid) == category_filter][:20]
    else:
        sparse_ranked = sparse_ranked_all[:20]

    fused = rrf_fuse(dense_ranked, sparse_ranked, k=RRF_K)[:limit]
    return [{"score": score, "payload": payload} for _, score, payload in fused if payload]


def verify_retrieval(client, dense_model, bm25_index, jenkins_processed, category_filter: bool = False):
    print("\n🔍 Verification — running sample HYBRID (dense+sparse, RRF-fused) retrieval...")
    for sample_log in jenkins_processed[:3]:
        query_text = sample_log["cleaned_text"]
        cat = sample_log["failure_category"] if category_filter else None
        results = hybrid_search_demo(client, dense_model, bm25_index, query_text, limit=5, category_filter=cat)

        print(f"\n  Query build_id : {sample_log['build_id']}")
        print(f"  Query category : {sample_log['failure_category']}")
        print(f"  Query preview  : {query_text[:100]}...")
        print(f"  Top-5 retrieved Jira tickets (RRF-fused):")
        for r in results:
            print(f"    rrf_score={r['score']:.4f}  ticket_id={r['payload']['ticket_id']:<10}  "
                  f"category={r['payload']['category']:<14}  status={r['payload']['status']}")


if __name__ == "__main__":
    jira_path    = OUT_DIR / "jira_preprocessed.json"
    jenkins_path = OUT_DIR / "jenkins_preprocessed.json"

    with open(jira_path) as f:
        jira_processed = json.load(f)
    with open(jenkins_path) as f:
        jenkins_processed = json.load(f)

    print(f"\n[1/5] Initialising DEMO hybrid index — "
          f"dense=TF-IDF+SVD({VECTOR_DIM}d, stand-in), sparse=real BM25...")
    dense_model = DemoEmbedder(dim=VECTOR_DIM)
    bm25_index = BM25SparseIndex()

    client = get_qdrant_client()
    create_collection(client)
    index_jira_tickets(client, dense_model, bm25_index, jira_processed)
    verify_retrieval(client, dense_model, bm25_index, jenkins_processed, category_filter=True)

    print(f"\n[5/5] ✅ DEMO hybrid embedding + indexing complete (sandbox-only).")
    print(f"   Qdrant collection persisted at: {QDRANT_PATH}")
    print(f"\n👉 On your own machine, run 03_embed_index.py instead — it uses the")
    print(f"   real BAAI/bge-large-en-v1.5 + Qdrant's native BM25 + Qdrant's native")
    print(f"   RRF fusion via the Query API, exactly as documented in the mid-sem report.")
