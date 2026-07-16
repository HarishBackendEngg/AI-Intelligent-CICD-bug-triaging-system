"""
03_embed_index.py
==================
Section: Offline Knowledge Ingestion Pipeline — Step (c) Embedding Model and Vector DB

HYBRID SEARCH UPGRADE (dense + sparse):
----------------------------------------
The original version of this script indexed Jira tickets using only dense
embeddings (BAAI/bge-large-en-v1.5). Evaluation showed dense-only retrieval
under-performs on queries containing exact identifiers (ticket IDs, error
codes, pool names) because transformer embeddings compress rare tokens into
"vaguely similar" vectors rather than preserving exact-match signal.

This version builds a HYBRID index:
  - DENSE vector  (BAAI/bge-large-en-v1.5, 1024-dim) — captures semantic /
    paraphrase similarity ("session drops" ≈ "connection terminates")
  - SPARSE vector (BM25)                              — captures exact
    keyword / identifier matches (PSTR-1242, ISCSI_ERR_TCP_CONN_CLOSE)

At query time (04_rag_pipeline.py) both retrieval branches are prefetched
and fused with Reciprocal Rank Fusion (RRF) — Qdrant's native, tuning-free
fusion method (Qdrant Query API, qdrant.tech/documentation). RRF operates
on rank position rather than raw score, which sidesteps the fact that BM25
scores are unbounded while cosine similarity is bounded in [-1, 1].

Run:
    python 03_embed_index.py
"""

import json
import sys
import time
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct, models
)

OUT_DIR        = Path(__file__).resolve().parent.parent / "outputs"
QDRANT_PATH    = OUT_DIR / "qdrant_storage"
COLLECTION     = "jira_bug_tickets"
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
VECTOR_DIM     = 1024   # bge-large-en-v1.5 native output dimension
BM25_MODEL_ID  = "Qdrant/bm25"

DENSE_VEC_NAME  = "dense"
SPARSE_VEC_NAME = "bm25_sparse"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the dense embedding model + sparse BM25 model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Loads BAAI/bge-large-en-v1.5 locally via sentence-transformers.
    First run downloads ~1.3GB of model weights from Hugging Face;
    subsequent runs load from the local cache (~/.cache/huggingface).

    NOTE: requires outbound access to huggingface.co. If you are running
    this behind a corporate proxy/firewall (e.g. Dell on-prem network),
    either:
      (a) pre-download the model once on a machine with internet access:
          huggingface-cli download BAAI/bge-large-en-v1.5
          then copy the ~/.cache/huggingface folder to the offline machine, or
      (b) set HF_HUB_OFFLINE=1 once the model is cached locally.
    """
    print(f"\n[1/5] Loading dense embedding model: {EMBED_MODEL_ID} ...")
    t0 = time.time()
    try:
        model = SentenceTransformer(EMBED_MODEL_ID)
        print(f"  ✓ Model loaded in {time.time()-t0:.1f}s  (output dim = {model.get_sentence_embedding_dimension()})")
        return model
    except Exception as e:
        print(f"  ⚠ Could not reach Hugging Face Hub ({type(e).__name__}).")
        print(f"    This will work on your local machine / Dell on-prem environment")
        print(f"    once huggingface.co is reachable, or the model is pre-cached.")
        raise


def load_sparse_model():
    """
    Loads Qdrant's built-in BM25 sparse embedding model via fastembed.
    This requires the `qdrant-client[fastembed]` extra:
        pip install "qdrant-client[fastembed]"
    First run downloads a small (~few MB) vocabulary/IDF file from
    Hugging Face; cached locally afterwards — same network note as above.
    """
    print(f"\n[2/5] Loading sparse BM25 model: {BM25_MODEL_ID} ...")
    from fastembed import SparseTextEmbedding
    t0 = time.time()
    done = threading.Event()

    def heartbeat():
        while not done.wait(10):
            elapsed = time.time() - t0
            print(f"  ... still loading BM25 model after {elapsed:.0f}s (first run may download from Hugging Face)")

    watcher = threading.Thread(target=heartbeat, daemon=True)
    watcher.start()
    try:
        model = SparseTextEmbedding(model_name=BM25_MODEL_ID)
        print(f"  ✓ BM25 model loaded in {time.time()-t0:.1f}s")
        return model
    except Exception as e:
        print(f"  ⚠ Could not load BM25 model ({type(e).__name__}): {e}")
        raise
    finally:
        done.set()


def embed_dense_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,   # required for correct cosine similarity in Qdrant
    )
    return embeddings.tolist()


def embed_dense_query(model: SentenceTransformer, text: str) -> list[float]:
    """BGE-recommended instruction prefix for query-side dense embeddings."""
    instruction = "Represent this sentence for searching relevant passages: "
    return model.encode(instruction + text, normalize_embeddings=True).tolist()


def embed_sparse_texts(model, texts: list[str]) -> list[dict]:
    """Returns a list of {indices, values} sparse vector dicts, one per text."""
    sparse_embeddings = list(model.embed(texts))
    return [{"indices": e.indices.tolist(), "values": e.values.tolist()} for e in sparse_embeddings]


def embed_sparse_query(model, text: str) -> dict:
    sparse_embeddings = list(model.query_embed(text))
    e = sparse_embeddings[0]
    return {"indices": e.indices.tolist(), "values": e.values.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Set up Qdrant collection with BOTH a named dense vector and a
#          named sparse vector
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Local on-disk Qdrant instance — no server/Docker required for this
    dissertation prototype. For production, swap to QdrantClient(url=...)
    pointing at a Qdrant Docker container or Qdrant Cloud instance.
    """
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    return client


def create_collection(client: QdrantClient) -> None:
    print(f"\n[3/5] Creating HYBRID Qdrant collection '{COLLECTION}' "
          f"(dense dim={VECTOR_DIM} cosine + sparse BM25)...")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print("  (existing collection deleted — rebuilding fresh)")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VEC_NAME: VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VEC_NAME: SparseVectorParams(),
        },
    )
    print(f"  ✓ Hybrid collection '{COLLECTION}' created "
          f"(named vectors: '{DENSE_VEC_NAME}', '{SPARSE_VEC_NAME}')")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Embed Jira tickets (dense + sparse) and upsert into Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def index_jira_tickets(client: QdrantClient, dense_model: SentenceTransformer,
                        sparse_model, jira_processed: list[dict]) -> None:
    print(f"\n[4/5] Embedding {len(jira_processed)} Jira ticket text blobs "
          f"(dense BGE-large + sparse BM25)...")
    texts = [t["text_blob"] for t in jira_processed]

    t0 = time.time()
    dense_vectors = embed_dense_texts(dense_model, texts)
    print(f"  ✓ Dense embeddings done in {time.time()-t0:.1f}s")

    t0 = time.time()
    sparse_vectors = embed_sparse_texts(sparse_model, texts)
    print(f"  ✓ Sparse (BM25) embeddings done in {time.time()-t0:.1f}s")

    print(f"\n[5/5] Upserting hybrid vectors + payload metadata into Qdrant...")
    points = []
    for i, (ticket, dvec, svec) in enumerate(zip(jira_processed, dense_vectors, sparse_vectors)):
        payload = {
            "ticket_id": ticket["ticket_id"],
            "text_blob": ticket["text_blob"][:500],  # store a preview, not full blob
            **ticket["metadata"],
        }
        points.append(PointStruct(
            id=i,
            vector={
                DENSE_VEC_NAME: dvec,
                SPARSE_VEC_NAME: models.SparseVector(indices=svec["indices"], values=svec["values"]),
            },
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  ✓ {len(points)} hybrid points upserted into Qdrant collection '{COLLECTION}'")


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL — dense + sparse prefetch, fused with RRF
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(client: QdrantClient, dense_model: SentenceTransformer, sparse_model,
                   query_text: str, limit: int = 5, category_filter: str = None) -> list:
    """
    Qdrant native hybrid search: prefetch top candidates from BOTH the dense
    and sparse branches, then fuse with Reciprocal Rank Fusion (RRF) — no
    manual score normalization needed, since RRF operates on rank position.
    """
    dense_q  = embed_dense_query(dense_model, query_text)
    sparse_q = embed_sparse_query(sparse_model, query_text)

    qfilter = None
    if category_filter:
        qfilter = models.Filter(must=[
            models.FieldCondition(key="category", match=models.MatchValue(value=category_filter))
        ])

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_q, using=DENSE_VEC_NAME, limit=20, filter=qfilter),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
                using=SPARSE_VEC_NAME, limit=20, filter=qfilter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    ).points
    return results


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION — run one sample query end-to-end to prove the index works
# ─────────────────────────────────────────────────────────────────────────────

def verify_retrieval(client: QdrantClient, dense_model, sparse_model, jenkins_processed: list[dict]) -> None:
    print("\n🔍 Verification — running a sample HYBRID retrieval query...")
    sample_log = jenkins_processed[0]
    query_text = sample_log["cleaned_text"]

    results = hybrid_search(client, dense_model, sparse_model, query_text, limit=5)

    print(f"\n  Query build_id : {sample_log['build_id']}")
    print(f"  Query category : {sample_log['failure_category']}")
    print(f"  Query preview  : {query_text[:120]}...")
    print(f"\n  Top-5 retrieved Jira tickets (RRF-fused dense + sparse):")
    for r in results:
        print(f"    rrf_score={r.score:.4f}  ticket_id={r.payload['ticket_id']:<10}  "
              f"category={r.payload['category']:<14}  status={r.payload['status']}")
    client.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jira_path    = OUT_DIR / "jira_preprocessed.json"
    jenkins_path = OUT_DIR / "jenkins_preprocessed.json"

    if not jira_path.exists():
        raise FileNotFoundError("Run 02_preprocess.py first to generate jira_preprocessed.json")

    with open(jira_path) as f:
        jira_processed = json.load(f)
    with open(jenkins_path) as f:
        jenkins_processed = json.load(f)

    # Step 1 & 2: load dense + sparse models
    dense_model = load_embedding_model()
    sparse_model = load_sparse_model()

    # Step 3: create hybrid Qdrant collection
    client = get_qdrant_client()
    create_collection(client)

    # Step 4 & 5: embed + index (both branches)
    index_jira_tickets(client, dense_model, sparse_model, jira_processed)

    # Verification
    verify_retrieval(client, dense_model, sparse_model, jenkins_processed)

    print(f"\n✅ Hybrid embedding + indexing complete.")
    print(f"   Qdrant collection persisted at: {QDRANT_PATH}")
    print(f"   Proceed to 04_rag_pipeline.py for the full LLM reasoning chain")
