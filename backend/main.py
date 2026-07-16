"""
main.py — FastAPI backend for the Intelligent Bug Triage system
==================================================================
Exposes the RAG pipeline (preprocessing → BAAI/bge-large-en-v1.5 embedding →
Qdrant retrieval → Qwen3-8B reasoning chain → state-aware decision engine)
as a REST API consumed by the React dashboard.

Endpoints
---------
GET  /health                  → liveness check
GET  /builds                  → list available Jenkins build_ids (demo data)
POST /triage                  → run full triage pipeline on a build_id or raw log
GET  /tickets/{ticket_id}     → fetch a single Jira ticket's details
GET  /stats                   → aggregate dashboard stats (action distribution etc.)

Run:
    uvicorn main:app --reload --port 8000
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make the pipeline scripts importable ─────────────────────────────────────
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PIPELINE_ROOT / "scripts"))

DATA_DIR = PIPELINE_ROOT / "data"
OUT_DIR  = PIPELINE_ROOT / "outputs"

app = FastAPI(
    title="Intelligent Bug Triage API",
    description="RAG-based duplicate detection and workaround retrieval for Jenkins CI/CD failures",
    version="1.0.0",
)

# Allow the local React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load static demo data (Jenkins logs + Jira tickets) once at startup ──────
with open(DATA_DIR / "jenkins_logs.json") as f:
    JENKINS_LOGS = {l["build_id"]: l for l in json.load(f)}

with open(DATA_DIR / "jira_tickets.json") as f:
    JIRA_TICKETS = {t["ticket_id"]: t for t in json.load(f)}

with open(OUT_DIR / "jenkins_preprocessed.json") as f:
    JENKINS_PREPROCESSED = {l["build_id"]: l for l in json.load(f)}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    build_id: Optional[str] = Field(None, description="Existing Jenkins build_id from the demo dataset")
    raw_log_text: Optional[str] = Field(None, description="Raw console log text, if build_id is not supplied")


class BugFields(BaseModel):
    bug_title: str
    component: str
    error_message: str
    symptoms: str
    category: str


class TriageResponse(BaseModel):
    build_id: str
    failure_category: str
    bug_fields: BugFields
    top_match_ticket_id: Optional[str]
    top_match_score: float
    is_duplicate: bool
    confidence: float
    reason: str
    action: Literal["DUPLICATE", "WORKAROUND_AVAILABLE", "NEW_ISSUE"]
    workaround_steps: list[str]
    matched_ticket: Optional[dict] = None
    elapsed_seconds: float


class BuildSummary(BaseModel):
    build_id: str
    job_name: str
    failure_category: str
    failure_summary: str
    environment: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration (lazy-loaded singletons — model loads once)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_state = {"client": None, "retrieval_args": None, "mode": None}


def get_pipeline():
    """
    Lazily initialise the embedding model(s) + Qdrant client on first request.
    Tries the REAL pipeline (BAAI/bge-large-en-v1.5 + Qdrant's native BM25 +
    native RRF) first; falls back to the DEMO hybrid pipeline (TF-IDF dense
    stand-in + real BM25 via rank_bm25, RRF-fused in Python) if the real
    embedding model can't be downloaded (e.g. no internet access).

    `retrieval_args` holds the positional arguments expected after `client`
    and before `jenkins_log` by the chosen triage function.
    """
    if _pipeline_state["client"] is not None:
        return _pipeline_state

    try:
        from qdrant_client import QdrantClient
        import importlib

        embed_index = importlib.import_module("03_embed_index")
        dense_model = embed_index.load_embedding_model()
        sparse_model = embed_index.load_sparse_model()
        client = QdrantClient(path=str(OUT_DIR / "qdrant_storage"))

        _pipeline_state.update(client=client, retrieval_args=(dense_model, sparse_model), mode="production")
        print("✓ Loaded PRODUCTION pipeline (BAAI/bge-large-en-v1.5 + Qdrant)")

    except Exception as e:
        print(f"⚠ Could not load production embedder ({e}); falling back to DEMO pipeline")
        from importlib import import_module
        demo = import_module("03b_embed_index_demo")

        with open(OUT_DIR / "jira_preprocessed.json") as f:
            jira_processed = json.load(f)

        demo_dense_model = demo.DemoEmbedder(dim=1024)
        demo_bm25_index = demo.BM25SparseIndex()
        demo_client = demo.get_qdrant_client()
        demo.create_collection(demo_client)
        demo.index_jira_tickets(demo_client, demo_dense_model, demo_bm25_index, jira_processed)

        _pipeline_state.update(
            client=demo_client,
            retrieval_args=(demo_dense_model, demo_bm25_index),
            mode="demo",
        )
        print("✓ Loaded DEMO pipeline (TF-IDF + real BM25 hybrid stand-in)")

    return _pipeline_state


def get_llm_chain(mode: str):
    """Returns the appropriate triage function depending on which mode loaded."""
    from importlib import import_module
    if mode == "production":
        try:
            import ollama
            ollama.show("qwen3:8b")
            real = import_module("04_rag_pipeline")
            return real.triage_one_failure, real.COLLECTION
        except Exception as e:
            print(f"⚠ Qwen3-8B/Ollama not available ({e}); using mock LLM chain for reasoning only")
    demo_chain = import_module("04b_rag_pipeline_demo")
    return demo_chain.triage_one_failure_demo, demo_chain.COLLECTION


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Intelligent Bug Triage API"}


@app.get("/builds", response_model=list[BuildSummary])
def list_builds(limit: int = 30):
    """Returns a list of available Jenkins builds for the dashboard's picker."""
    builds = list(JENKINS_LOGS.values())[:limit]
    return [
        BuildSummary(
            build_id=b["build_id"],
            job_name=b["job_name"],
            failure_category=b["failure_category"],
            failure_summary=b["failure_summary"],
            environment=b["environment"],
            timestamp=b["timestamp"],
        )
        for b in builds
    ]


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    ticket = JIRA_TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


@app.get("/stats")
def get_stats():
    """Aggregate stats for the dashboard header cards."""
    from collections import Counter
    cat_counts = Counter(l["failure_category"] for l in JENKINS_LOGS.values())
    status_counts = Counter(t["status"] for t in JIRA_TICKETS.values())
    return {
        "total_jenkins_builds": len(JENKINS_LOGS),
        "total_jira_tickets": len(JIRA_TICKETS),
        "failure_category_distribution": dict(cat_counts),
        "ticket_status_distribution": dict(status_counts),
    }


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    if not req.build_id and not req.raw_log_text:
        raise HTTPException(400, "Provide either build_id or raw_log_text")

    if req.build_id:
        preprocessed = JENKINS_PREPROCESSED.get(req.build_id)
        if not preprocessed:
            raise HTTPException(404, f"build_id {req.build_id} not found in preprocessed logs")
    else:
        # Minimal on-the-fly preprocessing for raw text pasted by the user
        preprocessed = {
            "build_id": "AD-HOC-" + str(int(time.time())),
            "failed_stage": "Unknown",
            "failure_category": "product",  # default guess; LLM Prompt 2 will refine in real mode
            "failure_component": "Unknown",
            "cleaned_text": req.raw_log_text[:3000],
        }

    state = get_pipeline()
    triage_fn, _ = get_llm_chain(state["mode"])

    t0 = time.time()
    verdict = triage_fn(state["client"], *state["retrieval_args"], preprocessed)
    elapsed = time.time() - t0

    verdict_dict = verdict.__dict__ if hasattr(verdict, "__dict__") else verdict

    matched_ticket = None
    if verdict_dict.get("top_match_ticket_id"):
        matched_ticket = JIRA_TICKETS.get(verdict_dict["top_match_ticket_id"])

    return TriageResponse(
        build_id=verdict_dict["build_id"],
        failure_category=verdict_dict["failure_category"],
        bug_fields=verdict_dict["bug_fields"],
        top_match_ticket_id=verdict_dict.get("top_match_ticket_id"),
        top_match_score=verdict_dict.get("top_match_score") or 0.0,
        is_duplicate=verdict_dict.get("is_duplicate", False),
        confidence=verdict_dict.get("confidence", 0.0),
        reason=verdict_dict.get("reason", ""),
        action=verdict_dict["action"],
        workaround_steps=verdict_dict.get("workaround_steps", []),
        matched_ticket=matched_ticket,
        elapsed_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
