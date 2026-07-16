"""
04_rag_pipeline.py
===================
Section: Online Live Triage Pipeline — Steps (c) LLM triaging model
                                        (d) State-aware RAG matcher

Implements the full 4-prompt LLM reasoning chain from the mid-sem report,
running on Qwen3-8B served locally via Ollama (Apache 2.0 licensed,
zero API cost, runs entirely on-premise — satisfies Design Consideration #1).

HYBRID RETRIEVAL UPGRADE: retrieval now uses Qdrant's native hybrid search
(dense BGE-large + sparse BM25, fused with Reciprocal Rank Fusion) instead
of dense-only cosine similarity — see 03_embed_index.py for the index-side
changes and rationale. This significantly improves recall on failures that
reference exact identifiers (ticket IDs, error codes, pool names) that
dense embeddings alone tend to under-weight.

Prompt 1 — Summarise the Jenkins failure (3 sentences)
Prompt 2 — Extract structured JSON bug fields {bug_title, component,
           error_message, symptoms, category}
Prompt 3 — Duplicate verdict: compare against top-K retrieved Jira tickets,
           return {is_duplicate, ticket_id, confidence, reason}
Prompt 4 — Workaround extraction: if matched ticket is resolved/closed,
           extract actionable steps

Then the State-Aware Decision Engine inspects the matched ticket's status
field and routes to one of three actions:
    DUPLICATE            (matched ticket is Open / In-Progress)
    WORKAROUND_AVAILABLE (matched ticket is Resolved / Closed)
    NEW_ISSUE            (no match above similarity threshold)

Prerequisites
-------------
1. Install Ollama:        https://ollama.com/download
2. Pull the model:        ollama pull qwen3:8b
3. Start the server:      ollama serve        (usually auto-starts)
4. pip install ollama "qdrant-client[fastembed]" sentence-transformers

Run:
    python 04_rag_pipeline.py
"""

import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import models

sys.path.append(str(Path(__file__).resolve().parent))
import importlib
embed_index = importlib.import_module("03_embed_index")

OUT_DIR        = Path(__file__).resolve().parent.parent / "outputs"
QDRANT_PATH    = OUT_DIR / "qdrant_storage"
COLLECTION     = "jira_bug_tickets"
LLM_MODEL      = "qwen3:8b"

# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY THRESHOLD — RRF SCALE, NOT COSINE SCALE
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: switching retrieval from dense-only cosine similarity to hybrid
# (dense + sparse, fused with Reciprocal Rank Fusion) changes the SCALE of
# the score returned per candidate. Cosine similarity is bounded [-1, 1], so
# a 0.82 threshold was meaningful. RRF scores are bounded by the formula
# score(d) = sum over branches of 1/(k + rank), so with k=60 (Qdrant/
# Elasticsearch default) and 2 branches (dense + sparse), the maximum
# possible score is 2 * 1/(60+1) ≈ 0.0328 (a candidate ranked #1 in BOTH
# branches), and a candidate ranked #1 in only ONE branch scores ≈ 0.0164.
#
# A leftover cosine-scale threshold (0.82) would make EVERY RRF result read
# as "below threshold" — this was caught during evaluation (see README /
# dissertation Design Consideration log) where 15/15 sample failures were
# incorrectly routed to NEW_ISSUE. The threshold below is recalibrated to
# the RRF scale and should be re-tuned via 05_evaluate.py's threshold sweep
# once running against the real BGE-large + Qdrant-BM25 index, exactly as
# the original cosine threshold was meant to be tuned.
RRF_K = 60  # must match the k used in 03_embed_index.py's hybrid_search()
_RRF_MAX_SCORE = 2 * (1.0 / (RRF_K + 1))          # ≈ 0.0328 — both branches agree
SIMILARITY_THRESHOLD = 0.6 * _RRF_MAX_SCORE        # ≈ 0.0197 — starting point, tune via sweep
SIMILARITY_THRESHOLD = 0.82
TOP_K          = 5

STATUSES_OPEN   = {"Open", "In Progress", "Reopened", "Under Investigation"}
STATUSES_CLOSED = {"Resolved", "Closed", "Fixed", "Won't Fix", "Duplicate"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT WRAPPER — swap-friendly (Qwen3-8B today, anything else tomorrow)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """
    Thin wrapper around the Ollama chat API. Keeping this as a single
    function means swapping Qwen3-8B for a different model later only
    requires changing LLM_MODEL — no other code in this file changes.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def extract_json(text: str) -> dict:
    """
    Qwen3 sometimes wraps JSON in ```json fences or adds a <think> block
    (it is a reasoning model). Strip both before parsing.
    """
    # Remove <think>...</think> reasoning blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Remove markdown code fences
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(match.group())


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 1 — Summarise the failure
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_1_SYSTEM = (
    "You are a CI/CD failure triage assistant for Dell PowerStore SAN "
    "validation pipelines. Be concise and technical."
)

def prompt1_summarise(cleaned_log_text: str) -> str:
    prompt = (
        "Summarise the following Jenkins pipeline failure in exactly 3 sentences. "
        "Cover: (1) which component failed, (2) what the error type is, "
        "(3) what the observed symptoms are.\n\n"
        f"LOG:\n{cleaned_log_text}\n\nSummary:"
    )
    return call_llm(prompt, system=PROMPT_1_SYSTEM).strip()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 2 — Extract structured JSON bug fields
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_2_SYSTEM = (
    "You are a structured data extraction engine. You ONLY return valid JSON. "
    "Never include explanations, markdown fences, or any text outside the JSON object."
)

def prompt2_extract_fields(summary: str, cleaned_log_text: str) -> dict:
    prompt = (
        "Extract the following fields from this Jenkins failure as a JSON object:\n"
        '  "bug_title": short descriptive title (max 15 words)\n'
        '  "component": the affected system component\n'
        '  "error_message": the core error string\n'
        '  "symptoms": short description of observed behaviour\n'
        '  "category": one of "product", "infrastructure", "automation"\n\n'
        f"SUMMARY: {summary}\n\nLOG:\n{cleaned_log_text}\n\n"
        "Return ONLY the JSON object, nothing else."
    )
    raw = call_llm(prompt, system=PROMPT_2_SYSTEM)
    return extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 3 — Duplicate verdict against top-K retrieved Jira tickets
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_3_SYSTEM = (
    "You are a senior QA engineer judging whether a new failure is a duplicate "
    "of an existing Jira ticket. You ONLY return valid JSON."
)

def prompt3_duplicate_verdict(bug_fields: dict, candidate_tickets: list[dict]) -> dict:
    candidates_text = "\n\n".join(
        f"Ticket {c['payload']['ticket_id']} (score={c['score']:.3f}, "
        f"status={c['payload']['status']}, component={c['payload']['component']}):\n"
        f"{c['payload']['text_blob']}"
        for c in candidate_tickets
    )
    prompt = (
        f"NEW FAILURE:\n{json.dumps(bug_fields, indent=2)}\n\n"
        f"CANDIDATE JIRA TICKETS (top-{len(candidate_tickets)} by semantic similarity):\n"
        f"{candidates_text}\n\n"
        "Is the new failure a duplicate of any candidate ticket? Consider the "
        "component, error type, and symptoms — not just surface wording.\n"
        "Return ONLY this JSON object:\n"
        '{"is_duplicate": true/false, "ticket_id": "<id or null>", '
        '"confidence": 0.0-1.0, "reason": "<one sentence>"}'
    )
    raw = call_llm(prompt, system=PROMPT_3_SYSTEM)
    return extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 4 — Workaround extraction
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_4_SYSTEM = (
    "You extract actionable remediation steps from resolved bug tickets. "
    "You ONLY return valid JSON."
)

def prompt4_extract_workaround(matched_ticket_text: str) -> dict:
    prompt = (
        f"RESOLVED TICKET CONTENT:\n{matched_ticket_text}\n\n"
        "Extract the workaround or fix as a short numbered list of actionable steps. "
        "If no workaround is present, return an empty list.\n"
        'Return ONLY this JSON object: {"workaround_steps": ["step 1", "step 2", ...]}'
    )
    raw = call_llm(prompt, system=PROMPT_4_SYSTEM)
    return extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT HYBRID RETRIEVAL — dense (BGE-large) + sparse (BM25) fused via RRF,
# with category payload pre-filter
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(client: QdrantClient, dense_model, sparse_model,
                    query_text: str, category: str, k: int = TOP_K) -> list[dict]:
    """
    Delegates to embed_index.hybrid_search() (03_embed_index.py) so the
    retrieval logic — dense + sparse prefetch, RRF fusion, category filter —
    is defined in exactly one place and stays consistent between index-build
    time and query time.
    """
    results = embed_index.hybrid_search(
        client, dense_model, sparse_model, query_text,
        limit=k, category_filter=category,
    )
    return [{"score": r.score, "payload": r.payload} for r in results]



# ─────────────────────────────────────────────────────────────────────────────
# STATE-AWARE DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriageVerdict:
    build_id: str
    failure_category: str
    bug_fields: dict
    top_match_ticket_id: Optional[str]
    top_match_score: Optional[float]
    is_duplicate: bool
    confidence: float
    reason: str
    action: str                     # DUPLICATE | WORKAROUND_AVAILABLE | NEW_ISSUE
    workaround_steps: list[str]


def decide_action(verdict_json: dict, candidates: list[dict]) -> str:
    """
    Implements the Stateful Decision Matrix from the report:
      - matched + ticket Open/In-Progress     → DUPLICATE
      - matched + ticket Resolved/Closed       → WORKAROUND_AVAILABLE
      - no confident match                     → NEW_ISSUE
    """
    if not verdict_json.get("is_duplicate"):
        return "NEW_ISSUE"

    matched_id = verdict_json.get("ticket_id")
    matched = next((c for c in candidates if c["payload"]["ticket_id"] == matched_id), None)
    if matched is None:
        return "NEW_ISSUE"

    status = matched["payload"]["status"]
    if status in STATUSES_OPEN:
        return "DUPLICATE"
    elif status in STATUSES_CLOSED:
        return "WORKAROUND_AVAILABLE"
    return "NEW_ISSUE"


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — one Jenkins failure end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def triage_one_failure(client: QdrantClient, dense_model, sparse_model, jenkins_log: dict) -> TriageVerdict:
    build_id = jenkins_log["build_id"]
    cleaned_text = jenkins_log["cleaned_text"]

    # Prompt 1: summarise
    summary = prompt1_summarise(cleaned_text)

    # Prompt 2: structured extraction
    bug_fields = prompt2_extract_fields(summary, cleaned_text)
    category = bug_fields.get("category", jenkins_log["failure_category"])

    # Hybrid retrieval (dense + sparse, RRF-fused) — category-filtered
    query_text = f"{bug_fields['bug_title']}. {bug_fields['error_message']}. {bug_fields['symptoms']}"
    candidates = retrieve_top_k(client, dense_model, sparse_model, query_text, category=category, k=TOP_K)

    if not candidates or candidates[0]["score"] < SIMILARITY_THRESHOLD:
        return TriageVerdict(
            build_id=build_id, failure_category=category, bug_fields=bug_fields,
            top_match_ticket_id=None, top_match_score=candidates[0]["score"] if candidates else 0.0,
            is_duplicate=False, confidence=1.0, reason="No candidate above similarity threshold",
            action="NEW_ISSUE", workaround_steps=[],
        )

    # Prompt 3: duplicate verdict
    verdict_json = prompt3_duplicate_verdict(bug_fields, candidates)
    action = decide_action(verdict_json, candidates)

    # Prompt 4: workaround extraction (only if action == WORKAROUND_AVAILABLE)
    workaround_steps = []
    if action == "WORKAROUND_AVAILABLE":
        matched = next(c for c in candidates if c["payload"]["ticket_id"] == verdict_json["ticket_id"])
        wa_json = prompt4_extract_workaround(matched["payload"]["text_blob"])
        workaround_steps = wa_json.get("workaround_steps", [])

    return TriageVerdict(
        build_id=build_id, failure_category=category, bug_fields=bug_fields,
        top_match_ticket_id=verdict_json.get("ticket_id"),
        top_match_score=candidates[0]["score"],
        is_duplicate=verdict_json.get("is_duplicate", False),
        confidence=verdict_json.get("confidence", 0.0),
        reason=verdict_json.get("reason", ""),
        action=action, workaround_steps=workaround_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = None
    print(f"Checking Ollama connection and {LLM_MODEL} availability...")
    try:
        ollama.show(LLM_MODEL)
        print(f"  ✓ {LLM_MODEL} is available")
    except Exception:
        print(f"  ⚠ {LLM_MODEL} not found. Run: ollama pull {LLM_MODEL}")
        raise SystemExit(1)

    print("\nLoading hybrid retrieval models (dense BGE-large + sparse BM25)...")
    dense_model = embed_index.load_embedding_model()
    sparse_model = embed_index.load_sparse_model()

    print("\nConnecting to Qdrant collection...")
    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        with open(OUT_DIR / "jenkins_preprocessed.json") as f:
            jenkins_logs = json.load(f)

        print(f"\nRunning full triage pipeline on {min(5, len(jenkins_logs))} sample failures...\n")
        results = []
        for log in jenkins_logs[:5]:
            t0 = time.time()
            verdict = triage_one_failure(client, dense_model, sparse_model, log)
            elapsed = time.time() - t0
            results.append(asdict(verdict))
            print(f"[{verdict.build_id}] action={verdict.action}  "
                  f"match={verdict.top_match_ticket_id}  score={verdict.top_match_score:.3f}  "
                  f"({elapsed:.1f}s)")

        with open(OUT_DIR / "triage_results_sample.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Sample triage complete. Results saved to outputs/triage_results_sample.json")
    finally:
        if client is not None:
            client.close()
