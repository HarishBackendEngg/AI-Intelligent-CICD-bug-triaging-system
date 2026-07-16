"""
02_preprocess.py
=================
Section: Offline Knowledge Ingestion Pipeline — Step (b) Pre-Processing of Jira data
                Online Live Triage Pipeline      — Step (b) Logs pre-processing

Jira side  : concatenate title + description + resolution into one text blob
             per ticket, ready for embedding, with ticket_id + metadata kept
             alongside for the Qdrant payload.

Jenkins side: strip ANSI escape codes, split by pipeline stage using the
             [Pipeline] markers, extract only ERROR / WARN / EXCEPTION /
             Traceback lines, and truncate to a token-safe character limit.

Run:
    python 02_preprocess.py
"""

import re
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from importlib import import_module
load_data = import_module("01_load_data")

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
OUT_DIR    = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ANSI_RE       = re.compile(r"\x1b\[[0-9;]*m")
ERROR_LINE_RE = re.compile(r"(ERROR|WARN|EXCEPTION|Traceback|FAILED?)", re.IGNORECASE)
TOKEN_CHAR_LIMIT = 3000  # ≈ 750 tokens, matches report spec "3000-char token limit"


# ─────────────────────────────────────────────────────────────────────────────
# JIRA SIDE — Pre-Processing of Jira data
# ─────────────────────────────────────────────────────────────────────────────

def build_jira_text_blob(ticket: dict) -> str:
    """
    Concatenate title + description + resolution into a single text blob.
    This is the exact text that will be embedded in 03_embed_index.py.
    """
    parts = [
        ticket.get("summary", ""),
        ticket.get("description", ""),
    ]
    if ticket.get("resolution"):
        parts.append(ticket["resolution"])
    if ticket.get("workaround"):
        parts.append(ticket["workaround"])
    return " . ".join(p.strip() for p in parts if p)


def preprocess_jira_tickets(tickets: list[dict]) -> list[dict]:
    """Returns a list of {ticket_id, text_blob, metadata} ready for embedding."""
    processed = []
    for t in tickets:
        text_blob = build_jira_text_blob(t)
        metadata = {
            "ticket_id":  t["ticket_id"],
            "status":     t["status"],
            "component":  t["component"],
            "category":   t["category"],
            "priority":   t["priority"],
            "has_workaround": t.get("workaround") is not None,
        }
        processed.append({
            "ticket_id": t["ticket_id"],
            "text_blob": text_blob,
            "metadata":  metadata,
        })
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# JENKINS SIDE — Logs pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def strip_ansi_codes(log_text: str) -> str:
    """Remove ANSI colour escape sequences from raw Jenkins console output."""
    return ANSI_RE.sub("", log_text)


def split_by_stage(log_text: str) -> list[str]:
    """Split a Jenkins console log into per-stage chunks using [Pipeline] markers."""
    # Jenkins stage markers look like: [Pipeline] stage (Build)
    stage_pattern = re.compile(r"\[Pipeline\]\s*stage\s*\(([^)]+)\)")
    chunks, last_idx, last_name = [], 0, "PRELUDE"
    for match in stage_pattern.finditer(log_text):
        chunks.append((last_name, log_text[last_idx:match.start()]))
        last_name, last_idx = match.group(1), match.start()
    chunks.append((last_name, log_text[last_idx:]))
    return chunks


def extract_error_lines(log_text: str) -> str:
    """Keep only lines matching ERROR / WARN / EXCEPTION / Traceback / FAILED."""
    lines = log_text.splitlines()
    error_lines = [ln for ln in lines if ERROR_LINE_RE.search(ln)]
    return "\n".join(error_lines)


def preprocess_jenkins_log(log: dict) -> dict:
    """
    Full preprocessing for a single Jenkins build failure:
      1. Strip ANSI codes
      2. Split by stage (informational — failure stage isolated)
      3. Extract ERROR/WARN/EXCEPTION lines
      4. Truncate to TOKEN_CHAR_LIMIT characters
    Returns a clean text ready for the LLM extraction step (Prompt 1 & 2).
    """
    raw_log = log.get("console_log", "")
    cleaned = strip_ansi_codes(raw_log)

    stage_chunks  = split_by_stage(cleaned)
    failed_stage  = log.get("failed_stage", "")
    stage_text    = next((txt for name, txt in stage_chunks if name == failed_stage), cleaned)

    error_only = extract_error_lines(stage_text) or extract_error_lines(cleaned)
    if not error_only.strip():
        error_only = log.get("error_message", "")

    # Always prepend the structured error_message/failure_summary fields —
    # these came from the dataset and guarantee the LLM has something to work with
    context = f"Failure summary: {log.get('failure_summary','')}\n" \
              f"Component: {log.get('failure_component','')}\n" \
              f"Error: {log.get('error_message','')}\n" \
              f"--- Extracted log lines ---\n{error_only}"

    truncated = context[:TOKEN_CHAR_LIMIT]

    return {
        "build_id":           log["build_id"],
        "failed_stage":       failed_stage,
        "failure_category":   log["failure_category"],
        "failure_component":  log["failure_component"],
        "cleaned_text":       truncated,
    }


def preprocess_jenkins_logs(logs: list[dict]) -> list[dict]:
    return [preprocess_jenkins_log(l) for l in logs]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    datasets = load_data.load_all_datasets()

    print("\n[1/2] Preprocessing Jira tickets (concat title + description + resolution)...")
    jira_processed = preprocess_jira_tickets(datasets["jira_tickets"])
    print(f"  ✓ {len(jira_processed)} Jira tickets preprocessed")
    print(f"  Sample text_blob (first 200 chars):\n  {jira_processed[0]['text_blob'][:200]}...")

    print("\n[2/2] Preprocessing Jenkins logs (ANSI strip + stage chunk + error extract)...")
    jenkins_processed = preprocess_jenkins_logs(datasets["jenkins_logs"])
    print(f"  ✓ {len(jenkins_processed)} Jenkins logs preprocessed")
    print(f"  Sample cleaned_text (first 300 chars):\n  {jenkins_processed[0]['cleaned_text'][:300]}...")

    # Persist intermediate outputs so 03_embed_index.py can consume them directly
    with open(OUT_DIR / "jira_preprocessed.json", "w") as f:
        json.dump(jira_processed, f, indent=2)
    with open(OUT_DIR / "jenkins_preprocessed.json", "w") as f:
        json.dump(jenkins_processed, f, indent=2)

    print(f"\n✅ Preprocessing complete. Saved to:")
    print(f"   {OUT_DIR / 'jira_preprocessed.json'}")
    print(f"   {OUT_DIR / 'jenkins_preprocessed.json'}")
    print("   Proceed to 03_embed_index.py")
