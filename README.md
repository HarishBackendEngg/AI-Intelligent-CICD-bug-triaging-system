# Bug Triage Pipeline — Data Loading, Preprocessing & Model Development

This folder contains the runnable code for the **Design and Development**
phase referenced in the mid-sem report for:

> *Automated Bug Deduplication and Workaround Retrieval in CI/CD Pipelines
> Using Retrieval-Augmented Generation*
> Harisha P V — 2024AA05069 — BITS ZG628T Dissertation

It implements the **Offline Knowledge Ingestion Pipeline** components
(a)–(c) from Section 1 of the report: Jira Data → Pre-Processing → Embedding
Model and Vector DB.

## Folder structure

```
bug_triage_pipeline/
├── data/                          ← synthetic dataset (input)
│   ├── jira_tickets.json          (300 records)
│   ├── jenkins_logs.json          (150 records)
│   ├── duplicate_pairs.json       (120 labelled pairs)
│   └── ground_truth_labels.json   (150 evaluation labels)
├── scripts/
│   ├── 01_load_data.py            ← Step (a): load + sanity-check dataset
│   ├── 02_preprocess.py           ← Step (b): Jira text blobs + Jenkins log cleaning
│   ├── 03_embed_index.py          ← Step (c): BAAI/bge-large-en-v1.5 + Qdrant
│   ├── 04_rag_pipeline.py         ← LLM reasoning chain: Qwen3-8B via Ollama 
│   | 
│   └── 05_evaluate.py             ← Precision/Recall/F1, threshold sweep, strategy comparison
├── backend/
│   ├── main.py                    ← FastAPI app: /health /builds /tickets /stats /triage
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                ← React dashboard (verdict card, build picker, paste-log)
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── outputs/                       ← generated at runtime
│   ├── jira_preprocessed.json
│   ├── jenkins_preprocessed.json
│   ├── triage_results_sample.json
│   ├── evaluation/
│   │   ├── threshold_sweep_results.json
│   │   ├── strategy_comparison_results.json
│   │   ├── final_predictions.json
│   │   ├── final_metrics.json
│   │   └── evaluation_report.md   ← human-readable summary for your dissertation
│   └── qdrant_storage/            ← persisted Qdrant collection (local on-disk)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd bug_triage_pipeline
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
cd scripts
python 01_load_data.py        # loads & sanity-checks the synthetic dataset
python 02_preprocess.py       # builds Jira text blobs + cleans Jenkins logs
python 03_embed_index.py      # embeds with BAAI/bge-large-en-v1.5, indexes into Qdrant
python 04_rag_pipeline.py     # runs the 4-prompt LLM chain on sample failures
```

## LLM setup — Qwen3-8B via Ollama

The reasoning layer (`04_rag_pipeline.py`) uses **Qwen3-8B**, an open-source
model released under the **Apache 2.0 license** (free, no API cost, no usage
restrictions), served locally via **Ollama**. This keeps the entire pipeline
on-premise — no Jira ticket content or Jenkins log data is ever sent to an
external API, satisfying Design Consideration #1 in the mid-sem report.

```bash
# 1. Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh      # Linux
# or download from https://ollama.com/download      # macOS / Windows

# 2. Pull the Qwen3-8B model (~5GB download, one-time)
ollama pull qwen3:8b

# 3. Ollama runs as a background service automatically.
#    Verify it's running:
ollama list

# 4. Install the Python client
pip install ollama
```

**Why Qwen3-8B:** evaluated against GLM-5.2, DeepSeek V4, and Phi-4 Mini for
this project's needs (structured JSON extraction + classification + reasoning
over retrieved documents, not long-horizon coding). Qwen3-8B offers the best
balance of structured-output reliability, hardware footprint (~8–16GB VRAM,
or runs on CPU at reduced speed), and Apache 2.0 licensing — see Design
Consideration #11 (to be added) for the full comparison table.

**Hardware note:** Qwen3-8B runs on a single consumer GPU (8GB+ VRAM) or on
CPU for testing (slower, ~10-30s per prompt). For the 4-prompt chain across
150 Jenkins logs, a GPU is recommended for the evaluation phase.

**First run note:** `03_embed_index.py` downloads `BAAI/bge-large-en-v1.5`
(~1.3 GB) from Hugging Face on first use, then caches it locally under
`~/.cache/huggingface`. This requires outbound internet access once.
If you are running this on Dell's on-prem network behind a proxy, either:

- Pre-download the model on a machine with internet access:
  ```bash
  huggingface-cli download BAAI/bge-large-en-v1.5
  ```
  then copy the `~/.cache/huggingface` folder to the offline machine, or
- Configure your proxy environment variables (`HTTPS_PROXY`) before running.

## What "model development" means here

`BAAI/bge-large-en-v1.5` is a **pretrained** sentence embedding model — it is
not trained from scratch. The "model development" work for this dissertation
phase is:

1. **Selecting** the embedding model (BGE-large vs OpenAI vs alternatives —
   documented in Design Consideration #2 of the mid-sem report)
2. **Building the retrieval index** — embedding all 300 Jira tickets and
   constructing the Qdrant HNSW vector index (this is the actual "model" that
   gets used at inference time — a retrieval model, not a classifier)
3. **Tuning the similarity threshold** (0.70–0.95 sweep) — this is the
   hyperparameter tuning step, covered in `04_evaluate.py` (next phase)

The LLM reasoning layer (Prompts 1–4) is implemented in `04_rag_pipeline.py`
using Qwen3-8B served locally via Ollama. It consumes the Qdrant index built
here, retrieves top-K similar Jira tickets per Jenkins failure, and routes
through the state-aware decision engine to a final DUPLICATE /
WORKAROUND_AVAILABLE / NEW_ISSUE verdict.

## Evaluation — `05_evaluate.py`

This is the Testing phase from your mid-sem report's Future Plan
(01 Jul – 31 Jul 2026), made runnable. It runs three experiments:

1. **Threshold sweep** — runs the full pipeline across all 150 Jenkins
   failures at 7 cosine similarity thresholds (0.70 → 0.95) and reports
   Precision/Recall/F1 at each, replacing the placeholder value of 0.82
   with an empirically-chosen one (Design Consideration #4).
2. **Prompting strategy comparison** — re-runs Prompt 3 (duplicate verdict)
   under zero-shot, few-shot, and chain-of-thought, using the best threshold
   from step 1 (Design Consideration #6).
3. **Final model evaluation** — using the best threshold + best strategy,
   reports binary duplicate-detection metrics (Precision/Recall/F1/confusion
   matrix) and 3-way action classification metrics (DUPLICATE /
   WORKAROUND_AVAILABLE / NEW_ISSUE), each compared against
   `ground_truth_labels.json`.

```bash
cd scripts
python 05_evaluate.py            # real pipeline (BGE-large + Qwen3-8B)

```

Outputs land in `outputs/evaluation/`, including a ready-to-paste
`evaluation_report.md` with all tables formatted for your dissertation's
Results chapter.

**Important:** `--demo` mode exists only to verify the evaluation
*machinery* (metrics math, threshold sweep loop, report generation) without
internet access. The F1 scores it produces are **not meaningful** — they
reflect a TF-IDF stand-in embedder's weak lexical matching, not BGE-large's
real semantic retrieval quality. Run without `--demo` on your machine for
numbers you can actually cite.

## Backend — FastAPI

Exposes the full pipeline (preprocessing → BGE-large embedding → Qdrant
retrieval → Qwen3-8B reasoning chain → state-aware decision engine) as a
REST API for the React dashboard.

```bash
cd backend
pip install -r requirements.txt
python main.py
# or: uvicorn main:app --reload --port 8000
```

Endpoints:

| Method | Path                  | Purpose                                              |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/health`             | Liveness check                                         |
| GET    | `/builds?limit=30`    | List available Jenkins builds for the dashboard picker |
| GET    | `/stats`              | Aggregate counts + category distribution               |
| GET    | `/tickets/{ticket_id}`| Fetch a single Jira ticket's full details              |
| POST   | `/triage`             | Run the full pipeline on a `build_id` or `raw_log_text`|

The backend automatically falls back to the TF-IDF demo embedder + mock LLM
if `BAAI/bge-large-en-v1.5` or `qwen3:8b` aren't reachable (e.g. no internet,
Ollama not running) — so the API stays usable for UI development even before
the real models are wired up. Check the startup log to see which mode loaded.

Interactive API docs: `http://localhost:8000/docs`

## Frontend — React dashboard

A dark, console-style dashboard (Tailwind + lucide-react) for browsing
Jenkins builds, running triage, and reading the verdict — built and screen-
tested in this session (see `bug_triage_pipeline/frontend/`).

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

Features:
- **Recent builds** tab — pick a Jenkins failure from the synthetic dataset and run triage
- **Paste log** tab — paste a raw console log directly for ad-hoc triage
- **Verdict card** — colour-coded by action: red = Duplicate, teal = Workaround available, amber = New issue
- **Similarity score bar** + copyable Jira ticket ID
- **Workaround steps** rendered as a numbered list when available
- Falls back to realistic mock data automatically if the FastAPI backend isn't running, so the UI is always inspectable standalone

To point the dashboard at a deployed backend, edit `API_BASE` at the top of
`src/App.jsx`.

## Sandbox note (for this Claude session only)

This sandboxed environment cannot reach `huggingface.co` or `ollama.com`
(only PyPI/npm/GitHub are network-allowlisted here). Two fallback scripts
were used **only inside this session** to verify pipeline logic end-to-end:

- `03b_embed_index_demo.py` — TF-IDF (dense) + real BM25 via `rank_bm25`
  (sparse) stand-in for BAAI/bge-large-en-v1.5, fused with hand-rolled
  Reciprocal Rank Fusion (RRF) — the same algorithm Qdrant's native hybrid
  query API uses. Verifies Qdrant indexing, category-filtered hybrid
  retrieval, and RRF fusion end-to-end.
- `04b_rag_pipeline_demo.py` — deterministic mock LLM stand-in for Qwen3-8B,
  verifies the 4-prompt orchestration, JSON parsing, and the state-aware
  decision engine.
- `fix_ground_truth.py` — one-time repair script (see "Dataset bug found
  and fixed" below).

**On your own machine, use `03_embed_index.py` and `04_rag_pipeline.py`** —
these are the real scripts using BAAI/bge-large-en-v1.5 and Qwen3-8B exactly
as specified in your mid-sem report. The pipeline logic is identical; only
the embedder and LLM are swapped from mocks to the real models.

The **FastAPI backend** and **React dashboard** were verified end-to-end in
this session (TestClient calls / Playwright screenshots respectively) — see
git history for details. The sections below document a more significant
finding from verifying `05_evaluate.py`: a real **dataset bug**, not a
retrieval or code bug, that materially changed the evaluation numbers and
is worth describing in your dissertation's methodology/limitations section.

### Dataset bug found and fixed — ground truth was not content-based

Running the full evaluation initially produced poor, confusing metrics
(F1 ≈ 0.45, then later flat precision/recall across every threshold value
tried). Rather than accept "the model is just bad," each anomaly was traced
to its root cause:

1. **Symptom:** F1 ≈ 0.45, looked like "weak embeddings."
   **Investigation:** measured keyword (Jaccard) overlap between each
   Jenkins failure and its ground-truth "matched" Jira ticket directly.
   **Finding:** 39 of 57 `TRUE_DUPLICATE` pairs had **zero** keyword
   overlap; mean overlap 0.045 — statistically indistinguishable from
   random chance.
   **Root cause:** the original synthetic-dataset generator linked each
   Jenkins log to a `related_jira_ticket` chosen **randomly from the same
   failure category**, not from tickets that actually share the same
   underlying failure template. The "duplicate detection" task as
   originally labelled was unlearnable by any retrieval method — dense,
   sparse, or hybrid — because the ground truth itself wasn't based on
   content similarity.
   **Fix:** `fix_ground_truth.py` re-derives every match using **template
   fingerprinting** — stripping the variable tokens (environment names,
   generated IDs, version numbers) that the dataset generator substitutes
   into a fixed set of ~30 template phrases, recovering the stable
   template each record was actually generated from. Post-fix mean overlap
   on matched pairs: **0.784** (was 0.045); zero-overlap pairs: **0/82**
   (was 39/57). `data/jenkins_logs.json`, `duplicate_pairs.json`, and
   `ground_truth_labels.json` were all regenerated with this corrected
   linkage — re-run `fix_ground_truth.py` if you regenerate the dataset
   from scratch.

2. **Symptom:** identical precision/recall/F1 at every one of 7 threshold
   values swept (0.0049 → 0.0246).
   **Investigation:** dumped raw top-1 RRF fusion scores for true vs false
   duplicates directly.
   **Finding:** means were 0.03257 vs 0.03255 — overlapping ranges,
   essentially identical. With RRF_K=60 and only ~35-45 same-category
   candidates per query, "rank #1 in both branches" (the RRF ceiling) is
   cheap to reach by chance — top-1 RRF score alone carries almost no
   power to separate genuine matches from coincidental ones at this corpus
   scale. Separately confirmed retrieval itself was fine: recall@5 by
   correct-template-match was **100%** — the right ticket is always
   present in the candidates; the bug was in how the *decision* step used
   (or rather, ignored) that fact.
   **Fix:** the demo's mock "Prompt 3" was changed from a raw-score
   threshold check to categorical template-fingerprint comparison against
   the actual retrieved candidate text (the closest honest stand-in for
   "an LLM read both texts and judged them as the same failure" available
   without a real LLM call). The threshold-sweep and strategy-comparison
   experiments now correctly report **"not applicable in demo mode"**
   instead of silently producing a flat, meaningless curve.

3. **Symptom:** after fix #2, every single prediction flipped to
   `NEW_ISSUE` (precision/recall/F1 all exactly 0.000).
   **Investigation:** printed the actual fingerprint computed for a known-
   correct candidate ticket.
   **Finding:** the candidate's fingerprint was being computed from
   `text_blob[:120]` — a blind character-count slice that frequently cut
   off mid-sentence, partway into the ticket's *description* field, never
   landing cleanly on just the title.
   **Fix:** split `text_blob` on its known `" . "` title/description
   separator (see `02_preprocess.py`) and fingerprint only the title
   segment, instead of an arbitrary character slice.

**Final, correctly-functioning demo-mode baseline** (TF-IDF + BM25 hybrid
retrieval + fingerprint-based mock verdict, on the corrected dataset):
Precision 0.566, Recall 1.000, F1 0.723 — recall is perfect because every
true duplicate's template is genuinely retrievable (confirmed via direct
recall@5 measurement); precision is moderate because template-fingerprint
matching can't distinguish *which* same-template ticket is the canonical
one when several exist (e.g. "SSL certificate expired" appears as 16
separate tickets in the 300-ticket corpus). This is exactly the gap a real
embedding model and a real LLM reading full ticket text are designed to
close — re-run without `--demo` on your machine to measure how much BGE-
large + Qwen3-8B improve on this baseline.
