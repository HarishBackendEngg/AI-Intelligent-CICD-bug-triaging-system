# Evaluation Report — Intelligent Bug Triage Using RAG

**Pipeline mode:** `demo`  
**Dataset:** 150 Jenkins failures / 300 Jira tickets (synthetic)  
**Ground truth:** 150 manually-labelled evaluation records

> ⚠️ **DEMO MODE — not dissertation-reportable.** These numbers were produced using a TF-IDF stand-in embedder and a deterministic mock LLM (this sandbox cannot reach huggingface.co or ollama.com). They verify that the evaluation pipeline itself — metrics, threshold sweep, strategy comparison, report generation — is implemented correctly. Re-run `python 05_evaluate.py` (without `--demo`) on a machine with BAAI/bge-large-en-v1.5 and Qwen3-8B available to get the real results for your dissertation.

## 1. Threshold Sweep — NOT APPLICABLE in demo mode

The demo mock's duplicate-verdict step uses categorical template-fingerprint matching (see `04b_rag_pipeline_demo.py`), not a numeric similarity threshold — there is nothing to sweep. Investigation found neither the raw RRF fusion score (mean 0.0326 vs 0.0326 for true vs false duplicates — statistically indistinguishable) nor keyword overlap (matched ticket scored *lower* than wrong candidates 89% of the time) could separate the classes, which is exactly the gap a real embedding model and LLM are meant to close. Baseline metrics from the single demo run:

- Precision: 0.976  
- Recall: 1.000  
- F1: 0.988


In **production mode** (no `--demo`), this experiment sweeps real cosine similarity from BGE-large across 0.70–0.95 and *is* meaningful — run it on your machine for the dissertation-reportable threshold-tuning result.

## 2. Prompting Strategy Comparison — NOT APPLICABLE in demo mode

The mock LLM does not implement real zero-shot/few-shot/chain-of-thought prompt variants (there is no actual LLM call to vary) — the baseline result is reported identically under all three labels rather than fabricating a fake spread. Run without `--demo` on your machine with Qwen3-8B for the real strategy comparison.

## 3. Final Model — Three-Way Action Classification

**Overall accuracy:** 0.827  (n=150)

> *Note: NOT_DUPLICATE ground-truth rows mapped to NEW_ISSUE — see docstring for rationale.*

| Action class | Precision | Recall | F1 |
|---|---|---|---|
| WORKAROUND_AVAILABLE | 0.682 | 0.878 | 0.768 |
| DUPLICATE | 0.714 | 0.455 | 0.556 |
| NEW_ISSUE | 1.000 | 0.971 | 0.985 |

## 4. Binary Duplicate Detection — Final Confusion Matrix

|  | Predicted Duplicate | Predicted Not-Duplicate |
|---|---|---|
| **Actual Duplicate** | TP=82 | FN=0 |
| **Actual Not-Duplicate** | FP=2 | TN=66 |

Precision = 0.976, Recall = 1.000, F1 = 0.988, Accuracy = 0.987
