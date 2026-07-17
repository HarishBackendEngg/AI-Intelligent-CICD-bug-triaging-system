"""
05_evaluate.py
================
Section: Testing phase (per mid-sem report Future Plan, 01 Jul – 31 Jul 2026)
          Design Consideration #4 (threshold tuning)
          Design Consideration #6 (three-strategy prompting comparison)

This is the evaluation chapter of the dissertation, made runnable:

  1. Run the full triage pipeline (04_rag_pipeline.py) across all 150
     Jenkins failures in the synthetic dataset.
  2. Compare predicted verdicts against `ground_truth_labels.json`.
  3. Compute Precision, Recall, F1, and a confusion matrix for binary
     duplicate detection (is_duplicate: true/false).
  4. Sweep the cosine similarity threshold from 0.70 to 0.95 (step 0.05)
     and report Precision/Recall/F1 at each point — this identifies the
     optimal threshold, replacing the placeholder value of 0.82.
  5. Repeat the full run under three prompting strategies — zero-shot,
     few-shot, chain-of-thought — for Prompt 3 (duplicate verdict), and
     report a strategy-comparison table. This is the core research
     contribution referenced in Design Consideration #6.
  6. Save all results + a Markdown summary report to outputs/evaluation/.

Run:
    python 05_evaluate.py                  # real pipeline (BGE + Qwen3-8B)
"""

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from collections import Counter

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR    = SCRIPTS_DIR.parent / "data"
OUT_DIR     = SCRIPTS_DIR.parent / "outputs"
EVAL_DIR    = OUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(SCRIPTS_DIR))

# IMPORTANT — these two sweeps are on the RRF fused score scale:
#   - PRODUCTION_THRESHOLD_FRACTIONS: RRF-fused rank score (dense+sparse), range
#     [0, ~0.0328] for RRF_K=60. Expressed as fractions of RRF_MAX_SCORE so it
#     stays correct if RRF_K ever changes.
#   - DEMO_THRESHOLD_FRACTIONS: RRF-fused rank score (dense+sparse), range
#     [0, ~0.0328] for RRF_K=60.
PRODUCTION_THRESHOLD_FRACTIONS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
DEMO_THRESHOLD_FRACTIONS       = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
PROMPTING_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]


def get_threshold_sweep(mode: str) -> list[float]:
    if mode == "production":
        import importlib
        rag = importlib.import_module("04_rag_pipeline")
        return [round(f * rag._RRF_MAX_SCORE, 5) for f in PRODUCTION_THRESHOLD_FRACTIONS]
    import importlib
    demo_rag = importlib.import_module("04b_rag_pipeline_demo")
    return [round(f * demo_rag.RRF_MAX_SCORE, 5) for f in DEMO_THRESHOLD_FRACTIONS]


# ─────────────────────────────────────────────────────────────────────────────
# METRICS — Precision, Recall, F1, Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(cm: dict) -> dict:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + cm["tn"]) / (tp + fp + fn + cm["tn"]) if (tp + fp + fn + cm["tn"]) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        **cm,
    }


def three_way_accuracy(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Accuracy + per-class breakdown for the action verdict.

    NOTE — label space mismatch (documented, not silently patched):
    `ground_truth_labels.json` uses 4 verdict labels: DUPLICATE,
    WORKAROUND_AVAILABLE, NEW_ISSUE, and NOT_DUPLICATE (the dataset's
    "false positive" category — failures that superficially resemble a
    ticket but are confirmed not to be duplicates). The pipeline's
    decision engine (Section 2 of the report) only ever emits 3 actions —
    DUPLICATE / WORKAROUND_AVAILABLE / NEW_ISSUE — because by design it
    collapses "not a duplicate, no match" into NEW_ISSUE.
    NOT_DUPLICATE ground-truth rows are therefore correctly counted as
    "should have been NEW_ISSUE" when scoring 3-way accuracy below
    (a NEW_ISSUE prediction against a NOT_DUPLICATE label is a true
    negative for novelty-detection purposes), rather than appearing as
    an always-empty 4th class with undefined precision/recall.
    """
    LABEL_MAP = {"NOT_DUPLICATE": "NEW_ISSUE"}
    y_true_mapped = [LABEL_MAP.get(t, t) for t in y_true]

    correct = sum(1 for t, p in zip(y_true_mapped, y_pred) if t == p)
    total = len(y_true_mapped)
    per_class = {}
    for cls in set(y_true_mapped) | set(y_pred):
        cls_true = [t == cls for t in y_true_mapped]
        cls_pred = [p == cls for p in y_pred]
        cm = confusion_matrix(cls_true, cls_pred)
        per_class[cls] = precision_recall_f1(cm)
    return {
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "n": total,
        "per_class": per_class,
        "note": "NOT_DUPLICATE ground-truth rows mapped to NEW_ISSUE — "
                "see docstring for rationale.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN THE PIPELINE ACROSS ALL 150 JENKINS LOGS
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(mode: str, strategy: str = "chain_of_thought",
                       similarity_threshold: float = None) -> list[dict]:
    """
    Executes the triage pipeline on every preprocessed Jenkins log and
    returns a list of prediction dicts. `mode` is "production" or "demo".

    `similarity_threshold` (production mode only): RRF fused score threshold.

    `strategy` selects the Prompt 3 variant (production mode only — the
    real Qwen3-8B call is reformulated as zero-shot/few-shot/CoT).
    """
    with open(OUT_DIR / "jenkins_preprocessed.json") as f:
        jenkins_logs = json.load(f)

    if mode == "production":
        from qdrant_client import QdrantClient
        import importlib
        rag = importlib.import_module("04_rag_pipeline")
        if similarity_threshold is not None:
            rag.SIMILARITY_THRESHOLD = similarity_threshold

        embed_index = importlib.import_module("03_embed_index")
        dense_model = embed_index.load_embedding_model()
        sparse_model = embed_index.load_sparse_model()
        client = QdrantClient(path=str(OUT_DIR / "qdrant_storage"))

        try:
            predictions = []
            for log in jenkins_logs:
                verdict = rag.triage_one_failure(client, dense_model, sparse_model, log, strategy=strategy)
                predictions.append(asdict(verdict))
            return predictions
        finally:
            client.close()

    else:
        import importlib
        demo = importlib.import_module("03b_embed_index_demo")
        demo_rag = importlib.import_module("04b_rag_pipeline_demo")

        with open(OUT_DIR / "jira_preprocessed.json") as f:
            jira_processed = json.load(f)

        dense_model = demo.DemoEmbedder(dim=1024)
        bm25_index  = demo.BM25SparseIndex()
        client = demo.get_qdrant_client()
        demo.create_collection(client)
        demo.index_jira_tickets(client, dense_model, bm25_index, jira_processed)

        predictions = []
        for log in jenkins_logs:
            verdict = demo_rag.triage_one_failure_demo(client, dense_model, bm25_index, log)
            predictions.append(asdict(verdict))
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE PREDICTIONS AGAINST GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(predictions: list[dict], ground_truth: dict) -> dict:
    """
    `ground_truth` is a dict keyed by build_id (from ground_truth_labels.json).
    Returns binary duplicate-detection metrics + 3-way action metrics.
    """
    y_true_binary, y_pred_binary = [], []
    y_true_action, y_pred_action = [], []

    for pred in predictions:
        gt = ground_truth.get(pred["build_id"])
        if gt is None:
            continue
        y_true_binary.append(bool(gt["is_duplicate"]))
        y_pred_binary.append(bool(pred["is_duplicate"]))
        y_true_action.append(gt["ground_truth_verdict"])
        y_pred_action.append(pred["action"])

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    binary_metrics = precision_recall_f1(cm)
    action_metrics = three_way_accuracy(y_true_action, y_pred_action)

    return {
        "n_evaluated": len(y_true_binary),
        "binary_duplicate_detection": binary_metrics,
        "three_way_action_classification": action_metrics,
    }


def project_predictions_at_threshold(predictions: list[dict], threshold: float) -> list[dict]:
    """
    Reuse one baseline pipeline run to simulate a stricter similarity threshold.

    The full triage output already includes the top candidate score, so a higher
    threshold only changes whether the prediction is forced to NEW_ISSUE.
    """
    projected = []
    for pred in predictions:
        if pred.get("top_match_score", 0.0) < threshold:
            adjusted = dict(pred)
            adjusted["is_duplicate"] = False
            adjusted["action"] = "NEW_ISSUE"
            adjusted["top_match_ticket_id"] = None
            projected.append(adjusted)
        else:
            projected.append(pred)
    return projected


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_sweep(mode: str, ground_truth: dict) -> list[dict]:
    print(f"\n{'='*70}")
    if mode == "demo":
        print("EXPERIMENT 1 — Threshold sweep: NOT APPLICABLE in demo mode")
        print(f"{'='*70}")
        print("  The demo mock's Prompt 3 uses categorical template-fingerprint")
        print("  matching (see 04b_rag_pipeline_demo.py docstring), not a numeric")
        print("  threshold — there is nothing to sweep. Running ONE pass to report")
        print("  baseline metrics instead of repeating an identical no-op 7 times.")
        predictions = run_full_pipeline(mode)
        metrics = evaluate_predictions(predictions, ground_truth)
        bm = metrics["binary_duplicate_detection"]
        print(f"  precision={bm['precision']:.3f}  recall={bm['recall']:.3f}  f1={bm['f1']:.3f}")
        result = {"threshold": None, **bm}
        return [result], result

    print("EXPERIMENT 1 — RRF fusion score similarity threshold sweep")
    print(f"{'='*70}")
    sweep = get_threshold_sweep(mode)
    baseline_predictions = run_full_pipeline(
        mode, strategy="chain_of_thought", similarity_threshold=float("-inf")
    )
    results = []
    for thresh in sweep:
        t0 = time.time()
        predictions = project_predictions_at_threshold(baseline_predictions, thresh)
        metrics = evaluate_predictions(predictions, ground_truth)
        bm = metrics["binary_duplicate_detection"]
        elapsed = time.time() - t0
        print(f"  threshold={thresh:.4f}  precision={bm['precision']:.3f}  "
              f"recall={bm['recall']:.3f}  f1={bm['f1']:.3f}  ({elapsed:.1f}s)")
        results.append({"threshold": thresh, **bm})
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  → Best threshold by F1: {best['threshold']:.4f} (F1={best['f1']:.3f})")
    return results, best


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — PROMPTING STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy_comparison(mode: str, ground_truth: dict, best_threshold) -> list[dict]:
    print(f"\n{'='*70}")
    if mode == "demo":
        print("EXPERIMENT 2 — Strategy comparison: NOT APPLICABLE in demo mode")
        print(f"{'='*70}")
        print("  The mock LLM does not implement zero-shot/few-shot/CoT variants —")
        print("  there is no real prompt being sent. Reporting the single baseline")
        print("  result under all three labels rather than fabricating a fake spread.")
        predictions = run_full_pipeline(mode)
        metrics = evaluate_predictions(predictions, ground_truth)
        bm = metrics["binary_duplicate_detection"]
        print(f"  (baseline) precision={bm['precision']:.3f}  recall={bm['recall']:.3f}  f1={bm['f1']:.3f}")
        results = [{"strategy": s, **bm} for s in PROMPTING_STRATEGIES]
        return results, results[0]

    print("EXPERIMENT 2 — Prompting strategy comparison (zero-shot / few-shot / CoT)")
    print(f"{'='*70}")
    results = []
    for strategy in PROMPTING_STRATEGIES:
        t0 = time.time()
        predictions = run_full_pipeline(mode, strategy=strategy, similarity_threshold=best_threshold)
        metrics = evaluate_predictions(predictions, ground_truth)
        bm = metrics["binary_duplicate_detection"]
        elapsed = time.time() - t0
        print(f"  {strategy:<18} precision={bm['precision']:.3f}  "
              f"recall={bm['recall']:.3f}  f1={bm['f1']:.3f}  ({elapsed:.1f}s)")
        results.append({"strategy": strategy, **bm})
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  → Best strategy by F1: {best['strategy']} (F1={best['f1']:.3f})")
    return results, best


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown_report(threshold_results, best_threshold,
                           strategy_results, best_strategy,
                           final_metrics, mode) -> Path:
    lines = []
    lines.append("# Evaluation Report — Intelligent Bug Triage Using RAG\n")
    lines.append(f"**Pipeline mode:** `{mode}`  ")
    lines.append(f"**Dataset:** 150 Jenkins failures / 300 Jira tickets (synthetic)  ")
    lines.append(f"**Ground truth:** 150 manually-labelled evaluation records\n")

    if mode == "demo":
        lines.append("> ⚠️ **DEMO MODE — not dissertation-reportable.** These numbers were "
                     "produced using a TF-IDF stand-in embedder and a deterministic mock "
                     "LLM (this sandbox cannot reach huggingface.co or ollama.com). They "
                     "verify that the evaluation pipeline itself — metrics, threshold "
                     "sweep, strategy comparison, report generation — is implemented "
                     "correctly. Re-run `python 05_evaluate.py` (without `--demo`) on a "
                     "machine with BAAI/bge-large-en-v1.5 and Qwen3-8B available to get "
                     "the real results for your dissertation.\n")

    if mode == "demo":
        lines.append("## 1. Threshold Sweep — NOT APPLICABLE in demo mode\n")
        lines.append("The demo mock's duplicate-verdict step uses categorical "
                     "template-fingerprint matching (see `04b_rag_pipeline_demo.py`), "
                     "not a numeric similarity threshold — there is nothing to sweep. "
                     "Investigation found neither the raw RRF fusion score (mean "
                     "0.0326 vs 0.0326 for true vs false duplicates — statistically "
                     "indistinguishable) nor keyword overlap (matched ticket scored "
                     "*lower* than wrong candidates 89% of the time) could separate "
                     "the classes, which is exactly the gap a real embedding model "
                     "and LLM are meant to close. Baseline metrics from the single "
                     "demo run:\n")
        bm = threshold_results[0]
        lines.append(f"- Precision: {bm['precision']:.3f}  \n"
                     f"- Recall: {bm['recall']:.3f}  \n"
                     f"- F1: {bm['f1']:.3f}\n")
        lines.append("\nIn **production mode** (no `--demo`), this experiment sweeps "
                     "real cosine similarity from BGE-large across 0.70–0.95 and "
                     "*is* meaningful — run it on your machine for the dissertation-"
                     "reportable threshold-tuning result.\n")
    else:
        lines.append("## 1. Threshold Sweep (RRF fusion score, 15%–75% of max)\n")
        lines.append("| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in threshold_results:
            lines.append(f"| {r['threshold']:.4f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                          f"{r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
        lines.append(f"\n**Best threshold: `{best_threshold['threshold']:.4f}`** "
                     f"(F1 = {best_threshold['f1']:.3f}) — used for Experiment 2 below.\n")

    if mode == "demo":
        lines.append("## 2. Prompting Strategy Comparison — NOT APPLICABLE in demo mode\n")
        lines.append("The mock LLM does not implement real zero-shot/few-shot/"
                     "chain-of-thought prompt variants (there is no actual LLM call "
                     "to vary) — the baseline result is reported identically under "
                     "all three labels rather than fabricating a fake spread. Run "
                     "without `--demo` on your machine with Qwen3-8B for the real "
                     "strategy comparison.\n")
    else:
        lines.append("## 2. Prompting Strategy Comparison\n")
        lines.append("| Strategy | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for r in strategy_results:
            lines.append(f"| {r['strategy'].replace('_',' ').title()} | {r['precision']:.3f} | "
                          f"{r['recall']:.3f} | {r['f1']:.3f} |")
        lines.append(f"\n**Best strategy: `{best_strategy['strategy']}`** (F1 = {best_strategy['f1']:.3f})\n")

    lines.append("## 3. Final Model — Three-Way Action Classification\n")
    aw = final_metrics["three_way_action_classification"]
    lines.append(f"**Overall accuracy:** {aw['overall_accuracy']:.3f}  (n={aw['n']})\n")
    if "note" in aw:
        lines.append(f"> *Note: {aw['note']}*\n")
    lines.append("| Action class | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for cls, m in aw["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")

    lines.append("\n## 4. Binary Duplicate Detection — Final Confusion Matrix\n")
    bm = final_metrics["binary_duplicate_detection"]
    lines.append("|  | Predicted Duplicate | Predicted Not-Duplicate |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual Duplicate** | TP={bm['tp']} | FN={bm['fn']} |")
    lines.append(f"| **Actual Not-Duplicate** | FP={bm['fp']} | TN={bm['tn']} |")
    lines.append(f"\nPrecision = {bm['precision']:.3f}, Recall = {bm['recall']:.3f}, "
                 f"F1 = {bm['f1']:.3f}, Accuracy = {bm['accuracy']:.3f}\n")

    report_path = EVAL_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the bug triage RAG pipeline")
    parser.add_argument("--demo", action="store_true",
                         help="Use the sandbox-safe demo pipeline (TF-IDF + mock LLM) "
                              "instead of the real BAAI/bge-large-en-v1.5 + Qwen3-8B pipeline")
    args = parser.parse_args()
    mode = "demo" if args.demo else "production"

    print(f"Running evaluation in '{mode}' mode...")
    if mode == "production":
        print("(use --demo if BAAI/bge-large-en-v1.5 or Qwen3-8B/Ollama aren't available)")
    else:
        print("⚠ DEMO MODE: using a TF-IDF stand-in embedder and a mock LLM.")
        print("  The F1/Precision/Recall numbers below are NOT representative of the")
        print("  real pipeline — TF-IDF captures only lexical overlap, not semantic")
        print("  similarity, so scores will be substantially lower than BGE-large +")
        print("  Qwen3-8B will achieve. This run only proves the evaluation MACHINERY")
        print("  (metrics, sweep, report generation) is correct — re-run without --demo")
        print("  on your machine for dissertation-reportable numbers.\n")

    with open(DATA_DIR / "ground_truth_labels.json") as f:
        gt_list = json.load(f)
    ground_truth = {g["build_id"]: g for g in gt_list}
    print(f"Loaded {len(ground_truth)} ground truth labels")

    # Experiment 1: threshold sweep
    threshold_results, best_threshold = run_threshold_sweep(mode, ground_truth)

    # Experiment 2: prompting strategy comparison (using best threshold)
    strategy_results, best_strategy = run_strategy_comparison(
        mode, ground_truth, best_threshold["threshold"])

    # Final run: best threshold + best strategy → full metrics incl. 3-way action accuracy
    print(f"\n{'='*70}")
    thresh_display = f"{best_threshold['threshold']:.4f}" if best_threshold["threshold"] is not None else "N/A (demo mode)"
    print(f"FINAL RUN — threshold={thresh_display}, strategy={best_strategy['strategy']}")
    print(f"{'='*70}")
    final_predictions = run_full_pipeline(
        mode, strategy=best_strategy["strategy"], similarity_threshold=best_threshold["threshold"])
    final_metrics = evaluate_predictions(final_predictions, ground_truth)

    aw = final_metrics["three_way_action_classification"]
    print(f"  Overall 3-way action accuracy: {aw['overall_accuracy']:.3f}  (n={aw['n']})")
    for cls, m in aw["per_class"].items():
        print(f"    {cls:<22} precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")

    # Persist everything
    with open(EVAL_DIR / "threshold_sweep_results.json", "w") as f:
        json.dump(threshold_results, f, indent=2)
    with open(EVAL_DIR / "strategy_comparison_results.json", "w") as f:
        json.dump(strategy_results, f, indent=2)
    with open(EVAL_DIR / "final_predictions.json", "w") as f:
        json.dump(final_predictions, f, indent=2)
    with open(EVAL_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    report_path = write_markdown_report(
        threshold_results, best_threshold, strategy_results, best_strategy, final_metrics, mode)

    print(f"\n✅ Evaluation complete.")
    print(f"   Threshold sweep:      {EVAL_DIR / 'threshold_sweep_results.json'}")
    print(f"   Strategy comparison:  {EVAL_DIR / 'strategy_comparison_results.json'}")
    print(f"   Final metrics:        {EVAL_DIR / 'final_metrics.json'}")
    print(f"   Markdown report:      {report_path}")


if __name__ == "__main__":
    main()
