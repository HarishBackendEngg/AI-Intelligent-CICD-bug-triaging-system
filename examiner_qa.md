# Dissertation Defense: Examiner Q&A Guide

This guide compiles the expected questions an examiner might ask during your dissertation defense for **"Automated Bug Deduplication and Workaround Retrieval in CI/CD Pipelines Using Retrieval-Augmented Generation"** and provides clear, technically precise answers based on your implementation.

---

## Table of Contents
1. [Core Architecture & RAG Choice](#1-core-architecture--rag-choice)
2. [Hybrid Search & Fusion Mechanism](#2-hybrid-search--fusion-mechanism)
3. [LLM Reasoning Chain & Prompt Engineering](#3-llm-reasoning-chain--prompt-engineering)
4. [State-Aware Decision Engine](#4-state-aware-decision-engine)
5. [Ground Truth Dataset Bug & Template Fingerprinting](#5-ground-truth-dataset-bug--template-fingerprinting)
6. [Evaluation, Threshold Sweeps & Strategy Comparison](#6-evaluation-threshold-sweeps--strategy-comparison)
7. [Engineering Decisions & Technical Stack](#7-engineering-decisions--technical-stack)

---

## 1. Core Architecture & RAG Choice

### Q1.1: Why did you use Retrieval-Augmented Generation (RAG) instead of fine-tuning a classifier model or training one from scratch?
*   **Answer**: 
    *   **Data Scarcity**: Training a classifier from scratch requires thousands of labelled failures per class. In production pipelines, bug categories change dynamically, and we only have a limited set of historical tickets.
    *   **Dynamic Knowledge updates**: If a new bug is filed or resolved in Jira, a RAG system indexes it instantly. A fine-tuned classifier would require expensive retraining cycles to learn about the new ticket.
    *   **Actionable Remediation**: Classification models only output a category label (e.g., `Class 42`). RAG retrieves the actual Jira ticket description and workaround field, allowing the LLM to extract concrete steps for the developer.

### Q1.2: How does the overall pipeline flow from a Jenkins failure to a developer verdict?
*   **Answer**: The pipeline is split into offline and online phases:
    1.  **Offline Ingestion**: Jira tickets are preprocessed into text blobs, embedded using a hybrid model, and stored in a local Qdrant collection.
    2.  **Online Triage**: 
        *   The raw Jenkins log is stripped of ANSI escape codes, split into stage chunks, and filtered for error patterns ([02_preprocess.py:L106-141](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/02_preprocess.py#L106-141)).
        *   **Prompt 1** summarizes the failure; **Prompt 2** extracts structured fields like error messages and components.
        *   **Hybrid Search** queries Qdrant using dense semantic vectors and sparse lexical vectors, returning the top-5 tickets.
        *   **Prompt 3** compares the failure against the candidates to judge if it is a duplicate.
        *   **State-Aware Decision Engine** inspects the candidate's status to route the ticket to `DUPLICATE`, `WORKAROUND_AVAILABLE`, or `NEW_ISSUE`.
        *   **Prompt 4** extracts the workaround list if the action is `WORKAROUND_AVAILABLE`.

---

## 2. Hybrid Search & Fusion Mechanism

### Q2.1: Why did you upgrade from dense-only semantic search to a hybrid search?
*   **Answer**: 
    *   **Semantic Compression Loss**: Transformer models (like BGE-large) compress text into dense vectors. While they excel at paraphrase similarity ("database down" $\approx$ "SQL connection timed out"), they compress rare alphanumeric tokens (e.g., error codes like `ISCSI_ERR_TCP_CONN_CLOSE` or ticket IDs like `PSTR-1242`) into vaguely similar vectors.
    *   **Keyword Precision**: Sparse retrieval (BM25) preserves exact-match keyword signals.
    *   **The Hybrid Solution**: By combining both, we get the best of both worlds: semantic understanding of symptoms (dense) + exact-match accuracy for trace identifiers and error codes (sparse).

### Q2.2: What is Reciprocal Rank Fusion (RRF), and why is it used instead of simply adding dense and sparse similarity scores?
*   **Answer**: 
    *   **Score Incompatibility**: Cosine similarity (dense) is bounded in $[-1, 1]$ (or $[0, 1]$ normalized), whereas BM25 (sparse) scores are unbounded and depend on document length and term frequency. Adding them directly is mathematically invalid.
    *   **Rank-Based Fusion**: RRF solves this by ignoring raw scores and focusing only on the *rank position* of a document in each retrieval branch. The RRF formula is:
        \[RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]
        where $M$ is the set of retrieval strategies (dense and sparse), $r_m(d)$ is the rank of document $d$ in strategy $m$, and $k$ is a constant (typically $60$) that smooths the impact of low-ranked items.
    *   This ensures a document ranked #1 in both strategies is prioritized without needing complex score calibration.

---

## 3. LLM Reasoning Chain & Prompt Engineering

### Q3.1: What are the specific roles of the 4 prompts in the RAG chain?
*   **Answer**:
    *   **Prompt 1 (Summarize)**: Condenses raw, noisy console output into a 3-sentence summary highlighting the failing component, error type, and symptoms ([04_rag_pipeline.py:L140-148](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/04_rag_pipeline.py#L140-148)).
    *   **Prompt 2 (JSON Extraction)**: Extracts structured fields (`bug_title`, `component`, `error_message`, `symptoms`, `category`) to standardize input for the retrieval database ([04_rag_pipeline.py:L159-171](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/04_rag_pipeline.py#L159-171)).
    *   **Prompt 3 (Duplicate Verdict)**: Acts as the decision judge, comparing the structured new bug fields against the top-$K$ retrieved Jira tickets to determine if there is a match ([04_rag_pipeline.py:L183-201](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/04_rag_pipeline.py#L183-201)).
    *   **Prompt 4 (Workaround Extraction)**: Parses resolved tickets to extract a clean, numbered list of actionable developer steps ([04_rag_pipeline.py:L213-221](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/04_rag_pipeline.py#L213-221)).

---

## 4. State-Aware Decision Engine

### Q4.1: How does the State-Aware Decision Engine operate, and why can't the LLM make the final action decision directly?
*   **Answer**: 
    *   **Stateful Rules**: The LLM's role is to verify *content similarity* (is this failure the same bug as that ticket?). The final business action depends on the *state* of the ticket in Jira, which the LLM shouldn't have to guess or hardcode.
    *   **The Matrix**: The engine routes the action programmatically:
        *   If a duplicate is found and its status is **Open/In Progress**, it routes to `DUPLICATE` (developers should link/subscribe to the existing ticket).
        *   If a duplicate is found and its status is **Resolved/Closed**, it routes to `WORKAROUND_AVAILABLE` (developers get immediate instructions on how to bypass the failure).
        *   If no duplicate exists above the similarity threshold, it routes to `NEW_ISSUE` (drafts a new bug report).
    *   This decouples logical comparison (LLM) from state-based workflow rules (deterministic code, see [04_rag_pipeline.py:L263-283](file:///Users/harisha/Downloads/final_project_post_evaluation/bug_triage_pipeline/scripts/04_rag_pipeline.py#L263-283)).

---

## 5. Ground Truth Dataset Bug & Template Fingerprinting

### Q5.1: Can you explain the dataset bug you discovered during evaluation, and how you fixed it?
*   **Answer**:
    *   **Symptom**: During initial runs, duplicate detection metrics were extremely low, and the similarity threshold sweep was unresponsive.
    *   **Root Cause**: Investigation revealed the synthetic-dataset generator linked Jenkins logs to a `related_jira_ticket` chosen *randomly* from the same category (e.g., any product failure), rather than one sharing the same error. 39 of 57 ground-truth duplicate pairs had **zero** keyword overlap, making the task mathematically unlearnable.
    *   **The Fix**: I wrote [fix_ground_truth.py](file:///Users/harisha/Downloads/final_project_post_evaluation/fix_ground_truth.py) to re-derive the linkages using **template fingerprinting**. It extracts stable error patterns, strips variable parameters, and re-maps true duplicates.
    *   **Result**: Post-fix keyword overlap jumped from **0.045 to 0.784**, and zero-overlap pairs fell to **0/82**, restoring the validation signal.

### Q5.2: What is template fingerprinting? How did you implement it?
*   **Answer**:
    *   **Concept**: Since logs and tickets were generated from about 30 underlying templates with variable tokens substituted in (like environment names, volume IDs, percentages, IP addresses), stripping these dynamic variables reveals the stable template phrase.
    *   **Implementation**: In [fix_ground_truth.py:L56-67](file:///Users/harisha/Downloads/final_project_post_evaluation/fix_ground_truth.py#L56-67), the `fingerprint` function:
        1.  Converts the text to lowercase.
        2.  Removes environment names (e.g., `powerstore-lab-01`).
        3.  Uses regex to strip alphanumeric IDs (`[a-z]*\d[a-z0-9]{3,}`) and numbers/percentages (`\d+(\.\d+)*%?`).
        4.  Collapses excess whitespace.
    *   Matching identical fingerprints guarantees that the log and ticket describe the exact same underlying failure.

---

## 6. Evaluation, Threshold Sweeps & Strategy Comparison

### Q6.1: How did the threshold sweep differ between "demo" mode and "production" mode?
*   **Answer**:
    *   **Production Mode**: Sweeps cosine similarity from BAAI/bge-large-en-v1.5 across $[0.70, 0.95]$. This is a standard similarity metric representing document proximity in vector space.
    *   **Demo Mode**: Uses RRF rank scores. Because RRF score ranges depend on rank ($1/(60+r)$) and are bounded by the number of branches (max $\approx 0.0328$), sweeping $0.70$–$0.95$ would result in $0\%$ duplicate predictions. Demo mode instead sweeps fractions of the maximum possible RRF score ($15\%$ to $75\%$, or $\approx 0.0049$ to $0.0246$) to ensure evaluation logic works end-to-end.

### Q6.2: How did the different prompting strategies (zero-shot, few-shot, CoT) perform, and what did it tell you?
*   **Answer**:
    *   **Zero-shot**: Directly asks if the failure is a duplicate. It runs fastest but struggles with nuanced differences (e.g., same component, different error codes).
    *   **Few-shot**: Includes 2-3 examples of duplicate and non-duplicate decisions in the system prompt. It improves precision but increases context size.
    *   **Chain-of-Thought (CoT)**: Instructs the model to generate its step-by-step reasoning *before* outputting the final JSON verdict (e.g., comparing error codes, components, then symptoms). It produces the highest F1 score by ensuring the LLM evaluates structural matches rather than surface-level terminology.

---

## 7. Engineering Decisions & Technical Stack

### Q7.1: Why did you choose Qwen3-8B and BAAI/bge-large-en-v1.5?
*   **Answer**:
    *   **Data Privacy & Compliance**: CI/CD logs and internal Jira tickets contain sensitive infrastructure names, IP addresses, and code snippets. Using local models via **Ollama** and **sentence-transformers** guarantees that no data is sent to external APIs (like OpenAI), adhering to corporate security guidelines.
    *   **Qwen3-8B**: Top-tier open-source reasoning model under Apache 2.0. It excels at JSON extraction, reasoning, and follows system instructions reliably on consumer hardware.
    *   **BGE-Large-en-v1.5**: Ranked highly on the MTEB (Massive Text Embedding Benchmark) for retrieval tasks, providing 1024-dimensional semantic coverage.

### Q7.2: Why did you choose a local, file-based Qdrant client over a running Docker container or Qdrant Cloud?
*   **Answer**:
    *   **Simplicity & Resource Constraints**: For a dissertation prototype, running a separate Docker service increases setup friction.
    *   **Local Storage**: Qdrant's file-based storage path (`qdrant_storage/`) runs natively inside Python, allowing database persistence and indexing without requiring external port bindings or service management. In production, this can be seamlessly swapped to a server-client model by changing the client initialization URL.
