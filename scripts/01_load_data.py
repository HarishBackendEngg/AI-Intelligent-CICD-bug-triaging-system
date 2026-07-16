"""
01_load_data.py
================
Section: Offline Knowledge Ingestion Pipeline — Step (a) Jira Data
                Online Live Triage Pipeline      — Step (a) Jenkins console logs

Loads the synthetic dataset (jira_tickets.json, jenkins_logs.json) generated
for the "Automated Bug Deduplication and Workaround Retrieval" project and
performs basic sanity checks before handing off to preprocessing.

Run:
    python 01_load_data.py
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_json(filename: str) -> list[dict]:
    """Load a JSON dataset file and return a list of records."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure the synthetic dataset "
            f"files are placed inside the 'data/' folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {filename:<28} → {len(records)} records")
    return records


def load_all_datasets() -> dict:
    """Load every dataset file needed for the pipeline."""
    print("\n[1/1] Loading synthetic dataset files...")
    datasets = {
        "jira_tickets":        load_json("jira_tickets.json"),
        "jenkins_logs":        load_json("jenkins_logs.json"),
        "duplicate_pairs":     load_json("duplicate_pairs.json"),
        "ground_truth_labels": load_json("ground_truth_labels.json"),
    }
    return datasets


def sanity_check(datasets: dict) -> None:
    """Print basic distribution statistics to confirm the data loaded correctly."""
    tickets = datasets["jira_tickets"]
    logs    = datasets["jenkins_logs"]

    print("\n📊 Sanity check — Jira tickets")
    print("  Category distribution :", Counter(t["category"] for t in tickets))
    print("  Status distribution   :", Counter(t["status"]   for t in tickets))
    print("  Sample ticket_id      :", tickets[0]["ticket_id"])
    print("  Sample summary        :", tickets[0]["summary"][:80], "...")

    print("\n📊 Sanity check — Jenkins logs")
    print("  Category distribution :", Counter(l["failure_category"] for l in logs))
    print("  Sample build_id        :", logs[0]["build_id"])
    print("  Sample failure_summary :", logs[0]["failure_summary"][:80], "...")


if __name__ == "__main__":
    datasets = load_all_datasets()
    sanity_check(datasets)
    print("\n✅ Data loading complete. Proceed to 02_preprocess.py")
