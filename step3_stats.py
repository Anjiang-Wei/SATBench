"""
SAT/UNSAT Evaluation Stats Summary
=====================================================

This script computes accuracy statistics for SAT/UNSAT classification by difficulty level.

Usage:
  python step3_stats.py                        # Automatically evaluates all *_eval.jsonl files
  python step3_stats.py --models openai_gpt-4o

The output includes:
- Per-bucket accuracy by SAT/UNSAT × difficulty
- LaTeX table row for easy copy-paste
- Average row across models
"""

import os
import json
import pandas as pd
import re
import argparse
from collections import defaultdict

def load_eval_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def classify_difficulty(num_clauses):
    if 4 <= num_clauses <= 15:
        return "easy"
    elif 20 <= num_clauses <= 30:
        return "medium"
    elif 31 <= num_clauses <= 50:
        return "hard"
    else:
        return "unknown"

def compute_accuracy(eval_records):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in eval_records:
        sat = row.get("satisfiable")
        correct = row.get("correct_prediction", False)
        num_clauses = row.get("num_clauses")
        difficulty = classify_difficulty(num_clauses)
        if difficulty == "unknown":
            continue
        key = f"{'SAT' if sat else 'UNSAT'}-{difficulty}"
        buckets[key]["total"] += 1
        if correct:
            buckets[key]["correct"] += 1

    # Compute and print accuracy
    accs_by_difficulty = {"easy": [], "medium": [], "hard": []}
    latex_vals = []  # will hold all 9 values

    print("\nDetailed Accuracy by Bucket:")
    for sat_label in ["SAT", "UNSAT"]:
        for difficulty in ["easy", "medium", "hard"]:
            key = f"{sat_label}-{difficulty}"
            stat = buckets[key]
            total = stat["total"]
            acc = stat["correct"] / total if total > 0 else 0.0
            accs_by_difficulty[difficulty].append(acc)
            latex_vals.append(acc * 100)
            print(f"{key:15} | Accuracy: {acc * 100:.1f}% ({stat['correct']} / {total})")

    # Add averaged accs (SAT + UNSAT) for each difficulty
    for d in ["easy", "medium", "hard"]:
        avg = sum(accs_by_difficulty[d]) / 2 if len(accs_by_difficulty[d]) == 2 else 0.0
        latex_vals.append(avg * 100)
    overall_avg = sum(latex_vals[-3:]) / 3  # average of easy, medium, hard averages
    latex_vals.append(overall_avg)

    latex_row = " & ".join(f"{a:.1f}" for a in latex_vals)
    return latex_row

def run_eval_summary(model_names):
    all_latex_rows = []

    for model in model_names:
        file_path = f"{model}_eval.jsonl"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"\n=== Model: {model} ===")
        records = load_eval_data(file_path)
        latex_row = compute_accuracy(records)

        all_latex_rows.append(f"{model:<25} & {latex_row} \\\\")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="List of model prefixes to evaluate")
    args = parser.parse_args()

    if args.models:
        model_names = args.models
    else:
        model_names = [
            fname.replace("_eval.jsonl", "")
            for fname in os.listdir(".")
            if fname.endswith("_eval.jsonl")
        ]

    run_eval_summary(model_names)
