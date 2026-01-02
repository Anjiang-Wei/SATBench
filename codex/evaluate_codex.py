#!/usr/bin/env python3
"""
Codex Agent Evaluation Script for SATBench
==========================================

Usage Example:
--------------
python evaluate_codex.py --model gpt-5-nano --limit 100

This script evaluates the Codex agent on SATBench tasks.

Arguments:
----------
--model: The model to use with codex (default: gpt-5-nano)
--limit: Maximum number of samples to evaluate (default: 100)
--output: Path to save evaluation results (default: codex_eval_results.jsonl)
--n-concurrent: Number of concurrent evaluations (default: 5)
"""

import json
import re
import asyncio
import argparse
import os
import shlex
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from tqdm.asyncio import tqdm
import yaml
from datasets import load_dataset


# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Evaluate Codex agent on SATBench")
parser.add_argument("--model", type=str, default="gpt-5-nano", help="Model to use with codex")
parser.add_argument("--limit", type=int, default=100, help="Maximum number of samples to evaluate")
parser.add_argument("--output", type=str, default="codex_eval_results.jsonl", help="Path to save results")
parser.add_argument("--n-concurrent", type=int, default=20, help="Number of concurrent evaluations")
args = parser.parse_args()


# === Configuration ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = SCRIPT_DIR.parent / "prompts" / "eval_prompts.yaml"
WORK_DIR = Path("/tmp/codex_satbench")
MODEL_NAME = args.model
OUTPUT_FILE = args.output
N_CONCURRENT = args.n_concurrent


def load_eval_prompt_template(path: Path = PROMPTS_PATH, key: str = "sat_prediction") -> str:
    """Load the SAT prediction prompt template from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[key]["system"]


def format_eval_prompt(template: str, entry: dict) -> str:
    """Format the prompt template with task data."""
    return template.format(
        scenario=entry["scenario"],
        conditions="\n".join(entry["conditions"]),
        question=entry["question"],
        readable=entry.get("readable", ""),
        variable_mapping=entry.get("variable_mapping", ""),
        unsat_reason=entry.get("unsat_reason", ""),
        model_trace=entry.get("model_trace", ""),
    )


def extract_sat_label(text: str) -> Optional[str]:
    """Extract SAT/UNSAT label from codex output."""
    text = text.upper()
    tokens = re.findall(r"\b[A-Z]{3,6}\b", text)

    if "UNSAT" in tokens:
        return "UNSAT"
    elif "SAT" in tokens:
        return "SAT"
    else:
        return None


def run_codex_sync(prompt: str, task_id: str, work_dir: Path) -> str:
    """Run codex agent with the given prompt."""
    task_work_dir = work_dir / f"task_{task_id}"
    task_work_dir.mkdir(parents=True, exist_ok=True)

    escaped_prompt = shlex.quote(prompt)

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--model", MODEL_NAME,
        "--temperature", "0",
        escaped_prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=task_work_dir,
            env={**os.environ},
            timeout=300,
        )
        output = result.stdout + "\n" + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return "[ERROR: Codex execution timed out]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"
    finally:
        try:
            shutil.rmtree(task_work_dir, ignore_errors=True)
        except Exception:
            pass


async def evaluate_single_task(entry: dict, task_idx: int, semaphore: asyncio.Semaphore) -> dict:
    """Evaluate a single SATBench task with codex agent."""
    async with semaphore:
        template = load_eval_prompt_template()
        prompt = format_eval_prompt(template, entry)

        output = await asyncio.to_thread(run_codex_sync, prompt, str(task_idx), WORK_DIR)

        pred_label = extract_sat_label(output)
        gold_label = "SAT" if entry["satisfiable"] else "UNSAT"
        correct = pred_label == gold_label

        return {
            "task_idx": task_idx,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "correct": correct,
            "output": output,
        }


async def run_evaluation():
    """Main evaluation loop."""
    print(f"Loading SATBench dataset...")
    ds = load_dataset("LLM4Code/SATBench", split="train")
    entries = [dict(x) for x in ds][:args.limit]

    print(f"Evaluating {len(entries)} tasks with codex (model: {MODEL_NAME})")

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(N_CONCURRENT)
    tasks = [evaluate_single_task(entry, idx, semaphore) for idx, entry in enumerate(entries)]

    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await fut
        results.append(result)

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Compute accuracy
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"\nFINAL ACCURACY: {accuracy:.2%} ({correct_count}/{total_count})")

    # Clean up
    try:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(run_evaluation())
