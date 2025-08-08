"""
SAT/UNSAT Puzzle Evaluation Script (step2_evaluation.py)
=========================================================

Usage Example:
--------------
python step2_evaluation.py \
    --mode sat \
    --eval_model openai_gpt-4o \
    --limit 10

This script supports two evaluation modes:
- --mode sat: Performs prediction of satisfiability (SAT/UNSAT) using a specified model.
- --mode trace: Evaluates reasoning traces for logical correctness, using a judge model.

Arguments:
----------
--sat_output: Path to save the model prediction results (optional; defaults to <eval_model>_eval.jsonl).
--trace_output: Path to save the trace evaluation results (optional; defaults to <eval_model>_trace.jsonl).
--eval_model: The model used to predict SAT/UNSAT and generate reasoning traces.
--judge_model: The model used to verify trace validity and evaluate logic consistency.
--limit: (Optional) Maximum number of samples to evaluate. Default is None (evaluate all).

"""

import json
import re
import asyncio
import argparse
import os
import yaml
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from pysat.formula import CNF
from pysat.solvers import Glucose3
from datasets import load_dataset
from eval_utils.generate import llm_generate
from eval_utils.type import EvalInput, EvalOutput, EvalError, EvalErrorCode


# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sat", "trace"], required=True)
parser.add_argument("--sat_output", type=str)
parser.add_argument("--trace_output", type=str)
parser.add_argument("--eval_model", type=str, default="openai_gpt-4o-mini")
parser.add_argument("--judge_model", type=str, default="gpt-4o")
parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to evaluate")
args = parser.parse_args()

# === CONFIG ===
EVAL_MODEL = args.eval_model
JUDGE_MODEL = args.judge_model
SAT_OUTPUT_FILE = args.sat_output or f"{EVAL_MODEL}_eval.jsonl"
TRACE_OUTPUT_FILE = args.trace_output or f"{EVAL_MODEL}_trace.jsonl"


# === Load prompt template from YAML ===
def load_eval_prompt_template(path="prompts/eval_prompts.yaml", key="sat_prediction") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[key]["system"]

def format_eval_prompt(template: str, entry: dict) -> str:
    return template.format(
        scenario=entry["scenario"],
        conditions="\n".join(entry["conditions"]),
        question=entry["question"],
        readable=entry.get("readable", ""),
        variable_mapping=entry.get("variable_mapping", ""),
        unsat_reason=entry.get("unsat_reason", ""),
        model_trace=entry.get("model_trace", "")
    )


def get_remaining(entries, existing_results, mode):
    remaining = []
    for entry in entries:
        key = entry.get("readable")
        result = existing_results.get(key)
        if mode == "sat_only":
            if not result or "pred_label" not in result:
                remaining.append(entry)
        elif mode == "trace_only":
            if key not in existing_results:
                remaining.append(entry)
    return remaining


def write_result(entry, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


# === GPT judge for UNSAT trace ===
async def ask_gpt_trace_valid(entry, model_trace):
    template = load_eval_prompt_template(key="unsat_judgment")
    prompt = format_eval_prompt(template, {**entry, "model_trace": model_trace})

    res = await AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()


# === GPT assignment request for SAT trace ===
async def ask_gpt_for_assignment(entry, model_trace):
    template = load_eval_prompt_template(key="sat_assignment")
    prompt = format_eval_prompt(template, {**entry, "model_trace": model_trace})

    res = await AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()


# === Evaluation ===
async def evaluate_sat_only(entry):
    template = load_eval_prompt_template(key="sat_prediction")
    prompt = format_eval_prompt(template, entry)
    input = EvalInput(model_with_platform=EVAL_MODEL, prompt=prompt)

    try:
        trace = await llm_generate(input)
    except EvalError as err:
        trace = str(err)

    trace = trace.strip()
    text = trace.upper()
    tokens = re.findall(r"\b[A-Z]{3,6}\b", text)  # e.g., "SAT", "UNSAT"

    if "UNSAT" in tokens:
        pred_label = "UNSAT"
    elif "SAT" in tokens:
        pred_label = "SAT"
    else:
        pred_label = None

    gold_label = "SAT" if entry["satisfiable"] else "UNSAT"
    correct = pred_label == gold_label

    eval_result = {
        **entry,
        "model_trace": trace,
        "pred_label": pred_label,
        "correct_prediction": correct,
    }

    return eval_result


async def evaluate_trace_only(entry):
    trace = entry["model_trace"]
    gold_label = "SAT" if entry["satisfiable"] else "UNSAT"
    correct = entry.get("correct_prediction", False)

    eval_result = {**entry}

    if not correct:
        eval_result["trace_correct"] = False
        eval_result["trace_eval_reason"] = "Prediction incorrect."
        return eval_result

    if gold_label == "UNSAT":
        gpt_reason = await ask_gpt_trace_valid(entry, trace)
        valid = bool(re.search(r"\[\s*YES\s*\]|\{\s*YES\s*\}|\\text\{\s*YES\s*\}", gpt_reason.upper()))
        eval_result["trace_correct"] = valid
        eval_result["trace_eval_reason"] = gpt_reason
    else:
        gpt_resp = await ask_gpt_for_assignment(entry, trace)
        assignment = []
        match = re.search(r"Assignment:\s*(\[\[.*\]\])", gpt_resp, re.DOTALL)
        if match:
            try:
                assignment = json.loads(match.group(1))
            except Exception:
                assignment = []
        valid = assignment and all(any(lit == 1 for lit in clause) for clause in assignment)
        eval_result["trace_correct"] = valid
        eval_result["trace_eval_reason"] = f"Assignment (per clause): {assignment}\nGPT said:\n{gpt_resp}"

    return eval_result

def read_existing_results(output_file):
    results = {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = obj.get("readable")
                results[key] = obj
    except FileNotFoundError:
        pass
    return results

async def run_sat_only():
    ds = load_dataset("LLM4Code/SATBench", split="train")
    entries = [dict(x) for x in ds]
    if args.limit:
        entries = entries[:args.limit]

    if not os.path.exists(SAT_OUTPUT_FILE):
        with open(SAT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass
    existing = read_existing_results(SAT_OUTPUT_FILE)
    remaining = get_remaining(entries, existing, "sat_only")

    semaphore = asyncio.Semaphore(100)

    async def evaluate_with_semaphore(entry):
        async with semaphore:
            try:
                return await evaluate_sat_only(entry)
            except Exception as e:
                print(f"Error processing entry: {e}")
                return None

    tasks = [evaluate_with_semaphore(entry) for entry in remaining]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await fut
        if result:
            await asyncio.to_thread(write_result, result, SAT_OUTPUT_FILE)


async def run_trace_only():
    entries = [json.loads(line) for line in open(SAT_OUTPUT_FILE, encoding="utf-8")]
    existing = read_existing_results(TRACE_OUTPUT_FILE)
    remaining = get_remaining(entries, existing, "trace_only")

    semaphore = asyncio.Semaphore(100)

    async def evaluate_with_semaphore(entry):
        async with semaphore:
            try:
                return await evaluate_trace_only(entry)
            except Exception as e:
                print(f"Error processing entry: {e}")
                return None

    tasks = [evaluate_with_semaphore(entry) for entry in remaining]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await fut
        if result:
            await asyncio.to_thread(write_result, result, TRACE_OUTPUT_FILE)


# === Main process with resume ===
if __name__ == "__main__":

    if args.mode == "sat":
        asyncio.run(run_sat_only())
    elif args.mode == "trace":
        asyncio.run(run_trace_only())
