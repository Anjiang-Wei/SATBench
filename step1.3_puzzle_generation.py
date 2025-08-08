"""
step1.3_puzzle_generation.py
=============================

Generate natural language logic puzzles from scenario mappings.

Usage Example:
--------------
Run the following command to start generation:

python step1.3_puzzle_generation.py \
  --input scenario_and_mapping.jsonl \
  --output puzzle_problems.jsonl \
  --target_per_clause 5 \
  --clause_values "[4,5,6]"
"""
import json
import asyncio
import os
import re
import yaml
import random
import argparse
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from collections import defaultdict
from typing import List, Tuple
from pysat.formula import CNF
from pysat.solvers import Glucose3


# === CONFIG ===
MODEL = "gpt-4o"
semaphore = asyncio.Semaphore(150)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


# === Prompt Utilities ===
def load_prompt_template(path="prompts/gen_prompts.yaml", key="consistency_check") -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data[key]["system"]

def format_prompt(template: str, entry: dict, extra: dict = None) -> str:
    fields = {
        "scenario": entry.get("scenario", ""),
        "variable_mapping": entry.get("variable_mapping", ""),
        "conditions": "\n".join(entry.get("conditions", [])),
        "question": entry.get("question", ""),
        "readable": entry.get("readable", ""),
    }
    if extra:
        fields.update(extra)
    return template.format(**fields)


# === I/O Utilities ===
def write_json_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

def encode(index: Tuple[int], dims: List[int]) -> int:
    flat = 0
    multiplier = 1
    for i in reversed(range(len(dims))):
        flat += index[i] * multiplier
        multiplier *= dims[i]
    return flat + 1

def readable_clause_to_dimacs(clause_str: str, dims: List[int]) -> List[int]:
    clause_str = clause_str.strip("() ").replace("\n", " ")
    literals = clause_str.split(" ∨ ")
    dimacs_clause = []

    for lit in literals:
        lit = lit.strip()
        is_neg = lit.startswith("¬")
        lit = lit[1:] if is_neg else lit

        match = re.match(r"x\(\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*\)?", lit)
        if not match:
            # print(f"[Warning] failed to parse literal: {lit}")
            continue

        idx = tuple(int(g) for g in match.groups() if g is not None)
        var = encode(idx, dims)
        # print(f"Parsed {lit} → idx={idx} → var={var}")
        dimacs_clause.append(-var if is_neg else var)

    return dimacs_clause


# === GPT Functions ===
async def ask_gpt_consistency_check(entry):
    template = load_prompt_template(key="consistency_check")
    prompt = format_prompt(template, entry)

    res = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()

async def ask_gpt_recover_readable_formula(entry):
    template = load_prompt_template(key="clause_recovery")
    prompt = format_prompt(template, entry)

    res = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()

async def ask_gpt_formula_equivalence_check(entry, recovered_formula_text):
    template = load_prompt_template(key="equivalence_check")
    prompt = format_prompt(template, entry, extra={"recovered": recovered_formula_text})
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

async def generate_and_parse(entry, retry_context=None):
    try:
        header = load_prompt_template(key="puzzle_generation")
        key = "generation_retry" if retry_context else "generation"
        template = load_prompt_template(key=key)

        extra = None
        if retry_context:
            extra = {
                "previous_conditions": "\n".join(retry_context["conditions"]),
                "previous_question": retry_context["question"],
                "consistency_feedback": retry_context["last_consistency_trace"].strip()
            }

        user_prompt = header.strip() + "\n\n" + format_prompt(template, entry, extra)

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        full_text = response.choices[0].message.content.strip()

        conditions_match = re.search(r"<conditions>\s*((?:\d+\..*?\n)+)", full_text)
        final_question_match = re.search(r"<final question>\s*(.+)", full_text, re.DOTALL)

        if not (conditions_match and final_question_match):
            print(f"[Parse Failed] Missing conditions or final question:\n{full_text[:300]}...\n")
            return None

        condition_lines = [line.strip() for line in conditions_match.group(1).strip().splitlines()]
        question = final_question_match.group(1).strip()

        return {
            **entry,
            "conditions": condition_lines,
            "question": question
        }

    except Exception as e:
        print(f"[GPT call failed] {entry.get('readable', '')[:60]} → {e}")
        return None


def parse_formula(readable: str, dims: List[int]) -> List[List[int]]:
    """Convert a human-readable CNF formula to DIMACS format."""
    return [readable_clause_to_dimacs(c, dims) for c in readable.split(" ∧ ")]

def entails(src: List[List[int]], tgt: List[List[int]]) -> bool:
    """
    Check if CNF src entails tgt (src ⊨ tgt). If src ∧ ¬clause is satisfiable for any clause in tgt, return False.
    """
    for clause in tgt:
        neg_units = [[-lit] for lit in clause]
        if Glucose3(bootstrap_with=src + neg_units).solve():
            return False
    return True

def cnf_equivalent(a: List[List[int]], b: List[List[int]]) -> bool:
    """Check if two CNF formulas are logically equivalent."""
    return entails(a, b) and entails(b, a)

def check_formula_equivalence(original_formula: str, recovered_formula: str, dims: List[int]) -> bool:
    """
    Check if the original and recovered formulas are logically equivalent using a SAT solver.
    """
    try:
        a = parse_formula(original_formula, dims)
        b = parse_formula(recovered_formula, dims)
        return cnf_equivalent(a, b)
    except Exception as e:
        return False

# === Main Worker ===
async def generate_and_write(entry, seen, bank, output_path):
    async with semaphore:
        attempts = 0
        retry_context = None
        consistency_history = []

        while attempts < 3:
            candidate = entry  # always use passed-in entry
            if candidate.get("readable") in seen:
                return  # already done, skip

            # Step 1: Generate
            result = await generate_and_parse(candidate, retry_context)
            if not result:
                attempts += 1
                continue

            # Step 2: Consistency Check
            gpt_verdict = await ask_gpt_consistency_check(result)
            consistency_history.append({
                "attempt": attempts + 1,
                "trace": gpt_verdict,
                "conditions": result.get("conditions"),
                "question": result.get("question")
            })

            if not re.search(r"\[\s*CONSISTENT\s*\]", gpt_verdict.upper()):
                retry_context = {
                    "last_consistency_trace": gpt_verdict,
                    "conditions": result.get("conditions"),
                    "question": result.get("question")
                }
                attempts += 1
                print("[Inconsistent and retry]")
                continue

            # Step 3: Clause Recovery Check
            clause_formula_text = await ask_gpt_recover_readable_formula(result)
            match = re.search(r"\[\s*(.*?)\s*\]", clause_formula_text, re.DOTALL)
            if not match:
                print("[Clause parse fail]")
                return

            clause_str = match.group(1)
            clause_strs = clause_str.split(" ∧ ")

            try:
                clauses = [readable_clause_to_dimacs(cs, candidate["dims"]) for cs in clause_strs]
                with Glucose3(bootstrap_with=CNF(from_clauses=clauses)) as solver:
                    actual_sat = solver.solve()
                if actual_sat != candidate["satisfiable"]:
                    print(f"[SAT check mismatch]: expected {candidate['satisfiable']}, got {actual_sat}")
                    return
            except Exception as e:
                print(f"[Clause decode error]: {e}")
                return

            equiv_result = await ask_gpt_formula_equivalence_check(result, clause_str)
            result["formula_equivalence_check"] = equiv_result
            if not re.search(r"\[\s*EQUIVALENT\s*\]", equiv_result.upper()):
                print("[Formula mismatch]")
                return

            # Step 4: SAT Solver Equivalence Check
            is_equivalent = check_formula_equivalence(
                result["readable"],
                clause_str,
                candidate["dims"]
            )
            if not is_equivalent:
                print("[SAT solver equivalence check failed]")
                print(f"Original: {result['readable']}")
                print(f"Recovered: {clause_str}")
                return

            # Step 5: Save
            result["recovered_formula"] = clause_str
            result["recovered_formula_full_text"] = clause_formula_text
            result["consistency_check_trace_history"] = consistency_history
            result["sat_solver_equivalence_check"] = "PASSED"
            await asyncio.to_thread(write_json_line, output_path, result)
            seen.add(candidate.get("readable"))
            return

        print(f"[Failed after 3 attempts] {entry.get('readable', '')[:60]}...")


# === Dataset Indexing ===
def build_bank_index(data):
    bank = defaultdict(list)
    for item in data:
        key = (item["num_clauses"], item["satisfiable"])
        bank[key].append(item)
    return bank


# === MAIN ===
async def main():
    parser = argparse.ArgumentParser(description="Puzzle Problem Generator with GPT Validation")
    parser.add_argument("--input", type=str, default="scenario_and_mapping.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default="puzzle_problems.jsonl", help="Path to output JSONL file")
    parser.add_argument("--target_per_clause", type=int, default=100,
                        help="Total entries per clause count (half SAT, half UNSAT)")
    parser.add_argument("--max_rounds", type=int, default=10, help="Maximum retry rounds")
    parser.add_argument("--clause_values", type=str, default="[4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]",
                        help="List of clause counts to cover")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    target_per_clause = args.target_per_clause
    max_rounds = args.max_rounds
    clause_values = json.loads(args.clause_values)

    # Load full dataset once
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    bank = build_bank_index(data)

    for round_id in range(1, max_rounds + 1):
        print(f"\nRound {round_id}")

        seen = set()
        seen_counts = defaultdict(int)
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    seen.add(obj.get("readable"))
                    key = (obj["num_clauses"], obj["satisfiable"])
                    seen_counts[key] += 1
        except FileNotFoundError:
            pass

        additional_entries = []

        for num_clauses in clause_values:
            for satisfiable in [True, False]:
                key = (num_clauses, satisfiable)
                needed = (target_per_clause // 2) - seen_counts.get(key, 0)

                if needed > 0:
                    candidates = [e for e in bank.get(key, []) if e.get("readable") not in seen]
                    random.shuffle(candidates)
                    additional_entries.extend(candidates[:needed])
                    print(f"key={key}, needed={needed}, available_candidates={len(candidates)}")

        if not additional_entries:
            print("All num_clauses categories filled!")
            break

        print(f"Need to generate {len(additional_entries)} more entries to fill gaps...")

        tasks = [generate_and_write(entry, seen, bank, output_path) for entry in additional_entries]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await future

    else:
        print("Max retry rounds reached, but some clause categories may still have gaps.")


if __name__ == "__main__":
    asyncio.run(main())
