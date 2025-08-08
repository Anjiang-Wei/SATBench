"""
SAT Problem Generator
======================
Generates structured SAT/UNSAT problems using conflict strategies.

Each generated CNF is constructed with known unsatisfiable cores (e.g., cycle, frozen chains, xor-chains),
mixed with filler clauses, and optionally repaired to create satisfiable instances.

Usage (CLI):
    python step1.1_sat_problem_generation.py \
        --output sat_problems.json \
        --num_per_config 10 \
        --max_clause_len 3 \
        --sat_ratio 0.5 \
        --dimensions "[[3,2],[3,3,2]]" \
        --clauses "[4,5,6,7,8,9,10]"

"""
import argparse
import itertools
import random
import json
import os
from typing import List, Tuple
from pysat.formula import CNF
from pysat.solvers import Glucose3
from tqdm import tqdm

from generators.structured_cores import (
    build_structured_sat_from_unsat,
    build_structured_unsat_cnf
)

from generators.cnf_utils import readable_cnf_formula


def generate_dataset(
    output_path: str,
    dimensions: List[List[int]],
    clause_values: List[int],
    max_clause_len: int,
    num_problems_per_config: int,
    sat_ratio: float,
    seed: int = 42
):
    random.seed(seed)
    dataset = []

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            dataset = json.load(f)

    existing_counts = {}
    for item in dataset:
        key = (tuple(item["dims"]), item["num_clauses"], item["satisfiable"])
        existing_counts[key] = existing_counts.get(key, 0) + 1

    strategies = ['cycle', 'frozen', 'exactly_one_blocked', 'full_block', 'xor_chain']

    valid_pairs = []
    for dim_sizes, num_clauses in itertools.product(dimensions, clause_values):
        num_vars = 1
        for d in dim_sizes:
            num_vars *= d
        if num_clauses <= 20:
            if len(dim_sizes) == 3 and dim_sizes[2] > 2:
                continue
        if num_clauses <= num_vars:
            valid_pairs.append((dim_sizes, num_clauses))

    total_tasks = len(valid_pairs) * num_problems_per_config
    pbar = tqdm(total=total_tasks, desc="Generating SAT/UNSAT dataset")

    for dim_sizes, num_clauses in valid_pairs:
        sat_target = round(num_problems_per_config * sat_ratio)
        unsat_target = num_problems_per_config - sat_target
        key_sat = (tuple(dim_sizes), num_clauses, True)
        key_unsat = (tuple(dim_sizes), num_clauses, False)
        sat_count = existing_counts.get(key_sat, 0)
        unsat_count = existing_counts.get(key_unsat, 0)

        while sat_count < sat_target:
            strategy = strategies[sat_count % len(strategies)]
            try:
                cnf, num_vars, sat_reason = build_structured_sat_from_unsat(dim_sizes, num_clauses, max_clause_len, strategy)
                with Glucose3(bootstrap_with=cnf) as solver:
                    if solver.solve():
                        dataset.append({
                            "dims": dim_sizes,
                            "num_vars": num_vars,
                            "num_clauses": num_clauses,
                            "clauses": cnf.clauses,
                            "readable": readable_cnf_formula(cnf, dim_sizes),
                            "satisfiable": True,
                            "sat_reason": sat_reason
                        })
                        sat_count += 1
                        pbar.update(1)
            except Exception as e:
                print(f"[Warning] SAT generation failed: {e}")

        while unsat_count < unsat_target:
            strategy = strategies[unsat_count % len(strategies)]
            try:
                cnf, num_vars, unsat_reason = build_structured_unsat_cnf(dim_sizes, num_clauses, max_clause_len, strategy)
                with Glucose3(bootstrap_with=cnf) as solver:
                    if not solver.solve():
                        dataset.append({
                            "dims": dim_sizes,
                            "num_vars": num_vars,
                            "num_clauses": num_clauses,
                            "clauses": cnf.clauses,
                            "readable": readable_cnf_formula(cnf, dim_sizes),
                            "satisfiable": False,
                            "unsat_reason": unsat_reason
                        })
                        unsat_count += 1
                        pbar.update(1)
            except Exception as e:
                print(f"[Warning] UNSAT generation failed: {e}")

    pbar.close()

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} problems to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured SAT/UNSAT Problem Generator")
    parser.add_argument("--output", type=str, default="sat_problems.json", help="Output file path")
    parser.add_argument("--num_per_config", type=int, default=80, help="# problems per (dims, clause_num)")
    parser.add_argument("--max_clause_len", type=int, default=3, help="Max clause length")
    parser.add_argument("--sat_ratio", type=float, default=0.5, help="Fraction of SAT problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dimensions", type=str, required=True, help="List of dimensions, e.g., '[[3,2],[3,3]]'")
    parser.add_argument("--clauses", type=str, required=True, help="List of clause counts, e.g., '[4,5,6]'")

    args = parser.parse_args()
    dimensions = json.loads(args.dimensions)
    clause_values = json.loads(args.clauses)

    generate_dataset(
        output_path=args.output,
        dimensions=dimensions,
        clause_values=clause_values,
        max_clause_len=args.max_clause_len,
        num_problems_per_config=args.num_per_config,
        sat_ratio=args.sat_ratio,
        seed=args.seed
    )
