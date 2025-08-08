"""
structured_cores.py
====================
Conflict-based structured SAT/UNSAT problem construction (cycle, frozen, XOR, etc).
"""
import random
from typing import List, Tuple
from pysat.formula import CNF
from pysat.solvers import Glucose3
from .cnf_utils import extract_readable_subset, generate_clause_space

def select_n_vars_for_strategy(strategy: str, num_clauses: int) -> int:
    if strategy == "cycle" or strategy == "frozen":
        return max(2, num_clauses - 1)
    elif strategy == "exactly_one_blocked":
        n = 1
        while (1 + n + n * (n - 1) // 2) <= num_clauses:
            n += 1
        return max(1, n - 1)
    elif strategy == "full_block":
        n = 1
        while (2 ** (n + 1)) <= num_clauses:
            n += 1
        return max(1, n)
    elif strategy == "xor_chain":
        return max(2, num_clauses // 2)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def build_cycle_conflict(n_vars: int, dims: List[int]) -> Tuple[List[List[int]], str]:
    if n_vars < 2:
        raise ValueError("Cycle conflict requires at least 2 variables.")
    num_total_vars = _compute_total_vars(dims)
    if n_vars > num_total_vars:
        raise ValueError(f"Cannot pick {n_vars} vars from {num_total_vars} total variables.")
    vars_chosen = random.sample(range(1, num_total_vars + 1), n_vars)
    core = [[-vars_chosen[i], vars_chosen[i + 1]] for i in range(n_vars - 1)]
    core.append([-vars_chosen[-1], -vars_chosen[0]])
    core.append([vars_chosen[0]])
    reason = "Cycle conflict across variables, forming a logical loop that cannot be satisfied: " + extract_readable_subset(core, dims)
    return core, reason

def build_frozen_conflict(n_vars: int, dims: List[int]) -> Tuple[List[List[int]], str]:
    if n_vars < 2:
        raise ValueError("Frozen conflict requires at least 2 variables.")
    num_total_vars = _compute_total_vars(dims)
    if n_vars > num_total_vars:
        raise ValueError(f"Cannot pick {n_vars} vars from {num_total_vars} total variables.")
    vars_chosen = random.sample(range(1, num_total_vars + 1), n_vars)
    core = [[vars_chosen[0]]]
    core += [[-vars_chosen[i], vars_chosen[i + 1]] for i in range(n_vars - 1)]
    core.append([-vars_chosen[-1]])
    reason = "Frozen conflict chain: sequential forced assignments leading to contradiction: " + extract_readable_subset(core, dims)
    return core, reason

def build_exactly_one_conflict(n_vars: int, dims: List[int]) -> Tuple[List[List[int]], str]:
    if n_vars < 2:
        raise ValueError("Exactly-one conflict requires at least 2 variables.")
    num_total_vars = _compute_total_vars(dims)
    if n_vars > num_total_vars:
        raise ValueError(f"Cannot pick {n_vars} vars from {num_total_vars} total variables.")
    vars_chosen = random.sample(range(1, num_total_vars + 1), n_vars)
    core = [vars_chosen[:]]
    core += [[-vars_chosen[i], -vars_chosen[j]] for i in range(n_vars) for j in range(i + 1, n_vars)]
    core += [[-v] for v in vars_chosen]
    reason = "Exactly-one conflict: forced to select at least one, but every option is individually forbidden: " + extract_readable_subset(core, dims)
    return core, reason

def build_full_block_conflict(n_vars: int, dims: List[int]) -> Tuple[List[List[int]], str]:
    if n_vars < 1:
        raise ValueError("Full block conflict requires at least 1 variable.")
    num_total_vars = _compute_total_vars(dims)
    if n_vars > num_total_vars:
        raise ValueError(f"Cannot pick {n_vars} vars from {num_total_vars} total variables.")
    vars_chosen = random.sample(range(1, num_total_vars + 1), n_vars)
    core = []
    for bits in range(2 ** n_vars):
        clause = [ -var if (bits >> idx) & 1 else var for idx, var in enumerate(vars_chosen) ]
        core.append(clause)
    reason = f"Full block: all {2 ** n_vars} possible combinations forbidden for variables {vars_chosen}: " + extract_readable_subset(core, dims)
    return core, reason

def build_xor_chain_conflict(n_vars: int, dims: List[int]) -> Tuple[List[List[int]], str]:
    if n_vars < 2:
        raise ValueError("XOR chain requires at least 2 variables.")
    num_total_vars = _compute_total_vars(dims)
    if n_vars > num_total_vars:
        raise ValueError(f"Cannot pick {n_vars} vars from {num_total_vars} total variables.")
    vars_chosen = random.sample(range(1, num_total_vars + 1), n_vars)
    core = []
    for i in range(n_vars - 1):
        xi = vars_chosen[i]
        xj = vars_chosen[i + 1]
        core.append([xi, -xj])
        core.append([-xi, xj])
    xi = vars_chosen[0]
    xj = vars_chosen[-1]
    core.append([xi, xj])
    core.append([-xi, -xj])
    reason = f"XOR Chain contradiction over variables {vars_chosen}: " + extract_readable_subset(core, dims)
    return core, reason

def _compute_total_vars(dims: List[int]) -> int:
    num_total_vars = 1
    for d in dims:
        num_total_vars *= d
    return num_total_vars

def build_structured_sat_from_unsat(dims: List[int], num_clauses: int, max_clause_len: int, strategy: str) -> Tuple[CNF, int, str]:
    while True:
        unsat_cnf, num_vars, unsat_reason = build_structured_unsat_cnf(dims, num_clauses + 1, max_clause_len, strategy)
        with Glucose3(bootstrap_with=unsat_cnf) as solver:
            if not solver.solve():
                break
    clauses_copy = unsat_cnf.clauses.copy()
    removed_idx = random.randint(0, len(clauses_copy) - 1)
    removed_clause = clauses_copy.pop(removed_idx)
    final_cnf = CNF(from_clauses=clauses_copy)
    removed_clause_readable = extract_readable_subset([removed_clause], dims)
    sat_reason = (
        f"Originally UNSAT because: {unsat_reason}. After removing clause {removed_clause_readable}, the conflict is resolved."
    )
    return final_cnf, num_vars, sat_reason

def build_structured_unsat_cnf(dims: List[int], num_clauses: int, max_clause_len: int, strategy: str) -> Tuple[CNF, int, str]:
    num_vars = _compute_total_vars(dims)
    n_vars = select_n_vars_for_strategy(strategy, num_clauses)
    if strategy == "cycle":
        core, reason = build_cycle_conflict(n_vars, dims)
    elif strategy == "frozen":
        core, reason = build_frozen_conflict(n_vars, dims)
    elif strategy == "exactly_one_blocked":
        core, reason = build_exactly_one_conflict(n_vars, dims)
    elif strategy == "full_block":
        core, reason = build_full_block_conflict(n_vars, dims)
    elif strategy == "xor_chain":
        core, reason = build_xor_chain_conflict(n_vars, dims)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    clause_space = generate_clause_space(num_vars, max_clause_len)
    clause_space_set = {tuple(sorted(c)) for c in core}
    pool = [c for c in clause_space if tuple(sorted(c)) not in clause_space_set]
    fillers = random.sample(pool, min(num_clauses - len(core), len(pool))) if num_clauses > len(core) else []
    full_clauses = core + fillers
    if len(core) > num_clauses:
        raise ValueError(f"Core too large: strategy={strategy}, core={len(core)}, allowed={num_clauses}")
    cnf = CNF(from_clauses=full_clauses)
    return cnf, num_vars, reason
