"""
cnf_utils.py
============
Utility functions for SAT variable encoding, decoding, clause formatting, and clause pool generation.
"""
import itertools
import random
from typing import List, Tuple
from pysat.formula import CNF


def encode(index: Tuple[int], dims: List[int]) -> int:
    """Encodes a multi-dimensional index into a flat integer variable ID (1-based)."""
    flat = 0
    multiplier = 1
    for i in reversed(range(len(dims))):
        flat += index[i] * multiplier
        multiplier *= dims[i]
    return flat + 1


def decode(var_id: int, dims: List[int]) -> Tuple[int]:
    """Decodes a 1-based variable ID into its multi-dimensional index."""
    var_id -= 1
    out = []
    for d in reversed(dims):
        out.append(var_id % d)
        var_id //= d
    return tuple(reversed(out))


def extract_readable_subset(clauses: List[List[int]], dims: List[int]) -> str:
    """Returns a human-readable string for a set of clauses."""
    def literal_to_str(lit: int) -> str:
        idx = decode(abs(lit), dims)
        var_str = f"x{idx}"
        return var_str if lit > 0 else f"¬x{idx}"

    return ", ".join(f"({' ∨ '.join([literal_to_str(l) for l in clause])})" for clause in clauses)


def generate_clause_space(num_vars: int, max_clause_len: int) -> List[List[int]]:
    """Generates the full space of clauses (no tautologies) up to a given length."""
    literals = [i + 1 for i in range(num_vars)] + [-(i + 1) for i in range(num_vars)]
    clause_space = []
    for r in range(1, max_clause_len + 1):
        for combo in itertools.combinations(literals, r):
            if not any(-x in combo for x in combo):
                clause_space.append(list(combo))
    return clause_space


def readable_cnf_formula(cnf: CNF, dims: List[int]) -> str:
    """Generates a pretty CNF string from a list of clauses."""
    def literal_to_str(lit: int) -> str:
        index = decode(abs(lit), dims)
        var_str = "x" + str(index)
        return var_str if lit > 0 else f"¬{var_str}"

    clause_strs = []
    for clause in cnf.clauses:
        lits = [literal_to_str(l) for l in clause]
        clause_strs.append(f"({' ∨ '.join(lits)})")

    random.shuffle(clause_strs)
    return " ∧ ".join(clause_strs)
