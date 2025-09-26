from __future__ import annotations
from typing import List


BASES = ("A", "U", "G", "C")


# Canonical + wobble pairs (toy)
ALLOWED = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def is_valid_pair(b1: str, b2: str) -> bool:
    return (b1, b2) in ALLOWED


def min_hairpin_loop_ok(i: int, j: int, min_loop: int = 3) -> bool:
    return (j - i - 1) >= min_loop


def to_dot_bracket(pairing: List[int]) -> str:
    n = len(pairing)
    s = ["."] * n
    for i, j in enumerate(pairing):
        if j > i:
            s[i] = "("
            s[j] = ")"
    return "".join(s)