from __future__ import annotations
from typing import List, Tuple

BASES = ("A", "U", "G", "C")
ALLOWED = {("A","U"), ("U","A"), ("G","C"), ("C","G"), ("G","U"), ("U","G")}

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

# --- Parsing structural components ---

def decompose_stems(pairing: List[int]) -> List[Tuple[int,int,int]]:
    """Return list of stems as (i_start, j_start, length) with contiguous stacks.
    i_start pairs with j_start, and k-th layer is (i_start+k, j_start-k).
    """
    n = len(pairing)
    stems = []
    i = 0
    seen = set()
    while i < n:
        j = pairing[i]
        if j > i and (i, j) not in seen:
            # start a new stem
            k = 1
            while i+k < j-k and pairing[i+k] == j-k:
                seen.add((i+k, j-k))
                k += 1
            stems.append((i, j, k))
            i += k
        else:
            i += 1
    return stems

def loop_regions(pairing: List[int]) -> List[Tuple[int,int,int,int]]:
    """Identify loop closures and classify roughly.
    Returns list of (i, j, left_len, right_len) for internal/bulge; hairpins have (i,j,0,0).
    Multibranch loops are identified separately by branch counting around a closing pair.
    """
    n = len(pairing)
    loops = []
    for i in range(n):
        j = pairing[i]
        if j > i:
            # inside region (i, j)
            # check if stacked immediately
            if not (i+1 < j-1 and pairing[i+1] == j-1):
                # either hairpin or internal/bulge/multiloop
                # Count paired segments inside to detect multibranch
                k = i+1
                branches = 0
                last_end = i
                gaps = []
                while k < j:
                    if pairing[k] == -1:
                        k += 1
                        continue
                    kk = pairing[k]
                    if kk > k:
                        branches += 1
                        gaps.append((last_end, k))
                        # jump into this helix to its end
                        # find its stacked length
                        t = 1
                        while k+t < kk-t and pairing[k+t] == kk-t:
                            t += 1
                        last_end = kk
                        k = kk + 1
                    else:
                        k += 1
                if branches == 0:
                    # hairpin
                    loops.append((i, j, 0, 0))
                elif branches == 1:
                    # internal/bulge: left_len/right_len between outer stem and inner stem
                    left_len = gaps[0][1] - gaps[0][0] - 1
                    right_len = j - last_end - 1
                    loops.append((i, j, left_len, right_len))
                else:
                    # mark multibranch via negative lengths
                    loops.append((i, j, -branches, -1))
    return loops