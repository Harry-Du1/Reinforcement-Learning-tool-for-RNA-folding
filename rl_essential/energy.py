# rna_rl/energy.py
from __future__ import annotations
import math
from typing import List, Tuple, Dict

BASES = "AUGC"
PAIR_OK = {("A","U"),("U","A"),("G","C"),("C","G"),("G","U"),("U","G")}

# Very small, approximate Turner-like stack table (kcal/mol).
# Keys are ((b_i,b_j),(b_ip1,b_jm1)) for stacked pairs (i,j) over (i+1,j-1), 5'->3' on left, 3'->5' on right.
STACK_DG: Dict[Tuple[Tuple[str,str],Tuple[str,str]], float] = {}
def _add(sym_left: str, sym_right: str, dg: float):
    # helper to fill both orientation keys
    l = (sym_left[0], sym_left[1])
    r = (sym_right[0], sym_right[1])
    STACK_DG[(l, r)] = dg

# Coarse values (not exhaustive Turner 2004, but sign-correct and reasonable)
# GC/CG stacks strongest, AU/UA weaker, GU wobble weakest
_add("GC","CG", -2.4); _add("CG","GC", -2.4)
_add("GC","GC", -2.3); _add("CG","CG", -2.3)
_add("AU","UA", -1.1); _add("UA","AU", -1.1)
_add("AU","UA", -1.1); _add("UA","UA", -1.0)
_add("GU","UG", -0.9); _add("UG","GU", -0.9)
# generic fallback for other legal stacks
DEFAULT_STACK = -1.1

# Tetraloop bonuses (very rough; negative => stabilizing)
TETRA_BONUS = {
    "GNRA": -0.8,  # e.g., GAAA
    "UNCG": -1.3,  # e.g., UUCG
    "CUUG": -1.0,
}

def _is_pair(a: str, b: str) -> bool:
    return (a, b) in PAIR_OK

# --- FIX in energy.py ---

def _helices(pairing: List[int]) -> List[Tuple[int,int,int]]:
    """Return list of helices as (i0, j0, L) with consecutive (i+k, j-k) pairs."""
    n = len(pairing)
    seen = set()
    helices = []
    i = 0
    while i < n:
        j = pairing[i]
        if j > i and (i, j) not in seen:
            L = 1
            while i+L < j-L and pairing[i+L] == j-L:
                seen.add((i+L, j-L))
                L += 1
            helices.append((i, j, L))
            i += L
        else:
            i += 1
    return helices


def _loops(seq: str, pairing: List[int]) -> Dict[str, List[Tuple]]:
    """
    Decompose into loops around helices: hairpin at helix end, internal/bulge between two helices,
    multibranch when >2 branches meet. Lightweight parse (sign-correct, not exhaustive).
    """
    n = len(seq)
    helices = _helices(pairing)
    paired = [p != -1 for p in pairing]

    # Hairpins at helix termini
    hairpins = []
    for i, j, L in helices:                     # <-- unpack triples
        il = i + L
        jr = j - L
        if il <= jr and not paired[il] and not paired[jr]:
            hairpins.append((il, jr))

    internals = []
    multibranch = []

    # Mark paired interval coverage from helices (use i..j ranges)
    visited = [False] * n
    for i, j, L in helices:                      # <-- unpack triples
        for k in range(i, j + 1):
            visited[k] = True

    # Scan unpaired runs between paired regions and classify
    k = 0
    while k < n:
        if visited[k] or pairing[k] != -1:
            k += 1
            continue
        start = k
        while k + 1 < n and (not visited[k + 1]) and pairing[k + 1] == -1:
            k += 1
        end = k

        # Look for nearest paired positions around this run
        left_pair = None
        right_pair = None
        li = start - 1
        ri = end + 1
        while li >= 0 and pairing[li] == -1:
            li -= 1
        while ri < n and pairing[ri] == -1:
            ri += 1
        if li >= 0 and pairing[li] > li:
            left_pair = (li, pairing[li])
        if ri < n and pairing[ri] > ri:
            right_pair = (ri, pairing[ri])

        if left_pair and right_pair:
            internals.append((start, end))
        elif left_pair or right_pair:
            multibranch.append((start, end))
        k += 1

    return {"hairpin": hairpins, "internal": internals, "multibranch": multibranch}
class TurnerEnergyModel:
    """
    Simplified, sign-correct RNA ΔG model (kcal/mol).
    - total_energy(...) returns NEGATIVE values for stable folds.
    - No per-base unpaired penalties.
    """
    def __init__(self):
        pass

    def total_energy(self, seq: str, pairing: List[int]) -> float:
        n = len(seq)
        # 1) stacking (negative)
        dg_stack = 0.0
        for i, j, L in _helices(pairing):
            # each step in the helix (i+k, j-k) over (i+k+1, j-k-1)
            for k in range(L-1):
                left = (seq[i+k], seq[j-k])
                right = (seq[i+k+1], seq[j-k-1])
                if not _is_pair(left[0], left[1]) or not _is_pair(right[0], right[1]):
                    continue
                dg = STACK_DG.get((left, right), DEFAULT_STACK)
                dg_stack += dg

        # 2) loops (positive)
        loops = _loops(seq, pairing)
        dg_loop = 0.0
        # hairpin: a + b*ln(L) + tetraloop bonus
        for il, jr in loops["hairpin"]:
            L = jr - il + 1
            if L < 3:
                # prohibit tiny hairpins heavily
                dg_loop += 50.0
                continue
            a, b = 3.4, 1.3
            term = a + b * math.log(L)
            # tetraloop bonuses
            if L == 4:
                loop = seq[il:jr+1]
                if loop.startswith("G") and loop.endswith("A") and loop[2] == "A":
                    term += TETRA_BONUS["GNRA"]
                if loop == "UUCG":
                    term += TETRA_BONUS["UNCG"]
                if loop == "CUUG":
                    term += TETRA_BONUS["CUUG"]
            dg_loop += term

        # internal/bulge: c + d*ln(L) + asymmetry penalty
        for il, jr in loops["internal"]:
            L = jr - il + 1
            c, d = 0.8, 1.1
            term = c + d * math.log(max(1, L))
            dg_loop += term

        # multibranch: a_mb + b_mb * branches + c_mb * unpaired (very rough)
        # Here we approximate each multibranch segment as contributing a small cost.
        for il, jr in loops["multibranch"]:
            L = jr - il + 1
            a_mb, b_mb, c_mb = 3.2, 0.4, 0.2
            term = a_mb + c_mb * L
            dg_loop += term

        return dg_stack + dg_loop
