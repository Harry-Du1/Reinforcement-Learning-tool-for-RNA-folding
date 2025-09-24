from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import json
import os
from .utils.structures import is_valid_pair

# --- Data loading ---
_DATA_DIR = os.path.join(os.path.dirname(__file__), "param")


with open(os.path.join(_DATA_DIR, "turner_stack_2004.json")) as f:
    TURNER_STACK: Dict[str, Dict[str, float]] = json.load(f)


with open(os.path.join(_DATA_DIR, "hairpin_special.json")) as f:
    HAIRPIN_SPECIAL: Dict[str, float] = json.load(f)

class TurnerEnergyModel:
    """Nearest-neighbor energy model (Turner 2004 subset).


    Supports:
    - Stacking energies (base pair stacks)
    - Hairpin loops (length-dependent + special tetraloops)
    - Simple unpaired penalty fallback for unhandled structures


    Notes:
    - Units: kcal/mol
    - Lower = more stable
    - No multiloops or internal loops yet
    """


    def __init__(self):
        pass


   # --- helpers ---
def _stack_key(self, b1: str, b2: str, b3: str, b4: str) -> Tuple[str, str]:
    # stack = closing pair b1-b4, stacked with b2-b3
    return b1 + b4, b2 + b3


def pair_energy(self, b1: str, b2: str) -> float:
    # used for isolated pairs if needed
    if not is_valid_pair(b1, b2):
        return 0.0
    if (b1 == "G" and b2 == "C") or (b1 == "C" and b2 == "G"):
        return -3.0
    if (b1 == "A" and b2 == "U") or (b1 == "U" and b2 == "A"):
        return -2.0
    if (b1 == "G" and b2 == "U") or (b1 == "U" and b2 == "G"):
        return -1.0
    return 0.0


def hairpin_energy(self, seq: str, i: int, j: int) -> float:
    loop_len = j - i - 1
    if loop_len < 3:
        return 10.0 # invalid tiny hairpin, huge penalty


    loop_seq = seq[i:j+1]
    core = seq[i+1:j]


    # Tetraloop bonus
    if len(core) == 4 and core in HAIRPIN_SPECIAL:
        return HAIRPIN_SPECIAL[core]


    # General length-dependent model (Turner 2004, approximate)
    # dG = a + b * ln(loop_len)
    a, b = 3.0, 0.5
    return a + b * np.log(loop_len)

    def total_energy(self, seq: str, pairing: List[int]) -> float:
        n = len(seq)
        E = 0.0
        paired = np.zeros(n, dtype=bool)


    # --- stacking ---
        for i in range(n):
            j = pairing[i]
            if j > i:
                paired[i] = paired[j] = True
                # check for stack (i,j) with (i+1, j-1)
                if i+1 < j-1 and pairing[i+1] == j-1:
                    key_outer, key_inner = self._stack_key(seq[i], seq[j], seq[i+1], seq[j-1])
                    if key_outer in TURNER_STACK and key_inner in TURNER_STACK[key_outer]:
                        E += TURNER_STACK[key_outer][key_inner]
                    else:
                        E += self.pair_energy(seq[i], seq[j])
                else:
                    # isolated pair fallback
                    E += self.pair_energy(seq[i], seq[j])


        # --- hairpins ---
        for i in range(n):
            j = pairing[i]
        if j > i:
            # if no stacking inside (i,j)
            if not (i+1 < j-1 and pairing[i+1] == j-1):
                E += self.hairpin_energy(seq, i, j)


        # --- unpaired penalty fallback ---
        E += 0.2 * float((~paired).sum())
        return E