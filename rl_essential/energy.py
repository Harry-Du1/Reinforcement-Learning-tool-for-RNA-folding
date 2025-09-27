from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import json, os
from .utils.structures import is_valid_pair, min_hairpin_loop_ok, decompose_stems, loop_regions

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(_DATA_DIR, "turner_stack_2004.json")) as f:
    TURNER_STACK: Dict[str, Dict[str, float]] = json.load(f)
with open(os.path.join(_DATA_DIR, "hairpin_special.json")) as f:
    HAIRPIN_SPECIAL: Dict[str, float] = json.load(f)

class TurnerEnergyModel:
    """Approximate Turner-like model.

    Includes:
      - Stacking energies (nearest neighbor)
      - Hairpins: length penalty + tetraloop bonuses
      - Internal/Bulge loops: length + asymmetry
      - Multibranch loops: a + b * unpaired + c * branches
      - Optional coaxial stacking heuristic

    Units: kcal/mol; lower is better. Constants are approximate.
    """
    def __init__(self, enable_coaxial: bool = True):
        self.enable_coaxial = enable_coaxial
        # Multibranch parameters (rough)
        self.mb_a = 3.4
        self.mb_b = 0.4
        self.mb_c = 0.9
        self.unpaired_penalty = 0.1

    # --- stacking ---
    def _stack_key(self, b1: str, b2: str, b3: str, b4: str) -> Tuple[str, str]:
        return b1 + b2, b3 + b4

    def _stack_energy(self, s: str, i: int, j: int, k: int) -> float:
        # stack layer k on stem (i,j): pairs (i+k, j-k) with (i+k+1, j-k-1)
        a = s[i+k] + s[j-k]
        b = s[i+k+1] + s[j-k-1]
        return TURNER_STACK.get(a, {}).get(b, 0.0)

    def _hairpin_energy(self, s: str, i: int, j: int) -> float:
        L = j - i - 1
        if L < 3:
            return 10.0
        core = s[i+1:j]
        if len(core) == 4 and core in HAIRPIN_SPECIAL:
            return HAIRPIN_SPECIAL[core]
        a, b = 3.0, 0.5
        return a + b * np.log(L)

    def _internal_bulge_energy(self, left_len: int, right_len: int) -> float:
        # symmetric favored; asymmetry penalty
        L = left_len + right_len
        if L == 0:
            return 0.0
        base = 0.3 * np.log(max(2, L)) + 0.2 * L
        asym = 0.2 * abs(left_len - right_len)
        # 1-nt bulge favorable-ish
        if L == 1:
            base -= 0.2
        return base + asym

    def _multibranch_energy(self, branches: int, unpaired: int) -> float:
        return self.mb_a + self.mb_b * unpaired + self.mb_c * branches

    def _coaxial_bonus(self, s: str, stems: List[Tuple[int,int,int]]) -> float:
        if not self.enable_coaxial:
            return 0.0
        # Heuristic: bonus per adjacency of stem ends (not physically rigorous)
        bonus = 0.0
        for (i,j,k) in stems:
            # if two stems share adjacent ends, add small bonus
            for (i2,j2,k2) in stems:
                if (i,j) == (i2,j2):
                    continue
                if abs(i - i2) <= 1 or abs(j - j2) <= 1:
                    bonus -= 0.2
        return bonus

    def total_energy(self, seq: str, pairing: List[int]) -> float:
        n = len(seq)
        E = 0.0
        paired = np.zeros(n, dtype=bool)

        # Mark paired
        for i in range(n):
            j = pairing[i]
            if j > i:
                paired[i] = paired[j] = True

        # Stems & stacking
        stems = decompose_stems(pairing)
        for (i, j, L) in stems:
            if L >= 2:
                for k in range(L - 1):
                    E += self._stack_energy(seq, i, j, k)
            else:
                # isolated pair fallback
                E += -1.0  # weak stabilization

        # Loops
        loops = loop_regions(pairing)
        for (i, j, a, b) in loops:
            if a == 0 and b == 0:
                E += self._hairpin_energy(seq, i, j)
            elif a < 0 and b == -1:
                # multibranch: a = -branches
                branches = -a
                # estimate unpaired inside (i, j)
                unpaired = sum(1 for k in range(i+1, j) if pairing[k] == -1)
                E += self._multibranch_energy(branches, unpaired)
            else:
                # internal/bulge
                E += self._internal_bulge_energy(a, b)

        # coaxial heuristic
        E += self._coaxial_bonus(seq, stems)

        # Mild penalty for unpaired
        E += self.unpaired_penalty * float((~paired).sum())
        return E