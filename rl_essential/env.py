from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from .energy import TurnerEnergyModel
from .utils.structures import is_valid_pair, min_hairpin_loop_ok, to_dot_bracket

@dataclass
class StepResult:
    state: Tuple[int, List[int]]
    reward: float
    done: bool
    info: Dict

class RNARLEnv:
    def __init__(self, seq: str, energy_model: Optional[TurnerEnergyModel] = None, seed: Optional[int] = None):
        assert all(b in "AUGC" for b in seq), "Sequence must be A/U/G/C"
        self.seq = seq
        self.n = len(seq)
        self.rng = np.random.default_rng(seed)
        self.energy = energy_model or TurnerEnergyModel()
        self.reset()

    def reset(self):
        self.i = 0
        self.pairing = [-1] * self.n
        self.E = self.energy.total_energy(self.seq, self.pairing)
        return (self.i, self.pairing.copy())

    @property
    def state(self):
        return (self.i, self.pairing.copy())

    def _advance_i(self):
        while self.i < self.n and self.pairing[self.i] != -1:
            self.i += 1

    # rna_rl/env.py
    def _noncrossing_ok(self, i: int, j: int) -> bool:
        """Ensure adding (i,j) doesn't cross any existing pair (k,l)."""
        for k, l in enumerate(self.pairing):
            if l == -1 or k >= l:  # only check one orientation k<l
                continue
            # crossing if i<k<j and (l<i or l>j)
            if i < k < j and not (i < l < j):
                return False
            # also disallow nesting conflicts with occupied endpoints
            if (k == i) or (l == j):
                return False
        return True

    def valid_actions(self):
        if self.i >= self.n:
            return []
        acts = [("skip", None)]
        for j in range(self.i + 1, self.n):
            if self.pairing[j] != -1:
                continue
            # 1) Watson–Crick / GU pairing only
            if not is_valid_pair(self.seq[self.i], self.seq[j]):
                continue
            # 2) No pseudoknots/crossings
            if not self._noncrossing_ok(self.i, j):
                continue
            # NOTE: do NOT enforce min hairpin loop here; the energy
            # model already penalizes illegal tiny hairpins when they
            # are truly hairpins (i.e., no inner stack).
            acts.append(("pair", j))
        return acts


    def step(self, action):
        if self.i >= self.n:
            return StepResult(self.state, 0.0, True, {"reason": "done"})
        a, j = action
        old_E = self.E
        if a == "skip":
            self.i += 1
            self._advance_i()
        elif a == "pair":
            assert j is not None and j > self.i
            assert self.pairing[self.i] == -1 and self.pairing[j] == -1
            self.pairing[self.i] = j
            self.pairing[j] = self.i
            self.i += 1
            self._advance_i()
        else:
            raise ValueError(f"Unknown action {action}")
        self.E = self.energy.total_energy(self.seq, self.pairing)
        r = (old_E - self.E)
        done = self.i >= self.n
        if done:
            r += -self.E
        return StepResult(self.state, r, done, {"dot_bracket": to_dot_bracket(self.pairing), "energy": self.E})