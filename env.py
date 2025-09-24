from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from .energy import ToyEnergyModel
from .utils.structures import is_valid_pair, min_hairpin_loop_ok, to_dot_bracket

@dataclass
class StepResult:
    state: Tuple[int, List[int]]
    reward: float
    done: bool
    info: Dict
class RNARLEnv:
    def __init__(self, seq: str, energy_model: Optional[ToyEnergyModel] = None, seed: Optional[int] = None):
        assert all(b in "AUGC" for b in seq), "Sequence must be A/U/G/C only (RNA)"
        self.seq = seq
        self.n = len(seq)
        self.rng = np.random.default_rng(seed)
        self.energy = energy_model or ToyEnergyModel()
        self.reset()


    # --- Core API ---
    def reset(self) -> Tuple[int, List[int]]:
        self.i = 0
        self.pairing = [-1] * self.n
        self.E = self.energy.total_energy(self.seq, self.pairing)
        return (self.i, self.pairing.copy())


    def _advance_i(self):
        # If current i is already paired (from a prior action pairing j with i), skip it
        while self.i < self.n and self.pairing[self.i] != -1:
            self.i += 1


    def valid_actions(self) -> List[Tuple[str, Optional[int]]]:
        if self.i >= self.n:
            return []
        acts: List[Tuple[str, Optional[int]]] = [("skip", None)]
        for j in range(self.i + 1, self.n):
            if self.pairing[j] != -1:
                continue
            if not is_valid_pair(self.seq[self.i], self.seq[j]):
                continue
            if not min_hairpin_loop_ok(self.i, j):
                continue
            acts.append(("pair", j))
        return acts


    def step(self, action: Tuple[str, Optional[int]]) -> StepResult:
        assert self.i <= self.n, "Episode already terminated"
        if self.i == self.n:
            return StepResult((self.i, self.pairing.copy()), 0.0, True, {"reason": "done"})


        a, j = action
        old_E = self.E


        if a == "skip":
        # leave i unpaired
            self.i += 1
            self._advance_i()
        elif a == "pair":
            assert j is not None and j > self.i
            assert self.pairing[self.i] == -1 and self.pairing[j] == -1
            # apply pairing
            self.pairing[self.i] = j
            self.pairing[j] = self.i
            # advance i
            self.i += 1
            self._advance_i()
        else:
            raise ValueError(f"Unknown action: {action}")


        # Dense reward = energy drop
        self.E = self.energy.total_energy(self.seq, self.pairing)
        r = (old_E - self.E)


        done = self.i >= self.n
        if done:
        # terminal bonus toward lowest final energy
            r += -self.E


        return StepResult((self.i, self.pairing.copy()), r, done, {
        "dot_bracket": to_dot_bracket(self.pairing),
        "energy": self.E,
        })