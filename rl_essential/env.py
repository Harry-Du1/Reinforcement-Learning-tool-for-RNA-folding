# rna_rl/env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .energy import TurnerEnergyModel
from .utils.structures import is_valid_pair, to_dot_bracket

Action = Tuple[str, Optional[int]]  # ("pair", j) only


@dataclass
class StepResult:
    state: Tuple[int, List[int]]
    reward: float
    done: bool
    info: Dict


class RNARLEnv:
    """
    Pairs-only RNA secondary-structure environment (no explicit 'skip').
    - No pseudoknots (non-crossing arcs enforced).
    - If no legal pair exists at position i, env auto-advances i (implicit skip).
    - Reward = ΔE per step (E_{t-1} - E_t); terminal bonus adds -E_T.
      With a correctly signed energy model, lower energies are better.
    """
    def _would_make_too_small_hairpin(self, i: int, j: int) -> bool:
        """
        Return True iff adding (i,j) would create a *new* hairpin whose loop length < 3.
        Allowed cases that do NOT trigger the hairpin check:
        - Stack extension: (i+1, j-1) already paired.
        - Enclosing an existing interior structure: any k in (i,j) is already paired.
        """
        # 1) stack extension? then OK regardless of gap
        if i + 1 < j and self.pairing[i + 1] == j - 1:
            return False

        # 2) enclosing existing interior pairs? then not a hairpin closure
        for k in range(i + 1, j):
            if self.pairing[k] != -1:
                return False

        # 3) otherwise we'd close a fresh hairpin; require loop >= 3
        loop_len = (j - i - 1)
        return loop_len < 3

    def __init__(self,
        seq: str,
        energy_model: Optional[TurnerEnergyModel] = None,
        min_pair_separation: int = 4,):
        assert all(b in "AUGC" for b in seq), "Sequence must contain only A/U/G/C"
        self.seq = seq
        self.n = len(seq)
        self.energy = energy_model or TurnerEnergyModel()
        self.min_pair_separation = int(min_pair_separation)
        self.reset()

    # ---------- helpers (relations) ----------
    @staticmethod
    def _cross(i: int, j: int, k: int, l: int) -> bool:
        return (i < k < j < l) or (k < i < l < j)

    def _pair_allowed(self, i: int, j: int) -> bool:
        if not (0 <= i < j < self.n):
            return False

        # NEW: hard minimum separation (no close pairs *anywhere*)
        if (j - i) < self.min_pair_separation:
            return False

        # base compatibility
        if not is_valid_pair(self.seq[i], self.seq[j]):
            return False

        # no pseudoknots (non-crossing)
        for k, l in enumerate(self.pairing):
            if l == -1 or not (k < l):
                continue
            if self._cross(i, j, k, l):
                return False
        return True

    # ---------- API ----------
    def reset(self) -> Tuple[int, List[int]]:
        self.i = 0
        self.pairing = [-1] * self.n
        self.E = self.energy.total_energy(self.seq, self.pairing)
        # advance i to the first index that *could* pair with someone
        self._advance_until_candidate()
        return (self.i, self.pairing.copy())

    @property
    def state(self) -> Tuple[int, List[int]]:
        return (self.i, self.pairing.copy())

    def _advance_i(self) -> None:
        """Advance i to next free site."""
        while self.i < self.n and self.pairing[self.i] != -1:
            self.i += 1

    def _advance_until_candidate(self) -> None:
        """Advance i while no legal pair exists for i (implicit skip)."""
        while self.i < self.n:
            if self.pairing[self.i] != -1:
                self._advance_i()
                continue
            has_candidate = any(
                self._pair_allowed(self.i, j) and self.pairing[j] == -1
                for j in range(self.i + 1, self.n)
            )
            if has_candidate:
                break
            self.i += 1  # implicit skip
        # if i==n, terminal

    def valid_actions(self) -> List[Action]:
        """Pairs-only actions at current i. Empty => auto-advance on step()."""
        if self.i >= self.n:
            return []
        acts: List[Action] = []
        # collect pair candidates i<j
        for j in range(self.i + 1, self.n):
            if self.pairing[j] != -1:
                continue
            if self._pair_allowed(self.i, j):
                acts.append(("pair", j))
        return acts

    def step(self, action: Action) -> StepResult:
        """Apply ('pair', j) if available; otherwise implicitly advance i."""
        if self.i >= self.n:
            return StepResult(self.state, 0.0, True, {"reason": "done"})

        old_E = self.E
        acts = self.valid_actions()

        if not acts:
            # no legal pair: implicit skip
            self.i += 1
            self._advance_until_candidate()
        else:
            a, j = action
            assert a == "pair" and j is not None and j > self.i
            assert self.pairing[self.i] == -1 and self.pairing[j] == -1
            # final guard: never allow a crossing
            if not self._pair_allowed(self.i, j):
                raise ValueError(
                    f"Invalid pair ({self.i},{j}) attempts crossing with existing "
                    f"{[(k,l) for k,l in enumerate(self.pairing) if l>k]}"
                )
            # commit pair
            self.pairing[self.i] = j
            self.pairing[j] = self.i
            # move to next actionable i
            self.i += 1
            self._advance_until_candidate()

        # recompute energy and reward
        self.E = self.energy.total_energy(self.seq, self.pairing)
        reward = (old_E - self.E)  # positive if energy decreased

        done = self.i >= self.n
        if done:
            reward += -self.E  # terminal bonus: lower final E => larger reward

        info = {"dot_bracket": to_dot_bracket(self.pairing), "energy": self.E}
        return StepResult(self.state, reward, done, info)
