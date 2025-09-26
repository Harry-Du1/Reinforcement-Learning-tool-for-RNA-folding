from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from .base import Policy


Action = Tuple[str, Optional[int]]


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)


    def act(self, state, valid_actions: list[Action]):
        idx = self.rng.integers(len(valid_actions))
        return valid_actions[idx]