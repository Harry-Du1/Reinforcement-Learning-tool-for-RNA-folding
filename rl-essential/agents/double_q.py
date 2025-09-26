from __future__ import annotations
from collections import defaultdict
from typing import Tuple, Optional, List
import numpy as np


Action = Tuple[str, Optional[int]]

class DoubleQ:
    """Sophisticated Double Q-learning agent.


    Features:
    - Maintains two Q-tables (Q1, Q2) to reduce overestimation bias.
    - ε-greedy with linear decay.
    - State hashing with richer features (index, paired count, % bases paired).
    - Learning-rate scheduler (decay).
    - Optional reward normalization (running mean/std).
    """


    def __init__(self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps: float = 0.3,
        eps_min: float = 0.05,
        eps_decay: float = 0.999,
        alpha_decay: float = 0.9995,
        normalize_rewards: bool = True,
        seed: int | None = None,
        ):
        self.Q1 = defaultdict(lambda: defaultdict(float))
        self.Q2 = defaultdict(lambda: defaultdict(float))


        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.alpha_decay = alpha_decay
        self.normalize_rewards = normalize_rewards


        self.rng = np.random.default_rng(seed)


        # for reward normalization
        self.r_mean = 0.0
        self.r_var = 1.0
        self.r_count = 1e-6
    
    def _hash_state(self, state) -> tuple:
        i, pairing = state
        paired_count = sum(1 for p in pairing if p != -1) // 2
        n = len(pairing)
        frac = paired_count / max(1, n // 2)
        return (i, paired_count, round(frac, 2))
    def _normalize_reward(self, r: float) -> float:
        if not self.normalize_rewards:
            return r
        # online Welford update
        self.r_count += 1
        delta = r - self.r_mean
        self.r_mean += delta / self.r_count
        self.r_var += delta * (r - self.r_mean)
        std = (self.r_var / self.r_count) ** 0.5
        return (r - self.r_mean) / (std + 1e-8)


# --- policy ---
def act(self, state, valid_actions: List[Action]):
    s = self._hash_state(state)
    if not valid_actions:
        return ("skip", None)


    # ε-greedy
    if self.rng.random() < self.eps:
        return valid_actions[self.rng.integers(len(valid_actions))]


# greedy under Q1+Q2
    best, vbest = None, -1e18
    for a in valid_actions:
        v = self.Q1[s][a] + self.Q2[s][a]
    if v > vbest:
        best, vbest = a, v
    return best if best is not None else valid_actions[0]

# --- learning ---
def update(self, state, action, reward, next_state, next_valid_actions: List[Action]):
    s = self._hash_state(state)
    ns = self._hash_state(next_state)
    r = self._normalize_reward(reward)


    if self.rng.random() < 0.5:
        # update Q1
        if next_valid_actions:
            a_star = max(next_valid_actions, key=lambda a: self.Q1[ns][a])
            target = r + self.gamma * self.Q2[ns][a_star]
        else:
            target = r
            self.Q1[s][action] += self.alpha * (target - self.Q1[s][action])
    else:
    # update Q2
        if next_valid_actions:
            a_star = max(next_valid_actions, key=lambda a: self.Q2[ns][a])
            target = r + self.gamma * self.Q1[ns][a_star]
        else:
            target = r
            self.Q2[s][action] += self.alpha * (target - self.Q2[s][action])


    # decay schedules
    self.alpha = max(1e-4, self.alpha * self.alpha_decay)
    self.eps = max(self.eps_min, self.eps * self.eps_decay)