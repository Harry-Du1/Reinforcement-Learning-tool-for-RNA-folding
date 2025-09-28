from __future__ import annotations
from collections import defaultdict
from typing import Tuple, Optional, List
import numpy as np

Action = Tuple[str, Optional[int]]

class DoubleQ:
    # Add these guards inside your DoubleQ class.

    def act(self, state, actions):
        # If no legal actions, return a harmless dummy; env will auto-advance.
        if not actions:
            return ("pair", None)
        import random
        if random.random() < self.eps:
            return random.choice(actions)
        hs = self._hash_state(state)
        # Greedy on Q1+Q2
        return max(actions, key=lambda a: self.Q1[hs][a] + self.Q2[hs][a])

    def update(self, s, a, r, s_next, next_actions):
        # Standard Double Q-learning, but handle terminal / no-action next states.
        hs  = self._hash_state(s)
        hs2 = self._hash_state(s_next)
        if not next_actions:
            target = r      # no bootstrap when no next actions (terminal-like)
        else:
            # pick argmax under Q1 then evaluate with Q2 (or vice versa)
            import random
            if random.random() < 0.5:
                a_star = max(next_actions, key=lambda aa: self.Q1[hs2][aa])
                target = r + self.gamma * self.Q2[hs2][a_star]
            else:
                a_star = max(next_actions, key=lambda aa: self.Q2[hs2][aa])
                target = r + self.gamma * self.Q1[hs2][a_star]
        # do the update on Q1 or Q2
        import random
        if random.random() < 0.5:
            self.Q1[hs][a] += self.alpha * (target - self.Q1[hs][a])
        else:
            self.Q2[hs][a] += self.alpha * (target - self.Q2[hs][a])

    def __init__(self, alpha=0.1, gamma=0.99, eps=0.3, eps_min=0.05, eps_decay=0.999, alpha_decay=0.9995, seed: int | None = None):
        self.Q1 = defaultdict(lambda: defaultdict(float))
        self.Q2 = defaultdict(lambda: defaultdict(float))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.alpha_decay = alpha_decay
        self.rng = np.random.default_rng(seed)

    def _hash_state(self, state) -> tuple:
        i, pairing = state
        paired_count = sum(1 for p in pairing if p != -1) // 2
        n = len(pairing)
        frac = paired_count / max(1, n // 2)
        return (i, paired_count, round(frac, 2))

    def act(self, state, valid_actions: List[Action]):
        s = self._hash_state(state)
        if not valid_actions:
            return ("skip", None)
        if self.rng.random() < self.eps:
            return valid_actions[self.rng.integers(len(valid_actions))]
        best, val = None, -1e18
        for a in valid_actions:
            q = self.Q1[s][a] + self.Q2[s][a]
            if q > val: best, val = a, q
        return best if best is not None else valid_actions[0]

    def update(self, state, action, reward, next_state, next_valid_actions):
        s = self._hash_state(state)
        ns = self._hash_state(next_state)
        if self.rng.random() < 0.5:
            if next_valid_actions:
                a_star = max(next_valid_actions, key=lambda a: self.Q1[ns][a])
                target = reward + self.gamma * self.Q2[ns][a_star]
            else:
                target = reward
            self.Q1[s][action] += self.alpha * (target - self.Q1[s][action])
        else:
            if next_valid_actions:
                a_star = max(next_valid_actions, key=lambda a: self.Q2[ns][a])
                target = reward + self.gamma * self.Q1[ns][a_star]
            else:
                target = reward
            self.Q2[s][action] += self.alpha * (target - self.Q2[s][action])
        self.alpha = max(1e-4, self.alpha * self.alpha_decay)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)