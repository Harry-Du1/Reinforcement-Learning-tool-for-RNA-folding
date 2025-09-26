from __future__ import annotations
import numpy as np


def seed_all(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)