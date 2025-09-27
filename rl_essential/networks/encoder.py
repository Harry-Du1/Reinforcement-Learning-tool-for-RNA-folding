from __future__ import annotations
from typing import List, Tuple
import torch

BASE2IDX = {"A":0, "U":1, "G":2, "C":3}

class SimpleEncoder:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _dist_bucket(self, d: int) -> int:
        if d <= 0: return 0
        if d <= 3: return 1
        if d <= 7: return 2
        if d <= 15: return 3
        if d <= 31: return 4
        return 5

    def encode(self, seq: str, state: Tuple[int, List[int]]) -> torch.Tensor:
        i, pairing = state
        feats = []
        for k, b in enumerate(seq):
            x = [0.0]*4
            x[BASE2IDX[b]] = 1.0
            paired = 1.0 if pairing[k] != -1 else 0.0
            dist = 0 if pairing[k] == -1 else abs(pairing[k]-k)
            db = [0.0]*6
            db[self._dist_bucket(dist)] = 1.0
            cur = 1.0 if k == i else 0.0
            feats.append(x + [paired] + db + [cur])
        X = torch.tensor(feats, dtype=torch.float32, device=self.device)
        mean = X.mean(dim=0)
        mx, _ = X.max(dim=0)
        return torch.cat([mean, mx], dim=0)  # (24,)