from __future__ import annotations
from typing import List, Tuple
import torch

BASE2IDX = {"A":0, "U":1, "G":2, "C":3}

class GraphEncoder:
    """Graph-aware encoder: nodes = positions; edges = backbone (k,k+1) and pairing (k, pairing[k]).
    Features per node: one-hot base(4) + paired flag(1) + degree(1) + helix_id one-hot (pooled) + is_current(1).
    Aggregation: mean + max over nodes; plus global helix histogram (5 bins by length).
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _helices(self, pairing: List[int]) -> List[Tuple[int,int,int]]:
        n = len(pairing)
        helices = []
        i = 0
        used = set()
        while i < n:
            j = pairing[i]
            if j > i and (i,j) not in used:
                L = 1
                while i+L < j-L and pairing[i+L] == j-L:
                    used.add((i+L, j-L))
                    L += 1
                helices.append((i, j, L))
                i += L
            else:
                i += 1
        return helices

    def _helix_bin(self, L: int) -> int:
        if L <= 2: return 0
        if L <= 4: return 1
        if L <= 8: return 2
        if L <= 16: return 3
        return 4

    def encode(self, seq: str, state: Tuple[int, List[int]]) -> torch.Tensor:
        i, pairing = state
        n = len(seq)
        helices = self._helices(pairing)
        # Map each node to helix bin id (coarse). If unassigned, -1.
        helix_id = [-1]*n
        for h_idx, (a,b,L) in enumerate(helices):
            for k in range(L):
                helix_id[a+k] = self._helix_bin(L)
                helix_id[b-k] = self._helix_bin(L)
        # node features
        feats = []
        for k, b in enumerate(seq):
            xb = [0.0]*4; xb[BASE2IDX[b]] = 1.0
            paired = 1.0 if pairing[k] != -1 else 0.0
            deg = 2.0  # backbone deg (k-1,k+1)
            if k == 0 or k == n-1: deg -= 1.0
            if pairing[k] != -1: deg += 1.0
            hbins = [0.0]*5
            if helix_id[k] >= 0: hbins[helix_id[k]] = 1.0
            cur = 1.0 if k == i else 0.0
            feats.append(xb + [paired, deg] + hbins + [cur])
        X = torch.tensor(feats, dtype=torch.float32, device=self.device)
        mean = X.mean(dim=0)
        mx, _ = X.max(dim=0)
        # global helix histogram
        hist = [0]*5
        for _,_,L in helices:
            hist[self._helix_bin(L)] += 1
        H = torch.tensor(hist, dtype=torch.float32, device=self.device)
        return torch.cat([mean, mx, H], dim=0)  # ( (4+1+1+5+1)*2 + 5 = (12)*2 + 5 = 29 )