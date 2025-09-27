from __future__ import annotations
import torch
import torch.nn as nn

class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int, a_max: int = 256, hidden: int = 256):
        super().__init__()
        self.a_max = a_max
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden, a_max)
        self.v_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())
    def forward(self, x):
        h = self.trunk(x)
        logits = self.pi_head(h)
        v = self.v_head(h).squeeze(-1)
        return logits, v