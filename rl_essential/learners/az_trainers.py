from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from ..networks.encoder import SimpleEncoder
from ..networks.graph_encoder import GraphEncoder
from ..networks.policy_value import PolicyValueNet

class AlphaZeroTrainer:
    def __init__(self, encoder: str = "graph", a_max: int = 256, lr=1e-3, device="cpu"):
        self.device = device
        if encoder == "graph":
            self.encoder = GraphEncoder(device=device)
            in_dim = 29
        else:
            self.encoder = SimpleEncoder(device=device)
            in_dim = 24
        self.net = PolicyValueNet(in_dim=in_dim, a_max=a_max).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.pi_loss = nn.KLDivLoss(reduction="batchmean")
        self.v_loss = nn.MSELoss()

    def batch(self, samples):
        X, P, V = [], [], []
        for (seq, state, pi, v) in samples:
            x = self.encoder.encode(seq, state)
            X.append(x)
            import torch as T
            P.append(T.tensor(pi, dtype=T.float32, device=self.device))
            V.append(T.tensor([v], dtype=T.float32, device=self.device))
        import torch as T
        X = T.stack(X).to(self.device)
        P = T.stack(P)
        V = T.cat(V)
        return X, P, V

    def train_step(self, batch_samples):
        X, P, V = self.batch(batch_samples)
        logits, v_pred = self.net(X)
        k = P.size(1)
        logits = logits[:, :k]
        logp = torch.log_softmax(logits, dim=-1)
        loss_pi = self.pi_loss(logp, P)
        loss_v = self.v_loss(v_pred, V)
        loss = loss_pi + loss_v
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item()), float(loss_pi.item()), float(loss_v.item())