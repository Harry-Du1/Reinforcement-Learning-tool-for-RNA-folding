from __future__ import annotations
from typing import List, Tuple
from .mcts import PUCT

class SelfPlay:
    def __init__(self, env_cls, encoder, net, mcts_sims=100, device="cpu"):
        self.env_cls = env_cls
        self.encoder = encoder
        self.net = net
        self.device = device
        self.mcts_sims = mcts_sims

    def play_episode(self, seq: str):
        env = self.env_cls(seq)
        s = env.reset()
        traj = []
        while True:
            mcts = PUCT(self.env_cls, self.encoder, self.net, n_sim=self.mcts_sims, device=self.device)
            pi, acts = mcts.search(env)
            import numpy as np
            a_idx = np.random.choice(len(acts), p=pi)
            a = acts[a_idx]
            sr = env.step(a)
            traj.append((seq, s, pi, 0.0))
            s = sr.state
            if sr.done:
                final_reward = -sr.info["energy"]
                traj = [(seq, st, pi, final_reward) for (seq, st, pi, _) in traj]
                return traj, sr.info