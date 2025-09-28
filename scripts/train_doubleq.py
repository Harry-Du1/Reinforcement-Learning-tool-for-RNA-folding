# scripts/train_doubleq.py
from __future__ import annotations
import argparse, os, pickle
from rl_essential.env import RNARLEnv
from rl_essential.agents.double_q import DoubleQ

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--ckpt", type=str, default="checkpoints/dq.pkl")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
agent = DoubleQ()  # no env seed needed; env is deterministic

def run_episode(seq: str):
    env = RNARLEnv(seq)  # <— removed seed
    s = env.reset()
    G = 0.0
    while True:
        acts = env.valid_actions()
        if not acts:
            # no legal pair at i → env auto-advances on dummy step
            sr = env.step(("pair", None))
        else:
            a = agent.act(s, acts)
            sr = env.step(a)
            nacts = env.valid_actions()
            agent.update(s, a, sr.reward, sr.state, nacts)
        G += sr.reward
        s = sr.state
        if sr.done:
            return G, sr.info

for ep in range(1, args.episodes + 1):
    G, info = run_episode(args.seq)
    if ep % 100 == 0:
        print(f"Episode {ep:5d}  Return {G:8.3f}  DB {info['dot_bracket']}  E {info['energy']:.2f}", flush=True)

# save trained tables
Q1 = {k: dict(v) for k, v in agent.Q1.items()}
Q2 = {k: dict(v) for k, v in agent.Q2.items()}
with open(args.ckpt, "wb") as f:
    pickle.dump({"Q1": Q1, "Q2": Q2}, f)
print(f"Saved DoubleQ checkpoint to {args.ckpt}")
