# scripts/train_doubleq.py
from __future__ import annotations
import argparse, os, pickle
from rl_essential.env import RNARLEnv
from rl_essential.agents.double_q import DoubleQ

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ckpt", type=str, default="checkpoints/dq.pkl")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
agent = DoubleQ(seed=args.seed)

def run_episode(seq: str):
    env = RNARLEnv(seq, seed=args.seed)
    s = env.reset()
    G = 0.0
    while True:
        acts = env.valid_actions()
        a = agent.act(s, acts)
        sr = env.step(a)
        nacts = env.valid_actions()
        agent.update(s, a, sr.reward, sr.state, nacts)
        G += sr.reward
        s = sr.state
        if sr.done:
            return G

def greedy_eval(seq: str):
    env = RNARLEnv(seq, seed=args.seed)
    s = env.reset()
    # greedy under current Q1+Q2
    while True:
        acts = env.valid_actions()
        if not acts:
            break
        hs = agent._hash_state(s)
        a = max(acts, key=lambda act: agent.Q1[hs][act] + agent.Q2[hs][act])
        sr = env.step(a)
        s = sr.state
        if sr.done:
            return sr.info["dot_bracket"], sr.info["energy"]

for ep in range(1, args.episodes + 1):
    G = run_episode(args.seq)
    if ep % 100 == 0:
        db, e = greedy_eval(args.seq)
        print(f"Episode {ep:5d}  Return {G:8.3f}  DB {db}  E {e:.2f}")

# --- save trained tables ---
# convert defaultdict -> dict for pickle stability
Q1 = {k: dict(v) for k, v in agent.Q1.items()}
Q2 = {k: dict(v) for k, v in agent.Q2.items()}
with open(args.ckpt, "wb") as f:
    pickle.dump({"Q1": Q1, "Q2": Q2}, f)
print(f"Saved DoubleQ checkpoint to {args.ckpt}")
