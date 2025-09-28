# scripts/eval_doubleq.py
from __future__ import annotations
import argparse, pickle
from rl_essential.env import RNARLEnv
from rl_essential.agents.double_q import DoubleQ
from rl_essential.utils.visualizer import plot_rainbow

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--ckpt", type=str, default="checkpoints/dq.pkl")
parser.add_argument("--out", type=str, default="final.png")
args = parser.parse_args()

agent = DoubleQ()
with open(args.ckpt, "rb") as f:
    data = pickle.load(f)
agent.Q1.update({k: {tuple(a): v for a, v in vv.items()} for k, vv in data["Q1"].items()})
agent.Q2.update({k: {tuple(a): v for a, v in vv.items()} for k, vv in data["Q2"].items()})

env = RNARLEnv(args.seq)  # <— removed seed
s = env.reset()
while True:
    acts = env.valid_actions()
    if not acts:
        sr = env.step(("pair", None))  # auto-advance
    else:
        hs = agent._hash_state(s)
        a = max(acts, key=lambda act: agent.Q1[hs][act] + agent.Q2[hs][act])
        sr = env.step(a)
    s = sr.state
    if sr.done:
        print("DB:", sr.info["dot_bracket"], "Energy:", sr.info["energy"])
        plot_rainbow(args.seq, env.pairing, title="Final", save_path=args.out)
        print(f"Saved plot to {args.out}")
        break
