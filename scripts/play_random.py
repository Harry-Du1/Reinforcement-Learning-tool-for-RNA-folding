from __future__ import annotations
import argparse
from rl_essential.env import RNARLEnv
from rl_essential.agents.double_q import DoubleQ

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = RNARLEnv(args.seq, seed=args.seed)
agent = DoubleQ(seed=args.seed)

s = env.reset()
ret = 0.0
step = 0
while True:
    acts = env.valid_actions()
    a = agent.act(s, acts)
    sr = env.step(a)
    ret += sr.reward
    print(f"t={step:02d} i={sr.state[0]} a={a} E={sr.info['energy']:.3f} r={sr.reward:.3f} db={sr.info['dot_bracket']}")
    step += 1
    s = sr.state
    if sr.done:
        print(f"Return: {ret:.3f}")
        break