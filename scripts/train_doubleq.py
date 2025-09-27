from __future__ import annotations
import argparse
from rl_essential.env import RNARLEnv
from rl_essential.agents.double_q import DoubleQ

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

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

for ep in range(1, args.episodes + 1):
    G = run_episode(args.seq)
    if ep % 100 == 0:
        print(f"Episode {ep:5d}  Return {G:8.3f}")