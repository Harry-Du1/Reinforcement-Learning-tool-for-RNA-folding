from __future__ import annotations
import argparse
from .env import RNARLEnv
from .policies.random_policy import RandomPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


env = RNARLEnv(args.seq, seed=args.seed)
policy = RandomPolicy(seed=args.seed)


state = env.reset()
step = 0
ret = 0.0
while True:
    acts = env.valid_actions()
    a = policy.act(state, acts)
    sr = env.step(a)
    ret += sr.reward
    print(f"t={step:02d} i={sr.state[0]} a={a} E={sr.info['energy']:.3f} r={sr.reward:.3f} db={sr.info['dot_bracket']}")
    step += 1
    state = sr.state
    if sr.done:
        print(f"Return: {ret:.3f}")
    break