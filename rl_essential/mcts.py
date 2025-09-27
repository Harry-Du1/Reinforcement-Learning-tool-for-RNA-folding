from __future__ import annotations
import math
import torch
from typing import List, Tuple, Optional, Dict

Action = Tuple[str, Optional[int]]

class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.children: Dict[Action, Node] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P: Dict[Action, float] = {}
        self.terminal = False
        self.value = 0.0
        self.state = None

class PUCT:
    """PUCT that supports pointer-style policy with unbounded actions.
    If net exposes pointer interface, we score only the valid j's per state.
    """
    def __init__(self, env_cls, encoder, net, use_pointer: bool = True, n_sim=100, c_puct=1.4, device="cpu"):
        self.env_cls = env_cls
        self.encoder = encoder
        self.net = net
        self.use_pointer = use_pointer
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.device = device

    def _policy_priors(self, env):
        valid = env.valid_actions()
        if not self.use_pointer:
            # fallback: require fixed-size logits; mask invalid outside caller
            raise RuntimeError("Pointer disabled; use fixed policy head.")
        # Build candidate list: all pair actions first, then skip as last
        acts = []
        j_candidates = []
        for a in valid:
            if a[0] == "pair":
                acts.append(a)
                j_candidates.append(a[1])
        acts.append(("skip", None))
        # Encode
        X_nodes = self.encoder.node_features(env.seq, env.state).unsqueeze(0)
        g_vec = self.encoder.encode(env.seq, env.state).unsqueeze(0)
        i_idx = torch.tensor([env.state[0]], dtype=torch.long)
        j_idx = [torch.tensor(j_candidates, dtype=torch.long)]
        with torch.no_grad():
            logits_list, v = self.net(g_vec, X_nodes, i_idx, j_idx)
        logits = logits_list[0]
        pi = torch.softmax(logits, dim=-1)
        P = {a: float(pi[k]) for k, a in enumerate(acts)}
        return acts, P, float(v.item())

    def _puct_score(self, node: Node, a: Action, c: float) -> float:
        child = node.children.get(a)
        if child is None:
            return c * node.P.get(a, 0.0) * math.sqrt(node.N + 1e-8)
        return child.Q + c * node.P.get(a, 0.0) * math.sqrt(node.N + 1e-8) / (1 + child.N)

    def search(self, root_env):
        root = Node(parent=None)
        root.state = root_env.state
        acts, P, v = self._policy_priors(root_env)
        root.P = P
        root.value = v

        for _ in range(self.n_sim):
            node = root
            env = self.env_cls(root_env.seq, energy_model=root_env.energy)
            env.i, env.pairing, env.E = root_env.i, root_env.pairing.copy(), root_env.E
            path = []
            while True:
                if env.i >= env.n:
                    node.terminal = True
                    break
                valid = env.valid_actions()
                if not valid:
                    node.terminal = True
                    break
                if len(node.children) == 0:
                    acts, P, v = self._policy_priors(env)
                    node.P = P
                    node.value = v
                    break
                best_a, best_s = None, -1e18
                for a in valid + [("skip", None)]:
                    s = self._puct_score(node, a, self.c_puct)
                    if s > best_s:
                        best_a, best_s = a, s
                path.append((node, best_a))
                sr = env.step(best_a)
                if best_a not in node.children:
                    node.children[best_a] = Node(parent=node)
                    node.children[best_a].state = sr.state
                node = node.children[best_a]
                if sr.done:
                    node.terminal = True
                    node.value = -sr.info["energy"]
                    break
            v = node.value
            for parent, a in reversed(path):
                child = parent.children[a]
                child.N += 1
                child.W += v
                child.Q = child.W / child.N
                parent.N += 1
        # Build improved policy from visit counts
        valid = root_env.valid_actions()
        acts = [a for a in valid] + [("skip", None)]
        counts = [(root.children[a].N if a in root.children else 0) for a in acts]
        total = sum(counts) + 1e-8
        pi = [c/total for c in counts]
        return pi, acts