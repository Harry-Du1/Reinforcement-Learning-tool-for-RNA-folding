from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional


Pairing = list[int] # partner index or -1
Action = Tuple[str, Optional[int]] # ("pair", j) or ("skip", None)


@dataclass
class Transition:
    state: Tuple[int, Pairing]
    action: Action
    reward: float
    next_state: Tuple[int, Pairing]
    done: bool