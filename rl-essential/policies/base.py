from __future__ import annotations
from typing import Tuple, Optional


Action = Tuple[str, Optional[int]]


class Policy:
    def act(self, state, valid_actions: list[Action]):
        raise NotImplementedError 

  