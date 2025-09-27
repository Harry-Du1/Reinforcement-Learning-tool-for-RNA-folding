from __future__ import annotations
from collections import deque
import random

class Replay:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def add(self, item):
        self.buf.append(item)
    def sample(self, batch):
        return random.sample(self.buf, min(batch, len(self.buf)))
    def __len__(self):
        return len(self.buf)