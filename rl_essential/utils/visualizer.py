from __future__ import annotations
import math
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .structures import to_dot_bracket
import numpy as np




def _rainbow_color(t: float):
    """t in [0,1] → RGBA from hsv colormap."""
    return cm.hsv(t)




def plot_rainbow(
    seq: str,
    pairing: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_bases: bool = True,
    height: float = 0.65,         # fixed arc height (0..~1); single “rainbow” look
    linewidth: float = 1.6,
):
    """
    Draw a clean rainbow arc diagram:
      - All arcs same height (fixed curvature).
      - Colors form ONE spectrum left→right by arc *midpoint* (not by i),
        so you don't get multiple mini-rainbows.
    """
    n = len(seq)
    assert n == len(pairing), "seq/pairing length mismatch"

    # collect pairs and sort by midpoint so colors sweep left→right ONCE
    pairs = [(i, j) for i, j in enumerate(pairing) if j > i]
    pairs_sorted = sorted(pairs, key=lambda p: (p[0] + p[1]) / 2.0)
    m = max(1, len(pairs_sorted))

    # figure / axes
    fig_h = 3.5
    fig_w = max(6, n * 0.12)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    ax.set_xlim(-1, n)
    ax.set_ylim(-0.15, height + 0.1)
    ax.axis("off")

    # baseline
    ax.hlines(0, 0, n - 1, linewidth=1)

    # ticks & bases
    for i, b in enumerate(seq):
        ax.vlines(i, 0, 0.03, linewidth=1)
        if show_bases:
            ax.text(i, -0.08, b, ha="center", va="top", fontsize=9)

    # draw arcs with a single rainbow sweep by midpoint rank
    for rank, (i, j) in enumerate(pairs_sorted):
        cx = (i + j) / 2.0
        r = (j - i) / 2.0

        # semicircle at fixed height factor
        ts = [t / 60.0 * math.pi for t in range(61)]
        xs = [cx + r * math.cos(t) for t in ts]
        ys = [height * math.sin(t) for t in ts]

        color = cm.hsv(rank / (m - 1) if m > 1 else 0.0)
        ax.plot(xs, ys, linewidth=linewidth, color=color)

    # paired markers
    for i, j in enumerate(pairing):
        if j == -1:
            ax.plot(i, 0, marker="o", ms=2.4, color="#444444")
        elif j > i:
            ax.plot([i, j], [0, 0], ls="None", marker="o", ms=3.0, color="#000000")

    db = to_dot_bracket(pairing)
    ttl = (title + " — " if title else "") + db
    ax.set_title(ttl, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax