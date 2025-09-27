from __future__ import annotations
import math
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .structures import to_dot_bracket




def _rainbow_color(t: float):
    """t in [0,1] → RGBA from hsv colormap."""
    return cm.hsv(t)




def plot_rainbow(seq: str, pairing: List[int], title: Optional[str] = None, save_path: Optional[str] = None, show_bases: bool = True):
    """Plot a 2D rainbow arc diagram of an RNA secondary structure.


    - x-axis: sequence positions (0..n-1)
    - arcs: semicircles for base pairs (i,j), colored by normalized i (rainbow)
    - unpaired bases shown as ticks; paired bases highlighted


    Args:
    seq: RNA sequence (A,U,G,C)
    pairing: list[int], partner index or -1
    title: optional figure title (dot-bracket is appended automatically)
    save_path: if provided, save to this path (PNG/SVG)
    show_bases: if True, annotate bases along the axis
    """
    n = len(seq)
    assert n == len(pairing), "seq and pairing length mismatch"


    fig, ax = plt.subplots(figsize=(max(6, n*0.12), 3.5), dpi=120)


    # axis styling
    ax.set_xlim(-1, n)
    ax.set_ylim(0, 1.05)
    ax.axis("off")


    # baseline
    ax.hlines(0, 0, n-1, linewidth=1)


    # draw ticks and bases
    for i, b in enumerate(seq):
        ax.vlines(i, 0, 0.03, linewidth=1)
        if show_bases:
            ax.text(i, -0.06, b, ha="center", va="top", fontsize=9)


    # draw arcs for pairs (i<j)
    for i, j in enumerate(pairing):
        if j > i:
            x0, x1 = i, j
            cx = (x0 + x1) / 2.0
            r = (x1 - x0) / 2.0
            # parametric semicircle
            ts = [t/60.0*math.pi for t in range(61)]
            xs = [cx + r*math.cos(t) for t in ts]
            ys = [0 + r*math.sin(t)/r for t in ts] # normalized to height 1
            # color by i position
            c = _rainbow_color(i / max(1, n-1))
            ax.plot(xs, ys, linewidth=1.8, color=c)


    # paired markers
    for i, j in enumerate(pairing):
        if j == -1:
            ax.plot(i, 0, marker="o", markersize=2.5, color="#444444")
        elif j > i:
            ax.plot([i, j], [0, 0], linestyle="None", marker="o", markersize=3.2, color="#000000")


    db = to_dot_bracket(pairing)
    ttl = (title + " — " if title else "") + db
    ax.set_title(ttl, fontsize=11)


    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax