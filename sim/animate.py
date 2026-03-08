"""
Matplotlib 3D animation of origami folding using OrigamiSimulator.

Usage:
    python -m sim.animate [target_name]

    target_name defaults to 'half_horizontal', resolved against
    env/targets/<target_name>.fold relative to this file's parent directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .simulator import OrigamiSimulator

# ── Design system colours ─────────────────────────────────────────────────────
BG_COLOR     = '#0d0d14'
AX_COLOR     = '#13131d'
PAPER_FACE   = '#fafaf5'
PAPER_EDGE   = '#2a2a3a'
MOUNTAIN_CLR = '#f59e0b'   # amber
VALLEY_CLR   = '#38bdf8'   # sky


# ── Public API ────────────────────────────────────────────────────────────────

def animate_fold(fold_file: str,
                 n_frames: int = 80,
                 steps_per_frame: int = 40,
                 target_name: str = 'origami') -> None:
    """
    Animate folding from 0% → 100% → 0% in a triangle-wave loop.

    Parameters
    ----------
    fold_file : str
        Path to the .fold JSON file.
    n_frames : int
        Total animation frames (default 80 → ~40 in, 40 out).
    steps_per_frame : int
        Physics steps executed per frame.
    target_name : str
        Display name shown in the title.
    """
    fold_data = json.loads(Path(fold_file).read_text())
    sim = OrigamiSimulator(fold_data, subdivisions=2)

    # Triangle-wave fold percents: 0 → 1 → 0
    half = n_frames // 2
    fold_percents = np.concatenate([
        np.linspace(0.0, 1.0, half),
        np.linspace(1.0, 0.0, n_frames - half),
    ])

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(AX_COLOR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_axis_off()

    def update(frame: int) -> list:
        pct = fold_percents[frame]
        sim.set_fold_percent(pct)
        sim.step(steps_per_frame)

        ax.clear()
        ax.set_facecolor(AX_COLOR)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.set_axis_off()

        # ── Paper surface ─────────────────────────────────────────────────────
        verts = [sim.pos[tri] for tri in sim.triangles]
        poly = Poly3DCollection(
            verts,
            alpha=0.85,
            facecolor=PAPER_FACE,
            edgecolor=PAPER_EDGE,
            linewidth=0.2,
            zorder=1,
        )
        ax.add_collection3d(poly)

        # ── Crease / fold edges ───────────────────────────────────────────────
        for i in range(len(sim._crease_a)):
            if sim._crease_assign[i] not in ('M', 'V'):
                continue
            a, b = sim._crease_a[i], sim._crease_b[i]
            color = MOUNTAIN_CLR if sim._crease_assign[i] == 'M' else VALLEY_CLR
            ax.plot(
                [sim.pos[a, 0], sim.pos[b, 0]],
                [sim.pos[a, 1], sim.pos[b, 1]],
                [sim.pos[a, 2], sim.pos[b, 2]],
                color=color,
                linewidth=2.5,
                zorder=2,
            )

        # ── Axis limits & style ───────────────────────────────────────────────
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_zlim(-0.6, 0.6)
        ax.set_box_aspect([1.4, 1.4, 1.0])
        ax.set_title(
            f'OPTIGAMI — {target_name}  fold: {pct * 100:.0f}%',
            color='#e0e0f0',
            fontsize=13,
            pad=10,
        )

        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=40,   # ms between frames (~25 fps)
        blit=False,
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else 'half_horizontal'
    fold_file = Path(__file__).parent.parent / 'env' / 'targets' / f'{target}.fold'
    if not fold_file.exists():
        print(f'Error: fold file not found: {fold_file}', file=sys.stderr)
        sys.exit(1)
    animate_fold(str(fold_file), target_name=target)


if __name__ == '__main__':
    main()
