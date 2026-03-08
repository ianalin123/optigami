"""
Matplotlib-based crease pattern renderer.
Used for quick observability during training and debugging.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Optional


# Design system colors
_COLOR_MOUNTAIN = "#f59e0b"
_COLOR_VALLEY = "#38bdf8"
_COLOR_PAPER = "#fafaf5"
_COLOR_PAPER_EDGE = "#e2e8f0"
_COLOR_AX_BG = "#1a1a2e"
_COLOR_ANCHOR = "#4a4a6a"
_COLOR_REWARD_BG = "#13131d"
_COLOR_GRID = "#2a2a3a"
_COLOR_VALIDITY = "#22d3ee"
_COLOR_PROGRESS = "#22c55e"
_COLOR_ECONOMY = "#a78bfa"


def draw_paper_state(ax, paper_state, target=None, step=None, reward=None):
    """
    Draw the current crease pattern on a matplotlib axes object.

    Args:
        ax: matplotlib axes
        paper_state: PaperState instance
        target: optional FOLD dict for target crease ghost overlay
        step: step number for title (None = "Initial")
        reward: unused, kept for signature compatibility
    """
    ax.set_facecolor(_COLOR_AX_BG)

    # Unit square paper
    square = patches.Rectangle(
        (0, 0), 1, 1,
        facecolor=_COLOR_PAPER,
        edgecolor=_COLOR_PAPER_EDGE,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(square)

    # Target ghost overlay
    if target is not None:
        verts = target["vertices_coords"]
        edges_v = target["edges_vertices"]
        edges_a = target["edges_assignment"]
        for (v1, v2), assignment in zip(edges_v, edges_a):
            if assignment not in ("M", "V"):
                continue
            x1, y1 = verts[v1]
            x2, y2 = verts[v2]
            color = _COLOR_MOUNTAIN if assignment == "M" else _COLOR_VALLEY
            ax.plot(
                [x1, x2], [y1, y2],
                color=color,
                alpha=0.2,
                linewidth=1,
                linestyle="--",
                zorder=2,
            )

    # Current crease edges
    for edge in paper_state.crease_edges():
        x1, y1 = edge["v1"]
        x2, y2 = edge["v2"]
        assignment = edge["assignment"]
        color = _COLOR_MOUNTAIN if assignment == "M" else _COLOR_VALLEY
        ax.plot(
            [x1, x2], [y1, y2],
            color=color,
            linewidth=2.5,
            linestyle="-",
            solid_capstyle="round",
            zorder=3,
        )
        # Endpoint dots
        ax.plot(
            [x1, x2], [y1, y2],
            color=color,
            marker="o",
            markersize=5,
            linestyle="none",
            zorder=4,
        )

    # Anchor points as gray crosses
    for x, y in paper_state.anchor_points():
        ax.plot(
            x, y,
            color=_COLOR_ANCHOR,
            marker="+",
            markersize=3,
            linestyle="none",
            zorder=5,
        )

    # Title
    title = f"Step {step}" if step is not None else "Initial"
    ax.set_title(title, color="white", fontfamily="monospace", fontsize=10, pad=6)

    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")


def draw_reward_bars(ax, reward: dict):
    """
    Draw a horizontal bar chart of reward components.

    Args:
        ax: matplotlib axes
        reward: dict with keys kawasaki, maekawa, blb, progress, economy (all 0-1)
    """
    components = ["kawasaki", "maekawa", "blb", "progress", "economy"]
    colors = {
        "kawasaki": _COLOR_VALIDITY,
        "maekawa": _COLOR_VALIDITY,
        "blb": _COLOR_VALIDITY,
        "progress": _COLOR_PROGRESS,
        "economy": _COLOR_ECONOMY,
    }

    values = [float(reward.get(c, 0.0)) for c in components]

    ax.set_facecolor(_COLOR_REWARD_BG)

    bar_colors = [colors[c] for c in components]
    bars = ax.barh(
        components,
        values,
        height=0.6,
        color=bar_colors,
        zorder=2,
    )

    # Value labels at end of each bar
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            color="white",
            fontfamily="monospace",
            fontsize=8,
            zorder=3,
        )

    # Y-axis label style
    ax.tick_params(axis="y", colors="white", labelsize=8)
    for label in ax.get_yticklabels():
        label.set_fontfamily("monospace")

    # Subtle x gridlines
    for x_pos in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x_pos, color=_COLOR_GRID, linewidth=0.8, zorder=1)

    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.tick_params(axis="x", colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Reward Breakdown", color="white", fontfamily="monospace", fontsize=10, pad=6)


def render_episode(fold_history, target, rewards_history, save_path=None):
    """
    Create a multi-panel figure showing an entire episode.

    Args:
        fold_history: list of PaperState snapshots (one per step)
        target: FOLD dict of target crease pattern
        rewards_history: list of reward dicts (one per step)
        save_path: if provided, save PNG here; otherwise plt.show()

    Returns:
        matplotlib Figure
    """
    n_states = len(fold_history)
    show_states = min(n_states, 4)

    fig = plt.figure(figsize=(4 * show_states + 4, 5), facecolor="#0d0d14")
    gs = fig.add_gridspec(
        1, show_states + 1,
        width_ratios=[1] * show_states + [1.2],
        wspace=0.3,
    )

    # Paper state panels (up to 4)
    for i in range(show_states):
        # Evenly sample from fold_history if more than 4 steps
        idx = int(i * (n_states - 1) / max(show_states - 1, 1)) if show_states > 1 else 0
        ax = fig.add_subplot(gs[0, i])
        draw_paper_state(
            ax,
            fold_history[idx],
            target=target,
            step=idx + 1,
            reward=rewards_history[idx] if idx < len(rewards_history) else None,
        )

    # Reward curves panel
    ax_reward = fig.add_subplot(gs[0, show_states])
    ax_reward.set_facecolor(_COLOR_REWARD_BG)

    steps = list(range(1, len(rewards_history) + 1))
    curve_specs = [
        ("progress", _COLOR_PROGRESS, "progress"),
        ("kawasaki", _COLOR_VALIDITY, "kawasaki"),
        ("total", "#f8fafc", "total"),
    ]

    for key, color, label in curve_specs:
        vals = [r.get(key, 0.0) for r in rewards_history]
        ax_reward.plot(steps, vals, color=color, linewidth=1.5, label=label)

    ax_reward.set_xlim(1, max(len(rewards_history), 1))
    ax_reward.set_title("Reward Curves", color="white", fontfamily="monospace", fontsize=10, pad=6)
    ax_reward.tick_params(colors="white", labelsize=8)
    ax_reward.legend(
        fontsize=7,
        facecolor=_COLOR_REWARD_BG,
        edgecolor=_COLOR_GRID,
        labelcolor="white",
    )
    for spine in ax_reward.spines.values():
        spine.set_color(_COLOR_GRID)

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="#0d0d14", bbox_inches="tight")
    else:
        plt.show()

    return fig


def render_training_curves(log_path: str):
    """
    Read a JSONL log file and plot training curves.

    Each line must be a JSON object with reward component keys.

    Args:
        log_path: path to JSONL training log

    Returns:
        matplotlib Figure
    """
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    episodes = list(range(1, len(records) + 1))

    keys_to_plot = [
        ("total", "#f8fafc", "total reward"),
        ("progress", _COLOR_PROGRESS, "progress"),
        ("kawasaki", _COLOR_VALIDITY, "kawasaki"),
        ("maekawa", _COLOR_VALIDITY, "maekawa"),
        ("blb", _COLOR_VALIDITY, "blb"),
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 6),
        facecolor="#0d0d14",
        gridspec_kw={"hspace": 0.4},
    )

    # Top: total + progress
    ax_top = axes[0]
    ax_top.set_facecolor(_COLOR_REWARD_BG)
    for key, color, label in keys_to_plot[:2]:
        vals = [r.get(key, 0.0) for r in records]
        ax_top.plot(episodes, vals, color=color, linewidth=1.5, label=label)
    ax_top.set_title("Training: Total & Progress", color="white", fontfamily="monospace", fontsize=10)
    ax_top.tick_params(colors="white", labelsize=8)
    ax_top.legend(fontsize=8, facecolor=_COLOR_REWARD_BG, edgecolor=_COLOR_GRID, labelcolor="white")
    for spine in ax_top.spines.values():
        spine.set_color(_COLOR_GRID)

    # Bottom: kawasaki, maekawa, blb
    ax_bot = axes[1]
    ax_bot.set_facecolor(_COLOR_REWARD_BG)
    for key, color, label in keys_to_plot[2:]:
        vals = [r.get(key, 0.0) for r in records]
        ax_bot.plot(episodes, vals, color=color, linewidth=1.5, label=label, alpha=0.85)
    ax_bot.set_title("Training: Validity Checks", color="white", fontfamily="monospace", fontsize=10)
    ax_bot.set_xlabel("Episode", color="white", fontsize=9)
    ax_bot.tick_params(colors="white", labelsize=8)
    ax_bot.legend(fontsize=8, facecolor=_COLOR_REWARD_BG, edgecolor=_COLOR_GRID, labelcolor="white")
    for spine in ax_bot.spines.values():
        spine.set_color(_COLOR_GRID)

    return fig
