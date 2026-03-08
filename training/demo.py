"""
training/demo.py — Run 8 zero-shot rollouts and stream them to the grid viewer.

Usage:
    cd /path/to/optigami
    python -m training.demo

Then open: http://localhost:9001/viewer/training.html

Each of the 8 "strategies" is a heuristic that mimics what a pretrained LLM might
produce for different tasks — varying from near-optimal to poor.  This exercises
the full broadcast → grid viewer pipeline without requiring an LLM API key.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable

import uvicorn

from server.app import app, broadcast
from training.runner import run_batch


# ── 8 zero-shot heuristic strategies ──────────────────────────────────────────
# Each is a callable: paper_state (dict) → fold_dict
# These represent the range of strategies a pretrained LLM might generate.

def strategy_perfect_half(paper_state: dict) -> dict:
    """Valley fold exactly at horizontal midline — optimal for half_fold."""
    return {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0}


def strategy_slight_offset(paper_state: dict) -> dict:
    """Valley fold slightly off-center — almost optimal."""
    return {"type": "valley", "line": {"start": [0.0, 0.48], "end": [1.0, 0.48]}, "angle": 180.0}


def strategy_thirds(paper_state: dict) -> dict:
    """Letter fold at one-third — wrong for half_fold, generates interesting geometry."""
    fold_count = paper_state.get("fold_count", 0)
    positions = [0.333, 0.667]
    if fold_count >= len(positions):
        return {"type": "stop", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0.0}
    return {
        "type": "valley" if fold_count == 0 else "mountain",
        "line": {"start": [0.0, positions[fold_count]], "end": [1.0, positions[fold_count]]},
        "angle": 180.0,
    }


def strategy_vertical(paper_state: dict) -> dict:
    """Vertical fold — gets compactness but in wrong dimension for target_box."""
    return {"type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180.0}


def strategy_mountain(paper_state: dict) -> dict:
    """Mountain fold at midline — same geometry, different assignment."""
    return {"type": "mountain", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0}


def strategy_accordion(paper_state: dict) -> dict:
    """Accordion 3-fold — overfolds, achieves high compactness but more folds."""
    fold_count = paper_state.get("fold_count", 0)
    positions = [0.25, 0.5, 0.75]
    assignments = ["valley", "mountain", "valley"]
    if fold_count >= len(positions):
        return {"type": "stop", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0.0}
    return {
        "type": assignments[fold_count],
        "line": {"start": [0.0, positions[fold_count]], "end": [1.0, positions[fold_count]]},
        "angle": 180.0,
    }


def strategy_diagonal(paper_state: dict) -> dict:
    """Diagonal fold — achieves compactness but irregular bounding box."""
    return {"type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180.0}


def strategy_quarter(paper_state: dict) -> dict:
    """Two perpendicular folds — 4x compactness for quarter_fold task."""
    fold_count = paper_state.get("fold_count", 0)
    if fold_count == 0:
        return {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0}
    if fold_count == 1:
        return {"type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180.0}
    return {"type": "stop", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0.0}


STRATEGIES: list[tuple[str, Callable]] = [
    ("perfect_half",  strategy_perfect_half),
    ("slight_offset", strategy_slight_offset),
    ("thirds_fold",   strategy_thirds),
    ("vertical_fold", strategy_vertical),
    ("mountain_fold", strategy_mountain),
    ("accordion_3",   strategy_accordion),
    ("diagonal",      strategy_diagonal),
    ("quarter_fold",  strategy_quarter),
]


# ── Demo runner ────────────────────────────────────────────────────────────────

async def run_demo(task_name: str = "half_fold", delay_s: float = 0.5) -> None:
    """Wait for server to be ready, then fire 8 episodes."""
    # Give uvicorn time to bind and call startup hook (sets broadcast._loop)
    await asyncio.sleep(1.5)

    batch_id = 1
    names, fns = zip(*STRATEGIES)
    ep_ids = [f"ep_{name}" for name in names]

    print(f"\n[demo] Starting batch {batch_id} — task: {task_name}")
    print(f"[demo] Open http://localhost:9001/viewer/training.html\n")

    # Signal grid to clear and show G=8
    await broadcast.start_batch(batch_id, len(fns))

    await asyncio.sleep(delay_s)

    # Run all 8 episodes in the thread pool; broadcast_fn fires into this loop
    results = await asyncio.gather(*[
        asyncio.to_thread(
            _run_one,
            fn,
            task_name,
            ep_id,
            broadcast.publish,
        )
        for fn, ep_id in zip(fns, ep_ids)
    ])

    scores = [r["score"] for r in results]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    await broadcast.finish_batch(batch_id, scores, best_episode_id=ep_ids[best_idx])

    print("\n[demo] Results:")
    for name, result in zip(names, results):
        print(f"  {name:20s}  score={result['score']:+.2f}  status={result['status']}")
    print(f"\n[demo] Best: {names[best_idx]} (score={scores[best_idx]:+.2f})")
    print("\n[demo] Grid viewer running. Press Ctrl+C to stop.\n")


def _run_one(
    strategy_fn: Callable,
    task_name: str,
    ep_id: str,
    broadcast_fn: Callable,
) -> dict:
    """Thin wrapper: adds a small sleep between steps so the viewer can animate."""
    from server.models import OrigamiAction
    from server.origami_environment import OrigamiEnvironment

    env = OrigamiEnvironment()
    obs = env.reset(task_name=task_name)

    broadcast_fn(ep_id, {
        "type": "episode_update",
        "episode_id": ep_id,
        "task_name": task_name,
        "step": 0,
        "observation": _obs_dict(obs),
    })

    max_steps = env._task.get("max_folds", 10) if env._task else 10
    status = "done"

    for step_idx in range(max_steps):
        if obs.done:
            break

        time.sleep(0.3)  # pace so the viewer can animate each step

        fold_dict = strategy_fn(obs.paper_state)

        if fold_dict.get("type") == "stop":
            break

        action = OrigamiAction(
            fold_type=fold_dict["type"],
            fold_line=fold_dict["line"],
            fold_angle=float(fold_dict.get("angle", 180.0)),
        )
        obs = env.step(action)

        broadcast_fn(ep_id, {
            "type": "episode_update",
            "episode_id": ep_id,
            "task_name": task_name,
            "step": step_idx + 1,
            "observation": _obs_dict(obs),
        })

        if obs.done:
            break
    else:
        status = "timeout"

    score = obs.reward if obs.reward is not None else env._total_reward or 0.0

    broadcast_fn(ep_id, {
        "type": "episode_done",
        "episode_id": ep_id,
        "status": status,
        "score": float(score),
        "final_metrics": obs.metrics,
    })

    return {
        "episode_id": ep_id,
        "score": float(score),
        "final_metrics": obs.metrics,
        "status": status,
    }


def _obs_dict(obs) -> dict:
    try:
        return obs.model_dump()
    except AttributeError:
        return {
            "paper_state": getattr(obs, "paper_state", {}),
            "metrics": getattr(obs, "metrics", {}),
            "fold_history": getattr(obs, "fold_history", []),
            "done": getattr(obs, "done", False),
            "reward": getattr(obs, "reward", None),
        }


# ── Entry point ────────────────────────────────────────────────────────────────

async def _main() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=9001, log_level="warning")
    server = uvicorn.Server(config)

    # Run demo concurrently with the uvicorn server
    await asyncio.gather(
        server.serve(),
        run_demo(task_name="half_fold"),
    )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[demo] Stopped.")
