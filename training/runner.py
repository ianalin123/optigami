"""
TrainingRunner — parallel episode executor for GRPO training.

Each episode runs in a ThreadPoolExecutor thread.
After every env.step(), observations are pushed to the broadcast server (fire-and-forget).
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from server.models import OrigamiAction
from server.origami_environment import OrigamiEnvironment


BroadcastFn = Callable[[str, dict], None]


def run_episode(
    strategy_fn: Callable[[dict], dict],
    task_name: str,
    ep_id: Optional[str] = None,
    broadcast_fn: Optional[BroadcastFn] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Run a single origami episode with a given strategy function.

    Args:
        strategy_fn: Callable that receives paper_state dict and returns a fold dict:
                     {"type": "valley"|"mountain"|"pleat"|"crimp"|"stop",
                      "line": {"start": [x, y], "end": [x, y]},
                      "angle": 180.0}
        task_name: Name of the task (from server/tasks.py)
        ep_id: Episode identifier for broadcast; auto-generated if None
        broadcast_fn: Optional callback(ep_id, data) for live streaming
        max_steps: Override task's max_folds if provided

    Returns:
        dict with keys: episode_id, score, final_metrics, fold_history, status
    """
    ep_id = ep_id or str(uuid.uuid4())[:8]
    env = OrigamiEnvironment()

    obs = env.reset(task_name=task_name)

    if broadcast_fn:
        broadcast_fn(ep_id, {
            "type": "episode_update",
            "episode_id": ep_id,
            "task_name": task_name,
            "step": 0,
            "observation": _obs_to_dict(obs),
        })

    step_limit = max_steps or env._task.get("max_folds", 20) if env._task else 20
    status = "done"

    for step_idx in range(step_limit):
        if obs.done:
            break

        # Strategy generates a fold dict
        try:
            fold_dict = strategy_fn(obs.paper_state)
        except Exception as exc:
            status = "error"
            if broadcast_fn:
                broadcast_fn(ep_id, {
                    "type": "episode_done",
                    "episode_id": ep_id,
                    "status": "error",
                    "score": obs.reward or 0.0,
                    "final_metrics": obs.metrics,
                    "error": str(exc),
                })
            break

        fold_type = fold_dict.get("type", "valley")
        fold_line = fold_dict.get("line", {"start": [0, 0.5], "end": [1, 0.5]})
        fold_angle = float(fold_dict.get("angle", 180.0))

        action = OrigamiAction(
            fold_type=fold_type,
            fold_line=fold_line,
            fold_angle=fold_angle,
        )
        obs = env.step(action)

        if broadcast_fn:
            broadcast_fn(ep_id, {
                "type": "episode_update",
                "episode_id": ep_id,
                "task_name": task_name,
                "step": step_idx + 1,
                "observation": _obs_to_dict(obs),
            })

        if obs.done:
            break
    else:
        status = "timeout"

    score = obs.reward if obs.reward is not None else (env._total_reward or 0.0)

    if broadcast_fn:
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
        "fold_history": obs.fold_history,
        "status": status,
    }


def run_batch(
    strategy_fns: list[Callable[[dict], dict]],
    task_name: str,
    broadcast_fn: Optional[BroadcastFn] = None,
    batch_id: Optional[int] = None,
    max_workers: int = 8,
) -> list[dict]:
    """Run G episodes in parallel with a ThreadPoolExecutor.

    Args:
        strategy_fns: List of G strategy callables (one per completion)
        task_name: Task to use for all episodes
        broadcast_fn: Optional broadcast callback, called after each step
        batch_id: Batch identifier for broadcast
        max_workers: Max parallel threads (bounded by G)

    Returns:
        List of episode result dicts, in same order as strategy_fns
    """
    n = len(strategy_fns)
    ep_ids = [f"ep_{(batch_id or 0):04d}_{i:02d}" for i in range(n)]
    workers = min(max_workers, n)

    results: list[dict] = [{}] * n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_episode,
                fn,
                task_name,
                ep_ids[i],
                broadcast_fn,
            ): i
            for i, fn in enumerate(strategy_fns)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "episode_id": ep_ids[idx],
                    "score": 0.0,
                    "final_metrics": {},
                    "fold_history": [],
                    "status": "error",
                    "error": str(exc),
                }

    return results


def _obs_to_dict(obs) -> dict:
    """Convert OrigamiObservation to a JSON-serializable dict."""
    try:
        return obs.model_dump()
    except AttributeError:
        return {
            "task": obs.task if hasattr(obs, "task") else {},
            "paper_state": obs.paper_state if hasattr(obs, "paper_state") else {},
            "metrics": obs.metrics if hasattr(obs, "metrics") else {},
            "fold_history": obs.fold_history if hasattr(obs, "fold_history") else [],
            "done": obs.done if hasattr(obs, "done") else False,
            "reward": obs.reward if hasattr(obs, "reward") else None,
            "error": obs.error if hasattr(obs, "error") else None,
        }
