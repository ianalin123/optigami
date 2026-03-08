"""
training/demo_llm.py — 8 rollouts using Claude as the zero-shot fold strategist.

Usage:
    cd /path/to/optigami
    ANTHROPIC_API_KEY=sk-... python -m training.demo_llm

Each of the 8 episodes calls Claude (claude-haiku-4-5) once per fold step.
Claude sees the current paper_state metrics and decides the next fold.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import anthropic
import uvicorn

from server.app import app, broadcast
from server.models import OrigamiAction
from server.origami_environment import OrigamiEnvironment
from server.tasks import get_task_by_name


TASK_NAME = "half_fold"
NUM_EPISODES = 8
MODEL = "claude-haiku-4-5-20251001"


# ── LLM strategy factory ───────────────────────────────────────────────────────

def make_llm_strategy(client: anthropic.Anthropic, task: dict, episode_num: int):
    """Return a strategy_fn for one episode. Each episode gets its own call history."""
    history: list[dict[str, Any]] = []

    def strategy(paper_state: dict) -> dict:
        fold_count = paper_state.get("fold_count", 0)
        compactness = paper_state.get("compactness", 0)
        bb = paper_state.get("bounding_box", [1, 1, 0])
        target_box = task.get("target_box", [1, 0.5, 0.02])
        max_folds = task.get("max_folds", 3)

        user_msg = f"""You are folding a {task['width']}x{task['height']} sheet of {task['material']}.
Task: {task['description']}
Target box to fit inside: {target_box}
Max folds allowed: {max_folds}

Current state (fold {fold_count}/{max_folds}):
  compactness: {compactness:.3f}  (1.0 = fully packed, 0.0 = flat)
  bounding_box: [{bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.4f}]
  fits_target_box: {paper_state.get('fits_target_box', False)}

Choose the next fold. Respond with ONLY valid JSON, no other text:
{{
  "type": "valley" or "mountain" or "stop",
  "line": {{"start": [x, y], "end": [x, y]}},
  "angle": 180
}}

Coordinates are normalized 0-1. Use "stop" if done."""

        history.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model=MODEL,
            max_tokens=120,
            messages=history,
        )
        reply = response.content[0].text.strip()
        history.append({"role": "assistant", "content": reply})

        # Extract JSON — handle markdown code blocks
        match = re.search(r'\{[^{}]+\}', reply, re.DOTALL)
        if not match:
            return {"type": "stop", "line": {"start": [0, 0.5], "end": [1, 0.5]}, "angle": 0.0}

        fold_dict = json.loads(match.group())
        # Normalize: ensure required keys
        fold_dict.setdefault("type", "valley")
        fold_dict.setdefault("line", {"start": [0.0, 0.5], "end": [1.0, 0.5]})
        fold_dict.setdefault("angle", 180.0)
        return fold_dict

    return strategy


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode_llm(
    strategy_fn,
    task_name: str,
    ep_id: str,
    broadcast_fn,
) -> dict:
    env = OrigamiEnvironment()
    obs = env.reset(task_name=task_name)
    task = env._task or {}

    broadcast_fn(ep_id, {
        "type": "episode_update",
        "episode_id": ep_id,
        "task_name": task_name,
        "step": 0,
        "observation": _obs_dict(obs),
    })

    max_steps = task.get("max_folds", 5)
    status = "done"

    for step_idx in range(max_steps):
        if obs.done:
            break

        # Build a flat paper_state dict for the LLM (add metrics inline)
        ps = dict(obs.paper_state)
        ps.update(obs.metrics)  # compactness, fits_target_box, etc.
        ps["fold_count"] = step_idx

        try:
            fold_dict = strategy_fn(ps)
        except Exception as exc:
            broadcast_fn(ep_id, {
                "type": "episode_done", "episode_id": ep_id,
                "status": "error", "score": 0.0,
                "final_metrics": obs.metrics, "error": str(exc),
            })
            return {"episode_id": ep_id, "score": 0.0, "status": "error"}

        if fold_dict.get("type") == "stop":
            break

        time.sleep(0.4)  # pace for viewer animation

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

    score = obs.reward if obs.reward is not None else (env._total_reward or 0.0)
    broadcast_fn(ep_id, {
        "type": "episode_done",
        "episode_id": ep_id,
        "status": status,
        "score": float(score),
        "final_metrics": obs.metrics,
    })

    return {"episode_id": ep_id, "score": float(score), "status": status}


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


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_demo() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)
    task = get_task_by_name(TASK_NAME)

    await asyncio.sleep(1.5)  # wait for server startup

    print(f"\n[llm-demo] Model: {MODEL}")
    print(f"[llm-demo] Task: {TASK_NAME} — {task['description']}")
    print(f"[llm-demo] Open: http://localhost:9001/viewer/training.html\n")

    await broadcast.start_batch(1, NUM_EPISODES)

    ep_ids = [f"ep_{i:02d}" for i in range(NUM_EPISODES)]
    strategies = [make_llm_strategy(client, task, i) for i in range(NUM_EPISODES)]

    # Run all episodes concurrently (each makes its own Claude API calls)
    results = await asyncio.gather(*[
        asyncio.to_thread(run_episode_llm, fn, TASK_NAME, ep_id, broadcast.publish)
        for fn, ep_id in zip(strategies, ep_ids)
    ])

    scores = [r["score"] for r in results]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    await broadcast.finish_batch(1, scores, best_episode_id=ep_ids[best_idx])

    print("\n[llm-demo] Results:")
    for i, result in enumerate(results):
        print(f"  ep_{i:02d}  score={result['score']:+.2f}  status={result['status']}")
    print(f"\n[llm-demo] Best: ep_{best_idx:02d} (score={scores[best_idx]:+.2f})")
    print("\n[llm-demo] Press Ctrl+C to stop.\n")


async def _main() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=9001, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), run_demo())


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[llm-demo] Stopped.")
