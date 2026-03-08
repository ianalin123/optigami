"""
training/demo_llm.py — 8 rollouts using Claude as the zero-shot fold strategist.

Usage:
    cd /path/to/optigami
    ANTHROPIC_API_KEY=sk-... python -m training.demo_llm

Each of the 8 episodes calls Claude (claude-haiku-4-5) once per fold step.
Claude receives the current paper state (metrics + fold history) and decides
the next fold action.  Episodes run concurrently; all stream to the grid viewer.
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


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an origami folding agent controlling a robotic paper-folding system.

COORDINATE SYSTEM
- Paper starts as a flat sheet; coordinates are normalized to the sheet's original size.
- x=0 is the left edge, x=1 is the right edge.
- y=0 is the bottom edge, y=1 is the top edge.
- Fold line endpoints must be on or outside the paper boundary (0.0–1.0 range).
- A fold line that runs off the edge is fine — it just doesn't affect paper outside the sheet.

FOLD TYPES
- "valley": folds the paper toward you (creates a V crease when viewed from above).
- "mountain": folds the paper away from you (creates a ^ crease).
- "stop": you are satisfied — no more folds needed.

PHYSICS
- angle=180 means a fully flat fold (paper halved).
- Smaller angles (e.g. 90) create partial folds.
- Each fold updates compactness, bounding_box, and strain readings.
- Kawasaki/Maekawa violations indicate geometrically invalid crease patterns.

RESPONSE FORMAT — output ONLY valid JSON, no markdown, no explanation:
{"type": "valley", "line": {"start": [x, y], "end": [x, y]}, "angle": 180}
"""

# Eight distinct approach hints — gives diversity across the parallel episodes.
APPROACH_HINTS = [
    "Try a single clean horizontal fold at the exact midline.",
    "Try a single clean vertical fold at the exact midline.",
    "Use two folds: first horizontal then vertical, to create quarters.",
    "Use a diagonal fold from one corner to the opposite.",
    "Try folding at y=0.333 and y=0.667 to create thirds.",
    "Try a single fold but vary the position slightly off-center to explore.",
    "Use a mountain fold instead of valley for the primary crease.",
    "Try to reach the target box in as few folds as possible — stop early if done.",
]


# ── LLM strategy factory ───────────────────────────────────────────────────────

def make_llm_strategy(client: anthropic.Anthropic, task: dict, episode_num: int):
    """Return a strategy_fn for one episode.

    Each episode has its own conversation history (multi-turn) and a unique
    approach hint so the 8 concurrent episodes explore different strategies.
    """
    history: list[dict[str, Any]] = []
    hint = APPROACH_HINTS[episode_num % len(APPROACH_HINTS)]
    prev_compactness: list[float] = [0.0]  # mutable cell for delta tracking

    def strategy(paper_state: dict, fold_history: list[dict]) -> dict:
        fold_count = paper_state.get("fold_count", 0)
        compactness = float(paper_state.get("compactness", 0))
        bb = paper_state.get("bounding_box", [1, 1, 0])
        fits = paper_state.get("fits_target_box", False)
        strain = paper_state.get("max_strain", 0.0)
        kaw = paper_state.get("kawasaki_violations", 0)
        target_box = task.get("target_box", [1, 0.5, 0.02])
        max_folds = task.get("max_folds", 3)

        delta = compactness - prev_compactness[0]
        prev_compactness[0] = compactness

        # Summarise what has been done so far
        history_lines = ""
        if fold_history:
            history_lines = "Folds applied so far:\n"
            for i, f in enumerate(fold_history, 1):
                t = f.get("type", "?")
                ln = f.get("line", {})
                s = ln.get("start", [0, 0])
                e = ln.get("end", [1, 1])
                ang = f.get("angle", 180)
                history_lines += (
                    f"  {i}. {t} fold  "
                    f"from ({s[0]:.3f},{s[1]:.3f}) to ({e[0]:.3f},{e[1]:.3f})  "
                    f"angle={ang}\n"
                )
        else:
            history_lines = "No folds applied yet — paper is flat.\n"

        sign = "+" if delta >= 0 else ""
        user_msg = (
            f"Task: {task['description']}\n"
            f"Sheet: {task['width']}×{task['height']} {task['material']}\n"
            f"Target bounding box: {target_box}  (must fit inside to succeed)\n"
            f"Max folds remaining: {max_folds - fold_count}\n"
            f"\n"
            f"{history_lines}"
            f"\n"
            f"Current state after fold {fold_count}/{max_folds}:\n"
            f"  compactness : {compactness:.4f}  (Δ {sign}{delta:.4f})\n"
            f"  bounding_box: [{bb[0]:.4f}, {bb[1]:.4f}, {bb[2]:.5f}]\n"
            f"  fits_target : {'YES ✓' if fits else 'no'}\n"
            f"  max_strain  : {strain:.5f}\n"
            f"  kaw_violations: {kaw}\n"
            f"\n"
            f"Approach hint: {hint}\n"
            f"\n"
            f"What is your next fold action?  "
            f"Return \"stop\" if the target is already achieved or no useful fold remains."
        )

        history.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model=MODEL,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        reply = response.content[0].text.strip()
        history.append({"role": "assistant", "content": reply})

        # Handle explicit "stop" text before JSON parse
        if reply.lower().startswith("stop") or '"type": "stop"' in reply:
            return {"type": "stop", "line": {"start": [0, 0.5], "end": [1, 0.5]}, "angle": 0.0}

        # Extract JSON — handles markdown code fences
        match = re.search(r'\{[^{}]+\}', reply, re.DOTALL)
        if not match:
            # Malformed response — default safe fold then stop next turn
            return {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0}

        try:
            fold_dict = json.loads(match.group())
        except json.JSONDecodeError:
            return {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0}

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

        # Merge paper_state + metrics for the strategy
        ps = dict(obs.paper_state)
        ps.update(obs.metrics)
        ps["fold_count"] = step_idx

        try:
            fold_dict = strategy_fn(ps, list(obs.fold_history))
        except Exception as exc:
            broadcast_fn(ep_id, {
                "type": "episode_done", "episode_id": ep_id,
                "status": "error", "score": 0.0,
                "final_metrics": obs.metrics, "error": str(exc),
            })
            return {"episode_id": ep_id, "score": 0.0, "status": "error"}

        if fold_dict.get("type") == "stop":
            break

        time.sleep(0.5)  # pace for viewer animation

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

    print(f"\n[llm-demo] Model  : {MODEL}")
    print(f"[llm-demo] Task   : {TASK_NAME} — {task['description']}")
    print(f"[llm-demo] Open   : http://localhost:9001/viewer/training.html\n")
    print(f"[llm-demo] Episodes:")
    for i, hint in enumerate(APPROACH_HINTS):
        print(f"  ep_{i:02d}  hint: {hint}")
    print()

    await broadcast.start_batch(1, NUM_EPISODES)

    ep_ids = [f"ep_{i:02d}" for i in range(NUM_EPISODES)]
    strategies = [make_llm_strategy(client, task, i) for i in range(NUM_EPISODES)]

    results = await asyncio.gather(*[
        asyncio.to_thread(run_episode_llm, fn, TASK_NAME, ep_id, broadcast.publish)
        for fn, ep_id in zip(strategies, ep_ids)
    ])

    scores = [r["score"] for r in results]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    await broadcast.finish_batch(1, scores, best_episode_id=ep_ids[best_idx])

    print("\n[llm-demo] Results:")
    for i, (result, hint) in enumerate(zip(results, APPROACH_HINTS)):
        marker = " ← best" if i == best_idx else ""
        print(f"  ep_{i:02d}  score={result['score']:+.2f}  status={result['status']}  hint: {hint}{marker}")
    print(f"\n[llm-demo] Press Ctrl+C to stop.\n")


async def _main() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=9001, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), run_demo())


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[llm-demo] Stopped.")
