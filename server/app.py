"""
server/app.py — Training WebSocket server for Colab environment.

Provides /ws/training for live streaming of RL training episodes to browsers.
Mount at a publicly accessible URL in Colab (e.g., via ngrok or Colab's proxy).

Usage in training:
    from server.app import broadcast
    broadcast.publish(episode_id, {"type": "episode_update", ...})
"""
from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.training_broadcast import TrainingBroadcastServer

app = FastAPI(title="Optigami Training Server", version="1.0")

# Allow cross-origin connections (Colab public URL → browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global broadcast server — import and use from training code
broadcast = TrainingBroadcastServer()


@app.on_event("startup")
async def _store_loop() -> None:
    """Capture the asyncio event loop so training threads can schedule coroutines."""
    import asyncio
    broadcast._loop = asyncio.get_running_loop()


@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket) -> None:
    """Spectator WebSocket endpoint. Viewers connect here to watch training."""
    await broadcast.connect_spectator(websocket)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "spectators": broadcast.spectator_count,
        "active_episodes": broadcast.active_episodes,
    }


# ── Demo endpoints (same as openenv_server/app.py so the React UI works) ──

@app.get("/targets")
def get_targets() -> dict:
    from server.tasks import available_task_names, get_task_by_name
    return {
        name: {
            "name": name,
            "level": t["difficulty"],
            "description": t.get("description", ""),
            "n_creases": t.get("max_folds", 3),
            "difficulty": t["difficulty"],
            "material": t.get("material", "paper"),
        }
        for name in available_task_names()
        if (t := get_task_by_name(name))
    }


_DEMO_SEQUENCES: dict[str, list[dict]] = {
    "half_fold":    [{"type": "valley",   "line": {"start": [0.0, 0.5],   "end": [1.0, 0.5]},   "angle": 180.0}],
    "quarter_fold": [{"type": "valley",   "line": {"start": [0.0, 0.5],   "end": [1.0, 0.5]},   "angle": 180.0},
                     {"type": "valley",   "line": {"start": [0.5, 0.0],   "end": [0.5, 1.0]},   "angle": 180.0}],
    "letter_fold":  [{"type": "valley",   "line": {"start": [0.0, 0.333], "end": [1.0, 0.333]}, "angle": 180.0},
                     {"type": "mountain", "line": {"start": [0.0, 0.667], "end": [1.0, 0.667]}, "angle": 180.0}],
    "map_fold":     [{"type": "valley",   "line": {"start": [0.0, 0.5],   "end": [1.0, 0.5]},   "angle": 180.0},
                     {"type": "mountain", "line": {"start": [0.5, 0.0],   "end": [0.5, 1.0]},   "angle": 180.0}],
    "solar_panel":  [{"type": "valley",   "line": {"start": [0.0, 0.25],  "end": [1.0, 0.25]},  "angle": 180.0},
                     {"type": "mountain", "line": {"start": [0.0, 0.5],   "end": [1.0, 0.5]},   "angle": 180.0},
                     {"type": "valley",   "line": {"start": [0.0, 0.75],  "end": [1.0, 0.75]},  "angle": 180.0}],
}


@app.get("/episode/demo")
def demo_episode(target: str = "half_fold") -> dict:
    from server.origami_environment import OrigamiEnvironment
    from server.models import OrigamiAction as NewAction
    from server.tasks import get_task_by_name

    folds = _DEMO_SEQUENCES.get(target, _DEMO_SEQUENCES["half_fold"])
    env = OrigamiEnvironment()
    obs = env.reset(task_name=target)
    steps: list[dict] = []

    for i, fold_dict in enumerate(folds):
        action = NewAction(
            fold_type=fold_dict["type"],
            fold_line=fold_dict["line"],
            fold_angle=float(fold_dict.get("angle", 180.0)),
        )
        obs = env.step(action)
        steps.append({"step": i + 1, "fold": fold_dict,
                       "paper_state": obs.paper_state, "metrics": obs.metrics,
                       "done": obs.done})
        if obs.done:
            break

    return {"task_name": target, "task": get_task_by_name(target) or {},
            "steps": steps, "final_metrics": obs.metrics if steps else {}}


@app.get("/episode/replay/{ep_id}")
def replay_episode(ep_id: str) -> dict:
    """Return a stored training episode in the same format as /episode/demo."""
    from server.tasks import get_task_by_name
    ep = broadcast._registry.get(ep_id)
    if not ep:
        raise HTTPException(status_code=404, detail=f"Episode '{ep_id}' not found in registry")
    return {
        "task_name": ep.task_name,
        "task": get_task_by_name(ep.task_name) or {},
        "steps": ep.steps,
        "final_metrics": ep.final_metrics or (ep.steps[-1]["metrics"] if ep.steps else {}),
    }


# ── Static files — viewer first, then React app (LAST, catch-all) ──

_VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
_BUILD_DIR  = Path(__file__).resolve().parent.parent / "build"

if _VIEWER_DIR.exists():
    app.mount("/viewer", StaticFiles(directory=str(_VIEWER_DIR), html=True), name="viewer")


if _BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_BUILD_DIR), html=True), name="react")
else:
    @app.get("/", include_in_schema=False)
    def _no_build() -> HTMLResponse:
        return HTMLResponse(
            "<p>React build not found. Run <code>npm run build</code> in the frontend directory.</p>"
            "<p>Training viewer: <a href='/viewer/training.html'>/viewer/training.html</a></p>"
        )


def run(host: str = "0.0.0.0", port: int = 9001) -> None:
    """Start the training server. Call from Colab notebook."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
