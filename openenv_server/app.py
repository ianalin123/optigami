from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from openenv_runtime.environment import OpenEnvOrigamiEnvironment
from openenv_runtime.models import OrigamiAction, OrigamiObservation


app = create_app(
    env=lambda: OpenEnvOrigamiEnvironment(),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="optigami",
)


# ---------------------------------------------------------------------------
# Demo fold sequences — new format: type, line {start, end}, angle
# ---------------------------------------------------------------------------

DEMO_SEQUENCES: dict[str, list[dict]] = {
    "half_fold": [
        {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0},
    ],
    "quarter_fold": [
        {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0},
        {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0},
    ],
    "letter_fold": [
        {"type": "valley", "line": {"start": [0.0, 0.333], "end": [1.0, 0.333]}, "angle": 180.0},
        {"type": "mountain", "line": {"start": [0.0, 0.667], "end": [1.0, 0.667]}, "angle": 180.0},
    ],
    "map_fold": [
        {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0},
        {"type": "mountain", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180.0},
    ],
    "solar_panel": [
        {"type": "valley", "line": {"start": [0.0, 0.25], "end": [1.0, 0.25]}, "angle": 180.0},
        {"type": "mountain", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180.0},
        {"type": "valley", "line": {"start": [0.0, 0.75], "end": [1.0, 0.75]}, "angle": 180.0},
    ],
    "shelter_wall": [
        {"type": "valley", "line": {"start": [0.0, 0.333], "end": [1.0, 0.333]}, "angle": 180.0},
        {"type": "valley", "line": {"start": [0.0, 0.667], "end": [1.0, 0.667]}, "angle": 180.0},
    ],
    "stent": [
        {"type": "valley", "line": {"start": [0.0, 0.25], "end": [1.0, 0.25]}, "angle": 90.0},
        {"type": "mountain", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 90.0},
        {"type": "valley", "line": {"start": [0.0, 0.75], "end": [1.0, 0.75]}, "angle": 90.0},
        {"type": "stop", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 0.0},
    ],
}


# ---------------------------------------------------------------------------
# API routes — must be registered BEFORE the StaticFiles catch-all mount
# ---------------------------------------------------------------------------

@app.get("/targets", include_in_schema=True)
def get_targets() -> dict:
    """Return available task names and metadata for the frontend."""
    from server.tasks import get_task_by_name, available_task_names

    result: dict[str, dict] = {}
    for name in available_task_names():
        t = get_task_by_name(name)
        result[name] = {
            "name": name,
            "level": t.get("difficulty", 1),
            "description": t.get("description", ""),
            "n_creases": t.get("max_folds", 3),
            "difficulty": t.get("difficulty", 1),
            "material": t.get("material", "paper"),
        }
    return result


@app.get("/episode/demo", include_in_schema=True)
def demo_episode(target: str = "half_fold") -> dict:
    """Return a pre-solved demo episode for the given task."""
    from server.origami_environment import OrigamiEnvironment
    from server.models import OrigamiAction as NewOrigamiAction
    from server.tasks import get_task_by_name

    # Fall back to half_fold if target not found
    folds = DEMO_SEQUENCES.get(target, DEMO_SEQUENCES["half_fold"])

    env = OrigamiEnvironment()
    obs = env.reset(task_name=target)

    steps: list[dict] = []

    for i, fold_dict in enumerate(folds):
        if fold_dict.get("type") == "stop":
            break

        action = NewOrigamiAction(
            fold_type=fold_dict["type"],
            fold_line=fold_dict["line"],
            fold_angle=float(fold_dict.get("angle", 180.0)),
        )

        obs = env.step(action)

        steps.append({
            "step": i + 1,
            "fold": fold_dict,
            "paper_state": obs.paper_state,
            "metrics": obs.metrics,
            "done": obs.done,
        })

        if obs.done:
            break

    task_def = get_task_by_name(target) if target else {}

    return {
        "task_name": target,
        "task": task_def,
        "steps": steps,
        "final_metrics": obs.metrics if steps else {},
    }


# ---------------------------------------------------------------------------
# Static file serving — must come LAST so API routes take priority
# ---------------------------------------------------------------------------

_BUILD_DIR = Path(__file__).resolve().parent.parent / "build"

if _BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_BUILD_DIR), html=True), name="renderer")
else:
    @app.get("/", include_in_schema=False)
    def missing_renderer_build() -> HTMLResponse:
        return HTMLResponse(
            """
            <html><body style="font-family: sans-serif; margin: 24px;">
            <h3>Renderer build not found</h3>
            <p>No <code>build/</code> directory is present in the container.</p>
            <p>OpenEnv API docs are available at <a href="/docs">/docs</a>.</p>
            </body></html>
            """,
            status_code=200,
        )
