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
# Demo routes required by the React frontend.
# These must be registered BEFORE the StaticFiles catch-all mount.
# ---------------------------------------------------------------------------

DEMO_COMPLETIONS: dict[str, str] = {
    "half_horizontal": '<folds>[{"instruction": "Valley fold along horizontal center line", "from": [0, 0.5], "to": [1, 0.5], "assignment": "V"}]</folds>',
    "half_vertical": '<folds>[{"instruction": "Mountain fold along vertical center line", "from": [0.5, 0], "to": [0.5, 1], "assignment": "M"}]</folds>',
    "diagonal_main": '<folds>[{"instruction": "Valley fold along main diagonal", "from": [0, 0], "to": [1, 1], "assignment": "V"}]</folds>',
    "diagonal_anti": '<folds>[{"instruction": "Mountain fold along anti-diagonal", "from": [1, 0], "to": [0, 1], "assignment": "M"}]</folds>',
    "thirds_h": '<folds>[{"instruction": "Valley fold at one-third height", "from": [0, 0.333], "to": [1, 0.333], "assignment": "V"}, {"instruction": "Valley fold at two-thirds height", "from": [0, 0.667], "to": [1, 0.667], "assignment": "V"}]</folds>',
    "thirds_v": '<folds>[{"instruction": "Mountain fold at one-third width", "from": [0.333, 0], "to": [0.333, 1], "assignment": "M"}, {"instruction": "Mountain fold at two-thirds width", "from": [0.667, 0], "to": [0.667, 1], "assignment": "M"}]</folds>',
    "accordion_3h": '<folds>[{"instruction": "Valley fold at quarter height", "from": [0, 0.25], "to": [1, 0.25], "assignment": "V"}, {"instruction": "Mountain fold at half height", "from": [0, 0.5], "to": [1, 0.5], "assignment": "M"}, {"instruction": "Valley fold at three-quarter height", "from": [0, 0.75], "to": [1, 0.75], "assignment": "V"}]</folds>',
    "accordion_4h": '<folds>[{"instruction": "Valley fold at 0.2", "from": [0, 0.2], "to": [1, 0.2], "assignment": "V"}, {"instruction": "Mountain fold at 0.4", "from": [0, 0.4], "to": [1, 0.4], "assignment": "M"}, {"instruction": "Valley fold at 0.6", "from": [0, 0.6], "to": [1, 0.6], "assignment": "V"}, {"instruction": "Mountain fold at 0.8", "from": [0, 0.8], "to": [1, 0.8], "assignment": "M"}]</folds>',
}


@app.get("/targets", include_in_schema=True)
def get_targets() -> dict:
    """Return available target names and metadata for the frontend."""
    from env.environment import OrigamiEnvironment

    env = OrigamiEnvironment()
    result: dict[str, dict] = {}
    for name in env.available_targets():
        t = env._targets[name]
        result[name] = {
            "name": name,
            "level": t.get("level", 1),
            "description": t.get("description", ""),
            "n_creases": sum(1 for a in t["edges_assignment"] if a in ("M", "V")),
        }
    return result


@app.get("/episode/run", include_in_schema=True)
def run_episode(target: str = "half_horizontal", completion: str = "") -> dict:
    """Run a fold-sequence episode and return step-by-step data."""
    from env.environment import OrigamiEnvironment
    from env.prompts import parse_fold_list, step_level_prompt
    from env.rewards import compute_reward

    env = OrigamiEnvironment(mode="step")
    obs = env.reset(target_name=target)

    if not completion:
        return {"prompt": obs["prompt"], "steps": [], "target": env.target}

    try:
        folds = parse_fold_list(completion)
    except ValueError as exc:
        return {"error": str(exc), "steps": []}

    steps: list[dict] = []
    for i, fold in enumerate(folds):
        result = env.paper.add_crease(fold["from"], fold["to"], fold["assignment"])
        reward = compute_reward(env.paper, result, env.target)

        paper_state = {
            "vertices": {str(k): list(v) for k, v in env.paper.graph.vertices.items()},
            "edges": [
                {
                    "id": k,
                    "v1": list(env.paper.graph.vertices[v[0]]),
                    "v2": list(env.paper.graph.vertices[v[1]]),
                    "assignment": v[2],
                }
                for k, v in env.paper.graph.edges.items()
            ],
            "anchor_points": [list(p) for p in env.paper.anchor_points()],
        }

        step_prompt = step_level_prompt(
            target=env.target,
            paper_state=env.paper,
            step=i + 1,
            max_steps=env.max_steps,
            last_reward=reward,
        )

        steps.append(
            {
                "step": i + 1,
                "fold": {
                    "from_point": fold["from"],
                    "to_point": fold["to"],
                    "assignment": fold["assignment"],
                    "instruction": fold.get("instruction", ""),
                },
                "paper_state": paper_state,
                "anchor_points": [list(p) for p in env.paper.anchor_points()],
                "reward": reward,
                "done": reward.get("completion", 0) > 0,
                "info": env._info(),
                "prompt": step_prompt,
            }
        )

        if reward.get("completion", 0) > 0:
            break

    return {
        "target_name": target,
        "target": env.target,
        "steps": steps,
        "final_reward": steps[-1]["reward"] if steps else {},
    }


@app.get("/episode/demo", include_in_schema=True)
def demo_episode(target: str = "half_horizontal") -> dict:
    """Return a pre-solved demo episode for the given target."""
    completion = DEMO_COMPLETIONS.get(target, DEMO_COMPLETIONS["half_horizontal"])
    return run_episode(target=target, completion=completion)


# ---------------------------------------------------------------------------
# Static file serving — must come LAST so API routes take priority.
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
