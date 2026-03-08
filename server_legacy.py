"""
FastAPI server for the origami RL environment.
Serves episode data to the React frontend.

Usage: uvicorn server:app --reload --port 8000
"""

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("Run: pip install fastapi uvicorn pydantic")
    raise

from typing import Optional


app = FastAPI(title="OrigamiRL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost:3000 for React dev
    allow_methods=["*"],
    allow_headers=["*"],
)


class FoldAction(BaseModel):
    from_point: list[float]  # [x, y]
    to_point: list[float]    # [x, y]
    assignment: str          # 'M' or 'V'
    instruction: str = ""


class EpisodeStep(BaseModel):
    step: int
    fold: Optional[FoldAction]
    paper_state: dict        # FOLD JSON of current crease graph
    anchor_points: list[list[float]]
    reward: dict
    done: bool
    info: dict
    prompt: str              # LLM prompt at this step


class EpisodeResult(BaseModel):
    target_name: str
    target: dict             # FOLD JSON of target
    steps: list[EpisodeStep]
    final_reward: dict


@app.get("/")
def health_check():
    """Health check — returns status and available target names."""
    from env.environment import OrigamiEnvironment
    env = OrigamiEnvironment()
    return {"status": "ok", "targets": env.available_targets()}


@app.get("/targets")
def get_targets():
    """Return list of available target names and their metadata."""
    from env.environment import OrigamiEnvironment
    env = OrigamiEnvironment()
    targets = {}
    for name in env.available_targets():
        t = env._targets[name]
        targets[name] = {
            "name": name,
            "level": t.get("level", 1),
            "description": t.get("description", ""),
            "n_creases": sum(1 for a in t["edges_assignment"] if a in ("M", "V")),
        }
    return targets


@app.get("/episode/run")
def run_episode(target: str = "half_horizontal", completion: str = ""):
    """
    Run a code-as-policy episode with a provided completion string.

    If completion is empty, returns the prompt so the caller knows what to send.
    Returns full episode result with all steps.
    """
    from env.environment import OrigamiEnvironment
    from env.prompts import parse_fold_list, code_as_policy_prompt
    from env.rewards import compute_reward, target_crease_edges

    env = OrigamiEnvironment(mode="step")
    obs = env.reset(target_name=target)

    if not completion:
        return {"prompt": obs["prompt"], "steps": [], "target": env.target}

    try:
        folds = parse_fold_list(completion)
    except ValueError as e:
        return {"error": str(e), "steps": []}

    steps = []
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

        # Build per-step prompt reflecting current state
        from env.prompts import step_level_prompt
        step_prompt = step_level_prompt(
            target=env.target,
            paper_state=env.paper,
            step=i + 1,
            max_steps=env.max_steps,
            last_reward=reward,
        )

        steps.append({
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
        })

        if reward.get("completion", 0) > 0:
            break

    return {
        "target_name": target,
        "target": env.target,
        "steps": steps,
        "final_reward": steps[-1]["reward"] if steps else {},
    }


@app.get("/episode/demo")
def demo_episode(target: str = "half_horizontal"):
    """Return a pre-solved demo episode for each target."""
    DEMO_COMPLETIONS = {
        "half_horizontal": '<folds>[{"instruction": "Valley fold along horizontal center line", "from": [0, 0.5], "to": [1, 0.5], "assignment": "V"}]</folds>',
        "half_vertical": '<folds>[{"instruction": "Mountain fold along vertical center line", "from": [0.5, 0], "to": [0.5, 1], "assignment": "M"}]</folds>',
        "diagonal_main": '<folds>[{"instruction": "Valley fold along main diagonal", "from": [0, 0], "to": [1, 1], "assignment": "V"}]</folds>',
        "diagonal_anti": '<folds>[{"instruction": "Mountain fold along anti-diagonal", "from": [1, 0], "to": [0, 1], "assignment": "M"}]</folds>',
        "thirds_h": '<folds>[{"instruction": "Valley fold at one-third height", "from": [0, 0.333], "to": [1, 0.333], "assignment": "V"}, {"instruction": "Valley fold at two-thirds height", "from": [0, 0.667], "to": [1, 0.667], "assignment": "V"}]</folds>',
        "thirds_v": '<folds>[{"instruction": "Mountain fold at one-third width", "from": [0.333, 0], "to": [0.333, 1], "assignment": "M"}, {"instruction": "Mountain fold at two-thirds width", "from": [0.667, 0], "to": [0.667, 1], "assignment": "M"}]</folds>',
        "accordion_3h": '<folds>[{"instruction": "Valley fold at quarter height", "from": [0, 0.25], "to": [1, 0.25], "assignment": "V"}, {"instruction": "Mountain fold at half height", "from": [0, 0.5], "to": [1, 0.5], "assignment": "M"}, {"instruction": "Valley fold at three-quarter height", "from": [0, 0.75], "to": [1, 0.75], "assignment": "V"}]</folds>',
        "accordion_4h": '<folds>[{"instruction": "Valley fold at 0.2", "from": [0, 0.2], "to": [1, 0.2], "assignment": "V"}, {"instruction": "Mountain fold at 0.4", "from": [0, 0.4], "to": [1, 0.4], "assignment": "M"}, {"instruction": "Valley fold at 0.6", "from": [0, 0.6], "to": [1, 0.6], "assignment": "V"}, {"instruction": "Mountain fold at 0.8", "from": [0, 0.8], "to": [1, 0.8], "assignment": "M"}]</folds>',
    }
    completion = DEMO_COMPLETIONS.get(target, DEMO_COMPLETIONS["half_horizontal"])
    return run_episode(target=target, completion=completion)
