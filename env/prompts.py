import json
import re
from typing import Optional

_CORNERS = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
_BOUNDARY_X = {0.0, 1.0}
_BOUNDARY_Y = {0.0, 1.0}


def _is_corner(x: float, y: float) -> bool:
    return (round(x, 4), round(y, 4)) in _CORNERS


def _is_boundary(x: float, y: float) -> bool:
    return x in _BOUNDARY_X or y in _BOUNDARY_Y


def format_target_for_prompt(target: dict) -> str:
    vertices = target["vertices_coords"]
    edges_v = target["edges_vertices"]
    edges_a = target["edges_assignment"]

    lines = []
    for (v1, v2), assignment in zip(edges_v, edges_a):
        if assignment not in ("M", "V"):
            continue
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]
        label = "Mountain" if assignment == "M" else "Valley"
        lines.append(
            f"{label} fold: ({round(x1, 4)}, {round(y1, 4)}) -> ({round(x2, 4)}, {round(y2, 4)})"
        )
    return "\n".join(lines)


def format_anchor_points(paper_state) -> str:
    corners = []
    boundary_pts = []
    intersections = []

    for x, y in paper_state.anchor_points():
        rx, ry = round(x, 4), round(y, 4)
        if _is_corner(rx, ry):
            corners.append((rx, ry))
        elif _is_boundary(rx, ry):
            boundary_pts.append((rx, ry))
        else:
            intersections.append((rx, ry))

    def fmt_pts(pts: list[tuple[float, float]]) -> str:
        return "  ".join(f"({x},{y})" for x, y in pts)

    lines = []
    if corners:
        lines.append(f"  Corners:       {fmt_pts(corners)}")
    if boundary_pts:
        lines.append(f"  Boundary pts:  {fmt_pts(boundary_pts)}")
    if intersections:
        lines.append(f"  Intersections: {fmt_pts(intersections)}")

    return "\n".join(lines)


def format_crease_history(paper_state) -> str:
    history = paper_state.fold_history
    if not history:
        return "none"

    lines = []
    for i, fold in enumerate(history, 1):
        p1, p2 = fold["p1"], fold["p2"]
        assignment = fold["assignment"]
        label = "Mountain" if assignment == "M" else "Valley"
        x1, y1 = round(p1[0], 4), round(p1[1], 4)
        x2, y2 = round(p2[0], 4), round(p2[1], 4)
        lines.append(f"  {i}. {label} fold: ({x1}, {y1}) -> ({x2}, {y2})")

    return "\n".join(lines)


def format_reward_feedback(reward: Optional[dict]) -> str:
    if not reward:
        return "(no feedback yet)"

    keys = ["kawasaki", "maekawa", "blb", "progress", "economy", "total"]
    parts = []
    for k in keys:
        if k in reward:
            parts.append(f"{k}={reward[k]:.2f}")

    for k, v in reward.items():
        if k not in keys:
            parts.append(f"{k}={v:.2f}")

    return "  " + "  ".join(parts)


def code_as_policy_prompt(target: dict, max_folds: int = 8) -> str:
    formatted_target = format_target_for_prompt(target)
    return f"""You are an origami designer. Generate a fold sequence for a unit square [0,1]x[0,1].

TARGET CREASE PATTERN:
{formatted_target}

RULES (must hold at every interior vertex):
  - Kawasaki: alternating sector angles sum equally (each half = 180 degrees)
  - Maekawa: |mountain_count - valley_count| = 2
  - Big-Little-Big: folds bounding the smallest sector must have opposite types (one M, one V)

INITIAL ANCHOR POINTS (valid fold endpoints — new ones appear when creases intersect):
  Corners:      (0.0,0.0)  (1.0,0.0)  (1.0,1.0)  (0.0,1.0)
  Midpoints:    (0.0,0.5)  (0.5,0.0)  (1.0,0.5)  (0.5,1.0)
  Note: new anchor points are created at crease intersections.

Output at most {max_folds} folds. Both endpoints must be valid anchor points.
Output ONLY the JSON list, wrapped in <folds> tags:

<folds>
[
  {{"instruction": "Describe the fold in plain English", "from": [x1, y1], "to": [x2, y2], "assignment": "V"}},
  {{"instruction": "...", "from": [x1, y1], "to": [x2, y2], "assignment": "M"}}
]
</folds>"""


def step_level_prompt(
    target: dict,
    paper_state,
    step: int,
    max_steps: int,
    last_reward: Optional[dict] = None,
) -> str:
    formatted_target = format_target_for_prompt(target)
    formatted_history = format_crease_history(paper_state)
    formatted_anchors = format_anchor_points(paper_state)
    formatted_reward = format_reward_feedback(last_reward)

    return f"""You are an origami designer building a crease pattern step by step.

TARGET:
{formatted_target}

CURRENT STATE (step {step} of {max_steps}):
  Creases placed:
{formatted_history}

AVAILABLE ANCHOR POINTS:
{formatted_anchors}

LAST REWARD:
{formatted_reward}

Add the NEXT crease. Both endpoints must be listed anchor points above.
Output ONLY valid JSON (no extra text):
{{"instruction": "...", "from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}"""


def parse_fold_list(completion: str) -> list[dict]:
    match = re.search(r"<folds>(.*?)</folds>", completion, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("No <folds>...</folds> tags found in completion")

    raw = match.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON inside <folds> tags: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list inside <folds> tags, got {type(data).__name__}")

    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Fold {i} is not a dict: {item!r}")

        for field in ("from", "to", "assignment"):
            if field not in item:
                raise ValueError(f"Fold {i} missing required field '{field}'")

        from_pt = item["from"]
        to_pt = item["to"]

        if (
            not isinstance(from_pt, list)
            or len(from_pt) != 2
            or not all(isinstance(v, (int, float)) for v in from_pt)
        ):
            raise ValueError(f"Fold {i} 'from' must be a list of 2 numbers, got {from_pt!r}")

        if (
            not isinstance(to_pt, list)
            or len(to_pt) != 2
            or not all(isinstance(v, (int, float)) for v in to_pt)
        ):
            raise ValueError(f"Fold {i} 'to' must be a list of 2 numbers, got {to_pt!r}")

        if not isinstance(item["assignment"], str):
            raise ValueError(f"Fold {i} 'assignment' must be a string")

        cleaned.append(
            {
                "from": [float(from_pt[0]), float(from_pt[1])],
                "to": [float(to_pt[0]), float(to_pt[1])],
                "assignment": item["assignment"],
                "instruction": item.get("instruction", ""),
            }
        )

    return cleaned


def parse_single_fold(completion: str) -> dict:
    start = completion.find("{")
    end = completion.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in completion")

    raw = completion[start : end + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from completion: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    for field in ("from", "to", "assignment"):
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in fold JSON")

    return data
