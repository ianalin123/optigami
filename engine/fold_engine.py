"""
Fold execution engine.

Applies fold operations (valley / mountain) to a Paper object using
Rodrigues' rotation formula, face splitting, and layer tracking.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .paper import Paper


# ────────────────────────────────────────────────────────────────────
# Rodrigues' rotation
# ────────────────────────────────────────────────────────────────────

def _rodrigues_rotate(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate *points* (N, 3) around an axis defined by a point and direction
    using Rodrigues' rotation formula.  Returns rotated points (N, 3)."""
    k = axis_dir / (np.linalg.norm(axis_dir) + 1e-30)
    translated = points - axis_point
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dot_term = np.dot(translated, k).reshape(-1, 1) * k  # (N,1)*(3,) broadcast
    rotated = (
        translated * cos_a
        + np.cross(k, translated) * sin_a
        + dot_term * (1.0 - cos_a)
    )
    return rotated + axis_point


# ────────────────────────────────────────────────────────────────────
# Single fold
# ────────────────────────────────────────────────────────────────────

def apply_fold(
    paper: Paper,
    fold_dict: dict,
) -> tuple[Paper, str | None]:
    """Apply a single fold to *paper* and return ``(new_paper, error_or_None)``.

    *fold_dict* has the form::

        {
            "type": "valley" | "mountain",
            "line": {"start": [x, y], "end": [x, y]},
            "angle": 0-180,
        }

    Steps:
      1. Validate inputs.
      2. Split faces along the fold line.
      3. Determine vertices to rotate (one side of fold line).
      4. Rodrigues' rotation of those vertices.
      5. Update edge assignments for new fold-line edges.
      6. Update fold angles.
      7. Update layer tracking.
    """

    # ── 0. parse & validate ─────────────────────────────────────────
    fold_type = fold_dict.get("type", "valley")
    line = fold_dict.get("line", {})
    angle_deg = fold_dict.get("angle", 180)

    if fold_type not in ("valley", "mountain"):
        return paper, f"Unknown fold type: {fold_type}"

    try:
        start_2d = np.array(line["start"], dtype=np.float64)[:2]
        end_2d = np.array(line["end"], dtype=np.float64)[:2]
    except (KeyError, TypeError, IndexError) as exc:
        return paper, f"Invalid fold line: {exc}"

    if np.linalg.norm(end_2d - start_2d) < 1e-12:
        return paper, "Fold line has zero length"

    if not (0 < angle_deg <= 180):
        return paper, f"Angle must be in (0, 180], got {angle_deg}"

    # ── 1. deep copy so the original is untouched ───────────────────
    new_paper = paper.copy()

    # ── 2. split faces along fold line ──────────────────────────────
    try:
        fold_edge_ids = new_paper.split_faces_along_line(start_2d, end_2d)
    except Exception as exc:
        return paper, f"Face split failed: {exc}"

    # ── 3. determine vertices to rotate ─────────────────────────────
    rotate_ids = new_paper.get_vertices_on_side(start_2d, end_2d, "positive")

    if not rotate_ids:
        # Try the other side — maybe the fold line is at the boundary
        rotate_ids = new_paper.get_vertices_on_side(start_2d, end_2d, "negative")
        if not rotate_ids:
            return paper, "No vertices to rotate — fold line may not intersect paper"

    # ── 4. Rodrigues' rotation ──────────────────────────────────────
    sign = 1.0 if fold_type == "valley" else -1.0
    angle_rad = sign * math.radians(angle_deg)

    axis_point = np.array([start_2d[0], start_2d[1], 0.0])
    axis_dir = np.array([end_2d[0] - start_2d[0], end_2d[1] - start_2d[1], 0.0])

    pts = new_paper.vertices[rotate_ids]
    rotated = _rodrigues_rotate(pts, axis_point, axis_dir, angle_rad)
    new_paper.vertices[rotate_ids] = rotated

    # ── 5. update edge assignments ──────────────────────────────────
    assignment = "V" if fold_type == "valley" else "M"
    for eidx in fold_edge_ids:
        if eidx < len(new_paper.assignments):
            new_paper.assignments[eidx] = assignment

    # ── 6. update fold angles ───────────────────────────────────────
    for eidx in fold_edge_ids:
        if eidx < len(new_paper.fold_angles):
            new_paper.fold_angles[eidx] = angle_deg * sign

    # ── 7. update layer tracking ────────────────────────────────────
    # For each pair of faces on opposite sides of the fold line, record
    # layer ordering.  Simple heuristic: faces that were rotated are now
    # on top (sign +1) of faces that stayed put.
    rotated_set = set(rotate_ids)

    def _face_side(face_verts: list[int]) -> str:
        """Classify a face as 'rotated', 'fixed', or 'mixed'."""
        r_count = sum(1 for v in face_verts if v in rotated_set)
        if r_count == len(face_verts):
            return "rotated"
        if r_count == 0:
            return "fixed"
        return "mixed"

    face_sides = [_face_side(f) for f in new_paper.faces]
    for i in range(len(new_paper.faces)):
        for j in range(i + 1, len(new_paper.faces)):
            if face_sides[i] == "rotated" and face_sides[j] == "fixed":
                new_paper.face_orders.append((i, j, 1))
            elif face_sides[i] == "fixed" and face_sides[j] == "rotated":
                new_paper.face_orders.append((j, i, 1))

    new_paper.fold_count += 1

    return new_paper, None


# ────────────────────────────────────────────────────────────────────
# Strategy executor (matches mock_env.execute_fold_strategy signature)
# ────────────────────────────────────────────────────────────────────

def execute_fold_strategy(
    strategy_fn: Callable,
    paper: Paper,
    max_folds: int = 20,
) -> tuple[Paper, list[dict], str | None]:
    """Execute a ``fold_strategy`` function against the real physics engine.

    Signature matches ``mock_env.execute_fold_strategy`` so the trainer
    reward functions can swap engines transparently.

    Parameters
    ----------
    strategy_fn : callable
        ``strategy_fn(paper_state_dict) -> list[dict]``
    paper : Paper
        The initial paper state.
    max_folds : int
        Maximum number of folds to apply.

    Returns
    -------
    (final_paper, applied_folds, error_or_None)
    """
    state_dict = paper.to_dict()
    try:
        folds = strategy_fn(state_dict)
    except Exception as exc:
        return paper, [], f"Strategy function raised: {exc}"

    if not isinstance(folds, list):
        return paper, [], "Strategy must return a list of fold dicts"

    applied: list[dict] = []
    current = paper

    for i, fold in enumerate(folds):
        if i >= max_folds:
            break
        if not isinstance(fold, dict):
            return current, applied, f"Fold {i} is not a dict"

        current, error = apply_fold(current, fold)
        if error:
            return current, applied, f"Fold {i} failed: {error}"
        applied.append(fold)

    return current, applied, None


def apply_pleat(
    paper: Paper,
    line1: dict,
    line2: dict,
    angle: float = 180.0,
) -> tuple[Paper, str | None]:
    """Pleat fold: valley at line1, mountain at line2 (two parallel folds).

    Both line dicts have the form: {"start": [x, y], "end": [x, y]}
    Returns (new_paper, error_or_None).
    """
    paper, err = apply_fold(paper, {"type": "valley", "line": line1, "angle": angle})
    if err:
        return paper, f"Pleat valley fold failed: {err}"
    paper, err = apply_fold(paper, {"type": "mountain", "line": line2, "angle": angle})
    if err:
        return paper, f"Pleat mountain fold failed: {err}"
    return paper, None


def apply_crimp(
    paper: Paper,
    line1: dict,
    line2: dict,
    angle: float = 180.0,
) -> tuple[Paper, str | None]:
    """Crimp fold: mountain at line1, valley at line2 (reverse of pleat).

    Both line dicts have the form: {"start": [x, y], "end": [x, y]}
    Returns (new_paper, error_or_None).
    """
    paper, err = apply_fold(paper, {"type": "mountain", "line": line1, "angle": angle})
    if err:
        return paper, f"Crimp mountain fold failed: {err}"
    paper, err = apply_fold(paper, {"type": "valley", "line": line2, "angle": angle})
    if err:
        return paper, f"Crimp valley fold failed: {err}"
    return paper, None
