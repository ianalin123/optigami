"""
Task decomposer: breaks a parsed instruction into sequential sub-goals
with concrete fold operations on a unit square.
"""

from __future__ import annotations

import copy
from planner.knowledge import (
    ORIGAMI_MODELS,
    ORIGAMI_BASES,
    FOLD_OPERATIONS,
    get_model_steps,
    get_base_steps,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _step_to_fold_operation(step: dict) -> dict:
    """
    Convert a knowledge-base step dict into the engine's fold operation format:
      {"type": ..., "line": {"start": [...], "end": [...]}, "angle": ...}
    """
    op = {
        "type": step["type"],
        "line": copy.deepcopy(step["line"]),
        "angle": step.get("angle", 180),
    }
    if "layer_select" in step:
        op["layer_select"] = step["layer_select"]
    return op


def _expected_state_after_fold(fold_type: str, prev_state: dict | None) -> dict:
    """
    Produce a lightweight expected-state dict describing what the paper
    should look like after a fold.  This is intentionally approximate --
    the real simulation engine computes exact geometry.
    """
    state = dict(prev_state or {"layers": 1, "shape": "square", "phase": "flat"})
    if fold_type in ("valley", "mountain"):
        state["layers"] = state.get("layers", 1) * 2
    elif fold_type == "petal":
        state["shape"] = "narrow_diamond"
    elif fold_type == "squash":
        state["shape"] = "diamond"
    elif fold_type == "reverse_inside":
        state["shape"] = "pointed_flap_reversed"
    elif fold_type == "inflate":
        state["phase"] = "3d"
    elif fold_type == "turn_over":
        state["flipped"] = not state.get("flipped", False)
    elif fold_type == "unfold":
        # Layers don't literally halve on every unfold, but this is a hint
        state["layers"] = max(1, state.get("layers", 1) // 2)
    return state


def _validation_for_fold(fold_type: str) -> dict:
    """Return a simple validation dict for a step."""
    checks: dict = {"flat_foldable": True}
    if fold_type in ("valley", "mountain"):
        checks["kawasaki_check"] = True
        checks["maekawa_check"] = True
    if fold_type == "inflate":
        checks["is_3d"] = True
        checks["flat_foldable"] = False
    return checks


# ---------------------------------------------------------------------------
# Known-model decomposition
# ---------------------------------------------------------------------------

def _decompose_known_model(parsed: dict) -> list[dict]:
    """Decompose a known model into sub-goal steps."""
    model_name: str = parsed["model_name"]
    model_info = ORIGAMI_MODELS.get(model_name)
    if model_info is None:
        return _decompose_free_fold(parsed)

    base_name = model_info.get("base")
    steps = get_model_steps(model_name)
    sub_goals: list[dict] = []
    running_state: dict = {"layers": 1, "shape": "square", "phase": "flat"}

    for i, step in enumerate(steps):
        fold_op = _step_to_fold_operation(step)
        running_state = _expected_state_after_fold(step["type"], running_state)

        sub_goals.append({
            "step_number": i + 1,
            "description": step.get("description", f"Step {i + 1}"),
            "base_required": base_name if i == 0 else None,
            "fold_operations": [fold_op],
            "expected_state": dict(running_state),
            "validation": _validation_for_fold(step["type"]),
        })

    return sub_goals


# ---------------------------------------------------------------------------
# Packing / optimization decomposition
# ---------------------------------------------------------------------------

def _decompose_packing(parsed: dict) -> list[dict]:
    """
    Decompose an optimize_packing task into sub-goals.
    Returns a Miura-ori-style fold plan on a unit square.
    """
    w = parsed["dimensions"]["width"]
    h = parsed["dimensions"]["height"]
    material = parsed["material"]
    constraints = parsed.get("constraints", {})
    max_folds = constraints.get("max_folds", 20)

    sub_goals: list[dict] = []
    step_num = 0

    # Horizontal valley/mountain pleats (zigzag in Y)
    n_horizontal = min(4, max_folds // 4)
    spacing_y = 1.0 / (n_horizontal + 1)
    for i in range(n_horizontal):
        step_num += 1
        y = spacing_y * (i + 1)
        fold_type = "valley" if i % 2 == 0 else "mountain"
        sub_goals.append({
            "step_number": step_num,
            "description": f"Horizontal {fold_type} fold at y={y:.3f} (pleat {i + 1}/{n_horizontal})",
            "base_required": None,
            "fold_operations": [{
                "type": fold_type,
                "line": {"start": [0.0, y], "end": [1.0, y]},
                "angle": 180,
                "layer_select": "all",
            }],
            "expected_state": {"layers": i + 2, "phase": "flat", "pattern": "miura_horizontal"},
            "validation": {"flat_foldable": True, "kawasaki_check": True},
        })

    # Vertical zigzag valley/mountain pleats (Miura-ori angle offsets)
    n_vertical = min(4, (max_folds - n_horizontal) // 2)
    spacing_x = 1.0 / (n_vertical + 1)
    for i in range(n_vertical):
        step_num += 1
        x = spacing_x * (i + 1)
        fold_type = "valley" if i % 2 == 0 else "mountain"
        # Miura-ori: alternate slight angle offset to create parallelogram cells
        angle_offset = 0.02 * (1 if i % 2 == 0 else -1)
        sub_goals.append({
            "step_number": step_num,
            "description": f"Vertical {fold_type} fold at x={x:.3f} (Miura-ori column {i + 1}/{n_vertical})",
            "base_required": None,
            "fold_operations": [{
                "type": fold_type,
                "line": {"start": [x, 0.0 + angle_offset], "end": [x, 1.0 - angle_offset]},
                "angle": 180,
                "layer_select": "all",
            }],
            "expected_state": {
                "layers": (n_horizontal + 1) * (i + 2),
                "phase": "flat",
                "pattern": "miura_complete" if i == n_vertical - 1 else "miura_partial",
            },
            "validation": {"flat_foldable": True, "kawasaki_check": True, "maekawa_check": True},
        })

    # Final collapse
    step_num += 1
    sub_goals.append({
        "step_number": step_num,
        "description": "Collapse all creases simultaneously into compact Miura-ori stack",
        "base_required": None,
        "fold_operations": [{
            "type": "valley",
            "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
            "angle": 180,
            "layer_select": "all",
        }],
        "expected_state": {
            "layers": (n_horizontal + 1) * (n_vertical + 1),
            "phase": "compact",
            "pattern": "miura_ori",
        },
        "validation": {
            "flat_foldable": True,
            "check_bounding_box": constraints.get("target_box"),
            "check_deployable": constraints.get("must_deploy", False),
        },
    })

    return sub_goals


# ---------------------------------------------------------------------------
# Free-fold / unknown model decomposition
# ---------------------------------------------------------------------------

def _decompose_free_fold(parsed: dict) -> list[dict]:
    """
    Generic decomposition for an unknown model or free-form folding task.
    Returns a minimal plan that an LLM can expand upon.
    """
    return [
        {
            "step_number": 1,
            "description": "Create reference creases (diagonals and midlines)",
            "base_required": None,
            "fold_operations": [
                {"type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180},
                {"type": "unfold", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 0},
                {"type": "valley", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 180},
                {"type": "unfold", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 0},
                {"type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180},
                {"type": "unfold", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0},
            ],
            "expected_state": {"layers": 1, "shape": "square", "phase": "creased"},
            "validation": {"flat_foldable": True},
        },
        {
            "step_number": 2,
            "description": "Collapse into a base form using reference creases",
            "base_required": "preliminary_base",
            "fold_operations": [
                {"type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},
            ],
            "expected_state": {"layers": 4, "shape": "diamond", "phase": "base"},
            "validation": {"flat_foldable": True},
        },
        {
            "step_number": 3,
            "description": "Shape the model with additional folds (LLM determines specifics)",
            "base_required": None,
            "fold_operations": [],  # Left empty for LLM to fill
            "expected_state": {"phase": "shaped"},
            "validation": {"flat_foldable": True},
        },
    ]


# ---------------------------------------------------------------------------
# Fold-pattern decomposition
# ---------------------------------------------------------------------------

def _decompose_pattern(parsed: dict) -> list[dict]:
    """Decompose a tessellation/pattern task."""
    # For now, delegate to packing which generates a Miura-ori pattern
    return _decompose_packing(parsed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_task(parsed: dict) -> list[dict]:
    """
    Decompose a parsed instruction into sequential sub-goals.

    Args:
        parsed: Output of parse_instruction()

    Returns:
        List of sub-goal dicts, each with:
          - step_number: int
          - description: str
          - base_required: str or None
          - fold_operations: list[dict]  (engine-format fold dicts)
          - expected_state: dict
          - validation: dict
    """
    intent = parsed.get("intent", "free_fold")

    if intent == "fold_model" and parsed.get("model_name"):
        return _decompose_known_model(parsed)
    elif intent == "optimize_packing":
        return _decompose_packing(parsed)
    elif intent == "fold_pattern":
        return _decompose_pattern(parsed)
    else:
        return _decompose_free_fold(parsed)
