"""
Reward functions for origami GRPO training.

Three reward functions following the 2048 pattern:
  1. code_valid     — Does the generated code parse and produce fold instructions?
  2. physically_valid — Are the folds geometrically/physically valid?
  3. fold_quality   — How good is the folding solution (compactness, efficiency)?

Lexicographic gating (from SpatialThinker): if code doesn't parse,
all downstream rewards are 0. This prevents reward hacking.
"""

import ast
import sys
import math
import traceback
from typing import Callable

# Use real engine if available, fall back to mock
try:
    from engine.paper import Paper
    from engine.fold_engine import execute_fold_strategy
    from engine.materials import Material, get_material
    from engine.validation import validate_paper
    from engine.metrics import compute_metrics

    def _create_sheet(width, height, material):
        return Paper.create_flat_sheet(width, height, material)

    USE_REAL_ENGINE = True
except ImportError:
    from trainer.mock_env import (
        PaperState as Paper, create_flat_sheet, execute_fold_strategy, Material
    )

    def _create_sheet(width, height, material):
        return create_flat_sheet(width, height, material)

    def validate_paper(p):
        from types import SimpleNamespace
        return SimpleNamespace(
            is_valid=p.is_valid, kawasaki_valid=True, maekawa_valid=True,
            kawasaki_violation=p.kawasaki_violation,
            maekawa_violation=p.maekawa_violation,
            self_intersection_count=p.self_intersections,
        )

    def compute_metrics(p, orig):
        return {
            "deployment_ratio": p.deployment_ratio,
            "fold_count": sum(1 for a in p.assignments if a in ("M", "V")),
            "max_strain": float(p.strain.max()) if len(p.strain) > 0 else 0.0,
        }

    USE_REAL_ENGINE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_function(text: str) -> str | None:
    """Extract a Python function from triple-backtick code blocks."""
    if text.count("```") < 2:
        return None
    first = text.find("```") + 3
    second = text.find("```", first)
    fx = text[first:second].strip()
    fx = fx.removeprefix("python\n").removeprefix("python\r\n")
    # Find the def statement
    def_idx = fx.find("def ")
    if def_idx == -1:
        return None
    fx = fx[def_idx:]
    if fx.startswith("def fold_strategy("):
        return fx
    return None


def check_imports_stdlib_only(code: str) -> tuple[bool, str]:
    """Check that code only imports from Python stdlib."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    ALLOWED_MODULES = {
        "math", "itertools", "functools", "collections", "copy",
        "operator", "typing", "random", "heapq", "bisect",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"non-stdlib import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"non-stdlib import: {node.module}"

    return True, "ok"


def create_sandboxed_function(code: str) -> Callable:
    """
    Execute the function code in a restricted namespace.
    Returns the fold_strategy function object.
    """
    allowed_builtins = {
        "range", "len", "int", "float", "str", "list", "dict", "tuple",
        "set", "bool", "abs", "min", "max", "sum", "sorted", "reversed",
        "enumerate", "zip", "map", "filter", "round", "isinstance",
        "True", "False", "None", "print",
    }
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict)
                     else getattr(__builtins__, k)
                     for k in allowed_builtins
                     if (k in __builtins__ if isinstance(__builtins__, dict)
                         else hasattr(__builtins__, k))}
    safe_builtins["__import__"] = __import__  # needed for stdlib imports

    namespace = {"__builtins__": safe_builtins}
    exec(code, namespace)

    if "fold_strategy" not in namespace:
        raise ValueError("No fold_strategy function defined")

    return namespace["fold_strategy"]


# ---------------------------------------------------------------------------
# State for strategy execution
# ---------------------------------------------------------------------------

# Current task config (set by train.py before training starts)
if USE_REAL_ENGINE:
    _default_material = get_material("paper")
else:
    _default_material = Material()

_current_task = {
    "width": 1.0,
    "height": 1.0,
    "material": _default_material,
    "target_ratio": 0.5,
    "max_folds": 3,
}

PRINT_EVERY = 5
_print_counter = 0


def set_task_config(width=1.0, height=1.0, material=None,
                    target_ratio=0.5, max_folds=3):
    global _current_task
    _current_task = {
        "width": width,
        "height": height,
        "material": material or Material(),
        "target_ratio": target_ratio,
        "max_folds": max_folds,
    }


# ---------------------------------------------------------------------------
# Reward 1: code_valid
# ---------------------------------------------------------------------------

def code_valid(completions, **kwargs) -> list[float]:
    """
    Does the generated code parse as valid Python and produce a callable?

    +1.0  — valid function that can be created
    -0.5  — correct structure but exec/sandbox fails
    -2.0  — no function found or syntax error
    -20.0 — non-stdlib imports (heavy penalty)
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function_code = extract_function(response)

        if function_code is None:
            scores.append(-2.0)
            continue

        ok, info = check_imports_stdlib_only(function_code)
        if not ok:
            if "syntax error" in info:
                scores.append(-2.0)
            else:
                scores.append(-20.0)  # non-stdlib imports
            continue

        try:
            create_sandboxed_function(function_code)
            scores.append(1.0)
        except Exception:
            scores.append(-0.5)

    return scores


# ---------------------------------------------------------------------------
# Reward 2: physically_valid
# ---------------------------------------------------------------------------

def physically_valid(completions, **kwargs) -> list[float]:
    """
    Are the folds physically possible?

    +1.0  — all folds valid, no violations
    -2.0  — per Kawasaki/Maekawa violation
    -5.0  — any self-intersection
    -1.0  — strain exceeds material limit
     0.0  — function broken / can't run
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function_code = extract_function(response)

        if function_code is None:
            scores.append(0.0)
            continue

        ok, info = check_imports_stdlib_only(function_code)
        if not ok:
            scores.append(0.0)
            continue

        try:
            strategy_fn = create_sandboxed_function(function_code)
        except Exception:
            scores.append(0.0)
            continue

        try:
            paper = _create_sheet(
                _current_task["width"],
                _current_task["height"],
                _current_task["material"],
            )
            original = paper
            final_state, applied, error = execute_fold_strategy(
                strategy_fn, paper, _current_task["max_folds"]
            )

            if error:
                scores.append(0.0)
                continue

            if len(applied) == 0:
                scores.append(0.0)
                continue

            # Score based on validity using engine validation
            val = validate_paper(final_state)
            metrics = compute_metrics(final_state, original)

            score = 1.0
            score -= 2.0 * val.kawasaki_violation
            score -= 2.0 * val.maekawa_violation
            if val.self_intersection_count > 0:
                score -= 5.0
            max_strain = metrics.get("max_strain", 0.0)
            if max_strain > _current_task["material"].max_strain:
                score -= 1.0

            scores.append(score)

        except TimeoutError:
            scores.append(-1.0)
        except Exception:
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Reward 3: fold_quality
# ---------------------------------------------------------------------------

def fold_quality(completions, **kwargs) -> list[float]:
    """
    How good is the folding solution?

    +20.0 * compactness — main reward (1 - deployment_ratio)
    +10.0 bonus         — if meets target ratio
    -0.5 per fold       — efficiency penalty
    -3.0 * overstrain   — material stress penalty
    -1.0                — timeout
    -3.0                — exception
     0.0                — function broken
    """
    global _print_counter
    scores = []

    for completion in completions:
        response = completion[0]["content"]
        function_code = extract_function(response)

        should_print = (_print_counter % PRINT_EVERY == 0)
        _print_counter += 1

        if should_print:
            print(f"\n--- Strategy (sample {_print_counter}) ---")
            print(function_code if function_code else "[no function extracted]")

        if function_code is None:
            scores.append(0.0)
            continue

        ok, info = check_imports_stdlib_only(function_code)
        if not ok:
            scores.append(0.0)
            continue

        try:
            strategy_fn = create_sandboxed_function(function_code)
        except Exception:
            scores.append(0.0)
            continue

        try:
            paper = _create_sheet(
                _current_task["width"],
                _current_task["height"],
                _current_task["material"],
            )
            original = paper
            final_state, applied, error = execute_fold_strategy(
                strategy_fn, paper, _current_task["max_folds"]
            )

            if error:
                if should_print:
                    print(f"Error: {error}")
                scores.append(0.0)
                continue

            num_folds = len(applied)
            if num_folds == 0:
                scores.append(0.0)
                continue

            # Use engine metrics
            metrics = compute_metrics(final_state, original)
            deploy_ratio = metrics.get("deployment_ratio", 1.0)
            max_strain = metrics.get("max_strain", 0.0)

            # Compactness: main reward signal
            compactness = 1.0 - deploy_ratio
            score = 20.0 * compactness

            # Bonus for meeting target
            if deploy_ratio <= _current_task["target_ratio"]:
                score += 10.0

            # Fold efficiency penalty
            score -= 0.5 * num_folds

            # Strain penalty
            mat_limit = _current_task["material"].max_strain
            if max_strain > mat_limit:
                score -= 3.0 * (max_strain / mat_limit)

            if should_print:
                print(f"Folds: {num_folds}, Ratio: {deploy_ratio:.3f}, "
                      f"Compactness: {compactness:.3f}, Score: {score:.2f}")
                bb = metrics.get("bounding_box", {})
                print(f"BBox: {bb.get('x',0):.3f} x {bb.get('y',0):.3f} x {bb.get('z',0):.3f}")

            scores.append(score)

        except TimeoutError:
            if should_print:
                print("Timeout!")
            scores.append(-1.0)
        except Exception as e:
            if should_print:
                print(f"Exception: {e}")
            scores.append(-3.0)

    return scores
