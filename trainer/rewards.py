"""
Reward functions for origami GRPO training.

SpatialThinker-style dense rewards (arXiv 2511.07403):
  1. format_reward     (0.10) — All 4 tags present, valid JSON plan, valid function
  2. spatial_reward    (0.20) — Fold coordinates in plan are within bounds, lines valid
  3. execution_reward  (0.50) — Physical validity + fold quality (code execution)
  4. consistency_reward(0.20) — Plan matches code, verify matches actual results

Plus legacy rewards for backwards compatibility:
  - code_valid, physically_valid, fold_quality

Lexicographic gating: if code doesn't parse, downstream rewards are 0.
"""

import ast
import re
import sys
import json
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
    """Extract fold_strategy() from <code> blocks or triple-backtick code blocks."""
    # Try <code> block first (SpatialThinker format)
    code_match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if code_match:
        code_block = code_match.group(1).strip()
    elif text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        code_block = text[first:second].strip()
    else:
        return None

    code_block = code_block.removeprefix("```python\n").removeprefix("```python\r\n")
    code_block = code_block.removeprefix("python\n").removeprefix("python\r\n")
    code_block = code_block.rstrip("`").strip()

    # Find the def statement
    def_idx = code_block.find("def ")
    if def_idx == -1:
        return None
    fx = code_block[def_idx:]
    if fx.startswith("def fold_strategy("):
        return fx
    return None


def extract_section(text: str, tag: str) -> str | None:
    """Extract content between <tag>...</tag>."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_plan_json(text: str) -> dict | None:
    """Extract and parse the JSON fold plan from <plan> block."""
    plan_text = extract_section(text, "plan")
    if not plan_text:
        return None
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        # Try to find JSON object within the plan text
        brace_start = plan_text.find("{")
        brace_end = plan_text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(plan_text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
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


# ---------------------------------------------------------------------------
# SpatialThinker Dense Rewards (weight 0.10 + 0.20 + 0.50 + 0.20 = 1.0)
# ---------------------------------------------------------------------------

REQUIRED_TAGS = ["observe", "plan", "code", "verify"]


def format_reward(completions, **kwargs) -> list[float]:
    """
    SpatialThinker format reward (weight: 0.10).

    Checks that the response has all 4 structured tags, valid JSON in <plan>,
    and a parseable function in <code>.

    Score range: [0.0, 1.0]
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0

        # Check each required tag (0.15 each = 0.60 for all 4)
        tags_present = 0
        for tag in REQUIRED_TAGS:
            if extract_section(response, tag) is not None:
                tags_present += 1
        score += 0.15 * tags_present

        # Valid JSON in <plan> (0.20)
        plan = extract_plan_json(response)
        if plan is not None:
            score += 0.20
            # Plan has required fields (0.05 bonus)
            if "folds" in plan and isinstance(plan["folds"], list):
                score += 0.05

        # Valid function in <code> (0.15)
        fn = extract_function(response)
        if fn is not None:
            score += 0.15

        scores.append(score)
    return scores


def spatial_reward(completions, **kwargs) -> list[float]:
    """
    SpatialThinker spatial plan quality reward (weight: 0.20).

    Checks that fold coordinates in <plan> are geometrically valid:
    - Within paper bounds
    - Line endpoints form valid fold lines (cross the paper)
    - Fold types are valid
    - Expected ratio/count are reasonable

    Score range: [0.0, 1.0]
    """
    w = _current_task["width"]
    h = _current_task["height"]

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        plan = extract_plan_json(response)

        if plan is None:
            scores.append(0.0)
            continue

        score = 0.0
        folds = plan.get("folds", [])

        if not folds:
            scores.append(0.0)
            continue

        # Score each fold in the plan
        valid_folds = 0
        for fold in folds:
            fold_score = 0.0

            # Has required fields
            has_type = fold.get("type") in ("valley", "mountain")
            has_start = isinstance(fold.get("line_start"), list) and len(fold.get("line_start", [])) == 2
            has_end = isinstance(fold.get("line_end"), list) and len(fold.get("line_end", [])) == 2

            if has_type:
                fold_score += 0.25
            if has_start and has_end:
                fold_score += 0.25
                # Coordinates within paper bounds (with small tolerance)
                sx, sy = fold["line_start"]
                ex, ey = fold["line_end"]
                tol = 0.01
                in_bounds = (
                    -tol <= sx <= w + tol and -tol <= sy <= h + tol and
                    -tol <= ex <= w + tol and -tol <= ey <= h + tol
                )
                if in_bounds:
                    fold_score += 0.25

                # Start != end (not a degenerate line)
                dist = math.sqrt((ex - sx)**2 + (ey - sy)**2)
                if dist > 0.01:
                    fold_score += 0.25

            if fold_score > 0.5:
                valid_folds += 1

        # Proportion of valid folds
        score = valid_folds / len(folds) if folds else 0.0

        # Bonus: expected_ratio is reasonable (0.0 to 1.0)
        expected = plan.get("expected_ratio")
        if isinstance(expected, (int, float)) and 0.0 < expected <= 1.0:
            score = min(1.0, score + 0.1)

        scores.append(min(1.0, score))
    return scores


def execution_reward(completions, **kwargs) -> list[float]:
    """
    SpatialThinker execution/accuracy reward (weight: 0.50).

    Combines code validity, physical validity, and fold quality into
    one normalized score. This is the main reward signal.

    Score range: [0.0, 1.0]
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function_code = extract_function(response)

        # Gate: no function → 0
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

            if error or len(applied) == 0:
                scores.append(0.0)
                continue

            val = validate_paper(final_state)
            metrics = compute_metrics(final_state, original)
            deploy_ratio = metrics.get("deployment_ratio", 1.0)
            max_strain = metrics.get("max_strain", 0.0)

            # Physical validity component (0-0.3)
            phys = 0.3
            if not val.is_valid:
                phys -= 0.1 * val.kawasaki_violation
                phys -= 0.1 * val.maekawa_violation
                if val.self_intersection_count > 0:
                    phys -= 0.15
            mat_limit = _current_task["material"].max_strain
            if max_strain > mat_limit:
                phys -= 0.05
            phys = max(0.0, phys)

            # Quality component (0-0.5)
            compactness = 1.0 - deploy_ratio
            quality = 0.5 * compactness

            # Target bonus (0-0.2)
            target = 0.0
            if deploy_ratio <= _current_task["target_ratio"]:
                target = 0.2

            score = phys + quality + target
            scores.append(min(1.0, score))

        except Exception:
            scores.append(0.0)

    return scores


def consistency_reward(completions, **kwargs) -> list[float]:
    """
    SpatialThinker consistency reward (weight: 0.20).

    Checks that <plan> matches <code> and <verify> matches actual results.
    - Plan fold count matches code fold count
    - Verify predictions close to actual metrics

    Score range: [0.0, 1.0]
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        plan = extract_plan_json(response)
        verify = extract_section(response, "verify")
        function_code = extract_function(response)

        # Need at least plan + code to check consistency
        if plan is None or function_code is None:
            scores.append(0.0)
            continue

        score = 0.0

        # 1. Plan fold count vs code fold count (0.4)
        plan_folds = plan.get("folds", [])
        plan_count = len(plan_folds)

        try:
            strategy_fn = create_sandboxed_function(function_code)
            paper = _create_sheet(
                _current_task["width"],
                _current_task["height"],
                _current_task["material"],
            )
            original = paper
            final_state, applied, error = execute_fold_strategy(
                strategy_fn, paper, _current_task["max_folds"]
            )
            if error or len(applied) == 0:
                scores.append(0.0)
                continue

            actual_count = len(applied)
            if plan_count == actual_count:
                score += 0.4
            elif abs(plan_count - actual_count) <= 1:
                score += 0.2

            # 2. Verify predictions vs actual (0.6)
            if verify:
                metrics = compute_metrics(final_state, original)
                actual_ratio = metrics.get("deployment_ratio", 1.0)

                # Extract predicted ratio from verify text
                ratio_match = re.search(
                    r'deployment\s*ratio[:\s]*([\d.]+)', verify, re.IGNORECASE)
                if ratio_match:
                    predicted_ratio = float(ratio_match.group(1))
                    error_pct = abs(predicted_ratio - actual_ratio)
                    if error_pct < 0.05:
                        score += 0.4
                    elif error_pct < 0.15:
                        score += 0.2
                    elif error_pct < 0.3:
                        score += 0.1

                # Extract predicted fold count
                count_match = re.search(
                    r'fold\s*count[:\s]*(\d+)', verify, re.IGNORECASE)
                if count_match:
                    predicted_count = int(count_match.group(1))
                    if predicted_count == actual_count:
                        score += 0.2
                    elif abs(predicted_count - actual_count) <= 1:
                        score += 0.1

        except Exception:
            scores.append(0.0)
            continue

        scores.append(min(1.0, score))
    return scores
