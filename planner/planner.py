"""
OrigamiPlanner: the main orchestrator that converts human instructions
into structured fold plans with LLM-ready prompts.

Usage:
    from planner.planner import OrigamiPlanner

    planner = OrigamiPlanner()
    plan = planner.plan("make a paper crane")
    print(plan.summary())
    prompt = plan.get_prompt_for_step(0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from planner.parser import parse_instruction
from planner.decomposer import decompose_task
from planner.knowledge import ORIGAMI_MODELS, FOLD_OPERATIONS


# ---------------------------------------------------------------------------
# Material defaults (mirrors trainer/prompts.py TASK_CONFIGS)
# ---------------------------------------------------------------------------

_MATERIAL_DEFAULTS = {
    "paper":     {"thickness_mm": 0.1,  "youngs_modulus_gpa": 2.0,  "max_strain_pct": 3},
    "mylar":     {"thickness_mm": 0.05, "youngs_modulus_gpa": 4.0,  "max_strain_pct": 3},
    "aluminum":  {"thickness_mm": 0.02, "youngs_modulus_gpa": 69.0, "max_strain_pct": 1},
    "metal":     {"thickness_mm": 0.05, "youngs_modulus_gpa": 200.0,"max_strain_pct": 0.5},
    "nitinol":   {"thickness_mm": 0.1,  "youngs_modulus_gpa": 75.0, "max_strain_pct": 8},
    "cardboard": {"thickness_mm": 1.0,  "youngs_modulus_gpa": 1.0,  "max_strain_pct": 2},
    "cardstock": {"thickness_mm": 0.3,  "youngs_modulus_gpa": 1.5,  "max_strain_pct": 2},
    "fabric":    {"thickness_mm": 0.2,  "youngs_modulus_gpa": 0.1,  "max_strain_pct": 15},
}


# ---------------------------------------------------------------------------
# FoldPlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoldPlan:
    """
    A complete, executable fold plan produced by OrigamiPlanner.

    Attributes:
        instruction: The original human instruction.
        parsed: Structured parse result (intent, model, material, etc.).
        steps: Ordered list of sub-goal dicts from the decomposer.
        prompts: Pre-built LLM prompts, one per step.
    """

    instruction: str
    parsed: dict
    steps: list[dict]
    prompts: list[str]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines: list[str] = []
        lines.append(f"Origami Plan: {self.instruction}")
        lines.append(f"  Intent : {self.parsed['intent']}")
        if self.parsed.get("model_name"):
            model = ORIGAMI_MODELS.get(self.parsed["model_name"], {})
            lines.append(f"  Model  : {model.get('name', self.parsed['model_name'])}")
            lines.append(f"  Difficulty: {model.get('difficulty', 'unknown')}")
        lines.append(f"  Material: {self.parsed['material']}")
        dims = self.parsed["dimensions"]
        lines.append(f"  Sheet  : {dims['width']}m x {dims['height']}m")
        lines.append(f"  Steps  : {len(self.steps)}")
        lines.append("")
        lines.append("Step-by-step:")
        for s in self.steps:
            n = s["step_number"]
            desc = s["description"]
            n_ops = len(s.get("fold_operations", []))
            lines.append(f"  {n:>3}. {desc}  ({n_ops} fold op{'s' if n_ops != 1 else ''})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt access
    # ------------------------------------------------------------------

    def get_prompt_for_step(self, step_index: int, current_state: dict | None = None) -> str:
        """
        Get the LLM prompt for a specific step, optionally enriched with
        the current paper state from the simulation engine.

        Args:
            step_index: Zero-based index into self.steps.
            current_state: Optional live paper_state dict from the engine.

        Returns:
            A fully-formatted prompt string ready for the LLM.
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(f"step_index {step_index} out of range (0..{len(self.steps) - 1})")

        base_prompt = self.prompts[step_index]

        if current_state is None:
            return base_prompt

        # Inject live state into the prompt
        state_block = _format_state_block(current_state)
        return base_prompt.replace("{{CURRENT_STATE}}", state_block)

    # ------------------------------------------------------------------
    # Convenience: all fold operations flattened
    # ------------------------------------------------------------------

    def all_fold_operations(self) -> list[dict]:
        """Return every fold operation across all steps, in order."""
        ops: list[dict] = []
        for step in self.steps:
            ops.extend(step.get("fold_operations", []))
        return ops

    def total_fold_count(self) -> int:
        """Total number of fold operations in the plan."""
        return sum(len(s.get("fold_operations", [])) for s in self.steps)


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

def _format_state_block(state: dict) -> str:
    """Format a paper_state dict as a human-readable block for the prompt."""
    lines = ["CURRENT STATE:"]
    if "bounding_box" in state:
        bb = state["bounding_box"]
        if isinstance(bb, dict):
            lines.append(f"  Bounding box: {bb.get('x', '?')}m x {bb.get('y', '?')}m x {bb.get('z', '?')}m")
        elif isinstance(bb, (list, tuple)) and len(bb) >= 3:
            lines.append(f"  Bounding box: {bb[0]}m x {bb[1]}m x {bb[2]}m")
    if "num_layers_at_center" in state:
        lines.append(f"  Layers at center: {state['num_layers_at_center']}")
    if "deployment_ratio" in state:
        lines.append(f"  Deployment ratio: {state['deployment_ratio']:.3f}")
    if "fold_angles" in state:
        n_folds = sum(1 for a in state["fold_angles"] if a != 0)
        lines.append(f"  Active folds: {n_folds}")
    return "\n".join(lines)


def _format_fold_ops_as_code(ops: list[dict]) -> str:
    """Format fold operations as Python list literal for inclusion in a prompt."""
    if not ops:
        return "    # (LLM: determine fold operations for this step)\n    return []"

    lines = ["    return ["]
    for op in ops:
        clean = {
            "type": op["type"],
            "line": op.get("line", {"start": [0, 0], "end": [1, 1]}),
            "angle": op.get("angle", 180),
        }
        lines.append(f"        {json.dumps(clean)},")
    lines.append("    ]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OrigamiPlanner
# ---------------------------------------------------------------------------

class OrigamiPlanner:
    """
    Full pipeline: human instruction -> structured plan -> executable fold operations.

    The planner:
      1. Parses the instruction (parser.py)
      2. Decomposes into sub-goals (decomposer.py)
      3. Builds LLM-ready prompts matching trainer/prompts.py format
    """

    def plan(self, instruction: str) -> FoldPlan:
        """
        Plan an origami task from a human instruction.

        Args:
            instruction: e.g. "make a paper crane", "pack a 1m mylar sheet"

        Returns:
            A FoldPlan with steps and LLM prompts.
        """
        parsed = parse_instruction(instruction)
        steps = decompose_task(parsed)
        prompts = [self._build_prompt(step, i, parsed) for i, step in enumerate(steps)]
        return FoldPlan(
            instruction=instruction,
            parsed=parsed,
            steps=steps,
            prompts=prompts,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, step: dict, step_index: int, parsed: dict) -> str:
        """
        Build an LLM-ready prompt for a single sub-goal step.

        The format matches trainer/prompts.py: task description at top,
        material/constraints in the middle, and a fold_strategy() code
        block wrapped in triple backticks at the bottom.
        """
        material = parsed["material"]
        mat_info = _MATERIAL_DEFAULTS.get(material, _MATERIAL_DEFAULTS["paper"])
        dims = parsed["dimensions"]
        constraints = parsed.get("constraints", {})
        total_steps = len(parsed.get("_all_steps", [])) or step.get("step_number", 1)

        # ---- Header ----
        intent = parsed["intent"]
        if intent == "fold_model" and parsed.get("model_name"):
            model_info = ORIGAMI_MODELS.get(parsed["model_name"], {})
            task_line = (
                f"TASK: Step {step['step_number']} of {total_steps} — "
                f"{step['description']}\n"
                f"MODEL: {model_info.get('name', parsed['model_name'])} "
                f"(difficulty: {model_info.get('difficulty', 'unknown')})"
            )
        elif intent == "optimize_packing":
            task_line = (
                f"TASK: Step {step['step_number']} — {step['description']}\n"
                f"GOAL: Minimize packed volume while maintaining deployability."
            )
        else:
            task_line = f"TASK: Step {step['step_number']} — {step['description']}"

        # ---- Material ----
        material_block = (
            f"MATERIAL:\n"
            f"  - Name: {material}\n"
            f"  - Thickness: {mat_info['thickness_mm']}mm\n"
            f"  - Max strain: {mat_info['max_strain_pct']}%"
        )

        # ---- Constraints ----
        constraint_lines = ["CONSTRAINTS:"]
        if "max_folds" in constraints:
            constraint_lines.append(f"  - Maximum {constraints['max_folds']} fold operations")
        if "target_box" in constraints:
            tb = constraints["target_box"]
            constraint_lines.append(
                f"  - Must pack into bounding box <= "
                f"{tb[0]*100:.0f}cm x {tb[1]*100:.0f}cm x {tb[2]*100:.0f}cm"
            )
        if constraints.get("must_deploy"):
            constraint_lines.append("  - Must deploy to >= 80% of original area")
        constraint_lines.append("  - No self-intersections allowed")
        constraints_block = "\n".join(constraint_lines)

        # ---- State placeholder ----
        state_block = (
            f"CURRENT STATE:\n"
            f"  Sheet: {dims['width']}m x {dims['height']}m\n"
            f"  {{{{CURRENT_STATE}}}}"
        )

        # ---- Fold operations hint ----
        ops = step.get("fold_operations", [])
        ops_code = _format_fold_ops_as_code(ops)

        # ---- Expected result ----
        expected = step.get("expected_state", {})
        expected_block = ""
        if expected:
            expected_block = f"\nEXPECTED RESULT: {json.dumps(expected)}"

        # ---- Code block (matches trainer/prompts.py format) ----
        code_block = (
            f'Write a fold_strategy(paper_state) function that returns a list of fold operations.\n'
            f'Each fold: {{"type": "valley"|"mountain", "line": {{"start": [x,y], "end": [x,y]}}, "angle": 0-180}}\n'
            f'\n'
            f'```python\n'
            f'def fold_strategy(paper_state):\n'
            f'{ops_code}\n'
            f'```'
        )

        # ---- Assemble ----
        sections = [
            task_line,
            "",
            material_block,
            "",
            constraints_block,
            "",
            state_block,
            expected_block,
            "",
            code_block,
        ]
        return "\n".join(sections)
