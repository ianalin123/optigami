"""
Prompt templates for origami fold strategy generation.

Inspired by SpatialThinker (arXiv 2511.07403): the model must produce
a structured spatial representation BEFORE generating code.

Output format (4 stages):
  <observe>  — Describe the paper geometry and constraints
  <plan>     — Structured fold plan with coordinates and reasoning
  <code>     — The fold_strategy() function
  <verify>   — Predict expected outcome (deployment ratio, fold count)

Dense rewards check each stage independently, not just code execution.
"""

# ---------------------------------------------------------------------------
# System prompt — defines the structured output format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an origami engineer specializing in computational fold design.
You solve folding tasks by reasoning spatially about paper geometry.

You MUST respond in exactly this 4-stage format:

<observe>
Describe the paper: dimensions, material, coordinate system.
Identify key geometric features (center, edges, diagonals, symmetry axes).
Note constraints (max strain, max folds, target ratio).
</observe>

<plan>
{
  "strategy": "description of overall approach",
  "folds": [
    {
      "description": "what this fold does",
      "type": "valley or mountain",
      "line_start": [x, y],
      "line_end": [x, y],
      "angle": 180,
      "reasoning": "why these coordinates"
    }
  ],
  "expected_ratio": 0.5,
  "expected_folds": 1
}
</plan>

<code>
```python
def fold_strategy(paper_state):
    # Implementation matching the plan above
    return [...]
```
</code>

<verify>
Expected deployment ratio: X.XX
Expected fold count: N
Expected max strain: X.XXXX
Potential issues: ...
</verify>

Rules:
- Only use native Python (no imports except math, itertools, functools)
- Each fold: {"type": "valley"|"mountain", "line": {"start": [x,y], "end": [x,y]}, "angle": 0-180}
- Fold lines must cross the paper boundary (intersect at least 2 edges)
- Valley = fold toward you (+Z), Mountain = fold away (-Z)
- angle=180 = fully folded, smaller = partial fold
- Each fold changes the geometry — later folds operate on already-folded paper
- Fewer folds is better (efficiency matters)
- Respect material strain limits\
"""


# ---------------------------------------------------------------------------
# Task templates — each includes spatial context
# ---------------------------------------------------------------------------

TASK_TEMPLATES = {
    "half_fold": {
        "name": "half_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m {material} sheet in half to minimize one dimension.

PAPER GEOMETRY:
  Corners: (0,0), ({width},0), ({width},{height}), (0,{height})
  Center: ({cx},{cy})
  Horizontal midline: y={cy} from (0,{cy}) to ({width},{cy})
  Vertical midline: x={cx} from ({cx},0) to ({cx},{height})
  Diagonals: (0,0)→({width},{height}) and ({width},0)→(0,{height})

MATERIAL: {material} (thickness: {thickness_mm}mm, max strain: {max_strain_pct}%)
CONSTRAINTS: Maximum {max_folds} fold operations.
TARGET: Deployment ratio <= 0.5""",
        "target_ratio": 0.5,
        "max_folds": 3,
    },

    "letter_fold": {
        "name": "letter_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m {material} sheet into thirds (like a letter).

PAPER GEOMETRY:
  Corners: (0,0), ({width},0), ({width},{height}), (0,{height})
  Third lines: y={t1:.4f} and y={t2:.4f}
  Center: ({cx},{cy})

MATERIAL: {material} (thickness: {thickness_mm}mm, max strain: {max_strain_pct}%)
CONSTRAINTS: Maximum {max_folds} fold operations.
TARGET: Deployment ratio <= 0.33""",
        "target_ratio": 0.33,
        "max_folds": 5,
    },

    "solar_panel": {
        "name": "solar_panel",
        "prompt": """\
TASK: Fold a {width}m x {height}m Mylar sheet to minimize packed volume for a solar panel.
The folded panel must be deployable (unfold cleanly to near-original area).

PAPER GEOMETRY:
  Corners: (0,0), ({width},0), ({width},{height}), (0,{height})
  Center: ({cx},{cy})
  Area: {area}m²

MATERIAL: Mylar (thickness: 0.05mm, Young's modulus: 4 GPa, max strain: 3%)
CONSTRAINTS:
  - Maximum {max_folds} fold operations
  - Must pack into bounding box <= 15cm x 15cm x 5cm
  - No self-intersections

TARGET: Deployment ratio <= 0.05 (95% area reduction)

HINT: Tessellated patterns (alternating M/V folds in a grid) achieve high
compaction with single-DOF deployment. Consider dividing the sheet into
a regular grid of panels.""",
        "target_ratio": 0.05,
        "max_folds": 20,
    },

    "stent_fold": {
        "name": "stent_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m Nitinol sheet into a compact cylinder for a medical stent.

PAPER GEOMETRY:
  Corners: (0,0), ({width},0), ({width},{height}), (0,{height})
  Center: ({cx},{cy})

MATERIAL: Nitinol (thickness: 0.1mm, Young's modulus: 75 GPa, max strain: 8%)
CONSTRAINTS:
  - Maximum {max_folds} fold operations
  - Compressed diameter: 3mm, Deployed diameter: 10mm

TARGET: Deployment ratio <= 0.1""",
        "target_ratio": 0.1,
        "max_folds": 15,
    },
}


# ---------------------------------------------------------------------------
# Config and builders
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "half_fold": {
        "width": 1.0, "height": 1.0, "material": "paper",
        "thickness_mm": 0.1, "max_strain_pct": 3, "max_folds": 3,
    },
    "letter_fold": {
        "width": 1.0, "height": 1.0, "material": "paper",
        "thickness_mm": 0.1, "max_strain_pct": 3, "max_folds": 5,
    },
    "solar_panel": {
        "width": 1.0, "height": 1.0, "material": "mylar",
        "thickness_mm": 0.05, "max_strain_pct": 3, "max_folds": 20,
    },
    "stent_fold": {
        "width": 0.1, "height": 0.03, "material": "nitinol",
        "thickness_mm": 0.1, "max_strain_pct": 8, "max_folds": 15,
    },
}


def build_prompt(task_name: str = "half_fold", **overrides) -> str:
    """Build a complete user prompt for a given task."""
    task = TASK_TEMPLATES[task_name]
    config = {**TASK_CONFIGS[task_name], **overrides}

    # Add computed geometry values
    w = config["width"]
    h = config["height"]
    config["cx"] = w / 2
    config["cy"] = h / 2
    config["area"] = w * h
    config["t1"] = h / 3
    config["t2"] = 2 * h / 3

    return task["prompt"].format(**config)


def get_task_target_ratio(task_name: str) -> float:
    return TASK_TEMPLATES[task_name]["target_ratio"]


def get_task_max_folds(task_name: str) -> int:
    return TASK_TEMPLATES[task_name]["max_folds"]
