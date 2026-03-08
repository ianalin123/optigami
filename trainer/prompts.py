"""
Prompt templates for origami fold strategy generation.

The LLM receives a task description and paper state, then generates
a fold_strategy(paper_state) function that returns fold operations.
"""

SYSTEM_PROMPT = """\
You are an origami engineer. You design fold patterns for real-world applications \
like solar panel packing, deployable shelters, and medical stents.

You will be given a folding task with material constraints. Write a Python function \
`fold_strategy(paper_state)` that returns a list of fold operations to achieve the goal.

Rules:
- Only use native Python (no imports except math, itertools, functools)
- Each fold: {"type": "valley"|"mountain", "line": {"start": [x,y], "end": [x,y]}, "angle": 0-180}
- Fold lines must intersect the paper boundaries
- Fewer folds is better (efficiency matters)
- Respect material strain limits
- Output ONLY the function in ```python ... ``` backticks\
"""


TASK_TEMPLATES = {
    "half_fold": {
        "name": "half_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m {material} sheet in half to minimize one dimension.

MATERIAL: {material} (thickness: {thickness_mm}mm, max strain: {max_strain_pct}%)
CONSTRAINTS: Maximum {max_folds} fold operations.
TARGET: Deployment ratio <= 0.5 (folded area is half or less of original)

CURRENT STATE:
  Sheet: {width}m x {height}m, flat (0 folds applied)
  Bounding box: {width}m x {height}m x 0.0m

Write a fold_strategy(paper_state) function that returns a list of fold operations.
Each fold: {{"type": "valley"|"mountain", "line": {{"start": [x,y], "end": [x,y]}}, "angle": 0-180}}

```python
def fold_strategy(paper_state):
    # Your code here
    return [...]
```""",
        "target_ratio": 0.5,
        "max_folds": 3,
    },

    "letter_fold": {
        "name": "letter_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m {material} sheet into thirds (like a letter).

MATERIAL: {material} (thickness: {thickness_mm}mm, max strain: {max_strain_pct}%)
CONSTRAINTS: Maximum {max_folds} fold operations.
TARGET: Deployment ratio <= 0.33

CURRENT STATE:
  Sheet: {width}m x {height}m, flat (0 folds applied)

Write a fold_strategy(paper_state) function that returns a list of fold operations.
Each fold: {{"type": "valley"|"mountain", "line": {{"start": [x,y], "end": [x,y]}}, "angle": 0-180}}

```python
def fold_strategy(paper_state):
    # Your code here
    return [...]
```""",
        "target_ratio": 0.33,
        "max_folds": 5,
    },

    "solar_panel": {
        "name": "solar_panel",
        "prompt": """\
TASK: Fold a {width}m x {height}m Mylar sheet to minimize packed volume for a solar panel.
The folded panel must be deployable (unfold cleanly to near-original area).

MATERIAL: Mylar (thickness: 0.05mm, Young's modulus: 4 GPa, max strain: 3%)
CONSTRAINTS:
  - Maximum {max_folds} fold operations
  - Must pack into bounding box <= 15cm x 15cm x 5cm
  - Must deploy to >= 80% of original area
  - No self-intersections

TARGET: Deployment ratio <= 0.05 (95% volume reduction)

CURRENT STATE:
  Sheet: {width}m x {height}m, flat (0 folds applied)
  Bounding box: {width}m x {height}m x 0.0m

HINT: Consider tessellated patterns like Miura-ori — alternating mountain and valley
folds in a grid create a highly compact, single-DOF deployable structure.

Write a fold_strategy(paper_state) function that returns a list of fold operations.
Each fold: {{"type": "valley"|"mountain", "line": {{"start": [x,y], "end": [x,y]}}, "angle": 0-180}}

```python
def fold_strategy(paper_state):
    # Your code here
    return [...]
```""",
        "target_ratio": 0.05,
        "max_folds": 20,
    },

    "stent_fold": {
        "name": "stent_fold",
        "prompt": """\
TASK: Fold a {width}m x {height}m Nitinol sheet into a compact cylinder for a medical stent.

MATERIAL: Nitinol (thickness: 0.1mm, Young's modulus: 75 GPa, max strain: 8%)
CONSTRAINTS:
  - Maximum {max_folds} fold operations
  - Compressed diameter: 3mm
  - Deployed diameter: 10mm
  - Must be radially deployable

TARGET: Minimize packed cross-section while maintaining deployability.

Write a fold_strategy(paper_state) function that returns a list of fold operations.

```python
def fold_strategy(paper_state):
    # Your code here
    return [...]
```""",
        "target_ratio": 0.1,
        "max_folds": 15,
    },
}


# Default task configs for each level
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
    return task["prompt"].format(**config)


def get_task_target_ratio(task_name: str) -> float:
    return TASK_TEMPLATES[task_name]["target_ratio"]


def get_task_max_folds(task_name: str) -> int:
    return TASK_TEMPLATES[task_name]["max_folds"]
