"""
Task pool and curriculum for the origami RL environment.

7 tasks across 4 difficulty levels.
"""
from __future__ import annotations

import random
from typing import Optional


TASKS: dict[str, dict] = {
    "half_fold": {
        "name": "half_fold",
        "description": "Fold a 1x1 paper sheet in half along the horizontal midline.",
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.50,
        "max_folds": 3,
        "target_box": [1.0, 0.5, 0.02],
        "must_deploy": False,
        "difficulty": 1,
    },
    "quarter_fold": {
        "name": "quarter_fold",
        "description": "Fold a 1x1 paper sheet into quarters using two perpendicular folds.",
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.25,
        "max_folds": 5,
        "target_box": [0.5, 0.5, 0.04],
        "must_deploy": False,
        "difficulty": 1,
    },
    "letter_fold": {
        "name": "letter_fold",
        "description": "Fold a 1x1 paper into thirds (letter fold) using two parallel folds.",
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.33,
        "max_folds": 5,
        "target_box": [1.0, 0.34, 0.03],
        "must_deploy": False,
        "difficulty": 2,
    },
    "map_fold": {
        "name": "map_fold",
        "description": "Fold a 1x1 paper into eighths using a grid fold pattern. Must be re-deployable.",
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.125,
        "max_folds": 8,
        "target_box": [0.5, 0.25, 0.08],
        "must_deploy": True,
        "difficulty": 2,
    },
    "solar_panel": {
        "name": "solar_panel",
        "description": "Pack a 1x1 Mylar solar panel into a compact configuration using a Miura-ori style fold. Must deploy.",
        "width": 1.0,
        "height": 1.0,
        "material": "mylar",
        "target_ratio": 0.05,
        "max_folds": 20,
        "target_box": [0.25, 0.25, 0.05],
        "must_deploy": True,
        "difficulty": 3,
    },
    "shelter_wall": {
        "name": "shelter_wall",
        "description": "Fold a 1x1 aluminum sheet into a compact structural panel within strain limits.",
        "width": 1.0,
        "height": 1.0,
        "material": "aluminum",
        "target_ratio": 0.10,
        "max_folds": 15,
        "target_box": [0.5, 0.25, 0.1],
        "must_deploy": False,
        "difficulty": 3,
    },
    "stent": {
        "name": "stent",
        "description": "Fold a 0.5x1.5 nitinol sheet into a compact tube configuration for a medical stent. Superelastic material.",
        "width": 0.5,
        "height": 1.5,
        "material": "nitinol",
        "target_ratio": 0.09,
        "max_folds": 25,
        "target_box": [0.1, 0.1, 0.15],
        "must_deploy": True,
        "difficulty": 4,
    },
}


def get_task_by_name(name: str) -> Optional[dict]:
    """Return task dict by name, or None if not found."""
    return TASKS.get(name)


def sample_task(seed: Optional[int] = None, difficulty: Optional[int] = None) -> dict:
    """Sample a random task, optionally filtered by difficulty level."""
    rng = random.Random(seed)
    pool = list(TASKS.values())
    if difficulty is not None:
        pool = [t for t in pool if t["difficulty"] == difficulty]
    if not pool:
        pool = list(TASKS.values())
    return dict(rng.choice(pool))


def get_tasks_by_difficulty(level: int) -> list[dict]:
    """Return all tasks at a given difficulty level."""
    return [dict(t) for t in TASKS.values() if t["difficulty"] == level]


def available_task_names() -> list[str]:
    """Return sorted list of all task names."""
    return sorted(TASKS.keys())
