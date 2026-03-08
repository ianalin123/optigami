"""
OrigamiEnvironment — OpenEnv environment wrapping the origami physics engine.

Implements reset() / step() / state following the OpenEnv interface.
Engine (physics, fold, validation, metrics) lives in engine/.
No server-side image rendering — paper_state contains all geometry data.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from engine.paper import Paper
from engine.fold_engine import apply_fold
from engine.physics import simulate
from engine.validation import validate_state
from engine.metrics import compute_all_metrics
from server.models import OrigamiAction, OrigamiObservation, OrigamiState
from server.tasks import get_task_by_name, sample_task


def _get_material(name: str):
    """Get material by name, falling back to paper."""
    try:
        from engine.materials import get_material
        return get_material(name)
    except Exception:
        from engine.materials import get_material
        return get_material("paper")


class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    """Origami folding RL environment.

    Each episode: agent receives paper_state + task, applies folds one at a
    time via step(), receives metrics + reward, ends with 'stop' action or
    when max_folds is reached.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paper: Optional[Paper] = None
        self._task: Optional[dict] = None
        self._fold_history: list[dict] = []
        self._metrics: dict = {}
        self._validation: dict = {}
        self._error: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._fold_history = []
        self._error = None
        self._total_reward = 0.0

        # Select task
        task_name = kwargs.get("task_name")
        if task_name:
            self._task = get_task_by_name(task_name)
        if not self._task:
            self._task = sample_task(seed=seed)

        # Create flat sheet
        mat = _get_material(self._task["material"])
        self._paper = Paper.create_flat_sheet(
            width=self._task["width"],
            height=self._task["height"],
            material=mat,
        )

        # Initial validation + metrics (no physics needed for flat sheet)
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        return self._make_observation(done=False, reward=None)

    # ── step ──────────────────────────────────────────────────────────

    def step(
        self,
        action: OrigamiAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        if self._paper is None or self._task is None:
            return self._make_observation(done=True, reward=-5.0)

        self._step_count += 1
        self._error = None

        # ── Stop action ───────────────────────────────────────────────
        if action.fold_type == "stop":
            return self._finalize_episode()

        # ── Build fold dict ───────────────────────────────────────────
        fold_dict = {
            "type": action.fold_type,
            "line": action.fold_line,
            "angle": action.fold_angle,
        }

        # ── Apply fold ────────────────────────────────────────────────
        new_paper, err = apply_fold(self._paper, fold_dict)
        if err:
            self._error = err
            return self._make_observation(done=True, reward=-5.0)

        self._paper = new_paper
        self._fold_history.append({**fold_dict, "step": self._step_count})

        # ── Physics relaxation ────────────────────────────────────────
        try:
            self._paper = simulate(self._paper, fold_percent=1.0)
        except Exception as exc:
            self._error = f"Physics failed: {exc}"
            # Continue — don't abort episode on physics failure

        # ── Validate ──────────────────────────────────────────────────
        self._validation = validate_state(self._paper)

        # ── Metrics ───────────────────────────────────────────────────
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # ── Check termination ─────────────────────────────────────────
        max_folds = self._task.get("max_folds", 50)
        if self._step_count >= max_folds:
            return self._finalize_episode()

        if self._validation.get("self_intersections", 0) > 0:
            self._error = "Self-intersection detected"
            return self._finalize_episode()

        return self._make_observation(done=False, reward=None)

    # ── state ─────────────────────────────────────────────────────────

    @property
    def state(self) -> OrigamiState:
        return OrigamiState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task.get("name", "") if self._task else "",
            num_folds_applied=len(self._fold_history),
            is_valid=self._metrics.get("is_valid", True),
            total_reward=self._total_reward,
        )

    # ── internals ─────────────────────────────────────────────────────

    def _finalize_episode(self) -> OrigamiObservation:
        reward = self._compute_reward()
        self._total_reward = reward
        return self._make_observation(done=True, reward=reward)

    def _make_observation(self, done: bool, reward: Optional[float]) -> OrigamiObservation:
        return OrigamiObservation(
            done=done,
            reward=reward,
            task=self._task or {},
            paper_state=self._paper.to_observation_dict() if self._paper else {},
            metrics=self._metrics,
            fold_history=self._fold_history,
            error=self._error,
        )

    def _compute_reward(self) -> float:
        m = self._metrics
        reward = 0.0

        # Compactness is the main signal
        reward += m.get("compactness", 0.0) * 20.0

        # Bonus for fitting in target box
        if m.get("fits_target_box", False):
            reward += 10.0

        # Bonus for deployability (if task requires it)
        if m.get("is_deployable", False):
            reward += 5.0

        # Penalties for violations
        reward -= m.get("kawasaki_violations", 0) * 2.0
        reward -= m.get("maekawa_violations", 0) * 2.0
        reward -= m.get("self_intersections", 0) * 5.0

        # Penalty for too many folds (encourage efficiency)
        reward -= m.get("fold_count", 0) * 0.5

        # Penalty for exceeding material strain limit
        max_strain = m.get("max_strain", 0.0)
        strain_limit = self._paper.material.max_strain if self._paper else 0.05
        if max_strain > strain_limit:
            reward -= 3.0 * (max_strain / strain_limit)

        return float(reward)
