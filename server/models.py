"""
OpenEnv Pydantic models for the origami RL environment.

OrigamiAction  — one fold per step
OrigamiObservation — everything the LLM and Three.js viewer need
OrigamiState   — server-side episode tracking
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class OrigamiAction(Action):
    """One fold operation sent by the client each step."""

    fold_type: str = Field(
        default="valley",
        description="'valley' | 'mountain' | 'pleat' | 'crimp' | 'stop'",
    )
    fold_line: dict[str, list[float]] = Field(
        default_factory=lambda: {"start": [0.0, 0.5], "end": [1.0, 0.5]},
        description="{'start': [x, y], 'end': [x, y]} normalized 0-1",
    )
    fold_angle: float = Field(
        default=180.0,
        description="Fold angle in degrees, 0-180",
    )
    layer_select: str = Field(
        default="all",
        description="'all' | 'top' | 'bottom'",
    )


class OrigamiObservation(Observation):
    """Everything the LLM and Three.js viewer need.

    paper_state contains FOLD-compatible geometry + physics data.
    metrics contains all computed quality metrics.
    No render_urls — the browser renders from paper_state directly.
    """

    task: dict[str, Any] = Field(default_factory=dict)
    paper_state: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    fold_history: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


class OrigamiState(State):
    """Server-side episode tracking."""

    task_name: str = Field(default="")
    num_folds_applied: int = Field(default=0)
    is_valid: bool = Field(default=True)
    total_reward: float = Field(default=0.0)
