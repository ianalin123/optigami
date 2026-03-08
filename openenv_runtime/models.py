from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from openenv.core.env_server.types import Action, Observation, State


class OrigamiFold(BaseModel):
    """Single fold action payload for step-level execution."""

    from_point: list[float] = Field(..., description="Fold line start [x, y]")
    to_point: list[float] = Field(..., description="Fold line end [x, y]")
    assignment: Literal["M", "V"] = Field(..., description="Mountain or valley")
    instruction: str = Field(default="", description="Optional natural language instruction")

    @field_validator("from_point", "to_point")
    @classmethod
    def _validate_point(cls, point: list[float]) -> list[float]:
        if len(point) != 2:
            raise ValueError("Point must contain exactly 2 coordinates")
        return [float(point[0]), float(point[1])]


class OrigamiAction(Action):
    """
    OpenEnv action for Optigami.

    Modes:
    - single: execute one fold (pass `fold` or JSON `completion` for a single-fold object)
    - sequence: execute a full <folds>[...]</folds> completion in one step
    """

    mode: Literal["single", "sequence"] = Field(default="single")
    fold: Optional[OrigamiFold] = Field(default=None)
    completion: Optional[str] = Field(default=None)
    target_name: Optional[str] = Field(
        default=None,
        description="Optional target override; reset to this target before stepping",
    )


class OrigamiObservation(Observation):
    """OpenEnv observation payload returned by Optigami."""

    prompt: str = Field(default="")
    target_name: Optional[str] = Field(default=None)
    step: int = Field(default=0)
    paper_state: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)
    reward_components: dict[str, float | int | str] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)


class OrigamiState(State):
    """OpenEnv state payload for Optigami."""

    mode: str = Field(default="step")
    target_name: Optional[str] = Field(default=None)
    paper: dict[str, Any] = Field(default_factory=dict)
    last_reward: dict[str, Any] = Field(default_factory=dict)
    available_targets: list[str] = Field(default_factory=list)
