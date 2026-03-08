from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from env.environment import OrigamiEnvironment

from .models import OrigamiAction, OrigamiObservation, OrigamiState


class OpenEnvOrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    """OpenEnv adapter over the existing OrigamiEnvironment implementation."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        default_mode: str = "step",
        max_steps: int = 8,
        targets_dir: Optional[str] = None,
    ):
        super().__init__()
        self.default_mode = default_mode
        self.max_steps = max_steps
        self.targets_dir = targets_dir
        self._env: Optional[OrigamiEnvironment] = None
        self._episode_id: Optional[str] = None

    def _new_env(self, mode: Optional[str] = None) -> OrigamiEnvironment:
        return OrigamiEnvironment(
            mode=mode or self.default_mode,
            max_steps=self.max_steps,
            targets_dir=self.targets_dir,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        del seed  # deterministic seed plumbing can be added later

        mode = kwargs.get("mode", self.default_mode)
        target_name = kwargs.get("target_name")

        self._env = self._new_env(mode=mode)
        self._episode_id = episode_id
        obs_dict = self._env.reset(target_name=target_name)

        return OrigamiObservation(
            done=False,
            reward=None,
            metadata={"available_targets": self._env.available_targets()},
            prompt=obs_dict.get("prompt", ""),
            target_name=obs_dict.get("target_name"),
            step=obs_dict.get("step", 0),
            paper_state=self._paper_state_snapshot(),
            info=self._env._info(),
            reward_components={},
        )

    def step(
        self,
        action: OrigamiAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        del timeout_s, kwargs

        if self._env is None:
            self.reset(target_name=action.target_name)

        assert self._env is not None

        if action.target_name and action.target_name != self._env.target_name:
            self.reset(target_name=action.target_name, mode=self._env.mode)

        try:
            if action.mode == "sequence":
                if not action.completion:
                    return self._error_observation("sequence mode requires completion")

                seq_env = self._new_env(mode="code_as_policy")
                seq_env.reset(target_name=self._env.target_name)
                obs_dict, reward_dict, done, info = seq_env.step(action.completion)
                self._env = seq_env
            else:
                if action.fold is not None:
                    fold_payload = {
                        "from": list(action.fold.from_point),
                        "to": list(action.fold.to_point),
                        "assignment": action.fold.assignment,
                        "instruction": action.fold.instruction,
                    }
                    env_action: Any = fold_payload
                elif action.completion:
                    env_action = action.completion
                else:
                    return self._error_observation("single mode requires fold or completion")

                obs_dict, reward_dict, done, info = self._env.step(env_action)

            total = reward_dict.get("total") if isinstance(reward_dict, dict) else None
            return OrigamiObservation(
                done=bool(done),
                reward=float(total) if isinstance(total, (int, float)) else None,
                metadata={"target_name": self._env.target_name},
                prompt=obs_dict.get("prompt", ""),
                target_name=obs_dict.get("target_name", self._env.target_name),
                step=obs_dict.get("step", self._env.step_count),
                paper_state=self._paper_state_snapshot(),
                info=info or {},
                reward_components=reward_dict or {},
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return self._error_observation(str(exc))

    @property
    def state(self) -> OrigamiState:
        if self._env is None:
            tmp_env = self._new_env(mode=self.default_mode)
            return OrigamiState(
                episode_id=self._episode_id,
                step_count=0,
                mode=tmp_env.mode,
                target_name=None,
                paper={},
                last_reward={},
                available_targets=tmp_env.available_targets(),
            )

        env_state = self._env.state()
        return OrigamiState(
            episode_id=self._episode_id,
            step_count=env_state.get("step", self._env.step_count),
            mode=env_state.get("mode", self._env.mode),
            target_name=env_state.get("target", self._env.target_name),
            paper=env_state.get("paper", {}),
            last_reward=self._env.last_reward or {},
            available_targets=self._env.available_targets(),
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _paper_state_snapshot(self) -> dict[str, Any]:
        if self._env is None or self._env.paper is None:
            return {"vertices": {}, "edges": [], "anchor_points": []}

        graph = self._env.paper.graph
        return {
            "vertices": {str(k): [float(v[0]), float(v[1])] for k, v in graph.vertices.items()},
            "edges": [
                {
                    "id": int(eid),
                    "v1": [float(graph.vertices[v1][0]), float(graph.vertices[v1][1])],
                    "v2": [float(graph.vertices[v2][0]), float(graph.vertices[v2][1])],
                    "assignment": assignment,
                }
                for eid, (v1, v2, assignment) in graph.edges.items()
            ],
            "anchor_points": [
                [float(x), float(y)] for (x, y) in self._env.paper.anchor_points()
            ],
        }

    def _error_observation(self, message: str) -> OrigamiObservation:
        return OrigamiObservation(
            done=False,
            reward=-0.1,
            metadata={"error": True},
            prompt="",
            target_name=self._env.target_name if self._env else None,
            step=self._env.step_count if self._env else 0,
            paper_state=self._paper_state_snapshot(),
            info=self._env._info() if self._env else {},
            reward_components={"format": 0.0, "total": -0.1, "error": message},
            error=message,
        )
