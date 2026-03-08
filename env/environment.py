import json
import os
import copy
from pathlib import Path
from typing import Optional

from .paper_state import PaperState
from .rewards import compute_reward, compute_terminal_reward, load_target, target_crease_edges
from .prompts import (
    code_as_policy_prompt,
    step_level_prompt,
    parse_fold_list,
    parse_single_fold,
)
from .verifier import check_all_vertices


TARGETS_DIR = Path(__file__).parent / 'targets'


class OrigamiEnvironment:
    """
    OpenEnv-compatible origami crease pattern environment.

    Supports two modes:
    - code_as_policy: model outputs complete fold sequence, gets terminal reward
    - step: model outputs one fold at a time, gets per-step reward
    """

    def __init__(
        self,
        mode: str = 'code_as_policy',  # 'code_as_policy' or 'step'
        max_steps: int = 8,
        targets_dir: Optional[str] = None,
    ):
        assert mode in ('code_as_policy', 'step'), f"Unknown mode: {mode}"
        self.mode = mode
        self.max_steps = max_steps
        self.targets_dir = Path(targets_dir) if targets_dir else TARGETS_DIR

        self.paper: Optional[PaperState] = None
        self.target: Optional[dict] = None
        self.target_name: Optional[str] = None
        self.step_count: int = 0
        self.last_reward: Optional[dict] = None

        # Cache all available targets
        self._targets = self._load_all_targets()

    def _load_all_targets(self) -> dict[str, dict]:
        targets = {}
        for fold_file in self.targets_dir.glob('*.fold'):
            with open(fold_file) as f:
                targets[fold_file.stem] = json.load(f)
        return targets

    def available_targets(self) -> list[str]:
        return sorted(self._targets.keys())

    def reset(self, target_name: Optional[str] = None) -> dict:
        """
        Reset environment to start of a new episode.

        Args:
            target_name: name of target (stem of .fold file). If None, picks level-1 randomly.

        Returns:
            observation dict with 'prompt' key containing the LLM prompt string.
        """
        import random

        if target_name:
            assert target_name in self._targets, f"Unknown target: {target_name}"
            self.target_name = target_name
        else:
            # Default to level-1 targets
            level1 = [k for k, v in self._targets.items() if v.get('level', 1) == 1]
            self.target_name = random.choice(level1 if level1 else list(self._targets.keys()))

        self.target = self._targets[self.target_name]
        self.paper = PaperState()
        self.step_count = 0
        self.last_reward = None

        return self._get_observation()

    def step(self, action) -> tuple[dict, dict, bool, dict]:
        """
        Execute an action.

        In code_as_policy mode: action is a string (model completion with <folds> tags)
            OR a list of fold dicts already parsed.
        In step mode: action is a string (single fold JSON) or dict.

        Returns:
            (observation, reward, done, info)
        """
        if self.mode == 'code_as_policy':
            return self._step_sequence(action)
        else:
            return self._step_single(action)

    def _step_sequence(self, action) -> tuple[dict, dict, bool, dict]:
        """Execute a complete fold sequence (code-as-policy mode)."""
        # Parse action if it's a string
        if isinstance(action, str):
            try:
                folds = parse_fold_list(action)
            except ValueError as e:
                bad_reward = {'format': 0.0, 'total': -0.1, 'error': str(e)}
                return self._get_observation(), bad_reward, True, self._info()
        else:
            folds = action  # already a list of dicts

        # Execute each fold sequentially
        last_result = {'valid': True, 'anchored': True, 'new_vertices': [], 'errors': []}
        for fold in folds:
            try:
                p1 = fold['from']
                p2 = fold['to']
                assignment = fold['assignment']
            except (KeyError, TypeError) as e:
                last_result = {'valid': False, 'anchored': False, 'new_vertices': [], 'errors': [str(e)]}
                break

            last_result = self.paper.add_crease(p1, p2, assignment)
            self.step_count += 1
            if not last_result['valid']:
                break  # stop at first invalid fold, partial credit

        reward = compute_terminal_reward(self.paper, self.target)
        self.last_reward = reward
        return self._get_observation(), reward, True, self._info()

    def _step_single(self, action) -> tuple[dict, dict, bool, dict]:
        """Execute a single fold (step mode)."""
        if isinstance(action, str):
            try:
                fold = parse_single_fold(action)
            except ValueError as e:
                bad_reward = {'format': 0.0, 'total': -0.1, 'error': str(e)}
                self.last_reward = bad_reward
                done = self.step_count >= self.max_steps
                return self._get_observation(), bad_reward, done, self._info()
        else:
            fold = action

        try:
            p1 = fold['from']
            p2 = fold['to']
            assignment = fold['assignment']
        except (KeyError, TypeError) as e:
            bad_reward = {'format': 0.0, 'total': -0.1, 'error': str(e)}
            self.last_reward = bad_reward
            done = self.step_count >= self.max_steps
            return self._get_observation(), bad_reward, done, self._info()

        result = self.paper.add_crease(p1, p2, assignment)
        self.step_count += 1

        reward = compute_reward(self.paper, result, self.target)
        self.last_reward = reward

        done = (
            self.step_count >= self.max_steps or
            reward.get('completion', 0) > 0
        )
        return self._get_observation(), reward, done, self._info()

    def _get_observation(self) -> dict:
        """Returns observation dict with the LLM prompt and raw state."""
        if self.mode == 'code_as_policy':
            prompt = code_as_policy_prompt(self.target, max_folds=self.max_steps)
        else:
            prompt = step_level_prompt(
                target=self.target,
                paper_state=self.paper,
                step=self.step_count,
                max_steps=self.max_steps,
                last_reward=self.last_reward,
            )

        return {
            'prompt': prompt,
            'target_name': self.target_name,
            'step': self.step_count,
            'paper_fold_json': self.paper.graph.edges if self.paper else {},
        }

    def _info(self) -> dict:
        """Returns diagnostic info dict for logging."""
        if self.paper is None:
            return {}

        interior = self.paper.graph.interior_vertices()
        vertex_scores = check_all_vertices(self.paper.graph)

        return {
            'local_foldability': (
                vertex_scores['kawasaki'] == 1.0 and
                vertex_scores['maekawa'] == 1.0
            ),
            'blb_satisfied': vertex_scores['blb'] == 1.0,
            'global_foldability': 'not_checked',  # NP-complete (Bern-Hayes 1996)
            'n_interior_vertices': len(interior),
            'n_creases': len(self.paper.graph.crease_edges()),
            'target_name': self.target_name,
        }

    def state(self) -> dict:
        """Returns current environment state for logging/inspection."""
        return {
            'paper': {
                'vertices': dict(self.paper.graph.vertices),
                'edges': {
                    k: v for k, v in self.paper.graph.edges.items()
                    if v[2] in ('M', 'V')
                },
                'fold_history': self.paper.fold_history,
            },
            'target': self.target_name,
            'step': self.step_count,
            'mode': self.mode,
        }

    def close(self):
        """Cleanup."""
        pass

    def clone(self) -> 'OrigamiEnvironment':
        """Return a deep copy for parallel evaluation (used in GRPO)."""
        new_env = OrigamiEnvironment(
            mode=self.mode,
            max_steps=self.max_steps,
            targets_dir=str(self.targets_dir),
        )
        if self.paper is not None:
            new_env.paper = copy.deepcopy(self.paper)
        new_env.target = self.target
        new_env.target_name = self.target_name
        new_env.step_count = self.step_count
        new_env.last_reward = self.last_reward
        return new_env
