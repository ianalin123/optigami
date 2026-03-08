from openenv_runtime.environment import OpenEnvOrigamiEnvironment
from openenv_runtime.models import OrigamiAction, OrigamiFold, OrigamiObservation


def test_openenv_reset_returns_observation():
    env = OpenEnvOrigamiEnvironment(default_mode="step", max_steps=8)
    obs = env.reset(target_name="half_horizontal", episode_id="ep-1")

    assert isinstance(obs, OrigamiObservation)
    assert obs.done is False
    assert obs.target_name == "half_horizontal"
    assert "prompt" in obs.model_fields_set


def test_openenv_step_single_fold_completes_simple_target():
    env = OpenEnvOrigamiEnvironment(default_mode="step", max_steps=8)
    env.reset(target_name="half_horizontal")

    action = OrigamiAction(
        mode="single",
        fold=OrigamiFold(
            from_point=[0.0, 0.5],
            to_point=[1.0, 0.5],
            assignment="V",
            instruction="Valley fold along horizontal center line",
        ),
    )
    obs = env.step(action)

    assert obs.reward is not None
    assert obs.reward > 1.0
    assert obs.done is True
    assert obs.reward_components.get("completion", 0.0) >= 10.0


def test_openenv_step_sequence_mode_executes_completion():
    env = OpenEnvOrigamiEnvironment(default_mode="step", max_steps=8)
    env.reset(target_name="half_vertical")

    completion = (
        '<folds>[{"instruction": "Mountain fold vertical center", '
        '"from": [0.5, 0.0], "to": [0.5, 1.0], "assignment": "M"}]</folds>'
    )

    obs = env.step(OrigamiAction(mode="sequence", completion=completion))

    assert obs.done is True
    assert obs.reward is not None
    assert obs.reward > 1.0


def test_openenv_state_contains_targets_and_step_count():
    env = OpenEnvOrigamiEnvironment(default_mode="step", max_steps=8)
    env.reset(target_name="half_horizontal", episode_id="ep-state")

    state = env.state

    assert state.episode_id == "ep-state"
    assert state.step_count == 0
    assert "half_horizontal" in state.available_targets
