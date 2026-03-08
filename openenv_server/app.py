from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from openenv_runtime.environment import OpenEnvOrigamiEnvironment
from openenv_runtime.models import OrigamiAction, OrigamiObservation


app = create_app(
    env=lambda: OpenEnvOrigamiEnvironment(),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="optigami",
)
