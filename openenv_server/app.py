from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from openenv_runtime.environment import OpenEnvOrigamiEnvironment
from openenv_runtime.models import OrigamiAction, OrigamiObservation


app = create_app(
    env=lambda: OpenEnvOrigamiEnvironment(),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="optigami",
)

_BUILD_DIR = Path(__file__).resolve().parent.parent / "build"

if _BUILD_DIR.exists():
    # Serve compiled React dashboard at "/" while preserving existing OpenEnv API routes.
    app.mount("/", StaticFiles(directory=str(_BUILD_DIR), html=True), name="renderer")
else:
    @app.get("/", include_in_schema=False)
    def missing_renderer_build() -> HTMLResponse:
        return HTMLResponse(
            """
            <html><body style="font-family: sans-serif; margin: 24px;">
            <h3>Renderer build not found</h3>
            <p>No <code>build/</code> directory is present in the container.</p>
            <p>OpenEnv API docs are available at <a href="/docs">/docs</a>.</p>
            </body></html>
            """,
            status_code=200,
        )
