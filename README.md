---
title: Optigami
emoji: 🐠
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 8000
license: mit
short_description: OpenEnv origami environment and demo
---

# Optigami

OpenEnv-compatible origami RL environment with:
- environment + reward checks in `env/`
- OpenEnv server adapter in `openenv_runtime/` and `openenv_server/`
- Dockerized deployment for Hugging Face Spaces

Entry point: `openenv_server.app:app`  
Manifest: `openenv.yaml`  
Container: `Dockerfile`

## Local Run

```bash
uvicorn openenv_server.app:app --host 0.0.0.0 --port 8000
```

## Frontend (optional local React demo)

```bash
npm install
npm start
```

This serves the dashboard against the FastAPI API.
