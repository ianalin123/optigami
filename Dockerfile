FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "openenv-core[core]>=0.2.1"

ENV ENABLE_WEB_INTERFACE=false

CMD ["uvicorn", "openenv_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
