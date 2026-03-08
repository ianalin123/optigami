FROM node:20-alpine AS web-builder

WORKDIR /web
COPY package*.json ./
RUN npm install --no-audit --no-fund
COPY public ./public
COPY src ./src
RUN npm run build

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY . /app
COPY --from=web-builder /web/build /app/build

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "openenv-core[core]>=0.2.1"

ENV ENABLE_WEB_INTERFACE=false

CMD ["uvicorn", "openenv_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
