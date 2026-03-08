FROM node:20-alpine AS web-builder

WORKDIR /web
COPY package*.json ./
RUN npm ci --no-audit --no-fund
COPY public ./public
COPY src ./src
RUN npm run build

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "openenv-core[core]>=0.2.1"

# Copy application source
COPY . /app

# Overlay the compiled React frontend
COPY --from=web-builder /web/build /app/build

EXPOSE 8000

CMD ["uvicorn", "openenv_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
