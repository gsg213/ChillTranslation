# Dev image: Python + uv + command-line tools. Nothing project-specific is baked in:
# the virtualenv lives in a named volume (/opt/venv), so you rarely need to rebuild.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl make bzip2 unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.7

# Run as your host user, so files created in the mounted repo belong to you, not root.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -o -g "${HOST_GID}" dev \
 && useradd -o -m -u "${HOST_UID}" -g "${HOST_GID}" dev \
 && mkdir -p /opt/venv /opt/uv-cache /home/dev/.cache/huggingface \
 && chown -R dev:dev /opt/venv /opt/uv-cache /home/dev/.cache

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

USER dev
WORKDIR /app
