# syntax=docker/dockerfile:1

# ==========================================================================
# ai-gateway-provider — OpenAI/Anthropic-compatible API gateway
#
#   Build:  docker build -t ai-gateway-provider .
#   Run:    docker run --env-file .env -p 1221:1221 ai-gateway-provider
# ==========================================================================

# --------------------------------------------------------------------------
# Stage 1: builder — install uv and create a fully synced virtualenv
# --------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

# Install uv (single binary) into the builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

WORKDIR /app

# Compile bytecode, and copy instead of symlink into the venv (container-friendly)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Install only dependencies first — this layer is cached until
#    pyproject.toml / uv.lock change, so rebuilds stay fast.
COPY pyproject.toml uv.lock ./
RUN /uv/bin/uv sync --frozen --no-dev --no-install-project

# 2) Copy the full source (README needed by hatchling to build the
#    project wheel), then install the project itself into the venv.
COPY . .
RUN /uv/bin/uv sync --frozen --no-dev

# --------------------------------------------------------------------------
# Stage 2: runtime — lean image; app is started with `uv run`
# --------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Bring uv along so the image can be started with `uv run` as requested
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the synced venv plus everything `uv run` needs to verify sync state
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/uv.lock /app/uv.lock
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/gateway /app/gateway
COPY --from=builder /app/providers /app/providers
COPY --from=builder /app/config /app/config
# Web UI (dashboard served by gateway/routes/ui.py)
COPY --from=builder /app/tool /app/tool
# Top-level modules imported by gateway/app.py (registry/router) and skeletons
COPY --from=builder /app/registry.py /app/registry.py
COPY --from=builder /app/router.py /app/router.py
COPY --from=builder /app/scheduler.py /app/scheduler.py
COPY --from=builder /app/judge.py /app/judge.py

# Run as non-root. The app writes runtime data (chat history -> /app/data,
# freebuff tokens -> /app/config), so both dirs must exist and be owned by
# appuser. Docker named volumes (see docker-compose.yml) inherit this
# ownership on first mount, so the container can write to them out of the box.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 1221 2222

# `uv run` verifies the lockfile is in sync, then executes the console script
# (main:main -> uvicorn). Env vars come from --env-file / -e at run time.
CMD ["uv", "run", "ai-gateway"]
