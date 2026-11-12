# Deep auto-regressive chapkit service. The chapkit-py base ships Python 3.13,
# uv and an empty venv at /app/.venv (on PATH). Multi-arch: jax has wheels for
# both linux/amd64 and linux/arm64, so no platform is pinned.
FROM ghcr.io/dhis2-chap/chapkit-py:latest@sha256:107479a76c4964c08e2125433d7728894d93e2cd5695a51b621b206c299cffd2

USER root
WORKDIR /work
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies first, so this layer caches independently of code changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Commit the image was built from, reported as git_revision on /api/v1/info.
# The publish workflow passes it; locally: --build-arg GIT_REVISION=$(git rev-parse HEAD)
ARG GIT_REVISION=""
ENV GIT_REVISION=${GIT_REVISION}

# The package itself, installed (not editable) into the venv: the runner copies
# /work into a per-job workspace and runs `python -m chapkit_deep_auto_regressive.*`
# there, which must resolve to the installed package.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

COPY main.py MLproject ./

# Writable paths at runtime are /work/data (SQLite) and /tmp (job workspaces,
# and HOME for jax/matplotlib caches); everything else stays read-only.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /work/data \
    && chown app:app /work/data
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
