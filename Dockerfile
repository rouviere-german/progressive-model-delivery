# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="progressive-model-delivery" \
      org.opencontainers.image.licenses="MIT"
# LightGBM and scikit-learn dlopen the OpenMP runtime when they are imported, and python:*-slim
# does not ship it. Without libgomp1 every command that touches a model -- bootstrap, demo, and
# the router the first time it loads a release -- dies on
# `OSError: libgomp.so.1: cannot open shared object file`. Plain `serve` survives, because the
# import is lazy, which is exactly why a healthz-only smoke test misses this.
# libgomp1 is deliberately not version-pinned: the base image is digest-pinned but apt still
# resolves against a live Debian mirror, so an exact pin breaks the build the day the package is
# superseded.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
# The Settings defaults are relative (`var/`, `var/control.sqlite`, ...), which is right for a
# source checkout and wrong here: WORKDIR is /app, and every hardened deployment runs with a
# read-only root filesystem, so the router died on `OSError: Read-only file system: 'var'` before
# it ever bound a port. Point the state paths at one absolute directory -- the same
# /var/lib/pmd docker-compose.yml mounts -- and own it as the app user, so the image needs only a
# writable volume (or tmpfs) there and no environment at all to start.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_DATA_DIR=/var/lib/pmd \
    APP_REGISTRY_DIR=/var/lib/pmd/registry \
    APP_CONTROL_DB=/var/lib/pmd/control.sqlite \
    APP_PREDLOG_DIR=/var/lib/pmd/predlog
RUN install --directory --owner=10001 --group=10001 --mode=0750 /var/lib/pmd
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "progressive_model_delivery.core.healthcheck"]
ENTRYPOINT ["python", "-m", "progressive_model_delivery"]
CMD ["serve"]
