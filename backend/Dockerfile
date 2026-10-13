# Stages: base (app dependencies) -> test (lint + unit tests) -> runtime (the image we ship).
# `docker build .` builds runtime; `docker build --target test .` runs the checks (used by Jenkins).
FROM python:3.12-alpine AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt


FROM base AS test

COPY requirements-dev.txt .
RUN python3 -m pip install -r requirements-dev.txt
COPY . .
# setup.cfg adds -q, so pytest prints one dot per test and a summary.
RUN flake8 . && pytest


FROM base AS runtime

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}
LABEL org.opencontainers.image.title="sitesentinel-backend" \
      org.opencontainers.image.description="SiteSentinel PPE compliance backend: dashboard, detection API, metrics and demo camera simulator" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="https://github.com/"

COPY . .
# Non-root user; its home directory holds gunicorn's control socket.
RUN rm -rf tests requirements-dev.txt setup.cfg \
    && adduser -D -u 10001 sitesentinel
USER sitesentinel

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/healthz > /dev/null || exit 1

# Exactly one worker: the event store lives in memory.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "app:app"]
