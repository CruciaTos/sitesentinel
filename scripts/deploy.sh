#!/usr/bin/env bash
# Deploy the docker compose stack (backend, simulator, Prometheus, Grafana) with the given backend
# image, remembering the previously running image so scripts/rollback.sh can go back to it.
# Usage: bash scripts/deploy.sh IMAGE TAG
#   e.g. bash scripts/deploy.sh localhost:5000/sitesentinel-backend sha-1a2b3c4
set -euo pipefail

IMAGE="${1:?usage: deploy.sh IMAGE TAG}"
TAG="${2:?usage: deploy.sh IMAGE TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.deploy-state"
cd "$ROOT"
mkdir -p "$STATE"

container="$(docker compose ps -q backend 2>/dev/null || true)"
if [ -n "$container" ]; then
  previous="$(docker inspect --format '{{.Config.Image}}' "$container")"
  echo "Currently running backend image: $previous"
  if [ "$previous" != "$IMAGE:$TAG" ]; then
    echo "$previous" > "$STATE/previous_image"
  fi
else
  echo "No backend running yet."
fi

echo "Building monitoring images"
docker compose build prometheus grafana

echo "Deploying $IMAGE:$TAG"
export IMAGE TAG
# --wait fails the deploy if the backend's HEALTHCHECK does not pass within the timeout.
docker compose up -d --no-build --remove-orphans --wait --wait-timeout 120
echo "$IMAGE:$TAG" > "$STATE/current_image"

if health="$(docker compose exec -T backend wget -qO- http://127.0.0.1:8080/healthz 2>/dev/null)"; then
  echo "Backend /healthz: $health"
fi
echo "DEPLOYED $IMAGE:$TAG"
