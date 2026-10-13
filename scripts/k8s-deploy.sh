#!/usr/bin/env bash
# Deploy SiteSentinel to the current kubectl context, smoke test it through a port-forward,
# and roll the deployments back if anything fails.
# Usage: bash scripts/k8s-deploy.sh IMAGE:TAG
set -euo pipefail

IMAGE="${1:?usage: k8s-deploy.sh IMAGE:TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE=sitesentinel
LOCAL_PORT="${LOCAL_PORT:-18080}"
DEPLOYMENTS=(sitesentinel-backend sitesentinel-simulator)

# The app reports its image tag as its version (APP_VERSION build-arg).
TAG="${IMAGE##*:}"
case "$TAG" in */*) TAG="" ;; esac
EXPECTED_VERSION="${EXPECTED_VERSION-$TAG}"

pf_pid=""
pf_log="$(mktemp)"
cleanup() {
  if [ -n "$pf_pid" ]; then
    kill "$pf_pid" 2>/dev/null || true
  fi
  rm -f "$pf_log"
}
trap cleanup EXIT

rollback() {
  echo "DEPLOY FAILED: $1"
  for d in "${DEPLOYMENTS[@]}"; do
    echo "Rolling back deployment/$d"
    if kubectl -n "$NAMESPACE" rollout undo "deployment/$d"; then
      kubectl -n "$NAMESPACE" rollout status "deployment/$d" --timeout=120s || true
    else
      echo "No previous revision of deployment/$d to roll back to."
    fi
  done
  exit 1
}

render() {
  for f in "$ROOT"/k8s/*.yaml; do
    [ "$(basename "$f")" = namespace.yaml ] && continue
    echo "---"
    sed "s|__IMAGE__|$IMAGE|g" "$f"
  done
}

echo "Deploying $IMAGE to namespace $NAMESPACE"
kubectl apply -f "$ROOT/k8s/namespace.yaml" || rollback "could not create namespace $NAMESPACE"
render | kubectl apply -f - || rollback "kubectl apply failed"

for d in "${DEPLOYMENTS[@]}"; do
  kubectl -n "$NAMESPACE" rollout status "deployment/$d" --timeout=180s \
    || rollback "deployment/$d did not become ready in 180s"
done

kubectl -n "$NAMESPACE" port-forward svc/sitesentinel-backend "$LOCAL_PORT:8080" > "$pf_log" 2>&1 &
pf_pid=$!

if ! bash "$ROOT/scripts/smoke-test.sh" "http://127.0.0.1:$LOCAL_PORT" "$EXPECTED_VERSION"; then
  echo "--- port-forward log ---"
  cat "$pf_log" || true
  rollback "smoke test failed"
fi

echo "DEPLOYED $IMAGE to namespace $NAMESPACE"
