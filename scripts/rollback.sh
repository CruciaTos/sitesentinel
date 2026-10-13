#!/usr/bin/env bash
# Roll the compose deployment back to the previous backend image (or to the IMAGE:TAG given).
# Usage: bash scripts/rollback.sh [IMAGE:TAG]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREVIOUS_FILE="$ROOT/.deploy-state/previous_image"

target="${1:-}"
if [ -z "$target" ] && [ -s "$PREVIOUS_FILE" ]; then
  target="$(head -n 1 "$PREVIOUS_FILE")"
fi
if [ -z "$target" ]; then
  echo "Nothing to roll back to: pass IMAGE:TAG or deploy at least twice with scripts/deploy.sh." >&2
  exit 1
fi

# Split on the last ':' so registry ports (localhost:5000/...) stay in the image name.
image="${target%:*}"
tag="${target##*:}"
if [ "$image" = "$target" ] || [ -z "$tag" ] || [[ "$tag" == */* ]]; then
  echo "Cannot roll back to '$target': expected IMAGE:TAG." >&2
  exit 1
fi

echo "ROLLING BACK to $image:$tag"
bash "$ROOT/scripts/deploy.sh" "$image" "$tag"
