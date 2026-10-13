#!/usr/bin/env bash
# Smoke test a running SiteSentinel backend: health, version, one compliant and one violating
# frame through the model, the dashboard summary and the Prometheus metrics.
# Usage: bash scripts/smoke-test.sh BASE_URL [EXPECTED_VERSION]
set -euo pipefail

BASE_URL="${1:?usage: smoke-test.sh BASE_URL [EXPECTED_VERSION]}"
BASE_URL="${BASE_URL%/}"
EXPECTED_VERSION="${2:-}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

failures=0
body_file="$(mktemp)"
trap 'rm -f "$body_file"' EXIT

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

# request METHOD PATH [JSON]: prints the HTTP status code, leaves the body in $body_file.
request() {
  local code args=(-s --max-time 10 -o "$body_file" -w '%{http_code}' -X "$1")
  if [ $# -ge 3 ]; then
    args+=(-H 'Content-Type: application/json' --data "$3")
  fi
  : > "$body_file"
  code="$(curl "${args[@]}" "$BASE_URL$2")" || true
  echo "${code:-000}"
}

json_field() {  # json_field NAME: first string value of "NAME" in $body_file
  sed -n "s/.*\"$1\": *\"\\([^\"]*\\)\".*/\\1/p" "$body_file" | head -n 1
}

echo "Smoke testing $BASE_URL${EXPECTED_VERSION:+ (expecting version $EXPECTED_VERSION)}"

# 1. Wait for /healthz.
deadline=$((SECONDS + WAIT_SECONDS))
code="000"
while [ "$SECONDS" -lt "$deadline" ]; do
  code="$(request GET /healthz)"
  [ "$code" = "200" ] && break
  sleep 2
done
if [ "$code" != "200" ]; then
  fail "GET /healthz did not return 200 within ${WAIT_SECONDS}s (last status $code)"
  echo "FAIL: smoke test aborted"
  exit 1
fi
if grep -q '"status": *"ok"' "$body_file"; then
  pass "GET /healthz -> 200, status ok ($(tr -d '\n' < "$body_file"))"
else
  fail "GET /healthz body has no \"status\": \"ok\": $(cat "$body_file")"
fi
if [ -n "$EXPECTED_VERSION" ]; then
  version="$(json_field version)"
  if [ "$version" = "$EXPECTED_VERSION" ]; then
    pass "running version is $version"
  else
    fail "running version is '$version', expected '$EXPECTED_VERSION'"
  fi
fi

# 2. Frames through the model: one person wearing both items, one person wearing neither.
PERSON='[0.2,0.3,0.9,0.5]'
HARDHAT='[0.18,0.36,0.27,0.44]'
VEST='[0.35,0.31,0.6,0.49]'
check_frame() {  # check_frame NAME EXPECTED_STATUS JSON
  local code
  code="$(request POST /v1/frames "$3")"
  if [ "$code" != "201" ]; then
    fail "POST /v1/frames ($1) -> $code, expected 201: $(cat "$body_file")"
  elif grep -q "\"status\": *\"$2\"" "$body_file"; then
    pass "POST /v1/frames ($1) -> 201, status $2"
  else
    fail "POST /v1/frames ($1) did not return status $2: $(cat "$body_file")"
  fi
}
check_frame "compliant" clear \
  "{\"cameraId\":\"smoke-test\",\"persons\":[$PERSON],\"hardhats\":[$HARDHAT],\"vests\":[$VEST]}"
check_frame "violating" violation \
  "{\"cameraId\":\"smoke-test\",\"persons\":[$PERSON],\"hardhats\":[],\"vests\":[]}"

# 3. Dashboard API and metrics.
code="$(request GET /api/summary)"
if [ "$code" = "200" ] && grep -q '"checks_total"' "$body_file"; then
  pass "GET /api/summary -> 200"
else
  fail "GET /api/summary -> $code: $(head -c 300 "$body_file")"
fi

code="$(request GET /metrics)"
if [ "$code" = "200" ] && grep -q '^sitesentinel_checks_total' "$body_file"; then
  pass "GET /metrics -> 200, exposes sitesentinel_checks_total"
else
  fail "GET /metrics -> $code without sitesentinel_checks_total"
fi

if [ "$failures" -gt 0 ]; then
  echo "FAIL: $failures smoke check(s) failed"
  exit 1
fi
echo "PASS: all smoke checks passed"
