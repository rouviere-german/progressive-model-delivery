#!/usr/bin/env bash
#
# End-to-end check of the deployed topology: router and shadow worker in separate containers on
# separate networks. This is the only test that exercises the real network hop and the real
# isolation boundary; everything else runs the shadow worker in-process (bench/bench_serving.py
# explains why that distinction matters).
#
# It asserts, in order:
#   1. the signed registry is built, and the router is live but *not ready* before a champion
#      exists, because it has nothing it is willing to load;
#   2. a release manager can bootstrap a champion and the router then serves predictions;
#   3. shadow requests really cross the container boundary (the shadow worker's own counter moves);
#   4. the shadow worker has no egress: it cannot open a connection off the internal bridge, and
#      no write access to the control database or the router's prediction log;
#   5. killing the shadow worker does not fail a single champion response, and the undeliverable
#      jobs are counted rather than retried into the request path;
#   6. the image can run the whole offline demo (rollback with audit evidence, then an approved
#      promotion) inside the container.
#
# It runs in CI (.github/workflows/ci.yml#compose-e2e) and needs docker compose, jq and awk.
#
# Usage: scripts/compose_e2e.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ROUTER=http://127.0.0.1:8000
CONTROLLER=http://127.0.0.1:8001

random_token() { head -c 48 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 32; }
sha256() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }

DEPLOYER_KEY="deployer-$(random_token)"
MANAGER_KEY="manager-$(random_token)"

export APP_SHADOW_TOKEN="shadow-$(random_token)"
export APP_SUBJECT_HASH_SECRET="subject-$(random_token)"
export APP_API_KEY_SHA256="$(sha256 "$DEPLOYER_KEY"),$(sha256 "$MANAGER_KEY")"
export APP_ROLE_BINDINGS="$(sha256 "$DEPLOYER_KEY"):dana:deployer,$(sha256 "$MANAGER_KEY"):ravi:release_manager"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

cleanup() {
  echo "--- container logs (tail) ---"
  docker compose logs --tail 40 router shadow controller || true
  docker compose down --volumes --remove-orphans || true
}
trap cleanup EXIT

wait_for() { # wait_for <url> <what>
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 "$1" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  fail "$2 never became reachable at $1"
}

status_of() { # status_of <url>
  curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1"
}

# Sum every exported sample of <name>, probed from inside the container so the shadow worker
# never needs a published port.
metric() { # metric <service> <name>
  docker compose exec -T -e PROBE_KEY="$DEPLOYER_KEY" "$1" python - "$2" <<'PY'
import http.client
import os
import sys

name = sys.argv[1]
conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
conn.request("GET", "/metrics", headers={"X-API-Key": os.environ["PROBE_KEY"]})
body = conn.getresponse().read().decode()
print(
    sum(
        float(line.rsplit(" ", 1)[1])
        for line in body.splitlines()
        if line.startswith((name + " ", name + "{"))
    )
)
PY
}

positive() { # positive <value> <message>
  awk -v value="$1" 'BEGIN { exit (value + 0 > 0) ? 0 : 1 }' || fail "$2"
}

predict() { # predict <subject>
  curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST "$ROUTER/v1/predict" \
    -H "X-API-Key: $DEPLOYER_KEY" -H 'Content-Type: application/json' \
    -d "{\"subject_id\": \"$1\", \"features\": {\"tenure_months\": 14, \"monthly_spend\": 72.5, \"support_tickets\": 3, \"num_products\": 2, \"discount_rate\": 0.15, \"usage_trend\": -0.2, \"region\": \"south\"}}"
}

# --- 1. a signed registry, and a router that refuses to be ready without a champion -------------
echo "== build"
docker compose build

echo "== bootstrap: train and sign the demo releases"
BOOTSTRAP="$(docker compose run --rm bootstrap 2>/dev/null | sed -n '/^{/,$p')"
PUBLIC_KEY="$(printf '%s' "$BOOTSTRAP" | jq -r '.public_key_b64')"
CHAMPION="$(printf '%s' "$BOOTSTRAP" | jq -r '.releases.champion')"
CHALLENGER="$(printf '%s' "$BOOTSTRAP" | jq -r '.releases.subgroup_worse')"
test -n "$PUBLIC_KEY" || fail "bootstrap did not print a signing public key"
test -n "$CHAMPION" || fail "bootstrap did not publish a champion release"
export APP_TRUSTED_SIGNING_KEYS="$PUBLIC_KEY"

echo "== up: router, shadow, controller"
docker compose up --detach --wait --wait-timeout 180 router shadow controller
wait_for "$ROUTER/healthz" "router"
wait_for "$CONTROLLER/healthz" "controller"

code="$(status_of "$ROUTER/readyz")"
test "$code" = "503" || fail "/readyz returned $code before a champion was bootstrapped (want 503)"

# --- 2. a champion, and then predictions --------------------------------------------------------
echo "== bootstrap the champion through the control API"
curl -fsS -X POST "$CONTROLLER/v1/champion" \
  -H "X-API-Key: $MANAGER_KEY" -H 'Content-Type: application/json' \
  -d "{\"release_id\": \"$CHAMPION\"}" >/dev/null
for _ in $(seq 1 30); do
  code="$(status_of "$ROUTER/readyz")"
  test "$code" = "200" && break
  sleep 1
done
test "$code" = "200" || fail "/readyz still $code after the champion was bootstrapped"

echo "== propose a rollout so the shadow stage is active"
curl -fsS -X POST "$CONTROLLER/v1/rollouts" \
  -H "X-API-Key: $DEPLOYER_KEY" -H 'Content-Type: application/json' \
  -d "{\"challenger\": \"$CHALLENGER\", \"spec\": \"standard\"}" >/dev/null
sleep 3

for i in $(seq 1 40); do
  code="$(predict "subject-$i")"
  test "$code" = "200" || fail "/v1/predict returned $code for subject-$i"
done

# --- 3. the shadow hop really crosses the container boundary -------------------------------------
sleep 5
scored="$(metric shadow app_predictions_total)"
echo "shadow worker scored: $scored"
positive "$scored" "no request reached the shadow worker across the container boundary"

# --- 4. the shadow worker has no egress ---------------------------------------------------------
echo "== the shadow worker cannot reach anything off the internal bridge"
if docker compose exec -T shadow python -c "
import socket

socket.setdefaulttimeout(5)
socket.create_connection(('1.1.1.1', 443))
" 2>/dev/null; then
  fail "the shadow worker opened an outbound connection; the internal network is not isolating it"
fi

echo "== the shadow worker can write its own log and nothing else"
docker compose exec -T shadow python - <<'PY' || exit 1
import pathlib
import sys

denied = [
    "/var/lib/pmd/control.sqlite",
    "/var/lib/pmd/predlog/router/probe.parquet",
    "/var/lib/pmd/registry/probe",
]
for name in denied:
    try:
        with pathlib.Path(name).open("ab"):
            pass
    except OSError:
        continue
    print(f"FAIL: the shadow worker can write {name}")
    sys.exit(1)
own = pathlib.Path("/var/lib/pmd/predlog/shadow/probe")
own.write_bytes(b"")
own.unlink()
print("shadow filesystem: own log writable, control plane and router log are not")
PY

# --- 5. a dead shadow worker does not touch the champion ----------------------------------------
echo "== kill the shadow worker; champion responses must not change"
docker compose kill shadow
for i in $(seq 41 120); do
  code="$(predict "subject-$i")"
  test "$code" = "200" || fail "/v1/predict returned $code after the shadow worker was killed"
done
dropped="$(metric router app_shadow_dropped_total)"
failed="$(metric router app_shadow_failed_total)"
echo "router shadow drops: $dropped, failures: $failed"
positive "$(awk -v a="$dropped" -v b="$failed" 'BEGIN { print a + b }')" \
  "the router neither dropped nor failed a shadow job while the worker was down"

# --- 6. the image runs the whole offline demo ----------------------------------------------------
echo "== the offline demo inside the image"
DEMO="$(docker compose run --rm demo | sed -n '/^{/,$p')"
printf '%s' "$DEMO" | jq -e '.rollback_scenario.outcome == "rolled_back"' >/dev/null \
  || fail "the in-container demo did not roll the subgroup-regressed challenger back"
printf '%s' "$DEMO" | jq -e '.promotion_scenario.status == "promoted"' >/dev/null \
  || fail "the in-container demo did not promote the better challenger"
printf '%s' "$DEMO" | jq -e '.audit.verified == true' >/dev/null \
  || fail "the in-container demo produced an audit chain that does not verify"

echo "compose end-to-end OK"
