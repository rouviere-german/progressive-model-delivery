# Runbook: a rollout is misbehaving

For whoever is on call. The controller rolls back on its own and needs no approval to do it, so most
of this runbook is about the cases where it *did not*, or where it did and you need to know why.

Everything below assumes the control API at `$CONTROLLER` and an API key bound to at least the
`viewer` role. Rollback needs `deployer` or `release_manager`.

## 0. The 30-second version

```bash
# What does the controller think is happening?
curl -fsS -H "X-API-Key: $KEY" "$CONTROLLER/v1/rollouts/current" | jq
curl -fsS -H "X-API-Key: $KEY" "$CONTROLLER/v1/rollouts/current/evidence" | jq

# Stop it now. This is always allowed and always safe.
curl -fsS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"reason": "on-call rollback: <what you saw>"}' \
  "$CONTROLLER/v1/rollouts/<rollout_id>/rollback" | jq
```

A rollback sets the challenger's lifecycle to `rolled_back` and the champion stays exactly as it was.
Within `state_refresh_seconds` (default 1 s) every router has re-read the state and sends 100% of
traffic to the champion again. **There is no rollback of the rollback**: a rolled-back release is
terminal and needs a new release to try again. That is deliberate.

## 1. Decide which situation you are in

| Symptom | Situation | Go to |
|---|---|---|
| `app_rollout_stage` dropped to 5 and you were not paged by a human | The controller rolled back by itself | §2 |
| Customer-visible errors or latency, rollout still advancing | A regression the guardrails do not measure | §3 |
| `app_canary_weight > 0` and stuck; `app_reconcile_total{outcome="hold"}` climbing | Held, probably on drift or minimums | §4 |
| `app_lease_held == 0` on every controller | No controller is acting | §5 |
| `app_shadow_dropped_total` climbing fast | Shadow coverage is degraded | §6 |
| `/readyz` on a router returns 503 | The router has no model it is willing to serve | §7 |

## 2. The controller rolled back by itself

This is the system working. You need the *reason*, which is in the audit log, not in the metrics.

```bash
curl -fsS -H "X-API-Key: $KEY" "$CONTROLLER/v1/audit?rollout_id=<rollout_id>" | jq '.[-1]'
```

The last entry has `action: "rolled_back"` and `data.evidence` containing:

- `tests`: every guardrail with its `log_e`, its `threshold` and whether it `crossed`. The one that
  crossed is your answer.
- `stage_kind` and `canary_weight_percent`: the exposure the decision was made under. If
  `stage_kind` is `shadow`, **no customer was ever served by the challenger.**
- `stage_samples`, `challenger_exposure`: how much evidence, how many served requests.
- `drift`: feature and score PSI at the time.
- `spec_sha256`: the policy version it was judged under.

Then:

1. Check the chain verifies: `uv run progressive-model-delivery audit verify --database <db>`.
2. Map the guardrail to the failure mode. `worst_subgroup_error[<region>]` means the model regressed
   for one group and probably looked fine globally -- that is the case you would have missed.
   `latency_slo` usually means the model got bigger, not worse. `log_loss` with `error_rate` fine
   means calibration moved.
3. Hand the evidence to whoever owns the model. The rollout id, the crossed e-value and the spec
   hash are the whole story; they do not need your interpretation.

No action is needed to protect traffic. The champion never stopped serving.

## 3. Customers are unhappy and the rollout is still advancing

The guardrails measure error, worst-subgroup error, SLO exceedance and log loss. If the damage is
somewhere else -- a downstream system, a calibration-sensitive threshold, a feature pipeline -- the
controller has no reason to stop.

Roll back manually (§0), then decide whether the missing signal should become a guardrail. Adding one
is a spec change with a version bump, not a code change.

## 4. The rollout is held and not advancing

Read the evidence:

```bash
curl -fsS -H "X-API-Key: $KEY" "$CONTROLLER/v1/rollouts/current/evidence" | jq '.evidence | {minimums_met, drift, stage_samples, stage_elapsed_seconds}'
```

Three causes, in order of likelihood:

- **`drift.drift_ok == false`.** A feature's PSI is over threshold, so the controller will not
  advance on evidence gathered from traffic the model was not trained for. Check
  `app_feature_psi{feature=...}` to see which one. This is usually real: an upstream change, a
  seasonal shift, or a broken feature. Fix the input, or accept the drift and re-baseline the
  champion; do not override.
- **`minimums_met == false`.** Not enough matured samples or not enough wall time yet. Compare
  `stage_samples` against the stage's `min_samples` and `stage_elapsed_seconds` against its
  `min_duration_seconds`. If sample counts are far below traffic volume, suspect the label path
  (§6 and `app_feedback_total{outcome=...}`): labels that never arrive never mature.
- **Non-inferiority not proven.** The stage requires it and the e-value has not crossed
  `1/alpha_promotion`. This is the test saying "not enough evidence yet", which on a genuinely
  equal model can take a while. Waiting is the correct action.

Every stage has a `max_duration_seconds`; a stage that never meets its gate aborts the rollout
rather than holding forever. An aborted rollout has `status: "aborted"` and needs a human to decide
what happens next.

**Forcing it needs two people.** If you are certain the gate is wrong:

```bash
curl -fsS -X POST -H "X-API-Key: $KEY_A" -H 'Content-Type: application/json' \
  -d '{"action": "force_advance", "reason": "<why>"}' \
  "$CONTROLLER/v1/rollouts/<rollout_id>/overrides" | jq   # returns request_id

curl -fsS -X POST -H "X-API-Key: $KEY_B" \
  "$CONTROLLER/v1/overrides/<request_id>/approve" | jq     # a DIFFERENT release manager
```

The requester counts as the first signature, so this needs two distinct release managers. The
override expires if the rollout moves on in the meantime. Both signatures and the reason land in the
audit chain. Automatic rollback remains armed the whole time -- an override advances a stage, it does
not disable the guardrails.

## 5. No controller holds the lease

`app_lease_held == 0` everywhere and `/readyz` on the controller returns 503.

Serving is unaffected: routers read state, they do not need the controller. What stops is *evaluation*
-- nobody is watching the guardrails, so a bad challenger will keep its current traffic share
indefinitely.

1. Check the controller process is alive and reaching the control database.
2. A lease is taken over automatically after `lease_seconds` (default 15 s). If no controller has
   taken it, none is running.
3. If you cannot restore a controller quickly and a rollout is at a non-zero weight, roll it back
   manually (§0). An unwatched canary is worse than no canary.

A controller that lost its lease cannot write, so a partially-restored controller is not a
correctness risk -- only a liveness one.

## 6. Shadow coverage is degraded

`app_shadow_dropped_total` or `app_shadow_failed_total` climbing means shadow jobs are not reaching
the worker. Customer requests are **not** affected: the enqueue is non-blocking and the champion
answers before it happens (see `docs/adr/0005-shadow-traffic-is-dropped-not-queued.md`).

The consequence is statistical: a shadow stage accumulates evidence more slowly, so it holds longer.
Check the shadow container's health, its CPU share, and `app_shadow_queue_depth`. If drops are
sustained, the shadow worker is undersized for current traffic.

## 7. A router reports not ready

`/readyz` reports which check failed by name:

```bash
curl -s "$ROUTER/readyz" | jq
```

`champion_verified: "fail"` means the router has no champion it is willing to load. Either no
champion is bootstrapped, or the champion's signature no longer verifies against
`APP_TRUSTED_SIGNING_KEYS`. **This is a fail-closed design**: a router that cannot verify its model
refuses traffic rather than serving something unverified. Check the trusted key list, then the
registry directory. `app_challenger_rejected_total` climbing instead means the *challenger* failed
verification and the champion is still serving, which is the safe degradation and not an outage.

## Metrics worth a dashboard

| Metric | Read it as |
|---|---|
| `app_rollout_stage{release}` | 0 staged, 1 shadow, 2 canary, 3 production, 4 retired, 5 rolled back |
| `app_canary_weight` | Share of traffic on the challenger right now |
| `app_e_value{guardrail,test}` | Current evidence per guardrail; crossing `1/alpha'` is a decision |
| `app_guardrail_breaches_total{guardrail}` | Which guardrail has ever fired |
| `app_reconcile_total{outcome}` | `hold` climbing with nothing else moving = §4 |
| `app_lease_held` | 1 on exactly one controller |
| `app_feature_psi{feature}`, `app_score_psi` | Drift preconditions |
| `app_predictions_total{release,arm}`, `app_prediction_latency_seconds` | Per-version traffic and latency |
| `app_prediction_errors_total{release}`, `app_challenger_rejected_total` | Scoring and verification failures |
| `app_shadow_dropped_total`, `app_shadow_failed_total`, `app_shadow_queue_depth` | Shadow coverage (§6) |
| `app_feedback_total{outcome}` | Label path health; `refused` climbing means late or unknown labels |

The names and label sets above are pinned by
`tests/unit/test_observability.py::test_every_documented_rollout_metric_is_exported_with_its_value`,
so a rename breaks the build rather than this runbook.

## What never needs approval

Automatic rollback by the controller, and manual rollback by any deployer or release manager. If you
are wondering whether you are allowed to stop a rollout: you are.
