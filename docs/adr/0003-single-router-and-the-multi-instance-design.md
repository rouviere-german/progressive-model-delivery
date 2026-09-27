# 3. One router instance, with the multi-instance design written down

- Status: accepted
- Date: 2026-09-27

## Context

A real serving tier runs several router replicas behind a load balancer. This repository runs one.
That is a scope decision, and the interesting question is not "is one enough" (it is not) but
"which parts of the design would have to change, and did we accidentally build something that
*cannot* scale out".

Four pieces of state sit on the router's path:

1. **Arm assignment.** Which arm a subject is on.
2. **The prediction log.** Rows the controller later joins.
3. **The request index.** The set of request ids that may still receive a label.
4. **The serving state.** Which release is champion, which is challenger, at what weight.

Each has a different answer.

## Decision

Run a single router, and design each piece of state so that scale-out is a deployment change rather
than a redesign:

- **Assignment is stateless and deterministic.** The arm is
  `SHA-256(rollout_salt || HMAC(secret, subject_id)) mod 10000 < weight_bp`. Two replicas with the
  same salt and secret agree without talking to each other, and because the threshold only moves up
  (1% -> 5% -> 25% -> 50%) a subject on the challenger stays on it as the weight rises. No sticky
  sessions, no shared assignment table.
- **The prediction log is append-only and partitioned by writer.** Each process writes its own
  Parquet segments under a prefix; the controller's DuckDB query reads the whole directory. Adding
  replicas adds files, not coordination.
- **The controller is already single-writer by construction**, via a SQLite lease with a monotonic
  fencing token that every write re-checks inside its transaction. Several controllers may run; only
  the lease holder acts, and a controller that lost its lease cannot write. That is the part that
  genuinely must be exclusive, and it is the part that is tested for it.
- **Serving state is read, not held.** Each router re-reads the champion, challenger and weight from
  the control database every `state_refresh_seconds`. There is no push, no invalidation and no
  cache coherence problem -- only a bounded staleness window.

Accept two limitations, and name them in the README Non-goals rather than implying otherwise:

- **The request index is per process.** A label is accepted only for a request id *this* router
  issued. With several replicas, a label arriving at the replica that did not serve the request
  would be refused. The index would have to move to a shared store (Redis with a TTL, or a small
  table), which is the one real change scale-out requires.
- **Nothing here runs or tests more than one router.** The claims in this repository are about one
  process.

## Consequences

**What we get.** The whole system fits in one `docker compose up` and one offline demo, the
statistical work stays the focus, and the pieces that would have to change are one bounded
component rather than the architecture. The bounded-staleness model also makes a real property
testable: a weight change reaches serving within a refresh interval, and the demo exercises that
path explicitly instead of mocking it.

**What it costs.** Three concrete gaps, in the order they would bite:

1. Labels would be refused by the wrong replica (the request index).
2. A weight change reaches replicas at slightly different times, so for one refresh interval the
   effective weight is a blend of the old and new values. The statistics are unaffected -- the
   canary e-value conditions on the *recorded* per-request weight, not on the configured one -- but
   the `canary_weight` gauge would briefly disagree between replicas.
3. Metrics are per process, so a scrape has to aggregate across replicas before the numbers in
   `docs/RUNBOOK_ROLLBACK.md` mean what they say.

**Cost of changing course.** Low, and deliberately so. The shared request index is a single
interface (`accepts(request_id)` plus a TTL) with one in-process implementation today. Nothing else
on the router's path holds state that two processes would have to agree on.
