# 5. Shadow traffic is dropped on overflow, never buffered or retried

- Status: accepted
- Date: 2026-09-27

## Context

The shadow stage exists so a challenger can be evaluated on real traffic before any customer is
served by it. That means the router has to get every request to a second model. The obvious
implementations all quietly put the customer's request behind the challenger:

- **Score both models inline.** Simplest, and wrong: the customer waits for a model that is, by
  definition, not trusted yet. A challenger with a pathological input, a corrupted artifact or a
  slow code path becomes a latency incident.
- **Await the shadow call after responding.** Better, but in an async server "after responding"
  still occupies the event loop and the task. A hanging challenger accumulates tasks until something
  runs out.
- **Durable queue with retries** (Kafka, Redis Streams, an outbox table). No request loss, at the
  cost of an extra dependency on the serving path and, more subtly, a *backlog*: jobs retried
  minutes later are scored by a model version that may no longer be the challenger, and they arrive
  in the log with a timestamp that no longer matches the stage they belong to.

The statistical requirement is weaker than "no loss", and that is the key observation. The paired
shadow test needs pairs whose *presence* is independent of the outcome -- it does not need every
request. Dropping because a queue is full is caused by load, not by what either model predicted.
Retrying, by contrast, would bias the sample toward whatever the shadow worker finds slow, which
is exactly the correlation that would break the test.

## Decision

Forward shadow traffic through a **bounded in-memory queue that drops on overflow, and never
retries**:

- The champion's response is produced first. Only then does the router call `offer()`, a
  non-blocking `put_nowait` on a queue of `shadow_queue_size` jobs.
- A full queue **drops the job and increments `app_shadow_dropped_total`**. `offer()` never blocks,
  never awaits and never raises.
- A fixed pool of worker tasks drains the queue and POSTs each job to the shadow worker with a hard
  timeout (`shadow_timeout_seconds`). A failure is counted by reason in
  `app_shadow_failed_total`; the job is discarded.
- The shadow worker is a separate container on a network with no gateway. It verifies the release
  signature, scores, and writes only to the prediction log.
- Delivered, dropped, failed and queue depth are all exported, so shadow coverage is an observable
  quantity rather than an assumption.

## Consequences

**What we get.** The only shadow work on the customer's path is the enqueue: p99 **1.9 us**, and
1.5 us on the drop path (`bench/bench_serving.py`, gated at 50 us in CI). A shadow worker that never
answers costs **0 failed requests** and about 1,640 dropped jobs out of 2,200, while the champion's
own percentiles move only within this machine's run-to-run spread (0.4-2.1 ms at p99 across runs;
see the README Results section, which states that spread rather than picking the best run). Fault
injection covers crash, hang and error
(`tests/integration/test_shadow_isolation.py`), and the compose end-to-end script kills the worker
container outright and asserts that 80 subsequent champion requests all return 200.

**What it costs.**

- **Shadow coverage is not 100% under load.** The shadow stage's sample minimum is reached in wall
  time, not request count, so a period of heavy load lengthens the stage instead of corrupting it.
  A deployment that sustained heavy drops would advance slowly and should alert on
  `app_shadow_dropped_total`, which the runbook says to do.
- **Drops are not perfectly outcome-independent.** If a *particular kind of request* were both
  slow for the shadow worker and correlated with the label, the dropped sample would be biased. We
  do not claim otherwise: the mechanism removes the large, obvious correlation (retry-on-slow) and
  leaves a small one that nothing here measures. It is listed as a residual risk in the threat
  model (D3) rather than argued away.
- **No replay.** A rollout cannot be re-judged on traffic that was dropped. Re-running a shadow
  stage means running it again on new traffic.

**Cost of changing course.** Moderate. Swapping the in-memory queue for a durable one is a change
behind `ShadowForwarder`'s interface (`offer()` plus a drain task), but the *statistical* argument
above would have to be revisited: a durable queue that retries would need the join to reject jobs
whose scoring timestamp has drifted out of their stage, and the bias question would need an answer
rather than a bound.
