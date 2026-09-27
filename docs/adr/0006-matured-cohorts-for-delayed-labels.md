# 6. Matured cohorts for delayed labels

- Status: accepted
- Date: 2026-09-27

## Context

Churn labels do not arrive with the prediction. In the simulator, and in the systems it is modelled
on, a negative is confirmed sooner than a positive: a customer who did not churn is settled by the
end of the observation window, a customer who did needs the event to happen first. Label delay is
therefore correlated with the outcome.

That leaves the controller with a choice at every reconciliation pass. It knows, for a set of
requests, that some labels have arrived and some have not. Three options:

1. **Score every label that has arrived.** The natural reading of the log, and wrong: at any cutoff
   the labelled subset over-represents whichever class matures faster, so the measured error rate is
   biased, and the bias moves as the cutoff moves. Two models that differ in *which* class they get
   wrong are compared on a sample that favours one of them.
2. **Impute the missing labels.** Needs a delay model, and its errors land exactly in the guardrails
   the system exists to protect. Out of scope (README Non-goals).
3. **Only read requests whose label window has closed.** A request older than
   `label_window_seconds` is *matured*: either its label arrived inside the window, or it never
   will. Within a matured cohort, arrival order carries no information about the outcome.

## Decision

Evaluate matured cohorts only, and make the window a property of the data rather than a promise
about configuration.

- `EvidenceReader` counts a request as labelled when `ts <= now - label_window_seconds` **and** the
  label's `received_at` is no later than `ts + label_window_seconds`. Requests whose label never
  arrived are dropped, assumed missing at random *with respect to the arm* -- both models see the
  same customers, so a delay mechanism that is independent of the arm cancels.
- The second clause is enforced in the reader, not assumed from the router's configuration. The
  router refuses a label after `APP_LABEL_TTL_SECONDS`, but that TTL is a separate setting from the
  spec's window, and a deployment can set it longer. Such labels are counted in
  `late_labels_ignored` and never enter a test.
- `ControlService.propose` refuses a spec whose `label_window_seconds` is shorter than the router's
  TTL outright, so the mismatch is visible at proposal time instead of quietly shrinking the
  effective window.
- Latency guardrails need no label and use every request in the stage.

## Consequences

**What we get.** The join is a deterministic function of the logs and the cutoff: the same cohort at
a later cutoff yields the same numbers plus whatever fully matured since. That is what makes the
per-stage e-processes legitimate -- each pass adds increments from cohorts that were not scored
before -- and what makes a replay reproduce the same audit hashes.

**What it costs.** Latency. A regression is provable only after the window has closed on enough
requests, so the shadow stage of the production policy needs at least an hour of maturation before
any label-dependent guardrail can fire. The label-free guards (score PSI, latency exceedance) exist
partly to cover that hour, and they have no error guarantee (`docs/ROLLOUT_POLICY.md`).

**What breaks if the rule is violated.** A label folded into an already-scored cohort adds an
increment to an e-process for data it has already published. Ville's inequality still bounds the
supremum of a genuine e-process, but the statistic being computed is no longer one: the same
observation can be counted under two different filtrations. The measured A/A rate would stay
plausible while the guarantee behind it no longer held, which is the worst failure mode available
here -- hence the reader-side enforcement and
`tests/integration/test_evidence.py::test_a_label_that_arrived_after_its_window_never_counts`.
