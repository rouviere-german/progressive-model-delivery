# 4. Mixture prior strength as a policy knob, not a tuning constant

- Status: accepted
- Date: 2026-09-27

## Context

Every e-value here mixes a likelihood ratio over a prior on the alternative. For the Bernoulli
guardrails the prior is `Beta(c*pi0, c*(1-pi0))` truncated to the alternative side of the null,
where `pi0` is the null value and `c` -- `prior_strength` in the rollout spec -- is the total prior
mass. For the mean-difference tests the analogue is the mixing scale `tau` (`mixture_scale`).

Two facts about `c` pull in opposite directions:

- **It cannot break validity.** *Any* prior gives a valid e-value; the supermartingale property
  holds component-wise, so it survives any mixing measure. That is the point of mixture methods and
  it is what `tests/unit/test_stats_evalues.py::test_mixture_is_a_supermartingale_under_every_null_rate`
  checks.
- **It decides what the test is good at.** A mixture is a bet spread over effect sizes. Large `c`
  concentrates the prior near the null, betting on tiny regressions: the test accumulates evidence
  slowly but eventually detects very small effects. Small `c` spreads the prior wide, betting on
  large regressions: it fires fast on a broken model and is weak against a 0.5pp drift.

So `c` is not a constant to be tuned until the numbers look good. It is a statement about which
regressions this system exists to catch. Getting that backwards is how a guardrail ends up
technically valid and operationally useless -- it never fires on the failures that actually happen.

## Decision

Treat the mixing parameters as **part of the versioned rollout policy**, not as implementation
constants:

- `prior_strength` and each guardrail's `mixture_scale` live in the rollout spec YAML, are covered
  by the spec's SHA-256, and are therefore recorded in every audit entry. A decision can always be
  re-derived from the policy it was made under.
- The default is `prior_strength = 10`, chosen for the effect sizes in
  `bench/bench_power.py`: error-rate regressions of 1pp and up, a latency regression that moves SLO
  exceedance by tens of percent, and a subgroup regression of a few points on ~12% of traffic. With
  that value the power study detects 1/2/5pp error regressions in 100% of rollouts and the
  subgroup-only regression in 87.5-96%.
- `mixture_scale` defaults to 0.05 for log loss, roughly the scale of a log-loss shift worth acting
  on for this model, and is set per guardrail because a log-loss margin and an error-rate ratio are
  not on the same scale.
- The default is *not* claimed to be optimal, and nothing in the code tunes it against the data.
  Tuning a prior on the same data the test then judges would be a garden path; if a team needs
  different sensitivity they change the policy, bump its version, and the change is visible in the
  audit trail.

## Consequences

**What we get.** The sensitivity of the release gate is reviewable. "Why did this not fire?" has a
document-shaped answer -- the policy says which regressions it bets on -- instead of an argument
about a magic number in a module. Because the spec hash covers it, a rollout judged under one
sensitivity cannot be quietly compared to one judged under another.

**What it costs.** Someone has to choose. A team whose real risk is a slow 0.3pp drift will find
`prior_strength = 10` underpowered, and the honest answer is to raise it and accept slower reaction
to large breakage, not to add a second test at the same alpha. The power study is the tool for that
choice: it reports detection probability and median customer exposure per effect size, so the
trade-off is measured rather than argued.

**A limitation worth stating.** Because the mixing parameters do not affect validity, there is no
test that can catch a *bad* choice -- only the power study can show it, and only for the effect
sizes someone thought to simulate. An effect the scenario catalogue does not contain is an effect
nobody has measured the power against.

**Cost of changing course.** Very low. Changing a value is a spec edit and a version bump. Changing
the *family* of prior (say, to a discrete mixture over a grid of effect sizes) is a change inside
`stats/evalues.py` with no effect on the controller or the spec format.
