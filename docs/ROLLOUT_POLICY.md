# Rollout policy and what its guarantees actually say

This document is the contract between the controller and the people who trust it. It states the
policy in `src/progressive_model_delivery/specs/standard.yaml`, the two guarantees that policy
carries, and -- at least as important -- the things it does **not** say.

## The policy

| Stage | Weight | Min samples | Min duration | Max duration | Non-inferiority required |
|---|---|---|---|---|---|
| shadow | 0% (paired) | 2,000 | 1 h | 48 h | no |
| canary | 1% | 100 | 30 min | default | no |
| canary | 5% | 500 | 1 h | default | no |
| canary | 25% | 2,000 | 2 h | default | yes |
| canary | 50% | 5,000 | 4 h | default | yes |

Guardrails:

| Name | Metric | Harm test | Promotion test |
|---|---|---|---|
| `error_rate` | classification error | ratio > 1 | ratio <= 1.10 |
| `worst_subgroup_error` | error per subgroup (region) | ratio > 1, Bonferroni over subgroups | -- |
| `latency_slo` | P(latency > 25 ms) | ratio > 1 | -- |
| `log_loss` | mean log loss | difference > 0 | difference <= 0.02 |

Error budget: `alpha_rollback = 0.025`, `alpha_promotion = 0.025`. Labels are accepted for one hour
(`label_window_seconds`); only requests whose window has closed enter the tests. Drift
preconditions: feature PSI <= 0.2, score PSI <= 0.25, from 500 samples.

`prior_strength = 10` and each guardrail's `mixture_scale` decide which effect sizes the tests have
power against; see `docs/adr/0004-mixture-prior-strength.md`. The spec is versioned and its SHA-256
is written into every audit entry, so a decision can always be re-derived from the policy it was
made under.

## The two guarantees

**Rollback (type-I error on harm).** For a challenger that is genuinely no worse than the champion
on every guardrail, the probability that the controller *ever* rolls it back -- across every
reconciliation pass, at every stage, for any stopping rule -- is at most `alpha_rollback = 0.025`.

Each harm test is an e-value: a nonnegative statistic whose expectation is at most 1 under the null.
Ville's inequality then bounds `P(exists t: E_t >= 1/alpha')` by `alpha'`, with no reference to a
sample size or a look schedule. `alpha_rollback` is split Bonferroni-style over the `K` harm tests
(each subgroup counting separately), so each runs at `alpha_rollback / K` and the union bound gives
the overall budget.

Measured: **0.6%** of 2,000 A/A rollouts, 95% CI [0.31%, 1.05%]
(`uv run python bench/bench_validity.py --check`).

**Promotion (type-I error on inferiority).** For a challenger that is inferior on any *tested*
guardrail -- error ratio above 1.10, or log loss more than 0.02 worse -- the probability that it
reaches production is at most `alpha_promotion = 0.025`, plus whatever the human approver lets
through.

Promotion is an intersection-union test: every non-inferiority test must reject, so no correction is
needed. Promotion evidence comes only from canary stages, because evidence about traffic the
challenger actually served is the only evidence that answers the question being asked.

Measured: **0 of 2,000** rollouts, 95% CI [0.00%, 0.18%].

## What these guarantees do not say

**They are about the tested guardrails only.** A challenger that is worse in a way no guardrail
measures -- calibration, a subgroup not in the data contract, a failure mode that shows up in a
downstream system -- passes. The system controls the error rate of the questions it asks. It has
nothing to say about the questions nobody asked.

**"No worse" means "no worse than the champion".** Both models can be bad. The subgroup guardrail
catches a *regression* for a region; it does not catch a model that has always been bad for that
region. This is not a fairness control.

**Non-inferiority is not superiority.** Passing the 1.10 error ratio means "not detectably more than
10% worse", not "better". A guardrail may add a superiority requirement; none of the default ones
do.

**The mean-difference tests are asymptotically valid, not exact.** The log-loss mSPRT uses running
sample variances, which makes its guarantee asymptotic (Waudby-Smith, Arbour, Sinha, Kennedy and
Ramdas, 2024). Log losses are right-skewed, so early sample means and variances are biased together;
the statistic is held at zero for a 200-sample burn-in per arm, and the resulting false-harm rate is
measured on deliberately skewed losses
(`tests/unit/test_stats_msprt.py::test_plug_in_variance_msprt_keeps_false_harm_rate_below_alpha_on_skewed_losses`).
The Bernoulli guardrails -- error, subgroup error, latency -- are exact supermartingales and tested
as such.

**The drift precondition is conventional, not derived.** PSI thresholds of 0.2 and 0.25 are industry
convention. They gate *advancement*, so a drifting rollout holds rather than advancing on evidence
gathered from traffic the model was never trained for. They are not a hypothesis test and carry no
error rate.

**The score guard trades precision for speed.** A label-free score-PSI breach can roll a rollout
back on its own. A score shift is not proof of harm, and that guard has no error guarantee; it
exists because a broken model often shows in its outputs hours before any label matures.

**Late labels are lost, not modelled.** A label that misses its one-hour window is refused, counted
and ignored. If label loss correlated with the outcome, the estimate would be biased; nothing here
detects that.

**Exposure is bounded by the policy, not by the statistics.** How many customers a bad model reaches
before rollback is decided by the stage schedule and by how fast the evidence accumulates. The power
study measures it per failure mode; the median is 0 for globally-worse challengers (caught in
shadow) and 1,652 served requests for the subgroup-only regression.

**Power is not guaranteed at all.** The guarantees above are one-sided: they bound false decisions.
Nothing bounds the probability of *missing* a real regression. `bench/bench_power.py` measures
detection probability for the failure modes in the scenario catalogue -- 87.5% to 100% -- and a
failure mode not in that catalogue has never been measured.

## Changing the policy

Edit the YAML, bump `version`, and the new hash appears in every audit entry written under it.
Loosening `alpha_rollback` or dropping a guardrail is a policy change with a paper trail, which is
the point. Two things are *not* policy knobs: the controller always allows automatic rollback, and
final promotion always needs a second human.
