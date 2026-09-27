# 2. Anytime-valid e-values instead of fixed-horizon tests

- Status: accepted
- Date: 2026-09-27

## Context

A reconciliation controller re-evaluates its guardrails on every pass -- here every few seconds,
and in the study up to 90 times per rollout. That is exactly the pattern classical hypothesis tests
forbid: a two-proportion z-test controls its type-I error at *one* pre-specified sample size, and
acting on it whenever it happens to cross the threshold does not.

The alternative is not hypothetical. We measured it: `bench/bench_validity.py` runs 2,000 A/A
rollouts (champion and challenger identical) and judges each one two ways. The anytime-valid
guardrails act wrongly in 0.6% of rollouts. The same arrival process judged by a z-test re-read at
every pass acts wrongly in **36.7%**, against a nominal 5%. Read once at the horizon, the same test
sits at 5.1%. So a controller built on fixed-horizon tests would roll back roughly one in three
perfectly good models, and each of those rollbacks would look like a real finding, complete with a
p-value.

Three families could fix this:

1. **Group-sequential designs** (O'Brien-Fleming, Pocock). Valid, well understood, and standard in
   clinical trials -- but they require the number and timing of looks to be fixed in advance. A
   reconciliation loop does not know how many passes a rollout will take, and a stage that waits for
   a sample minimum takes as many passes as the traffic needs.
2. **Always-valid p-values via mixture SPRTs** (Johari, Koomen, Pekelis and Walsh, "Peeking at A/B
   Tests", 2017). No fixed horizon, closed form, and the natural fit for mean differences such as
   log loss.
3. **E-values and e-processes** (Robbins 1970; Howard, Ramdas, McAuliffe and Sekhon, "Time-uniform
   Chernoff bounds via nonnegative supermartingales", 2021; Grünwald, de Heide and Koolen on safe
   testing). An e-value is a nonnegative statistic with expectation at most 1 under the null;
   Ville's inequality gives `P(exists t: E_t >= 1/alpha) <= alpha` for *any* stopping rule, and
   e-values multiply across independent batches and combine across hypotheses by averaging.

## Decision

Use anytime-valid statistics throughout, choosing the family per metric:

- **Bernoulli guardrails** (error rate, worst-subgroup error, latency > SLO) use **e-values**: the
  likelihood ratio mixed over a Beta prior truncated to the alternative side of the null. Both
  rollout designs first reduce the two-model comparison to a single Bernoulli parameter with a
  *known* null value, so there is no nuisance parameter:
  - shadow stage: among discordant pairs (exactly one model made the bad event),
    `pi = p01/(p01+p10)`, and "no worse" is exactly `pi <= 1/2` -- the sequential form of McNemar's
    test;
  - canary stage: given a bad event, `pi = w r / (w r + 1 - w)` where `w` is the known stage weight
    and `r` the rate ratio, independent of the champion's own rate.
- **Mean differences** (log loss) use the **mixture SPRT** with Robbins' normal mixture, paired in
  the shadow stage and two-sample (Welch variance) in canary stages.
- Per-stage statistics are computed on disjoint requests, so e-values **multiply** across stages
  (optional continuation).
- `alpha_rollback` is split Bonferroni-style over every harm test, each subgroup counting as its own
  test; promotion is an intersection-union test at `alpha_promotion`.
- The operator-facing intervals are **confidence sequences** from the same mixtures, so the number
  on the dashboard and the decision come from one object.

## Consequences

**What we get.** The controller may be watched continuously, by a human or by itself, without an
error-rate penalty. A/A false rollback measures 0.6% [0.31%, 1.05%] against a 2.5% budget. A
challenger past the non-inferiority margin was promoted in 0 of 2,000 rollouts. Confidence
sequences keep 96.5-97.3% coverage checked at every one of 1,500 steps. Rollback can be
*immediate*: there is no minimum sample size below which the harm test is invalid, only a threshold
that strong evidence crosses sooner.

**What it costs.** Validity under arbitrary stopping is bought with power. `bench/bench_power.py`
measures the bill: large regressions are caught in one or two passes and at zero customer exposure,
because the shadow stage sees them first; but `subgroup_worse` -- 2pp worse for 12% of traffic -- is
detected in 87.5% of rollouts rather than 100%, and costs a median 1,652 served requests. A
fixed-horizon test sized for 80% power at that effect would need about 21,000 samples and could not
stop early at all, but it would be more powerful at its own horizon. We accept that trade: a missed
subtle regression is caught by the next rollout's evidence, whereas a 36.7% false-rollback rate
would make the whole system untrustworthy within a week.

**Where the guarantee is weaker than it looks.** The Bernoulli e-values are exact
supermartingales, tested as such. The mean-difference tests use running (plug-in) variances, which
makes their guarantee asymptotic rather than exact (Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas,
2024). Log losses are right-skewed, so early sample means and variances are biased together; we
hold the statistic at zero for a 200-sample burn-in per arm and measure the resulting false-harm
rate on deliberately skewed losses
(`tests/unit/test_stats_msprt.py::test_plug_in_variance_msprt_keeps_false_harm_rate_below_alpha_on_skewed_losses`).
This is stated in `docs/ROLLOUT_POLICY.md` rather than hidden behind the word "anytime-valid".

**Cost of changing course.** Low for the statistics, high for the shape. Swapping a mixture for a
different e-process is a change inside `stats/`. Going back to fixed-horizon tests would mean giving
the controller a fixed look schedule, which changes the stage rule, the spec format and the meaning
of every recorded decision.
