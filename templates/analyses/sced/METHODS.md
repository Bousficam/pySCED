# SCED inference methods - a didactic reference

This document explains the statistical methods behind the SCED (single-case
experimental design) toolbox: what question each method answers, the principle
that makes it valid, the exact function that runs it, and how to turn its output
into a conclusion. It is written for two readers at once:

- A statistician who knows R but not Python. Every function reference carries an
  `R bridge` line naming the closest R package or idiom. When a package is only a
  partial match (constrained randomization schemes, the two-piece coding, the
  change-point models), the bridge says so explicitly ("potential equivalent, to
  test") rather than overclaiming.
- An engineer who is weak in statistics. Every method starts with a plain-language
  `Problem`: the question being asked and why the obvious approach fails.

No method is left with the inference "up to the reader": each fiche states the
decision rule (the threshold, and how a p-value / HDI / ROPE / LOO gap becomes a
verdict).

Convention used throughout: `b0` intercept, `b1` baseline trend, `b2` immediate
level change at the intervention, `b3` slope change after the intervention;
`A` = baseline phase / reference condition, `B` = treatment phase / target
condition; `tau` = between-case SD of true effects; `rho` = lag-1
autocorrelation. All module paths are relative to the repository root
`functions/sced/`. In function signatures `*,` marks the start of keyword-only
(named) arguments; a returned `{a, b}` is an R named list; `**kwargs` is R's
`...`; `X | None` means X or None (a null/NA return).

---

## 0. Reading guide and design x approach matrix

The design is chosen at PLANNING time, from what was actually randomized, not from
the shape of the collected data. Two orthogonal axes decide the analysis:

1. The SCED design (how the intervention was scheduled).
2. The inferential approach: design-based (randomization / permutation),
   model-based frequentist (piecewise multilevel GLM), or model-based Bayesian.

The randomization test is the PRIMARY confirmatory inference in every design,
because its validity comes from the physical randomization, not from distributional
assumptions. Model-based methods add estimation (effect magnitude, heterogeneity,
per-case shrinkage (pulling noisy per-case estimates toward the group mean)) and
comparability across studies.

| SCED design | Design-based (randomization) | Model-based frequentist | Model-based Bayesian | Entry-point function (module) | Template |
|---|---|---|---|---|---|
| Alternating / ATD and N-of-1 (single case) | Fiche 1, Fiche 2 (label / residual permutation, condition effect) | Fiche 3 effect sizes; segmented ITS | Fiche 7 per-case | `pipeline_sced_alternating` (`alternating/core.py`) via `report_sced_alternating` (`alternating/run.py`); Bayes `report_sced_bayesian_condition` (`bayes/condition.py`) | `template_sced_alternating.py`; `template_sced_bayesian.py` |
| Alternating replicated across cases (group) | Fiche 1, Fiche 2 stratified | Fiche 3 | Fiche 8 hierarchical | `pipeline_sced_alternating_group` (`alternating/group.py`) via `report_sced_alternating(unit_col=...)` | `template_sced_alternating.py` |
| A-B phase (AB / ABA / ABAB) | Fiche 1 change-point randomization | Fiche 3 segmented ITS | Fiche 7 (onset known/unknown) | `report_sced_phase_design` (`phase_design.py`); Bayes `report_sced_bayesian_mbd` (`bayes/report.py`) | `template_sced_phase_design.py` |
| Multiple-baseline (MBD) | Fiche 1 (MB/WW/MB-R/KL/Rev) | Fiche 4 piecewise multilevel; Fiche 5 BC-SMD | Fiche 8, Fiche 9, Fiche 10 | `report_sced_multiple_baseline` (`mbd/report.py`); Bayes `report_sced_bayesian_mbd`, `report_sced_model_compare`, `report_sced_bayesian_percase` (`bayes/report.py`) | `template_sced_multiple_baseline.py`; `template_sced_bayesian_mbd.py`; `template_sced_model_compare.py`; `template_sced_bayesian_percase.py` |
| 3-level within-visit (measures > visit > patient) | visit-level randomization (Fiche 1 stratified) | Fiche 6 variance decomposition | (via Fiche 8 on visit means) | `variance_decomposition_3level`, `dispersion_test`, `within_visit_slope_test` (`intravisit.py`) via `report_sced_alternating(level3=True, dispersion=..., within_visit_slope=True)` | `template_sced_alternating.py` |

Two cross-cutting design flowcharts are maintained alongside this file:
`FLOWCHARTS.md` (how each model is assembled from its parameters) and `README.md`
(the full decision tree, statistic choice, effect-size tradeoffs).

---

# Part I - Design-based inference (randomization / permutation tests)

## Fiche 1 - Randomization test by label permutation

**Problem.** You alternated two or more conditions across sessions (or staggered an
intervention across baselines) using a known random schedule, and you want a
p-value for "does the condition matter?". A t-test or ANOVA assumes independent,
normally distributed observations. Single-case series violate both: sessions are
few, serially dependent, and often skewed or bounded. The classical p-value is
then untrustworthy.

**Principle.** The randomization test (Edgington and Onghena 2007) builds its
reference distribution from the assignments that COULD have occurred under the
actual randomization scheme. You compute a statistic on the observed labelling
(e.g. mean(B) - mean(A), or the between-condition variance for >=3 conditions),
then recompute it for every admissible relabelling the schedule allows, and read
off where the observed value falls. Because the schedule is what is permuted, the
p-value is exact under the sharp null "the condition changes nothing", for ANY
statistic, with no distributional assumption. Serial autocorrelation does not
break it: exactness comes from the physical randomization, not from independence
(verified in the code up to AR(1) rho = 0.8). The p-value never reaches zero:
`p = (1 + #{stat* as-or-more-extreme}) / (1 + B)` (Phipson and Smyth 2010).

The permutation set MUST mirror how conditions were really assigned. The toolbox
provides three schedules: free relabelling that preserves each condition count
(the default, exact for a count-balanced design); `max_consecutive` (reject draws
with more than k identical conditions in a row); and `block_scheme` (permute only
within consecutive blocks of size k, the randomized-block design). Using the wrong
schedule makes the p inexact.

**In this toolbox.**
- Module: `functions/sced/core.py`.
- Generic engine:
  `randomization_test(labels, values, *, statistic, scheme, n_perm=5000, random_state=0, two_sided=True)`
  returns `{observed, p_value, n_perm, two_sided}`.
- Schedules: `alternating_scheme(max_consecutive=None)` and `block_scheme(block_size)`
  each return a `scheme(labels, rng)` callable.
- Ready statistics: `diff_in_means(labels, values, *, target, reference)` and
  `omnibus_variance(labels, values)`; ordered/dose-response statistics via
  `make_ordered_trend_stat(condition_order, *, kind="slope", dose=None, improvement="increase")`
  (`kind` in {slope, jonckheere}; jonckheere = ordered-alternative trend test).
- Group (replicated single case): `stratified_ordered_trend_test(...)` for the dose
  trend; the common-effect stratified test is in Fiche 2.
- Single-case pipeline (`alternating/core.py`):
  `pipeline_sced_alternating(df, *, session_col, condition_col, outcome_col, conditions=None, outcome_type="continuous", improvement="increase", max_consecutive=None, block_size=None, time_covariate="none", detrend=None, perm_method="freedman-lane", covariate_cols=None, trend_test=True, condition_order=None, dose=None, power_analysis=False, ..., n_perm=5000, random_state=0, save_path=None, plot_path=None, verbose=True)`
  returns `(model_info: dict, randomization_result: dict)`.
- Driving template: `template_sced_alternating.py` -> `report_sced_alternating`
  (`alternating/run.py`). For A-B phase and MBD, the change-point / start-point
  randomization variants are Fiche 1 applied to a different schedule, run by
  `phase_design_test` / `report_sced_phase_design` and `multiple_baseline_test` /
  `report_sced_multiple_baseline` (see Fiche 4's neighbours).
- Methods paragraph generator: `sced_methods_text(...)` emits a citable Methods
  sentence describing the exact test that was run.
- R bridge: `SCRT` (Single-Case Randomization Tests, Bulte and Onghena) is the
  genuine equivalent for SCED design-based randomization tests (alternation CRD/RBD,
  phase, multiple-baseline). Generic label permutation: `coin::independence_test`
  or `perm`. The `max_consecutive` / block-constrained schedules and the ordered
  dose-response statistic are potential equivalents, to test (no turnkey R function
  reproduces those exact constrained reference sets).

**Reading the output.** Report `p_value` against your pre-set alpha (typically
0.05); with `two_sided=True` it tests any difference, with `two_sided=False` a
pre-specified beneficial direction. `n_perm` (or `n_eval` when the space was fully
enumerated) documents the resolution: the smallest attainable p is `1/(1+n_perm)`,
so `n_perm` must be large enough that this floor sits below alpha. Pre-specify ONE
primary statistic (mean difference by default); running several and keeping the
significant one inflates the error rate (Manolov, Moeyaert and Fingerhut 2021). An
undefined observed statistic returns `p_value = NaN` by design, never a spurious
`1/(1+n_perm)`.

**References.** Edgington and Onghena (2007); Heyvaert and Onghena (2014);
Michiels, Heyvaert and Onghena (2017); Phipson and Smyth (2010); Kratochwill et al.
(2010, 2012); Tate et al. (2016).

## Fiche 2 - Freedman-Lane residual permutation (controlling a nuisance)

**Problem.** Single-case series often drift over time (a learning or spontaneous-
recovery trend). If condition B tends to fall late in the series, a plain label
permutation can confound the condition effect with that trend, and you lose power
or bias the test. You want to test the condition effect NET of a time trend (and
possibly other covariates) while keeping the randomization logic.

**Principle.** Put the time trend (and covariates) into a nuisance design `Z`
(intercept plus session or log-session, plus covariate columns), and test the extra
sum of squares that CONDITION explains beyond `Z` with a partial-F statistic (its
null distribution is free of the nuisance parameters) (Winkler et al. 2014).
Variants of residual permutation differ in WHAT is permuted and when the nuisance is
fitted (Winkler et al. 2014, Table):

- Draper-Stoneman (permute the condition labels, `Z` and the outcome fixed, refit).
  For a randomized SCED with condition orthogonal to time this IS the exact
  Edgington-Onghena test with a richer statistic. It can misbehave only under strong
  condition-time collinearity.
- Freedman-Lane (fit the reduced model `Y ~ Z`, permute its reduced-model residuals,
  rebuild `Y* = Z gamma_hat + P resid`, refit the full `Y* ~ [Z | condition]`,
  recompute the statistic). It keeps the nuisance relationship intact and is the
  best-behaved scheme across regimes, in particular under temporal confounding. It is
  the default.
- ter Braak permutes full-model residuals (robust, slightly conservative in small
  samples).
- Manly permutes the raw outcome (simple, valid only when the nuisance is orthogonal
  to the regressor of interest).
- Kennedy permutes residualized outcome against residualized regressor (can be
  anti-conservative).

Use Freedman-Lane unless you have a specific reason; the toolbox implements
Draper-Stoneman and Freedman-Lane and selects between them automatically. For a
truly randomized SCED the exact design-based route stays the label-permutation
schedule of Fiche 1; Freedman-Lane is the model-based scheme for the confounded /
observational regime.

**In this toolbox.**
- Module: `functions/sced/core.py`.
- Single series:
  `condition_permutation_test(values, labels, sessions, *, detrend="none", method="draper-stoneman", n_perm=5000, max_consecutive=None, block_size=None, covariates=None, enumerate_cap=40320, random_state=0)`
  returns `{observed_F, p_value, method, detrend, df, n_perm, n_eval, exact}`.
  `detrend` in {none, linear, log}; `method` in {draper-stoneman, freedman-lane, auto};
  for Freedman-Lane the n! residual reference set is enumerated exactly when
  `n! <= enumerate_cap` (n <= 8), else Monte-Carlo.
- Group (unit = block):
  `stratified_condition_permutation_test(values, labels, units, sessions, *, detrend="none", method="freedman-lane", n_perm=5000, max_consecutive=None, block_size=None, standardize=True, covariates=None, random_state=0)`
  returns `{observed_F, p_value, method, detrend, df, n_perm, n_units}`. Labels are
  permuted WITHIN each unit (the exact joint reference set); `standardize` z-scores
  the outcome within unit (permutation-invariant, so still exact).
- Scheme chooser:
  `choose_permutation_method(labels, sessions, *, detrend="linear", threshold=0.15)`
  and its group form `choose_permutation_method_group(...)` return
  `{condition_time_R2, threshold, recommended_method, rationale, decision_tree}`;
  the recommendation is Freedman-Lane when condition explains more than `threshold`
  of the temporal variance, else Draper-Stoneman. The static tree is the module
  constant `PERMUTATION_DECISION_TREE`.
- Nuisance internals (documented for auditors): `_nuisance_basis(sessions, detrend)`
  builds the time design; `_append_covariates(Z, covariates, n)` adds ANCOVA-style
  covariate columns. Adjust only for covariates NOT affected by the condition
  (a mediator biases the effect).
- Driving templates: `template_sced_alternating.py` (single and group, via
  `report_sced_alternating`, `perm_method` and `time_covariate` arguments).
- R bridge: `permuco::lmperm` is the direct equivalent - it implements the same GLM
  permutation methods by name (`freedman_lane`, `terBraak`, `manly`, `kennedy`,
  `draper_stoneman`) with the partial statistic. `lmPerm` is a lighter alternative.
  The WITHIN-UNIT stratified permutation (exchangeability blocks: units whose
  labels can be swapped without changing the null) is a potential
  equivalent, to test: `permuco` supports some blocking but not this exact
  z-scored-within-unit stratified scheme; the closest full match is FSL PALM
  (Winkler's own tool) rather than an R package.

**Reading the output.** `observed_F` is the partial F for condition; `p_value` is
its permutation tail (compare to alpha). `df = (df_num, df_den)` and `exact`
document whether the p is exact (full enumeration) or Monte-Carlo. If
`choose_permutation_method` reports `condition_time_R2` above `threshold`, prefer
Freedman-Lane and say so; if it is near zero, Draper-Stoneman is exact and simpler.
The randomization keeps the test valid under autocorrelation; the covariate only
restores the power a trend would otherwise cost.

**References.** Freedman and Lane (1983) (not in wiki); ter Braak (1992) (not in
wiki); Manly (1997) (not in wiki); Kennedy (1995) (not in wiki); Winkler et al.
(2014) (primary reference for residual permutation and the variants; referenced in
the wiki via wawrzyniak-2022 for sign-flipping within exchangeability blocks, no
dedicated source page); Draper and Stoneman (1966); Edgington and Onghena (2007).

## Fiche 3 - Nonparametric effect sizes and segmented regression

**Problem.** A p-value says whether an effect exists, not how large it is. You want
a magnitude that is robust to non-normality and to a single extreme point, ideally
one that also accounts for a baseline trend, and a way to separate an immediate
level jump from a change of slope.

**Principle.** Three complementary descriptive estimands, tied to the same
rank-based logic as the randomization test:

- NAP (nonoverlap of all pairs): the share of A-B pairs in the improving direction,
  ties counted 0.5. NAP = U / (n_A n_B), 0.5 = chance, 1 = full separation.
  Equivalent to `2*NAP - 1 = tau` (A vs B). Preferred over PND (which one extreme
  baseline point dominates). NAP is NOT trend-corrected, so it inflates under a
  baseline drift.
- Tau-U family (Parker, Vannest, Davis and Sauber 2011), bounded in [-1, +1] by
  construction. The toolbox refuses the original unbounded `(S_AB - S_A)/(n_A n_B)`
  denominator (which yields |tau| > 1) and offers two bounded, trend-controlling
  variants selected by a decision flowchart (Fingerhut, Xu and Moeyaert 2021):
  `adj` = Tarlow (2016) Baseline Corrected Tau (Theil-Sen (robust trend-line slope)
  detrend of the whole
  series, then Kendall tau-b on residuals; preferred when n_A >= 7) and `trend_a` =
  Brossart (2018) bounded denominator `(S_AB - S_A)/(n_AB + n_A)` (fallback for short
  baselines). `method="auto"` corrects only when a baseline trend is present
  (trend p < 0.05 OR |Tau_A| >= 0.40).
- Segmented (interrupted time series) regression: a GLM
  `outcome ~ time + phase + time_since_intervention` (Wagner 2002 coding), where
  `time_since` is 0 in A and 0 at the first B session then 1, 2, ..., so the `phase`
  coefficient reads as the immediate LEVEL change and `time_since` as the SLOPE
  change. Family in {gaussian, poisson, binomial} for continuous / count / binary.

**In this toolbox.**
- Module: `functions/sced/core.py`.
- `nap(baseline, treatment, *, improvement="increase")` -> float; with inference
  `nap_ci(baseline, treatment, *, improvement="increase", se_method="unbiased", confidence=0.95)`
  -> `{nap, se, ci_lower, ci_upper, method, n_baseline, n_treatment}` (verified
  against SingleCaseES 0.7.4).
- `tau_u(baseline, treatment, *, correct_baseline_trend=True, improvement="increase", method="auto", min_baseline_for_adj=7, alternative="two-sided")`
  -> `{tau, tau_u, variant, bounded, alternative, baseline_trend_tau, baseline_trend_p, se, p_value, ci, critical, n_baseline, n_treatment}`
  (verified against R `scan::tau_u` and `scan::corrected_tau`).
- `segmented_regression(df, *, session_col, phase_col, outcome_col, baseline_label, treatment_label, family="gaussian")`
  -> (the fitted results, or None if the fit failed, plus an info dict) with
  `info["Level change (coef, p)"]` and
  `info["Slope change (coef, p)"]`; tidy view via
  `segmented_table(res, *, exponentiate=False)` (verified against `scan::plm`).
- Descriptive backbone: `describe_by_phase(...)`; magnitude bands
  `nap_magnitude`, `tau_magnitude`, `pnd_magnitude`, `pem_magnitude`.
  `pnd` / `pem` are provided for comparability only (PND rejected by WWC 2012).
- These are reported automatically by `pipeline_sced_alternating`,
  `report_sced_phase_design` and `report_sced_multiple_baseline`.
- R bridge: `SingleCaseES` (Pustejovsky) for NAP with SE/CI and Tau variants, and
  `scan` (Wilbert and Lueke) for `tau_u` / `corrected_tau` / `plm` - the code is
  numerically verified against both, so these are exact equivalents. Segmented ITS
  also maps to `nlme::gls` with `corAR1` or the `segmented` package.

**Reading the output.** Read the magnitude WITH its interval, never the point value
alone. NAP bands (Parker and Vannest 2009): 0.50-0.65 weak, 0.66-0.92 medium,
0.93-1 strong. Tau-U bands (Vannest and Ninci 2015, indicative only): < 0.20
negligible, 0.20-0.60 moderate, 0.60-0.80 strong, > 0.80 very strong; the more
principled decision is the Critical Tau-U (the value the estimate must exceed at
alpha, `critical` in the returned dict). Check `variant` to see which trend
correction fired and `bounded` to confirm the coefficient stayed in [-1, 1]. For
segmented regression, read the level and slope coefficients with their p-values
together: a significant level with a null slope is an immediate, sustained shift; a
null level with a significant slope is a progressive effect. The magnitude complements
the randomization p; it does not replace it (Manolov et al. 2021).

**References.** Parker and Vannest (2009); Parker, Vannest, Davis and Sauber (2011);
Brossart, Laird and Armstrong (2018); Tarlow (2016); Fingerhut, Xu and Moeyaert
(2021); Manolov, Moeyaert and Fingerhut (2021); Kratochwill et al. (2012); Wagner
(2002) (not in wiki).

---

# Part II - Model-based frequentist (piecewise multilevel GLM)

## Fiche 4 - Piecewise interrupted-time-series mixed model (MBD)

**Problem.** In a multiple-baseline design several cases receive the intervention at
staggered times. You want one model that pools the cases, estimates the average
immediate jump AND the average change of slope, reports how much cases differ, and
still gives a trustworthy p-value even though there are only a handful of cases.

**Principle.** A piecewise (interrupted-time-series) mixed model (Van den Noortgate
and Onghena 2003; Moeyaert, Ferron, Beretvas and Van den Noortgate 2014) fits, per
case,

    Y = b0 + b1*time + b2*phase + b3*(treatment trajectory) + random effects + e

with `b1` = baseline trend, `b2` = immediate level change at onset, `b3` = slope
change. Two codings of the treatment trajectory are offered: `continuous`
(`time_since` = max(0, session - start), a single sustained B-phase slope) and
`two_piece` (a ramp `h` that changes linearly over the first `ramp` treatment
observations then plateaus; Cheng et al. 2025). Under `two_piece`, `b2` is the
STABILISED plateau effect and `b3` is the slope of the initial rise - the
recommended coding for an effect that grows then levels off (a TMS-like pattern).

Crucially, the mixed model supplies the ESTIMATES (b1, b2, b3, variance components,
ICC (intraclass correlation: share of variance between cases)) but the P-VALUES come
from the design-based randomization test on the
intervention start (Fiche 1 applied to the MBD schedule). The reason: with the few
cases typical of an MBD, the mixed model's asymptotic p is untrustworthy, and
statsmodels offers neither AR(1) errors nor Kenward-Roger (a small-sample correction
to the mixed-model degrees of freedom and p-value) degrees of freedom. When R
is available, the toolbox additionally reports the canonical Kenward-Roger p from
`lmerTest`. The randomization schedule for the start points follows the Levin et al.
(2016) taxonomy: MB (Marascuilo-Busk, start with replacement), WW (Wampold-Worsham,
case permutation), MB-R (restricted, without replacement), and the ordered /
stepwise variants; the exact p is obtained by enumeration or, for the additive
independent-MB statistic, by convolution (no Monte-Carlo error).

**In this toolbox.**
- Modules: `functions/sced/mbd/multilevel.py`, `mbd/baseline.py`, `mbd/procedures.py`,
  `mbd/report.py`.
- Piecewise model:
  `multilevel_mbd_model(df, *, tier_col, session_col, outcome_col, phase_col=None, intervention_starts=None, treatment_phase=None, outcome_type="continuous", improvement="increase", treatment_trajectory="continuous", ramp=3, min_baseline=3, min_treatment=3, n_perm=5000, random_state=0)`
  -> `(summary, sheet)`; `summary` holds `level_change_b2`, `slope_change_b3`,
  `baseline_trend_b1`, ICC, `p_level_change_perm`, `p_slope_change_perm`, and, when R
  is present, `p_b1_kr` / `p_b2_kr` / `p_b3_kr` with `canonical_ddf`. Design builder:
  `_piecewise_design(df, *, tier_col, session_col, outcome_col, starts, treatment_trajectory, ramp)`
  -> `(frame, slope_col_name)`.
- Randomization engine:
  `multiple_baseline_test(df, *, tier_col, session_col, outcome_col, phase_col=None, baseline_phase=None, treatment_phase=None, intervention_starts=None, candidates=None, entry_points=None, scheme="independent", statistic="level", improvement="increase", min_baseline=3, min_treatment=3, min_gap=None, n_perm=5000, enumerate_cap=200000, conv_cap=20000000, random_state=0)`
  -> `{observed, p_value, n_eval, exact, scheme, statistic, starts, per_tier}`
  (verified against `scan::rand_test`). Feature-specified dispatcher:
  `mbd_randomization_test(df, *, case_col, session_col, outcome_col, phase_col=None, intervention_starts=None, treatment_phase=None, case_randomization, start_point_randomization, replacement=False, comparison="within", windows=None, improvement="increase", statistic="level", n_perm=5000, random_state=0)`;
  procedure recommender `choose_mbd_procedure(*, n_cases, equal_series, n_start_points=1, effect_type="immediate", require_stagger=True)`;
  helper `_derive_starts(df, case_col, session_col, phase_col, intervention_starts, treatment_phase=None)`
  -> `{case: start_session}`.
- Model selection / sensitivity:
  `mbd_model_selection(df, *, tier_col, session_col, outcome_col, phase_col=None, intervention_starts=None, treatment_phase=None, ramp=3, random_structure=False, candidates=("immediate", "continuous", "two_piece"))`
  (AIC / BIC / LRT reported as a multiverse check on the fixed structure, NOT as a
  selector; Manolov and Moeyaert 2025).
- Whole pipeline:
  `pipeline_sced_multiple_baseline(df, *, tier_col, session_col, outcome_col, phase_col=None, ..., case_randomization=False, start_point_randomization=True, replacement=False, comparison="within", statistic="level", multilevel=False, treatment_trajectory="continuous", ramp=3, model_selection=False, ..., n_perm=5000, random_state=0, ...)`
  -> `(model_info, result)`.
- Driving template: `template_sced_multiple_baseline.py` ->
  `report_sced_multiple_baseline(df, *, tier_col="tier", session_col="session", phase_col="phase", ..., statistic="level", multilevel=True, treatment_trajectory="continuous", ramp=3, model_selection=False, bc_smd=True, bc_smd_trend=True, n_perm=5000, save_path=None, output_dir=None, ...)`.
- R bridge: `nlme::lme` or `lme4::lmer` for the mixed model, with `lmerTest` for the
  Kenward-Roger p (the toolbox literally calls `lmerTest::lmer` for its canonical
  path); `scan::hplm` is the exact SCED wrapper the fixed effects were verified
  against. The design-based p on the start points maps to `SCRT` / the `scan`
  randomization functions - the MB / WW / MB-R / KL / Rev schemes and the convolution
  exact path are potential equivalents, to test (no single R function reproduces all
  of them).

**Reading the output.** Inference verdict follows this rule: prefer the canonical
Kenward-Roger p (`p_b2_kr`, `p_b3_kr`) when R is available, else the permutation p
(`p_level_change_perm`, `p_slope_change_perm`); compare to alpha. Read `b2` and `b3`
in original outcome units: `b2` is the immediate jump, `b3` the slope change (under
`two_piece`, `b2` is the plateau and `b3` the initial rise). ICC is the share of
variance between cases. Treat `mbd_model_selection` AIC/BIC/LRT as a sensitivity
check: report whether the sign and significance of the effect hold across the
immediate / continuous / two-piece specifications, do not use it to pick the
hypothesis (that is a theory decision).

**References.** Van den Noortgate and Onghena (2003); Moeyaert, Ferron, Beretvas and
Van den Noortgate (2014); Ferron, Bell, Hess, Rendina-Gobioff and Hibbard (2009);
Cheng, Yi, Moeyaert, Beretvas, Van den Noortgate and Ferron (2025); Manolov and
Moeyaert (2025); Levin, Ferron and Gafurov (2016); Bouwmeester and Jongerling
(2020); De, Michiels, Tanious and Onghena (2020); Baek et al. (2014); Wagner (2002)
(not in wiki).

## Fiche 5 - Design-comparable between-case SMD (scdhlm)

**Problem.** The `b2` from Fiche 4 is in raw outcome units, so it cannot be compared
with a Cohen's d from a between-subjects RCT or meta-analysed across studies on
different scales. You want a standardized effect on the SAME metric as a group d.

**Principle.** The between-case standardized mean difference g_AB (Hedges,
Pustejovsky and Shadish; Pustejovsky, Hedges and Shadish 2014) standardizes the
treatment effect by the BETWEEN-case standard deviation `sqrt(tau_intercept^2 +
sigma^2)`, putting it on the same scale as a group Cohen's d and making it
meta-analysable. The canonical estimator fits `lme(y ~ phase, random = ~1|case,
corAR1(~time|case))` by REML and applies `g_mlm` with a Satterthwaite small-sample
correction. Compare the raw specification (`y ~ phase`) against the trend-corrected
one (`y ~ time + phase`): a g that collapses once the baseline trend is removed is an
effect largely carried by maturation.

**In this toolbox.**
- Module: `functions/sced/mbd/multilevel.py`.
- Canonical (R) estimator:
  `bc_smd_scdhlm(df, *, tier_col, session_col, outcome_col, phase_col, baseline=None, treatment=None, improvement="increase", trend=False, timeout=180)`
  -> `{available, g_AB, SE, df, ci_low, ci_high, method, trend}` when R and `scdhlm`
  are present, else `{available: False, message}`. `trend=False` uses
  `g_mlm(p_const=c(0,1), r_const=c(1,0,1))` (raw level); `trend=True` uses
  `y ~ time + phase` with `p_const=c(0,0,1)` (level net of a linear baseline trend).
- Python approximation (no R needed):
  `between_case_smd(df, *, tier_col, session_col, outcome_col, phase_col=None, intervention_starts=None, treatment_phase=None, model="level", treatment_trajectory="continuous", ramp=3, min_baseline=3, min_treatment=3)`
  -> `{d, g, se, ci_low, ci_high, J, df, n_cases, n_obs, b2, between_var, within_var, model}`;
  Welch-Satterthwaite df (small-sample degrees-of-freedom correction), Hedges J
  correction, Student CI. It closely approximates
  scdhlm but statsmodels MixedLM has no AR(1), so a residual ~5-10% gap on g remains -
  use `bc_smd_scdhlm` for a publishable value.
- Also exposed: the canonical fixed-effect model
  `hplm_R(df, *, tier_col, session_col, outcome_col, phase_col, baseline=None, treatment=None, improvement="increase", ddf="Kenward-Roger", timeout=240)`
  -> `{available, b0..b3, se_b*, df_b*, p_b* (KR), icc, random, ddf, method}`.
- Driving template: `template_sced_multiple_baseline.py` (`bc_smd=True`,
  `bc_smd_trend=True` emit both columns).
- R bridge: `scdhlm` (Pustejovsky) `lme` + `g_mlm` - this is not an approximation, the
  toolbox calls scdhlm directly; the exact equivalent IS the R package.

**Reading the output.** Read `g_AB` (or `g`) against Cohen's benchmarks (0.2 small,
0.5 medium, 0.8 large) with its CI; if the CI excludes 0 the effect is
design-comparable-significant. Always report the raw and the `trend=True` values side
by side: a large drop signals maturation rather than treatment. Requires >= 2 cases.
When `available` is False, fall back to `between_case_smd` and flag it as an
approximation.

**References.** Pustejovsky, Hedges and Shadish (2014); Valentine, Tanner-Smith,
Pustejovsky and Lau (2016); Shadish (2014); Ugille, Moeyaert, Beretvas, Ferron and
Van den Noortgate (2012).

## Fiche 6 - 3-level within-visit variance decomposition

**Problem.** Each visit carries several raw measures (for example 8 trials). Ignoring
that structure (averaging silently) throws away information and mis-states the
uncertainty. You want to know how the total variance splits between patients, between
visits within a patient, and between the repeated measures within a visit - and
whether the condition affects the visit mean, its dispersion, or its within-visit
slope.

**Principle.** Fit a 3-level mixed model `outcome ~ condition (+ time) + (1|patient) +
(1|visit-in-patient)` on the raw measures, which decomposes the variance into
between-patient, between-visit (within patient), and within-visit (residual
measurement noise) components, with the corresponding ICCs. The condition effect is
estimated there, but its p-value comes from a stratified randomization test on the
VISIT MEANS, because the visit is the unit of assignment (consistent with the rest of
the toolbox). Two focused companions test whether the condition changes the
within-visit dispersion (consistency) or the within-visit slope (fatigue / warm-up).

**In this toolbox.**
- Module: `functions/sced/intravisit.py`; reshapers in `functions/sced/prep.py`.
- `variance_decomposition_3level(df, *, unit_col, session_col, condition_col, outcome_col, conditions=None, detrend="none", method="freedman-lane", n_perm=5000, max_consecutive=None, standardize=True, random_state=0)`
  -> `(summary, sheet)` with `var_patient`, `var_visit_within_patient`,
  `var_within_visit`, their percentages, `icc_patient`, `condition_effect`,
  `p_common_effect_perm`.
- `dispersion_test(df, *, unit_col, session_col, condition_col, outcome_col, measure="sd", conditions=None, detrend="none", method="freedman-lane", n_perm=5000, max_consecutive=None, standardize=True, random_state=0)`
  -> `(result_dict, visit_level_df)` (`measure` in {sd, cv}).
- `within_visit_slope_test(df, *, unit_col, session_col, condition_col, outcome_col, trial_col, conditions=None, detrend="none", method="freedman-lane", n_perm=5000, max_consecutive=None, standardize=True, random_state=0)`
  -> `(result_dict, visit_level_df)`; the result carries `slope_by_condition`,
  `slope_by_patient`, `slope_by_patient_condition`.
- Reshapers: `aggregate_visits(df, *, unit_col, session_col, condition_col, outcome_col, stat="mean")`,
  `aggregate_runs(df, *, session_col, condition_col, run_cols, unit_col=None, agg="mean")`
  -> `(df_visit, long_runs, out_col)`,
  `collapse_long_measures(df, *, session_col, condition_col, value_col, unit_col=None, agg="mean", trial_col=None)`
  -> `(df_visit, long_measures, out_col)`,
  `unstack_outcomes(df, *, outcome_name_col, value_col, id_cols, trial_col=None, keep_cols=None)`.
- Driving template: `template_sced_alternating.py` via
  `report_sced_alternating(..., level3=True, dispersion="sd", within_visit_slope=True, trial_col=...)`.
- R bridge: `lme4::lmer(outcome ~ condition + (1|patient/visit))` or
  `nlme::lme` with nested random effects for the variance components; `performance::icc`
  for the ICCs. The visit-level stratified permutation p is a potential equivalent, to
  test (no standard R function ties a 3-level decomposition to a within-unit
  randomization p; closest: `SCRT` for the permutation part, `lme4` for the variance
  part, combined by hand).

**Reading the output.** Read the three variance shares (`pct_patient`, `pct_visit`,
`pct_within_visit`) to see where the noise lives: a high within-visit share means the
measurement itself is noisy and more trials per visit help; a high between-visit share
means visit-to-visit instability dominates. The condition verdict uses
`p_common_effect_perm` (visit-level permutation) against alpha, NOT the model's
asymptotic p. `dispersion_test` answers "does the condition change consistency?" and
`within_visit_slope_test` "does it change the within-visit trajectory?" (negative mean
slope = decline / fatigue, positive = warm-up).

**References.** Baek, Moeyaert, Petit-Bois, Beretvas, Van den Noortgate and Ferron
(2014); Ferron, Bell, Hess, Rendina-Gobioff and Hibbard (2009); Moeyaert, Ferron,
Beretvas and Van den Noortgate (2014); Cheng et al. (2025).

---

# Part III - Model-based Bayesian

## Fiche 7 - Per-case Bayesian model (simple / BITS / BUCP)

**Problem.** For a single case with a short series you want more than a yes/no
p-value: a full probability statement about the effect size, the immediate jump, the
slope change, and the total effect at the end of the phase, that behaves sensibly
with few data points and accounts for autocorrelation - and, optionally, that
estimates WHEN the change actually happened rather than assuming the designed
boundary.

**Principle.** Fit an AB Bayesian regression on one time-ordered series with priors
that regularize the few-data regime. Three models: `simple` (i.i.d. noise, no
autocorrelation), `bits` (Bayesian interrupted time series with a lag-1 AR(1) on the
residuals; Natesan and Hedges 2017), and `bucp` (Bayesian unknown change point,
treating the switch index as an estimated integer parameter rather than fixing it at
the designed boundary). The estimands, all oriented so positive = improvement, are:
`es` (standardized effect (beta_B - beta_A)/sigma), `level_change` = b2 (raw jump),
`slope_change` = b3, `effect_end` = b2 + b3*duration (the total effect at the end of B,
which is better identified than b2 or b3 alone because the level-slope tradeoff
cancels in the sum), and for BUCP the posterior of the change point `cp` (its
concentration measures immediacy). Inference is Bayesian: the Highest Density Interval
(HDI, the narrowest interval holding a given posterior mass), the probability of
direction (pd, the share of the posterior on the sign of the median), and the ROPE
(Region of Practical Equivalence) decision (Kruschke 2018).

**In this toolbox.**
- Module: `functions/sced/mbd/bayesian.py`.
- `bayes_phase_model(y, *, start_index, model="bits", improvement="increase", rope=0.5, trend=False, baseline_trend=False, prior_mean=None, sigma_max=None, draws=2000, tune=2000, chains=4, seed=42, progressbar=False, cache_dir=None, force_refit=False)`
  (`model` in {simple, bits, bucp}; `start_index` = 0-based first B index) ->
  dict `{model, n, n_a, n_b, es, level_change, rho, cp (bucp), p_es_gt_rope, baseline_trend_tau, baseline_trend_p, rhat_max, ess_min, diverging, ...}`;
  with `trend=True` it also returns `slope_change`, `effect_end`, `es_end`,
  `p_es_end_gt_rope`. Each estimand sub-dict has `{median, mean, sd, hdi_low, hdi_high, pd}`.
- ROPE decision helper:
  `_rope_stats(draws, rope, prob=0.95)` -> `{pct_in_rope, p_gt_rope, rope_decision, rope}`
  with the 3-way verdict effect / equivalence / undecided.
- Report level (per-case, alternating condition design):
  `report_sced_bayesian_condition(df, *, session_col="session", condition_col="condition", outcome_col="score", conditions=("A","B"), reference=None, target=None, outcome_type="continuous", improvement="increase", prior_mean=None, prior_sd=None, prior_scale=None, group_cols=None, random_slope=False, time_covariate="none", covariate_cols=None, hdi_prob=0.95, rope=None, draws=2000, tune=1000, chains=4, random_state=0, save_path=None, output_dir=None, diagnostics=False, verbose=True)`
  (module `functions/sced/bayes/condition.py`), which calls
  `bayesian_condition_effect(...)` and reports, per contrast, the effect size (Cohen's
  d / odds ratio / rate ratio), a 95% credible interval, `p_benefit`, and a
  Savage-Dickey Bayes factor `bf10` (evidence ratio for vs against the effect).
- Per-case MBD phase report with per-patient model selection:
  `report_sced_bayesian_percase(...)` (Fiche 10) and the phase engine
  `report_sced_bayesian_mbd(..., onset="unknown")` route to the BUCP path.
- Sampling cache: `fit_or_load(cache_dir, prefix, spec, fit_fn, *, force_refit=False, verbose=False)`
  (`functions/sced/model_cache.py`) reloads a matching netCDF instead of re-running MCMC.
- Driving templates: `template_sced_bayesian.py` (condition effect + Bayes factor) ->
  `report_sced_bayesian_condition`; `template_sced_bayesian_percase.py` ->
  `report_sced_bayesian_percase`.
- R bridge: `brms` for the `simple` and `bits` models - `brm(y ~ phase (+ time),
  autocor = ~ar(time, case, p = 1))` reproduces the likelihood and structure; the
  toolbox's priors (sceptical b2, Half-Normal SDs, Uniform(-1,1) AR) differ - set brms
  priors to match;
  `bayestestR` (Makowski, Ben-Shachar and Ludecke 2019) computes the HDI, pd and ROPE
  from the posterior. The BUCP unknown-change-point model has no turnkey R equivalent:
  the original implementations use JAGS / Stan directly (Natesan and Hedges 2017;
  Aerts 2015); a `brms` nonlinear / mixture change-point formulation is a potential
  equivalent, to test.

**Reading the output.** Decision rule, in order: (1) convergence gate - require
`rhat_max` < 1.01, `ess_min` adequate (a few hundred), and `diverging` = 0, else
increase `draws` / `tune` before interpreting anything. (2) Direction - `pd` (or
`p_benefit`) >= 0.95 is strong evidence of a beneficial effect; 0.90-0.95 is
suggestive. (3) Practical size - the ROPE decision: HDI entirely outside
[-rope, +rope] = a practically non-zero effect (accept the effect), HDI entirely
inside = practical equivalence (accept the null), overlap = undecided (need more
data). Set `rope` in outcome units or an MCID. Report `effect_end` (total end-of-B
effect) as the headline estimand. For BUCP, a concentrated `cp` posterior (high
`concentration`) means an abrupt, immediate change; a diffuse one means a gradual or
uncertain onset.

**References.** Natesan and Hedges (2017); Aerts (2015); Kruschke (2018)
(not in wiki); Makowski, Ben-Shachar and Ludecke (2019); Rindskopf (2014) (not in
wiki); the wiki method pages `BayesianUnknownChangePoint.md` and `bayestestR.md`.

## Fiche 8 - Hierarchical Bayesian model (partial pooling)

**Problem.** With several cases you want both the POPULATION effect and a stable
per-case estimate, without fitting each case in isolation (noisy) or ignoring that
they differ (over-confident). You also want to know how much the cases genuinely
differ.

**Principle.** One joint model over all cases with per-case random effects drawn from
a common population distribution (partial pooling): `y_it = b0_i + b1_i*time +
b2_i*phase + b3_i*time_B + AR(1)`, with `(b0, b1, b2, b3)_i ~ Normal(mu, diag(tau))`.
Partial pooling shrinks each case toward the population, borrowing strength across
cases so per-case estimates are stable even on short series, while `mu` gives the
population effect and `tau` its between-case SD. The mean structure is pre-specified by
a hypothesis (not data-driven): immediate_flat (b2 only), cumulative_flat (b2 + b3),
immediate_trend (b1 + b2), cumulative_trend (b1 + b2 + b3). Priors are weakly
informative (sceptical b2, regularizing b1/b3, AR(1) rho ~ Uniform(-1, 1), residual
and between-case SD Half-Normal). Reported effect: the population level in original
units with HDI, pd, and P(effect > ROPE). This is the Bayesian counterpart of the
frequentist multilevel model of Fiche 4.

**In this toolbox.**
- Module: `functions/sced/mbd/bayesian.py`.
- `bayes_hier_sced(df, *, tier_col, session_col, phase_col, outcome_col, baseline=None, treatment=None, improvement="increase", hypothesis="immediate_flat", terms=None, pooling="partial", family="gaussian", bounds=None, n_trials=None, ar=True, rope=0.5, draws=2000, tune=2000, chains=4, seed=42, target_accept=0.95, progressbar=False, cache_dir=None, force_refit=False)`
  -> `{population, heterogeneity, per_case, rhat_max, ess_min, diverging, ...}`.
  `pooling` in {partial, random_intercept, correlated, none, meta}; `family` in
  {gaussian, student, beta, binomial}; `terms` (explicit subset of trend/level/slope)
  overrides `hypothesis`. Parameter counter for complexity:
  `hier_param_count(terms, n_cases, pooling="partial", family="gaussian", ar=False)`.
- Report level:
  `report_sced_bayesian_mbd(df, *, pooling="partial", onset="known", ar=True, ..., tier_col="tier", session_col="session", phase_col="phase", ..., improvement="increase", rope=0.5, hypothesis="immediate_flat", terms=None, family="gaussian", bounds=(0,100), n_trials=None, target_accept=0.95, trend=False, baseline_trend=False, min_per_phase=3, draws=2000, tune=2000, chains=4, seed=42, save_path=None, output_dir=None, diagnostics=False, mcid=None, cache_dir=None, force_refit=False, forest_estimand="auto", ...)`
  (module `functions/sced/bayes/report.py`). Three orthogonal axes select the engine:
  `pooling` (sharing across cases), `onset` (known phase label -> hierarchical;
  unknown -> BUCP, forces `pooling="none"`), `ar` (intra-case AR(1)).
- Driving template: `template_sced_bayesian_mbd.py` -> `report_sced_bayesian_mbd`.
- R bridge: `brms` is the direct equivalent - a multilevel `brm` with per-case random
  effects (`(1 + phase + time_B | case)`), family gaussian / student / Beta / binomial,
  and `ar(p = 1)` autocorrelation reproduces the likelihood and structure; the
  toolbox's priors (sceptical b2, Half-Normal SDs, Uniform(-1,1) AR) differ - set brms
  priors to match; `bayestestR` reads the HDI /
  pd / ROPE. The `correlated` (LKJ (prior over correlation matrices)) pooling maps to
brms correlated random effects; the
  `meta` pooling is Fiche 9.

**Reading the output.** Same convergence gate as Fiche 7 (`rhat_max` < 1.01,
`diverging` = 0). Read `population` (median, HDI, pd) in outcome units against alpha /
ROPE exactly as in Fiche 7. Read `heterogeneity` (`tau`, and if reported I^2 and the
prediction interval) to judge how much cases differ - but at fewer than about 5 cases,
tau / I^2 are unstable, so do not conclude on heterogeneity there (Moeyaert et al.
2017; Baek et al. 2014). The shrunken `per_case` effects are the right estimate for
individual PREDICTION; for a responder DECISION prefer the raw per-case value (Fiche 9
note). If the population effect is credible but the prediction interval crosses 0, a
future patient may still not respond - report that.

**References.** Van den Noortgate and Onghena (2003); Moeyaert, Ferron, Beretvas and
Van den Noortgate (2014); Ferron, Bell, Hess, Rendina-Gobioff and Hibbard (2009);
Baek et al. (2014); Rindskopf (2014) (not in wiki); Williams, Rast and Burkner (2018)
(not in wiki); Makowski, Ben-Shachar and Ludecke (2019).

## Fiche 9 - Bayesian meta-analysis (two-stage)

**Problem.** You have a per-case effect and its standard error (from stage-1 fits) and
want to pool them into a population effect that acknowledges the cases are a sample
from a wider population, plus a prediction for a new case - without the one-stage model
over-shrinking away real between-case differences.

**Principle.** A two-stage random-effects meta-analysis. Stage 1: fit each case
independently (`pooling="none"`), extract the effect and its posterior SD. Stage 2: a
normal-normal random-effects model, `theta_i = mu + tau*z_i` (non-centered to avoid
funnel divergences), `obs_i ~ Normal(theta_i, se_i)`. It returns the population effect
`mu`, the between-case SD `tau`, the I^2 (share of total variability due to true
heterogeneity; Higgins and Thompson 2002), a prediction interval for a new case, and
shrunken per-case effects. Compared with the one-stage hierarchical model (Fiche 8),
the two-stage per-case effects are not contaminated by the group (they are shrunk only
at stage 2), which is preferable when the cases differ in the TYPE of effect (one level,
one slope) and the decision is per-case. Both are Bayesian meta-analyses; if they
diverge strongly on `mu`, that is a heterogeneity signal to report, not a bug.

**In this toolbox.**
- Module: `functions/sced/mbd/meta.py`.
- `bayes_meta_analysis(effects, ses, *, labels=None, improvement="increase", draws=2000, tune=2000, chains=4, seed=42, target_accept=0.95, prior_mu_sd=None, prior_tau_sd=None, progressbar=False)`
  -> `{population, heterogeneity{tau, I2, prediction_interval}, per_case, k, improvement, rhat_max, diverging, _idata}`.
- From a stage-1 idata:
  `meta_from_idata(idata, *, estimand="effect_end", labels=None, **kwargs)` (default
  estimand = the total end-of-B effect); row formatter `meta_summary_rows(result, *, label="effect")`;
  report writer `write_meta_report(results, save_path, *, unit="pts", estimand_label="Total end-B effect")`.
- Report level: reached via `report_sced_bayesian_mbd(..., pooling="meta")`.
- Driving template: `template_sced_bayesian_mbd.py` with `pooling="meta"`.
- R bridge: `brms` random-effects meta - `brm(effect | se(se) ~ 1 + (1 | case))` -
  reproduces the non-centered normal-normal pooling; `bayesmeta` (Rover) is a dedicated
  Bayesian random-effects meta package; frequentist counterpart `metafor::rma`
  (with `predict()` for the prediction interval and I^2). These are confident matches
  for the stage-2 model.

**Reading the output.** Convergence gate first (`rhat_max` < 1.01, `diverging` = 0).
Read `population` (mu with HDI / pd) against alpha / ROPE. Read `tau` (between-case
effect SD in outcome units) and `I2` (0.25 / 0.50 / 0.75 = weak / moderate / strong;
Higgins and Thompson 2002) together, and only when k >= 5. The `prediction_interval`
is the plausible effect for a NEW case: if it crosses 0, a future patient may not
respond even when `mu` is credible - this is the individual prognosis, wider than the
HDI of `mu`. Meta-analysis in raw points is valid only within one scale; across scales
use the BC-SMD of Fiche 5 (standardized SCED effects are inflated versus Cohen's d;
Ugille et al. 2012).

**References.** Van den Noortgate and Onghena (2008); Van den Noortgate and Onghena
(2024); Ugille, Moeyaert, Beretvas, Ferron and Van den Noortgate (2012); Shadish
(2014); Burke, Ensor and Riley (2017); Higgins and Thompson (2002) (not in wiki).

## Fiche 10 - Model comparison to identify the supposed effect (LOO / WAIC / stacking)

**Problem.** Competing hypotheses about the effect shape (level-only vs level+slope,
different likelihood families, different pooling) each fit a different model. You want
a principled, out-of-sample criterion to say which shape the data support, and an
honest verdict when the data cannot tell them apart.

**Principle.** Compare the models by their expected out-of-sample predictive accuracy.
PSIS-LOO (Pareto-smoothed importance-sampling leave-one-out) and WAIC (widely
applicable information criterion) estimate the expected log predictive density
(Vehtari, Gelman and Gabry 2017; Watanabe 2010). Stacking weights (Yao et al. 2018)
and pseudo-BMA+ combine the models into a predictive average. Only the MEAN structure
is compared here (terms among trend / level / slope); the error structure (AR) and the
pooling are held fixed as sensitivity axes. AR is off by default because
per-observation LOO/WAIC assumes conditional independence. The default candidate set
is the four nested models: M0 (trend only = "no effect"), Mi (trend + level =
immediate jump), Mg (trend + slope = gradual), Mf (trend + level + slope = both).

**In this toolbox.**
- Module: `functions/sced/mbd/bayesian.py`.
- `compare_sced_models(df, *, tier_col, session_col, phase_col, outcome_col, models=None, baseline=None, treatment=None, improvement="increase", family="gaussian", bounds=None, n_trials=None, pooling="partial", ar=False, rope=0.5, draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99, se_mult=2.0, progressbar=False, cache_dir=None, force_refit=False)`
  -> `{compare (LOO), compare_waic, weights (stacking), weights_pbma, best, best_waic, loo_waic_agree, decisive, decision, models}`.
  Default `models` = `_DEFAULT_MODELS` (M0 trend, Mi trend+level, Mg trend+slope, Mf
  trend+level+slope).
- Per-patient model selection (independent per-case, LOO-ranked):
  `select_percase_models(df, *, tier_col, session_col, phase_col, outcome_col, models=None, ..., family="beta", ..., se_mult=2.0, cache_dir=None, force_refit=False, progressbar=False)`.
- Report level:
  `report_sced_model_compare(df, *, tier_col="tier", session_col="session", phase_col="phase", ..., family="gaussian", bounds=(0,100), n_trials=None, models=None, pooling="partial", ar=False, rope=0.5, se_mult=2.0, draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99, save_path=None, output_dir=None, diagnostics=False, cache_dir=None, force_refit=False, per_model_reports=True, ...)`
  and the per-patient report
  `report_sced_bayesian_percase(df, *, tier_col="tier", ..., family="beta", bounds=(0,100), n_trials=None, models=None, rope="auto", se_mult=2.0, draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99, save_path=None, output_dir=None, ...)`
  (module `functions/sced/bayes/report.py`).
- Driving templates: `template_sced_model_compare.py` -> `report_sced_model_compare`;
  `template_sced_bayesian_percase.py` -> `report_sced_bayesian_percase`.
- R bridge: the `loo` package is the exact equivalent - `loo()`, `waic()`,
  `loo_compare()`, and `loo_model_weights(method = "stacking")` /
  `method = "pseudobma"` reproduce PSIS-LOO, WAIC, stacking and pseudo-BMA+; models
  fit with `brms` expose `add_criterion(fit, "loo")`. Confident match.

**Reading the output.** Decision rule (parsimony): the best model by LOO is declared
DECISIVE only if its `elpd_diff` advantage over the runner-up exceeds
`se_mult` * `dse` (default 2 standard errors of the difference); otherwise the verdict
is INDISTINGUISHABLE and you keep the simplest model or report the stacking-weighted
average effect - the field `decision` states which. Cross-check `loo_waic_agree` (LOO
and WAIC should pick the same winner) and the per-model `k>0.7` count (Pareto-k above
0.7 flags unreliable LOO for that model - refit or interpret with caution). The
`weights` (stacking) show how much predictive mass each model carries. Do not read a
decisive winner as "the effect is real"; it says which effect SHAPE predicts best,
conditional on the fixed AR / pooling axes.

**References.** Vehtari, Gelman and Gabry (2017) (not in wiki); Watanabe (2010) (not in
wiki); Yao, Vehtari, Simpson and Gelman (2018) (not in wiki); Manolov and Moeyaert
(2025); Natesan Batley et al. (2020).

---

## References

Wiki-sourced entries carry the DOI recorded in the source frontmatter. Entries marked
"(not in wiki)" are cited from established literature and should be verified before
external reuse.

- Aerts, X. Q. (2015). Time Series Data Analysis of Single Subject Experimental Designs
  Using Bayesian Estimation [Doctoral dissertation, University of North Texas]. ProQuest
  Dissertations and Theses (No. 10032200).
- Baek, E. K., Moeyaert, M., Petit-Bois, M., Beretvas, S. N., Van den Noortgate, W., and
  Ferron, J. M. (2014). The use of multilevel analysis for integrating single-case
  experimental design results within a study and across studies. Neuropsychological
  Rehabilitation. doi:10.1080/09602011.2013.835740
- Bouwmeester, S., and Jongerling, J. (2020). Power of a randomization test in a single
  case multiple baseline AB design. PLoS ONE. doi:10.1371/journal.pone.0228355
- Brossart, D. F., Laird, V. C., and Armstrong, T. W. (2018). Interpreting Kendall's Tau
  and Tau-U for single-case experimental designs. Cogent Psychology.
  doi:10.1080/23311908.2018.1518358
- Burke, D. L., Ensor, J., and Riley, R. D. (2017). Meta-analysis using individual
  participant data: one-stage and two-stage approaches, and why they may differ.
  Statistics in Medicine. doi:10.1002/sim.7141
- Cheng, K., Yi, Z., Moeyaert, M., Beretvas, S. N., Van den Noortgate, W., and Ferron, J.
  (2025). Synthesizing Single-Case Experimental Designs: Modeling Complex Data
  Structures. Journal of Behavioral Education. doi:10.1007/s10864-025-09602-x
- De, T. K., Michiels, B., Tanious, R., and Onghena, P. (2020). Handling missing data in
  randomization tests for single-case experiments. Behavior Research Methods.
  doi:10.3758/s13428-019-01320-3
- Draper, N. R., and Stoneman, D. M. (1966). Testing for the inclusion of variables in
  linear regression by a randomisation technique. Technometrics, 8(4), 695-699. (not in
  wiki)
- Edgington, E. S., and Onghena, P. (2007). Randomization Tests (4th ed.). Chapman and
  Hall/CRC. (not in wiki)
- Ferron, J. M., Bell, B. A., Hess, M. R., Rendina-Gobioff, G., and Hibbard, S. T. (2009).
  Making treatment effect inferences from multiple-baseline data: The utility of
  multilevel modeling approaches. Behavior Research Methods. doi:10.3758/BRM.41.2.372
- Fingerhut, J., Xu, X., and Moeyaert, M. (2021). Selecting the proper Tau-U measure for
  single-case experimental designs: Development of a decision flowchart. Evidence-Based
  Communication Assessment and Intervention. doi:10.1080/17489539.2021.1937851
- Freedman, D., and Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. Journal of Business and Economic Statistics, 1(4), 292-298.
  doi:10.1080/07350015.1983.10509354 (not in wiki as a source page)
- Hedges, L. V., Shadish, W. R., and Natesan Batley, P. (2022). Power analysis for
  single-case designs: Computations for (AB)k designs. Behavior Research Methods.
  doi:10.3758/s13428-022-01971-9
- Heyvaert, M., and Onghena, P. (2014). Analysis of single-case data: Randomisation tests
  for measures of effect size. Neuropsychological Rehabilitation.
  doi:10.1080/09602011.2013.818564
- Higgins, J. P. T., and Thompson, S. G. (2002). Quantifying heterogeneity in a
  meta-analysis. Statistics in Medicine, 21, 1539-1558. doi:10.1002/sim.1186 (not in wiki
  as a source page)
- Kennedy, P. E. (1995). Randomization tests in econometrics. Journal of Business and
  Economic Statistics, 13(1), 85-94. (not in wiki)
- Kratochwill, T. R., Hitchcock, J., Horner, R. H., Levin, J. R., Odom, S. L., Rindskopf,
  D. M., and Shadish, W. R. (2010). Single-Case Design Technical Documentation Version 1.0
  (Pilot). What Works Clearinghouse.
- Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom, S. L.,
  Rindskopf, D. M., and Shadish, W. R. (2012). Single-Case Intervention Research Design
  Standards. Remedial and Special Education. doi:10.1177/0741932512452794
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation.
  Advances in Methods and Practices in Psychological Science, 1(2), 270-280. (not in wiki)
- Levin, J. R., Ferron, J. M., and Gafurov, B. S. (2016). Comparison of randomization-test
  procedures for single-case multiple-baseline designs. Developmental Neurorehabilitation.
  doi:10.1080/17518423.2016.1197708
- Levin, J. R., and Kratochwill, T. R. (2021). Randomized Single-Case Intervention Designs
  and Analyses for Health Sciences. Therapeutic Innovation and Regulatory Science.
  doi:10.1007/s43441-021-00274-z
- Makowski, D., Ben-Shachar, M. S., and Ludecke, D. (2019). bayestestR: Describing effects
  and their uncertainty, existence and significance within the Bayesian framework. Journal
  of Open Source Software, 4(40), 1541. doi:10.21105/joss.01541
- Manly, B. F. J. (1997). Randomization, Bootstrap and Monte Carlo Methods in Biology
  (2nd ed.). Chapman and Hall. (not in wiki)
- Manolov, R., Moeyaert, M., and Fingerhut, J. E. (2021). A Priori Justification for Effect
  Measures in Single-Case Experimental Designs. Perspectives on Behavior Science.
  doi:10.31234/osf.io/uawzh (preprint)
- Manolov, R., and Moeyaert, M. (2025). Multilevel Model Selection Applied to Single-Case
  Experimental Design Data. Journal of Behavioral Education. doi:10.1007/s10864-025-09593-9
- Manolov, R., and Tanious, R. (2024). Type I error rates and power of two randomization
  test procedures for the changing criterion design. Behavior Research Methods.
  doi:10.3758/s13428-023-02303-1
- Michiels, B., Heyvaert, M., and Onghena, P. (2017). The conditional power of
  randomization tests for single-case effect sizes. Behavior Research Methods.
  doi:10.3758/s13428-017-0885-7
- Michiels, B., and Onghena, P. (2018). Randomized single-case AB phase designs: Prospects
  and pitfalls. Behavior Research Methods. doi:10.3758/s13428-018-1084-x
- Moeyaert, M., Ferron, J. M., Beretvas, S. N., and Van den Noortgate, W. (2014). From a
  single-level analysis to a multilevel analysis of single-case experimental designs.
  Journal of School Psychology. doi:10.1016/j.jsp.2013.11.003
- Natesan, P., and Hedges, L. V. (2017). Bayesian unknown change-point models to
  investigate immediacy in single-case designs. Psychological Methods, 22(4), 743-759.
  doi:10.1037/met0000134
- Natesan Batley, P., Contractor, A. A., and Caldas, S. V. (2020). Bayesian time-series
  models in single-case experimental designs: A tutorial for trauma researchers. Journal
  of Traumatic Stress. doi:10.1002/jts.22614
- Parker, R. I., and Vannest, K. (2009). An improved effect size for single-case research:
  Nonoverlap of All Pairs. Behavior Therapy. doi:10.1016/j.beth.2008.10.004
- Parker, R. I., Vannest, K. J., Davis, J. L., and Sauber, S. B. (2011). Combining
  nonoverlap and trend for single-case research: Tau-U. Behavior Therapy.
  doi:10.1016/j.beth.2010.08.006
- Phipson, B., and Smyth, G. K. (2010). Permutation p-values should never be zero.
  Statistical Applications in Genetics and Molecular Biology, 9(1), Art. 39.
  doi:10.2202/1544-6115.1585 (not in wiki as a source page)
- Pustejovsky, J. E., Hedges, L. V., and Shadish, W. R. (2014). Design-comparable effect
  sizes in multiple baseline designs: A general modeling framework. Journal of Educational
  and Behavioral Statistics. doi:10.3102/1076998614547577
- Rindskopf, D. (2014). Nonlinear Bayesian analysis for single case designs. Journal of
  School Psychology. (not in wiki)
- Shadish, W. R. (2014). Analysis and meta-analysis of single-case designs: An
  introduction. Journal of School Psychology. doi:10.1016/j.jsp.2013.11.009
- Tarlow, K. R. (2016). An improved rank correlation effect size statistic for single-case
  designs: Baseline Corrected Tau. Behavior Modification. doi:10.1177/0145445516676750
- Tate, R. L., Perdices, M., Rosenkoetter, U., Shadish, W., Vohra, S., Barlow, D. H., et
  al. (2016). The Single-Case Reporting Guideline In BEhavioural Interventions (SCRIBE)
  2016. Archives of Scientific Psychology. doi:10.1037/arc0000026.supp
- Ugille, M., Moeyaert, M., Beretvas, S. N., Ferron, J., and Van den Noortgate, W. (2012).
  Multilevel meta-analysis of single-subject experimental designs: A simulation study.
  Behavior Research Methods. doi:10.3758/s13428-012-0213-1
- Valentine, J. C., Tanner-Smith, E. E., Pustejovsky, J. E., and Lau, T. S. (2016).
  Between-case standardized mean difference effect sizes for single-case designs: A primer
  and tutorial using the scdhlm web application. The Campbell Collaboration.
  doi:10.4073/cmdp.2016.1
- Van den Noortgate, W., and Onghena, P. (2003). Hierarchical linear models for the
  quantitative integration of effect sizes in single-case research. Behavior Research
  Methods, Instruments, and Computers. doi:10.3758/bf03195492
- Van den Noortgate, W., and Onghena, P. (2008). A multilevel meta-analysis of
  single-subject experimental design studies. Evidence-Based Communication Assessment and
  Intervention. doi:10.1080/17489530802505362
- Van den Noortgate, W., and Onghena, P. (2024). Harnessing available evidence in
  single-case experimental studies: The use of multilevel meta-analysis. Psychologica
  Belgica. doi:10.5334/pb.1307
- Vannest, K. J., and Ninci, J. (2015). Evaluating intervention effects in single-case
  research designs. Journal of Counseling and Development, 93(4), 403-411. (not in wiki)
- Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC. Statistics and Computing, 27, 1413-1432.
  doi:10.1007/s11222-016-9696-4 (not in wiki)
- Wagner, A. K., Soumerai, S. B., Zhang, F., and Ross-Degnan, D. (2002). Segmented
  regression analysis of interrupted time series studies in medication use research.
  Journal of Clinical Pharmacy and Therapeutics, 27(4), 299-309. (not in wiki)
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely
  applicable information criterion in singular learning theory. Journal of Machine Learning
  Research, 11, 3571-3594. (no DOI; JMLR open access; not in wiki)
- Williams, D. R., Rast, P., and Burkner, P.-C. (2018). Bayesian meta-analysis with
  weakly informative prior distributions. Psychological Methods. (not in wiki)
- Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., and Nichols, T. E. (2014).
  Permutation inference for the general linear model. NeuroImage, 92, 381-397.
  doi:10.1016/j.neuroimage.2014.01.060 (referenced in the wiki via
  wawrzyniak-2022-disconnection-mapping.md; no dedicated source page)
- Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018). Using stacking to average
  Bayesian predictive distributions. Bayesian Analysis, 13(3), 917-1007.
  doi:10.1214/17-ba1091 (not in wiki)
