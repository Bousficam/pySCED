# pysced

Python toolbox for the statistical analysis of single-case experimental designs
(SCED): randomization/permutation tests, effect sizes, Bayesian models, plus a
general-statistics layer (longitudinal, multivariate, feature selection).

SCED inference rests on the **randomization actually implemented** in the design,
not on distributional assumptions: the tests stay valid under non-normality,
autocorrelation and small samples (Edgington & Onghena 2007).

## Installation

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, statsmodels, pymc/arviz...
pip install -r requirements-dev.txt  # pytest
pytest                               # run the test suite
```

## Layout

```
functions/
  sced/        SCED analysis (core of the project)
    core/      primitives: randomization, effect sizes, nuisance, diagnostics, permutation
    alternating/  alternating design (ATD / N-of-1)
    mbd/          multiple-baseline
    bayes/        Bayesian condition models
    cluster/      cluster / NBS permutation (spatial or network outcome)
    plots/        SCED figures
  general/     general statistics (longitudinal, multivariate, mixed, univariate)
  common/      figure style and viz utilities
templates/     copy-ready scripts, one per analysis type
```

Each analysis is driven from a **template** (links below) that calls the
functions in `functions/`. Detailed methodological documentation:
[`templates/analyses/sced/METHODS.md`](templates/analyses/sced/METHODS.md) and
[`templates/analyses/sced/FLOWCHARTS.md`](templates/analyses/sced/FLOWCHARTS.md).

---

## Axis 1 - Alternating design (ATD): permutation principle, ANOVA / ANCOVA

Conditions are assigned to the **measurement occasion** (not to a phase). Because
the condition is randomized, the natural exact test is a **randomization test**:
reshuffle the admissible labels and recompute a statistic.

- **Without a nuisance covariate**: the statistic is a **difference of means**
  (2 conditions) or a **between-condition variance** (>=3). Permuting the labels
  amounts to a permutation test of the one-factor model, i.e. the permuted
  version of **ANOVA**.
- **With a covariate (time trend)**: the condition effect is tested **beyond**
  time, i.e. a **permuted ANCOVA**. Here you do not permute freely: the scheme
  depends on condition-time collinearity, between **Draper-Stoneman** (permute
  the regressor) and **Freedman-Lane** (permute the reduced-model residuals).
  Reference GLM-permutation framework: Winkler et al. 2014.
- **Spatial or network outcome** (connectome, EEG): the same permuted ANCOVA is
  carried by a cluster-mass statistic or an NBS (`sced/cluster/`).

Key functions: `randomization_test`, `diff_in_means`, `omnibus_variance`,
`condition_permutation_test` (ANCOVA), `choose_permutation_method` (DS vs FL),
`huh_jhun_test`, `run_ancova`, `report_sced_cluster`.

Templates:
- [`template_sced_alternating.py`](templates/analyses/sced/template_sced_alternating.py) - full analysis of an alternating design (single or multi-unit)
- [`template_sced_power_planning.py`](templates/analyses/sced/template_sced_power_planning.py) - power planning / minimum n
- [`template_sced_visualize.py`](templates/analyses/sced/template_sced_visualize.py) - figures

---

## Axis 2 - Multiple-baseline (MBD): permutation and piecewise Bayesian

The intervention starts at staggered points across units. Two complementary
routes.

- **MBD permutation test**: randomization of the start point under a scheme
  (MB, MB-R, Wampold-Worsham, Koehler-Levin, Revusky), with procedure choice
  driven by the number of units and the stagger constraint. Statistics: level
  difference, ITEI (immediate effect), Tau-U.
- **Piecewise Bayesian**: hierarchical segmented model (level `b1`, level shift
  `b2`, slope change `b3`) with within-case AR(1), per-case and population-level
  effects, and model comparison (PSIS-LOO / canonical comparators) to identify
  the assumed effect shape.

Key functions: `multiple_baseline_test`, `pipeline_sced_multiple_baseline`,
`choose_mbd_procedure`, `bayes_hier_sced`, `multilevel_mbd_model`,
`bayes_phase_model`, `compare_sced_models`.

Templates:
- [`template_sced_multiple_baseline.py`](templates/analyses/sced/template_sced_multiple_baseline.py) - MBD permutation test
- [`template_sced_bayesian_mbd.py`](templates/analyses/sced/template_sced_bayesian_mbd.py) - hierarchical piecewise Bayesian (MBD)
- [`template_sced_bayesian.py`](templates/analyses/sced/template_sced_bayesian.py) - Bayesian condition model
- [`template_sced_bayesian_percase.py`](templates/analyses/sced/template_sced_bayesian_percase.py) - per-case Bayesian + model selection (LOO)
- [`template_sced_model_compare.py`](templates/analyses/sced/template_sced_model_compare.py) - SCED model comparison
- [`template_sced_phase_design.py`](templates/analyses/sced/template_sced_phase_design.py) - phase design (start-point)

---

## Axis 3 (secondary) - Univariate to multivariate with feature selection

General-statistics layer, outside SCED: from the univariate test to the
multivariate model with feature selection.

- **Univariate**: descriptive and bivariate tests (continuous, dichotomous,
  multiclass), paired pre-post.
- **Multivariate**: OLS / logistic / multivariate MANOVA, repeated measures.
- **Feature selection**: forward / backward stepwise (AIC), Lasso and ElasticNet,
  with collinearity diagnostics (VIF, correlated pairs).

Key functions: `pipeline_univariate_tests`, `pipeline_analysis_descr`,
`pipeline_multiv_ols`, `pipeline_multiv_logit`, `pipeline_paired_ols` / `_logit`
/ `_manova` / `_count`, `forward_selection_step`, `backward_selection_step`,
`optimize_lasso_ic`, `compute_vif_table`.

Templates:
- [`template_preprocessing.py`](templates/template_preprocessing.py) - column cleaning and typing
- [`template_essai_randomise_pre_post.py`](templates/analyses/general/template_essai_randomise_pre_post.py) - randomized pre-post trial
- [`template_outcome_binaire_repete.py`](templates/analyses/general/template_outcome_binaire_repete.py) - repeated binary outcome
- [`template_comptage_repete.py`](templates/analyses/general/template_comptage_repete.py) - repeated counts
- [`template_multivarie_permanova.py`](templates/analyses/general/template_multivarie_permanova.py) - multivariate PERMANOVA
- [`template_suivi_multi_temps.py`](templates/analyses/general/template_suivi_multi_temps.py) - multi-timepoint follow-up
- [`template_longitudinal_growth.py`](templates/analyses/general/template_longitudinal_growth.py) - growth curves
- [`template_donnees_manquantes_imputation.py`](templates/analyses/general/template_donnees_manquantes_imputation.py) - missing data / imputation

---

## Style

ASCII output, neutral tone. No em dash (U+2014) or en dash (U+2013): plain hyphen
or rephrase.
