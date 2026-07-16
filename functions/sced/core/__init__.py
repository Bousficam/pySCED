"""
SCED - Single-Case Experimental Designs: shared layer (core)
============================================================

Common building blocks for every SCED family (alternating/N-of-1, multiple
baseline, AB, ABAB). Following the adopted methodological choice, the PRIMARY
inference is the **randomization test** (Edgington/Onghena) - it matches the
actual experimental design by permuting only the assignments that could have
occurred - and the **nonoverlap indices** (Tau-U, NAP) act as descriptive
support.

Data format (long), one row per session:
    unit | session(integer order) | phase/condition | outcome

Contents:
  - validate_sced            : structure validation + short report
  - describe_by_phase        : level / trend / variability per phase
  - nap, tau_u, pnd, pem     : nonoverlap effect sizes
  - segmented_regression     : LEVEL + SLOPE change (ITS), continuous/
                               count/binary (gaussian/poisson/binomial)
  - mann_kendall, sens_slope, diagnose_series : trend/learning diagnostics
  - randomization_test       : generic permutation engine (stat + scheme)
  - alternating_scheme       : condition randomization scheme (family 1)
  - diff_in_means / omnibus_variance : ready-to-use test statistics
  - condition_permutation_test : time-adjusted condition-effect test, via
                               Draper-Stoneman OR Freedman-Lane
  - choose_permutation_method : helper for choosing between the two schemes

References (all verified online):
  - Edgington, E. S., & Onghena, P. (2007). Randomization Tests (4th ed.).
    Chapman & Hall/CRC.  [design-based randomization test, N-of-1]
  - Draper, N. R., & Stoneman, D. M. (1966). Testing for the inclusion of variables
    in linear regression by a randomisation technique. Technometrics, 8(4), 695-699.
  - Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
    significance levels. Journal of Business & Economic Statistics, 1(4), 292-298.
    doi:10.1080/07350015.1983.10509354
  - ter Braak, C. J. F. (1992). Permutation versus bootstrap significance tests in
    multiple regression and ANOVA. In Bootstrapping and Related Techniques
    (pp. 79-85). Springer.
  - Kennedy, P. E. (1995). Randomization tests in econometrics. Journal of Business
    & Economic Statistics, 13(1), 85-94.
  - Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E.
    (2014). Permutation inference for the general linear model. NeuroImage, 92,
    381-397. doi:10.1016/j.neuroimage.2014.01.060  [comparison of schemes]
  - Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for
    autocorrelated data. Journal of Hydrology, 204, 182-196.
    doi:10.1016/S0022-1694(97)00125-X
  - Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011). Combining
    nonoverlap and trend for single-case research: Tau-U. Behavior Therapy, 42(2),
    284-299. doi:10.1016/j.beth.2010.08.006  [Tau-U family; NB the 2011a hand-calc of
    the baseline-corrected coefficient has an UNBOUNDED denominator - see below]
  - Brossard, M., et al. (2018). [evaluation of the six Tau-U coefficients vs trained
    visual raters]. Cogent Psychology.  [documents the Parker 2011a arithmetic flaw
    (values > 1), gives the bounded denominator (S_AB-S_A)/(n_AB+n_A) and the
    SE = sqrt(2(2n+5)/(9n(n-1))) for weighted meta-analysis]
  - Tarlow, K. R. (2016). An improved rank correlation effect size statistic for
    single-case designs: Baseline Corrected Tau. Behavior Modification.
    doi:10.1177/0145445516676750  [Theil-Sen baseline detrend -> bounded Tau-U adj;
    trend control validated for n_A >= 7]
  - Fingerhut, J., Xu, X., & Moeyaert, M. (2021). [Tau-U variant decision flowchart].
    Evidence-Based Communication Assessment and Intervention.  [A-vs-B vs Trend-A vs
    adj selection; Critical Tau-U interpretation over Vannest & Ninci 2015 benchmarks]
  - Cochran, W. G. (1954). The combination of estimates from different experiments.
    Biometrics, 10, 101-129.  [heterogeneity Q]
  - Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a
    meta-analysis. Statistics in Medicine, 21, 1539-1558. doi:10.1002/sim.1186  [I^2]
  - Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size.
    Journal of Educational Statistics, 6(2), 107-128.  [Hedges' g]
  - Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero.
    Statistical Applications in Genetics and Molecular Biology, 9(1), Art. 39.
    doi:10.2202/1544-6115.1585  [p = (1 + #) / (1 + B)]
  - Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power. The American
    Statistician, 55(1), 19-24.  [why not "observed power" - see SCED_power]

No new dependency (numpy/pandas/scipy/statsmodels).
"""


from .nuisance import (
    _kendall_s,
    sens_slope,
    _nuisance_basis,
    make_condition_ss_stat,
    _rss_design,
    _append_covariates,
    _group_nuisance,
    _standardize_within,
    _cochran_q,
    _effect_values,
)
from .data_checks import (
    validate_sced,
    sced_data_warnings,
    missing_data_warnings,
    check_long_format,
    _ols_slope,
    describe_by_phase,
)
from .effect_sizes import (
    nap,
    nap_ci,
    _tau_significance,
    _tau_contrast_significance,
    hedges_g_within,
    tau_u,
    hedges_g,
    pnd,
    pem,
    nap_magnitude,
    tau_magnitude,
    pnd_magnitude,
    pem_magnitude,
    ICC_BANDS,
    I2_BANDS,
    G_BANDS,
    NAP_BANDS,
    TAU_BANDS,
    PND_BANDS,
    PEM_BANDS,
    icc_band,
    i2_band,
)
from .diagnostics import (
    _rank_autocorr,
    mann_kendall,
    durbin_watson,
    lag1_autocorr,
    diagnose_series,
    segmented_regression,
    segmented_table,
)
from .permutation import (
    condition_permutation_test,
    PERMUTATION_DECISION_TREE,
    choose_permutation_method,
    stratified_condition_permutation_test,
    choose_permutation_method_group,
    huh_jhun_test,
    recommend_scheme,
    heterogeneity_test,
    pairwise_contrasts,
    diff_in_means,
    omnibus_variance,
    _validate_condition_order,
    make_ordered_trend_stat,
    stratified_ordered_trend_test,
    randomization_test,
    alternating_scheme,
    block_scheme,
    sced_test_name,
    sced_methods_text,
)

__all__ = [
    "_kendall_s",
    "sens_slope",
    "_nuisance_basis",
    "make_condition_ss_stat",
    "_rss_design",
    "_append_covariates",
    "_group_nuisance",
    "_standardize_within",
    "_cochran_q",
    "_effect_values",
    "validate_sced",
    "sced_data_warnings",
    "missing_data_warnings",
    "check_long_format",
    "_ols_slope",
    "describe_by_phase",
    "nap",
    "nap_ci",
    "_tau_significance",
    "_tau_contrast_significance",
    "hedges_g_within",
    "tau_u",
    "hedges_g",
    "pnd",
    "pem",
    "nap_magnitude",
    "tau_magnitude",
    "pnd_magnitude",
    "pem_magnitude",
    "ICC_BANDS",
    "I2_BANDS",
    "G_BANDS",
    "NAP_BANDS",
    "TAU_BANDS",
    "PND_BANDS",
    "PEM_BANDS",
    "icc_band",
    "i2_band",
    "_rank_autocorr",
    "mann_kendall",
    "durbin_watson",
    "lag1_autocorr",
    "diagnose_series",
    "segmented_regression",
    "segmented_table",
    "condition_permutation_test",
    "PERMUTATION_DECISION_TREE",
    "choose_permutation_method",
    "stratified_condition_permutation_test",
    "choose_permutation_method_group",
    "huh_jhun_test",
    "recommend_scheme",
    "heterogeneity_test",
    "pairwise_contrasts",
    "diff_in_means",
    "omnibus_variance",
    "_validate_condition_order",
    "make_ordered_trend_stat",
    "stratified_ordered_trend_test",
    "randomization_test",
    "alternating_scheme",
    "block_scheme",
    "sced_test_name",
    "sced_methods_text",
]
