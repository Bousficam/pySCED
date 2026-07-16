"""Targeted tests on the pure / regression-prone logic of the pipeline.

Deliberately non-exhaustive safety net: it covers the functions whose bug
would go unnoticed (selection criteria, univariate table parsing, text
formatting, penalized selection).
"""
import numpy as np
import pandas as pd
import pytest

from functions.general.multivariate.selection import (
    get_criterion,
    select_significant_variables,
    penalized_selection,
    regression_model,
    _join_human,
    SIGNIFICANCE_CRITERIA,
    MAXIMIZE_CRITERIA,
    MINIMIZE_CRITERIA,
)
from functions.general.univariate import (
    perform_categorial_tests,
    perform_OR,
    z_test,
    perform_continuous_tests,
    logistic_regression_univariable,
    perform_multiclass_test,
    analyse_descriptive_continuous,
    export_dict_log_result,
)
from functions.general.collinearity import correlogram, detect_colinear_pairs
import statsmodels.api as sm


def _cohort_from_table(table):
    """Reconstruct a long cohort from a 2x2 table [[a,b],[c,d]]
    indexed (var_indep=1, var_indep=0) x (var_dep=1, var_dep=0)."""
    rows = []
    for i, indep in enumerate([1, 0]):
        for j, dep in enumerate([1, 0]):
            rows += [{"x": indep, "y": dep}] * table[i][j]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_df():
    """Binary dataset where y depends strongly on x1 and x2; x3/x4 = noise."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.randn(n)  # noise
    x4 = rng.randn(n)  # noise
    logit = 2.0 * x1 - 1.5 * x2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


@pytest.fixture(scope="module")
def logit_model(synthetic_df):
    return regression_model(synthetic_df, ["x1", "x2"], "y", model_type="Logit")


@pytest.fixture(scope="module")
def ols_model(synthetic_df):
    return regression_model(synthetic_df, ["x1", "x2"], "y", model_type="ols")


# --------------------------------------------------------------------------- #
# get_criterion
# --------------------------------------------------------------------------- #
class TestGetCriterion:
    def test_aic_bic_ll_match_model(self, logit_model):
        assert get_criterion(logit_model, "AIC") == logit_model.aic
        assert get_criterion(logit_model, "BIC") == logit_model.bic
        assert get_criterion(logit_model, "log_likelihood") == logit_model.llf
        assert get_criterion(logit_model, "ll") == logit_model.llf

    def test_mcfadden_in_unit_range(self, logit_model):
        r2 = get_criterion(logit_model, "explained_variance", pseudo_r2_type="McFadden")
        assert 0.0 < r2 < 1.0

    def test_auc_in_range(self, logit_model):
        assert 0.5 <= get_criterion(logit_model, "AUC") <= 1.0

    def test_wald_matches_formula(self, logit_model):
        expected = (logit_model.params["x1"] / logit_model.bse["x1"]) ** 2
        assert get_criterion(logit_model, "Wald", var="x1") == pytest.approx(expected)

    def test_pvalue_requires_var(self, logit_model):
        with pytest.raises(ValueError):
            get_criterion(logit_model, "pvalue")

    # --- Bug (a) regression: these branches were unreachable ---
    def test_adjusted_r2_reachable_and_not_llf(self, ols_model):
        val = get_criterion(ols_model, "adjusted_r2")
        assert val == ols_model.rsquared_adj
        assert val != ols_model.llf  # before the fix, returned llf

    def test_mae_reachable(self, ols_model):
        assert get_criterion(ols_model, "MAE") == pytest.approx(
            float(np.mean(np.abs(ols_model.resid)))
        )

    def test_unknown_criterion_raises(self, logit_model):
        # before the `or 'll'` fix, an unknown criterion silently returned llf
        with pytest.raises(ValueError):
            get_criterion(logit_model, "definitely_not_a_criterion")


# --------------------------------------------------------------------------- #
# select_significant_variables
# --------------------------------------------------------------------------- #
class TestSelectSignificantVariables:
    def _df(self):
        return pd.DataFrame(
            {
                "Variable": ["a", "b", "c", "d"],
                "Pval": [0.01, 0.04, 0.50, 0.001],
                "Missing data": ["0 (0.0%)", "30 (12.0%)", "0 (0.0%)", "2 (1.0%)"],
            }
        )

    def test_basic_split(self):
        # alpha=0.05, missing<=5%: a and d ok; b significant but 12% missing; c not significant
        ok, high_missing = select_significant_variables(self._df(), alpha=0.05, missing_threshold=5)
        assert ok == ["a", "d"]
        assert high_missing == ["b"]

    def test_threshold_as_fraction(self):
        # missing_threshold=0.05 must be interpreted as 5%
        ok_pct, _ = select_significant_variables(self._df(), alpha=0.05, missing_threshold=5)
        ok_frac, _ = select_significant_variables(self._df(), alpha=0.05, missing_threshold=0.05)
        assert ok_pct == ok_frac

    def test_alpha_filters(self):
        ok, _ = select_significant_variables(self._df(), alpha=0.02, missing_threshold=100)
        assert ok == ["a", "d"]  # b has p=0.04 > 0.02

    def test_missing_guard_raises_without_pvalue_column(self):
        df = pd.DataFrame({"Variable": ["a"], "foo": [1]})
        with pytest.raises(ValueError):
            select_significant_variables(df)


# --------------------------------------------------------------------------- #
# _join_human
# --------------------------------------------------------------------------- #
class TestJoinHuman:
    def test_single(self):
        assert _join_human(["A"]) == "A"

    def test_pair(self):
        assert _join_human(["A", "B"]) == "A and B"

    def test_oxford_comma(self):
        assert _join_human(["A", "B", "C"]) == "A, B, and C"


# --------------------------------------------------------------------------- #
# Criterion constants (P4)
# --------------------------------------------------------------------------- #
def test_criteria_sets_are_disjoint():
    assert SIGNIFICANCE_CRITERIA.isdisjoint(MAXIMIZE_CRITERIA)
    assert SIGNIFICANCE_CRITERIA.isdisjoint(MINIMIZE_CRITERIA)
    assert MAXIMIZE_CRITERIA.isdisjoint(MINIMIZE_CRITERIA)


# --------------------------------------------------------------------------- #
# penalized_selection
# --------------------------------------------------------------------------- #
class TestPenalizedSelection:
    def test_returns_subset_and_info(self, synthetic_df):
        candidates = ["x1", "x2", "x3", "x4"]
        selected, info = penalized_selection(synthetic_df, candidates, "y", C=1.0, l1_ratio=1.0)
        assert isinstance(selected, list)
        assert set(selected) <= set(candidates)
        # selected and excluded partition the candidates
        assert set(selected) | set(info["Variables excluded during penalty regression"]) == set(candidates)
        assert info["Total number of selected variables"] == len(selected)

    def test_strong_predictor_retained(self, synthetic_df):
        # With a lightly penalized Lasso, the dominant predictor x1 must survive.
        selected, _ = penalized_selection(synthetic_df, ["x1", "x2", "x3", "x4"], "y", C=1.0, l1_ratio=1.0)
        assert "x1" in selected


# --------------------------------------------------------------------------- #
# perform_categorial_tests: association tests + zero-cell handling
# --------------------------------------------------------------------------- #
class TestCategorialTests:
    @pytest.mark.parametrize(
        "table",
        [
            [[0, 15], [12, 3]],   # zero cell, expected >= 5
            [[14, 0], [3, 11]],   # zero cell, other corner
            [[0, 3], [8, 2]],     # zero cell, expected < 5
        ],
    )
    def test_zero_cell_never_returns_none(self, table):
        # Regression: a 2x2 table with a zero cell must produce a
        # p-value (Fisher or chi2), never None (otherwise the variable is
        # silently dropped from the significance filter).
        _, p, comment = perform_categorial_tests(_cohort_from_table(table), "x", "y")
        assert p is not None, comment
        assert 0.0 <= p <= 1.0

    def test_expected_lt5_uses_fisher(self):
        # Expected count < 5 -> Fisher exact; numerical match with scipy.
        from scipy.stats import fisher_exact
        table = [[2, 8], [7, 3]]
        _, p, comment = perform_categorial_tests(_cohort_from_table(table), "x", "y")
        assert "Fisher" in comment
        assert p == pytest.approx(fisher_exact(np.array(table))[1])

    def test_large_counts_use_chi2(self):
        # All expected >= 5 -> chi2.
        _, p, comment = perform_categorial_tests(_cohort_from_table([[30, 20], [18, 32]]), "x", "y")
        assert "Chi2" in comment
        assert p is not None

    def test_constant_variable_returns_none(self):
        const = pd.DataFrame({"x": [1, 1, 1, 1], "y": [0, 1, 0, 1]})
        _, p, comment = perform_categorial_tests(const, "x", "y")
        assert p is None
        assert "constant" in comment


# --------------------------------------------------------------------------- #
# perform_OR: Haldane-Anscombe correction on a zero cell
# --------------------------------------------------------------------------- #
class TestPerformOR:
    def test_haldane_correction_on_zero(self):
        OR, se, lo, hi, corrected, _ = perform_OR(_cohort_from_table([[0, 15], [12, 3]]), "x", "y")
        assert corrected == "yes"
        assert OR is not None and np.isfinite(OR)
        assert lo < hi  # bounded CI, no division by zero

    def test_no_correction_when_no_zero(self):
        OR, se, lo, hi, corrected, _ = perform_OR(_cohort_from_table([[5, 10], [8, 7]]), "x", "y")
        assert corrected == "no"
        assert OR == pytest.approx((5 * 7) / (10 * 8))


# --------------------------------------------------------------------------- #
# Continuous tests: z_test (known variances) and n<5 guard
# --------------------------------------------------------------------------- #
class TestContinuous:
    def test_z_test_known_variance_uses_correct_se(self):
        # Correct SE = sqrt(s1^2/n1 + s2^2/n2), without any extra factor.
        rng = np.random.RandomState(0)
        g1 = rng.normal(10, 2, 40)
        g2 = rng.normal(8, 2, 40)
        s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
        z, p = z_test(g1, g2, std_known=(s1, s2))
        se = np.sqrt(s1 ** 2 / len(g1) + s2 ** 2 / len(g2))
        assert z == pytest.approx((g1.mean() - g2.mean()) / se)

    def test_continuous_comment_is_string_when_small_group(self):
        # n1 < 5 -> comment must be a string, not a tuple.
        df = pd.DataFrame({"x": [1, 1, 1] + [0] * 7, "y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        stats, pval, comment = perform_continuous_tests(df, "x", "y")
        assert stats is None and pval is None
        assert isinstance(comment, str)
        assert "Not enough" in comment


# --------------------------------------------------------------------------- #
# Univariate logistic regression: (quasi-)separation warning
# --------------------------------------------------------------------------- #
def test_logit_univariable_warns_on_quasi_separation(recwarn):
    # Quasi-separation: pred=1 almost always implies out=1 (a single
    # contaminant). The MLE diverges (huge coef/se) -> warning expected.
    data = pd.DataFrame({
        "out": [0] * 10 + [1] * 10,
        "pred": [0] * 10 + [1] * 9 + [0],
    })
    res = logistic_regression_univariable(data, "out", "pred")
    assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
    assert res[0] is not None  # here the fit converges (but toward an aberrant OR)


def test_logit_univariable_perfect_separation_returns_none():
    # Complete separation: statsmodels raises -> tuple of None.
    data = pd.DataFrame({"out": [0] * 6 + [1] * 6, "pred": [0] * 6 + [1] * 6})
    res = logistic_regression_univariable(data, "out", "pred")
    assert res[0] is None


# --------------------------------------------------------------------------- #
# perform_multiclass_test: Cochran-Armitage, Kruskal-Wallis, Mann-Whitney
# --------------------------------------------------------------------------- #
class TestMulticlassTest:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "x": [1] * 20 + [0] * 20,
            "y": [0, 0, 1, 1, 2, 2, 3, 3, 1, 2] * 2 + [0, 0, 0, 1, 1, 0, 1, 2, 0, 1] * 2,
        })

    def test_cochran_armitage_runs(self, df):
        # Regression: test_ordinal_association must no longer raise AttributeError.
        stats, p, comment = perform_multiclass_test(df, "x", "y", test="Cochran-Armitage")
        assert p is not None and 0.0 <= p <= 1.0
        assert "Cochran" in comment

    def test_mann_whitney_runs(self, df):
        # Regression: the MW branch (dtype) was dead and returned None.
        stats, p, comment = perform_multiclass_test(df, "x", "y", test="Mann-Whitney")
        assert p is not None and 0.0 <= p <= 1.0
        assert "Mann" in comment

    def test_kruskal_runs(self, df):
        stats, p, comment = perform_multiclass_test(df, "x", "y", test="Kruskal-Wallis")
        assert p is not None


# --------------------------------------------------------------------------- #
# analyse_descriptive_continuous: mean+/-std (normal) vs median[IQR] (skewed)
# --------------------------------------------------------------------------- #
def test_descriptive_summary_by_normality():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "normal": rng.normal(10, 2, 200),
        "skewed": rng.exponential(3, 200),
    })
    out = analyse_descriptive_continuous(df, ["normal", "skewed"])
    cohort = out.set_index("Variable")["Cohort"]
    assert "±" in cohort["normal"]          # normal distribution -> mean +/- std
    assert "[" in cohort["skewed"]          # skewed distribution -> median [IQR]


# --------------------------------------------------------------------------- #
# Multiclass OR: versus-reference (not one-vs-rest) + ordinal trend
# --------------------------------------------------------------------------- #
class TestMulticlassOR:
    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.RandomState(0)
        n = 300
        lvl = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        y = rng.binomial(1, np.array([0.2, 0.5, 0.8])[lvl])
        return pd.DataFrame({"pred": lvl, "y": y})

    def test_level_or_matches_global_reference_model(self, data):
        # The per-level OR must equal exp(coef) of the global versus-reference model,
        # NOT a one-vs-rest.
        dummies = pd.get_dummies(data["pred"], drop_first=True).astype(float)
        m = sm.Logit(data["y"], sm.add_constant(dummies)).fit(disp=False)
        truth = {c: round(float(np.exp(m.params[c])), 3) for c in dummies.columns}

        res = export_dict_log_result(data, "pred", "y", continuous=False)
        got = {k: res[k]["OR"] for k in res if k != "ref"}
        assert got == truth
        assert res["ref"]["OR"] == 1.0

    def test_text_predictor_reference_not_duplicated(self, data):
        df = data.copy()
        df["pred"] = df["pred"].map({0: "A", 1: "B", 2: "C"})
        res = export_dict_log_result(df, "pred", "y", continuous=False)
        # ref + B + C, the reference category 'A' does not appear twice
        assert set(res.keys()) == {"ref", "B", "C"}

    def test_ordinal_adds_trend_entry(self, data):
        res = export_dict_log_result(data, "pred", "y", continuous=False, ordinal=True)
        assert "trend" in res
        assert res["trend"]["OR"] is not None
        # without ordinal, no trend entry
        res_nominal = export_dict_log_result(data, "pred", "y", continuous=False)
        assert "trend" not in res_nominal


# --------------------------------------------------------------------------- #
# Collinearity: the correlation method is honored (both r AND p), no side
# effect on the DataFrame, sorting by absolute correlation.
# --------------------------------------------------------------------------- #
class TestCorrelogram:
    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.RandomState(1)
        x = rng.uniform(0, 5, 120)
        return pd.DataFrame({
            "a": x,
            "b": np.exp(x) + rng.normal(0, 2, 120),   # nonlinear monotone
            "c": -x + rng.normal(0, 0.1, 120),         # strong negative correlation
        })

    def test_method_is_respected(self, data):
        from scipy.stats import pearsonr, spearmanr
        cp, pp = correlogram(data, ["a", "b"], method="pearson")
        cs, ps = correlogram(data, ["a", "b"], method="spearman")
        assert abs(cp.loc["a", "b"] - pearsonr(data["a"], data["b"])[0]) < 1e-9
        assert abs(cs.loc["a", "b"] - spearmanr(data["a"], data["b"])[0]) < 1e-9
        # Spearman must actually differ from Pearson (nonlinear relationship)
        assert abs(cs.loc["a", "b"] - cp.loc["a", "b"]) > 0.05
        # p-value consistent with the method
        assert abs(ps.loc["a", "b"] - spearmanr(data["a"], data["b"])[1]) < 1e-9

    def test_input_not_mutated(self, data):
        before = data.copy(deep=True)
        correlogram(data, ["a", "b", "c"], method="pearson")
        assert data.equals(before)

    def test_pairs_sorted_by_absolute_correlation(self, data):
        corr, _ = correlogram(data, ["a", "b", "c"], method="pearson")
        pairs = detect_colinear_pairs(corr, threshold=0.5)
        # the strongest pair (a~c, negative) must come first
        assert abs(pairs.iloc[0]["Correlation"]) == pairs["Correlation"].abs().max()

    def test_invalid_method_raises(self, data):
        with pytest.raises(ValueError):
            correlogram(data, ["a", "b"], method="bogus")


# --------------------------------------------------------------------------- #
# MM_Analysis: GLM family selection by equality (not `is`).
# --------------------------------------------------------------------------- #
class TestGLMFamilySelection:
    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        return pd.DataFrame({"x": x, "y": rng.poisson(np.exp(0.5 + 0.8 * x))})

    def test_runtime_built_string_selects_right_family(self, data):
        from functions.general.mixed.mm import regression_model
        fam = "Pois" + "son"  # built at runtime: `is` could fail
        m = regression_model(data, "x", "y", model_type="GLM", family_distribution=fam)
        assert m.model.family.__class__.__name__ == "Poisson"
        # and the coefficient recovers the DGP truth
        assert abs(m.params["x"] - 0.8) < 0.15

    def test_family_aliases_case_insensitive(self, data):
        from functions.general.mixed.mm import regression_model
        cases = {" ols ": "Gaussian", "Gamma": "Gamma", "NegativeBinomial": "NegativeBinomial"}
        for fam, expected in cases.items():
            m = regression_model(data, "x", "y", model_type="GLM", family_distribution=fam)
            assert m.model.family.__class__.__name__ == expected

    def test_invalid_family_raises(self, data):
        from functions.general.mixed.mm import regression_model
        with pytest.raises(ValueError):
            regression_model(data, "x", "y", model_type="GLM", family_distribution="bogus")


# --------------------------------------------------------------------------- #
# Stepwise: reproducible tie-breaking (supplied order, not set()).
# --------------------------------------------------------------------------- #
def test_forward_step_tie_break_is_deterministic():
    from functions.general.multivariate.selection import forward_selection_step
    rng = np.random.RandomState(0)
    n = 300
    base = rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-base)))
    # three identical predictors -> strictly equal addition AIC (perfect tie)
    df = pd.DataFrame({"dupA": base, "dupB": base.copy(), "dupC": base.copy(), "y": y})
    # whatever the supplied order, the 1st candidate is the one kept
    for order in (["dupA", "dupB", "dupC"], ["dupC", "dupB", "dupA"], ["dupB", "dupA", "dupC"]):
        _, _, chosen = forward_selection_step(
            df, [], order, "y", model_type="Logit", entry_criterion="AIC", verbose=False
        )
        assert chosen == order[0]


# --------------------------------------------------------------------------- #
# preprocessing.recode_time_difference: no SILENT destruction of dates.
# --------------------------------------------------------------------------- #
class TestRecodeTimeDifference:
    def test_datetime64_columns_not_corrupted_by_default_format(self):
        from functions.general.preprocessing import recode_time_difference
        df = pd.DataFrame({
            "t2": pd.to_datetime(["2021-05-31 10:00", "2021-06-01 09:00", "2021-06-02 09:00"]),
            "t1": pd.to_datetime(["2021-05-31 08:00", "2021-06-01 08:00", None]),
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no silent loss tolerated
            out = recode_time_difference(df, "delta", "t2", "t1", unit="hours")
        assert out["delta"].iloc[0] == 2.0
        assert out["delta"].iloc[1] == 1.0
        assert pd.isna(out["delta"].iloc[2])

    def test_format_mismatch_emits_warning(self):
        from functions.general.preprocessing import recode_time_difference
        df = pd.DataFrame({"t2": ["31/05/2021", "01/06/2021"], "t1": ["30/05/2021", "31/05/2021"]})
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recode_time_difference(df, "delta", "t2", "t1", unit="days", date_format="%d/%m/%y")
        assert any("NaT" in str(x.message) for x in w)

    def test_correct_format_parses_strings(self):
        from functions.general.preprocessing import recode_time_difference
        df = pd.DataFrame({"t2": ["31/05/2021", "01/06/2021"], "t1": ["30/05/2021", "31/05/2021"]})
        out = recode_time_difference(df, "delta", "t2", "t1", unit="days", date_format="%d/%m/%Y")
        assert out["delta"].tolist() == [1.0, 1.0]


# --------------------------------------------------------------------------- #
# preprocessing.encode_datetime / encode_variables
# --------------------------------------------------------------------------- #
class TestEncoding:
    def test_encode_datetime_does_not_mutate_input_and_fills(self):
        from functions.general.preprocessing import encode_datetime
        raw = pd.DataFrame({
            "d": pd.to_datetime(["2021-05-31", "2021-06-01"]),
            "h": ["10:00", np.nan],
        })
        before = raw.copy(deep=True)
        out = encode_datetime(raw, [("dt", "d", "h")], fill_missing="00:00", drop_original=False)
        assert raw.equals(before)                       # input not mutated
        assert out["dt"].iloc[1] == pd.Timestamp("2021-06-01 00:00")  # fillna effective

    def test_encode_variables_text_dichotomous_becomes_numeric(self):
        from functions.general.preprocessing import encode_variables
        df = pd.DataFrame({
            "sexe": ["F", "M", "F", np.nan, "M"],
            "tabac": [0, 1, 1, 0, 1],
            "grade": ["A", "B", "C", "A", "B"],
        })
        enc, edict, lab, dum = encode_variables(df, label_vars=["sexe", "tabac"], dummy_var=["grade"])
        # text dichotomous -> numeric 0/1, NaN preserved, astype(float)-compatible
        assert pd.api.types.is_numeric_dtype(enc["sexe"])
        assert enc["sexe"].tolist()[:3] == [0, 1, 0]
        assert pd.isna(enc["sexe"].iloc[3])
        enc["sexe"].astype(float)
        # already-numeric dichotomous: unchanged
        assert enc["tabac"].tolist() == [0, 1, 1, 0, 1]
        # multiclass: dummies
        assert "grade_B" in enc.columns and "grade" not in enc.columns

    def test_encode_variables_returns_four_tuple_when_both_lists_given(self):
        from functions.general.preprocessing import encode_variables
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "p"]})
        result = encode_variables(df, label_vars=["a"], dummy_var=["b"])
        assert result is not None and len(result) == 4


# --------------------------------------------------------------------------- #
# CV-validated AUC: honest (out-of-sample), <= apparent AUC, robust.
# --------------------------------------------------------------------------- #
class TestCrossValidatedAUC:
    @pytest.fixture(scope="class")
    def noisy_df(self):
        rng = np.random.RandomState(0)
        n = 120
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(8)})
        df["y"] = rng.binomial(1, 1 / (1 + np.exp(-1.2 * df["x0"])))  # 1 true + 7 noise
        return df

    def test_cv_auc_in_range_and_not_above_apparent(self, noisy_df):
        from functions.general.multivariate.selection import (
            regression_model, create_model_info_dict
        )
        preds = [f"x{i}" for i in range(8)]
        m = regression_model(noisy_df, preds, "y", model_type="Logit")
        info = create_model_info_dict(m, noisy_df, model_type="Logit")
        app = info["AUC apparent (in-sample, optimistic) (>0.7)"]
        cv = info["AUC 5-fold CV (out-of-sample) (>0.7)"]
        assert 0.5 <= cv <= 1.0
        assert cv <= app + 1e-9  # apparent optimism never underestimates

    def test_cv_auc_handles_degenerate_cases(self, noisy_df):
        from functions.general.multivariate.selection import cross_validated_auc
        preds = [f"x{i}" for i in range(8)]
        single = noisy_df.copy()
        single["y"] = 0
        assert np.isnan(cross_validated_auc(single, preds, "y"))   # a single class
        assert np.isnan(cross_validated_auc(noisy_df, [], "y"))     # no predictor


# --------------------------------------------------------------------------- #
# preprocessing user-friendly: data_overview, named return, zero mutation,
# friendly errors, and COMPATIBILITY with the ETIS script call patterns.
# --------------------------------------------------------------------------- #
class TestPreprocessingUX:
    def _df(self):
        rng = np.random.RandomState(0)
        n = 50
        return pd.DataFrame({
            "sexe": rng.randint(0, 2, n),
            "grade": rng.randint(0, 5, n),
            "age": rng.uniform(40, 90, n),
            "centre": rng.choice(["A", "B", "C"], n),
            "constante": [1] * n,
            "date_examen": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.randint(0, 100, n), "D"),
            "creat": pd.Series(rng.uniform(50, 120, n)).where(rng.rand(n) > 0.4),
        })

    # --- A: data_overview ---
    def test_data_overview_detects_types_and_flags(self):
        from functions.general.preprocessing import data_overview
        ov = data_overview(self._df())
        m = ov.set_index("variable")["detected_type"].to_dict()
        assert m["sexe"] == "dichotomous"
        assert m["grade"] == "multiclass"
        assert m["age"] == "continuous"
        assert m["centre"] == "categorical"
        assert m["constante"] == "constant"
        assert m["date_examen"] == "datetime"
        # sorted by descending % missing -> creat (40%) first
        assert ov.iloc[0]["variable"] == "creat"
        assert "constant" in ov.set_index("variable").loc["constante", "flags"]

    # --- B: opt-in named return + legacy unchanged ---
    def test_find_categorical_named_result(self):
        from functions.general.preprocessing import find_categorical_variables, VariableTypes
        res = find_categorical_variables(self._df(), as_result=True)
        assert isinstance(res, VariableTypes)
        assert "sexe" in res.dichotomous and "grade" in res.multiclass and "age" in res.continuous
        assert res.counts["sexe"] == 2
        assert set(res.to_frame().columns) == {"variable", "type", "n_unique"}

    def test_find_categorical_legacy_shape_unchanged(self):
        # EXACT pattern from the ETIS script (lines 75-76): 3-tuple of (vars, counts) + [0]
        from functions.general.preprocessing import find_categorical_variables
        a, b, c = find_categorical_variables(
            self._df(), unique_threshold=10, return_count=True,
            return_unique_values=False, return_continuous=True)
        assert isinstance(a, tuple) and isinstance(a[0], list)   # (dicho_vars, counts)
        assert isinstance(b[0], list) and isinstance(c[0], list)

    # --- C: zero mutation ---
    def test_recoders_do_not_mutate_input(self):
        from functions.general.preprocessing import (
            recode_formula_var, recode_categorial_var, replace_target_value, fill_na_columns
        )
        df = pd.DataFrame({"sexe": [0, 1, 0, 1], "a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 2.0, 2.0, 2.0]})
        ref = df.copy(deep=True)
        recode_formula_var(df, "r ~ a / b", output_type="Float64")
        recode_categorial_var(df, "sexe", lambda v: v, "k")
        replace_target_value(df, "sexe", target_value=0, new_value=9)
        fill_na_columns(df, ["a"])
        assert df.equals(ref)

    # --- C: ETIS recode chain compat (reassignment + var_list_to_fill) ---
    def test_etis_recode_chain_with_fill(self):
        from functions.general.preprocessing import recode_formula_var
        df = pd.DataFrame({
            "premiere_strategie_nbp": [1.0, 2.0, 1.0],
            "apres_premiere_strategie_nbp": pd.array([1, pd.NA, 2], dtype="Float64"),
        })
        out = recode_formula_var(
            df, "nb_passes ~ premiere_strategie_nbp + apres_premiere_strategie_nbp",
            output_type="Float64", var_list_to_fill=["apres_premiere_strategie_nbp"])
        assert "nb_passes" in out.columns
        # NaN filled to 0 -> row 2 = 2 + 0
        assert out["nb_passes"].iloc[1] == 2.0
        # temporary _filled column removed
        assert "apres_premiere_strategie_nbp_filled" not in out.columns

    # --- D: friendly errors ---
    def test_friendly_error_with_suggestion(self):
        from functions.general.preprocessing import recode_categorial_var
        df = pd.DataFrame({"sexe": [0, 1]})
        with pytest.raises(KeyError) as exc:
            recode_categorial_var(df, "sexee", lambda v: v, "z")
        assert "sexe" in str(exc.value)  # suggestion of the close name


# --------------------------------------------------------------------------- #
# The preprocessing template must run end to end (guard against an API
# drift that would break the example provided to users).
# --------------------------------------------------------------------------- #
def test_preprocessing_template_runs():
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "templates" / "template_preprocessing.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "RÉSUMÉ FINAL" in res.stdout
    assert "APERÇU DU JEU DE DONNÉES" in res.stdout


# --------------------------------------------------------------------------- #
# Collinearity / pipeline robustness when few candidate variables
# (regression of the "zero-size array" crash in compute_vif_table -> VIF).
# --------------------------------------------------------------------------- #
class TestFewCandidatesRobustness:
    def test_compute_vif_table_degenerate(self):
        from functions.general.collinearity import compute_vif_table
        df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 1, 4, 3, 5]})
        assert compute_vif_table(df, []).empty            # 0 variable
        assert compute_vif_table(df, ["a"]).loc[0, "VIF"] == 1.0   # 1 var -> VIF=1
        assert len(compute_vif_table(df, ["a", "b"])) == 3         # const + 2 vars
        # 0 complete observation -> VIF NaN, no crash
        empty = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        out = compute_vif_table(empty, ["a", "b"])
        assert out["VIF"].isna().all()

    def test_pipeline_no_candidate_does_not_crash(self):
        from functions.general.multivariate.selection import pipeline_multiv_logit
        rng = np.random.RandomState(0)
        n = 80
        db = pd.DataFrame({
            "age": rng.uniform(40, 80, n), "size": rng.uniform(1, 100, n),
            "sexe": rng.randint(0, 2, n), "loc": rng.randint(0, 4, n),
            "y": rng.randint(0, 2, n),  # independent outcome
        })
        var_model = {"Continuous": ["age", "size"], "Dicho": ["sexe"],
                     "Multiclass": ["loc"], "Baseline": ["age", "size", "sexe", "loc"],
                     "Outcomes": ["y"], "Describe": [], "interrater": []}
        # very strict pfilter -> 0 candidate: must finish cleanly (logModel None)
        info, model = pipeline_multiv_logit(db, var_model, "y", pfilter=0.001, missing_data=10,
                                            method_multiv="stepwise", direction="alternate", verbose=False)
        assert model is None
        assert info["Number of candidate variables for multivariate"] == 0
        assert "Warning" in info


# --------------------------------------------------------------------------- #
# Multivariate_OLS: OLS pipeline (continuous outcome), mirror of the logit
# --------------------------------------------------------------------------- #
class TestPipelineMultivOLS:
    def _data(self):
        rng = np.random.RandomState(1)
        n = 200
        age = rng.uniform(40, 80, n)
        taille = rng.uniform(1, 100, n)
        bruit = rng.normal(0, 1, n)
        sexe = rng.randint(0, 2, n)
        loc = rng.randint(0, 3, n)
        y = 0.05 * age - 0.03 * taille + 1.2 * sexe + rng.normal(0, 1, n)
        db = pd.DataFrame({"age": age, "taille": taille, "bruit": bruit,
                           "sexe": sexe, "loc": loc, "y": y})
        vm = {"Continuous": ["age", "taille", "bruit"], "Dicho": ["sexe"],
              "Multiclass": ["loc"],
              "Baseline": ["age", "taille", "bruit", "sexe", "loc"],
              "Outcomes": ["y"], "Describe": [], "interrater": []}
        return db, vm

    def test_univariate_ols_screen_columns(self):
        from functions.general.multivariate.ols import univariate_ols_screen
        db, _ = self._data()
        scr = univariate_ols_screen(db, "y", ["age", "taille", "bruit", "sexe"])
        assert set(["Variable", "Beta", "Pval", "Missing data"]) <= set(scr.columns)
        # a strong predictor (age) must be significant
        assert scr.set_index("Variable").loc["age", "Pval"] < 0.05

    def test_ols_pipeline_returns_model_and_metrics(self):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        info, model = pipeline_multiv_ols(db, vm, "y", pfilter=0.2, missing_data=10,
                                          direction="alternate", verbose=False)
        assert model is not None
        assert "Explained Variance (R²)(>0.25)" in info
        assert "F-statistic p-value (<5%)" in info
        # true strong predictors kept, noise dropped
        kept = [v for v in model.params.index if v != "const"]
        assert "age" in kept and "sexe" in kept
        assert "bruit" not in kept

    def test_ols_export_beta_and_std_beta(self, tmp_path):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        pipeline_multiv_ols(db, vm, "y", pfilter=0.2, missing_data=10,
                            save_path=str(tmp_path), verbose=False)
        f = tmp_path / "Multivariate models" / "Stepwise_OLS" / "y_OLS_stepwise_alternate_summary.xlsx"
        assert f.exists()
        adj = pd.read_excel(f, sheet_name="Adjusted Variables")
        assert list(adj.columns) == ["Independent Variable", "Beta (95% CI)", "Std. Beta", "P"]

    def test_ols_no_candidate_does_not_crash(self):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        # Forces 0 candidate by excluding all baseline variables.
        info, model = pipeline_multiv_ols(db, vm, "y", pfilter=1.0, missing_data=100,
                                          excluded_vars=vm["Baseline"], verbose=False)
        assert model is None
        assert info["Number of candidate variables for multivariate"] == 0
        assert "Warning" in info

    def test_ols_penalized_lasso(self):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        info, model = pipeline_multiv_ols(db, vm, "y", pfilter=0.2, missing_data=10,
                                          method_multiv="penalized", direction="lasso", verbose=False)
        assert model is not None
        assert info["Penalized selection"] == "lasso"
        assert "Alpha" in info
        kept = [v for v in model.params.index if v != "const"]
        assert "age" in kept and "sexe" in kept

    def test_ols_penalized_elasticnet(self):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        info, model = pipeline_multiv_ols(db, vm, "y", pfilter=0.2, missing_data=10,
                                          method_multiv="penalized", direction="elasticnet", verbose=False)
        assert model is not None
        assert info["Penalized selection"] == "elasticnet"
        assert "Alpha" in info and "L1_ratio" in info

    def test_ols_penalized_export(self, tmp_path):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        pipeline_multiv_ols(db, vm, "y", pfilter=0.2, missing_data=10,
                            method_multiv="penalized", direction="lasso",
                            save_path=str(tmp_path), verbose=False)
        f = tmp_path / "Multivariate models" / "Penalized_OLS" / "y_OLS_lasso_summary.xlsx"
        assert f.exists()
        adj = pd.read_excel(f, sheet_name="Adjusted Variables")
        assert list(adj.columns) == ["Independent Variable", "Beta (95% CI)", "Std. Beta", "P"]

    def test_ols_invalid_method_and_direction(self):
        from functions.general.multivariate.ols import pipeline_multiv_ols
        db, vm = self._data()
        with pytest.raises(ValueError):
            pipeline_multiv_ols(db, vm, "y", method_multiv="bogus", verbose=False)
        with pytest.raises(ValueError):
            # 'alternate' is not a valid direction for penalized
            pipeline_multiv_ols(db, vm, "y", method_multiv="penalized",
                                direction="alternate", verbose=False)
