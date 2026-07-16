"""Tests of the longitudinal / paired pipeline (Longitudinal_Analysis).

Covers: reshape wide->long (univariate and multivariate), description by time,
paired tests (t/Wilcoxon, McNemar, Friedman, Cochran-Q), Greenhouse-Geisser, Holm,
paired PERMANOVA (within-subject permutation, univariate and multivariate), models
(MixedLM, logistic GEE) and the Excel export of the three turnkey pipelines.
"""
import numpy as np
import pandas as pd
import pytest

from functions.general.longitudinal.analysis import (
    to_long,
    to_long_multi,
    validate_long,
    ensure_long,
    describe_by_time,
    paired_tests_continuous,
    paired_tests_binary,
    permanova_paired,
    fit_mixed_ols,
    fit_gee_logit,
    diagnostics_mixed_ols,
    pipeline_paired_ols,
    pipeline_paired_logit,
    pipeline_paired_manova,
    missingness_report,
    impute_long,
    prepare_repeated,
    compare_corr_structures,
    _holm_adjust,
    _greenhouse_geisser_epsilon,
)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prepost_continuous_wide():
    """Continuous pre/post with a real time effect (post > pre) and 2 arms."""
    rng = np.random.RandomState(0)
    n = 60
    base = rng.normal(50, 8, n)
    arm = rng.randint(0, 2, n)
    # arm 1 improves more (time x group interaction)
    post = base + 3 + 4 * arm + rng.normal(0, 3, n)
    return pd.DataFrame({"subject": np.arange(n), "arm": arm,
                         "score_pre": base, "score_post": post})


@pytest.fixture(scope="module")
def prepost_binary_long():
    """Binary pre/post: the success probability increases afterward (discordant pairs)."""
    rng = np.random.RandomState(1)
    n = 80
    pre = rng.binomial(1, 0.3, n)
    # post: the 0s often become 1, the 1s stay 1 -> significant McNemar
    post = np.where(pre == 1, 1, rng.binomial(1, 0.55, n))
    rows = []
    for i in range(n):
        rows.append({"subject": i, "visit": "pre", "success": pre[i]})
        rows.append({"subject": i, "visit": "post", "success": post[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def three_times_long():
    """Three continuous timepoints with an increasing within-subject trend."""
    rng = np.random.RandomState(2)
    n = 40
    rows = []
    for i in range(n):
        b = rng.normal(10, 2)
        for k, t in enumerate(["t0", "t1", "t2"]):
            rows.append({"subject": i, "time": t, "y": b + 2.0 * k + rng.normal(0, 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reshape & validation
# --------------------------------------------------------------------------- #
class TestReshape:
    def test_to_long_dict_map(self, prepost_continuous_wide):
        long = to_long(prepost_continuous_wide, "subject",
                       {"pre": "score_pre", "post": "score_post"},
                       outcome="score", group_col="arm")
        assert set(long.columns) == {"subject", "time", "score", "arm"}
        assert sorted(long["time"].unique()) == ["post", "pre"]
        assert len(long) == 2 * len(prepost_continuous_wide)

    def test_to_long_missing_column_raises(self, prepost_continuous_wide):
        with pytest.raises(KeyError):
            to_long(prepost_continuous_wide, "subject", {"pre": "absent"}, outcome="score")

    def test_validate_long_reports_balance_and_drops_nan(self):
        df = pd.DataFrame({"id": [1, 1, 2, 2], "t": ["a", "b", "a", "b"],
                           "y": [1.0, 2.0, 3.0, np.nan]})
        clean, rep = validate_long(df, "id", "t", "y")
        assert rep["Number of timepoints"] == 2
        assert rep["Observations dropped (missing outcome)"] == 1
        assert rep["Balanced design"] is False  # subject 2 incomplete after drop

    def test_ensure_long_requires_time_spec(self, prepost_continuous_wide):
        with pytest.raises(ValueError):
            ensure_long(prepost_continuous_wide, "subject", "score")


# --------------------------------------------------------------------------- #
# Description by time
# --------------------------------------------------------------------------- #
class TestDescribeByTime:
    def test_continuous_has_delta_row_for_two_times(self, prepost_continuous_wide):
        long = to_long(prepost_continuous_wide, "subject",
                       {"pre": "score_pre", "post": "score_post"},
                       outcome="score", group_col="arm")
        d = describe_by_time(long, "subject", "time", "score", continuous=True)
        assert any(str(t).startswith("Δ") for t in d["Time"])
        # mean +/- sd expected on ~normal data
        assert d["Summary"].str.contains("±").any()

    def test_binary_summary_is_count_percent(self, prepost_binary_long):
        d = describe_by_time(prepost_binary_long, "subject", "visit", "success",
                             continuous=False)
        assert d["Summary"].str.contains("%").all()


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #
class TestPairedTests:
    def test_paired_ttest_detects_effect(self, prepost_continuous_wide):
        long = to_long(prepost_continuous_wide, "subject",
                       {"pre": "score_pre", "post": "score_post"}, outcome="score")
        res = paired_tests_continuous(long, "subject", "time", "score")
        overall = res[res["Group"] == "All"].iloc[0]
        assert overall["Test"] in ("Paired t-test", "Wilcoxon signed-rank")
        assert overall["P"] < 0.05

    def test_rm_anova_added_without_group(self, three_times_long):
        res = paired_tests_continuous(three_times_long, "subject", "time", "y")
        tests = set(res["Test"])
        assert "Friedman" in tests
        assert "Repeated-measures ANOVA" in tests
        # >2 timepoints -> a Greenhouse-Geisser row must accompany the RM-ANOVA
        assert "RM-ANOVA (Greenhouse-Geisser)" in tests
        gg = res[res["Test"] == "RM-ANOVA (Greenhouse-Geisser)"].iloc[0]
        assert 0.0 <= gg["P"] <= 1.0
        assert "epsilon=" in gg["Comment"]

    def test_mcnemar_detects_shift(self, prepost_binary_long):
        res = paired_tests_binary(prepost_binary_long, "subject", "visit", "success")
        overall = res[res["Group"] == "All"].iloc[0]
        assert "McNemar" in overall["Test"]
        assert overall["P"] < 0.05

    def test_cochran_q_for_three_binary_times(self):
        rng = np.random.RandomState(3)
        n = 60
        rows = []
        for i in range(n):
            for k, t in enumerate(["t0", "t1", "t2"]):
                rows.append({"id": i, "t": t, "y": rng.binomial(1, 0.2 + 0.25 * k)})
        df = pd.DataFrame(rows)
        res = paired_tests_binary(df, "id", "t", "y")
        overall = res[res["Group"] == "All"].iloc[0]
        assert overall["Test"] == "Cochran's Q"
        assert overall["P"] < 0.05

    def test_insufficient_pairs_no_crash(self):
        df = pd.DataFrame({"id": [1, 2], "t": ["a", "a"], "y": [1.0, 2.0]})
        res = paired_tests_continuous(df, "id", "t", "y")
        assert np.isnan(res.iloc[0]["P"])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TestModels:
    def test_mixedlm_time_effect_significant(self, three_times_long):
        res, info = fit_mixed_ols(three_times_long, "subject", "time", "y")
        assert res is not None and info["Converged"]
        # a time term must be significant
        time_terms = [t for t in res.pvalues.index if "time" in t]
        assert any(res.pvalues[t] < 0.05 for t in time_terms)

    def test_gee_logit_returns_fit(self, prepost_binary_long):
        res, info = fit_gee_logit(prepost_binary_long, "subject", "visit", "success")
        assert res is not None
        assert info["Working correlation"] == "exchangeable"

    def test_gee_invalid_cov_struct_raises(self, prepost_binary_long):
        with pytest.raises(ValueError):
            fit_gee_logit(prepost_binary_long, "subject", "visit", "success",
                          cov_struct="bogus")


# --------------------------------------------------------------------------- #
# Mixed-model diagnostics: residual normality + autocorrelation
# --------------------------------------------------------------------------- #
class TestDiagnostics:
    def test_reports_normality_and_durbin_watson(self, three_times_long):
        res, _ = fit_mixed_ols(three_times_long, "subject", "time", "y")
        info, df = diagnostics_mixed_ols(res, three_times_long, "subject", "time")
        # both diagnostics are present
        assert "Residual normality p (>0.05 = ok)" in info
        assert "Within-subject Durbin-Watson (~2 = no autocorr.)" in info
        metrics = " ".join(df["Metric"])
        assert "Shapiro" in metrics and "Durbin-Watson" in metrics
        # plausible bounds: 0 < DW < 4, and residuals clearly gaussian (normal DGP)
        dw = info["Within-subject Durbin-Watson (~2 = no autocorr.)"]
        assert 0.0 < dw < 4.0
        assert info["Residual normality p (>0.05 = ok)"] > 0.01

    def test_durbin_watson_skipped_for_two_timepoints(self):
        # With 2 timepoints, the DW is not relevant (a single gap per subject): no
        # numeric key in info, but an explanatory row in the table.
        rng = np.random.RandomState(7)
        n = 40
        long = pd.concat([
            pd.DataFrame({"id": np.arange(n), "t": "pre", "y": rng.normal(0, 1, n)}),
            pd.DataFrame({"id": np.arange(n), "t": "post", "y": rng.normal(1, 1, n)}),
        ], ignore_index=True)
        res, _ = fit_mixed_ols(long, "id", "t", "y")
        info, df = diagnostics_mixed_ols(res, long, "id", "t")
        assert "Within-subject Durbin-Watson (~2 = no autocorr.)" not in info
        dw_row = df[df["Metric"].str.contains("Durbin-Watson")].iloc[0]
        assert dw_row["Value"] == " - "
        assert "2 timepoints" in dw_row["Comment"]
        # normality is still reported
        assert "Residual normality (Shapiro W)" in info

    def test_none_model_returns_empty(self):
        info, df = diagnostics_mixed_ols(None, pd.DataFrame(), "id", "t")
        assert info == {}
        assert df.empty

    def test_pipeline_folds_diagnostics_into_info(self, three_times_long):
        info, _ = pipeline_paired_ols(
            three_times_long, id_col="subject", outcome="y",
            time_col="time", verbose=False)
        assert "Within-subject Durbin-Watson (~2 = no autocorr.)" in info
        assert "Residual normality (Shapiro W)" in info


# --------------------------------------------------------------------------- #
# Turnkey pipelines + export
# --------------------------------------------------------------------------- #
class TestPipelines:
    def test_ols_pipeline_returns_info_and_model(self, prepost_continuous_wide):
        info, model = pipeline_paired_ols(
            prepost_continuous_wide, id_col="subject", outcome="score",
            time_map={"pre": "score_pre", "post": "score_post"},
            group_col="arm", verbose=False)
        assert model is not None
        assert info["Outcome type"] == "Continuous"
        assert info["Number of timepoints"] == 2

    def test_ols_pipeline_export_sheets(self, tmp_path, prepost_continuous_wide):
        pipeline_paired_ols(
            prepost_continuous_wide, id_col="subject", outcome="score",
            time_map={"pre": "score_pre", "post": "score_post"},
            save_path=str(tmp_path), verbose=False)
        f = tmp_path / "Longitudinal models" / "Paired_OLS" / "score_paired_ols_report.xlsx"
        assert f.exists()
        sheets = pd.ExcelFile(f).sheet_names
        assert {"Setup Info", "Model Effects", "Diagnostics",
                "Paired Tests", "By Time"} <= set(sheets)
        diag = pd.read_excel(f, sheet_name="Diagnostics")
        assert {"Metric", "Value", "Comment"} <= set(diag.columns)
        eff = pd.read_excel(f, sheet_name="Model Effects")
        assert "Beta (95% CI)" in eff.columns

    def test_logit_pipeline_export_or(self, tmp_path, prepost_binary_long):
        info, model = pipeline_paired_logit(
            prepost_binary_long, id_col="subject", outcome="success",
            time_col="visit", save_path=str(tmp_path), verbose=False)
        assert model is not None
        f = tmp_path / "Longitudinal models" / "Paired_Logit" / "success_paired_logit_report.xlsx"
        assert f.exists()
        eff = pd.read_excel(f, sheet_name="Model Effects")
        assert "OR (95% CI)" in eff.columns


# --------------------------------------------------------------------------- #
# Greenhouse-Geisser: epsilon~1 under compound symmetry, epsilon<1 under AR1 (sphericity violated).
# --------------------------------------------------------------------------- #
class TestGreenhouseGeisser:
    def test_epsilon_near_one_under_compound_symmetry(self):
        rng = np.random.RandomState(0)
        n, k = 200, 4
        common = rng.normal(0, 1, (n, 1))
        X = common + rng.normal(0, 1, (n, k))  # exchangeable -> sphericity OK
        eps = _greenhouse_geisser_epsilon(X)
        assert eps > 0.95

    def test_epsilon_below_one_under_ar1(self):
        rng = np.random.RandomState(0)
        n, k = 200, 4
        t = np.arange(k)
        cov = 0.9 ** np.abs(t[:, None] - t[None, :])
        X = rng.normal(0, 1, (n, k)) @ np.linalg.cholesky(cov).T
        eps = _greenhouse_geisser_epsilon(X)
        assert 1.0 / (k - 1) <= eps < 0.85  # sphericity clearly violated

    def test_epsilon_trivial_for_two_times(self):
        assert _greenhouse_geisser_epsilon(np.random.randn(20, 2)) == 1.0


# --------------------------------------------------------------------------- #
# Holm step-down (correction familiale entre outcomes).
# --------------------------------------------------------------------------- #
class TestHolm:
    def test_matches_reference(self):
        from statsmodels.stats.multitest import multipletests
        p = [0.01, 0.04, 0.03, 0.20]
        ref = multipletests(p, method="holm")[1]
        assert np.allclose(_holm_adjust(p), ref)

    def test_nan_preserved(self):
        out = _holm_adjust([0.01, np.nan, 0.03])
        assert np.isnan(out[1])
        assert not np.isnan(out[0]) and not np.isnan(out[2])


# --------------------------------------------------------------------------- #
# Paired PERMANOVA: restricted within-subject permutation (univariate and multivariate).
# --------------------------------------------------------------------------- #
class TestPermanovaPaired:
    def _trend_long(self, n=40, k=3, effect=1.5, seed=0):
        rng = np.random.RandomState(seed)
        rows = []
        for i in range(n):
            b = rng.normal(0, 1)
            for j in range(k):
                rows.append({"id": i, "t": f"t{j}", "y": b + effect * j + rng.normal(0, 1)})
        return pd.DataFrame(rows)

    def test_detects_time_effect(self):
        info = permanova_paired(self._trend_long(), "id", "t", "y", n_perm=499)
        assert info["P"] < 0.05
        assert info["Pseudo-F"] > 1

    def test_null_is_not_significant(self):
        df = self._trend_long(effect=0.0, seed=1)
        info = permanova_paired(df, "id", "t", "y", n_perm=499)
        assert info["P"] > 0.05

    def test_multivariate_response(self):
        rng = np.random.RandomState(2)
        n = 40
        rows = []
        for i in range(n):
            b = rng.normal(0, 1, 3)
            for j, t in enumerate(["pre", "post"]):
                v = b + np.array([1.0, 0.8, 1.2]) * j + rng.normal(0, 1, 3)
                rows.append({"id": i, "t": t, "y1": v[0], "y2": v[1], "y3": v[2]})
        info = permanova_paired(pd.DataFrame(rows), "id", "t", ["y1", "y2", "y3"], n_perm=499)
        assert info["P"] < 0.05
        assert len(info["Outcomes"]) == 3
        assert 0.0 <= info["R2"] <= 1.0

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError):
            permanova_paired(self._trend_long(), "id", "t", "y", distance="bogus", n_perm=9)

    def test_appears_in_paired_tests_continuous(self):
        res = paired_tests_continuous(self._trend_long(), "id", "t", "y", permanova_n_perm=199)
        assert res["Test"].str.contains("PERMANOVA").any()


# --------------------------------------------------------------------------- #
# MULTIVARIATE repeated measures: reshape + turnkey PERMANOVA pipeline.
# --------------------------------------------------------------------------- #
class TestMultivariatePipeline:
    def _wide(self, n=40, seed=0):
        rng = np.random.RandomState(seed)
        recs = []
        for i in range(n):
            b = rng.normal(0, 1, 3)
            arm = i % 2
            pre = b + rng.normal(0, 1, 3)
            post = b + np.array([1.0, 0.8, 1.2]) * (1 + arm) + rng.normal(0, 1, 3)
            recs.append({"id": i, "arm": arm,
                         "hr_V1": pre[0], "hr_V2": post[0],
                         "sbp_V1": pre[1], "sbp_V2": post[1],
                         "rr_V1": pre[2], "rr_V2": post[2]})
        return pd.DataFrame(recs)

    def _maps(self):
        return {"hr": {"pre": "hr_V1", "post": "hr_V2"},
                "sbp": {"pre": "sbp_V1", "post": "sbp_V2"},
                "rr": {"pre": "rr_V1", "post": "rr_V2"}}

    def test_to_long_multi_shape(self):
        lng = to_long_multi(self._wide(), "id", self._maps(), group_col="arm")
        assert list(lng.columns) == ["id", "time", "hr", "sbp", "rr", "arm"]
        assert len(lng) == 80  # 40 subjects x 2 timepoints

    def test_to_long_multi_mismatched_times_raises(self):
        bad = {"hr": {"pre": "hr_V1", "post": "hr_V2"},
               "sbp": {"pre": "sbp_V1"}}  # inconsistent timepoints
        with pytest.raises(ValueError):
            to_long_multi(self._wide(), "id", bad)

    def test_pipeline_omnibus_significant(self):
        info, perm = pipeline_paired_manova(
            self._wide(), id_col="id", outcomes=list(self._maps()),
            outcome_maps=self._maps(), group_col="arm", n_perm=499, verbose=False)
        assert info["Outcome type"] == "Multivariate (continuous)"
        assert info["PERMANOVA P (overall)"] < 0.05
        assert perm["Pseudo-F"] > 1

    def test_pipeline_export_sheets(self, tmp_path):
        pipeline_paired_manova(
            self._wide(), id_col="id", outcomes=list(self._maps()),
            outcome_maps=self._maps(), n_perm=199,
            save_path=str(tmp_path), verbose=False)
        d = tmp_path / "Longitudinal models" / "Paired_MANOVA"
        files = list(d.glob("*.xlsx"))
        assert len(files) == 1
        sheets = pd.ExcelFile(files[0]).sheet_names
        assert {"Setup Info", "PERMANOVA", "Per-Outcome Tests", "By Time"} <= set(sheets)
        fu = pd.read_excel(files[0], sheet_name="Per-Outcome Tests")
        assert "P (Holm)" in fu.columns

    def test_pipeline_requires_time_spec(self):
        with pytest.raises(ValueError):
            pipeline_paired_manova(self._wide(), id_col="id",
                                   outcomes=["hr"], verbose=False)


# --------------------------------------------------------------------------- #
# Completeness, imputation (MICE/LOCF) and AR1 correlation structure.
# --------------------------------------------------------------------------- #
class TestMissingnessImputationAR1:
    def _long_with_gaps(self, n=60, miss=0.15, seed=0):
        rng = np.random.RandomState(seed)
        visites = ["M0", "M3", "M6", "M12"]
        rows = []
        for i in range(n):
            b = rng.normal(10, 2)
            for k, v in enumerate(visites):
                rows.append({"id": i, "t": v, "y": b + 2 * np.log1p(k) + rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        df.loc[rng.rand(len(df)) < miss, "y"] = np.nan
        return df

    def test_missingness_report_counts_and_warns(self):
        df = self._long_with_gaps(miss=0.25)
        info, tbl = missingness_report(df, "id", "t", "y")
        assert info["Subjects total"] == 60
        assert info["Complete-case subjects dropped (%)"] > 0
        assert "Missingness warning" in info        # >20% dropped -> warning
        assert {"Group", "Time", "Observed", "Missing", "% missing"} <= set(tbl.columns)

    def test_mice_fills_all_missing(self):
        df = self._long_with_gaps()
        n_missing = int(df["y"].isna().sum())
        out, info = impute_long(df, "id", "t", "y", method="mice", n_iter=5)
        assert info["Imputation"] == "mice"
        assert info["Missing cells imputed"] == n_missing
        assert int(out["y"].isna().sum()) == 0
        assert "Rubin" in info["Imputation note"]   # single-imputation caveat present

    def test_locf_flagged_as_biased(self):
        df = self._long_with_gaps()
        out, info = impute_long(df, "id", "t", "y", method="locf")
        assert info["Imputation"] == "locf"
        assert "BIAS" in info["Imputation note"].upper()
        assert int(out["y"].isna().sum()) == 0

    def test_impute_invalid_method_raises(self):
        df = self._long_with_gaps()
        with pytest.raises(ValueError):
            impute_long(df, "id", "t", "y", method="bogus")

    def test_prepare_repeated_threads_report(self):
        df = self._long_with_gaps()
        db_long, time_col, report, miss_df, imp_info = prepare_repeated(
            df, "id", "y", time_col="t", impute="mice", n_iter=5)
        assert report["Imputation"] == "mice"
        assert "Imputation note" in report
        assert not miss_df.empty
        assert int(db_long["y"].isna().sum()) == 0

    def test_compare_corr_recommends_ar1_on_temporal(self):
        # AR1 data -> AR(1) must win on QIC
        rng = np.random.RandomState(1)
        n, k = 60, 4
        t = np.arange(k)
        cov = 0.8 ** np.abs(t[:, None] - t[None, :])
        X = rng.normal(0, 1, (n, k)) @ np.linalg.cholesky(cov).T
        rows = [{"id": i, "t": f"t{j}", "y": X[i, j]} for i in range(n) for j in range(k)]
        info, df = compare_corr_structures(pd.DataFrame(rows), "id", "t", "y")
        assert "Recommended working correlation (lower QIC)" in info
        assert set(df["Working correlation"]) == {"exchangeable", "ar1"}

    def test_pipeline_ols_with_mice_and_ar1(self):
        df = self._long_with_gaps()
        info, model = pipeline_paired_ols(df, id_col="id", outcome="y",
                                          time_col="t", impute="mice", verbose=False)
        assert info["Imputation"] == "mice"
        assert "Complete-case subjects dropped (%)" in info
        assert "Recommended working correlation (lower QIC)" in info   # >=3 timepoints
        assert model is not None

    def test_pipeline_ols_export_has_new_sheets(self, tmp_path):
        df = self._long_with_gaps()
        pipeline_paired_ols(df, id_col="id", outcome="y", time_col="t",
                            impute="mice", save_path=str(tmp_path), verbose=False)
        f = tmp_path / "Longitudinal models" / "Paired_OLS" / "y_paired_ols_report.xlsx"
        assert f.exists()
        sheets = pd.ExcelFile(f).sheet_names
        assert {"Missingness", "Correlation Structure"} <= set(sheets)


# --------------------------------------------------------------------------- #
# The template must run end to end (guard against an API drift).
# --------------------------------------------------------------------------- #
def test_longitudinal_template_runs():
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "templates" / "template_longitudinal.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "RÉSUMÉ FINAL" in res.stdout
