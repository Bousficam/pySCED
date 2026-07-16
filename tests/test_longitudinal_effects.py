"""Tests of the longitudinal extensions: count outcome (GEE Poisson/NB) and
effect sizes (dz, Hedges g, rank-biserial, eta2)."""
import numpy as np
import pandas as pd
import pytest

from functions.general.longitudinal.effects import (
    cohen_dz,
    rank_biserial_wilcoxon,
    eta_squared_rm,
    paired_effect_sizes,
    fit_gee_count,
    pipeline_paired_count,
    count_paired_tests,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prepost_continuous():
    rng = np.random.RandomState(0)
    n = 60
    pre = rng.normal(50, 8, n)
    post = pre + 5 + rng.normal(0, 4, n)        # clear progress
    return pd.concat([
        pd.DataFrame({"id": np.arange(n), "t": "pre", "y": pre}),
        pd.DataFrame({"id": np.arange(n), "t": "post", "y": post}),
    ], ignore_index=True)


@pytest.fixture(scope="module")
def threetimes_continuous():
    rng = np.random.RandomState(1)
    n = 40
    rows = []
    for i in range(n):
        b = rng.normal(10, 2)
        for k, t in enumerate(["t0", "t1", "t2"]):
            rows.append({"id": i, "t": t, "y": b + 2 * k + rng.normal(0, 1)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def count_prepost_overdispersed():
    rng = np.random.RandomState(2)
    n = 80
    # negative-binomial-like: overdispersion via a Poisson-Gamma mixture
    lam_pre = rng.gamma(2.0, 2.0, n)            # mean 4, large variance
    pre = rng.poisson(lam_pre)
    post = rng.poisson(lam_pre * 0.5)           # the rate halves
    return pd.concat([
        pd.DataFrame({"id": np.arange(n), "t": "pre", "events": pre}),
        pd.DataFrame({"id": np.arange(n), "t": "post", "events": post}),
    ], ignore_index=True)


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
class TestEffectSizes:
    def test_cohen_dz_matches_formula(self):
        pre = np.array([1.0, 2, 3, 4, 5])
        post = np.array([2.0, 3, 5, 6, 8])
        dz, g, n = cohen_dz(pre, post)
        diff = post - pre
        assert dz == pytest.approx(diff.mean() / diff.std(ddof=1))
        assert abs(g) < abs(dz)                 # Hedges correction shrinks
        assert n == 5

    def test_dz_large_for_clear_effect(self, prepost_continuous):
        w = prepost_continuous.pivot_table(index="id", columns="t", values="y")
        dz, g, n = cohen_dz(w["pre"], w["post"])
        assert dz > 0.8                          # large effect expected

    def test_rank_biserial_sign_and_range(self):
        pre = np.array([1.0, 2, 3, 4, 5])
        post = np.array([2.0, 3, 4, 5, 6])       # all +1 -> r = +1
        r, n = rank_biserial_wilcoxon(pre, post)
        assert r == pytest.approx(1.0)
        assert n == 5

    def test_eta_squared_in_unit_range(self, threetimes_continuous):
        es = eta_squared_rm(threetimes_continuous, "id", "t", "y")
        assert 0.0 <= es["eta_squared"] <= 1.0
        assert 0.0 <= es["partial_eta_squared"] <= 1.0
        assert es["eta_squared"] > 0.2           # true strong trend

    def test_paired_effect_sizes_two_times(self, prepost_continuous):
        df = paired_effect_sizes(prepost_continuous, "id", "t", "y")
        names = set(df["Effect size"])
        assert {"Cohen's dz (paired)", "Hedges' g", "Rank-biserial (Wilcoxon)"} <= names

    def test_paired_effect_sizes_multi_times(self, threetimes_continuous):
        df = paired_effect_sizes(threetimes_continuous, "id", "t", "y")
        assert "η² (time)" in set(df["Effect size"])


# --------------------------------------------------------------------------- #
# Count outcome
# --------------------------------------------------------------------------- #
class TestCountOutcome:
    def test_count_tests_wilcoxon(self, count_prepost_overdispersed):
        res = count_paired_tests(count_prepost_overdispersed, "id", "t", "events")
        assert res.iloc[0]["Test"] == "Wilcoxon signed-rank"
        assert res.iloc[0]["P"] < 0.05

    def test_gee_auto_detects_overdispersion(self, count_prepost_overdispersed):
        res, info = fit_gee_count(count_prepost_overdispersed, "id", "t", "events")
        assert res is not None
        # overdispersion detected -> switches to negative binomial
        assert info["Poisson Pearson dispersion (>1.5 => overdispersed)"] > 1.5
        assert info["Family"] == "Negative binomial"

    def test_gee_force_poisson(self, count_prepost_overdispersed):
        res, info = fit_gee_count(count_prepost_overdispersed, "id", "t", "events",
                                  family="poisson")
        assert info["Family"] == "Poisson"

    def test_gee_invalid_family_raises(self, count_prepost_overdispersed):
        with pytest.raises(ValueError):
            fit_gee_count(count_prepost_overdispersed, "id", "t", "events", family="bogus")

    def test_pipeline_returns_irr_and_detects_decrease(self, count_prepost_overdispersed):
        info, model = pipeline_paired_count(
            count_prepost_overdispersed, id_col="id", outcome="events",
            time_col="t", verbose=False)
        assert model is not None
        assert info["Outcome type"] == "Count"
        # the rate drops at 'post' -> at least one significant time term
        time_terms = [t for t in model.params.index if "t" in t.lower() and t != "Intercept"]
        assert any(model.pvalues[t] < 0.05 for t in time_terms)

    def test_pipeline_export_sheets(self, tmp_path, count_prepost_overdispersed):
        pipeline_paired_count(
            count_prepost_overdispersed, id_col="id", outcome="events",
            time_col="t", save_path=str(tmp_path), verbose=False)
        f = tmp_path / "Longitudinal models" / "Paired_Count" / "events_paired_count_report.xlsx"
        assert f.exists()
        sheets = pd.ExcelFile(f).sheet_names
        assert {"Setup Info", "Rate Ratios", "Tests", "By Time"} <= set(sheets)
        irr = pd.read_excel(f, sheet_name="Rate Ratios")
        assert "IRR (95% CI)" in irr.columns

    def test_pipeline_count_mice_imputation(self, count_prepost_overdispersed):
        df = count_prepost_overdispersed.copy()
        rng = np.random.RandomState(0)
        df.loc[rng.rand(len(df)) < 0.15, "events"] = np.nan
        info, model = pipeline_paired_count(df, id_col="id", outcome="events",
                                            time_col="t", impute="mice", verbose=False)
        assert info["Imputation"] == "mice"
        assert "Complete-case subjects dropped (%)" in info
        assert model is not None
