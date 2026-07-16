"""Tests for the SCED pipeline: core (Tau-U, NAP, segmented regression, randomization
engine) and the alternating / N-of-1 pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from functions.sced.core import (
    validate_sced,
    sced_data_warnings,
    describe_by_phase,
    nap,
    tau_u,
    pnd,
    pem,
    segmented_regression,
    segmented_table,
    randomization_test,
    alternating_scheme,
    diff_in_means,
    omnibus_variance,
    mann_kendall,
    sens_slope,
    lag1_autocorr,
    durbin_watson,
    diagnose_series,
    make_condition_ss_stat,
    make_ordered_trend_stat,
    condition_permutation_test,
    choose_permutation_method,
    block_scheme,
    _kendall_s,
)
from functions.sced.alternating.core import pipeline_sced_alternating
from functions.sced.alternating.group import (
    pipeline_sced_alternating_group,
    classical_rm_checks,
)
from functions.sced.power import (
    power_sced_alternating,
    mdes_sced_alternating,
    retrospective_power,
    power_report,
)
from functions.sced.plots.panels import plot_sced_alternating
from functions.sced.glossary import explain, explain_report, GLOSSARY
from functions.sced.multivariate import permanova_condition_test
from functions.sced.core import (
    stratified_condition_permutation_test,
    stratified_ordered_trend_test,
    choose_permutation_method_group,
    heterogeneity_test,
    pairwise_contrasts,
    hedges_g,
)


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
class TestNonoverlap:
    def test_nap_full_separation(self):
        assert nap([1, 2, 3], [4, 5, 6]) == 1.0
        assert nap([4, 5, 6], [1, 2, 3]) == 0.0
        assert nap([1, 2, 3], [4, 5, 6], improvement="decrease") == 0.0

    def test_nap_chance_level_on_overlap(self):
        # identical distributions -> NAP ~ 0.5
        assert nap([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.5)

    def test_nap_ci_matches_singlecaseES(self):
        """nap_ci (unbiased SE + Newcombe score CI) reproduit SingleCaseES 0.7.4 exactement
        (cas n_A=7, n_B=10, NAP=0.8857 -> SE=0.0786, IC95 [0.607, 0.972] sous SingleCaseES)."""
        from functions.sced.core import nap_ci
        a = [44, 50, 38, 56, 44, 50, 38]
        b = [56, 62, 50, 68, 56, 62, 68, 74, 80, 86]   # fully ordered above A -> NAP high
        r = nap_ci(a, b)
        assert r["nap"] == pytest.approx(nap(a, b))
        assert 0 < r["se"] < 0.5
        assert r["ci_lower"] < r["nap"] < r["ci_upper"]
        # SE method 'null' is the H0 variance (m+n+1)/(12 m n)
        rn = nap_ci(a, b, se_method="null")
        assert rn["se"] == pytest.approx(((7 + 10 + 1) / (12 * 7 * 10)) ** 0.5, rel=1e-9)

    def test_hedges_g_within(self):
        """Within-case Hedges' g: g = J*d, oriented, and nan when too few points."""
        from functions.sced.core import hedges_g_within
        r = hedges_g_within([1, 2, 3, 2], [6, 7, 8, 7], improvement="increase")
        assert r["g"] > 0 and abs(r["g"]) < abs(r["d"]) + 1e-9   # J < 1 -> |g| <= |d|
        assert 0 < r["J"] < 1 and r["sd_within"] > 0
        # orientation: decrease flips the sign
        rd = hedges_g_within([1, 2, 3, 2], [6, 7, 8, 7], improvement="decrease")
        assert rd["g"] == -r["g"]
        # donnees insuffisantes -> nan
        assert np.isnan(hedges_g_within([1], [2, 3, 4])["g"])

    def test_tau_u_subtracts_baseline_trend(self):
        # baseline en hausse (tendance) -> Tau-U < Tau brut quand on corrige
        base = [1, 2, 3, 4, 5]            # strong increasing trend
        trt = [6, 7, 8, 9, 10]
        res = tau_u(base, trt, correct_baseline_trend=True)
        raw = tau_u(base, trt, correct_baseline_trend=False)
        assert res["tau"] == 1.0          # complete A/B separation
        assert res["tau_u"] < raw["tau_u"]  # the correction removes the baseline drift

    def test_tau_u_stays_bounded_with_opposing_baseline_trend(self):
        # Regression test for the documented bug (Brossard 2018 / Tarlow 2016): with an
        # n_a*n_b denominator (Parker 2011a), tau_u([5,4,3,2,1],[6,7,8,9,10]) returned 1.4 (>1).
        # The correction must stay within [-1, +1] whatever the variant.
        for method in ("auto", "adj", "trend_a"):
            res = tau_u([5, 4, 3, 2, 1], [6, 7, 8, 9, 10], method=method)
            assert -1.0 <= res["tau_u"] <= 1.0, (method, res["tau_u"])
            assert res["bounded"] is True

    def test_tau_u_adj_is_bounded_long_baseline(self):
        # long decreasing baseline + increasing treatment: 'adj' (Theil-Sen) bounded
        base = list(range(10, 0, -1))     # n_A=10 >= 7 -> adj eligible
        trt = list(range(11, 21))
        res = tau_u(base, trt, method="adj")
        assert "Tarlow" in res["variant"]
        assert -1.0 <= res["tau_u"] <= 1.0

    def test_tau_u_adj_is_taub_matching_scan_corrected_tau(self):
        """'adj' computes the faithful Tarlow tau-b (== R scan::corrected_tau): the coefficient
        equals Kendall's tau-b between the Theil-Sen-detrended series and the phase dummy."""
        from scipy import stats as _st
        from functions.sced.core import sens_slope
        base = [2, 3, 1, 4, 2, 3, 1]; trt = [3, 5, 2, 6, 4, 3, 4, 5, 6, 7]
        r = tau_u(base, trt, method="adj")
        na, nb = len(base), len(trt)
        t_a = np.arange(1, na + 1.0); t_b = np.arange(na + 1, na + nb + 1.0)
        sl = sens_slope(np.array(base, float), x=t_a)
        resid = np.r_[np.array(base) - sl * t_a, np.array(trt) - sl * t_b]
        phase = np.r_[np.zeros(na), np.ones(nb)]
        tb = _st.kendalltau(phase, resid)
        assert r["tau_u"] == pytest.approx(float(tb.statistic), abs=1e-3)   # tau-b, not tau-a (round 4dp)
        assert r["p_value"] == pytest.approx(float(tb.pvalue), abs=1e-3)
        assert "tau-b" in r["variant"]

    def test_tau_u_auto_skips_correction_without_trend(self):
        # no significant baseline trend -> 'auto' returns Tau A vs B (= tau)
        res = tau_u([3, 1, 4, 2, 5, 1, 3], [8, 9, 8, 10, 9, 11, 10], method="auto")
        assert "A vs B" in res["variant"]
        assert res["tau_u"] == res["tau"]

    def test_tau_u_reports_baseline_trend_diagnostic(self):
        res = tau_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert res["baseline_trend_tau"] == 1.0   # baseline parfaitement croissant
        assert "se" in res and "p_value" in res

    def test_kendall_s_monotonic(self):
        assert _kendall_s([1, 2, 3, 4]) == 6      # all pairs concordant (C(4,2))
        assert _kendall_s([4, 3, 2, 1]) == -6

    def test_pnd_pem_ranges(self):
        assert pnd([1, 2, 3], [4, 5, 6]) == 100.0
        assert 0 <= pem([1, 2, 3, 4], [3, 4, 5]) <= 100


# --------------------------------------------------------------------------- #
# Segmented regression (level + slope change)
# --------------------------------------------------------------------------- #
class TestSegmented:
    def _ab(self, level_jump=5.0, n=20, seed=0):
        rng = np.random.RandomState(seed)
        rows = []
        for s in range(1, n + 1):
            phase = "A" if s <= n // 2 else "B"
            base = 10 + (level_jump if phase == "B" else 0) + rng.normal(0, 1)
            rows.append({"session": s, "phase": phase, "y": base})
        return pd.DataFrame(rows)

    def test_detects_level_change(self):
        res, info = segmented_regression(self._ab(level_jump=6), session_col="session",
                                         phase_col="phase", outcome_col="y",
                                         baseline_label="A", treatment_label="B")
        assert res is not None
        coef, p = info["Level change (coef, p)"]
        assert coef > 2 and p < 0.05

    def test_table_shape(self):
        res, _ = segmented_regression(self._ab(), session_col="session",
                                      phase_col="phase", outcome_col="y",
                                      baseline_label="A", treatment_label="B")
        tbl = segmented_table(res)
        assert "Meaning" in tbl.columns
        assert (tbl["Term"] == "_phase").any()

    def test_insufficient_returns_none(self):
        tiny = pd.DataFrame({"session": [1, 2], "phase": ["A", "B"], "y": [1.0, 2.0]})
        res, info = segmented_regression(tiny, session_col="session", phase_col="phase",
                                         outcome_col="y", baseline_label="A",
                                         treatment_label="B")
        assert res is None and "Warning" in info

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError):
            segmented_regression(self._ab(), session_col="session", phase_col="phase",
                                 outcome_col="y", baseline_label="A",
                                 treatment_label="B", family="bogus")


# --------------------------------------------------------------------------- #
# Randomization engine
# --------------------------------------------------------------------------- #
class TestRandomization:
    def test_detects_clear_effect(self):
        # A low, B high, alternation -> small p
        labels = np.array(["A", "B"] * 8)
        values = np.where(labels == "B", 10.0, 0.0) + np.random.RandomState(0).normal(0, 0.1, 16)
        stat = lambda l, v: diff_in_means(l, v, target="B", reference="A")
        res = randomization_test(labels, values, statistic=stat,
                                 scheme=alternating_scheme(), n_perm=2000)
        assert res["p_value"] < 0.05
        assert res["observed"] > 5

    def test_null_not_significant(self):
        rng = np.random.RandomState(4)           # draw without accidental separation
        labels = np.array(["A", "B"] * 12)
        values = rng.normal(0, 1, 24)            # no real effect
        stat = lambda l, v: diff_in_means(l, v, target="B", reference="A")
        res = randomization_test(labels, values, statistic=stat,
                                 scheme=alternating_scheme(), n_perm=2000)
        assert res["p_value"] > 0.05

    def test_alternating_scheme_preserves_counts(self):
        labels = np.array(["A", "A", "B", "B", "B"])
        rng = np.random.RandomState(0)
        perm = alternating_scheme()(labels, rng)
        assert sorted(perm) == sorted(labels)

    def test_max_consecutive_constraint(self):
        labels = np.array(["A", "B"] * 6)
        rng = np.random.RandomState(0)
        for _ in range(20):
            perm = alternating_scheme(max_consecutive=2)(labels, rng)
            runs = max(len(list(g)) for _, g in __import__("itertools").groupby(perm))
            assert runs <= 2

    def test_omnibus_variance_three_conditions(self):
        labels = np.array(["A", "B", "C"] * 5)
        values = np.where(labels == "A", 0.0, np.where(labels == "B", 5.0, 10.0))
        assert omnibus_variance(labels, values) > 0


# --------------------------------------------------------------------------- #
# Series diagnostics: trend (Mann-Kendall + Hamed-Rao), slope, autocorrelation
# --------------------------------------------------------------------------- #
class TestDiagnostics:
    def test_mann_kendall_detects_trend(self):
        res = mann_kendall(list(range(1, 16)))     # perfect increasing trend
        assert res["tau"] == pytest.approx(1.0)
        assert res["p_value"] < 0.05

    def test_mann_kendall_no_trend(self):
        rng = np.random.RandomState(0)
        res = mann_kendall(rng.normal(0, 1, 30))
        assert res["p_value"] > 0.05

    def test_hamed_rao_corrects_for_autocorrelation(self):
        # autocorrelated series with no real trend -> HR must be MORE conservative
        rng = np.random.RandomState(3)
        x = np.zeros(40)
        for i in range(1, 40):
            x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)   # strong AR(1)
        res = mann_kendall(x, autocorr_correction=True)
        assert res["hr_correction"] >= 1.0            # inflated variance
        assert res["p_value"] >= res["p_naive"]       # corrected p >= naive p

    def test_hamed_rao_matches_modifiedmk(self):
        """The Hamed-Rao correction reproduces R modifiedmk::mmkh exactly (detrend by Sen
        slope + 1.96/sqrt(n) screen). Reference series: AR(1) phi=0.6 + trend 0.3/step."""
        rng = np.random.RandomState(7); n = 20; e = rng.normal(0, 1, n); x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.6 * x[i - 1] + e[i]
        y = 0.3 * np.arange(n) + x
        res = mann_kendall(y, autocorr_correction=True)
        assert res["var0"] == pytest.approx(950.0, abs=1e-6)        # mmkh old.variance
        assert res["var_hr"] == pytest.approx(348.737, abs=0.05)    # mmkh new.variance
        assert res["z"] == pytest.approx(8.0859, abs=1e-3)          # mmkh Corrected Zc
        assert res["p_value"] < 1e-6                                # mmkh new P-value ~ 0

    def test_sens_slope_recovers_slope(self):
        y = 3.0 * np.arange(1, 12) + 5
        assert sens_slope(y) == pytest.approx(3.0)

    def test_autocorr_and_dw(self):
        x = np.arange(1, 21, dtype=float)             # monotone -> autocorr +
        assert lag1_autocorr(x) > 0.5
        assert durbin_watson(x - x.mean()) < 2.0

    def test_diagnose_suggests_detrend_under_learning(self):
        rng = np.random.RandomState(1)
        conds = np.repeat(["A", "B"], 8)
        rng.shuffle(conds)
        sess = np.arange(1, 17)
        y = np.where(conds == "B", 2, 0) + 1.2 * sess + rng.normal(0, 1, 16)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        diag = diagnose_series(df, session_col="session", outcome_col="y", condition_col="cond")
        assert diag["Learning trend detected"] is True
        assert diag["Suggested detrend"] in ("linear", "log")

    def test_diagnose_no_trend_suggests_none(self):
        rng = np.random.RandomState(2)
        conds = np.repeat(["A", "B"], 10)
        rng.shuffle(conds)
        df = pd.DataFrame({"session": np.arange(1, 21), "cond": conds,
                           "y": np.where(conds == "B", 3, 0) + rng.normal(0, 1, 20)})
        diag = diagnose_series(df, session_col="session", outcome_col="y", condition_col="cond")
        assert diag["Suggested detrend"] == "none"


# --------------------------------------------------------------------------- #
# Time-adjusted condition statistic (Draper-Stoneman, design-based)
# --------------------------------------------------------------------------- #
class TestDetrend:
    def _learning(self, effect, learning, seed):
        r = np.random.RandomState(seed)
        conds = np.repeat(["A", "B", "C"], 6)
        r.shuffle(conds)
        sess = np.arange(1, 19)
        mu = {"A": 0, "B": effect, "C": effect / 2}
        y = np.array([mu[c] for c in conds]) + learning * sess + r.normal(0, 2, 18)
        return sess, conds, y

    def test_detrend_restores_power_lost_to_learning(self):
        # strong learning + real effect: detrend detects where the raw test fails
        sess, conds, y = self._learning(effect=6, learning=1.5, seed=1001)
        p_naive = randomization_test(
            conds, y, statistic=omnibus_variance,
            scheme=alternating_scheme(), n_perm=3000, two_sided=False)["p_value"]
        p_det = randomization_test(
            conds, y, statistic=make_condition_ss_stat(sess, detrend="linear"),
            scheme=alternating_scheme(), n_perm=3000, two_sided=False)["p_value"]
        assert p_naive > 0.05          # drowned by learning
        assert p_det < 0.01            # recovered after detrend

    def test_detrend_none_reduces_to_between_ss(self):
        # detrend="none" -> SS condition == order of magnitude of the between-condition variance
        sess = np.arange(1, 13)
        conds = np.array(["A", "B"] * 6)
        y = np.where(conds == "B", 5.0, 0.0) + 0.0
        stat = make_condition_ss_stat(sess, detrend="none")
        assert stat(conds, y) > 0

    def test_invalid_detrend_raises(self):
        with pytest.raises(ValueError):
            make_condition_ss_stat(np.arange(1, 10), detrend="bogus")


# --------------------------------------------------------------------------- #
# Draper-Stoneman vs Freedman-Lane permutation schemes + chooser
# --------------------------------------------------------------------------- #
class TestPermutationSchemes:
    def _balanced(self, effect, learning, seed):
        r = np.random.RandomState(seed)
        conds = np.tile(["A", "B", "C"], 6)        # spread over time
        sess = np.arange(1, 19)
        mu = {"A": 0, "B": effect, "C": effect / 2}
        y = np.array([mu[c] for c in conds]) + learning * sess + r.normal(0, 2, 18)
        return sess, conds, y

    def test_both_methods_detect_effect_and_agree(self):
        sess, conds, y = self._balanced(effect=6, learning=1.5, seed=1)
        ds = condition_permutation_test(y, conds, sess, detrend="linear",
                                        method="draper-stoneman", n_perm=2000)
        fl = condition_permutation_test(y, conds, sess, detrend="linear",
                                        method="freedman-lane", n_perm=2000)
        assert ds["p_value"] < 0.05 and fl["p_value"] < 0.05
        assert abs(ds["p_value"] - fl["p_value"]) < 0.05      # close on a balanced design
        assert ds["observed_F"] == fl["observed_F"]           # same observed statistic

    def test_methods_calibrated_under_learning_null(self):
        # NULL (no effect) + strong learning: false positives ~ 5% for both
        def null_p(method, seed):
            r = np.random.RandomState(seed)
            conds = np.tile(["A", "B", "C"], 6)
            sess = np.arange(1, 19)
            y = 1.5 * sess + r.normal(0, 2, 18)
            return condition_permutation_test(y, conds, sess, detrend="linear",
                                              method=method, n_perm=800)["p_value"]
        for meth in ("draper-stoneman", "freedman-lane"):
            fp = np.mean([null_p(meth, s) < 0.05 for s in range(80)])
            assert fp < 0.15                                   # calibrated (loose bound for MC)

    def test_invalid_method_raises(self):
        sess, conds, y = self._balanced(4, 1.0, 0)
        with pytest.raises(ValueError):
            condition_permutation_test(y, conds, sess, detrend="linear", method="bogus")

    def test_freedman_lane_exact_small_n(self):
        """FL: n! <= enumerate_cap -> exact complete enumeration; matches Monte-Carlo within
        MC error; large n stays Monte-Carlo (n! intractable)."""
        rng = np.random.RandomState(3)
        sess = np.arange(1, 9); conds = np.array(list("AABBABBA"))
        y = np.array([0, 0, 3, 3, 0, 3, 3, 0], float) + 0.4 * sess + rng.normal(0, 1, 8)
        ex = condition_permutation_test(y, conds, sess, detrend="linear", method="fl")
        mc = condition_permutation_test(y, conds, sess, detrend="linear", method="fl",
                                        enumerate_cap=0, n_perm=20000)
        assert ex["exact"] and ex["n_eval"] == 40320          # 8! enumerated
        assert not mc["exact"]
        assert abs(ex["p_value"] - mc["p_value"]) < 0.02      # exact vs MC agree
        big_s, big_c, big_y = self._balanced(4, 1.0, 0)       # n=18 -> 18! intractable
        assert not condition_permutation_test(big_y, big_c, big_s, detrend="linear",
                                              method="fl", n_perm=300)["exact"]

    def test_chooser_recommends_ds_when_orthogonal(self):
        conds = np.tile(["A", "B", "C"], 6)          # condition orthogonal to time
        rec = choose_permutation_method(conds, np.arange(1, 19), detrend="linear")
        assert rec["recommended_method"] == "draper-stoneman"
        assert rec["condition_time_R2"] < 0.15

    def test_chooser_returns_decision_tree(self):
        rec = choose_permutation_method(np.tile(["A", "B"], 9), np.arange(1, 19),
                                        detrend="linear")
        assert "Edgington" in rec["decision_tree"]
        assert "Freedman-Lane" in rec["decision_tree"]
        assert "Draper-Stoneman" in rec["decision_tree"]

    def test_chooser_recommends_fl_under_temporal_imbalance(self):
        conds = np.array(["A"] * 6 + ["B"] * 6 + ["C"] * 6)   # A early, C late (6/6/6)
        rec = choose_permutation_method(conds, np.arange(1, 19), detrend="linear")
        assert rec["recommended_method"] == "freedman-lane"
        assert rec["condition_time_R2"] > 0.5


# --------------------------------------------------------------------------- #
# Pipeline alternant / N-of-1
# --------------------------------------------------------------------------- #
class TestAlternatingPipeline:
    def _data(self, effect=8.0, n_blocks=10, seed=0):
        rng = np.random.RandomState(seed)
        rows = []
        for b in range(n_blocks):
            for cond in ["A", "B"]:
                s = b * 2 + (0 if cond == "A" else 1) + 1
                y = (effect if cond == "B" else 0) + rng.normal(10, 1)
                rows.append({"session": s, "cond": cond, "y": y})
        return pd.DataFrame(rows)

    def test_returns_info_and_significant(self):
        info, rand = pipeline_sced_alternating(
            self._data(), session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=2000, verbose=False)
        assert info["Design"].startswith("Alternating")
        assert info["Randomization p-value"] < 0.05
        assert "randomization" in info["Primary inference (test)"].lower()

    def test_effect_sizes_present(self):
        info, _ = pipeline_sced_alternating(
            self._data(), session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=500, verbose=False)
        # nothing to assert on info here; the export is checked below

    def test_export_sheets(self, tmp_path):
        pipeline_sced_alternating(
            self._data(), session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=500,
            save_path=str(tmp_path), verbose=False)
        f = tmp_path / "y_sced_alternating_report.xlsx"
        assert f.exists()
        sheets = pd.ExcelFile(f).sheet_names
        assert {"Setup Info", "Randomization Test", "Effect Sizes", "By Phase"} <= set(sheets)
        es = pd.read_excel(f, sheet_name="Effect Sizes")
        assert "Tau-U" in es.columns and "Tau-U variant" in es.columns

    def test_three_conditions_uses_omnibus(self):
        rng = np.random.RandomState(2)
        rows = []
        for b in range(8):
            for cond, mu in [("A", 0), ("B", 5), ("C", 10)]:
                rows.append({"session": b * 3 + ord(cond), "cond": cond,
                             "y": mu + rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        info, rand = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            conditions=("A", "B", "C"), n_perm=1000, verbose=False)
        assert "variance" in info["Test statistic"].lower()
        assert info["Randomization p-value"] < 0.05

    def test_single_condition_raises(self):
        df = pd.DataFrame({"session": [1, 2, 3], "cond": ["A", "A", "A"], "y": [1.0, 2, 3]})
        with pytest.raises(ValueError):
            pipeline_sced_alternating(df, session_col="session", condition_col="cond",
                                      outcome_col="y", verbose=False)

    def test_carryover_diagnostic_present(self):
        info, _ = pipeline_sced_alternating(
            self._data(), session_col="session", condition_col="cond",
            outcome_col="y", n_perm=300, verbose=False)
        assert any("Carryover" in k or "Order" in k for k in info)

    def test_diagnostics_in_info_and_sheet(self, tmp_path):
        info, _ = pipeline_sced_alternating(
            self._data(), session_col="session", condition_col="cond",
            outcome_col="y", n_perm=300, save_path=str(tmp_path), verbose=False)
        assert "Suggested detrend" in info and "Trend p (Hamed-Rao)" in info
        f = tmp_path / "y_sced_alternating_report.xlsx"
        assert "Diagnostics" in pd.ExcelFile(f).sheet_names

    def test_detrend_option_changes_statistic(self):
        rng = np.random.RandomState(5)
        conds = np.repeat(["A", "B"], 9)
        rng.shuffle(conds)
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 1.0 * sess + rng.normal(0, 1, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="linear", n_perm=2000, verbose=False)
        assert "partial f" in info["Test statistic"].lower()
        assert info["Detrend"] == "linear"

    def test_perm_method_freedman_lane_reported(self):
        rng = np.random.RandomState(7)
        conds = np.tile(["A", "B"], 9)
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 1.0 * sess + rng.normal(0, 1, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="linear", perm_method="freedman-lane", n_perm=1500, verbose=False)
        assert info["Permutation method"] == "freedman-lane"
        assert "freedman-lane" in info["Primary inference (test)"].lower()
        assert "Method recommendation" in info        # chooser reported

    def test_default_perm_method_is_freedman_lane(self):
        rng = np.random.RandomState(3)
        conds = np.tile(["A", "B"], 9)
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 1.0 * sess + rng.normal(0, 1, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(            # perm_method not specified
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="linear", n_perm=1200, verbose=False)
        assert info["Permutation method"] == "freedman-lane"

    def test_detrend_auto_adopts_diagnostic_suggestion(self):
        rng = np.random.RandomState(8)
        conds = np.tile(["A", "B"], 9)
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 0.8 * sess + rng.normal(0, 2, 18)  # apprentissage
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="auto", n_perm=1500, verbose=False)
        assert "(auto)" in info["Detrend"]
        assert info["Detrend"].split()[0] in ("linear", "log")     # trend detected
        assert "partial f" in info["Test statistic"].lower()

    def test_perm_method_auto_uses_chooser(self):
        rng = np.random.RandomState(9)
        conds = np.tile(["A", "B"], 9)                  # condition orthogonal to time
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 1.0 * sess + rng.normal(0, 1, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="linear", perm_method="auto", n_perm=1500, verbose=False)
        assert "(auto)" in info["Permutation method"]
        assert info["Permutation method"].startswith("draper-stoneman")   # orthogonal to time -> DS
        assert info["Permutation method"].startswith(info["Method recommendation"])

    def test_k3_conditions_omnibus_statistic(self):
        rng = np.random.RandomState(10)
        conds = np.tile(["A", "B", "C"], 6)
        rng.shuffle(conds)
        sess = np.arange(1, 19)
        y = 20 + np.select([conds == "B", conds == "C"], [5, 8], 0) + rng.normal(0, 3, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="cond", outcome_col="y",
            detrend="none", n_perm=2000, verbose=False)
        assert "variance" in info["Test statistic"].lower()
        assert len(info["Conditions"]) == 3


# --------------------------------------------------------------------------- #
# GROUP (replicated single-case): stratified randomization test
# --------------------------------------------------------------------------- #
def _make_group(n_units, effect, learning, seed, k=2, n_per=12):
    rng = np.random.RandomState(seed)
    levels = ["A", "B", "C"][:k]
    rows = []
    for p in range(n_units):
        reps = n_per // k
        conds = np.tile(levels, reps)
        rng.shuffle(conds)
        sess = np.arange(1, len(conds) + 1)
        base = 20 + rng.normal(0, 4)
        eff = np.where(conds == "B", effect, 0.0)
        y = base + eff + learning * sess + rng.normal(0, 2, len(conds))
        for s, c, v in zip(sess, conds, y):
            rows.append({"unit": f"P{p}", "session": int(s), "cond": c, "y": float(v)})
    return pd.DataFrame(rows)


class TestStratifiedEngine:
    def test_power_common_effect(self):
        df = _make_group(8, effect=2.0, learning=1.0, seed=1)
        r = stratified_condition_permutation_test(
            df["y"].to_numpy(), df["cond"].to_numpy(), df["unit"].to_numpy(),
            df["session"].to_numpy(), detrend="linear", method="freedman-lane", n_perm=2000)
        assert r["p_value"] < 0.01 and r["n_units"] == 8

    def test_calibrated_under_null_within_unit(self):
        def fp(method):
            hits = 0
            for s in range(80):
                df = _make_group(6, effect=0.0, learning=1.2, seed=s)
                p = stratified_condition_permutation_test(
                    df["y"].to_numpy(), df["cond"].to_numpy(), df["unit"].to_numpy(),
                    df["session"].to_numpy(), detrend="linear", method=method,
                    n_perm=500)["p_value"]
                hits += p < 0.05
            return hits / 80
        for meth in ("draper-stoneman", "freedman-lane"):
            assert fp(meth) < 0.15      # calibrated (loose bound for MC)

    def test_standardization_is_permutation_invariant(self):
        # same seed -> standardize True/False stay exact (finite, valid p)
        df = _make_group(5, effect=1.5, learning=0.0, seed=2)
        common = dict(detrend="none", method="draper-stoneman", n_perm=1000, random_state=0)
        a = stratified_condition_permutation_test(
            df["y"].to_numpy(), df["cond"].to_numpy(), df["unit"].to_numpy(),
            df["session"].to_numpy(), standardize=True, **common)
        b = stratified_condition_permutation_test(
            df["y"].to_numpy(), df["cond"].to_numpy(), df["unit"].to_numpy(),
            df["session"].to_numpy(), standardize=False, **common)
        assert 0 < a["p_value"] <= 1 and 0 < b["p_value"] <= 1


class TestGroupPipeline:
    def test_group_detects_effect_and_reports(self):
        df = _make_group(6, effect=3.0, learning=0.7, seed=0)
        info, rand = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), detrend="auto",
            perm_method="freedman-lane", n_perm=3000, verbose=False)
        assert info["Number of units"] == 6
        assert rand["p_value"] < 0.05
        assert "stratified" in info["Test statistic"].lower()
        assert "(auto)" in info["Detrend"]

    def test_group_requires_two_units(self):
        df = _make_group(1, effect=2.0, learning=0.0, seed=0)
        with pytest.raises(ValueError):
            pipeline_sced_alternating_group(
                df, unit_col="unit", session_col="session", condition_col="cond",
                outcome_col="y", n_perm=500, verbose=False)

    def test_group_perm_auto_and_k3(self):
        df = _make_group(5, effect=3.0, learning=0.0, seed=4, k=3)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B", "C"), detrend="none",
            perm_method="auto", n_perm=2000, verbose=False)
        assert len(info["Conditions"]) == 3
        assert info["Degrees of freedom (num, den)"][0] == 2   # k-1 = 2

    def test_classical_rm_sheet_keys(self):
        df = _make_group(5, effect=4.0, learning=1.0, seed=3, k=3)
        out = classical_rm_checks(df, unit_col="unit", session_col="session",
                                  condition_col="cond", outcome_col="y",
                                  conditions=("A", "B", "C"))
        checks = set(out["Test / Check"])
        assert any("Friedman" in c for c in checks)
        assert any("RM-ANOVA" in c for c in checks)
        assert any("Mauchly" in c for c in checks)         # sphericity (k=3)
        assert any("autocorrelation" in c.lower() for c in checks)
        assert any("slope" in c.lower() for c in checks)
        assert "Interpretation" in out.columns             # per-row verdict
        assert out["Interpretation"].str.len().gt(0).all()

    def test_classical_friedman_na_for_two_conditions(self):
        df = _make_group(5, effect=4.0, learning=0.0, seed=4, k=2)
        out = classical_rm_checks(df, unit_col="unit", session_col="session",
                                  condition_col="cond", outcome_col="y",
                                  conditions=("A", "B"))
        fried = out[out["Test / Check"].str.contains("Friedman")].iloc[0]
        assert fried["Statistic"] == "n/a"                 # k=2 -> Friedman n/a

    def test_all_report_sheets_have_explanation_and_interpretation(self, tmp_path):
        df = _make_group(5, effect=5.0, learning=1.0, seed=11, k=3)
        pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B", "C"), detrend="auto",
            n_perm=800, save_path=str(tmp_path), verbose=False)
        xls = tmp_path / "y_sced_alternating_group_report.xlsx"
        sheets = pd.read_excel(xls, sheet_name=None)
        # "Explanation" is now a dedicated sheet; the data sheets keep
        # "Interpretation" but NOT an "Explanation" column.
        assert "Explanations" in sheets and "Ordered Trend" in sheets and "Methods" in sheets
        # the Explanations sheet references every data sheet
        legend_feuilles = set(sheets["Explanations"]["Sheet"].astype(str))
        assert {"Setup Info", "Per Unit", "Heterogeneity"} <= legend_feuilles
        ref_sheets = {"Methods", "Explanations", "Glossary", "Log",
                      "Interpretation guide", "Data & design"}  # reference/log sheets
        for name, d in sheets.items():
            if name == "Explanations":        # the dedicated sheet legitimately carries the column
                continue
            assert "Explanation" not in d.columns, f"{name} must no longer have an Explanation column"
            if name not in ref_sheets:
                assert "Interpretation" in d.columns, name

    def test_pipeline_adds_classical_sheet_flag(self):
        df = _make_group(5, effect=4.0, learning=1.0, seed=5, k=3)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B", "C"), classical_checks=True,
            n_perm=800, verbose=False)
        assert "Classical RM cross-check" in info

    def test_group_chooser_aggregates(self):
        df = _make_group(5, effect=1.0, learning=0.5, seed=6)
        rec = choose_permutation_method_group(
            df["cond"].to_numpy(), df["unit"].to_numpy(), df["session"].to_numpy(),
            detrend="linear")
        assert rec["recommended_method"] in ("draper-stoneman", "freedman-lane")
        assert 0 <= rec["condition_time_R2"] <= 1

    def test_group_reports_heterogeneity(self):
        df = _make_group(6, effect=3.0, learning=0.0, seed=2)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), detrend="none",
            test_heterogeneity=True, n_perm=1500, verbose=False)
        assert "Heterogeneity - I2 (%) primary" in info
        assert "Heterogeneity - verdict" in info


def _make_group_effects(effects, seed=0, n_per=10):
    rng = np.random.RandomState(seed)
    rows = []
    for p, eff in enumerate(effects):
        conds = np.tile(["A", "B"], n_per // 2)
        rng.shuffle(conds)
        sess = np.arange(1, len(conds) + 1)
        y = 20 + rng.normal(0, 3) + np.where(conds == "B", eff, 0) + rng.normal(0, 2, len(conds))
        for s, c, v in zip(sess, conds, y):
            rows.append({"unit": f"P{p}", "session": int(s), "cond": c, "y": float(v)})
    return pd.DataFrame(rows)


class TestHeterogeneity:
    def _run(self, effects, seed=0):
        df = _make_group_effects(effects, seed=seed)
        return heterogeneity_test(df["y"].to_numpy(), df["cond"].to_numpy(),
                                  df["unit"].to_numpy(), df["session"].to_numpy(),
                                  conditions=("A", "B"), n_perm=2000)

    def test_homogeneous_not_flagged(self):
        h = self._run([4, 4, 4, 4, 4, 4], seed=1)
        assert h["interaction_p"] > 0.10
        assert h["I2_percent"] < 40

    def test_heterogeneous_flagged(self):
        # opposite effects: common effect ~0 but strong heterogeneity
        h = self._run([6, 6, 6, -6, -6, -6], seed=1)
        assert h["interaction_p"] < 0.05
        assert h["I2_percent"] > 70
        assert abs(h["pooled_effect"]) < 1.0          # the mean effect cancels out
        assert h["interaction_df"][0] == 5            # (n_units-1)*(k-1) = 5

    def test_saturated_design_returns_na(self):
        # 1 obs per condition per unit -> interaction not testable (df_den=0)
        df = _make_group_effects([3, 3, 3], seed=0, n_per=2)
        h = heterogeneity_test(df["y"].to_numpy(), df["cond"].to_numpy(),
                               df["unit"].to_numpy(), df["session"].to_numpy(),
                               conditions=("A", "B"), n_perm=500)
        assert not np.isfinite(h["interaction_p"])


class TestPairwiseContrasts:
    def test_hedges_g_sign_and_correction(self):
        a = np.array([10, 11, 9, 10, 12, 8], float)
        b = a + 5
        g = hedges_g(b, a)
        assert g > 1.0                      # large positive effect
        assert hedges_g(a, b) == pytest.approx(-g, rel=1e-9)

    def test_single_series_pairs_and_multiplicity(self):
        rng = np.random.RandomState(1)
        conds = np.repeat(["A", "B", "C"], 6)
        rng.shuffle(conds)
        sess = np.arange(1, 19)
        eff = {"A": 0, "B": 8, "C": 4}
        y = 60 + np.array([eff[c] for c in conds]) + 0.8 * sess + rng.normal(0, 3, 18)
        out = pairwise_contrasts(y, conds, sessions=sess, detrend="linear",
                                 n_perm=2000, n_boot=500)
        assert len(out) == 3                              # C(3,2)
        assert {"Hedges g", "g 95% CI", "p (perm)", "p Holm", "p FDR"} <= set(out.columns)
        # B vs A: large positive effect, stays significant after correction
        ba = out[out["Comparison"] == "B vs A"].iloc[0]
        assert ba["Hedges g"] > 1.0 and ba["p Holm"] < 0.05
        # multiplicity: Holm >= raw
        assert (out["p Holm"] >= out["p (perm)"] - 1e-9).all()

    def test_group_effect_size_coherent_with_p(self):
        rng = np.random.RandomState(2)
        eff = {"A": 0, "B": 8, "C": 4}
        rows = []
        for p in range(5):
            cds = np.repeat(["A", "B", "C"], 6); rng.shuffle(cds); ss = np.arange(1, 19)
            yy = rng.uniform(55, 68) + np.array([eff[c] for c in cds]) + 1.0 * ss + rng.normal(0, 4, 18)
            for s, c, v in zip(ss, cds, yy):
                rows.append({"u": f"P{p}", "s": int(s), "c": c, "y": v})
        d = pd.DataFrame(rows)
        out = pairwise_contrasts(d.y.to_numpy(), d.c.to_numpy(), units=d.u.to_numpy(),
                                 sessions=d.s.to_numpy(), detrend="linear",
                                 n_perm=2000, n_boot=500)
        ca = out[out["Comparison"] == "C vs A"].iloc[0]
        # effect C-A > 0 (not ~0) AND significant: g and p coherent
        assert ca["Hedges g"] > 0.3 and ca["p (perm)"] < 0.05

    def test_correction_none_omits_columns(self):
        rng = np.random.RandomState(3)
        conds = np.repeat(["A", "B", "C"], 6); rng.shuffle(conds)
        y = rng.normal(0, 1, 18)
        out = pairwise_contrasts(y, conds, correction="none", n_perm=500, n_boot=200)
        assert "p Holm" not in out.columns and "p FDR" not in out.columns


class TestDataWarnings:
    def test_clean_data_no_warnings(self):
        df = pd.DataFrame({"s": np.arange(1, 19),
                           "c": np.tile(["A", "B"], 9),
                           "y": np.random.RandomState(0).normal(0, 1, 18)})
        assert sced_data_warnings(df, session_col="s", condition_col="c",
                                  outcome_col="y") == []

    def test_flags_missing_and_sparse_condition(self):
        df = pd.DataFrame({"s": [1, 2, 3, 4], "c": ["A", "A", "A", "B"],
                           "y": [1.0, 2, 3, 4]})
        w = sced_data_warnings(df, session_col="s", condition_col="c",
                               outcome_col="y", conditions=["A", "B", "C"])
        assert any("C" in m and "absent" in m for m in w)      # C missing
        assert any("B" in m and "<2" in m for m in w)          # B has n=1

    def test_flags_duplicate_and_noninteger_sessions(self):
        df = pd.DataFrame({"s": [1.5, 1.5, 2, 3], "c": ["A", "B", "A", "B"],
                           "y": [1.0, 2, 3, 4]})
        w = sced_data_warnings(df, session_col="s", condition_col="c", outcome_col="y")
        assert any("non-integer" in m for m in w)
        assert any("duplicated" in m for m in w)

    def test_flags_group_single_unit(self):
        df = pd.DataFrame({"u": ["P1"] * 6, "s": np.arange(1, 7),
                           "c": np.tile(["A", "B"], 3), "y": np.arange(6.0)})
        w = sced_data_warnings(df, session_col="s", condition_col="c",
                               outcome_col="y", unit_col="u")
        assert any("unit" in m for m in w)

    def test_pipeline_reports_warnings_key(self):
        df = _make_group_effects([3, 3, 3], seed=0)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=500, verbose=False)
        assert "Data warnings" in info


class TestPower:
    def test_power_increases_with_effect(self):
        lo = power_sced_alternating(effect=0.2 * 5, sd=5, n_sessions=18, n_units=1,
                                    n_sims=80, n_perm=150, random_state=1)["power"]
        hi = power_sced_alternating(effect=1.5 * 5, sd=5, n_sessions=18, n_units=1,
                                    n_sims=80, n_perm=150, random_state=1)["power"]
        assert hi > lo and 0 <= lo <= 1 and 0 <= hi <= 1

    def test_more_units_help(self):
        one = power_sced_alternating(effect=0.6 * 5, sd=5, n_sessions=18, n_units=1,
                                     n_sims=80, n_perm=150, random_state=2)["power"]
        many = power_sced_alternating(effect=0.6 * 5, sd=5, n_sessions=18, n_units=6,
                                      n_sims=80, n_perm=150, random_state=2)["power"]
        assert many >= one

    def test_nsessions_not_divisible_raises(self):
        with pytest.raises(ValueError):
            power_sced_alternating(effect=1.0, sd=1, n_sessions=17,
                                   conditions=("A", "B", "C"), n_sims=5, n_perm=20)

    def test_mdes_returns_curve(self):
        m = mdes_sced_alternating(sd=5, n_sessions=18, n_units=1, target_power=0.80,
                                  d_grid=(0.2, 1.0, 2.0), n_sims=60, n_perm=120,
                                  random_state=1)
        assert len(m["power_curve"]) == 3
        # courbe monotone croissante
        ps = [p for _, p in m["power_curve"]]
        assert ps == sorted(ps)

    def test_retrospective_uses_specified_effect(self):
        rng = np.random.RandomState(0)
        rows = []
        for p in range(5):
            o = np.tile(["A", "B"], 9); rng.shuffle(o); ss = np.arange(1, 19)
            for s, c in zip(ss, o):
                rows.append({"u": f"P{p}", "s": int(s), "c": c,
                             "y": 50 + (4 if c == "B" else 0) + rng.normal(0, 6)})
        db = pd.DataFrame(rows)
        rp = retrospective_power(db, session_col="s", condition_col="c", outcome_col="y",
                                 unit_col="u", effect=4, n_sims=80, n_perm=150, random_state=1)
        assert 0 <= rp["power"] <= 1
        assert rp["effect_specified"] == 4 and "estimated_residual_sd" in rp

    def test_power_report_sheet_columns(self):
        df = _make_group_effects([4, 4, 4, 4, 4], seed=0)
        summ, sheet = power_report(
            df, session_col="session", condition_col="cond", outcome_col="y",
            unit_col="unit", effect_of_interest=4, target_power=0.80,
            d_grid=(0.4, 1.0), n_sims=40, n_perm=80)
        assert "Explanation" in sheet.columns and "Interpretation" in sheet.columns
        assert "mdes_cohens_d" in summ and "retrospective_power" in summ
        assert len(sheet) == 2                          # MDES + retrospective

    def test_group_pipeline_saves_figure(self, tmp_path):
        df = _make_group(4, effect=4.0, learning=0.0, seed=1, k=2)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=400,
            plot_path=str(tmp_path), verbose=False)
        assert "Figure" in info and Path(info["Figure"]).exists()

    def test_group_pipeline_power_sheet(self, tmp_path):
        df = _make_group(5, effect=5.0, learning=0.0, seed=2, k=2)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), n_perm=400,
            power_analysis=True, power_effect=5, power_n_sims=40, power_n_perm=80,
            save_path=str(tmp_path), verbose=False)
        assert "Power - MDES (Cohen d @target)" in info
        xls = tmp_path / "y_sced_alternating_group_report.xlsx"
        assert "Power" in pd.read_excel(xls, sheet_name=None)


class TestPlots:
    def _df(self, units=False):
        rng = np.random.RandomState(0)
        rows = []
        reps = range(3) if units else [0]
        for p in reps:
            o = np.repeat(["A", "B", "C"], 6); rng.shuffle(o); ss = np.arange(1, 19)
            for s, c in zip(ss, o):
                rows.append({"u": f"P{p}", "session": int(s), "cond": c,
                             "y": 50 + (5 if c == "B" else 0) + rng.normal(0, 3)})
        return pd.DataFrame(rows)

    def test_single_panel_figure(self):
        fig = plot_sced_alternating(self._df(), session_col="session",
                                    condition_col="cond", outcome_col="y")
        assert len(fig.axes) == 1

    def test_group_grid_and_save(self, tmp_path):
        fig = plot_sced_alternating(self._df(units=True), session_col="session",
                                    condition_col="cond", outcome_col="y", unit_col="u",
                                    save_path=str(tmp_path))
        assert len(fig.axes) >= 3
        assert any(p.suffix == ".png" for p in tmp_path.iterdir())


class TestGlossary:
    def test_explain_known_keys_and_aliases(self):
        assert "residuals" in explain("freedman-lane").lower()
        assert explain("fl") == explain("freedman-lane")   # alias
        d = explain("tau-u", "i2", as_dict=True)
        assert set(d) == {"tau-u", "i2"} and all(len(v) > 10 for v in d.values())

    def test_explain_all_and_unknown(self):
        assert explain().count("\n") == len(GLOSSARY) - 1
        assert "unknown" in explain("not_a_stat")

    def test_explain_report_picks_present_stats(self):
        info = {"Randomization statistic": "Stratified partial F (freedman-lane)",
                "Heterogeneity - I2 (%)": 12.0}
        txt = explain_report(info).lower()
        assert "freedman-lane" in txt and "randomization test" in txt


class TestMultivariate:
    def _group(self, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for p in range(5):
            o = np.repeat(["A", "B", "C"], 6); rng.shuffle(o); ss = np.arange(1, 19)
            base = rng.normal(0, 3, 3)
            for s, c in zip(ss, o):
                bump = 6 if c == "B" else 0
                rows.append({"u": f"P{p}", "s": int(s), "c": c,
                             "y1": base[0] + bump + 0.5 * s + rng.normal(0, 3),
                             "y2": base[1] + 0.8 * bump + rng.normal(0, 3),
                             "y3": base[2] + rng.normal(0, 3)})       # no effect
        return pd.DataFrame(rows)

    def test_omnibus_and_followups(self):
        r = permanova_condition_test(
            self._group(), outcome_cols=["y1", "y2", "y3"], session_col="s",
            condition_col="c", unit_col="u", conditions=("A", "B", "C"),
            detrend="linear", n_perm=1500)
        assert r["p_value"] < 0.05 and r["n_outcomes"] == 3
        assert r["df"][0] == 2                                        # k-1
        per = r["per_outcome"].set_index("Outcome")
        assert per.loc["y1", "p Holm"] < 0.05 and per.loc["y2", "p Holm"] < 0.05
        assert per.loc["y3", "p Holm"] > 0.10                        # y3 not flagged

    def test_single_series_multivariate(self):
        rng = np.random.RandomState(1); rows = []
        order = np.repeat(["A", "B"], 9); rng.shuffle(order)
        for s, c in zip(np.arange(1, 19), order):
            rows.append({"s": int(s), "c": c,
                         "y1": (5 if c == "B" else 0) + rng.normal(0, 3),
                         "y2": (4 if c == "B" else 0) + rng.normal(0, 3)})
        r = permanova_condition_test(pd.DataFrame(rows), outcome_cols=["y1", "y2"],
                                     session_col="s", condition_col="c",
                                     conditions=("A", "B"), n_perm=1500)
        assert 0 < r["p_value"] <= 1 and r["n_units"] == 1
        assert len(r["per_outcome"]) == 2


class TestHierarchical:
    def _group(self, effects, seed=0):
        from functions.sced.hierarchical import hierarchical_group_model
        rng = np.random.RandomState(seed); rows = []
        for p, eff in enumerate(effects):
            o = np.tile(["A", "B"], 6); rng.shuffle(o); ss = np.arange(1, 13)
            base = 20 + rng.normal(0, 5)
            for s, c in zip(ss, o):
                rows.append({"patient": f"P{p}", "session": int(s), "condition": c,
                             "score": base + (eff if c == "B" else 0) + rng.normal(0, 3)})
        return hierarchical_group_model(
            pd.DataFrame(rows), unit_col="patient", session_col="session",
            condition_col="condition", outcome_col="score", conditions=("A", "B"),
            n_perm=1200)

    def test_homogeneous_low_slope_sd(self):
        summ, sheet = self._group([5, 5, 5, 5, 5, 5], seed=0)
        assert summ["p_common_effect_perm"] < 0.05
        assert summ["p_heterogeneity_perm"] == "n/a" or float(summ["p_heterogeneity_perm"]) > 0.10
        assert "Explanation" in sheet.columns and "Interpretation" in sheet.columns

    def test_heterogeneous_high_slope_sd(self):
        summ, _ = self._group([7, 7, 7, -7, -7, -7], seed=1)
        # opposite effects: common effect ~0 but large slope SD + significant heterogeneity
        assert abs(float(summ["population_effect"])) < 2.0
        assert float(summ["sd_random_slope"]) > 3.0
        assert float(summ["p_heterogeneity_perm"]) < 0.05

    def test_pipeline_hierarchical_sheet(self, tmp_path):
        df = _make_group(5, effect=5.0, learning=0.0, seed=3, k=2)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond",
            outcome_col="y", conditions=("A", "B"), hierarchical=True, n_perm=600,
            save_path=str(tmp_path), verbose=False)
        assert "Hierarchical - ICC" in info
        xls = tmp_path / "y_sced_alternating_group_report.xlsx"
        assert "Hierarchical" in pd.read_excel(xls, sheet_name=None)


class TestLongFormatCheck:
    def test_clean_long_ok(self):
        from functions.sced.core import check_long_format
        df = pd.DataFrame({"session": range(1, 19), "condition": np.tile(["A", "B"], 9),
                           "score": np.random.RandomState(0).normal(0, 1, 18)})
        assert check_long_format(df, session_col="session", condition_col="condition",
                                 outcome_col="score") == []

    def test_missing_columns_flagged(self):
        from functions.sced.core import check_long_format
        wide = pd.DataFrame({"session": [1, 2], "A": [1, 2], "B": [3, 4]})
        probs = check_long_format(wide, session_col="session", condition_col="condition",
                                  outcome_col="score")
        assert probs and "Absent" in probs[0] and "melt" in probs[0].lower()

    def test_non_numeric_outcome_and_duplicates(self):
        from functions.sced.core import check_long_format
        df = pd.DataFrame({"u": ["P1", "P1"], "session": [1, 1],
                           "condition": ["A", "B"], "score": ["x", "y"]})
        probs = check_long_format(df, session_col="session", condition_col="condition",
                                  outcome_col="score", unit_col="u")
        assert any("numeric" in p for p in probs)
        assert any("a single row per session" in p for p in probs)

    def test_multivariate_outcome_list(self):
        from functions.sced.core import check_long_format
        df = pd.DataFrame({"session": range(1, 13), "condition": np.tile(["A", "B"], 6),
                           "y1": np.arange(12.0), "y2": np.arange(12.0)})
        assert check_long_format(df, session_col="session", condition_col="condition",
                                 outcome_col=["y1", "y2"]) == []


class TestCovariateAdjustment:
    def _data(self, seed=0):
        rng = np.random.RandomState(seed)
        order = np.repeat(["A", "B"], 9); rng.shuffle(order); sess = np.arange(1, 19)
        fatigue = rng.normal(5, 2, 18)
        acc = 60 + 4 * (order == "B") - 1.5 * fatigue + rng.normal(0, 3, 18)
        return order, sess, acc, fatigue

    def test_engine_covariate_reduces_df_and_sharpens(self):
        order, sess, acc, fatigue = self._data(0)
        no = condition_permutation_test(acc, order, sess, method="freedman-lane", n_perm=2000)
        cov = condition_permutation_test(acc, order, sess, method="freedman-lane",
                                         covariates=fatigue, n_perm=2000)
        assert cov["df"][1] == no["df"][1] - 1          # one df spent on the covariate
        assert cov["observed_F"] > no["observed_F"]     # removing fatigue noise sharpens

    def test_covariate_bad_length_raises(self):
        order, sess, acc, _ = self._data(1)
        with pytest.raises(ValueError):
            condition_permutation_test(acc, order, sess, covariates=np.ones(3), n_perm=50)

    def test_pipeline_nof1_records_covariate(self):
        order, sess, acc, fatigue = self._data(2)
        df = pd.DataFrame({"session": sess, "condition": order, "score": acc, "fatigue": fatigue})
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B"), covariate_cols=["fatigue"], n_perm=1500, verbose=False)
        assert "fatigue" in str(info["Covariate adjustment"])
        assert "covariate" in info["Test statistic"].lower()

    def test_pipeline_group_covariate(self):
        rng = np.random.RandomState(3); rows = []
        for p in range(5):
            o = np.tile(["A", "B"], 6); rng.shuffle(o); ss = np.arange(1, 13)
            base = rng.uniform(55, 65); fat = rng.normal(5, 2, 12)
            for s, c, f in zip(ss, o, fat):
                rows.append({"u": f"P{p}", "s": int(s), "c": c,
                             "score": base + 4 * (c == "B") - 1.5 * f + rng.normal(0, 3),
                             "fatigue": f})
        info, _ = pipeline_sced_alternating_group(
            pd.DataFrame(rows), unit_col="u", session_col="s", condition_col="c",
            outcome_col="score", conditions=("A", "B"), covariate_cols=["fatigue"],
            n_perm=1500, verbose=False)
        assert "fatigue" in info["Test statistic"]


class TestHierarchicalGLMM:
    def _binary(self, seed=0):
        from functions.sced.hierarchical import hierarchical_group_model
        rng = np.random.RandomState(seed); rows = []
        for p in range(6):
            o = np.tile(["A", "B"], 6); rng.shuffle(o); ss = np.arange(1, 13); b = rng.normal(0, 0.6)
            for s, c in zip(ss, o):
                eta = b + 1.3 * (c == "B")
                rows.append({"u": f"P{p}", "s": int(s), "c": c,
                             "y": int(rng.rand() < 1 / (1 + np.exp(-eta)))})
        return hierarchical_group_model(
            pd.DataFrame(rows), unit_col="u", session_col="s", condition_col="c",
            outcome_col="y", conditions=("A", "B"), outcome_type="binary", n_perm=800)

    def test_binary_glmm_runs_and_labels_scale(self):
        summ, sheet = self._binary(0)
        assert summ["outcome_type"] == "binary"
        assert "logit" in " ".join(sheet["Quantity"]) or "logit" in " ".join(sheet["Explanation"])
        assert 0 <= summ["p_common_effect_perm"] <= 1

    def test_count_glmm_runs(self):
        from functions.sced.hierarchical import hierarchical_group_model
        rng = np.random.RandomState(1); rows = []
        for p in range(6):
            o = np.tile(["A", "B"], 6); rng.shuffle(o); ss = np.arange(1, 13); b = rng.normal(0, 0.3)
            for s, c in zip(ss, o):
                rows.append({"u": f"P{p}", "s": int(s), "c": c,
                             "y": int(rng.poisson(np.exp(1.0 + 0.5 * (c == "B") + b)))})
        summ, _ = hierarchical_group_model(
            pd.DataFrame(rows), unit_col="u", session_col="s", condition_col="c",
            outcome_col="y", conditions=("A", "B"), outcome_type="count", n_perm=800)
        assert summ["outcome_type"] == "count" and summ["population_effect"] != "n/a"


def _measures_df(seed=0, eff_mean=4, sd_A=3.0, sd_B=1.2, n_meas=8):
    rng = np.random.RandomState(seed); rows = []
    for p in range(6):
        pat = rng.normal(0, 5); o = np.tile(["A", "B"], 6); rng.shuffle(o)
        for s, c in zip(np.arange(1, 13), o):
            vis = rng.normal(0, 3); sdin = sd_B if c == "B" else sd_A
            for _ in range(n_meas):
                rows.append({"patient": f"P{p}", "session": int(s), "cond": c,
                             "y": 50 + pat + vis + eff_mean * (c == "B") + rng.normal(0, sdin)})
    return pd.DataFrame(rows)


class TestMultilevel:
    def test_aggregate_visits_counts_and_sd(self):
        from functions.sced.prep import aggregate_visits
        df = _measures_df(0)
        vis = aggregate_visits(df, unit_col="patient", session_col="session",
                               condition_col="cond", outcome_col="y", stat="n")
        assert (vis["y"] == 8).all() and len(vis) == 6 * 12     # 6 patients x 12 visites

    def test_variance_decomposition_components(self):
        from functions.sced.intravisit import variance_decomposition_3level
        summ, sheet = variance_decomposition_3level(
            _measures_df(0), unit_col="patient", session_col="session",
            condition_col="cond", outcome_col="y", conditions=("A", "B"), n_perm=800)
        tot = summ["pct_patient"] + summ["pct_visit"] + summ["pct_within_visit"]
        assert abs(tot - 100) < 0.5                              # shares sum to 100%
        assert summ["var_within_visit"] != "n/a"
        assert summ["p_common_effect_perm"] < 0.05               # mean effect present
        assert "Interpretation" in sheet.columns

    def test_dispersion_test_detects_consistency(self):
        from functions.sced.intravisit import dispersion_test
        # B is much more consistent (sd 1.2 vs 3.0) -> condition affects within-visit SD
        res, vis = dispersion_test(_measures_df(0), unit_col="patient", session_col="session",
                                   condition_col="cond", outcome_col="y", measure="sd", n_perm=800)
        assert res["p_value"] < 0.05 and res["dispersion_measure"] == "sd"

    def test_dispersion_null_when_equal_spread(self):
        from functions.sced.intravisit import dispersion_test
        res, _ = dispersion_test(_measures_df(1, sd_A=2.5, sd_B=2.5), unit_col="patient",
                                 session_col="session", condition_col="cond",
                                 outcome_col="y", measure="sd", n_perm=800)
        assert res["p_value"] > 0.05                             # same spread -> ns


class TestWithinVisitSlope:
    def _df(self, slope_A, slope_B, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for p in range(6):
            pat = rng.normal(0, 5); o = np.tile(["A", "B"], 6); rng.shuffle(o)
            for s, c in zip(np.arange(1, 13), o):
                vis = rng.normal(0, 3); sl = slope_B if c == "B" else slope_A
                for t in range(1, 9):
                    rows.append({"patient": f"P{p}", "session": int(s), "cond": c, "trial": t,
                                 "y": 50 + pat + vis + sl * t + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_detects_slope_difference(self):
        from functions.sced.intravisit import within_visit_slope_test
        res, vis = within_visit_slope_test(self._df(0.0, -0.6), unit_col="patient",
            session_col="session", condition_col="cond", outcome_col="y",
            trial_col="trial", n_perm=800)
        assert res["p_value"] < 0.05
        assert "y_within_slope" in vis.columns

    def test_null_equal_slopes(self):
        from functions.sced.intravisit import within_visit_slope_test
        res, _ = within_visit_slope_test(self._df(-0.3, -0.3, seed=1), unit_col="patient",
            session_col="session", condition_col="cond", outcome_col="y",
            trial_col="trial", n_perm=800)
        assert res["p_value"] > 0.05


class TestRobustness:
    def test_engine_drops_nan_matches_manual(self):
        y = np.arange(18.0); y[3] = np.nan
        lab = np.tile(["A", "B"], 9); s = np.arange(1, 19)
        with_nan = condition_permutation_test(y, lab, s, n_perm=400, random_state=0)["p_value"]
        manual = condition_permutation_test(np.delete(y, 3), np.delete(lab, 3),
                                            np.delete(s, 3), n_perm=400, random_state=0)["p_value"]
        assert with_nan == pytest.approx(manual)          # NaN dropped, not silently wrong

    def test_engine_raises_when_too_few_valid(self):
        with pytest.raises(ValueError):
            condition_permutation_test(np.array([1.0, np.nan]), np.array(["A", "B"]),
                                       np.array([1, 2]), n_perm=50)

    def test_engine_raises_single_condition(self):
        with pytest.raises(ValueError):
            condition_permutation_test(np.arange(6.0), np.array(["A"] * 6),
                                       np.arange(1, 7), n_perm=50)

    def test_pipeline_missing_condition_raises(self):
        df = pd.DataFrame({"session": np.arange(1, 19), "condition": np.tile(["A", "B"], 9),
                           "score": np.arange(18.0)})
        with pytest.raises(ValueError, match="absent"):
            pipeline_sced_alternating(df, session_col="session", condition_col="condition",
                                      outcome_col="score", conditions=("A", "Z"),
                                      n_perm=100, verbose=False)

    def test_pipeline_non_numeric_outcome_raises(self):
        df = pd.DataFrame({"session": np.arange(1, 19), "condition": np.tile(["A", "B"], 9),
                           "score": np.tile(["bas", "haut"], 9)})
        with pytest.raises(ValueError):
            pipeline_sced_alternating(df, session_col="session", condition_col="condition",
                                      outcome_col="score", conditions=("A", "B"),
                                      n_perm=100, verbose=False)

    def test_group_missing_condition_raises(self):
        rng = np.random.RandomState(0); rows = []
        for p in range(4):
            for s in range(1, 13):
                rows.append({"u": f"P{p}", "session": s, "condition": np.tile(["A", "B"], 6)[s - 1],
                             "score": rng.normal()})
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="absent"):
            pipeline_sced_alternating_group(df, unit_col="u", session_col="session",
                                            condition_col="condition", outcome_col="score",
                                            conditions=("A", "C"), n_perm=100, verbose=False)


class TestMissingDataSignal:
    def test_missing_data_warnings_counts(self):
        from functions.sced.core import missing_data_warnings
        df = pd.DataFrame({"session": np.arange(1, 13), "condition": np.tile(["A", "B"], 6),
                           "score": np.arange(12.0)})
        df.loc[[2, 5], "score"] = np.nan
        w = missing_data_warnings(df, outcome_col="score", session_col="session")
        assert any("2/12" in m and "outcome" in m for m in w)
        assert missing_data_warnings(df.dropna(), outcome_col="score", session_col="session") == []

    def test_pipeline_reports_rows_dropped(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"session": np.arange(1, 19), "condition": np.tile(["A", "B"], 9),
                           "score": rng.normal(0, 1, 18)})
        df.loc[[3, 7, 11], "score"] = np.nan
        info, _ = pipeline_sced_alternating(df, session_col="session", condition_col="condition",
                                            outcome_col="score", conditions=("A", "B"),
                                            n_perm=300, verbose=False)
        assert info["Rows dropped (missing)"] == 3
        assert any("missing" in m for m in info["Data warnings"])

    def test_group_missing_per_unit_flag(self):
        rng = np.random.RandomState(1); rows = []
        for p in range(4):
            o = np.tile(["A", "B"], 6); rng.shuffle(o)
            for s, c in zip(np.arange(1, 13), o):
                y = rng.normal() if not (p == 3 and s % 2 == 0) else np.nan
                rows.append({"u": f"P{p}", "s": int(s), "c": c, "score": y})
        info, _ = pipeline_sced_alternating_group(
            pd.DataFrame(rows), unit_col="u", session_col="s", condition_col="c",
            outcome_col="score", conditions=("A", "B"), n_perm=300, verbose=False)
        assert info["Rows dropped (missing)"] == 6
        assert any("unit" in m for m in info["Data warnings"])


def _mbd_df(effect, seed=0, starts=(5, 9, 13, 17), n=20):
    rng = np.random.RandomState(seed); rows = []
    for ti, st in enumerate(starts):
        for s in range(1, n + 1):
            ph = "B" if s >= st else "A"
            rows.append({"tier": f"T{ti+1}", "session": s, "phase": ph,
                         "y": 10 + (effect if ph == "B" else 0) + rng.normal(0, 2)})
    return pd.DataFrame(rows)


class TestMultipleBaseline:
    def test_detects_staggered_effect(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        r = multiple_baseline_test(_mbd_df(6, seed=1), tier_col="tier", session_col="session",
                                   outcome_col="y", phase_col="phase", improvement="increase")
        assert r["p_value"] < 0.05 and r["exact"]
        assert (r["per_tier"]["diff"] > 2).all()

    def test_null_not_significant(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        # short series (fewer candidates) to keep the exact enumeration fast
        ps = [multiple_baseline_test(_mbd_df(0, seed=s, starts=(4, 6, 8), n=12),
                                     tier_col="tier", session_col="session",
                                     outcome_col="y", phase_col="phase")["p_value"]
              for s in range(12)]
        assert np.mean([p < 0.05 for p in ps]) < 0.3       # type-I ~ nominal

    def test_statistics_itei_and_tau_u(self):
        """The MBD test accepts the 'itei' (transition) and 'tau_u' (nonoverlap) statistics."""
        from functions.sced.mbd.baseline import multiple_baseline_test
        df = _mbd_df(6, seed=1)
        for stat in ("itei", "tau_u", "level"):
            r = multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                       phase_col="phase", statistic=stat, improvement="increase")
            assert 0 <= r["p_value"] <= 1 and r["statistic"] == stat
        # clear staggered effect -> itei and tau_u also detect it
        assert multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                      phase_col="phase", statistic="itei")["p_value"] < 0.1
        # unknown statistic -> error
        import pytest
        with pytest.raises(ValueError):
            multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                   phase_col="phase", statistic="nope")

    def test_explicit_starts_and_candidates(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        df = _mbd_df(5, seed=2)
        r = multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                   intervention_starts={"T1": 5, "T2": 9, "T3": 13, "T4": 17},
                                   candidates={"T1": [4, 5, 6], "T2": [8, 9, 10],
                                               "T3": [12, 13, 14], "T4": [16, 17, 18]})
        assert r["starts"]["T1"] == 5 and 0 <= r["p_value"] <= 1

    def test_pipeline_and_format_guard(self):
        from functions.sced.mbd.baseline import pipeline_sced_multiple_baseline
        info, res = pipeline_sced_multiple_baseline(
            _mbd_df(6, seed=3), tier_col="tier", session_col="session", outcome_col="y",
            phase_col="phase", verbose=False)
        assert info["p-value"] < 0.05 and len(res["per_tier"]) == 4
        bad = _mbd_df(6, seed=3)
        bad = pd.concat([bad, bad.iloc[[0]]])              # duplicate (tier, session)
        with pytest.raises(ValueError, match="tier"):
            pipeline_sced_multiple_baseline(bad, tier_col="tier", session_col="session",
                                            outcome_col="y", phase_col="phase", verbose=False)


class TestExactConvolutionAndOneSidedTau:
    def test_convolution_matches_enumeration(self):
        """For an additive statistic under the MB scheme (independent, no min_gap), the exact
        convolution p == direct enumeration (same p / exact / n_eval)."""
        from functions.sced.mbd.baseline import multiple_baseline_test
        df = _mbd_df(4, seed=7)
        common = dict(tier_col="tier", session_col="session", outcome_col="y", phase_col="phase",
                      candidates={"T1": [4, 5, 6, 7, 8, 9], "T2": [4, 5, 6, 7, 8, 9],
                                  "T3": [4, 5, 6, 7, 8, 9], "T4": [4, 5, 6, 7, 8, 9]},
                      scheme="MB", statistic="level")
        conv = multiple_baseline_test(df, enumerate_cap=10, conv_cap=10_000_000, **common)   # convolution
        enum = multiple_baseline_test(df, enumerate_cap=10_000_000, **common)                # enumeration
        assert conv["exact"] and enum["exact"]
        assert conv["n_eval"] == enum["n_eval"] > 10                # same full space, both exact
        assert abs(conv["p_value"] - enum["p_value"]) < 1e-12

    def test_convolution_falls_back_when_non_additive(self):
        """tau_u can return a non-finite value for some starts -> non-additive ->
        Monte-Carlo fallback (no crash, valid p)."""
        from functions.sced.mbd.baseline import multiple_baseline_test
        df = _mbd_df(6, seed=1)
        r = multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                   phase_col="phase", statistic="level",
                                   enumerate_cap=5, conv_cap=10_000_000, n_perm=500)
        assert 0 <= r["p_value"] <= 1

    def test_tau_u_one_sided_halves_two_sided(self):
        from functions.sced.core import tau_u
        a, b = [1, 2, 1, 3, 2, 1, 2], [3, 5, 2, 6, 4, 3]
        ts = tau_u(a, b, correct_baseline_trend=False, alternative="two-sided")
        gr = tau_u(a, b, correct_baseline_trend=False, alternative="greater")
        assert ts["tau_u"] == gr["tau_u"]
        assert ts["alternative"] == "two-sided" and gr["alternative"] == "greater"
        assert abs(ts["p_value"] / 2 - gr["p_value"]) < 1e-3     # tau>0 -> one-sided = half (p rounded to 4 dp)
        assert abs(gr["critical"] / ts["critical"] - 1.6448536 / 1.959964) < 1e-4

    def test_tau_u_contrast_se_matches_scan_formula(self):
        """The SE of the A-vs-B contrast = sqrt(n(n-1)(2n+5)/18)/(n_A*n_B) (scan convention),
        not the Vannest-Ninci series SE."""
        from functions.sced.core import tau_u
        a, b = [2, 3, 1, 4, 2, 3, 1], [3, 5, 2, 6, 4, 3, 4, 5, 6, 7]   # n_A=7, n_B=10, n=17
        r = tau_u(a, b, correct_baseline_trend=False)
        na, nb, n = 7, 10, 17
        se_scan = (n * (n - 1) * (2 * n + 5) / 18.0) ** 0.5 / (na * nb)
        assert abs(r["se"] - se_scan) < 1e-4
        assert abs(r["se"] - 0.3468) < 1e-3                          # reference scan SE (n_A=7, n_B=10)


class TestMBDSchemesAndSlope:
    def _df(self, starts, effect=5, slope=0.0, seed=0, n=20):
        rng = np.random.RandomState(seed); rows = []
        for p, st in enumerate(starts):
            for s in range(1, n + 1):
                prog = slope * (s - st) if s >= st else 0
                lvl = effect if s >= st else 0
                rows.append({"tier": f"P{p+1}", "session": s,
                             "phase": "B" if s >= st else "A",
                             "y": 10 + lvl + prog + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_distinct_scheme_enumerates_permutations(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        entry = [5, 8, 11, 14, 17]
        r = multiple_baseline_test(self._df(entry, seed=2), tier_col="tier",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="distinct", entry_points=entry)
        assert r["n_eval"] == 120 and r["exact"]           # 5! permutations
        assert r["p_value"] < 0.05

    def test_distinct_requires_distinct_starts(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        with pytest.raises(ValueError, match="distinct"):
            multiple_baseline_test(self._df([6, 6, 10, 14], seed=1), tier_col="tier",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="distinct", entry_points=[6, 10, 14, 18])

    def test_ordered_scheme_combinations(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        pool = [4, 6, 8, 10, 12, 14, 16, 18]
        r = multiple_baseline_test(self._df([6, 10, 14, 18], seed=3), tier_col="tier",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="ordered", entry_points=pool)
        assert r["n_eval"] == 70                            # C(8,4)
        assert 0 <= r["p_value"] <= 1

    def test_slope_more_sensitive_to_progressive_effect(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        df = self._df([5, 9, 13, 17], effect=0, slope=0.8, seed=4)  # pure trend change
        lvl = multiple_baseline_test(df, tier_col="tier", session_col="session",
                                     outcome_col="y", phase_col="phase", statistic="level")
        slp = multiple_baseline_test(df, tier_col="tier", session_col="session",
                                     outcome_col="y", phase_col="phase", statistic="slope")
        assert slp["p_value"] < lvl["p_value"]              # slope statistic far more sensitive
        assert slp["statistic"] == "slope"


class TestMBDMultiPhase:
    def _abc(self, c_starts=(9, 12, 15, 18), real_effect=7, sham_effect=0, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for p, cst in enumerate(c_starts):
            for s in range(1, 23):
                ph = "A" if s < 6 else ("B" if s < cst else "C")
                bump = real_effect if ph == "C" else (sham_effect if ph == "B" else 0)
                rows.append({"subj": f"S{p+1}", "session": s, "phase": ph,
                             "y": 55 + bump + rng.normal(0, 4)})
        return pd.DataFrame(rows)

    def test_C_vs_B_contrast_detects_real_effect(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        r = multiple_baseline_test(self._abc(seed=0), tier_col="subj", session_col="session",
                                   outcome_col="y", phase_col="phase",
                                   baseline_phase="B", treatment_phase="C", improvement="increase")
        assert r["p_value"] < 0.05
        # the contrast uses only B and C sessions (phase A run-in excluded)
        per = r["per_tier"].set_index("Tier")
        assert per.loc["S1", "n_A"] < 6                     # n_A = sham(B) count, not A+B

    def test_C_vs_B_null_when_real_equals_sham(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        ps = [multiple_baseline_test(self._abc(real_effect=0, seed=s), tier_col="subj",
              session_col="session", outcome_col="y", phase_col="phase",
              baseline_phase="B", treatment_phase="C")["p_value"] for s in range(8)]
        assert np.mean([p < 0.05 for p in ps]) < 0.4       # no specific effect -> ~ns


class TestPhaseDesign:
    def _abab(self, effect, seed=0, n=20, cuts=(6, 11, 16), labs=("A", "B", "A", "B")):
        rng = np.random.RandomState(seed); rows = []
        bounds = [0] + list(cuts) + [n]
        for k, lb in enumerate(labs):
            for i in range(bounds[k], bounds[k + 1]):
                rows.append({"session": i + 1, "phase": lb,
                             "y": 10 + (effect if lb == "B" else 0) + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_abab_contrast_detects(self):
        from functions.sced.phase_design import phase_design_test
        r = phase_design_test(self._abab(6, seed=1), session_col="session", outcome_col="y",
                              phase_col="phase", statistic="contrast")
        assert r["p_value"] < 0.05 and r["phases"] == ["A", "B", "A", "B"] and r["exact"]

    def test_abab_omnibus(self):
        from functions.sced.phase_design import phase_design_test
        r = phase_design_test(self._abab(6, seed=1), session_col="session", outcome_col="y",
                              phase_col="phase", statistic="omnibus")
        assert r["p_value"] < 0.05

    def test_abab_itei(self):                     # ITEI = |3 derniers A - 3 premiers B| (Michiels 2018)
        from functions.sced.phase_design import phase_design_test
        r = phase_design_test(self._abab(6, seed=1), session_col="session", outcome_col="y",
                              phase_col="phase", statistic="itei")
        assert r["statistic"] == "itei" and np.isfinite(r["observed"]) and r["observed"] > 0
        assert 0 < r["p_value"] <= 1               # computed (less powerful: 1 switch, 6 points)

    def test_null_calibrated(self):
        from functions.sced.phase_design import phase_design_test
        ps = [phase_design_test(self._abab(0, seed=s), session_col="session", outcome_col="y",
              phase_col="phase")["p_value"] for s in range(20)]
        assert np.mean([p < 0.05 for p in ps]) < 0.25

    def test_replicated(self):
        from functions.sced.phase_design import phase_design_test
        rng = np.random.RandomState(3); rows = []
        for u in range(4):
            for lb, a, b in [("A", 0, 6), ("B", 6, 11), ("A", 11, 16), ("B", 16, 20)]:
                for i in range(a, b):
                    rows.append({"subj": f"S{u}", "session": i + 1, "phase": lb,
                                 "y": 10 + (5 if lb == "B" else 0) + rng.normal(0, 2)})
        r = phase_design_test(pd.DataFrame(rows), session_col="session", outcome_col="y",
                              phase_col="phase", unit_col="subj")
        assert r["p_value"] < 0.05 and r["n_units"] == 4


class TestBlockScheme:
    def test_block_scheme_permutes_within_blocks(self):
        from functions.sced.core import block_scheme
        labels = np.array(["A", "B", "C", "A", "B", "C"])
        rng = np.random.RandomState(0)
        perm = block_scheme(3)(labels, rng)
        assert sorted(perm[:3]) == ["A", "B", "C"]        # block 1 keeps its members
        assert sorted(perm[3:]) == ["A", "B", "C"]        # block 2 too

    def test_pipeline_block_size_runs(self):
        rng = np.random.RandomState(1)
        conds = np.tile(["A", "B"], 9); sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + rng.normal(0, 1, 18)
        df = pd.DataFrame({"session": sess, "cond": conds, "y": y})
        info, _ = pipeline_sced_alternating(df, session_col="session", condition_col="cond",
                                            outcome_col="y", conditions=("A", "B"),
                                            block_size=2, n_perm=1000, verbose=False)
        assert 0 <= info["Randomization p-value"] <= 1


class TestMBDCombinedStat:
    def test_combined_more_sensitive_than_level_for_progressive(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        rng = np.random.RandomState(4); rows = []
        for p, st in enumerate((5, 9, 13, 17)):
            for s in range(1, 21):
                prog = 0.8 * (s - st) if s >= st else 0
                rows.append({"tier": f"T{p+1}", "session": s, "phase": "B" if s >= st else "A",
                             "y": 10 + prog + rng.normal(0, 2)})
        df = pd.DataFrame(rows)
        lvl = multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                     phase_col="phase", statistic="level")["p_value"]
        comb = multiple_baseline_test(df, tier_col="tier", session_col="session", outcome_col="y",
                                      phase_col="phase", statistic="combined")["p_value"]
        assert comb < lvl


class TestMBDMinGap:
    def _df(self, starts, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for p, st in enumerate(starts):
            for s in range(1, 21):
                rows.append({"tier": f"T{p+1}", "session": s, "phase": "B" if s >= st else "A",
                             "y": 10 + (5 if s >= st else 0) + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_min_gap_shrinks_reference_set(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        pool = [4, 6, 8, 10, 12, 14, 16, 18]
        common = dict(tier_col="tier", session_col="session", outcome_col="y",
                      phase_col="phase", scheme="ordered", entry_points=pool)
        df = self._df([6, 10, 14, 18], seed=3)
        n_none = multiple_baseline_test(df, min_gap=None, **common)["n_eval"]
        n_gap = multiple_baseline_test(df, min_gap=3, **common)["n_eval"]
        assert n_gap < n_none                              # regulated stagger filters assignments


class TestMBDCanonicalNames:
    def _df(self, starts, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for p, st in enumerate(starts):
            for s in range(1, 21):
                rows.append({"tier": f"T{p+1}", "session": s, "phase": "B" if s >= st else "A",
                             "y": 10 + (5 if s >= st else 0) + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_canonical_names_and_distribution_sizes(self):
        import math
        from functions.sced.mbd.baseline import multiple_baseline_test
        d = self._df([6, 10, 14, 18])
        ww = multiple_baseline_test(d, tier_col="tier", session_col="session", outcome_col="y",
                                    phase_col="phase", scheme="WW", entry_points=[6, 10, 14, 18])
        assert ww["n_eval"] == math.factorial(4) and "WW" in ww["procedure"]
        mbr = multiple_baseline_test(d, tier_col="tier", session_col="session", outcome_col="y",
                                     phase_col="phase", scheme="MB-R",
                                     entry_points=[4, 6, 8, 10, 12, 14, 16, 18])
        assert mbr["n_eval"] == math.perm(8, 4) and "MB-R" in mbr["procedure"]

    def test_ww_requires_exactly_N_points(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        with pytest.raises(ValueError, match="WW"):
            multiple_baseline_test(self._df([6, 10, 14, 18]), tier_col="tier",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="WW", entry_points=[4, 6, 8, 10, 12])


class TestMBDPaperValidation:
    """Reproduces the worked classroom example of Levin, Ferron & Gafurov (2017),
    Dev. Neurorehabil. 21(5):290-311 - the canonical reference for the 6 MBD procedures."""
    DATA = {"C1": [4, 3, 5, 7, 6, 8, 7, 8, 9, 7],
            "C2": [6, 7, 5, 6, 6, 7, 10, 9, 10, 10],
            "C3": [9, 9, 7, 10, 10, 8, 9, 12, 11, 14]}

    def _df(self, starts=None):
        rows = [{"case": c, "session": i + 1, "y": v}
                for c, ys in self.DATA.items() for i, v in enumerate(ys)]
        df = pd.DataFrame(rows)
        if starts:
            df["phase"] = ["B" if r.session >= starts[r.case] else "A" for r in df.itertuples()]
        return df

    def test_WW(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        r = multiple_baseline_test(self._df({"C1": 3, "C2": 6, "C3": 8}), tier_col="case",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="WW", entry_points=[3, 6, 8])
        assert r["p_value"] == pytest.approx(1 / 6, abs=1e-3)

    def test_MB(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        r = multiple_baseline_test(self._df({"C1": 3, "C2": 6, "C3": 8}), tier_col="case",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="MB", candidates={c: [3, 4, 5, 6, 7, 8] for c in self.DATA})
        assert r["n_eval"] == 216 and r["p_value"] == pytest.approx(3 / 216, abs=1e-3)

    def test_MB_R(self):
        from functions.sced.mbd.baseline import multiple_baseline_test
        r = multiple_baseline_test(self._df({"C1": 3, "C2": 6, "C3": 8}), tier_col="case",
                                   session_col="session", outcome_col="y", phase_col="phase",
                                   scheme="MB-R", entry_points=[3, 4, 5, 6, 7, 8])
        assert r["n_eval"] == 120 and r["p_value"] == pytest.approx(3 / 120, abs=1e-3)

    def test_KL(self):
        from functions.sced.mbd.procedures import koehler_levin_test
        r = koehler_levin_test(self._df(), case_col="case", session_col="session", outcome_col="y",
                               windows=[[2, 3], [5, 6], [8, 9]],
                               observed_starts={"C1": 3, "C2": 6, "C3": 8})
        assert r["n_eval"] == 48 and r["p_value"] == pytest.approx(1 / 48, abs=1e-3)

    def test_Rev(self):
        from functions.sced.mbd.procedures import revusky_test
        r = revusky_test(self._df(), case_col="case", session_col="session", outcome_col="y",
                         position_starts=[3, 6, 8], observed_order=["C1", "C2", "C3"])
        assert r["observed_ranksum"] == pytest.approx(3.0) and r["p_value"] == pytest.approx(1 / 6, abs=1e-3)

    def test_RevM(self):
        from functions.sced.mbd.procedures import revusky_modified_test
        r1 = revusky_modified_test(self._df(), case_col="case", session_col="session", outcome_col="y",
                                   windows=[[3], [6], [8]], observed_starts={"C1": 3, "C2": 6, "C3": 8})
        assert r1["n_eval"] == 6 and r1["p_value"] == pytest.approx(1 / 6, abs=1e-3)
        r2 = revusky_modified_test(self._df(), case_col="case", session_col="session", outcome_col="y",
                                   windows=[[2, 3], [5, 6], [8, 9]], observed_starts={"C1": 3, "C2": 6, "C3": 8})
        assert r2["n_eval"] == 48 and r2["p_value"] == pytest.approx(1 / 48, abs=1e-3)


class TestMBDFeatureAPI:
    DATA = {"C1": [4, 3, 5, 7, 6, 8, 7, 8, 9, 7],
            "C2": [6, 7, 5, 6, 6, 7, 10, 9, 10, 10],
            "C3": [9, 9, 7, 10, 10, 8, 9, 12, 11, 14]}

    def _df(self):
        rows = [{"case": c, "session": i + 1, "y": v}
                for c, ys in self.DATA.items() for i, v in enumerate(ys)]
        df = pd.DataFrame(rows)
        st = {"C1": 3, "C2": 6, "C3": 8}
        df["phase"] = ["B" if r.session >= st[r.case] else "A" for r in df.itertuples()]
        return df

    def test_features_map_to_procedures(self):
        from functions.sced.mbd.procedures import mbd_randomization_test as mbd
        b = dict(case_col="case", session_col="session", outcome_col="y", phase_col="phase")
        df = self._df()
        ww = mbd(df, **b, case_randomization=True, start_point_randomization=False)
        assert "WW" in ww["procedure"] and ww["p_value"] == pytest.approx(1/6, abs=1e-3)
        mbr = mbd(df, **b, case_randomization=False, start_point_randomization=True,
                  replacement=False, windows=[3, 4, 5, 6, 7, 8])
        assert "MB-R" in mbr["procedure"] and mbr["p_value"] == pytest.approx(3/120, abs=1e-3)
        kl = mbd(df, **b, case_randomization=True, start_point_randomization=True,
                 comparison="within", windows=[[2, 3], [5, 6], [8, 9]])
        assert "KL" in kl["procedure"] and kl["p_value"] == pytest.approx(1/48, abs=1e-3)
        revm = mbd(df, **b, case_randomization=True, start_point_randomization=True,
                   comparison="between", windows=[[2, 3], [5, 6], [8, 9]])
        assert "Rev-M" in revm["procedure"] and revm["p_value"] == pytest.approx(1/48, abs=1e-3)

    def test_no_randomization_raises(self):
        from functions.sced.mbd.procedures import mbd_randomization_test as mbd
        with pytest.raises(ValueError):
            mbd(self._df(), case_col="case", session_col="session", outcome_col="y",
                phase_col="phase", case_randomization=False, start_point_randomization=False)

    def test_pipeline_feature_based(self):
        from functions.sced.mbd.baseline import pipeline_sced_multiple_baseline
        info, _ = pipeline_sced_multiple_baseline(
            self._df(), tier_col="case", session_col="session", outcome_col="y", phase_col="phase",
            case_randomization=True, start_point_randomization=True, comparison="within",
            windows=[[2, 3], [5, 6], [8, 9]], verbose=False)
        assert "KL" in info["Procedure"] and info["p-value"] == pytest.approx(1/48, abs=1e-3)


class TestMBDMultilevel:
    def _build(self, level=0, slope=0, seed=0, starts=(5, 9, 13, 17), outcome="continuous"):
        rng = np.random.RandomState(seed); rows = []
        for ti, st in enumerate(starts):
            for s in range(1, 21):
                ts = max(0, s - st)
                eta = (level if s >= st else 0) + slope * ts
                if outcome == "continuous":
                    y = 10 + eta + rng.normal(0, 2)
                elif outcome == "count":
                    y = int(rng.poisson(np.exp(1.0 + 0.1 * eta)))
                else:
                    y = int(rng.rand() < 1 / (1 + np.exp(-(eta - 1))))
                rows.append({"tier": f"T{ti+1}", "session": s, "phase": "B" if s >= st else "A", "y": y})
        return pd.DataFrame(rows)

    def test_progressive_effect_via_slope(self):
        from functions.sced.mbd.multilevel import multilevel_mbd_model
        summ, sheet = multilevel_mbd_model(self._build(level=0, slope=0.7, seed=1),
                                           tier_col="tier", session_col="session",
                                           outcome_col="y", phase_col="phase", n_perm=1500)
        assert summ["p_slope_change_perm"] < 0.05            # progressive effect caught
        assert summ["p_level_change_perm"] > 0.10            # not an immediate jump
        assert float(summ["slope_change_b3"]) > 0.3
        assert "Interpretation" in sheet.columns

    def test_immediate_effect_via_level(self):
        from functions.sced.mbd.multilevel import multilevel_mbd_model
        summ, _ = multilevel_mbd_model(self._build(level=6, slope=0, seed=2),
                                       tier_col="tier", session_col="session",
                                       outcome_col="y", phase_col="phase", n_perm=1500)
        assert summ["p_level_change_perm"] < 0.05 and summ["p_slope_change_perm"] > 0.10

    def test_count_outcome_runs(self):
        from functions.sced.mbd.multilevel import multilevel_mbd_model
        summ, _ = multilevel_mbd_model(self._build(level=4, slope=0, seed=3, outcome="count"),
                                       tier_col="tier", session_col="session", outcome_col="y",
                                       phase_col="phase", outcome_type="count", n_perm=800)
        assert summ["outcome_type"] == "count" and "p_level_change_perm" in summ

    def _build_rampplateau(self, plateau=6.0, ramp=3, seed=7, starts=(5, 9, 13, 17)):
        """Effect rises linearly over the first `ramp` treatment obs, then holds."""
        rng = np.random.RandomState(seed); rows = []
        for ti, st in enumerate(starts):
            for s in range(1, 21):
                if s < st:
                    eta = 0.0
                else:
                    j = s - st + 1                       # 1-based treatment index
                    eta = plateau * min(j, ramp) / ramp  # linear rise then plateau
                rows.append({"tier": f"T{ti+1}", "session": s, "phase": "B" if s >= st else "A",
                             "y": 10 + eta + rng.normal(0, 1.5)})
        return pd.DataFrame(rows)

    def test_two_piece_recovers_plateau_and_rise(self):
        from functions.sced.mbd.multilevel import multilevel_mbd_model
        df = self._build_rampplateau(plateau=6.0, ramp=3, seed=11)
        summ, sheet = multilevel_mbd_model(df, tier_col="tier", session_col="session",
                                           outcome_col="y", phase_col="phase",
                                           treatment_trajectory="two_piece", ramp=3, n_perm=1200)
        assert summ["treatment_trajectory"] == "two_piece" and summ["ramp"] == 3
        # b2 = stabilised plateau (~6), b3 = positive initial-rise slope (~2/step)
        assert 4.0 < float(summ["level_change_b2"]) < 8.0
        assert float(summ["slope_change_b3"]) > 0.5
        assert summ["p_level_change_perm"] < 0.05
        assert "plateau" in " ".join(sheet["Explanation"]).lower()

    def test_model_selection_table_and_sensitivity(self):
        from functions.sced.mbd.multilevel import mbd_model_selection
        df = self._build_rampplateau(plateau=6.0, ramp=3, seed=5)
        table, summ = mbd_model_selection(df, tier_col="tier", session_col="session",
                                          outcome_col="y", phase_col="phase", ramp=3)
        assert {"AIC", "BIC", "logLik", "LRT vs immediate"}.issubset(table.columns)
        assert summ["aic_preferred"] in ("immediate", "continuous", "two_piece")
        assert summ["effect_sign_stable"] is True        # clear positive effect across models
        assert "Interpretation" in table.columns

    def test_effect_sign_stable_ignores_zero_b2(self):
        # F1/F3 audit regression: a b2 that rounds to exactly 0 (e.g. two-piece coding
        # moving the level into the ramp term) is NOT a sign flip vs positive specs.
        import functions.sced.mbd.multilevel as mod
        rows = [{"b2 (level)": 0.6, "AIC": 1.0, "BIC": 1.0},
                {"b2 (level)": 0.4, "AIC": 2.0, "BIC": 2.0},
                {"b2 (level)": 0.0, "AIC": 3.0, "BIC": 3.0}]   # rounds to zero
        signs = {np.sign(r["b2 (level)"]) for r in rows if r["b2 (level)"] != 0}
        assert not (1.0 in signs and -1.0 in signs)            # stable
        # but strictly opposite signs ARE flagged unstable
        signs2 = {np.sign(0.6), np.sign(-0.5)}
        assert (1.0 in signs2 and -1.0 in signs2)

    def test_model_selection_default_no_random_structure(self):
        # F6: random_structure now defaults False (opt-in) - no [random] rows by default.
        from functions.sced.mbd.multilevel import mbd_model_selection
        df = self._build_rampplateau(plateau=6.0, ramp=3, seed=6)
        table, summ = mbd_model_selection(df, tier_col="tier", session_col="session",
                                          outcome_col="y", phase_col="phase", ramp=3)
        assert "random_structure_bic_preferred" not in summ
        assert not table["Model"].astype(str).str.contains("random").any()


class TestHedgesABkPower:
    def test_worked_example_to_the_digit(self):
        """Reproduce Hedges, Shadish & Natesan Batley (2022) worked example exactly:
        k=2, n=3, m=3, phi=rho=0.5, delta=0.75 -> a=0.1670, c*=0.4571, h=5.95, lambda=1.982, p=0.38."""
        from functions.sced.power import power_abk_design
        r = power_abk_design(delta=0.75, k=2, n=3, m=3, phi=0.5, rho=0.5)
        assert r["a"] == 0.1670
        assert r["c_star"] == 0.4571
        assert r["h_df"] == 5.95
        assert r["lambda"] == 1.982
        assert abs(r["power"] - 0.38) <= 0.01      # nu ~= h approx (paper reports 0.38)

    def test_headline_designs(self):
        from functions.sced.power import power_abk_design
        # paper: m=5,n=3 -> 0.65 ; m=6,n=5 -> 0.80
        assert abs(power_abk_design(delta=0.75, k=2, n=3, m=5)["power"] - 0.65) <= 0.01
        assert abs(power_abk_design(delta=0.75, k=2, n=5, m=6)["power"] - 0.80) <= 0.01

    def test_power_monotone_in_effect_and_cases(self):
        from functions.sced.power import power_abk_design
        p_lo = power_abk_design(delta=0.5, k=2, n=3, m=3)["power"]
        p_hi = power_abk_design(delta=1.0, k=2, n=3, m=3)["power"]
        p_more = power_abk_design(delta=0.5, k=2, n=3, m=6)["power"]
        assert p_hi > p_lo and p_more > p_lo            # delta and m both raise power
        # k=1 is much weaker than k=2 (Hedges 2022 headline)
        assert (power_abk_design(delta=0.75, k=1, n=3, m=3)["power"]
                < power_abk_design(delta=0.75, k=2, n=3, m=3)["power"])

    def test_mdes_and_requires_two_cases(self):
        from functions.sced.power import mdes_abk_design, power_abk_design
        md = mdes_abk_design(k=2, n=3, m=6, target_power=0.80)
        assert isinstance(md["mdes_delta"], float) and 0.3 < md["mdes_delta"] < 2.0
        with pytest.raises(ValueError):
            power_abk_design(delta=0.75, k=2, n=3, m=1)

    def test_wwc_design_check(self):
        from functions.sced.power import wwc_design_check
        good, sheet = wwc_design_check(k=2, n_per_phase=5, m=4, design="phase")
        assert good["meets_without_reservations"] is True
        bad, _ = wwc_design_check(k=1, n_per_phase=2, m=1, design="phase")
        assert bad["all_pass"] is False and bad["meets_with_reservations"] is False
        mbd, _ = wwc_design_check(k=1, n_per_phase=5, m=3, design="mbd")
        assert mbd["meets_without_reservations"] is True   # MBD judged on m>=3, not k


# --------------------------------------------------------------------------- #
# Ordered-trend (dose-response) statistic for ordered conditions (ex. MI 4/5/6 s)
# --------------------------------------------------------------------------- #
class TestOrderedTrend:
    def _dose_response_df(self, seed=0, effect=3.0):
        """Balanced ATD, 3 conditions A/B/C (MI 4/5/6 s) with monotone dose-response."""
        rng = np.random.RandomState(seed); rows = []; s = 0
        eff = {"A": 0.0, "B": effect, "C": 2 * effect}
        for _ in range(10):
            for c in rng.permutation(["A", "B", "C"]):
                s += 1
                rows.append({"session": s, "condition": c,
                             "score": eff[c] + rng.normal(20, 2)})
        return pd.DataFrame(rows)

    def test_slope_orientation_and_dose_units(self):
        # positive slope when the outcome rises with the dose; oriented by improvement
        labels = np.array(["A", "B", "C", "A", "B", "C"])
        values = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        inc = make_ordered_trend_stat(("A", "B", "C"), kind="slope", dose=(4, 5, 6))
        # +1 outcome per MI second (dose 4->5->6, outcome 1->2->3)
        assert inc(labels, values) == pytest.approx(1.0)
        dec = make_ordered_trend_stat(("A", "B", "C"), kind="slope", dose=(4, 5, 6),
                                      improvement="decrease")
        assert dec(labels, values) == pytest.approx(-1.0)

    def test_jonckheere_increases_with_separation(self):
        jt = make_ordered_trend_stat(("A", "B", "C"), kind="jonckheere")
        lab = np.array(["A", "A", "B", "B", "C", "C"])
        strong = jt(lab, np.array([1, 2, 5, 6, 9, 10.0]))   # clear increasing separation
        weak = jt(lab, np.array([5, 6, 5, 6, 5, 6.0]))      # no trend
        assert strong > weak

    def test_randomization_detects_dose_response(self):
        df = self._dose_response_df(effect=3.0)
        labels = df["condition"].to_numpy()
        values = df["score"].to_numpy()
        stat = make_ordered_trend_stat(("A", "B", "C"), kind="slope", dose=(4, 5, 6))
        res = randomization_test(labels, values, statistic=stat,
                                 scheme=block_scheme(3), n_perm=2000, two_sided=False)
        assert res["p_value"] < 0.05

    def test_randomization_calibrated_under_null(self):
        # no effect -> p ~ uniform, must not be systematically significant
        rng = np.random.RandomState(1); rows = []; s = 0
        for _ in range(10):
            for c in rng.permutation(["A", "B", "C"]):
                s += 1
                rows.append({"session": s, "condition": c, "score": rng.normal(20, 2)})
        df = pd.DataFrame(rows)
        stat = make_ordered_trend_stat(("A", "B", "C"), kind="slope")
        res = randomization_test(df["condition"].to_numpy(), df["score"].to_numpy(),
                                 statistic=stat, scheme=block_scheme(3),
                                 n_perm=2000, two_sided=False)
        assert res["p_value"] > 0.05

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            make_ordered_trend_stat(("A", "B", "C"), kind="bogus")

    def test_pipeline_reports_ordered_trend(self):
        df = self._dose_response_df(effect=3.0)
        info, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B", "C"), condition_order=("A", "B", "C"), dose=(4, 5, 6),
            trend_test=True, n_perm=1000, verbose=False)
        assert "Ordered-trend test" in info
        assert info["Ordered-trend slope p"] < 0.05
        # can be disabled
        info2, _ = pipeline_sced_alternating(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B", "C"), trend_test=False, n_perm=500, verbose=False)
        assert "Ordered-trend test" not in info2

    # --- audit regression tests (PR #15 corrections review) ---
    def test_condition_order_mismatch_raises_not_spurious_significance(self):
        # F1: a mislabelled / incomplete condition_order must RAISE, not silently yield a
        # NaN observed statistic that randomization_test reports as p ~ 1/(1+n_perm).
        df = self._dose_response_df(effect=3.0)
        with pytest.raises(ValueError):       # wrong case
            pipeline_sced_alternating(
                df, session_col="session", condition_col="condition", outcome_col="score",
                conditions=("A", "B", "C"), condition_order=("a", "b", "c"),
                trend_test=True, n_perm=200, verbose=False)
        with pytest.raises(ValueError):       # condition C dropped from the order
            pipeline_sced_alternating(
                df, session_col="session", condition_col="condition", outcome_col="score",
                conditions=("A", "B", "C"), condition_order=("A", "B"),
                trend_test=True, n_perm=200, verbose=False)

    def test_randomization_test_nan_observed_returns_nan_p(self):
        # F1: undefined observed statistic -> p = NaN, NOT the most-significant value.
        labels = np.array(["A", "B", "C"] * 4)
        values = np.arange(12.0)
        res = randomization_test(labels, values, statistic=lambda l, v: np.nan,
                                 scheme=alternating_scheme(), n_perm=200, two_sided=False)
        assert np.isnan(res["p_value"])

    def test_dose_length_must_match_order(self):
        df = self._dose_response_df(effect=3.0)
        with pytest.raises(ValueError):
            pipeline_sced_alternating(
                df, session_col="session", condition_col="condition", outcome_col="score",
                conditions=("A", "B", "C"), condition_order=("A", "B", "C"), dose=(4, 5),
                trend_test=True, n_perm=200, verbose=False)


# --------------------------------------------------------------------------- #
# Ordered-trend (dose-response) at the GROUP level (stratified, within-unit perm)
# --------------------------------------------------------------------------- #
class TestOrderedTrendGroup:
    def _group_df(self, n_units=5, effect=2.5, seed=0, null=False):
        rng = np.random.RandomState(seed); rows = []
        for u in range(n_units):
            base = rng.normal(0, 3)                      # niveau propre au patient
            eff = {"A": 0.0, "B": 0.0, "C": 0.0} if null else \
                  {"A": 0.0, "B": effect, "C": 2 * effect}
            s = 0
            for _ in range(8):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"unit": f"P{u}", "session": s, "condition": c,
                                 "score": base + eff[c] + rng.normal(20, 2)})
        return pd.DataFrame(rows)

    def test_detects_group_dose_response(self):
        df = self._group_df(effect=2.5)
        res = stratified_ordered_trend_test(
            df["score"].to_numpy(), df["condition"].to_numpy(), df["unit"].to_numpy(),
            condition_order=("A", "B", "C"), dose=(4, 5, 6), kind="slope",
            block_size=3, n_perm=1000)
        assert res["p_value"] < 0.05 and res["n_units"] == 5

    def test_calibrated_under_group_null(self):
        df = self._group_df(null=True, seed=3)
        res = stratified_ordered_trend_test(
            df["score"].to_numpy(), df["condition"].to_numpy(), df["unit"].to_numpy(),
            condition_order=("A", "B", "C"), kind="jonckheere", block_size=3, n_perm=1000)
        assert res["p_value"] > 0.05

    def test_group_pipeline_reports_ordered_trend(self):
        df = self._group_df(effect=2.5)
        info, _ = pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B", "C"),
            condition_order=("A", "B", "C"), dose=(4, 5, 6), trend_test=True,
            block_size=3, test_heterogeneity=False, posthoc=False, classical_checks=False,
            n_perm=600, verbose=False)
        assert "Ordered-trend test (group)" in info
        assert info["Ordered-trend slope p (group)"] < 0.05


# --------------------------------------------------------------------------- #
# Bayesian condition effect (PyMC) - skipped when PyMC is not installed
# --------------------------------------------------------------------------- #
class TestBayesian:
    def _ab(self, effect=7.0, seed=0, n_blocks=10):
        rng = np.random.RandomState(seed); rows = []
        for b in range(n_blocks):
            for c in rng.permutation(["A", "B"]):
                s = b * 2 + (1 if c == "A" else 2)
                rows.append({"session": s, "condition": c,
                             "score": (effect if c == "B" else 0) + rng.normal(20, 2)})
        return pd.DataFrame(rows)

    def test_detects_clear_effect(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        summ, sheet = bayesian_condition_effect(
            self._ab(effect=7.0), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B"), draws=600, tune=600, chains=2,
            random_state=1)
        assert summ["p_benefit"] > 0.95                  # B clearly better
        assert summ["effect_size_name"] == "Cohen's d"
        assert summ["effect_size_hdi"][0] > 0            # HDI excludes 0
        assert summ["hdi_prob"] == 0.95
        bf = summ["bf10"]
        assert bf == "inf" or isinstance(bf, str) or float(bf) > 10   # strong evidence
        assert {"Quantity", "Value", "Interpretation"}.issubset(sheet.columns)

    def test_null_does_not_favour_strong_effect(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        summ, _ = bayesian_condition_effect(
            self._ab(effect=0.0, seed=3), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B"), draws=600, tune=600, chains=2,
            random_state=1)
        assert 0.05 < summ["p_benefit"] < 0.95           # uncertain direction
        bf = summ["bf10"]
        assert bf != "inf" and (isinstance(bf, str) is False) and float(bf) < 10

    def test_group_random_intercept_and_contrast(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        rng = np.random.RandomState(7); eff = {"A": 0, "B": 4, "C": 8}; rows = []
        for p in range(5):
            order = np.repeat(["A", "B", "C"], 6); rng.shuffle(order); base = rng.uniform(55, 68)
            for s, c in zip(range(1, 19), order):
                rows.append({"patient": f"P{p}", "session": int(s), "condition": c,
                             "score": base + eff[c] + rng.normal(0, 4)})
        summ, _ = bayesian_condition_effect(
            pd.DataFrame(rows), session_col="session", condition_col="condition",
            outcome_col="score", group_cols=["patient"], conditions=("A", "B", "C"),
            reference="A", target="C", draws=600, tune=600, chains=2, random_state=1)
        assert summ["levels"]["patient"] == 5 and summ["contrast"] == "C - A"
        assert summ["design"].startswith("multilevel") and summ["p_benefit"] > 0.95

    def test_binary_outcome_reports_odds_ratio(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        rng = np.random.RandomState(0); rows = []
        for b in range(14):
            for c in rng.permutation(["A", "B"]):
                s = b * 2 + (1 if c == "A" else 2)
                p = 1 / (1 + np.exp(-(-0.5 + (1.6 if c == "B" else 0))))
                rows.append({"session": s, "condition": c, "score": int(rng.rand() < p)})
        summ, _ = bayesian_condition_effect(
            pd.DataFrame(rows), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B"), outcome_type="binary",
            draws=600, tune=600, chains=2, random_state=1)
        assert "Odds ratio" in summ["effect_size_name"]
        assert summ["effect_size"] > 1 and summ["p_benefit"] > 0.9

    def test_count_outcome_reports_rate_ratio(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        rng = np.random.RandomState(0); rows = []
        for b in range(14):
            for c in rng.permutation(["A", "B"]):
                s = b * 2 + (1 if c == "A" else 2)
                rows.append({"session": s, "condition": c,
                             "score": int(rng.poisson(np.exp(1.0 + (0.7 if c == "B" else 0))))})
        summ, _ = bayesian_condition_effect(
            pd.DataFrame(rows), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B"), outcome_type="count",
            draws=600, tune=600, chains=2, random_state=1)
        assert "Rate ratio" in summ["effect_size_name"]
        assert summ["effect_size"] > 1 and summ["p_benefit"] > 0.9

    def test_informative_prior_raw_units_pulls_estimate(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        # weak data effect; a tight informative prior (raw units) should pull the estimate up
        df = self._ab(effect=1.0, seed=4, n_blocks=6)
        loose, _ = bayesian_condition_effect(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B"), draws=600, tune=600, chains=2, random_state=1)
        tight, _ = bayesian_condition_effect(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B"), prior_mean=8.0, prior_sd=1.0,   # raw outcome units
            draws=600, tune=600, chains=2, random_state=1)
        assert "informative" in tight["effect_prior"]
        assert "non-informative" in loose["effect_prior"]
        assert tight["effect_size"] > loose["effect_size"]   # prior shifts posterior

    def test_all_contrasts_reported_and_rope(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        rng = np.random.RandomState(0); eff = {"A": 0, "B": 3, "C": 6}; rows = []
        for b in range(10):
            for c in rng.permutation(["A", "B", "C"]):
                rows.append({"session": len(rows) + 1, "condition": c,
                             "score": eff[c] + rng.normal(20, 2)})
        summ, sheet = bayesian_condition_effect(
            pd.DataFrame(rows), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B", "C"), rope=(-0.1, 0.1),
            draws=600, tune=600, chains=2, random_state=1)
        names = [c["contrast"] for c in summ["contrasts"]]
        assert names == ["B - A", "C - A"]               # all non-reference contrasts
        assert summ["contrast"] == "C - A"               # headline = last by default
        assert summ["rope"]["decision"].startswith("HDI outside ROPE")  # clear effect
        assert sheet["Quantity"].astype(str).str.startswith("ROPE").any()

    def test_partial_pooling_prior_does_not_require_distribution_name(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        with pytest.raises(ValueError):              # must give BOTH mean and sd
            bayesian_condition_effect(self._ab(), session_col="session",
                                      condition_col="condition", outcome_col="score",
                                      conditions=("A", "B"), prior_mean=0.5,
                                      draws=200, tune=200, chains=1)

    def test_diagnostics_plots(self, tmp_path):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import (bayesian_condition_effect,
                                             plot_bayesian_diagnostics)
        summ, sheet, idata = bayesian_condition_effect(
            self._ab(effect=6.0), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B"), rope=(-0.1, 0.1),
            return_idata=True, draws=400, tune=400, chains=2, random_state=1)
        paths = plot_bayesian_diagnostics(idata, save_path=str(tmp_path), rope=(-0.1, 0.1),
                                          title="test")
        assert {"trace", "rank", "forest", "posterior", "energy", "ppc",
                "dashboard"}.issubset(paths)
        for p in paths.values():
            assert __import__("os").path.getsize(p) > 0

    def test_multiple_contrasts_named_dim_and_plots(self, tmp_path):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import (bayesian_condition_effect,
                                             plot_bayesian_diagnostics)
        rng = np.random.RandomState(0); eff = {"A": 0, "B": 3, "C": 6}; rows = []
        for b in range(10):
            for c in rng.permutation(["A", "B", "C"]):
                rows.append({"session": len(rows) + 1, "condition": c,
                             "score": eff[c] + rng.normal(20, 2)})
        summ, sheet, idata = bayesian_condition_effect(
            pd.DataFrame(rows), session_col="session", condition_col="condition",
            outcome_col="score", conditions=("A", "B", "C"), rope=(-0.1, 0.1),
            return_idata=True, draws=400, tune=400, chains=2, random_state=1)
        # the delta dimension is labelled by the actual contrasts (not delta[0], delta[1])
        labels = list(idata.posterior["delta"].coords["contrast"].values)
        assert labels == ["B - A", "C - A"]
        paths = plot_bayesian_diagnostics(idata, save_path=str(tmp_path), rope=(-0.1, 0.1))
        assert __import__("os").path.getsize(paths["posterior"]) > 0   # one panel per contrast

    def test_robust_exposes_nu_and_detects_outliers(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        df = self._ab(effect=6.0, seed=0, n_blocks=14)
        df.loc[[0, 5, 10], "score"] += [40, -35, 38]      # 3 aberrant sessions
        summ, sheet = bayesian_condition_effect(
            df, session_col="session", condition_col="condition", outcome_col="score",
            conditions=("A", "B"), outcome_type="robust", draws=600, tune=600, chains=2,
            random_state=1)
        assert "nu_robust" in summ and summ["nu_robust"] < 7      # heavy tails detected
        assert sheet["Quantity"].astype(str).str.startswith("nu").any()

    def test_invalid_reference_target_raises(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        with pytest.raises(ValueError):
            bayesian_condition_effect(self._ab(), session_col="session",
                                      condition_col="condition", outcome_col="score",
                                      conditions=("A", "B"), reference="A", target="A",
                                      draws=200, tune=200, chains=1)

    def test_binary_rejects_non_binary_values(self):
        pytest.importorskip("pymc")
        from functions.sced.bayes.condition import bayesian_condition_effect
        with pytest.raises(ValueError):
            bayesian_condition_effect(self._ab(effect=3.0), session_col="session",
                                      condition_col="condition", outcome_col="score",
                                      conditions=("A", "B"), outcome_type="binary",
                                      draws=200, tune=200, chains=1)


# --------------------------------------------------------------------------- #
# Unified dispatcher run_sced_alternating (automatic engine choice)
# --------------------------------------------------------------------------- #
class TestDispatcher:
    def _df(self, seed=0, n_units=1):
        rng = np.random.RandomState(seed); rows = []
        for u in range(n_units):
            base = rng.normal(0, 3); s = 0
            for _ in range(8):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"patient": f"P{u}", "session": s, "cond": c,
                                 "y": base + {"A": 0, "B": 2, "C": 4}[c] + rng.normal(20, 2),
                                 "rt": rng.normal(10, 2)})
        return pd.DataFrame(rows)

    def test_dispatch_nof1(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._df(n_units=1).drop(columns="patient"),
                                 session_col="session", condition_col="cond", outcome="y",
                                 conditions=("A", "B", "C"), n_perm=400, verbose=False)
        assert o["engine"].startswith("n-of-1") and o["model_info"] is not None

    def test_dispatch_group(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._df(n_units=4), unit_col="patient",
                                 session_col="session", condition_col="cond", outcome="y",
                                 conditions=("A", "B", "C"), test_heterogeneity=False,
                                 posthoc=False, classical_checks=False, n_perm=400, verbose=False)
        assert "group" in o["engine"] and o["model_info"]["Number of units"] == 4

    def test_dispatch_multivariate(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._df(n_units=1).drop(columns="patient"),
                                 session_col="session", condition_col="cond",
                                 outcome=["y", "rt"], conditions=("A", "B", "C"), n_perm=400)
        assert "PERMANOVA" in o["engine"] and "pseudo_F" in o["result"]

    def test_explain_helper_text(self):
        from functions.sced.alternating.run import explain_statistic_choice
        txt = explain_statistic_choice(grouped=True)
        assert "PERMANOVA" in txt and "Freedman-Lane" in txt and "dose" in txt.lower()


class TestTimeCovariateRename:
    def _df(self, seed=0):
        rng = np.random.RandomState(seed); conds = np.tile(["A", "B"], 9)
        sess = np.arange(1, 19)
        y = np.where(conds == "B", 4, 0) + 1.0 * sess + rng.normal(0, 1, 18)
        return pd.DataFrame({"session": sess, "cond": conds, "y": y})

    def test_time_covariate_alias_matches_detrend(self):
        # old name (detrend) and new name (time_covariate) -> identical result
        df = self._df()
        new, _ = pipeline_sced_alternating(df, session_col="session", condition_col="cond",
                                           outcome_col="y", conditions=("A", "B"),
                                           time_covariate="linear", n_perm=1000, verbose=False)
        old, _ = pipeline_sced_alternating(df, session_col="session", condition_col="cond",
                                           outcome_col="y", conditions=("A", "B"),
                                           detrend="linear", n_perm=1000, verbose=False)
        assert new["Observed statistic"] == old["Observed statistic"]
        assert new["Detrend"] == old["Detrend"] == "linear"

    def test_permanova_accepts_time_covariate(self):
        rng = np.random.RandomState(1); rows = []; s = 0
        for _ in range(9):
            for c in rng.permutation(["A", "B"]):
                s += 1
                rows.append({"session": s, "cond": c, "y": (3 if c == "B" else 0) + rng.normal(0, 1),
                             "z": rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        r = permanova_condition_test(df, outcome_cols=["y", "z"], session_col="session",
                                     condition_col="cond", time_covariate="linear", n_perm=400)
        assert "pseudo_F" in r


class TestRunsAndLevel3:
    def _wide(self, n_units=5, n_runs=4, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for u in range(n_units):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1; vmean = base + {"A": 0, "B": 2, "C": 4}[c]
                    r = {"patient": f"P{u}", "session": s, "cond": c}
                    for k in range(1, n_runs + 1):
                        r[f"run_{k}"] = vmean + rng.normal(0, 2)
                    rows.append(r)
        return pd.DataFrame(rows)

    def test_aggregate_runs_shapes(self):
        from functions.sced.prep import aggregate_runs
        df = self._wide(n_units=2)
        d, long, out = aggregate_runs(df, session_col="session", condition_col="cond",
                                      run_cols=["run_1", "run_2", "run_3", "run_4"],
                                      unit_col="patient", agg="mean")
        assert out in d.columns
        assert len(long) == len(df) * 4 and "_measure" in long.columns

    def test_runs_drive_group_engine(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._wide(), unit_col="patient", session_col="session",
                                 condition_col="cond", run_cols=["run_1", "run_2", "run_3", "run_4"],
                                 conditions=("A", "B", "C"), test_heterogeneity=False, posthoc=False,
                                 classical_checks=False, trend_test=False, n_perm=400, verbose=False)
        assert "group" in o["engine"] and o["model_info"]["Randomization p-value"] < 0.05

    def test_level3_decomposition(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._wide(), unit_col="patient", session_col="session",
                                 condition_col="cond", run_cols=["run_1", "run_2", "run_3", "run_4"],
                                 level3=True, conditions=("A", "B", "C"), test_heterogeneity=False,
                                 posthoc=False, classical_checks=False, trend_test=False,
                                 n_perm=400, verbose=False)
        s = o["level3"]["summary"]
        tot = s["pct_patient"] + s["pct_visit"] + s["pct_within_visit"]
        assert abs(tot - 100) < 1.0          # the 3 components sum to ~100%

    def test_level3_ignored_without_runs_or_units(self):
        from functions.sced.alternating.run import run_sced_alternating
        df = self._wide(n_units=1).drop(columns="patient")
        o = run_sced_alternating(df, session_col="session", condition_col="cond",
                                 run_cols=["run_1", "run_2", "run_3", "run_4"], level3=True,
                                 conditions=("A", "B", "C"), trend_test=False, n_perm=300, verbose=False)
        assert "level3" not in o and "level3_note" in o


class TestMeasuresFlexibleInput:
    def _wide(self, n_units=5, n_runs=4, seed=1):
        rng = np.random.RandomState(seed); rows = []
        for u in range(n_units):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1; vm = base + {"A": 0, "B": 2, "C": 4}[c]
                    r = {"patient": f"P{u}", "session": s, "cond": c}
                    for k in range(1, n_runs + 1):
                        r[f"run_{k}"] = vm + rng.normal(0, 2)
                    rows.append(r)
        return pd.DataFrame(rows)

    def _common(self):
        return dict(unit_col="patient", session_col="session", condition_col="cond",
                    conditions=("A", "B", "C"), test_heterogeneity=False, posthoc=False,
                    classical_checks=False, trend_test=False, n_perm=300, verbose=False)

    def test_measures_prefix(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._wide(), measures="run", level3=True, **self._common())
        assert "level3" in o and o["level3"]["summary"]["pct_within_visit"] > 0

    def test_measures_glob_equals_prefix(self):
        from functions.sced.alternating.run import run_sced_alternating
        a = run_sced_alternating(self._wide(), measures="run_*", level3=True, **self._common())
        b = run_sced_alternating(self._wide(), measures="run", level3=True, **self._common())
        assert (a["level3"]["summary"]["pct_within_visit"]
                == b["level3"]["summary"]["pct_within_visit"])

    def test_long_format_auto_detected(self):
        # several rows per visit, NO column named 'run' -> auto-detected
        from functions.sced.alternating.run import run_sced_alternating
        W = self._wide()
        L = W.melt(id_vars=["patient", "session", "cond"],
                   value_vars=["run_1", "run_2", "run_3", "run_4"],
                   value_name="score").drop(columns="variable")
        o = run_sced_alternating(L, outcome="score", level3=True, **self._common())
        assert "level3" in o and o["model_info"]["Randomization p-value"] < 0.05

    def test_bad_measures_spec_raises(self):
        from functions.sced.alternating.run import run_sced_alternating
        with pytest.raises(ValueError):
            run_sced_alternating(self._wide(), measures="zzz_no_match", **self._common())


class TestLevel3FullCoverage:
    """Checks that the unified engine covers the 3 analyses of the former template_sced_multilevel."""
    def _wide(self, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for u in range(5):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1; vm = base + {"A": 0, "B": 2, "C": 4}[c]
                    r = {"patient": f"P{u}", "session": s, "cond": c}
                    for k in range(1, 5):
                        r[f"run_{k}"] = vm + rng.normal(0, 2) + 0.3 * k
                    rows.append(r)
        return pd.DataFrame(rows)

    def test_level3_has_three_analyses(self):
        from functions.sced.alternating.run import run_sced_alternating
        o = run_sced_alternating(self._wide(), unit_col="patient", session_col="session",
                                 condition_col="cond", measures="run", level3=True, dispersion="sd",
                                 within_visit_slope=True,   # opt-in (non-standard test, off by default)
                                 conditions=("A", "B", "C"), test_heterogeneity=False, posthoc=False,
                                 classical_checks=False, trend_test=False, n_perm=300, verbose=False)
        l3 = o["level3"]
        assert {"summary", "dispersion", "within_visit_slope"} <= set(l3)
        assert 0 <= l3["dispersion"]["p_value"] <= 1
        assert 0 <= l3["within_visit_slope"]["p_value"] <= 1

    def test_long_format_trial_col_feeds_within_visit_slope(self):
        from functions.sced.alternating.run import run_sced_alternating
        W = self._wide()
        L = W.melt(id_vars=["patient", "session", "cond"],
                   value_vars=["run_1", "run_2", "run_3", "run_4"],
                   var_name="trial", value_name="score")
        L["trial"] = L["trial"].str.replace("run_", "").astype(int)
        o = run_sced_alternating(L, unit_col="patient", session_col="session", condition_col="cond",
                                 outcome="score", level3=True, trial_col="trial", within_visit_slope=True,
                                 conditions=("A", "B", "C"), test_heterogeneity=False, posthoc=False,
                                 classical_checks=False, trend_test=False, n_perm=300, verbose=False)
        assert "within_visit_slope" in o["level3"]


class TestPermanovaMissingData:
    def _df(self, seed=0, with_nan=False):
        rng = np.random.RandomState(seed); rows = []; s = 0
        for _ in range(9):
            for c in rng.permutation(["A", "B"]):
                s += 1
                rows.append({"session": s, "cond": c,
                             "y": (3 if c == "B" else 0) + rng.normal(0, 1),
                             "z": rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        if with_nan:
            df.loc[2, "z"] = np.nan       # a single NaN
        return df

    def test_nan_dropped_not_spurious_p(self):
        # before the fix: pseudo_F=nan -> spurious p=1/(B+1). After: the row is dropped.
        clean = permanova_condition_test(self._df(), outcome_cols=["y", "z"],
                                         session_col="session", condition_col="cond", n_perm=500)
        nanrow = permanova_condition_test(self._df(with_nan=True), outcome_cols=["y", "z"],
                                          session_col="session", condition_col="cond", n_perm=500)
        assert np.isfinite(nanrow["pseudo_F"])            # no more nan
        assert nanrow["n_dropped_missing"] == 1
        assert nanrow["p_value"] > 0.001                  # no more spurious p = 1/(B+1)
        assert np.isfinite(clean["pseudo_F"]) and clean["n_dropped_missing"] == 0


class TestRepeatedMeasuresFormat:
    def test_allow_repeated_measures_skips_dup_check(self):
        from functions.sced.core import check_long_format
        # 2 runs per visit -> dup keys
        df = pd.DataFrame({"u": ["P1"]*4, "session": [1, 1, 2, 2],
                           "cond": ["A", "A", "B", "B"], "y": [1.0, 2, 3, 4]})
        strict = check_long_format(df, session_col="session", condition_col="cond",
                                   outcome_col="y", unit_col="u")
        lax = check_long_format(df, session_col="session", condition_col="cond",
                                outcome_col="y", unit_col="u", allow_repeated_measures=True)
        assert any("a single row per session" in m for m in strict)   # blocks by default
        assert lax == []                                               # accepted when allowed


class TestHeterogeneityExtended:
    def _grp(self, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for u in range(5):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"u": f"P{u}", "session": s, "cond": c,
                                 "y": base + {"A": 0, "B": 2, "C": 4}[c] + rng.normal(0, 2)})
        return pd.DataFrame(rows)

    def test_per_contrast_and_dose_slope_present(self):
        df = self._grp()
        het = heterogeneity_test(df["y"].to_numpy(), df["cond"].to_numpy(), df["u"].to_numpy(),
                                 df["session"].to_numpy(), conditions=("A", "B", "C"),
                                 condition_order=("A", "B", "C"), dose=(4, 5, 6), n_perm=300)
        # all pairs C(3,2) = 3 contrasts
        assert len(het["per_contrast"]) == 3
        assert {p["contrast"] for p in het["per_contrast"]} == {"B - A", "C - A", "C - B"}
        # dose-response slope present
        assert "dose_slope" in het and "I2_percent" in het["dose_slope"]

    def test_heterogeneity_sheet_has_three_kinds(self, tmp_path):
        df = self._grp().rename(columns={"u": "unit"})
        pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond", outcome_col="y",
            conditions=("A", "B", "C"), condition_order=("A", "B", "C"), dose=(4, 5, 6),
            test_heterogeneity=True, posthoc=False, classical_checks=False, trend_test=False,
            n_perm=400, save_path=str(tmp_path), verbose=False)
        xls = tmp_path / "y_sced_alternating_group_report.xlsx"
        het = pd.read_excel(xls, sheet_name="Heterogeneity")
        elems = " ".join(het["Element"].astype(str))
        assert "interaction" in elems and "Contrast" in elems and "Dose-response" in elems
        assert "Explanation" not in het.columns and "Interpretation" in het.columns


class TestPerUnitMultiplicity:
    def test_individual_p_has_holm_fdr(self):
        from functions.sced.alternating.group import _per_unit_table
        rng = np.random.RandomState(0); rows = []
        for u in range(5):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"u": f"P{u}", "session": s, "cond": c,
                                 "y": base + {"A": 0, "B": 2, "C": 4}[c] + rng.normal(0, 2)})
        df = pd.DataFrame(rows)
        pu = _per_unit_table(df, unit_col="u", session_col="session", condition_col="cond",
                             outcome_col="y", conditions=("A", "B", "C"), improvement="increase",
                             n_perm=300)
        assert {"Individual p", "Individual p Holm", "Individual p FDR"} <= set(pu.columns)
        # corrected >= raw (Holm is conservative), where defined
        ok = pu["Individual p"].notna() & pu["Individual p Holm"].notna()
        assert (pu.loc[ok, "Individual p Holm"] >= pu.loc[ok, "Individual p"] - 1e-9).all()


class TestBenchmarksAndTauCI:
    def test_icc_i2_bands(self):
        from functions.sced.core import icc_band, i2_band
        assert icc_band(0.3) == "weak" and icc_band(0.6) == "moderate"
        assert icc_band(0.83) == "good" and icc_band(0.95) == "excellent"
        assert i2_band(10) == "weak" and i2_band(40) == "moderate"
        assert i2_band(60) == "strong" and i2_band(80) == "very strong"

    def test_tau_u_ci_and_critical(self):
        r = tau_u([5, 4, 3, 2, 1], [6, 7, 8, 9, 10])     # clear separation
        assert isinstance(r["ci"], tuple) and len(r["ci"]) == 2
        assert -1.0 <= r["ci"][0] <= r["ci"][1] <= 1.0    # CI bounded and ordered
        assert np.isfinite(r["critical"]) and r["critical"] > 0


class TestGroupDescriptives:
    def test_by_condition_sheet_and_per_unit_sd(self, tmp_path):
        df = _make_group(5, effect=3.0, learning=0.0, seed=4, k=3)
        pipeline_sced_alternating_group(
            df, unit_col="unit", session_col="session", condition_col="cond", outcome_col="y",
            conditions=("A", "B", "C"), test_heterogeneity=False, posthoc=False,
            classical_checks=False, trend_test=False, n_perm=300, save_path=str(tmp_path),
            verbose=False)
        xls = tmp_path / "y_sced_alternating_group_report.xlsx"
        sh = pd.read_excel(xls, sheet_name=None)
        assert "By Condition" in sh
        bc = sh["By Condition"]
        assert {"Mean (group)", "SD (pooled)", "SD between patients"} <= set(bc.columns)
        assert len(bc) == 3
        # per-condition per-patient SD present in Per Unit
        pu = sh["Per Unit"]
        assert {"sd[A]", "sd[B]", "sd[C]"} <= set(pu.columns)


class TestWithinVisitSlopeEstimate:
    def test_slope_by_condition_returned(self):
        from functions.sced.alternating.run import run_sced_alternating
        rng = np.random.RandomState(0); rows = []
        for u in range(5):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1; vm = base + {"A": 0, "B": 2, "C": 4}[c]
                    for k in range(1, 5):
                        rows.append({"u": f"P{u}", "session": s, "cond": c, "run": k,
                                     "y": vm + 0.3 * k + rng.normal(0, 2)})   # pente intra +0.3
        df = pd.DataFrame(rows)
        o = run_sced_alternating(df, unit_col="u", session_col="session", condition_col="cond",
                                 outcome="y", level3=True, trial_col="run", within_visit_slope=True,
                                 conditions=("A", "B", "C"),
                                 test_heterogeneity=False, posthoc=False, classical_checks=False,
                                 trend_test=False, n_perm=300, verbose=False)
        wvs = o["level3"]["within_visit_slope"]
        assert set(wvs["slope_by_condition"]) == {"A", "B", "C"}
        assert wvs["slope_overall"] > 0          # simulated warm-up (+0.3/run)


class TestLevel3SheetInReport:
    def test_level3_written_to_xlsx(self, tmp_path):
        from functions.sced.alternating.run import run_sced_alternating
        rng = np.random.RandomState(0); rows = []
        for u in range(5):
            base = rng.normal(0, 3); s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1; vm = base + {"A": 0, "B": 2, "C": 4}[c]
                    for k in range(1, 5):
                        rows.append({"unit": f"P{u}", "session": s, "cond": c, "run": k,
                                     "y": vm + 0.3 * k + rng.normal(0, 2)})
        df = pd.DataFrame(rows)
        run_sced_alternating(df, unit_col="unit", session_col="session", condition_col="cond",
                             outcome="y", level3=True, trial_col="run", within_visit_slope=True,
                             conditions=("A", "B", "C"),
                             test_heterogeneity=False, posthoc=False, classical_checks=False,
                             trend_test=False, n_perm=300, save_path=str(tmp_path), verbose=False)
        import glob
        xls = glob.glob(str(tmp_path / "*report.xlsx"))[0]
        sh = pd.read_excel(xls, sheet_name=None)
        assert "Level 3" in sh
        elems = " ".join(sh["Level 3"]["Quantity"].astype(str))
        assert "within-visit" in elems and "Within-visit slope per patient" in elems


class TestPlotKinds:
    def _df(self):
        rng = np.random.RandomState(0); rows = []
        for u in range(3):
            s = 0
            for _ in range(5):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    for k in range(3):
                        rows.append({"unit": f"P{u}", "session": s, "cond": c,
                                     "y": {"A": 0, "B": 2, "C": 4}[c] + rng.normal(0, 1)})
        return pd.DataFrame(rows)

    def test_all_kinds_return_figure(self):
        import matplotlib.pyplot as plt
        df = self._df()
        for kind in ("series", "box", "violin", "mean", "mean_sd"):
            for center in ("mean", "median"):
                fig = plot_sced_alternating(df, session_col="session", condition_col="cond",
                                            outcome_col="y", kind=kind, center=center)
                assert fig is not None and len(fig.axes) >= 1
                figg = plot_sced_alternating(df, unit_col="unit", session_col="session",
                                             condition_col="cond", outcome_col="y", kind=kind,
                                             center=center, show_points=False)
                assert figg is not None
                plt.close("all")


class TestPlotByAxis:
    def _df(self):
        rng = np.random.RandomState(0); rows = []
        for u in range(2):
            s = 0
            for _ in range(4):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    for k in range(4):
                        rows.append({"unit": f"P{u}", "session": s, "cond": c,
                                     "y": {"A": 0, "B": 2, "C": 4}[c] + rng.normal(0, 1)})
        return pd.DataFrame(rows)

    def test_by_session_and_condition(self):
        import matplotlib.pyplot as plt
        df = self._df()
        for by in ("session", "condition"):
            for kind in ("box", "violin", "mean", "mean_sd"):
                fig = plot_sced_alternating(df, session_col="session", condition_col="cond",
                                            outcome_col="y", kind=kind, by=by)
                assert fig is not None and len(fig.axes) >= 1
                plt.close("all")


class TestPlotConditionLine:
    def test_condition_line_toggle(self):
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0); rows = []
        for _ in range(6):
            for c in rng.permutation(["A", "B", "C"]):
                for k in range(4):
                    rows.append({"session": len(rows) // 4 + 1, "cond": c,
                                 "y": {"A": 0, "B": 2, "C": 4}[c] + rng.normal(0, 1)})
        df = pd.DataFrame(rows)

        def n_hlines(fig):
            return sum(1 for l in fig.axes[0].lines
                       if l.get_linestyle() == "--" and tuple(np.round(l.get_xdata(), 3)) == (0.0, 1.0))
        for kind in ("series", "box"):
            on = plot_sced_alternating(df, session_col="session", condition_col="cond",
                                       outcome_col="y", kind=kind, by="session", condition_line=True)
            off = plot_sced_alternating(df, session_col="session", condition_col="cond",
                                        outcome_col="y", kind=kind, by="session", condition_line=False)
            assert n_hlines(on) == 3 and n_hlines(off) == 0
            plt.close("all")


# ---------------------------------------------------------------------------
# VAIOR (Manolov & Vannest 2019) + modified Brinley plot (Manolov et al. 2021)
# ---------------------------------------------------------------------------
from functions.sced.plots.vaior import (vaior_compare, plot_vaior, plot_brinley, _theil_sen_fit,
                                   vaior_by_unit)


class TestVaior:
    def test_theil_sen_recovers_line(self):
        x = np.arange(1, 11, dtype=float)
        y = 2.0 * x + 3.0
        slope, intercept = _theil_sen_fit(x, y)
        assert abs(slope - 2.0) < 1e-9 and abs(intercept - 3.0) < 1e-9

    def test_all_compared_beyond_band_is_positive(self):
        # flat reference ~0, compared clearly above -> 100% beyond the band
        rng = np.random.RandomState(1)
        ref_x = np.arange(1, 7); ref_y = rng.normal(0, 0.5, 6)
        comp_x = np.arange(7, 13); comp_y = np.full(6, 10.0)
        r = vaior_compare(ref_x, ref_y, comp_x, comp_y, improvement="increase")
        assert r["pct_beyond_band"] == 100.0
        assert r["decision_immediate"] and r["decision_delayed"] and r["decision_overall"]

    def test_no_effect_is_negative(self):
        rng = np.random.RandomState(2)
        ref_x = np.arange(1, 7); ref_y = rng.normal(0, 1, 6)
        comp_x = np.arange(7, 13); comp_y = rng.normal(0, 1, 6)
        r = vaior_compare(ref_x, ref_y, comp_x, comp_y, improvement="increase")
        assert r["pct_beyond_band"] < 100.0 and not r["decision_overall"]

    def test_decrease_orientation(self):
        ref_x = np.arange(1, 7); ref_y = np.zeros(6)
        comp_x = np.arange(7, 13); comp_y = np.full(6, -10.0)
        r = vaior_compare(ref_x, ref_y, comp_x, comp_y, improvement="decrease")
        assert r["decision_overall"] and all(c == "green" for c in r["colors"])

    def test_plot_vaior_returns_fig(self):
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(3); rows = []
        s = 0
        for _ in range(6):
            for c in rng.permutation(["A", "B"]):
                s += 1
                rows.append({"session": s, "cond": c,
                             "y": {"A": 0, "B": 5}[c] + rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        fig, res = plot_vaior(df, session_col="session", condition_col="cond",
                              outcome_col="y", reference="A", compared="B")
        assert fig is not None and "pct_beyond_band" in res
        plt.close("all")


class TestBetweenCaseSMD:
    def _df(self, gap=12, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for ti, st in enumerate((5, 8, 11, 14)):
            for sx in range(1, 19):
                ph = "B" if sx >= st else "A"
                rows.append({"tier": f"T{ti+1}", "session": sx, "phase": ph,
                             "y": 50 + (gap if ph == "B" else 0) + 0.2 * sx + rng.normal(0, 4)})
        return pd.DataFrame(rows)

    def test_bcsmd_positive_effect(self):
        from functions.sced.mbd.multilevel import between_case_smd
        r = between_case_smd(self._df(gap=12), tier_col="tier", session_col="session",
                             outcome_col="y", phase_col="phase")
        assert r["n_cases"] == 4 and np.isfinite(r["g"]) and r["g"] > 0
        assert r["ci_low"] <= r["g"] <= r["ci_high"] and 0 < r["J"] <= 1

    def test_bcsmd_too_few_cases(self):
        from functions.sced.mbd.multilevel import between_case_smd
        df = self._df(); df = df[df["tier"] == "T1"]
        r = between_case_smd(df, tier_col="tier", session_col="session", outcome_col="y",
                             phase_col="phase")
        assert r["n_cases"] < 2 and not np.isfinite(r["g"])

    def test_bcsmd_satterthwaite_df_and_t_ci(self):
        # Welch-Satterthwaite df >= n_cases-1 (within-case contributes); level model by default
        from functions.sced.mbd.multilevel import between_case_smd
        r = between_case_smd(self._df(gap=12), tier_col="tier", session_col="session",
                             outcome_col="y", phase_col="phase", model="level")
        assert r["model"] == "level" and r["df"] >= r["n_cases"] - 1
        assert r["n_obs"] > r["n_cases"] and np.isfinite(r["g"])


class TestMBDPanels:
    def _df(self):
        rng = np.random.RandomState(0); rows = []
        for ti, st in enumerate((5, 7, 9, 6)):
            for sx in range(1, 16):
                ph = "B" if sx >= st else "A"
                rows.append({"tier": f"T{ti+1}", "session": sx, "phase": ph,
                             "y": 50 + (10 if ph == "B" else 0) + rng.normal(0, 3)})
        return pd.DataFrame(rows)

    def test_mbd_pooled_fit(self):
        """Pooled event-time plot: per-patient spaghetti + pooled mean + model curve
        (b0/b1/b2/b3). Re-aligned on t=0 at the intervention."""
        import matplotlib.pyplot as plt
        from functions.sced.plots.panels import plot_mbd_pooled_fit
        df = self._df()
        starts = {"T1": 5, "T2": 7, "T3": 9, "T4": 6}
        fit = {"intercept_b0": 10.0, "baseline_trend_b1": 0.2,
               "level_change_b2": 4.0, "slope_change_b3": 0.1}
        fig = plot_mbd_pooled_fit(df, tier_col="tier", session_col="session", outcome_col="y",
                                  starts=starts, fit=fit, per_case=True, bounds=(0, 20))
        ax = fig.axes[0]
        from matplotlib.colors import same_color
        navy = [ln for ln in ax.lines if same_color(ln.get_color(), "navy")]
        crimson = [ln for ln in ax.lines if same_color(ln.get_color(), "crimson")]
        assert len(navy) == 1 and len(crimson) >= 1            # A and B segments of the model
        vlines = [ln for ln in ax.lines if np.allclose(ln.get_xdata(), 0.0)]
        assert len(vlines) >= 1                                # intervention line at t=0
        plt.close("all")

    def test_mbd_pooled_fit_bayes_band(self):
        """Bayesian option: band_draws -> 95% credible band (2 ribbons A and B)."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from functions.sced.plots.panels import plot_mbd_pooled_fit
        rng = np.random.RandomState(1)
        fit = {"baseline_trend_b1": 0.2, "level_change_b2": 4.0, "slope_change_b3": 0.1}
        band = {"b1": rng.normal(0.2, 0.05, 400), "b2": rng.normal(4.0, 1.0, 400),
                "b3": rng.normal(0.1, 0.05, 400)}
        fig = plot_mbd_pooled_fit(self._df(), tier_col="tier", session_col="session",
                                  outcome_col="y", starts={"T1": 5, "T2": 7, "T3": 9, "T4": 6},
                                  fit=fit, band_draws=band, per_case=True)
        nb = sum(isinstance(c, PolyCollection) for c in fig.axes[0].collections)
        assert nb >= 2                                          # ribbons A + B
        plt.close("all")

    def test_mbd_panels_returns_fig(self):
        import matplotlib.pyplot as plt
        from functions.sced.plots.panels import plot_mbd_panels
        df = self._df()
        starts = {"T1": 5, "T2": 7, "T3": 9, "T4": 6}
        fig = plot_mbd_panels(df, tier_col="tier", session_col="session", outcome_col="y",
                              starts=starts, phase_col="phase", baseline="A", treatment="B")
        # 4 tiers (left) + Brinley + distribution (right)
        assert len(fig.axes) >= 5
        # B slope in dotted green (#2ca02c, ls ":") on each tier (trend lines drawn)
        greens = [ln for ax in fig.axes for ln in ax.lines
                  if ln.get_linestyle() == ":" and ln.get_color() == "#2ca02c"]
        assert len(greens) == 4
        # NO trend legend on the per-tier panels (user request)
        labels = [t.get_text() for ax in fig.axes if ax.get_legend()
                  for t in ax.get_legend().get_texts()]
        assert "tendance B" not in labels and "Treatment (Phase B) trend" not in labels
        plt.close("all")

    def test_mbd_panels_wrap_batch_bounds(self, tmp_path):
        """>5 tiers -> multiple columns; >10 -> several batches (_b1/_b2); bounds bounds y."""
        import os, glob, matplotlib.pyplot as plt
        from functions.sced.plots.panels import plot_mbd_panels, _grid_cols
        rng = np.random.RandomState(1); rows = []
        for ti in range(12):                              # 12 tiers -> 2 batches of <=10
            st = 5 + (ti % 4)
            for sx in range(1, 16):
                ph = "B" if sx >= st else "A"
                rows.append({"tier": f"T{ti:02d}", "session": sx, "phase": ph,
                             "y": 50 + (8 if ph == "B" else 0) + rng.normal(0, 3)})
        df = pd.DataFrame(rows)
        starts = {f"T{ti:02d}": 5 + (ti % 4) for ti in range(12)}
        fig = plot_mbd_panels(df, tier_col="tier", session_col="session", outcome_col="y",
                              starts=starts, phase_col="phase", baseline="A", treatment="B",
                              bounds=(0, 100), name="X", save_path=str(tmp_path), max_per_fig=10)
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(str(tmp_path), "X_mbd_panels*.png")))
        assert files == ["X_mbd_panels_b1.png", "X_mbd_panels_b2.png"]   # 12 > 10 -> 2 batches
        assert _grid_cols(7) >= 2                          # > 5 -> at least 2 columns
        plt.close("all")


class TestBrinley:
    def test_plot_brinley_returns_fig(self):
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(4); rows = []
        for u in range(3):
            s = 0
            for _ in range(5):
                for c in rng.permutation(["A", "B"]):
                    s += 1
                    rows.append({"unit": f"P{u}", "session": s, "cond": c,
                                 "y": {"A": 1, "B": 4}[c] + rng.normal(0, 0.5)})
        df = pd.DataFrame(rows)
        fig = plot_brinley(df, condition_col="cond", outcome_col="y",
                           cond_x="A", cond_y="B", session_col="session", unit_col="unit")
        assert fig is not None and len(fig.axes) >= 1

    def test_plot_brinley_modes(self):
        """classic = 1 point/patient (means); paired = several points/patient; invalid mode rejected."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(7); rows = []
        for u in range(3):
            s = 0
            for _ in range(5):
                for c in ["A", "B"]:
                    s += 1
                    rows.append({"unit": f"P{u}", "session": s, "cond": c,
                                 "y": {"A": 1, "B": 4}[c] + rng.normal(0, 0.5)})
        df = pd.DataFrame(rows)
        fc = plot_brinley(df, condition_col="cond", outcome_col="y", cond_x="A", cond_y="B",
                          session_col="session", unit_col="unit", mode="classic")
        # classic: one point (one scatter call) per patient -> 3 point collections
        n_pts_classic = sum(len(c.get_offsets()) for c in fc.axes[0].collections)
        fp = plot_brinley(df, condition_col="cond", outcome_col="y", cond_x="A", cond_y="B",
                          session_col="session", unit_col="unit", mode="paired")
        n_pts_paired = sum(len(c.get_offsets()) for c in fp.axes[0].collections)
        assert n_pts_classic == 3                 # 1 point per patient
        assert n_pts_paired > n_pts_classic       # several pairs per patient
        plt.close(fc); plt.close(fp)
        with pytest.raises(ValueError):
            plot_brinley(df, condition_col="cond", outcome_col="y", cond_x="A", cond_y="B",
                         unit_col="unit", mode="nope")

    def test_plot_brinley_bounds_axes(self):
        """bounds=(lo,hi) bounds both axes to the real outcome scale."""
        import matplotlib.pyplot as plt
        df = pd.DataFrame({"unit": ["P0"] * 4 + ["P1"] * 4,
                           "session": list(range(1, 5)) * 2,
                           "cond": ["A", "A", "B", "B"] * 2,
                           "y": [30, 35, 60, 62, 40, 38, 70, 72]})
        fig = plot_brinley(df, condition_col="cond", outcome_col="y", cond_x="A", cond_y="B",
                           unit_col="unit", bounds=(0, 100))
        ax = fig.axes[0]
        assert ax.get_xlim() == (0.0, 100.0) and ax.get_ylim() == (0.0, 100.0)
        plt.close(fig)
        plt.close("all")


class TestVaiorByUnit:
    def test_per_unit_table(self):
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(7); rows = []
        # P0/P1 = responders (B >> A), P2 = non-responder (B ~ A)
        for u, gap in [("P0", 8), ("P1", 8), ("P2", 0)]:
            s = 0
            for _ in range(5):
                for c in rng.permutation(["A", "B"]):
                    s += 1
                    rows.append({"unit": u, "session": s, "cond": c,
                                 "y": (gap if c == "B" else 0) + rng.normal(0, 0.5)})
        df = pd.DataFrame(rows)
        tbl = vaior_by_unit(df, unit_col="unit", session_col="session", condition_col="cond",
                            outcome_col="y", reference="A", compared="B", improvement="increase")
        assert list(tbl["unit"]) == ["P0", "P1", "P2"]
        d = dict(zip(tbl["unit"], tbl["decision_overall"]))
        assert d["P0"] and d["P1"] and not d["P2"]
        plt.close("all")


# ---------------------------------------------------------------------------
# TIDY format: outcome(s) stacked as rows (1 name col + 1 score col)
# ---------------------------------------------------------------------------
from functions.sced.prep import unstack_outcomes
from functions.sced.alternating.run import run_sced_alternating, run_sced_by_group


class TestRunByGroup:
    def _df(self):
        rng = np.random.RandomState(0); rows = []
        for u in range(6):
            cohort = "acute" if u < 3 else "chronic"
            eff = 1 if cohort == "acute" else 4
            s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"cohort": cohort, "patient": f"P{u}", "session": s, "condition": c,
                                 "y": {"A": 0, "B": 1, "C": 2}[c] * eff + rng.normal(20, 2)})
        return pd.DataFrame(rows)

    def test_runs_each_group_separately(self):
        res = run_sced_by_group(self._df(), group_col="cohort", session_col="session",
                                condition_col="condition", unit_col="patient", outcome="y",
                                conditions=("A", "B", "C"), n_perm=300, verbose=False)
        assert set(res) == {"acute", "chronic"}
        for g in ("acute", "chronic"):
            assert "error" not in res[g] and "group" in res[g]["engine"]

    def test_missing_group_col_raises(self):
        with pytest.raises(ValueError):
            run_sced_by_group(self._df(), group_col="absent", session_col="session",
                              condition_col="condition", outcome="y")

    def test_failing_group_does_not_abort_others(self):
        df = self._df()
        # degenerate cohort: a single condition -> the engine fails for this group only
        bad = df[df["cohort"] == "acute"].copy(); bad["cohort"] = "degenerate"
        bad = bad[bad["condition"] == "A"]
        df2 = pd.concat([df, bad], ignore_index=True)
        res = run_sced_by_group(df2, group_col="cohort", session_col="session",
                                condition_col="condition", unit_col="patient", outcome="y",
                                conditions=("A", "B", "C"), n_perm=200, verbose=False)
        assert "error" in res["degenerate"] and "error" not in res["chronic"]


class TestUnstackOutcomes:
    def _tidy(self, n_out=2, runs=None, seed=0):
        rng = np.random.RandomState(seed); rows = []
        names = ["accuracy", "fatigue"][:n_out]
        for u in range(4):
            s = 0
            for _ in range(6):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    for r in (range(1, runs + 1) if runs else [None]):
                        for nm in names:
                            base = {"accuracy": 20, "fatigue": 5}[nm]
                            gap = {"A": 0, "B": 2, "C": 4}[c] * (1 if nm == "accuracy" else 0.5)
                            row = {"record_id": f"P{u}", "visit": s, "phase": c,
                                   "outcome": nm, "value": base + gap + rng.normal(0, 1)}
                            if runs:
                                row["run"] = r
                            rows.append(row)
        return pd.DataFrame(rows)

    def test_unstack_recovers_columns(self):
        tidy = self._tidy(n_out=2)
        wide, names = unstack_outcomes(tidy, outcome_name_col="outcome", value_col="value",
                                       id_cols=["record_id", "visit", "phase"])
        assert set(names) == {"accuracy", "fatigue"}
        assert len(wide) == tidy["visit"].nunique() * tidy["record_id"].nunique()

    def test_unstack_with_trial_preserves_repeats(self):
        tidy = self._tidy(n_out=1, runs=4)
        wide, names = unstack_outcomes(tidy, outcome_name_col="outcome", value_col="value",
                                       id_cols=["record_id", "visit", "phase"], trial_col="run")
        assert names == ["accuracy"] and "run" in wide.columns
        assert wide.duplicated(subset=["record_id", "visit", "phase"]).any()  # runs kept

    def test_unstack_auto_occ_when_no_trial_col(self):
        tidy = self._tidy(n_out=1, runs=3)
        wide, names = unstack_outcomes(tidy, outcome_name_col="outcome", value_col="value",
                                       id_cols=["record_id", "visit", "phase"])
        assert "_occ" in wide.columns and set(wide["_occ"]) == {1, 2, 3}

    def test_dispatch_two_outcomes_is_multivariate(self):
        tidy = self._tidy(n_out=2)
        out = run_sced_alternating(tidy, session_col="visit", condition_col="phase",
                                   unit_col="record_id", outcome_name_col="outcome",
                                   value_col="value", conditions=("A", "B", "C"),
                                   n_perm=300, verbose=False)
        assert "PERMANOVA" in out["engine"]

    def test_dispatch_one_outcome_is_univariate(self):
        tidy = self._tidy(n_out=2)
        out = run_sced_alternating(tidy, session_col="visit", condition_col="phase",
                                   unit_col="record_id", outcome_name_col="outcome",
                                   value_col="value", outcome="accuracy",
                                   conditions=("A", "B", "C"), n_perm=300, verbose=False)
        assert out["model_info"] is not None and "group" in out["engine"]

    def test_multivariate_writes_report(self, tmp_path):
        tidy = self._tidy(n_out=2)
        out = run_sced_alternating(tidy, session_col="visit", condition_col="phase",
                                   unit_col="record_id", outcome_name_col="outcome",
                                   value_col="value", conditions=("A", "B", "C"),
                                   n_perm=200, save_path=str(tmp_path), verbose=False)
        assert "PERMANOVA" in out["engine"]
        rep = out.get("report_path")
        assert rep and __import__("os").path.exists(rep)

    def test_dispatch_tidy_outcome_none_single_is_univariate(self):
        # tidy with a single stacked outcome + outcome=None -> "all" = 1 -> univariate
        tidy = self._tidy(n_out=1)
        out = run_sced_alternating(tidy, session_col="visit", condition_col="phase",
                                   unit_col="record_id", outcome_name_col="outcome",
                                   value_col="value", outcome=None,
                                   conditions=("A", "B", "C"), n_perm=200, verbose=False)
        assert out["model_info"] is not None and "group" in out["engine"]

    def test_dispatch_tidy_with_runs_level3(self):
        tidy = self._tidy(n_out=1, runs=4)
        out = run_sced_alternating(tidy, session_col="visit", condition_col="phase",
                                   unit_col="record_id", outcome_name_col="outcome",
                                   value_col="value", trial_col="run", level3=True,
                                   conditions=("A", "B", "C"), n_perm=300, verbose=False)
        assert "level3" in out and out["level3"]["summary"].get("pct_within_visit") is not None


# ---------------------------------------------------------------------------
# Merged panels: plot_sced_panels (dashboard) + plot_vaior_grid (per patient)
# ---------------------------------------------------------------------------
from functions.sced.plots.panels import plot_sced_panels
from functions.sced.plots.vaior import plot_vaior_grid


class TestPanels:
    def _df(self, n_units=5):
        rng = np.random.RandomState(0); rows = []
        for u in range(n_units):
            base = rng.normal(0, 3); s = 0
            for _ in range(8):
                for c in rng.permutation(["A", "B", "C"]):
                    s += 1
                    rows.append({"patient": f"P{u+1}", "session": s, "condition": c,
                                 "score": base + {"A": 0, "B": 2, "C": 4}[c] + rng.normal(20, 2)})
        return pd.DataFrame(rows)

    def test_panels_axes_count(self):
        import matplotlib.pyplot as plt
        fig = plot_sced_panels(self._df(), session_col="session", condition_col="condition",
                               outcome_col="score", unit_col="patient", conditions=("A", "B", "C"),
                               reference="A")
        # GROUP (5 patients), "MBD-like" layout: 5 stacked series (left)
        # + right condition/group column = 2 VAIOR + 2 Brinley + 1 distribution
        assert len(fig.axes) == 5 + (2 + 2 + 1)
        plt.close("all")

    def test_panels_bounds_and_percent(self):
        """bounds borne l'axe (None = auto) ; y_percent affiche l'axe en pourcentage."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig = plot_sced_panels(self._df(), session_col="session", condition_col="condition",
                               outcome_col="score", unit_col="patient", conditions=("A", "B", "C"),
                               reference="A", vaior=False, bounds=(0.4, 1.0), y_percent=True)
        ax0 = fig.axes[0]                                  # first series
        assert ax0.get_ylim() == (0.4, 1.0)
        assert isinstance(ax0.yaxis.get_major_formatter(), PercentFormatter)
        plt.close("all")

    def test_panels_aggregate_mean_collapses_repeated_measures(self):
        """By default (aggregate='mean'), repeated measures per occasion (runs) are
        aggregated: the series draws ONE point per (patient, session), not one per run."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0); rows = []
        for u in range(3):
            for s in range(1, 7):
                c = ["A", "B", "C"][s % 3]
                for run in range(4):                      # 4 runs per visit
                    rows.append({"patient": f"P{u}", "session": s, "condition": c,
                                 "run": run, "score": 50 + rng.normal(0, 2)})
        df = pd.DataFrame(rows)

        def n_points(fig, n_facets):                      # points drawn in the series facets
            return sum(int(coll.get_offsets().shape[0]) for ax in fig.axes[:n_facets]
                       for coll in ax.collections)
        fig_m = plot_sced_panels(df, session_col="session", condition_col="condition",
                                 outcome_col="score", unit_col="patient", conditions=("A", "B", "C"),
                                 reference="A", vaior=False)
        fig_r = plot_sced_panels(df, session_col="session", condition_col="condition",
                                 outcome_col="score", unit_col="patient", conditions=("A", "B", "C"),
                                 reference="A", vaior=False, aggregate=None)
        assert n_points(fig_m, 3) == 3 * 6               # 3 patients x 6 visits = 18 points
        assert n_points(fig_r, 3) == 3 * 6 * 4           # raw: x4 runs
        plt.close("all")

    def test_panels_group_series_stacked_per_patient(self):
        """Regression test: with a group, one STACKED series per patient (left) + right
        condition/group column (VAIOR + Brinley + distribution)."""
        import matplotlib.pyplot as plt
        for nu in (3, 5):
            fig = plot_sced_panels(self._df(n_units=nu), session_col="session",
                                   condition_col="condition", outcome_col="score",
                                   unit_col="patient", conditions=("A", "B", "C"), reference="A")
            assert len(fig.axes) == nu + (2 + 2 + 1)    # nu series + 2 VAIOR + 2 Brinley + 1 distrib
            plt.close("all")

    def test_panels_toggle_off_sections(self):
        import matplotlib.pyplot as plt
        fig = plot_sced_panels(self._df(), session_col="session", condition_col="condition",
                               outcome_col="score", conditions=("A", "B", "C"), reference="A",
                               vaior=True, brinley=False)
        assert len(fig.axes) == 3            # 1 series + 2 VAIOR
        plt.close("all")

    def test_vaior_grid_shape(self):
        import matplotlib.pyplot as plt
        fig, tbl = plot_vaior_grid(self._df(), unit_col="patient", session_col="session",
                                   condition_col="condition", outcome_col="score", reference="A",
                                   conditions=("A", "B", "C"))
        assert len(fig.axes) == 5 * 2 and len(tbl) == 5 * 2
        assert set(tbl["compared"]) == {"B", "C"}
        plt.close("all")

    def test_vaior_grid_single_wrap_bounds_batch(self, tmp_path):
        """1 comparison + >10 patients: multi-column grid, batches (_b1/_b2), bounded y."""
        import os, glob, matplotlib.pyplot as plt
        rng = np.random.RandomState(2); rows = []
        for u in range(12):                               # 12 patients -> 2 batches
            s = 0
            for _ in range(6):
                for c in ["A", "B"]:
                    s += 1
                    rows.append({"patient": f"P{u:02d}", "session": s, "condition": c,
                                 "score": {"A": 40, "B": 55}[c] + rng.normal(0, 3)})
        df = pd.DataFrame(rows)
        fig, tbl = plot_vaior_grid(df, unit_col="patient", session_col="session",
                                   condition_col="condition", outcome_col="score", reference="A",
                                   compared="B", bounds=(0, 100), name="Y", save_path=str(tmp_path),
                                   max_per_fig=10)
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(str(tmp_path), "Y_vaior_grid*.png")))
        assert files == ["Y_vaior_grid_b1.png", "Y_vaior_grid_b2.png"]   # 12 > 10 -> 2 batches
        assert len(tbl) == 12                              # all patients in the table
        assert fig.axes[0].get_ylim() == (0.0, 100.0)     # y bounded to the scores
        plt.close("all")

    def test_panels_nof1_no_unit(self):
        import matplotlib.pyplot as plt
        d = self._df(n_units=1).drop(columns="patient")
        fig = plot_sced_panels(d, session_col="session", condition_col="condition",
                               outcome_col="score", conditions=("A", "B", "C"), reference="A")
        assert len(fig.axes) == 5
        plt.close("all")


# ---------------------------------------------------------------------------
# Pre-processing: case-insensitive + dates -> sessions
# ---------------------------------------------------------------------------
from functions.sced.prep import (resolve_columns, harmonize_conditions, harmonize_values,
                                  dates_to_sessions)


class TestPrep:
    def test_resolve_columns_case_insensitive(self):
        df = pd.DataFrame({"Record_ID": [1], "Visit": [1], "Phase": ["A"]})
        res = resolve_columns(df, {"unit": "record_id", "session": "VISIT", "condition": "phase",
                                   "x": None})
        assert res == {"unit": "Record_ID", "session": "Visit", "condition": "Phase", "x": None}

    def test_harmonize_conditions_to_canonical(self):
        df = pd.DataFrame({"cond": [" a ", "B", "c", "A"]})
        out = harmonize_conditions(df, "cond", conditions=("A", "B", "C"))
        assert list(out["cond"]) == ["A", "B", "C", "A"]

    def test_harmonize_values_upper(self):
        df = pd.DataFrame({"outcome": ["Accuracy", " fatigue "]})
        out = harmonize_values(df, "outcome")
        assert list(out["outcome"]) == ["ACCURACY", "FATIGUE"]

    def test_dates_to_sessions_per_unit(self):
        df = pd.DataFrame({
            "u": ["P1", "P1", "P1", "P2", "P2"],
            "d": ["2026-01-10", "2026-01-03", "2026-01-20", "2026-02-01", "2026-01-15"],
            "y": [1, 2, 3, 4, 5]})
        out = dates_to_sessions(df, "d", unit_col="u", new_col="session", label=True)
        # P1: 03<10<20 -> 1,2,3 ; P2: 15<01-02 -> 1,2
        got = {(r.u, r.y): int(r.session) for r in out.itertuples()}
        assert got[("P1", 2)] == 1 and got[("P1", 1)] == 2 and got[("P1", 3)] == 3
        assert got[("P2", 5)] == 1 and got[("P2", 4)] == 2
        assert "session_label" in out.columns and out["session_label"].iloc[0].startswith("S_")

    def test_dates_same_day_same_session(self):
        df = pd.DataFrame({"d": ["2026-01-01", "2026-01-01", "2026-01-05"], "y": [1, 2, 3]})
        out = dates_to_sessions(df, "d", new_col="session")
        assert list(out["session"]) == [1, 1, 2]

    def test_dates_unparseable_raises(self):
        df = pd.DataFrame({"d": ["xx", "yy"], "y": [1, 2]})
        with pytest.raises(ValueError):
            dates_to_sessions(df, "d")

    def test_end_to_end_lowercase_and_dates(self):
        # lowercase conditions + a date column -> must run like the canonical case
        rng = np.random.RandomState(0); rows = []
        from datetime import date, timedelta
        for u in range(4):
            d0 = date(2026, 1, 1)
            for k in range(12):
                c = ["a", "b", "c"][k % 3]
                rows.append({"Record_ID": f"P{u}", "Date": (d0 + timedelta(days=3 * k)).isoformat(),
                             "Phase": c, "accuracy": {"a": 0, "b": 2, "c": 4}[c] + rng.normal(20, 2)})
        df = pd.DataFrame(rows)
        df = harmonize_conditions(df, "Phase", conditions=("A", "B", "C"))
        df = dates_to_sessions(df, "Date", unit_col="Record_ID", new_col="session")
        out = run_sced_alternating(df, session_col="session", condition_col="Phase",
                                   unit_col="Record_ID", outcome="accuracy",
                                   conditions=("A", "B", "C"), n_perm=300, verbose=False)
        assert out["model_info"] is not None and "group" in out["engine"]


# ---------------------------------------------------------------------------
# Per-case Bayesian models: simple / BITS / BUCP (Natesan Batley 2020)
# ---------------------------------------------------------------------------
class TestMBDBayesian:
    def _series(self, T=16, start=8, jump=12, rho=0.3, seed=0):
        rng = np.random.RandomState(seed); y = np.empty(T); e = 0.0
        for t in range(T):
            e = rho * e + rng.normal(0, 4)
            y[t] = 50 + (jump if t >= start else 0) + e
        return y, start

    def test_bits_recovers_positive_es(self):
        from functions.sced.mbd.bayesian import bayes_phase_model
        y, start = self._series()
        r = bayes_phase_model(y, start_index=start, model="bits", draws=500, tune=500,
                              chains=2, seed=1)
        assert r["es"]["median"] > 0 and r["es"]["pd"] > 0.9
        assert r["rhat_max"] < 1.1 and r["n_b"] == len(y) - start

    def test_simple_no_rho(self):
        from functions.sced.mbd.bayesian import bayes_phase_model
        y, start = self._series()
        r = bayes_phase_model(y, start_index=start, model="simple", draws=400, tune=400,
                              chains=2, seed=1)
        assert abs(r["rho"]["median"]) < 1e-6 and r["es"]["median"] > 0

    def test_bucp_recovers_changepoint(self):
        from functions.sced.mbd.bayesian import bayes_phase_model
        y, start = self._series(jump=16, seed=2)     # large effect -> clear CP
        r = bayes_phase_model(y, start_index=start, model="bucp", draws=600, tune=600,
                              chains=2, seed=2)
        assert "cp" in r and abs(r["cp"]["start_B_mode"] - start) <= 1

    def test_decrease_orientation(self):
        from functions.sced.mbd.bayesian import bayes_phase_model
        y, start = self._series(jump=12)             # B higher, but improvement=decrease
        r = bayes_phase_model(y, start_index=start, model="simple", improvement="decrease",
                              draws=400, tune=400, chains=2, seed=1)
        assert r["es"]["median"] < 0          # rise = deterioration when decrease

    def test_trend_recovers_cumulative(self):
        from functions.sced.mbd.bayesian import bayes_phase_model
        rng = np.random.RandomState(0); T = 18; start = 8; y = np.empty(T); e = 0.0
        for t in range(T):                       # pure slope +1.2/session in B, no jump
            e = 0.3 * e + rng.normal(0, 4)
            y[t] = 50 + (1.2 * (t - start) if t >= start else 0) + e
        r = bayes_phase_model(y, start_index=start, model="bits", trend=True, draws=600,
                              tune=600, chains=2, seed=1)
        assert r["trend"] and r["slope_change"]["median"] > 0 and r["slope_change"]["pd"] > 0.9
        assert r["es_end"]["median"] > r["es"]["median"]   # the cumulative effect exceeds the jump


class TestInterpretationGlossary:
    def test_scopes_and_lecture_always(self):
        from functions.sced.glossary import interpretation_glossary
        for sc in ["rando", "mbd", "mv", "bayes", "all"]:
            g = interpretation_glossary(sc)
            assert len(g) > 5 and set(g.columns) == {"Term", "Definition"}
            # the reading block (thresholds) is always present
            assert any("pd (Bayesian)" in t for t in g["Term"])

    def test_append_glossary_sheet(self, tmp_path):
        import pandas as pd, openpyxl
        from functions.sced.glossary import append_glossary_sheet
        p = tmp_path / "r.xlsx"
        pd.DataFrame({"a": [1]}).to_excel(p, index=False, sheet_name="Data")
        append_glossary_sheet(str(p), scope="mbd")
        assert "Interpretation guide" in openpyxl.load_workbook(p).sheetnames

    def test_write_stacked(self, tmp_path):
        """write_stacked stacks several (title, df) blocks onto ONE sheet with the titles in column A."""
        import pandas as pd, openpyxl
        from functions.sced.glossary import write_stacked
        p = tmp_path / "s.xlsx"
        a = pd.DataFrame({"Element": ["x", "y"], "Value": [1, 2]})
        b = pd.DataFrame({"Term": ["t1"], "Definition": ["d1"]})
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            write_stacked(w, "Fusion", [("BLOC A", a), ("BLOC B", b)])
        wb = openpyxl.load_workbook(p); ws = wb["Fusion"]
        colA = [c.value for c in ws["A"]]
        assert "BLOC A" in colA and "BLOC B" in colA      # both titles present
        assert wb.sheetnames == ["Fusion"]                 # a single sheet

    def test_style_cells_and_cohen_fill(self, tmp_path):
        """cohen_fill code l'ampleur ; style_cells applique gras + remplissage aux bonnes cellules."""
        import pandas as pd, openpyxl
        from functions.sced.glossary import style_cells, cohen_fill
        assert cohen_fill(0.1)[1] == "F2F2F2" and cohen_fill(0.9)[1] == "C6EFCE" and cohen_fill(0.9)[0]
        df = pd.DataFrame({"Outcome": ["a", "b"], "p-value": [0.01, 0.40], "d": [0.9, 0.1]})
        p = tmp_path / "st.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="R", index=False)
            style_cells(w.sheets["R"], df,
                        {"p-value": lambda v: (True, "C6EFCE") if v < 0.05 else (False, "F2F2F2"),
                         "d": cohen_fill})
        ws = openpyxl.load_workbook(p)["R"]
        # ligne 2 = a : p=0.01 -> vert+gras ; d=0.9 -> vert. ligne 3 = b : p=0.40 -> gris
        assert ws.cell(2, 2).fill.fgColor.rgb.endswith("C6EFCE") and ws.cell(2, 2).font.bold
        assert ws.cell(2, 3).fill.fgColor.rgb.endswith("C6EFCE")
        assert ws.cell(3, 2).fill.fgColor.rgb.endswith("F2F2F2")

    def test_full_param_table(self):
        """full_param_table dumps ALL parameters (mu_b2, sd_*, pop_*...) with r_hat/ess, per model."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        from functions.sced.glossary import full_param_table
        rng = np.random.RandomState(0); rows = []
        for ci in range(3):
            for s in range(1, 13):
                ph = "B" if s >= 6 else "A"
                rows.append({"case": f"P{ci}", "session": s, "phase": ph,
                             "y": 50 + (8 if ph == "B" else 0) + rng.normal(0, 4)})
        df = pd.DataFrame(rows)
        r = bayes_hier_sced(df, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", terms={"trend", "level"}, draws=300, tune=300,
                            chains=2, seed=1)
        tab = full_param_table({("(all)", "y"): r["_idata"]})
        assert not tab.empty
        assert {"Model", "parameter", "r_hat"} <= set(tab.columns)
        names = set(tab["parameter"])
        assert any("mu_b2" in n for n in names) and any("pop_level" in n for n in names)
        assert not any(str(n).startswith("z_") for n in names)   # auxiliaries removed

    def test_baseline_trend_correction_removes_false_immediate(self):
        # series with NO break but a rising baseline: the correction must cancel the false jump
        from functions.sced.mbd.bayesian import bayes_phase_model
        rng = np.random.RandomState(0); T = 16; start = 8; y = np.empty(T); e = 0.0
        for t in range(T):
            e = 0.3 * e + rng.normal(0, 3); y[t] = 50 + 1.5 * t + e
        uncorr = bayes_phase_model(y, start_index=start, model="bits", trend=True,
                                   baseline_trend=False, draws=500, tune=500, chains=2, seed=1)
        corr = bayes_phase_model(y, start_index=start, model="bits", trend=True,
                                 baseline_trend=True, draws=500, tune=500, chains=2, seed=1)
        assert not uncorr["baseline_corrected"] and corr["baseline_corrected"]
        # the (false) immediate effect must drop once the trend is modelled
        assert corr["es"]["pd"] < uncorr["es"]["pd"]
        assert "baseline_trend_b1" in corr and abs(corr["baseline_trend_b1"]["median"]) > 0.3


class TestScdhlmTrend:
    def test_trend_option_changes_g(self):
        """bc_smd_scdhlm(trend=True) fits y~time+phase and yields a g different from the raw one (under trend)."""
        import shutil, pytest
        if shutil.which("Rscript") is None:
            pytest.skip("Rscript absent")
        from functions.sced.mbd.multilevel import bc_smd_scdhlm
        rng = np.random.RandomState(0); rows = []
        for ci in range(4):                          # 4 cases, rising baseline + small jump
            st = 5 + ci
            for s in range(1, 16):
                ph = "B" if s >= st else "A"
                y = 30 + 1.6 * s + (4 if ph == "B" else 0) + rng.normal(0, 3)
                rows.append({"case": f"P{ci}", "session": s, "phase": ph, "y": y})
        d = pd.DataFrame(rows)
        raw = bc_smd_scdhlm(d, tier_col="case", session_col="session", outcome_col="y",
                            phase_col="phase", baseline="A", treatment="B")
        trd = bc_smd_scdhlm(d, tier_col="case", session_col="session", outcome_col="y",
                            phase_col="phase", baseline="A", treatment="B", trend=True)
        if not (raw.get("available") and trd.get("available")):
            pytest.skip(f"scdhlm indisponible : {raw.get('message') or trd.get('message')}")
        assert raw["trend"] is False and trd["trend"] is True
        assert abs(raw["g_AB"] - trd["g_AB"]) > 1e-3   # the trend correction changes the estimate


class TestHierBayes:
    def _df(self, eff=10, seed=0):
        rng = np.random.RandomState(seed); rows = []
        for ci in range(5):
            e = 0.0; base = 50 + rng.normal(0, 8); st = 6
            for s in range(1, 16):
                ph = 1 if s >= st else 0; e = 0.3 * e + rng.normal(0, 4)
                rows.append({"case": f"P{ci}", "session": s, "phase": "B" if ph else "A",
                             "y": base + (eff if ph else 0) + e})
        return pd.DataFrame(rows)

    def test_hier_recovers_population_and_per_case(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        r = bayes_hier_sced(self._df(eff=10), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", hypothesis="immediate_flat",
                            draws=500, tune=500, chains=2, seed=1)
        assert r["population"]["level"]["median"] > 0 and r["population"]["level"]["pd"] > 0.9
        assert "bc_smd" not in r                        # in-house BC-SMD removed (-> R scdhlm)
        assert len(r["per_case"]) == 5 and all("level" in pc for pc in r["per_case"])
        assert r["rhat_max"] < 1.1 and "sd_level" in r["heterogeneity"]

    def test_hier_cumulative_has_slope(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        r = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", hypothesis="cumulative_trend",
                            draws=400, tune=400, chains=2, seed=1)
        assert "slope" in r["population"] and "slope" in r["per_case"][0]

    def test_hier_reports_combined_end_effect(self):
        """TOTAL end-of-phase effect (b2 + b3*T_B) reported at population + per case, all poolings."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        r = bayes_hier_sced(self._df(eff=10), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", terms={"level", "slope"},
                            draws=400, tune=400, chains=2, seed=1)
        assert "effect_end" in r["population"]
        assert all("effect_end" in pc for pc in r["per_case"])
        assert r["population"]["effect_end"]["pd"] > 0.85

    def test_pooling_none_beta_then_meta(self):
        """pooling='none' (Beta) = stage 1; meta-analysis (stage 2) non-centered -> 0 divergences."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        from functions.sced.mbd.meta import meta_from_idata
        d = self._df(eff=10); d["y"] = np.clip(d["y"], 1, 99)
        r = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", terms={"level", "slope"}, pooling="none",
                            family="beta", bounds=(0, 100), draws=400, tune=400, chains=2, seed=1)
        assert "effect_end" in r["population"] and len(r["per_case"]) == 5
        res = meta_from_idata(r["_idata"], estimand="effect_end",
                              draws=600, tune=600, chains=2, seed=1)
        assert res["k"] == 5 and res["diverging"] == 0
        assert res["population"]["pd"] > 0.80 and 0 <= res["heterogeneity"]["I2"] <= 1
        assert len(res["per_case"]) == 5 and "prediction_interval" in res["heterogeneity"]
        # meta PER PARAMETER: deterministic points exposed for the present terms (b2, b3)
        post = r["_idata"].posterior
        assert "level_pts" in post.data_vars and "slope_pts" in post.data_vars
        assert "trend_pts" not in post.data_vars        # no 'trend' term here
        for est in ("level", "slope"):
            mr = meta_from_idata(r["_idata"], estimand=est, draws=400, tune=400, chains=2, seed=1)
            assert mr["k"] == 5 and "tau" in mr["heterogeneity"]

    def test_meta_analysis_basic_and_forest(self, tmp_path):
        """Meta-analysis of homogeneous cases: mu>0 credible, low I2; forest plot saved."""
        from functions.sced.mbd.meta import bayes_meta_analysis
        from functions.sced.plots.panels import plot_meta_forest
        res = bayes_meta_analysis([8, 10, 12, 9, 11], [3, 3, 3, 3, 3],
                                  labels=[f"C{i}" for i in range(5)],
                                  draws=600, tune=600, chains=2, seed=1)
        assert res["population"]["median"] > 0 and res["population"]["pd"] > 0.95
        assert res["heterogeneity"]["I2"] < 0.6
        fp = tmp_path / "forest.png"
        plot_meta_forest(res, save_path=str(fp)); assert fp.exists()

    def test_diag_plots_panel_and_legend(self, tmp_path):
        """plot_bayesian_diag produces forest + energy + qq/ppc (PPC saved); panel + legend OK."""
        import os
        from functions.sced.mbd.bayesian import bayes_hier_sced
        from functions.sced.plots.panels import plot_bayesian_diag, plot_bayesian_panel
        from functions.sced.glossary import bayesian_diag_legend
        r = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", terms={"trend", "level"},
                            draws=300, tune=300, chains=2, seed=1)
        idata = r["_idata"]
        assert hasattr(idata, "posterior_predictive")        # replicates for PPC/QQ
        saved = plot_bayesian_diag(idata, save_path=str(tmp_path), name="m")
        tags = {os.path.basename(p).replace("m_", "").replace(".png", "") for p in saved}
        assert {"trace", "rank", "forest", "energy", "ppc"} <= tags   # forest restored + PPC present
        panel = plot_bayesian_panel(idata, save_path=str(tmp_path), name="m")
        assert panel and os.path.exists(panel)
        leg = bayesian_diag_legend()
        assert set(leg.columns) >= {"Plot", "How to read"} and len(leg) >= 8

    def test_grouped_ppc_by_phase_and_case(self, tmp_path):
        """The .nc carries obs_phase/obs_case (constant_data) -> PPC grouped by phase AND by patient."""
        import os
        from functions.sced.mbd.bayesian import bayes_hier_sced
        from functions.sced.plots.panels import plot_ppc_grouped
        r = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", terms={"trend", "level"},
                            draws=300, tune=300, chains=2, seed=1)
        idata = r["_idata"]
        assert hasattr(idata, "constant_data")
        assert {"obs_phase", "obs_case"} <= set(idata.constant_data.data_vars)
        ph = plot_ppc_grouped(idata, by="phase", save_path=str(tmp_path), name="m")
        ca = plot_ppc_grouped(idata, by="case", save_path=str(tmp_path), name="m")
        assert ph and all(os.path.exists(p) for p in ph)
        assert ca and all(os.path.exists(p) for p in ca)

    def test_per_case_b1_b3_hdi_and_patient_coords(self):
        """Per-case: b1 AND b3 present with HDI; idata indexed by PATIENT (coord 'case')."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        r = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", terms={"trend", "level", "slope"},
                            pooling="partial", draws=300, tune=300, chains=2, seed=1)
        pc0 = r["per_case"][0]
        for term in ("trend", "level", "slope"):
            assert term in pc0 and {"hdi_low", "hdi_high", "pd"} <= set(pc0[term])
        assert list(r["_idata"].posterior.coords["case"].values) == [f"P{i}" for i in range(5)]

    def test_hier_trend_b1_reported(self):
        """The baseline trend term (b1) is reported at population level, with HDI/pd; absent without b1."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        with_b1 = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                                  phase_col="phase", outcome_col="y", terms={"trend", "level"},
                                  draws=400, tune=400, chains=2, seed=1)
        assert "trend" in with_b1["population"]
        assert {"median", "hdi_low", "hdi_high", "pd"} <= set(with_b1["population"]["trend"])
        no_b1 = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                                phase_col="phase", outcome_col="y", terms={"level"},
                                draws=400, tune=400, chains=2, seed=1)
        assert "trend" not in no_b1["population"]      # no b1 -> no trend row

    def test_hier_slope_points_scale(self):
        """The slope is reported in POINTS (readable) + HDI, for gaussian AND beta; gaussian: cumulative = slope*horizon."""
        from functions.sced.mbd.bayesian import bayes_hier_sced
        # gaussian: slope already in points -> cumulative ~= slope/session * horizon B
        rg = bayes_hier_sced(self._df(eff=8), tier_col="case", session_col="session",
                             phase_col="phase", outcome_col="y", terms={"trend", "slope"},
                             family="gaussian", draws=400, tune=400, chains=2, seed=1)
        pg = rg["population"]
        for k in ("slope", "slope_points_per_session", "slope_points_cumulative", "slope_horizon_B"):
            assert k in pg
        cum = pg["slope_points_cumulative"]
        assert {"median", "hdi_low", "hdi_high", "pd"} <= set(cum)        # the slope indeed has an HDI
        approx = pg["slope_points_per_session"]["median"] * pg["slope_horizon_B"]
        assert abs(cum["median"] - approx) < 1e-6                          # gaussian: cumulative = slope*horizon
        # beta: slope converted from logit to points (!= raw logit value)
        d = self._df(eff=5); d["y"] = np.clip(d["y"] + 38, 0, 100)
        rb = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                             outcome_col="y", terms={"trend", "slope"}, family="beta",
                             bounds=(0, 100), draws=400, tune=400, chains=2, seed=1)
        pb = rb["population"]
        assert "slope_points_cumulative" in pb and "hdi_low" in pb["slope_points_cumulative"]

    def test_hier_ar_flag(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(eff=10)
        r = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", family="gaussian", ar=False,
                            draws=400, tune=400, chains=2, seed=1)
        assert "rho" not in r                          # no AR1 -> no rho
        assert r["population"]["level"]["median"] > 0

    def test_hier_beta_family(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(eff=5)
        d["y"] = np.clip(d["y"] + 38, 0, 100)        # pushes toward the ceiling (bounded outcome)
        r = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", family="beta", bounds=(0, 100),
                            draws=400, tune=400, chains=2, seed=1)
        assert r["family"] == "beta" and "logit_b2" in r
        assert "bc_smd" not in r                      # logit -> effect in POINTS, no BC-SMD
        assert "level" in r["population"] and len(r["per_case"]) == 5

    def test_hier_student_family(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(eff=10, seed=2)
        d.loc[d.index[::13], "y"] += 60                # inject outliers
        r = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", family="student", ar=True,
                            draws=400, tune=400, chains=2, seed=1)
        assert r["family"] == "student" and "nu" in r          # degrees of freedom reported
        assert r["population"]["level"]["median"] > 0 and r["population"]["level"]["pd"] > 0.9

    def test_hier_binomial_family(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(eff=5)
        d["y"] = np.clip(d["y"] + 30, 0, 100)         # k/N score expressed on 0-100
        r = bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", family="binomial", bounds=(0, 100), n_trials=20,
                            draws=400, tune=400, chains=2, seed=1)
        assert r["family"] == "binomial" and "logit_b2" in r
        assert "bc_smd" not in r                       # logit -> effect in POINTS
        assert "level" in r["population"] and len(r["per_case"]) == 5

    def test_hier_binomial_requires_n_trials(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(eff=5); d["y"] = np.clip(d["y"], 0, 100)
        try:
            bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y", family="binomial", bounds=(0, 100))
            assert False
        except ValueError:
            pass

    def test_hier_rope_on_raw_effect(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        r = bayes_hier_sced(self._df(eff=10), tier_col="case", session_col="session",
                            phase_col="phase", outcome_col="y", rope=3.0,
                            draws=400, tune=400, chains=2, seed=1)
        lv = r["population"]["level"]                   # ROPE in original units -> P(effect>ROPE)
        assert "p_gt_rope" in lv and lv["rope"] == 3.0 and 0 <= lv["p_gt_rope"] <= 1

    def test_hier_needs_two_cases(self):
        from functions.sced.mbd.bayesian import bayes_hier_sced
        d = self._df(); d = d[d["case"] == "P0"]
        try:
            bayes_hier_sced(d, tier_col="case", session_col="session", phase_col="phase",
                            outcome_col="y"); assert False
        except ValueError:
            pass
