"""Tests - longitudinal multilevel pipeline, Phase 1 (Fatigue x Performance coupling).

Checks that the four engines recover a simulated within-cluster coupling (eva up, acc down),
and that the degenerate case (no coupling) does not fabricate an effect. MCMC kept short on purpose.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.general.longitudinal.growth import rmcorr, slope_vs_slope, fit_growth_lmm, fit_growth_bayes, vif_table


def _make(coupling=-0.35, n_pat=5, n_sess=18, seed=1, eva_noise=5.0):
    """5 patients x n_sess sessions x 4 blocks; eva rises with the block, acc = base - coupling*eva."""
    rng = np.random.RandomState(seed)
    rows = []
    for pi in range(n_pat):
        base = 70 + rng.normal(0, 5)
        for s in range(n_sess):
            sess = f"P{pi}_S{s}"; lvl = rng.normal(0, 2)
            for b in range(1, 5):
                eva = 30 + 7 * b + rng.normal(0, eva_noise)
                acc = base + lvl + coupling * eva + rng.normal(0, 3)
                rows.append({"patient": f"P{pi}", "session": sess, "bloc": b,
                             "eva": eva, "acc": acc})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def coupled():
    return _make(coupling=-0.35)


def test_rmcorr_recovers_negative_coupling(coupled):
    r = rmcorr(coupled, x="eva", y="acc", subject="session")
    assert r["available"]
    assert r["r_rm"] < -0.2                      # clear negative coupling
    assert r["ci95"][1] < 0                      # CI excludes 0
    assert r["slope"] < 0 and r["slope_ci"][1] < 0
    assert r["p"] < 0.05


def test_rmcorr_null_when_no_coupling():
    df = _make(coupling=0.0, seed=7)
    r = rmcorr(df, x="eva", y="acc", subject="session")
    assert abs(r["r_rm"]) < 0.2                  # no fabricated effect
    assert r["ci95"][0] < 0 < r["ci95"][1]       # CI contains 0


def test_slope_vs_slope(coupled):
    sv = slope_vs_slope(coupled, x="eva", y="acc", time="bloc", cluster="session")
    assert sv["available"]
    assert sv["var_slope_x"] > 0                 # variable fatigue slopes -> correlatable
    assert sv["n_clusters"] == coupled["session"].nunique()
    assert sv["r_slopes"] < 0                    # negatively coupled dynamics


def test_vif_detects_collinearity():
    vif = vif_table(_make(eva_noise=1.0), ["eva", "bloc"])    # eva near-monotone with bloc
    assert set(vif["Variable"]) == {"eva", "bloc"}
    assert float(vif.set_index("Variable").loc["eva", "VIF"]) > 1.0


def test_fit_growth_lmm(coupled):
    res, summ = fit_growth_lmm(coupled, outcome="acc", fixed=["eva", "C(bloc)"],
                               group="session", group_l3="patient", kr=False, vif_terms=["eva", "bloc"])
    assert res is not None
    eva = next(f for f in summ["fixed"] if f["term"] == "eva")
    assert eva["coef"] < 0 and eva["CI95"][1] < 0
    assert 0.0 <= summ["ICC"] <= 1.0
    assert summ["R2_marginal"] is not None


@pytest.mark.parametrize("family", ["gaussian", "beta"])
def test_fit_growth_bayes(coupled, family):
    r = fit_growth_bayes(coupled, outcome="acc", x="eva", cat=["bloc"], group="session",
                         group_l3="patient", family=family, rope="auto",
                         draws=200, tune=200, chains=2, seed=3, progressbar=False)
    b = r["b_x"]
    assert b["hdi_high"] < 0                     # credible negative effect
    assert b["pd"] > 0.9
    assert "tau_session" in r["variance"] and "tau_patient" in r["variance"]
    if family == "gaussian":
        assert r["b_x"].get("rope_decision", "").startswith("effect")


# --------------------------------------------------------------------------- #
#  Phase 2 - growth-curve
# --------------------------------------------------------------------------- #
from functions.general.longitudinal.growth import (growth_terms, add_within_between, fit_growth_curve,
                                           compare_growth_models, marginal_trajectory, blup_table)


def _make_growth(n=40, seed=0):
    rng = np.random.RandomState(seed); rows = []
    for s in range(n):
        a = rng.normal(50, 6); b = rng.normal(4, 1.5); arm = s % 2
        for t in range(6):
            y = a + b * t - 0.4 * t ** 2 + (2.0 * arm) * t + rng.normal(0, 2.5)
            rows.append({"subj": f"S{s}", "t": t, "y": y, "arm": f"arm{arm}"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def growth():
    return _make_growth()


def test_growth_terms():
    assert growth_terms("linear", "linear")[0] == ["linear"]
    assert growth_terms("t", "poly2")[0] == ["t", "I(t**2)"]
    assert growth_terms("t", "spline", knots=5)[0] == ["bs(t, df=5)"]
    assert growth_terms("t", "discrete")[0] == ["C(t)"]


def test_add_within_between(growth):
    d = add_within_between(growth, "y", "subj")
    assert {"y_within", "y_between"} <= set(d.columns)
    # within-subject mean of the within component ~ 0
    assert d.groupby("subj")["y_within"].mean().abs().max() < 1e-8


def test_fit_growth_curve_recovers_quadratic(growth):
    res, s = fit_growth_curve(growth, outcome="y", time="t", shape="poly2", by="arm",
                              group="subj", random="int_slope")
    assert res is not None
    quad = next(f for f in s["fixed"] if f["term"] == "I(t ** 2)")
    assert quad["coef"] < 0                       # negative curvature (true = -0.4)
    assert 0.0 <= s["ICC"] <= 1.0
    assert s["R2_conditional"] >= s["R2_marginal"]


def test_compare_growth_models(growth):
    cmp = compare_growth_models(growth, outcome="y", time="t", group="subj",
                                shapes=("linear", "poly2"), random="int_slope")
    assert "best" in cmp.columns and cmp["best"].sum() == 1
    poly = cmp[cmp["shape"].str.contains("quad")]
    assert float(poly["LRT_p"].iloc[0]) < 0.05    # quadratic significantly better


def test_marginal_trajectory(growth):
    res, _ = fit_growth_curve(growth, outcome="y", time="t", shape="poly2", by="arm",
                              group="subj", random="int_slope")
    mt = marginal_trajectory(res, growth, time="t", shape="poly2", by="arm", grid_n=10)
    assert {"t", "pred", "lo", "hi", "arm"} <= set(mt.columns)
    assert (mt["hi"] > mt["lo"]).all()
    assert set(mt["arm"]) == {"arm0", "arm1"}


def test_blup_table(growth):
    res, _ = fit_growth_curve(growth, outcome="y", time="t", shape="linear", group="subj",
                              random="int_slope")
    bl = blup_table(res, group="subj")
    assert len(bl) == growth["subj"].nunique()
    assert list(bl["blup"]) == sorted(bl["blup"])


# --------------------------------------------------------------------------- #
#  Phase 3 - Bayesian growth-curve
# --------------------------------------------------------------------------- #
from functions.general.longitudinal.growth import fit_growth_bayes_curve, posterior_trajectory


def test_fit_growth_bayes_curve(growth):
    r = fit_growth_bayes_curve(growth, outcome="y", time="t", shape="poly2", by="arm",
                               group="subj", random="int_slope", draws=200, tune=200, chains=2, seed=1)
    f = r["fixed"]
    assert f["I(t ** 2)"]["hdi_high"] < 0          # credible negative curvature (true -0.4)
    assert f["t"]["pd"] > 0.95                      # credible positive time slope
    assert "tau_subject" in r["variance"] and "tau_slope" in r["variance"]
    assert r.get("loo") is not None
    mt = posterior_trajectory(r, grid_n=8)
    assert {"t", "pred", "lo", "hi", "arm"} <= set(mt.columns)
    assert (mt["hi"] > mt["lo"]).all()


# --------------------------------------------------------------------------- #
#  Phase 4 - GAMM / GLMM ordinal / forecast / power
# --------------------------------------------------------------------------- #
from functions.general.longitudinal.growth import forecast_individual, power_growth


def test_growth_bayes_pspline_gamm(growth):
    r = fit_growth_bayes_curve(growth, outcome="y", time="t", shape="pspline", knots=5,
                               group="subj", random="int_slope", draws=150, tune=150, chains=2, seed=1)
    assert "tau_smooth" in r["variance"]          # estimated smoothing penalty
    assert r.get("loo") is not None
    assert r["shape_label"].startswith("P-spline")


def test_growth_bayes_ordinal():
    rng = np.random.RandomState(3); rows = []
    for s in range(40):
        a = rng.normal(0, 1)
        for t in range(5):
            lat = a + 0.6 * t + rng.normal(0, 1)
            rows.append({"subj": f"S{s}", "t": t, "yc": int(np.clip(np.digitize(lat, [-1, 0.5, 2]), 0, 3))})
    df = pd.DataFrame(rows)
    r = fit_growth_bayes_curve(df, outcome="yc", time="t", shape="linear", group="subj",
                               family="ordinal", random="intercept", draws=150, tune=150, chains=2, seed=2)
    assert r["family"] == "ordinal"
    assert r["fixed"]["t"]["pd"] > 0.9            # credible positive latent slope
    assert "Intercept" not in r["_colnames"]      # cutpoints absorb the intercept


def test_forecast_individual_clamp_and_extrapolate(growth):
    rl = fit_growth_bayes_curve(growth, outcome="y", time="t", shape="linear", group="subj",
                                random="int_slope", draws=120, tune=120, chains=2, seed=1)
    fc = forecast_individual(rl, subject="S0", horizon=10, grid_n=6)        # linear -> extrapolates
    assert {"t", "pred", "lo", "hi"} <= set(fc.columns)
    assert abs(fc["t"].max() - 10.0) < 1e-6
    rp = fit_growth_bayes_curve(growth, outcome="y", time="t", shape="spline", knots=4, group="subj",
                                random="int_slope", draws=120, tune=120, chains=2, seed=1)
    fp = forecast_individual(rp, subject="S0", horizon=20, grid_n=6)        # spline -> clamped to the range
    assert fp["t"].max() <= float(growth["t"].max()) + 1e-6


def test_power_growth():
    hi = power_growth(n_subj=30, n_times=5, beta_time=0.5, sd_resid=1.0, sd_intercept=1.0,
                      sd_slope=0.3, n_sim=60, seed=0)
    lo = power_growth(n_subj=30, n_times=5, beta_time=0.0, sd_resid=1.0, sd_intercept=1.0,
                      sd_slope=0.3, n_sim=60, seed=1)
    assert hi["power"] > 0.8                       # clear effect -> high power
    assert lo["power"] < 0.15                      # null -> ~ alpha (false positives)
