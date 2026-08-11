"""Cross-validation of perm_boot_slope_band (within-unit Freedman-Lane permutation p + within-unit
residual bootstrap band/CI) against an INDEPENDENT reference (scipy.stats.bootstrap) and OLS theory.

No off-the-shelf tool does the SCED-constrained WITHIN-UNIT residual bootstrap ; we reduce to the
plain iid residual bootstrap (a SINGLE unit -> within-unit = iid) where scipy.stats.bootstrap, driven
to resample the SAME residuals, must agree. This proves the custom loop implements the documented
method (Freedman 1981) correctly, rather than a home-made statistic.
"""
import numpy as np
import scipy.stats as st

from functions.sced.core.permutation import perm_boot_slope_band


def _sim(n=120, slope=0.5, sd=1.5, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    y = 2.0 + slope * x + rng.normal(0, sd, n)
    X = np.column_stack([np.ones(n), x])          # [intercept | x] ; one unit -> within-unit = iid
    units = np.zeros(n, dtype=int)
    Lg = np.array([[1.0, 0.0], [1.0, 10.0]])       # group line over x in {0, 10}
    return x, y, X, units, Lg


def test_residual_bootstrap_matches_scipy():
    """Our within-unit (=iid, one unit) residual bootstrap slope CI must match scipy.stats.bootstrap
    driven to do the SAME residual bootstrap (resample residuals, refit), up to Monte-Carlo error."""
    x, y, X, units, Lg = _sim()
    nb = 20000
    ours = perm_boot_slope_band(y, X, 1, units, Lg, n_perm=50, n_boot=nb, seed=0)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); fit = X @ b; resid = y - fit

    def stat_resid(rs):                            # rs = resampled residuals -> refit slope
        return np.linalg.lstsq(X, fit + rs, rcond=None)[0][1]

    ci = st.bootstrap((resid,), stat_resid, n_resamples=nb, method="percentile",
                      vectorized=False, confidence_level=0.95, random_state=1).confidence_interval
    assert abs(ours["coef_lo"][1] - ci.low) < 0.01, (ours["coef_lo"][1], ci.low)
    assert abs(ours["coef_hi"][1] - ci.high) < 0.01, (ours["coef_hi"][1], ci.high)


def test_bootstrap_ci_near_ols_theory():
    """For a homoscedastic linear model the residual-bootstrap CI must be close to the OLS Wald CI
    (b +/- 1.96 SE) - a sanity anchor independent of any resampling code."""
    x, y, X, units, Lg = _sim()
    b, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ b
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    t_lo, t_hi = b[1] - 1.96 * se, b[1] + 1.96 * se
    ours = perm_boot_slope_band(y, X, 1, units, Lg, n_perm=50, n_boot=20000, seed=0)
    assert abs(ours["coef_lo"][1] - t_lo) < 0.02
    assert abs(ours["coef_hi"][1] - t_hi) < 0.02
    assert abs(ours["slope"] - b[1]) < 1e-9        # point estimate is exact (OLS), not resampled


def test_fl_permutation_detects_effect():
    """The Freedman-Lane permutation p is small under a real slope and large under the null."""
    _, y, X, units, Lg = _sim(slope=0.5)
    assert perm_boot_slope_band(y, X, 1, units, Lg, n_perm=2000, n_boot=10, seed=0)["p"] < 0.01
    _, y0, X0, units0, Lg0 = _sim(slope=0.0, seed=7)
    assert perm_boot_slope_band(y0, X0, 1, units0, Lg0, n_perm=2000, n_boot=10, seed=0)["p"] > 0.05


def test_within_unit_blocking_runs():
    """Multi-unit call (within-unit residual permutation + bootstrap) returns coherent shapes."""
    rng = np.random.RandomState(1)
    parts = []
    for u in range(5):
        x = np.arange(18, dtype=float)
        y = 60 + 5 * u + 0.4 * x + rng.normal(0, 2, 18)
        parts.append((np.full(18, u), x, y))
    units = np.concatenate([p[0] for p in parts])
    xx = np.concatenate([p[1] for p in parts]); yy = np.concatenate([p[2] for p in parts])
    U = np.column_stack([(units == k).astype(float) for k in range(5)])
    X = np.column_stack([U, xx.reshape(-1, 1)])
    Lg = np.zeros((18, X.shape[1])); Lg[:, :5] = 1 / 5; Lg[:, 5] = np.arange(18)
    r = perm_boot_slope_band(yy, X, 5, units, Lg, n_perm=500, n_boot=500, seed=0)
    assert r["band_lo"].shape == (18,) and r["band_hi"].shape == (18,)
    assert r["coef_lo"].shape[0] == X.shape[1]
    assert r["band_lo"][8] <= r["yhat"][8] <= r["band_hi"][8]   # observed line inside its own band
    assert 0.30 < r["slope"] < 0.50 and r["p"] < 0.01
