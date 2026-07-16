"""
SCED core - nuisance / design-matrix base layer.

Lowest layer of the SCED core package: time-nuisance design matrices, residual
sum of squares helpers, within-unit standardisation, the condition partial-SS
statistic, and two leaf trend helpers (Kendall S, Theil-Sen slope) shared by the
effect-size and diagnostic layers. Depends only on numpy / pandas / scipy - never
on a sibling core submodule.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _kendall_s(values):
    """Kendall S of a sequence vs its time order: #concordant - #discordant pairs.

    References: Kendall 1938 (Kendall rank correlation S).
    R equivalent: Kendall::MannKendall (S component)."""
    v = np.asarray(values, float)
    n = len(v)
    s = 0
    for i in range(n - 1):
        d = v[i + 1:] - v[i]
        s += np.sum(d > 0) - np.sum(d < 0)
    return float(s)


def sens_slope(values, x=None):
    """Theil-Sen (Sen's) slope: median of all pairwise slopes - a robust,
    breakdown-resistant estimate of the per-session trend (learning magnitude).
    Verified == R ``scan::theil_sen``.

    References: Theil 1950; Sen 1968 (Theil-Sen median-of-slopes estimator).
    R equivalent: scan::theil_sen; trend::sens.slope; zyp::zyp.sen."""
    y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    t = np.arange(1, len(y) + 1, dtype=float) if x is None else \
        pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(t)
    y, t = y[ok], t[ok]
    if len(y) < 2:
        return np.nan
    slopes = [(y[j] - y[i]) / (t[j] - t[i])
              for i in range(len(y)) for j in range(i + 1, len(y)) if t[j] != t[i]]
    return float(np.median(slopes)) if slopes else np.nan


def _nuisance_basis(sessions, detrend):
    """Design matrix of the time nuisance: intercept (+ session or log-session)."""
    t = np.asarray(sessions, float)
    ones = np.ones_like(t)
    if detrend == "none":
        return ones[:, None]
    if detrend == "linear":
        return np.column_stack([ones, t])
    if detrend == "log":
        return np.column_stack([ones, np.log(t - t.min() + 1.0)])
    raise ValueError("detrend must be none / linear / log")


def make_condition_ss_stat(sessions, detrend="none"):
    """
    Build a statistic ``stat(labels, values)`` = the sum of squares explained by
    CONDITION **beyond the time trend** (ANCOVA-style partial SS).

    Permutation handling - to be precise about the scheme: the randomization test
    permutes the **condition labels** (the regressor of interest), keeping the
    outcome and the time nuisance ``Z`` fixed, and refits the full model
    ``[Z | condition]`` each time. In the Winkler/PALM taxonomy of GLM permutation
    methods this is **Draper-Stoneman** (permute X), *not* Freedman-Lane (which
    permutes the reduced-model residuals). That distinction matters for
    observational GLMs; here the conditions were physically randomised, so this is
    a **design-based randomization test** (Edgington/Onghena) and its permutation
    distribution is exact under the sharp null for *any* statistic - ``Z`` only
    enters the statistic to regain the power a learning trend would otherwise cost.

    Works for 2 or k conditions; the statistic is one-sided (larger = stronger
    condition effect). With ``detrend="none"`` it reduces to the plain
    between-condition sum of squares. (For a *non-randomised* or temporally
    confounded series, true Freedman-Lane residual permutation would be more
    robust - not implemented, as the SCED case is randomised.)

    References: Draper & Stoneman 1966 (permute-X partial SS); Winkler et al. 2014 (GLM
    permutation); Edgington & Onghena 2007 (design-based exactness).
    R equivalent: permuco::lmperm (draper_stoneman scheme) - potential equivalent, to test.
    """
    Z = _nuisance_basis(sessions, detrend)

    def _rss(design, y):
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        return float(resid @ resid)

    def stat(labels, values):
        y = np.asarray(values, float)
        levels = np.unique(labels)
        X = np.column_stack([(labels == c).astype(float) for c in levels])
        full = np.column_stack([Z, X])
        return _rss(Z, y) - _rss(full, y)

    return stat


def _rss_design(design, y):
    """Residual sum of squares of an OLS fit y ~ design (rank-deficiency safe)."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    r = y - design @ beta
    return float(r @ r)


def _append_covariates(Z, covariates, n):
    """Append session-level covariate column(s) to a nuisance design ``Z`` (ANCOVA-style
    adjustment). ``covariates`` may be a 1-D array, a 2-D array (n x q) or None."""
    if covariates is None:
        return Z
    C = np.asarray(covariates, float)
    if C.ndim == 1:
        C = C[:, None]
    if C.shape[0] != n:
        raise ValueError("covariates must have one row per observation.")
    return np.column_stack([Z, C])


def _group_nuisance(units, sessions, detrend):
    """Nuisance design ``Z`` for a replicated SCED: one intercept per unit, plus a
    unit-specific time slope (linear / log session) when detrending."""
    units = np.asarray(units)
    sessions = np.asarray(sessions, float)
    uniq = pd.unique(units)
    cols = [(units == u).astype(float) for u in uniq]            # per-unit intercept
    if detrend == "linear":
        for u in uniq:
            m = units == u
            c = np.zeros(len(units)); c[m] = sessions[m]
            cols.append(c)
    elif detrend == "log":
        for u in uniq:
            m = units == u
            c = np.zeros(len(units)); c[m] = np.log(sessions[m] - sessions[m].min() + 1.0)
            cols.append(c)
    elif detrend != "none":
        raise ValueError("detrend must be none / linear / log")
    return np.column_stack(cols)


def _standardize_within(values, units):
    """Z-score the outcome within each unit. Invariant to within-unit label
    permutation (the value set per unit is fixed), so it preserves exactness while
    putting units on a comparable scale."""
    values = np.asarray(values, float).copy()
    for u in pd.unique(np.asarray(units)):
        m = np.asarray(units) == u
        sd = values[m].std(ddof=0)
        values[m] = (values[m] - values[m].mean()) / sd if sd > 0 else 0.0
    return values


def _cochran_q(effects, variances):
    """Meta-analytic heterogeneity of per-unit effects: inverse-variance weighted
    Cochran's Q (Cochran 1954), its chi^2(df) p-value, and I^2 = max(0, (Q-df)/Q)
    (Higgins & Thompson 2002). Verified == R ``metafor::rma(method="FE")`` (Q, df, p, I^2).

    References: Cochran 1954 (Q); Higgins & Thompson 2002 (I^2).
    R equivalent: metafor::rma(method="FE") - verified."""
    effects = np.asarray(effects, float)
    w = 1.0 / np.asarray(variances, float)
    keep = np.isfinite(effects) & np.isfinite(w) & (w > 0)
    effects, w = effects[keep], w[keep]
    k = len(effects)
    if k < 2:
        return {"Q": np.nan, "Q_df": 0, "Q_p": np.nan, "I2_percent": np.nan,
                "pooled_effect": float(effects[0]) if k else np.nan, "k_units": k}
    pooled = float(np.sum(w * effects) / np.sum(w))
    Q = float(np.sum(w * (effects - pooled) ** 2))
    df = k - 1
    return {"Q": round(Q, 4), "Q_df": df, "Q_p": round(float(stats.chi2.sf(Q, df)), 4),
            "I2_percent": round(max(0.0, (Q - df) / Q) * 100, 1) if Q > 0 else 0.0,
            "pooled_effect": round(pooled, 4), "k_units": k}


def _effect_values(values, labels, units, sessions, detrend, grouped):
    """Values on which to compute the effect size, made **coherent with the test**:
    residualised against the time trend when detrending (per-unit for a group),
    else unit-mean-centred for a group, else raw. This keeps Hedges' g and the
    permutation p telling the same story."""
    if detrend != "none":
        Z = (_group_nuisance(units, sessions, detrend) if grouped
             else _nuisance_basis(sessions, detrend))
        beta, *_ = np.linalg.lstsq(Z, values, rcond=None)
        return values - Z @ beta
    if grouped:
        out = values.copy()
        for u in pd.unique(units):
            mm = units == u
            out[mm] = values[mm] - values[mm].mean()
        return out
    return values
