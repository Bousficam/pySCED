"""
SCED core - series diagnostics and segmented regression.

Mann-Kendall (+ Hamed-Rao), lag-1 autocorrelation, Durbin-Watson, the composite
pre-analysis series diagnostic, and segmented interrupted-time-series regression.
Imports the shared Kendall S / Theil-Sen leaf helpers from the nuisance base layer.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .nuisance import _kendall_s, sens_slope


def _rank_autocorr(values, lag):
    """Lag-`lag` autocorrelation of the ranks of ``values`` (used by Hamed-Rao)."""
    r = stats.rankdata(values)
    r = r - r.mean()
    n = len(r)
    if lag >= n:
        return 0.0
    denom = np.sum(r * r)
    if denom == 0:
        return 0.0
    return float(np.sum(r[:n - lag] * r[lag:]) / denom)


def mann_kendall(values, *, autocorr_correction=True, alpha=0.05):
    """
    Mann-Kendall monotonic trend test, with the **Hamed-Rao (1998)** variance
    correction for serial autocorrelation.

    Plain Mann-Kendall assumes independent observations; in a SCED the sessions are
    serially dependent, which makes the naive trend test declare spurious learning.
    Hamed-Rao inflates (or deflates) Var(S) by a factor built from the *significant*
    rank autocorrelations - the correct way to ask "is there a learning trend?"
    while respecting the dependence. Returns S, tau, the corrected/naive variances,
    z and p-values, plus the correction factor.

    Implementation - the autocorrelation correction follows Hamed & Rao (1998) exactly as
    implemented in R ``modifiedmk::mmkh``: the series is DETRENDED by its Sen's slope, the
    rank-autocorrelation of the residuals is computed, only lags above the screen
    |rho_k| > z_{1-alpha/2}/sqrtn are kept, and Var(S) is inflated by
    essf = 1 + (2/(n(n-1)(n-2))) Sum_k (n-k)(n-k-1)(n-k-2) rho_k. Ref: Hamed, K. H., & Rao,
    A. R. (1998). J. Hydrology 204, 182-196. doi:10.1016/S0022-1694(97)00125-X.

    Verified against R: the **naive p** (``p_naive``, ties-corrected variance) matches
    ``Kendall::MannKendall`` exactly, and the **autocorrelation-corrected** ``z`` / ``p_value``
    / ``var_hr`` match ``modifiedmk::mmkh`` exactly. ``tau`` here is **tau-a** (S / [n(n-1)/2]);
    R reports **tau-b** (ties in the denominator), so the coefficient differs slightly under
    ties while the p is identical.

    References: Mann 1945; Kendall 1975 (Mann-Kendall trend test); Hamed & Rao 1998
    (autocorrelation-corrected variance).
    R equivalent: modifiedmk::mmkh (Hamed-Rao); Kendall::MannKendall (naive).
    """
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 4:
        return {"S": np.nan, "tau": np.nan, "z": np.nan, "p_value": np.nan,
                "p_naive": np.nan, "var0": np.nan, "var_hr": np.nan,
                "hr_correction": np.nan, "n": n}
    s = _kendall_s(x)
    # Var(S) with ties correction
    _, counts = np.unique(x, return_counts=True)
    ties = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var0 = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    # Hamed-Rao (1998) correction factor, matching R modifiedmk::mmkh exactly: DETREND by
    # Sen's slope, take the rank-autocorrelation of the residuals, keep only lags above the
    # fixed screen |rho_k| > z_{1-alpha/2}/sqrt(n), and inflate Var(S) by
    # essf = 1 + (2/(n(n-1)(n-2))) Sum_k (n-k)(n-k-1)(n-k-2) rho_k.
    cf = 1.0
    if autocorr_correction and n >= 4:
        slope_hr = sens_slope(x)
        slope_hr = slope_hr if np.isfinite(slope_hr) else 0.0
        resid = x - slope_hr * np.arange(1, n + 1, dtype=float)   # detrend (Sen) before ranking
        sig = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)          # mmkh screen (fixed, not per-lag)
        acc = 0.0
        for k in range(1, n - 1):
            rho = _rank_autocorr(resid, k)                        # acf of the residual ranks
            if abs(rho) > sig:                                    # only significant lags enter
                acc += (n - k) * (n - k - 1) * (n - k - 2) * rho
        cf = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * acc
        cf = max(cf, 1e-6)
    var_hr = var0 * cf

    def _z_p(var):
        if var <= 0:
            return 0.0, 1.0
        z = (s - np.sign(s)) / np.sqrt(var)
        return float(z), float(2 * stats.norm.sf(abs(z)))

    z_hr, p_hr = _z_p(var_hr)
    _, p0 = _z_p(var0)
    tau = s / (0.5 * n * (n - 1))
    return {"S": float(s), "tau": round(tau, 4),
            "z": round(z_hr, 4), "p_value": round(p_hr, 4),
            "p_naive": round(p0, 4), "var0": round(var0, 2),
            "var_hr": round(var_hr, 2), "hr_correction": round(cf, 4), "n": n}


def durbin_watson(residuals):
    """Durbin-Watson statistic (~2 = no lag-1 autocorrelation, <2 positive).

    References: Durbin & Watson 1950, 1951 (serial-correlation statistic).
    R equivalent: lmtest::dwtest; car::durbinWatsonTest."""
    e = pd.to_numeric(pd.Series(residuals), errors="coerce").dropna().to_numpy()
    if len(e) < 2 or np.sum(e * e) == 0:
        return np.nan
    return float(np.sum(np.diff(e) ** 2) / np.sum(e * e))


def lag1_autocorr(values):
    """Lag-1 autocorrelation of a series (serial dependence of the sessions).

    References: Box & Jenkins 1970 (sample autocorrelation function).
    R equivalent: stats::acf (lag 1)."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[:-1] * x[1:]) / denom) if denom else np.nan


def diagnose_series(df, *, session_col, outcome_col, condition_col=None, alpha=0.05):
    """
    Pre-analysis diagnostic of a SCED series: is there a **learning/time trend**,
    and is the series **autocorrelated**? Run before the condition test to inform
    (not auto-trigger) the choice of ``detrend``.

    When ``condition_col`` is given, the condition means are removed first so the
    diagnostic isolates the *temporal* structure from genuine condition
    differences. Reports Mann-Kendall+Hamed-Rao (trend), Sen's slope (magnitude),
    lag-1 autocorrelation / Durbin-Watson (dependence) and a **suggested detrend
    type** (none / linear / log) by comparing a linear vs log fit of the trend.

    References: Hamed & Rao 1998 (trend); Theil 1950 / Sen 1968 (slope); Durbin & Watson 1950
    (dependence).
    R equivalent: no single R equivalent (compose Kendall::MannKendall / modifiedmk::mmkh +
    trend::sens.slope + stats::acf).
    """
    d = df.sort_values(session_col).copy()
    y = pd.to_numeric(d[outcome_col], errors="coerce")
    sess = pd.to_numeric(d[session_col], errors="coerce")
    resid = y.copy()
    if condition_col is not None:
        resid = y - d.groupby(condition_col)[outcome_col].transform(
            lambda s: pd.to_numeric(s, errors="coerce").mean())
    ok = resid.notna() & sess.notna()
    resid, sess_ok = resid[ok].to_numpy(), sess[ok].to_numpy()

    mk = mann_kendall(resid, alpha=alpha)
    slope = sens_slope(resid, x=sess_ok)
    r1 = lag1_autocorr(resid)
    dw = durbin_watson(resid - resid.mean())

    # linear vs log fit of the temporal trend, to suggest a detrend basis
    def _r2(basis):
        b = np.polyfit(basis, resid, 1)
        pred = b[0] * basis + b[1]
        tss = np.sum((resid - resid.mean()) ** 2)
        return 1 - np.sum((resid - pred) ** 2) / tss if tss > 0 else 0.0
    r2_lin = _r2(sess_ok.astype(float))
    r2_log = _r2(np.log(sess_ok - sess_ok.min() + 1.0))
    trend_sig = np.isfinite(mk["p_value"]) and mk["p_value"] < alpha
    suggested = "none" if not trend_sig else ("log" if r2_log >= r2_lin else "linear")

    return {
        "Trend test": "Mann-Kendall + Hamed-Rao (autocorrelation-corrected)",
        "Trend tau": mk["tau"], "Trend p (Hamed-Rao)": mk["p_value"],
        "Trend p (naive MK)": mk["p_naive"], "Hamed-Rao variance factor": mk["hr_correction"],
        "Sen's slope (per session)": round(slope, 4) if np.isfinite(slope) else np.nan,
        "Lag-1 autocorrelation": round(r1, 4) if np.isfinite(r1) else np.nan,
        "Durbin-Watson": round(dw, 4) if np.isfinite(dw) else np.nan,
        "R2 linear / log": (round(r2_lin, 3), round(r2_log, 3)),
        "Learning trend detected": bool(trend_sig),
        "Suggested detrend": suggested,
    }


def segmented_regression(df, *, session_col, phase_col, outcome_col,
                         baseline_label, treatment_label, family="gaussian"):
    """
    Interrupted-time-series regression of an AB contrast, estimating both a
    LEVEL change (immediate jump at intervention) and a SLOPE change
    (change in trend afterwards). Returns ``(results | None, info)``.

    Design matrix (Wagner 2002 parameterisation)::

        outcome ~ time + phase + time_since_intervention

    where ``phase`` = 0/1 (A/B), ``time`` = session index, and
    ``time_since_intervention`` = 0 in A **and 0 at the first treatment session**,
    then 1,2,... afterwards. With this centring the ``phase`` coefficient is the
    **immediate level change at the intervention point** (evaluated at the first B
    session, where time_since = 0), and ``time_since_intervention`` is the slope
    change. ``family`` in {gaussian, poisson, binomial} for continuous/count/binary.

    Verified against the R package ``scan`` 0.68.1 (``plm``): the slope-change
    coefficient is identical, and the level-change coefficient matches ``phaseB`` once
    time_since is centred at the first treatment session.

    References: Wagner et al. 2002 (segmented interrupted-time-series regression).
    R equivalent: scan::plm (verified); segmented::segmented; nlme::gls(corAR1).
    """
    sub = df[df[phase_col].isin([baseline_label, treatment_label])].copy()
    sub = sub.sort_values(session_col)
    sub["_time"] = pd.to_numeric(sub[session_col], errors="coerce")
    sub["_phase"] = (sub[phase_col] == treatment_label).astype(int)
    first_b = sub.loc[sub["_phase"] == 1, "_time"].min()
    sub["_tsince"] = np.where(sub["_phase"] == 1, sub["_time"] - first_b, 0)   # 0 at 1st B (scan::plm)
    sub["_y"] = pd.to_numeric(sub[outcome_col], errors="coerce")
    sub = sub.dropna(subset=["_y", "_time"])

    info = {"Model": f"Segmented ITS regression ({family})",
            "Formula": "_y ~ _time + _phase + _tsince",
            "Level change term": "_phase", "Slope change term": "_tsince"}
    if len(sub) < 5 or sub["_phase"].nunique() < 2:
        info["Warning"] = "Insufficient sample size/contrast for the segmented regression."
        return None, info
    fam = {"gaussian": sm.families.Gaussian(), "poisson": sm.families.Poisson(),
           "binomial": sm.families.Binomial()}.get(family.lower())
    if fam is None:
        raise ValueError("family must be gaussian / poisson / binomial")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.glm("_y ~ _time + _phase + _tsince", data=sub, family=fam).fit()
        info["Level change (coef, p)"] = (round(float(res.params["_phase"]), 4),
                                          round(float(res.pvalues["_phase"]), 4))
        info["Slope change (coef, p)"] = (round(float(res.params["_tsince"]), 4),
                                          round(float(res.pvalues["_tsince"]), 4))
    except Exception as exc:
        info["Warning"] = f"Segmented regression not fitted ({type(exc).__name__})."
        return None, info
    return res, info


def segmented_table(res, *, exponentiate=False):
    """Tidy coef (95% CI) + p table from a segmented_regression fit.

    R equivalent: broom::tidy (coef + CI + p) - potential equivalent, to test."""
    label = "exp(coef) (95% CI)" if exponentiate else "Coef (95% CI)"
    cols = ["Term", "Meaning", label, "P"]
    if res is None:
        return pd.DataFrame(columns=cols)
    meaning = {"_time": "Baseline trend", "_phase": "Level change at intervention",
               "_tsince": "Slope change after intervention", "Intercept": "Intercept"}
    conf, params, pv = res.conf_int(), res.params, res.pvalues
    rows = []
    for t in params.index:
        lo, hi = conf.loc[t]
        if exponentiate:
            val = f"{np.exp(params[t]):.3f} ({np.exp(lo):.3f} to {np.exp(hi):.3f})"
        else:
            val = f"{params[t]:.3f} ({lo:.3f} to {hi:.3f})"
        rows.append({"Term": t, "Meaning": meaning.get(t, t), label: val,
                     "P": f"{pv[t]:.4f}"})
    return pd.DataFrame(rows, columns=cols)
