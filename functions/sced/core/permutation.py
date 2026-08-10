"""
SCED core - permutation / randomization inference engines.

The primary design-based inference layer: single-series and stratified condition
permutation tests, scheme selection (Draper-Stoneman / Freedman-Lane / Huh-Jhun),
ordered-trend and heterogeneity tests, the generic randomization engine, the
alternating / block schedules, pairwise post-hoc contrasts, and the report naming
/ Methods-paragraph helpers. Top layer: may import from nuisance and effect_sizes.
"""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .nuisance import (
    _nuisance_basis,
    _rss_design,
    _append_covariates,
    _group_nuisance,
    _standardize_within,
    _cochran_q,
    _effect_values,
)
from .effect_sizes import hedges_g


def _require_schedule_aware_method(method, max_consecutive, block_size):
    """Guard: a schedule constraint can only be honoured by the label-permutation
    (Draper-Stoneman) route.

    In: the resolved method string and the two schedule constraints. Out: None; RAISES
    ValueError when a constraint was declared alongside Freedman-Lane. Side effect: none.

    Freedman-Lane permutes the reduced-model residuals, so ``max_consecutive`` / ``block_size``
    have no effect on its reference set. Accepting them silently produced a p computed under a
    schedule the study never used, while the report still displayed the constraint as applied
    (Edgington & Onghena 2007: the reference set must mirror the assignment actually made)."""
    if method not in ("freedman-lane", "fl"):
        return
    declared = [n for n, v in (("max_consecutive", max_consecutive),
                               ("block_size", block_size)) if v]
    if declared:
        raise ValueError(
            f"{' and '.join(declared)} is a randomisation-schedule constraint that the "
            "Freedman-Lane residual null cannot honour: FL permutes reduced-model residuals, "
            "not the condition labels, so the constraint would be silently ignored and the "
            "p-value would not correspond to the schedule actually used. Pass "
            "method='draper-stoneman' (the design-based route, exact for a randomised SCED) "
            "or drop the constraint if the assignment really was unconstrained.")


def condition_permutation_test(values, labels, sessions, *, detrend="none",
                               method="draper-stoneman", n_perm=5000,
                               max_consecutive=None, block_size=None, covariates=None,
                               enumerate_cap=40320, random_state=0):
    """
    Permutation test of a CONDITION effect while adjusting for a time trend
    (nuisance ``Z`` = intercept + session or log-session), with a choice of
    permutation scheme. The statistic is the **partial F** for condition (extra
    sum of squares beyond ``Z``), a pivotal statistic recommended for permutation
    GLMs (Winkler et al. 2014).

    ``method``:

    - ``"draper-stoneman"`` - permute the CONDITION LABELS (the regressor of
      interest), leaving the outcome and ``Z`` fixed, and refit (Draper & Stoneman
      1966). For a *randomised* SCED this is the exact design-based randomization
      test (Edgington & Onghena 2007); it is the simplest and is exact when
      condition is uncorrelated with time, but can misbehave under strong
      condition-time collinearity (Winkler et al. 2014).
    - ``"freedman-lane"`` - fit the reduced model ``Y ~ Z``, then permute its
      residuals and add them back to the fitted values, ``Y* = Z*gamma_hat + P*resid``,
      refitting ``Y* ~ [Z, condition]`` each draw (Freedman & Lane 1983). It keeps
      the nuisance relationship intact and is the best-behaved scheme across
      scenarios, in particular under temporal confounding (Winkler et al. 2014).
      This is exactly PALM's form ``Y* = (H_Z + P*R_Z)*Y`` with a pivotal (partial
      F) statistic; verified numerically identical to that matrix implementation.

    Both reduce to the ordinary one-way randomization test when ``detrend="none"``
    (no nuisance). ``max_consecutive`` constrains the label runs in the
    Draper-Stoneman schedule. ``method="auto"`` delegates the choice to
    ``choose_permutation_method`` (DS when condition orthogonal to time, FL under temporal
    collinearity).

    ``covariates`` (session-level array, n x q) adds **ANCOVA-style adjustment**: the
    columns enter the nuisance ``Z``, so the condition is tested *beyond* time **and**
    the covariate(s). Only adjust for a covariate that is **not affected by the
    condition** (exogenous/pre-session); adjusting for a consequence of the condition
    (a mediator) biases the effect. With a covariate correlated to the condition,
    ``method="freedman-lane"`` is the safer scheme.

    **Exact vs Monte-Carlo.** For ``freedman-lane`` the reference set is the n! permutations
    of the reduced-model residuals; when ``n! <= enumerate_cap`` (default 40320, i.e. n <= 8)
    the test **enumerates them all** for an exact p = #{F* >= F_obs}/n! (``exact=True``),
    otherwise it draws ``n_perm`` Monte-Carlo permutations, p = (1+#)/(1+n_perm). Note this
    n! exact set is rarely tractable for real SCED series (n >= 10 -> >= 3.6M); the exact
    *design-based* test for a randomised SCED is instead the **Draper-Stoneman /
    label-permutation** route, whose (small) reference set is the randomisation schedule
    itself - that is the exact path used by the alternating / MBD engines. Set
    ``enumerate_cap=0`` to force Monte-Carlo. Returns ``{observed_F, p_value, method,
    detrend, df, n_perm, n_eval, exact}``.

    References: Freedman & Lane 1983; Draper & Stoneman 1966; Winkler et al. 2014;
    Edgington & Onghena 2007 (design-based randomization).
    R equivalent: permuco::lmperm (freedman_lane / draper_stoneman schemes); lmPerm.
    """
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    sessions = np.asarray(sessions, float)
    cov = None if covariates is None else np.asarray(covariates, float).reshape(len(values), -1)
    ok = np.isfinite(values) & np.isfinite(sessions)                  # drop non-finite rows
    if cov is not None:
        ok = ok & np.isfinite(cov).all(axis=1)
    values, labels, sessions = values[ok], labels[ok], sessions[ok]
    cov = cov[ok] if cov is not None else None
    n = len(values)
    if n < 3 or len(np.unique(labels)) < 2:
        raise ValueError("Insufficient data after removing missing values "
                         "(>=3 observations and >=2 conditions required).")
    if str(method).lower() == "auto":
        method = choose_permutation_method(labels, sessions, detrend=detrend)["recommended_method"]
    Z = _append_covariates(_nuisance_basis(sessions, detrend), cov, n)
    levels = np.unique(labels)

    def Xof(lab):
        return np.column_stack([(lab == c).astype(float) for c in levels])

    full_obs = np.column_stack([Z, Xof(labels)])
    p_red = np.linalg.matrix_rank(Z)
    p_full = np.linalg.matrix_rank(full_obs)
    df_num, df_den = p_full - p_red, n - p_full
    if df_num <= 0 or df_den <= 0:
        raise ValueError("Insufficient degrees of freedom for the time-adjusted test.")

    def Fstat(Xmat, y, rss_red):
        rss_full = _rss_design(np.column_stack([Z, Xmat]), y)
        if rss_full <= 0:
            return np.inf
        return ((rss_red - rss_full) / df_num) / (rss_full / df_den)

    X_obs = Xof(labels)
    F_obs = Fstat(X_obs, values, _rss_design(Z, values))
    rng = np.random.RandomState(random_state)
    ge = 0
    exact = False
    n_eval = n_perm
    m = method.lower().replace("_", "-")
    _require_schedule_aware_method(m, max_consecutive, block_size)
    if m in ("draper-stoneman", "ds"):
        rss_red = _rss_design(Z, values)          # Y and Z fixed -> constant
        scheme = (block_scheme(block_size) if block_size
                  else alternating_scheme(max_consecutive=max_consecutive))
        for _ in range(n_perm):
            if Fstat(Xof(scheme(labels, rng)), values, rss_red) >= F_obs - 1e-12:
                ge += 1
        p = (1 + ge) / (1 + n_perm)
    elif m in ("freedman-lane", "fl"):
        beta_z, *_ = np.linalg.lstsq(Z, values, rcond=None)
        fitted_z = Z @ beta_z
        resid_z = values - fitted_z
        # EXACT when feasible: the Freedman-Lane reference set is the n! permutations of the
        # reduced-model residuals. Complete enumeration is only tractable for small n
        # (n! <= enumerate_cap, i.e. n <~ 8); otherwise Monte-Carlo. NB the exact *design-based*
        # test for a randomised SCED is the Draper-Stoneman / label-permutation route (a small
        # schedule reference set) - FL is the model-based scheme for the confounded regime, so
        # its exact set is the full n! and is rarely enumerable for real series.
        total = math.factorial(n)
        if enumerate_cap and total <= enumerate_cap:
            exact, n_eval = True, total
            for perm in itertools.permutations(range(n)):
                yp = fitted_z + resid_z[list(perm)]
                if Fstat(X_obs, yp, _rss_design(Z, yp)) >= F_obs - 1e-12:
                    ge += 1
            p = ge / total
        else:
            for _ in range(n_perm):
                yp = fitted_z + resid_z[rng.permutation(n)]
                if Fstat(X_obs, yp, _rss_design(Z, yp)) >= F_obs - 1e-12:   # rss_red recomputed for Y*
                    ge += 1
            p = (1 + ge) / (1 + n_perm)
    else:
        raise ValueError("method must be 'draper-stoneman' or 'freedman-lane'")
    return {"observed_F": round(float(F_obs), 4), "p_value": p,
            "method": m, "detrend": detrend, "df": (int(df_num), int(df_den)),
            "n_perm": int(n_perm), "n_eval": int(n_eval), "exact": bool(exact)}


PERMUTATION_DECISION_TREE = (
    "Is the condition PHYSICALLY randomised to sessions (known schedule)?\n"
    "|\n"
    "+-- YES  (true SCED / N-of-1)\n"
    "|   `-- Framework = Edgington & Onghena. Permute ACCORDING TO THE REAL SCHEDULE\n"
    "|       (respect blocks / max-consecutive constraint).\n"
    "|       +-- No learning trend -> simple statistic (mean difference)  [detrend=none]\n"
    "|       `-- Learning trend -> put time in the statistic, permute the LABELS\n"
    "|             = Draper-Stoneman -> ALWAYS EXACT (same E&O set, richer statistic).\n"
    "|             > Switch to Freedman-Lane ONLY if the condition is strongly collinear\n"
    "|               with time (temporal imbalance, e.g. A-early/C-late).\n"
    "`-- NO  (observational series, nothing was randomised)\n"
    "    `-- E&O does not apply -> GLM regime: Freedman-Lane (or ter Braak) on residuals.\n"
    "\n"
    "NB: autocorrelation alone does NOT require Freedman-Lane (exact in a randomised design,\n"
    "verified up to AR(1) phi=0.8). Only condition-time collinearity justifies it."
)


def choose_permutation_method(labels, sessions, *, detrend="linear", threshold=0.15):
    """
    Recommend **Draper-Stoneman** vs **Freedman-Lane** from the condition-time
    collinearity (temporal imbalance), following Winkler et al. (2014).

    Decision tree (the scheme choice only arises once a time trend is adjusted for,
    i.e. ``detrend != "none"``; with no nuisance both reduce to the ordinary
    Edgington & Onghena randomization test)::

        Is the condition PHYSICALLY randomised to sessions (known schedule)?
        |
        +-- YES  (true SCED / N-of-1)
        |   `-- Framework = Edgington & Onghena. Permute ACCORDING TO THE REAL
        |       schedule (respect blocking / max-consecutive constraints).
        |       |
        |       +-- No learning trend -> simple statistic (mean difference)  [detrend="none"]
        |       |     = "plain" Edgington & Onghena.
        |       |
        |       `-- Learning trend -> put time in the statistic and permute the
        |             condition LABELS = Draper-Stoneman -> STILL EXACT (it is the
        |             same E&O permutation set, just a richer statistic).
        |             > Switch to Freedman-Lane ONLY if condition is strongly
        |               collinear with time (temporal imbalance, e.g. A-early/C-late).
        |
        `-- NO  (observational series, nothing was randomised)
            `-- E&O does not apply (no randomisation to permute).
                -> GLM regime: Freedman-Lane (or ter Braak) on residuals.

    Key clarification on "temporal dependence": **autocorrelation alone does NOT
    require Freedman-Lane** - in a randomised design the sharp-null permutation
    distribution is exact regardless of serial dependence (verified by simulation
    up to AR(1) phi=0.8). The Draper-Stoneman caveat (Kennedy 1995; Winkler et al.
    2014) concerns **condition-time collinearity** in the *model-based* regime, and
    is often modest in magnitude; Freedman-Lane is preferred there only because it
    is robust across both regimes and never does worse.

    Note: Draper-Stoneman and Edgington & Onghena are NOT competing alternatives - 
    in a randomised SCED the label-permutation route IS the E&O test (its exactness
    comes from the physical randomisation; ``Z`` only enriches the statistic).

    Implementation: measures how much the CONDITION predicts the TIME nuisance - 
    R^2 of regressing the time basis on the condition dummies. Low R^2 (condition
    spread evenly across time) -> Draper-Stoneman; high R^2 (e.g. A clustered early,
    C late - *temporal* imbalance even with equal counts) -> Freedman-Lane. Returns
    ``{condition_time_R2, threshold, recommended_method, rationale, decision_tree}``.

    References: Winkler et al. 2014; Kennedy 1995 (scheme choice under condition-time
    collinearity).
    R equivalent: no direct R equivalent (heuristic scheme selector).
    """
    labels = np.asarray(labels)
    if detrend == "none":
        r2 = 0.0
    else:
        tcol = _nuisance_basis(sessions, detrend)[:, 1]
        levels = np.unique(labels)
        X = np.column_stack([np.ones(len(labels))]
                            + [(labels == c).astype(float) for c in levels[1:]])
        beta, *_ = np.linalg.lstsq(X, tcol, rcond=None)
        tss = float(np.sum((tcol - tcol.mean()) ** 2))
        r2 = 1 - float(np.sum((tcol - X @ beta) ** 2)) / tss if tss > 0 else 0.0
    rec = "freedman-lane" if r2 > threshold else "draper-stoneman"
    rationale = (
        f"The condition explains {r2 * 100:.1f}% of the temporal variance "
        f"(threshold {threshold * 100:.0f}%). "
        + ("Condition-time confounding -> Freedman-Lane (more robust; Winkler 2014)."
           if rec == "freedman-lane" else
           "Condition ~orthogonal to time -> Draper-Stoneman (exact, randomised design)."))
    return {"condition_time_R2": round(r2, 4), "threshold": threshold,
            "recommended_method": rec, "rationale": rationale,
            "decision_tree": PERMUTATION_DECISION_TREE}


def stratified_condition_permutation_test(values, labels, units, sessions, *,
                                          detrend="none", method="freedman-lane",
                                          n_perm=5000, max_consecutive=None,
                                          block_size=None, standardize=True,
                                          covariates=None, random_state=0):
    """
    **Stratified** (blocked) randomization test of a COMMON condition effect across
    several units - the replicated-single-case group analysis (Edgington & Onghena
    2007). Each unit was randomised independently, so under the sharp null the joint
    randomisation is the product of the per-unit schedules: the condition labels are
    permuted **within each unit** and a single pooled model is refit, giving an exact test.

    Model: ``Y ~ [unit intercepts (+ unit-specific time slope if detrend)] +
    condition``; the condition block (shared across units) is tested by its partial
    F (Winkler et al. 2014). ``standardize`` z-scores the outcome within each unit
    (default; permutation-invariant, so still exact) so units contribute comparably.

    ``method``: ``"draper-stoneman"`` permutes the labels within unit; ``"freedman-
    lane"`` permutes the reduced-model residuals within unit; ``"auto"`` delegates to
    ``choose_permutation_method_group``. ``max_consecutive`` constrains each unit's
    label runs. ``covariates`` (session-level, n x q) adds ANCOVA-style adjustment by
    entering the nuisance ``Z`` - adjust only for covariates **not affected by the
    condition** (a mediator would bias the effect). Returns ``{observed_F, p_value,
    method, detrend, df, n_perm, n_units}``.

    References: Edgington & Onghena 2007 (within-block randomization); Freedman & Lane 1983;
    Winkler et al. 2014.
    R equivalent: permuco::lmperm with blocking - potential equivalent, to test (closest full
    match FSL PALM).
    """
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    units = np.asarray(units)
    sessions = np.asarray(sessions, float)
    cov = None if covariates is None else np.asarray(covariates, float).reshape(len(values), -1)
    ok = np.isfinite(values) & np.isfinite(sessions)                  # drop non-finite rows
    if cov is not None:
        ok = ok & np.isfinite(cov).all(axis=1)
    values, labels, units, sessions = values[ok], labels[ok], units[ok], sessions[ok]
    cov = cov[ok] if cov is not None else None
    n = len(values)
    if n < 3 or len(np.unique(labels)) < 2:
        raise ValueError("Insufficient data after removing missing values "
                         "(>=3 observations and >=2 conditions required).")
    if str(method).lower() == "auto":
        method = choose_permutation_method_group(labels, units, sessions,
                                                 detrend=detrend)["recommended_method"]
    y = _standardize_within(values, units) if standardize else values.copy()
    Z = _append_covariates(_group_nuisance(units, sessions, detrend), cov, n)
    levels = np.unique(labels)
    uniq_u = pd.unique(units)

    def Xof(lab):
        return np.column_stack([(lab == c).astype(float) for c in levels])

    p_red = np.linalg.matrix_rank(Z)
    p_full = np.linalg.matrix_rank(np.column_stack([Z, Xof(labels)]))
    df_num, df_den = p_full - p_red, n - p_full
    if df_num <= 0 or df_den <= 0:
        raise ValueError("Insufficient degrees of freedom for the stratified test.")

    def Fstat(Xmat, yy, rss_red):
        rss_full = _rss_design(np.column_stack([Z, Xmat]), yy)
        if rss_full <= 0:
            return np.inf
        return ((rss_red - rss_full) / df_num) / (rss_full / df_den)

    X_obs = Xof(labels)
    rss_red_obs = _rss_design(Z, y)
    F_obs = Fstat(X_obs, y, rss_red_obs)
    rng = np.random.RandomState(random_state)
    # precompute per-unit row indices
    idx = {u: np.where(units == u)[0] for u in uniq_u}
    ge = 0
    m = method.lower().replace("_", "-")
    _require_schedule_aware_method(m, max_consecutive, block_size)
    if m in ("draper-stoneman", "ds"):
        scheme = (block_scheme(block_size) if block_size
                  else alternating_scheme(max_consecutive=max_consecutive))
        for _ in range(n_perm):
            lab = labels.copy()
            for u in uniq_u:
                lab[idx[u]] = scheme(labels[idx[u]], rng)
            if Fstat(Xof(lab), y, rss_red_obs) >= F_obs - 1e-12:
                ge += 1
    elif m in ("freedman-lane", "fl"):
        beta_z, *_ = np.linalg.lstsq(Z, y, rcond=None)
        fitted_z = Z @ beta_z
        resid_z = y - fitted_z
        for _ in range(n_perm):
            r = resid_z.copy()
            for u in uniq_u:
                ii = idx[u]
                r[ii] = resid_z[ii][rng.permutation(len(ii))]
            yp = fitted_z + r
            if Fstat(X_obs, yp, _rss_design(Z, yp)) >= F_obs - 1e-12:
                ge += 1
    else:
        raise ValueError("method must be 'draper-stoneman' or 'freedman-lane'")
    return {"observed_F": round(float(F_obs), 4), "p_value": (1 + ge) / (1 + n_perm),
            "method": m, "detrend": detrend, "df": (int(df_num), int(df_den)),
            "n_perm": int(n_perm), "n_units": int(len(uniq_u))}


def choose_permutation_method_group(labels, units, sessions, *, detrend="linear",
                                    threshold=0.15):
    """Group counterpart of ``choose_permutation_method``: the size-weighted mean of
    the per-unit condition-time R^2. Same rule - low -> Draper-Stoneman, high (temporal
    imbalance within units) -> Freedman-Lane (Winkler et al. 2014).

    References: Winkler et al. 2014; Kennedy 1995 (scheme choice).
    R equivalent: no direct R equivalent."""
    labels, units = np.asarray(labels), np.asarray(units)
    r2s, w = [], []
    for u in pd.unique(units):
        m = units == u
        r2 = (0.0 if detrend == "none" else
              choose_permutation_method(labels[m], np.asarray(sessions)[m],
                                        detrend=detrend, threshold=threshold)["condition_time_R2"])
        r2s.append(r2); w.append(int(m.sum()))
    r2 = float(np.average(r2s, weights=w)) if r2s else 0.0
    rec = "freedman-lane" if r2 > threshold else "draper-stoneman"
    return {"condition_time_R2": round(r2, 4), "threshold": threshold,
            "recommended_method": rec,
            "rationale": (f"Mean within-unit condition-time R^2 = {r2 * 100:.1f}% "
                          f"(threshold {threshold * 100:.0f}%) -> {rec}.")}


def huh_jhun_test(y, effect, nuisance=None, *, effect_kind="continuous", nuisance_kind="continuous",
                  stat="W", n_perm=10000, tail="both", seed=0):
    """SCALAR Huh-Jhun (2001) exact-exchangeability test of a FIXED nuisance-adjusted covariate (e.g.
    a session-time trend adjusting for condition) - the scalar sibling of the cluster
    ``spatial_huh_jhun``, reusing the SAME whitening core (``cluster.core.huh_jhun_whiten``).

    Whitens the response onto the orthogonal complement of the nuisance space and permutes the
    exactly-exchangeable whitened residuals, so type I is exact at small n where Freedman-Lane /
    Draper-Stoneman are only approximate (Helwig 2019). HJ is for a FIXED covariate ONLY - a
    randomized factor must use the randomization / Draper-Stoneman reference set instead.

    y : (n,) outcome. effect : (n,) tested fixed covariate (e.g. session index). nuisance : (n,)
    array or list of them to adjust for (e.g. condition dummies) ; None = intercept only.
    nuisance_kind : "continuous" | "discrete" (or a list, one per nuisance term). stat : "W" (robust
    HC0, default) | "t". Returns {scheme, stat, stat_kind, slope, p, n_perm} with slope = the
    ANCOVA-adjusted effect (FWL, original units) and a two-sided p on the signed statistic.

    References: Huh & Jhun (2001); Helwig (2019). R equivalent: permuco::aovperm (method='huh_jhun')."""
    from ..cluster.core import huh_jhun_whiten, _glm_statmap    # numpy/scipy-only shared HJ core
    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    nuis = [] if nuisance is None else nuisance
    w, Xw, stat = huh_jhun_whiten(Y, effect, nuis, effect_kind, nuisance_kind, stat)
    statmap, effmap, wv, _Z, _pZ, _thr, signed = _glm_statmap(
        w, Xw, list(range(Xw.shape[1])), [], stat, None, 0.5)
    m = wv.shape[0]
    if m < 2:                                                 # nuisance saturates the design : no dof left
        return {"scheme": "huh-jhun", "stat": np.nan, "stat_kind": stat, "slope": np.nan,
                "p": np.nan, "n_perm": 0}
    obs = float(statmap(wv)[0]); slope = float(effmap(wv)[0])
    rng = np.random.default_rng(seed)
    null = np.array([statmap(wv[rng.permutation(m)])[0] for _ in range(n_perm)], dtype=float)
    if signed:                                                # two-sided on the signed t / W
        p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n_perm + 1)
    else:                                                     # F / chi2 : upper tail
        p = (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)
    return {"scheme": "huh-jhun", "stat": obs, "stat_kind": stat, "slope": slope,
            "p": float(p), "n_perm": int(n_perm)}


def freedman_lane_test(y, effect, nuisance=None, *, effect_kind="continuous",
                       nuisance_kind="continuous", stat="W", n_perm=10000, tail="both", seed=0):
    """SCALAR Freedman-Lane (1983) test of a FIXED nuisance-adjusted covariate (e.g. a session-time
    trend adjusting for condition) - the scalar sibling of the cluster ``spatial_freedman_lane`` and
    the exact twin of ``huh_jhun_test`` EXCEPT for the permutation null : FL permutes the residuals of
    the REDUCED (nuisance-only) model and refits, WITHOUT the exchangeability whitening. Reuses the
    same design/statistic core (``_fl_design`` + ``_glm_statmap``) so discrete nuisances (condition)
    are expanded to dummies identically, and the robust HC0 Wald W is the same statistic.

    Preferred over Huh-Jhun as the small-n primary : FL needs no covariance whitening, so its p is
    stable to the outcome metric (raw contrast vs signed r2), where HJ's whitened p is fragile.

    y : (n,) outcome. effect : (n,) tested fixed covariate. nuisance : (n,) array / list to adjust
    for ; None = intercept only. Returns {scheme, stat, stat_kind, slope, p, n_perm}, same contract
    as huh_jhun_test. References : Freedman & Lane (1983) ; Winkler et al. (2014)."""
    from ..cluster.core import _fl_design, _glm_statmap
    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    nuis = [] if nuisance is None else nuisance
    design, test_cols, nuis_cols, stat = _fl_design(Y, effect, nuis, effect_kind, nuisance_kind, stat)
    statmap, effmap, Yv, Z, pZ, _thr, signed = _glm_statmap(
        Y, design, test_cols, nuis_cols, stat, None, 0.5)
    n = Yv.shape[0]
    if n - design.shape[1] <= 0:                              # nuisance saturates the design : no dof
        return {"scheme": "freedman-lane", "stat": np.nan, "stat_kind": stat, "slope": np.nan,
                "p": np.nan, "n_perm": 0}
    reduced = Z @ (pZ @ Yv) if Z.shape[1] else np.zeros_like(Yv)   # nuisance fit (kept across perms)
    resid = Yv - reduced                                     # reduced-model residuals (exchangeable)
    obs = float(statmap(Yv)[0]); slope = float(effmap(Yv)[0])
    rng = np.random.default_rng(seed)
    null = np.array([statmap(reduced + resid[rng.permutation(n)])[0] for _ in range(n_perm)],
                    dtype=float)
    if signed:                                               # two-sided on the signed t / W
        p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n_perm + 1)
    else:                                                    # F / chi2 : upper tail
        p = (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)
    return {"scheme": "freedman-lane", "stat": obs, "stat_kind": stat, "slope": slope,
            "p": float(p), "n_perm": int(n_perm)}


def recommend_scheme(*, role, kind, values=None, sessions=None, n=None, grouped=False, units=None,
                     detrend="linear", threshold=0.15, small_n=30):
    """UNIFIED permutation-scheme dispatcher for ONE model term - the single place that resolves
    Draper-Stoneman / Freedman-Lane / Huh-Jhun, so every engine (scalar alternating + cluster map)
    routes scheme choice through the same rule instead of hard-coding it per site.

      - kind='hinge' (relu change-point)          -> Freedman-Lane.
      - role='randomized' (factor / ordered dose) -> Draper-Stoneman (design-based : the assignment
        schedule is the exact reference set), UNLESS the assignment is strongly collinear with time
        -> Freedman-Lane (choose_permutation_method[_group], Winkler 2014). Candidates {DS, FL}.
      - role='fixed' covariate (e.g. the trend)   -> Huh-Jhun at small n (exact finite-sample
        exchangeability via the whitening projection, Helwig 2019), else Freedman-Lane ; FL is
        always the sensitivity comparator. Candidates {FL, HJ}. HJ applies ONLY to a fixed covariate
        (it is NOT an alternative to DS for a randomized factor - a different axis).

    Returns ``{primary, candidates, reason}``. Pass values+sessions (+ units if grouped) to enable
    the randomized collinearity check ; n (or len(sessions)) drives the small-n HJ rule."""
    if kind == "hinge":
        return {"primary": "freedman-lane", "candidates": ["freedman-lane"],
                "reason": "hinge change-point -> Freedman-Lane"}
    if role == "randomized":
        rec = "draper-stoneman"
        if values is not None and sessions is not None and detrend != "none":
            if grouped and units is not None:
                rec = choose_permutation_method_group(values, units, sessions, detrend=detrend,
                                                      threshold=threshold)["recommended_method"]
            else:
                rec = choose_permutation_method(values, sessions, detrend=detrend,
                                                threshold=threshold)["recommended_method"]
        return {"primary": rec, "candidates": ["draper-stoneman", "freedman-lane"],
                "reason": f"randomized factor -> {rec} (design-based unless collinear with time)"}
    # fixed covariate : Freedman-Lane is ALWAYS the primary (robust, whitening-free), Huh-Jhun is
    # kept only as a sensitivity comparator. The small-n Huh-Jhun rule was dropped : on few sessions
    # (n<=~20) HJ's exchangeability whitening is estimated from a poorly-conditioned residual
    # covariance, so the headline p becomes fragile to the outcome metric (raw contrast vs signed r2
    # flip the HJ p while FL stays stable). FL + robust W is the stable primary ; HJ still runs in the
    # {FL,HJ}x{F,W} grid and is reported alongside as sensitivity.
    nn = n if n is not None else (len(np.asarray(sessions)) if sessions is not None else None)
    return {"primary": "freedman-lane", "candidates": ["freedman-lane", "huh-jhun"],
            "reason": f"fixed covariate, n={nn} -> Freedman-Lane primary (HJ as sensitivity)"}


def heterogeneity_test(values, labels, units, sessions, *, conditions=None,
                       condition_order=None, dose=None,
                       detrend="none", n_perm=5000,
                       standardize=True, random_state=0):
    """
    Does the condition effect **vary across units** (unit x condition interaction)?
    Two complementary read-outs of the same data:

    1. **Permutation test** of the interaction (Freedman & Lane 1983): partial F for
       the ``unit x condition`` block beyond the reduced model ``[unit (+ unit-specific
       time slope) + COMMON condition]``; the reduced-model residuals are permuted
       **within unit** and the model refitted. This is **approximate**, not exact: the
       null here is "homogeneous effect" (a common effect may exist), which is *not*
       the sharp null, so label permutation would destroy the common effect - hence we
       permute residuals (FL) rather than labels (Winkler et al. 2014). Because the null is
       necessarily residual-based, a schedule constraint cannot apply here: the former
       ``max_consecutive`` argument was accepted and never used, and has been removed rather
       than left to suggest a constraint the test does not honour.
    2. **Cochran's Q** (Cochran 1954) and **I^2** (Higgins & Thompson 2002) on the
       per-unit effect (target - reference mean, inverse-variance weighted) - the
       interpretable meta-analytic heterogeneity index.

    ``conditions`` defaults to the sorted labels. The primary Q/I^2 uses the contrast
    ``conditions[1]`` vs ``conditions[0]``; with >=3 conditions the Q/I^2 is **also**
    reported for **every pair** (``per_contrast``) and, when the conditions are ordered
    doses (``condition_order`` / ``dose``), for the **per-unit dose-response slope**
    (``dose_slope`` - "does the dose-response vary across patients?"). ``standardize``
    z-scores within unit (default) so effects are comparably scaled. Returns a dict with
    the interaction F/p/df, the primary Q/I^2, ``per_contrast`` and ``dose_slope``.

    References: Freedman & Lane 1983; Winkler et al. 2014 (interaction permutation);
    Cochran 1954; Higgins & Thompson 2002 (Q / I^2).
    R equivalent: metafor::rma (Q / I^2) + permuco (interaction) combined - potential
    equivalent, to test.
    """
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    units = np.asarray(units)
    n = len(values)
    if conditions is None:
        conditions = tuple(np.unique(labels))
    y = _standardize_within(values, units) if standardize else values.copy()
    uniq_u = pd.unique(units)
    levels = np.unique(labels)

    # reduced design = unit (+time) + common condition ; full = + unit x condition
    Z_base = _group_nuisance(units, sessions, detrend)
    Xcond = np.column_stack([(labels == c).astype(float) for c in levels])
    Z_red = np.column_stack([Z_base, Xcond])
    X_int = np.column_stack([((units == u) & (labels == c)).astype(float)
                             for u in uniq_u for c in levels])

    p_red = np.linalg.matrix_rank(Z_red)
    p_full = np.linalg.matrix_rank(np.column_stack([Z_red, X_int]))
    df_num, df_den = p_full - p_red, n - p_full

    inter = {"interaction_F": np.nan, "interaction_p": np.nan,
             "interaction_df": (int(max(df_num, 0)), int(max(df_den, 0))),
             "interaction_note": "insufficient df (saturated design) - interaction not testable"}
    if df_num > 0 and df_den > 0:
        def Fstat(yy):
            rss_full = _rss_design(np.column_stack([Z_red, X_int]), yy)
            rss_red = _rss_design(Z_red, yy)
            if rss_full <= 0:
                return np.inf
            return ((rss_red - rss_full) / df_num) / (rss_full / df_den)

        F_obs = Fstat(y)
        beta_z, *_ = np.linalg.lstsq(Z_red, y, rcond=None)
        fitted, resid = Z_red @ beta_z, y - Z_red @ beta_z
        idx = {u: np.where(units == u)[0] for u in uniq_u}
        rng = np.random.RandomState(random_state)
        ge = 0
        for _ in range(n_perm):
            r = resid.copy()
            for u in uniq_u:
                ii = idx[u]
                r[ii] = resid[ii][rng.permutation(len(ii))]
            if Fstat(fitted + r) >= F_obs - 1e-12:
                ge += 1
        inter = {"interaction_F": round(float(F_obs), 4),
                 "interaction_p": (1 + ge) / (1 + n_perm),
                 "interaction_df": (int(df_num), int(df_den)),
                 "interaction_note": "Freedman-Lane (approx.), residuals permuted within unit"}

    # Cochran's Q / I^2 per CONTRAST (per-unit mean difference, inverse-variance weighted)
    def _pair_qi2(ref, trt):
        eff, var = [], []
        for u in uniq_u:
            a = y[(units == u) & (labels == ref)]
            b = y[(units == u) & (labels == trt)]
            if len(a) >= 2 and len(b) >= 2:
                eff.append(b.mean() - a.mean())
                var.append(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return _cochran_q(eff, var)

    ref, trt = conditions[0], conditions[1]
    q = _pair_qi2(ref, trt)                                  # primary contrast (backward-compat)
    per_contrast = [{"contrast": f"{b} - {a}", **_pair_qi2(a, b)}
                    for a, b in itertools.combinations(conditions, 2)]

    # Cochran's Q / I^2 on the per-unit DOSE-RESPONSE SLOPE (ordered conditions only):
    # "does the dose-response itself vary across patients?" - more meaningful than any single
    # pair when the conditions are ordered doses. Per unit: OLS slope of y on the dose, with
    # its squared standard error as the variance; then inverse-variance Q/I^2 across units.
    order = list(condition_order) if condition_order is not None else list(conditions)
    pos = {c: (float(dose[i]) if dose is not None else float(i)) for i, c in enumerate(order)}
    slopes, slope_var = [], []
    for u in uniq_u:
        m = units == u
        xu = np.array([pos.get(l, np.nan) for l in labels[m]], float)
        yu = y[m]
        ok2 = np.isfinite(xu) & np.isfinite(yu)
        xu, yu = xu[ok2], yu[ok2]
        if len(xu) >= 3 and len(np.unique(xu)) >= 2:
            lr = stats.linregress(xu, yu)
            if np.isfinite(lr.slope) and np.isfinite(lr.stderr) and lr.stderr > 0:
                slopes.append(float(lr.slope))
                slope_var.append(float(lr.stderr) ** 2)
    dose_slope = (_cochran_q(slopes, slope_var) if len(slopes) >= 2
                  else {"Q": np.nan, "Q_df": 0, "Q_p": np.nan, "I2_percent": np.nan,
                        "pooled_effect": (round(slopes[0], 4) if slopes else np.nan),
                        "k_units": len(slopes)})

    return {**inter, "n_units": int(len(uniq_u)),
            "per_contrast": per_contrast, "dose_slope": dose_slope,
            "contrast": f"{trt} - {ref}", "standardized": bool(standardize), **q}


def pairwise_contrasts(values, labels, *, units=None, sessions=None, detrend="none",
                       method="freedman-lane", improvement="increase", n_perm=5000,
                       n_boot=2000, correction="all", max_consecutive=None,
                       standardize=True, random_state=0):
    """
    All **pairwise post-hoc contrasts** between conditions, each with a standardized
    effect size + CI and a permutation p-value corrected for multiplicity.

    The omnibus randomization test says *some* conditions differ; this says *which*.
    For every pair it reports **Hedges' g** (treatment - reference, small-sample
    corrected; Hedges 1981) with a **bootstrap 95% CI** (cluster bootstrap over units
    when grouped, else resampling sessions within condition), and the permutation
    p-value from the project's own engine - ``condition_permutation_test`` for a single
    series, ``stratified_condition_permutation_test`` (unit = block) when ``units`` is
    given - so detrending / scheme choice stay consistent with the main analysis.

    The effect size is computed on the **same residualised values as the test** (time
    trend removed when detrending, unit means removed for a group), so g and p never
    contradict each other. Multiplicity: ``correction`` in {all, holm, fdr_bh, none}
    adds Holm (FWER) and/or Benjamini-Hochberg (FDR) adjusted p-values (statsmodels
    ``multipletests``). ``improvement`` only orients the reported sign. Returns a tidy
    DataFrame, one row per pair.

    References: Hedges 1981 (g); Holm 1979 (FWER); Benjamini & Hochberg 1995 (FDR).
    R equivalent: emmeans::contrast + stats::p.adjust; rstatix.
    """
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    sessions = None if sessions is None else np.asarray(sessions, float)
    levels = list(np.unique(labels))
    grouped = units is not None
    if grouped:
        units = np.asarray(units)
    rng = np.random.RandomState(random_state)

    rows, praw = [], []
    for ref, trt in itertools.combinations(levels, 2):
        m = np.isin(labels, [ref, trt])
        sess_m = (np.arange(1, m.sum() + 1) if sessions is None else sessions[m])
        u_m = units[m] if grouped else None
        if grouped:
            res = stratified_condition_permutation_test(
                values[m], labels[m], u_m, sess_m, detrend=detrend, method=method,
                n_perm=n_perm, max_consecutive=max_consecutive, standardize=standardize,
                random_state=random_state)
        else:
            res = condition_permutation_test(
                values[m], labels[m], sess_m, detrend=detrend, method=method,
                n_perm=n_perm, max_consecutive=max_consecutive, random_state=random_state)

        ve = _effect_values(values[m], labels[m], u_m, sess_m, detrend, grouped)
        a, b = ve[labels[m] == ref], ve[labels[m] == trt]
        g_obs = hedges_g(b, a)

        gboot = []
        if grouped:                                    # cluster bootstrap over units
            uu = pd.unique(u_m)
            for _ in range(n_boot):
                pick = rng.choice(uu, len(uu), replace=True)
                aa = np.concatenate([ve[(u_m == u) & (labels[m] == ref)] for u in pick])
                bb = np.concatenate([ve[(u_m == u) & (labels[m] == trt)] for u in pick])
                g = hedges_g(bb, aa)
                if np.isfinite(g):
                    gboot.append(g)
        else:                                          # resample within condition
            for _ in range(n_boot):
                g = hedges_g(b[rng.randint(0, len(b), len(b))],
                             a[rng.randint(0, len(a), len(a))])
                if np.isfinite(g):
                    gboot.append(g)
        ci = ((np.nan, np.nan) if len(gboot) < 10
              else (round(float(np.percentile(gboot, 2.5)), 3),
                    round(float(np.percentile(gboot, 97.5)), 3)))
        rows.append({"Comparison": f"{trt} vs {ref}", "Hedges g": round(g_obs, 3),
                     "g 95% CI": ci, "p (perm)": round(res["p_value"], 4)})
        praw.append(res["p_value"])

    out = pd.DataFrame(rows)
    if praw:
        if correction in ("holm", "all"):
            out["p Holm"] = multipletests(praw, method="holm")[1].round(4)
        if correction in ("fdr_bh", "all"):
            out["p FDR"] = multipletests(praw, method="fdr_bh")[1].round(4)
    return out


def diff_in_means(labels, values, *, target, reference):
    """Mean(target) - Mean(reference). Distribution-free statistic valid for
    continuous, count and binary (rate/proportion difference) outcomes.

    R equivalent: base R mean difference; coin::independence_test statistic."""
    values = np.asarray(values, float)
    m_t = values[labels == target].mean() if np.any(labels == target) else np.nan
    m_r = values[labels == reference].mean() if np.any(labels == reference) else np.nan
    return m_t - m_r


def omnibus_variance(labels, values):
    """Variance of the per-condition means - omnibus statistic for >=2 conditions.

    Unordered: it answers "do ANY of the condition means differ?" and ignores any
    ordering of the conditions. For an ORDERED factor (e.g. a dose: MI time 4/5/6 s) it
    wastes power on a monotone dose-response - use ``make_ordered_trend_stat`` IN ADDITION
    (not instead; see that function's note on validation status).

    R equivalent: base R var of the group means (the between-group ANOVA numerator)."""
    values = np.asarray(values, float)
    means = [values[labels == c].mean() for c in np.unique(labels) if np.any(labels == c)]
    return float(np.var(means)) if means else np.nan


def _validate_condition_order(condition_order, labels):
    """Guard the ordered-trend test against a ``condition_order`` that does not match the
    conditions present in the data. A mismatch (typo, wrong case, a forgotten or extra
    level) would otherwise be SILENT: conditions absent from ``condition_order`` get mapped
    to NaN and dropped from the statistic (testing a subset while claiming the whole dose
    range), or - if no label matches - the observed statistic is NaN and the permutation p
    collapses to the most-significant value. Raise instead."""
    present = set(map(str, pd.unique(np.asarray(labels))))
    order = set(map(str, condition_order))
    missing = present - order            # in data but not ordered -> silently dropped
    extra = order - present              # ordered but absent from data
    if missing or extra:
        raise ValueError(
            "condition_order does not match the conditions present: "
            + (f"absent from the order {sorted(missing)} " if missing else "")
            + (f"; absent from the data {sorted(extra)}" if extra else "")
            + f". Data = {sorted(present)}, order = {list(condition_order)}.")


def make_ordered_trend_stat(condition_order, *, kind="slope", dose=None,
                            improvement="increase"):
    """
    Statistic factory for an ORDERED-condition (dose-response) randomization test:
    *does the outcome change monotonically across ordered conditions?* Built for designs
    where the conditions are ordered levels of one factor - e.g. the BCINET pilot ATD with
    MI time demanded = 4 / 5 / 6 s. Plugs into ``randomization_test(labels, values,
    statistic=<this>, scheme=...)``.

    Why this is justified, and its limits (read before using) - 
      - **Framework**: the randomization test admits ANY statistic; its exactness comes
        from the physical randomisation, not the statistic (Edgington & Onghena 2007;
        Heyvaert & Onghena 2014). So an ordered/trend statistic is a legitimate RT choice.
      - **The statistics are canonical**: the OLS slope on the numeric dose is the linear
        polynomial-trend contrast of classical experimental design; Jonckheere-Terpstra is
        THE canonical nonparametric test for ordered alternatives (Terpstra 1952;
        Jonckheere 1954, Biometrika 41:133-145).
      - **Caveat (not SCED-validated)**: the only RT statistics empirically benchmarked for
        single-case ALTERNATION designs are MD and NAP; PND is rejected (Michiels, Heyvaert
        & Onghena 2017, Behavior Research Methods). A trend statistic was NOT in that
        simulation, so its SCED power/Type-I properties are not established. Therefore
        report it ALONGSIDE the omnibus (MD/NAP), and pre-specify it a priori to avoid
        post-hoc measure selection (Manolov, Moeyaert & Fingerhut 2021).

    ``condition_order`` : conditions from low to high (e.g. ("A","B","C")).
    ``dose``            : numeric level per condition (default 0,1,2,...). Pass e.g.
                          (4,5,6) so the ``slope`` reads in **outcome units per second of
                          MI** - directly interpretable.
    ``kind``            : ``"slope"`` (OLS slope on dose; larger = stronger beneficial
                          trend) or ``"jonckheere"`` (Jonckheere-Terpstra J; larger =
                          stronger beneficial trend).
    ``improvement``     : ``"increase"`` (higher outcome = better) or ``"decrease"``; the
                          statistic is oriented so larger always = more beneficial trend,
                          so the RT is run one-sided (``two_sided=False``).

    References: Terpstra 1952; Jonckheere 1954 (ordered-alternatives test); Edgington &
    Onghena 2007 (randomization framework).
    R equivalent: clinfun::jonckheere.test / DescTools::JonckheereTerpstraTest (jonckheere);
    base lm slope on the dose (slope) - potential equivalents, to test.
    """
    order = list(condition_order)
    pos = {c: (float(dose[i]) if dose is not None else float(i)) for i, c in enumerate(order)}
    sgn = 1.0 if improvement == "increase" else -1.0

    def _slope(labels, values):
        x = np.array([pos.get(l, np.nan) for l in labels], float)
        y = sgn * np.asarray(values, float)                 # orient so larger = beneficial
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2 or np.nanstd(x) == 0:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    def _jonckheere(labels, values):
        y = sgn * np.asarray(values, float)                 # orient so larger = beneficial
        groups = [y[labels == c] for c in order]
        groups = [g[np.isfinite(g)] for g in groups]
        total = 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):             # ordered pair low<high
                gi, gj = groups[i], groups[j]
                if len(gi) and len(gj):
                    d = gj[:, None] - gi[None, :]
                    total += np.sum(d > 0) + 0.5 * np.sum(d == 0)
        return float(total)

    if kind == "slope":
        return _slope
    if kind == "jonckheere":
        return _jonckheere
    raise ValueError("kind must be 'slope' or 'jonckheere'")


def stratified_ordered_trend_test(values, labels, units, *, condition_order, dose=None,
                                  kind="slope", improvement="increase", standardize=True,
                                  max_consecutive=None, block_size=None, n_perm=5000,
                                  random_state=0):
    """
    GROUP (replicated single-case) ordered-trend / dose-response test - the **stratified**
    counterpart of ``make_ordered_trend_stat`` (which serves one unit through
    ``randomization_test``). Tests a monotone dose effect across several units (e.g. the
    5 BCINET pilot patients, MI time 4/5/6 s).

    Each unit was randomised independently, so labels are permuted **within each unit** - 
    the exact joint reference set under the sharp null (Edgington & Onghena 2007). Outcomes
    are z-scored within unit by default (``standardize``), which is permutation-invariant
    (the per-unit value set is fixed), so the test stays exact while units contribute on a
    comparable scale. The pooled statistic is the ordered-trend statistic computed across
    all units' rows on the standardized values (pass ``dose`` so the slope reads per dose
    unit on the z-scale).

    Same caveat as the single-unit version: this **complements** the omnibus stratified F,
    it does not replace it; an ordered/trend statistic is NOT power-validated for SCED
    alternation designs (Michiels et al. 2017 benchmarked only MD/NAP), so report it
    alongside and pre-specify it a priori (Manolov et al. 2021).

    References: Terpstra 1952; Jonckheere 1954 (ordered-alternatives test); Edgington &
    Onghena 2007 (stratified randomization); Michiels et al. 2017 (SCED power caveat).
    R equivalent: no direct stratified equivalent (clinfun::jonckheere.test per unit + SCRT)
    - potential equivalent, to test.

    ``block_size`` permutes within consecutive blocks **inside each unit** (randomized-block
    schedule, e.g. one of each condition per block); else labels are permuted within unit
    preserving each condition's count (``max_consecutive`` optional). Requires the rows to
    be in session order within unit (the pipelines guarantee this). One-sided (larger =
    stronger beneficial trend). Returns ``{observed, p_value, n_perm, kind, n_units,
    two_sided}``.
    """
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    units = np.asarray(units)
    ok = np.isfinite(values)
    values, labels, units = values[ok], labels[ok], units[ok]
    _validate_condition_order(condition_order, labels)
    y = _standardize_within(values, units) if standardize else values.copy()
    stat = make_ordered_trend_stat(condition_order, kind=kind, dose=dose,
                                   improvement=improvement)
    obs = stat(labels, y)
    if not np.isfinite(obs):
        return {"observed": float(obs), "p_value": float("nan"), "n_perm": int(n_perm),
                "kind": kind, "n_units": int(len(pd.unique(units))), "two_sided": False,
                "note": "observed statistic undefined (NaN)"}
    uniq_u = pd.unique(units)
    idx = {u: np.where(units == u)[0] for u in uniq_u}
    scheme = (block_scheme(block_size) if block_size
              else alternating_scheme(max_consecutive=max_consecutive))
    rng = np.random.RandomState(random_state)
    ge = n_valid = 0
    for _ in range(n_perm):
        lab = labels.copy()
        for u in uniq_u:
            lab[idx[u]] = scheme(labels[idx[u]], rng)
        st = stat(lab, y)
        if not np.isfinite(st):
            continue
        n_valid += 1
        if st >= obs - 1e-12:
            ge += 1
    return {"observed": round(float(obs), 4), "p_value": (1 + ge) / (1 + n_valid),
            "n_perm": int(n_valid), "kind": kind, "n_units": int(len(uniq_u)),
            "two_sided": False}


def randomization_test(labels, values, *, statistic, scheme, n_perm=5000,
                       random_state=0, two_sided=True):
    """
    Generic Monte-Carlo randomization test (Edgington & Onghena 2007).

    ``statistic(labels, values) -> float`` is computed on the observed assignment,
    then under ``n_perm`` draws of ``scheme(labels, rng) -> permuted_labels`` (the
    randomization schedule that could have been used). The schedule, not a free
    shuffle, is what makes the test exact for the design.

    Returns ``{observed, p_value, n_perm, two_sided}`` with
    ``p = (1 + #{stat* as-or-more extreme}) / (1 + n_perm)``.

    References: Edgington & Onghena 2007 (randomization test); Phipson & Smyth 2010
    (p-value never zero).
    R equivalent: coin::independence_test; perm; SCRT (SCED design-based).
    """
    labels = np.asarray(labels)
    values = np.asarray(values, float)
    obs = statistic(labels, values)
    if not np.isfinite(obs):
        # an undefined observed statistic (e.g. a degenerate/mis-specified statistic)
        # must NOT be reported as p = 1/(1+n_perm) (the most significant value) - return
        # NaN so the caller surfaces the problem instead of a spurious "significant" result.
        return {"observed": float(obs), "p_value": float("nan"), "n_perm": int(n_perm),
                "two_sided": bool(two_sided), "note": "observed statistic undefined (NaN)"}
    rng = np.random.RandomState(random_state)
    ge = n_valid = 0
    for _ in range(n_perm):
        perm = scheme(labels, rng)
        st = statistic(perm, values)
        if not np.isfinite(st):
            continue                                  # drop from BOTH num. and denom.
        n_valid += 1
        if two_sided:
            if abs(st) >= abs(obs) - 1e-12:
                ge += 1
        elif st >= obs - 1e-12:
            ge += 1
    return {"observed": float(obs), "p_value": (1 + ge) / (1 + n_valid),
            "n_perm": int(n_valid), "two_sided": bool(two_sided)}


def frozen_sd_statistic(values, treated, *, standardize=True):
    """Mean-difference statistic whose standardizing SD is FROZEN on the observed data.

    In : the observed ``values`` and a boolean mask of the treated observations. Out : a callable
    ``statistic(labels_mask, values) -> float`` usable by ``randomization_test`` and
    ``randomization_interval``. No side effect.

    WHY THE SD MUST BE FROZEN. Inverting a test means re-running it on data from which a candidate
    effect has been subtracted. If the standardizing SD is recomputed on those shifted data, it
    GROWS with the candidate, the reference distribution is rescaled candidate by candidate, and
    the interval is judged against a moving yardstick - Michiels et al. (2017, p. 368) : the SD
    "is the standard deviation of all observed data before Delta is subtracted and is thus a
    constant for the calculation of the test statistic for each assignment and for each tested
    Delta", because the unit-treatment additivity model has no variance parameter to move.

    ``standardize=False`` returns the raw mean difference, for which the question does not arise.
    """
    sd = float(np.std(np.asarray(values, float), ddof=1)) if standardize else 1.0
    sd = sd if sd > 0 else 1.0

    def statistic(mask, v):
        m = np.asarray(mask, bool)
        v = np.asarray(v, float)
        if m.all() or not m.any():
            return float("nan")
        return float((v[m].mean() - v[~m].mean()) / sd)

    statistic.frozen_sd = sd
    return statistic


def randomization_interval(labels, values, *, statistic, scheme, treated, confidence=0.95,
                           n_perm=2000, random_state=0, precision=1e-3, max_expand=40,
                           max_iter=60, block_size=None, max_consecutive=None):
    """Confidence interval by RANDOMIZATION TEST INVERSION (Michiels et al. 2017).

    In : the observed ``labels`` (whatever ``scheme`` and ``statistic`` expect), the ``values``, the
    test statistic, the randomization ``scheme``, and ``treated``, a boolean mask of the
    observations the treatment applies to. Out : ``{estimate, ci_low, ci_high, confidence, p_value,
    n_perm, m_t, p_floor, effect_model, note}``. Side effect : none.

    THE PRINCIPLE. A two-sided level-alpha test and a 100(1 - alpha)% interval are the same object
    read twice : the interval is the set of point null values the test does NOT reject
    (Neyman 1937). So one tests a range of candidate effects and keeps the survivors -
    Michiels et al. (2017, p. 365).

    THE EFFECT MODEL, which is the price of an interval. Testing "the effect equals Delta" requires
    reconstructing the data as they would be under that hypothesis. Here the unit-treatment
    additivity model is used : the treatment shifts every treated score by a constant, so the null
    data are ``values - Delta`` on the treated observations - (p. 368). This assumption is NOT
    needed for a bare p-value, and the paper draws the general moral : "CIs require an additional
    assumption (viz., the hypothetical effect function), as compared to a bare-bones significance
    test. As such the advocacy for the 'new statistics' ... is far too simplistic" - (p. 375).
    A trend or a unit-varying effect would need a different reconstruction, which this function
    does not implement ; it refuses to guess one.

    THE FLOOR. A design whose schedule admits ``m_t`` assignments cannot attain a p below
    ``1 / (m_t + 1)``. When that floor exceeds alpha, NO interval exists at the requested
    confidence, and this function raises rather than returning a bracket that means nothing -
    Michiels et al. (2017, p. 376) on the same arithmetic for AB designs.

    CONSERVATIVE, NOT EXACT, at a round confidence level. The reference set is finite, so its
    attainable p-values form a grid, and a requested alpha rarely lands on it. Ernst (2004,
    Statistical Science 19(4):676-685, p. 680) : the interval is exact "if alpha is chosen as one
    of the achievable p values in the randomization distribution", and "otherwise, they are
    conservative intervals where the true coverage probability is at least as large as the chosen
    confidence coefficient". Erring toward over-coverage, which is the safe direction, but a
    reported "95 per cent" is a floor rather than an equality. Ernst quotes his own examples at
    94.29 and 94.90 per cent rather than rounding, which is the honest habit here too.

    THE TWO-SIDED CONVENTION is inherited from ``randomization_test``, which compares |statistic|
    against |observed|, i.e. distance from ZERO. That matches the equal-tailed construction only
    when the reference distribution is centred at zero, as it is for a difference in means under a
    count-preserving relabeling. For a statistic whose null distribution is asymmetric or off-centre
    it is the wrong measure of extremeness : "these distributions are not symmetric and so there is
    no justification for doubling the probability in one tail", and the distance should be taken
    from the MEAN of the reference distribution (Ernst 2004, p. 682). Not implemented ; a statistic
    with an off-centre null should not be handed to this function as it stands.

    MONTE-CARLO NOISE. Every candidate is tested with the SAME generator seed, so two candidates
    differ by their data and not by their draws. With independent seeds the accept / reject
    boundary would jitter by the Monte-Carlo error of p and the reported bounds would not be
    reproducible.

    The search brackets each side by expanding away from the estimate until rejection, then
    bisects to ``precision``. The bounds are found INDEPENDENTLY on each side rather than
    mirrored around the estimate, so an asymmetric acceptance region is reported as it is.

    THE STATISTIC MUST BE EFFECT INCREASING, and this is a condition on the CALLER's choice, not on
    this function. An inversion interval covers at the stated rate only for a statistic that cannot
    DECREASE when a positive constant is added to the treated units (Luo et al. 2021; Caughey et al.
    2021, via Fiksel 2024, Biometrics 80(2):ujae051, p. 3). The difference in means and the Wilcoxon
    rank sum qualify. STUDENTIZED statistics do not - "studentized test statistics are not
    effect-increasing, and thus we cannot guarantee exact coverage" (Fiksel 2024, p. 6) - so passing
    a t, or any ratio whose denominator is recomputed from the shifted data, voids the guarantee.
    ``frozen_sd_statistic`` is safe because a frozen denominator is a positive CONSTANT multiple of
    the difference in means, which preserves the property ; the same freeze is required for a second,
    independent reason in Michiels et al. (2017, p. 368). An adjusted (ANCOVA) coefficient is not
    guaranteed effect increasing either, and Fiksel's Theorem 1 (p. 3) gives a condition checkable
    on the observed data. No check is performed here : the statistic is a black box to this function.

    THIS SEARCH IS THE SLOW ONE, and knowingly so. It runs a FULL randomization test (`n_perm`
    draws) at every candidate, which is exactly the procedure Garthwaite (1996, Biometrics
    52:1387-1393, p. 1387) calls unsophisticated and blames for the rarity of these intervals in
    practice. Two better routes exist :

      - ROBBINS-MONRO (Garthwaite 1996, p. 1388, Eq. 2.1-2.2) : draw ONE permutation per candidate
        and step by c*alpha/i or c*(1-alpha)/i according to the side it falls on. Each step reduces
        the expected distance to the endpoint, the sequence converges in probability to it (Blum
        1954), and a whole endpoint costs "only slightly more permutations than the number needed
        for a randomization test" - about 6000 steps where the naive search spends 5000 permutations
        at each of seven candidates. Needs a step-length constant, starting values, a restart rule
        and monitoring of the realised tail proportion.
      - CLOSED FORM (Zhu & Liu 2022 ; Fiksel 2024, p. 3) where the statistic's crossing equation is
        solvable : solve per permutation for the candidate at which the permuted statistic equals
        the observed one, sort, read the bounds as order statistics. Measured at 8 per cent of the
        p-value's own cost, against roughly 30x for bisection (Fiksel 2024, p. 5, Table 1).

    The bisection is kept for now because it accepts an ARBITRARY statistic and an arbitrary
    schedule, needs no tuning, and is deterministic given the seed. Its cost is `2 * n_candidates *
    n_perm` statistic evaluations, which is fine for one scalar contrast and prohibitive at map
    scale. Validated against Garthwaite's own published example, see tests.

    KNOWN LIMIT : the acceptance set need not be an interval. The reference set is finite, so p is
    a STEP function of the candidate and, with a two-sided |statistic| comparison, not necessarily
    monotone - measured on the worked example of the source paper, the exact p is 0.066 at one
    candidate and 0.026 a little further out. Bisection therefore returns the FIRST crossing on
    each side, i.e. the convex hull of the acceptance set, which is the conservative reading. A
    non-convex acceptance region is not reported as such.
    """
    labels = np.asarray(labels)
    values = np.asarray(values, float)
    treated = np.asarray(treated, bool)
    alpha = 1.0 - float(confidence)
    counts = count_admissible_assignments(labels, block_size=block_size,
                                          max_consecutive=max_consecutive)
    if np.isfinite(counts["p_floor"]) and counts["p_floor"] > alpha:
        raise ValueError(
            f"randomization_interval : the design admits {counts['m_t']} assignment(s), so the "
            f"smallest attainable p is {counts['p_floor']:.4f}, above alpha = {alpha:.4f}. No "
            f"{100 * confidence:.0f}% interval exists for this design - the test can never reject, "
            "whatever the effect. Report a lower confidence level, or a p-value alone.")

    def p_of(delta):
        """p of the randomization test on the data with `delta` removed from the treated units."""
        shifted = values - delta * treated
        return randomization_test(labels, shifted, statistic=statistic, scheme=scheme,
                                  n_perm=n_perm, random_state=random_state,
                                  two_sided=True)["p_value"]

    # THE SEARCH RUNS IN DATA UNITS, NOT IN THE STATISTIC'S UNITS. The candidate is a SHIFT applied
    # to the treated observations, so it lives on the scale of `values`. Centring the search on a
    # standardized estimate instead would subtract 1.5 where the data need 3.4, leave a residual
    # effect in every candidate, reject them all, and collapse the interval onto the estimate.
    delta_hat = float(values[treated].mean() - values[~treated].mean())
    est = float(statistic(labels, values))
    p_obs = p_of(0.0)
    scale = float(np.std(values, ddof=1)) or 1.0
    tol = float(precision) * scale

    def bound(direction):
        """Largest (direction=+1) or smallest (direction=-1) retained candidate shift."""
        lo = delta_hat                                  # retained by construction: shifting by the
        step = scale                                    # observed difference nulls the treated group
        hi = None
        for _ in range(max_expand):                     # expand until one candidate is rejected
            cand = lo + direction * step
            if p_of(cand) < alpha:
                hi = cand
                break
            lo, step = cand, step * 2.0
        if hi is None:
            return float("inf") * direction             # never rejected within the search range
        for _ in range(max_iter):                       # bisect between retained lo and rejected hi
            if abs(hi - lo) <= tol:
                break
            mid = 0.5 * (lo + hi)
            if p_of(mid) < alpha:
                hi = mid
            else:
                lo = mid
        return lo

    lo_raw, hi_raw = bound(-1), bound(+1)
    out = {"estimate": delta_hat, "ci_low": lo_raw, "ci_high": hi_raw,
           "estimate_statistic": est, "units": "data units (the shift applied to the treated)",
           "confidence": float(confidence), "p_value": float(p_obs), "n_perm": int(n_perm),
           "m_t": counts["m_t"], "p_floor": counts["p_floor"],
           "effect_model": "unit-treatment additivity (constant shift on the treated units)",
           "note": "interval by randomization test inversion (Michiels et al. 2017); the effect "
                   "model is an assumption the p-value alone does not require"}
    # A statistic that carries its FROZEN standardizing SD (frozen_sd_statistic) is a constant
    # multiple of the shift, so its interval is the shift interval rescaled - exactly, not
    # approximately. Reported alongside rather than instead: the two scales answer different
    # questions and mixing them up is how a d gets read as a raw difference.
    sd = getattr(statistic, "frozen_sd", None)
    if sd:
        out.update({"ci_low_standardized": lo_raw / sd, "ci_high_standardized": hi_raw / sd,
                    "frozen_sd": float(sd)})
    return out


_MAX_REJECTION_DRAWS = 200          # rejection-sampling budget per constrained draw


def alternating_scheme(max_consecutive=None):
    """
    Randomization schedule for an alternating/N-of-1 design: reshuffle the
    condition labels while preserving each condition's count (the random
    assignment of conditions to sessions). With ``max_consecutive``, draws that
    exceed that run length of an identical condition are rejected (mirrors the
    typical no-more-than-k-in-a-row constraint). Returns a ``scheme(labels, rng)``.

    Rejection sampling keeps the draw UNIFORM over the constrained set, which is what makes
    the test exact; when the budget of ``_MAX_REJECTION_DRAWS`` attempts is exhausted the
    scheme RAISES rather than returning an unconstrained draw, since mixing inadmissible
    assignments into the reference set silently invalidates the p-value. Use
    ``count_admissible_assignments`` to check whether the constrained set is large enough
    to be sampled (and to be tested) at all.

    References: Edgington & Onghena 2007 (completely-randomized SCED alternation schedule).
    R equivalent: SCRT (CRD alternation) - potential equivalent, to test.
    """
    def _scheme(labels, rng):
        if max_consecutive is None:
            return rng.permutation(labels)
        for _ in range(_MAX_REJECTION_DRAWS):
            perm = rng.permutation(labels)
            runs = 1
            ok = True
            for i in range(1, len(perm)):
                runs = runs + 1 if perm[i] == perm[i - 1] else 1
                if runs > max_consecutive:
                    ok = False
                    break
            if ok:
                return perm
        raise RuntimeError(
            f"alternating_scheme: no draw satisfying max_consecutive={max_consecutive} was "
            f"found in {_MAX_REJECTION_DRAWS} attempts. The constraint is too tight for "
            "rejection sampling on this label vector, so the reference set cannot be sampled "
            "uniformly. Call count_admissible_assignments(labels, max_consecutive=...) to see "
            "how small the admissible set is (a design whose p-floor exceeds alpha cannot "
            "support a randomization test at all), or use block_scheme(k) if the schedule was "
            "block-randomised. Previously this fell back to an UNCONSTRAINED draw, which "
            "silently mixed inadmissible assignments into the null.")
    return _scheme


def _multiset_permutations(items):
    """Yield each DISTINCT ordering of ``items`` exactly once (a multiset permutation), unlike
    ``itertools.permutations`` which yields n! orderings with duplicates whenever labels repeat.

    In: a sequence of hashable labels (the observed condition assignment).
    Out: a generator of tuples. No side effect. Used only for counting/enumerating small
    admissible sets, never in the permutation loop itself."""
    pool = sorted(set(items), key=str)
    counts = {c: list(items).count(c) for c in pool}
    n = len(items)
    out = []

    def _rec():
        if len(out) == n:
            yield tuple(out)
            return
        for c in pool:
            if counts[c]:
                counts[c] -= 1
                out.append(c)
                yield from _rec()
                out.pop()
                counts[c] += 1

    return _rec()


def count_admissible_assignments(labels, *, block_size=None, max_consecutive=None,
                                 enumerate_cap=200000):
    """Size of the DESIGN's reference set ``m_t`` (the number of distinct condition assignments
    the randomisation schedule could have produced) and the resulting p-value floor.

    In: the observed ``labels`` (condition per session, in session order) plus the schedule
    constraints actually used. Out: ``{m_t, exact, p_floor, basis}``; no side effect.

    - unconstrained count-preserving relabeling -> multinomial ``n! / prod(n_c!)``, exact;
    - ``block_size`` -> product of the per-block multinomials, exact (a block whose labels
      repeat has fewer than ``block_size!`` distinct orders);
    - ``max_consecutive`` -> has no closed form, so the distinct assignments are enumerated
      and filtered when the multinomial is at most ``enumerate_cap``; beyond that the
      multinomial is returned as an UPPER BOUND with ``exact=False``.

    ``p_floor = 1 / (m_t + 1)`` is the smallest p the design can attain; when it exceeds the
    intended alpha the study cannot yield a significant randomization test whatever the effect
    size (Heyvaert & Onghena 2014: at least ~20 admissible assignments are needed at alpha=.05;
    Levin et al. 2016; Phipson & Smyth 2010 for the (1+k)/(1+B) convention this floor sits
    under). Reported for information: the engines keep the Monte-Carlo p, they do not switch
    to k/m_t."""
    labels = list(np.asarray(labels))
    n = len(labels)
    if n == 0:
        return {"m_t": 0, "exact": True, "p_floor": np.nan, "basis": "empty"}

    def _multinomial(seq):
        m = math.factorial(len(seq))
        for c in set(seq):
            m //= math.factorial(list(seq).count(c))
        return m

    if block_size:
        m_t, basis, exact = 1, f"within-block permutation (block_size={block_size})", True
        for start in range(0, n, block_size):
            m_t *= _multinomial(labels[start:start + block_size])
    elif max_consecutive is not None:
        total = _multinomial(labels)
        if total <= enumerate_cap:
            m_t, exact = 0, True
            for perm in _multiset_permutations(labels):
                runs, ok = 1, True
                for i in range(1, n):
                    runs = runs + 1 if perm[i] == perm[i - 1] else 1
                    if runs > max_consecutive:
                        ok = False
                        break
                m_t += 1 if ok else 0
            basis = f"count-preserving relabeling with max_consecutive={max_consecutive}"
        else:
            m_t, exact = total, False
            basis = (f"UPPER BOUND: unconstrained multinomial ({total} > enumerate_cap="
                     f"{enumerate_cap}); the max_consecutive={max_consecutive} subset is smaller")
    else:
        m_t, exact, basis = _multinomial(labels), True, "count-preserving relabeling"
    return {"m_t": int(m_t), "exact": bool(exact),
            "p_floor": (1.0 / (m_t + 1)) if m_t else np.nan, "basis": basis}


def block_scheme(block_size):
    """
    Randomized-block schedule (RBD; Edgington & Onghena 2007): sessions are grouped into
    consecutive blocks of ``block_size`` and the condition labels are permuted **within
    each block** only. With ``block_size`` = number of conditions, each block holds one of
    each condition in a random order - the canonical randomized-block alternating design.
    The reference distribution is the within-block permutations, which is what makes the
    test exact for that constrained schedule. Returns a ``scheme(labels, rng)``.

    References: Edgington & Onghena 2007 (randomized-block SCED schedule).
    R equivalent: SCRT (RBD alternation) - potential equivalent, to test.
    """
    def _scheme(labels, rng):
        out = np.array(labels).copy()
        for start in range(0, len(out), block_size):
            block = out[start:start + block_size]
            out[start:start + block_size] = rng.permutation(block)
        return out
    return _scheme


def sced_test_name(*, n_conditions, detrend, grouped, method=None):
    """
    Canonical NAME of the primary condition test (not just its statistic), for reports.

    A SCED condition test by randomization is, in classical terms, an ANOVA/ANCOVA on a
    linear model whose p-value comes from permutation instead of the F distribution: the
    ANOVA F IS the partial-F of the categorical condition block (ANOVA = OLS with
    categorical predictors; ANCOVA = + a continuous covariate). Naming rule:
      - a time covariate present (``detrend`` != none) -> **ANCOVA**; else **ANOVA**;
      - grouped (patient = within-subject block) -> "repeated-measures";
      - exactly 2 conditions, no covariate -> the ANOVA reduces to a **difference of means**.
    The permutation scheme (Freedman-Lane / Draper-Stoneman) is appended when a covariate
    is present, because that scheme is what realises a permutation ANCOVA (Freedman & Lane
    1983; Winkler et al. 2014). Design-based exactness: Edgington & Onghena (2007).
    """
    has_cov = detrend not in (None, "none")
    if n_conditions <= 2 and not has_cov:
        return ("Stratified randomization test - mean difference" if grouped
                else "Randomization test - mean difference")
    name = ("ANCOVA" if has_cov else "ANOVA") + (" repeated-measures" if grouped
                                                 else " one-way") + " by permutation"
    if has_cov and method:
        m = ("Freedman-Lane" if "freedman" in str(method).lower()
             else "Draper-Stoneman" if "draper" in str(method).lower() else str(method))
        name += f" ({m})"
    return name


def sced_methods_text(*, design, n_units, conditions, outcome, detrend, method, n_perm,
                      standardize=None, improvement="increase", statistics_used=None,
                      dose=None):
    """
    Generate a **Methods** paragraph (scientific-article style, English) describing the
    SCED analysis actually run, with citations - drop-in for a manuscript's Statistical
    analysis subsection. Composed from the realised configuration so it never overstates
    what was done. ``statistics_used`` is an optional list of extra analyses
    (e.g. "trend", "heterogeneity", "posthoc") to mention.
    """
    stats_used = set(statistics_used or [])
    grouped = (n_units or 1) > 1
    conds = ", ".join(map(str, conditions))
    test = sced_test_name(n_conditions=len(conditions), detrend=detrend,
                          grouped=grouped, method=method)
    s = []
    s.append(
        f"The outcome '{outcome}' was analysed using a single-case experimental design "
        f"{'replicated across ' + str(n_units) + ' patients' if grouped else '(one patient)'} "
        f"({design}), with {len(conditions)} conditions ({conds}) randomly assigned "
        "to the sessions. The primary inference is a randomization test (Edgington & "
        "Onghena, 2007): the observed statistic is compared to its distribution over "
        f"the set of admissible assignments (B = {n_perm} permutations; "
        "p = (1 + #{stat* >= stat}) / (1 + B), Phipson & Smyth, 2010).")
    if detrend not in (None, "none"):
        m = ("Freedman-Lane" if "freedman" in str(method).lower()
             else "Draper-Stoneman" if "draper" in str(method).lower() else str(method))
        s.append(
            f"A temporal trend ({detrend}) was modelled as a nuisance covariate; "
            f"the condition effect is therefore estimated by a {test}, the p-value being "
            f"obtained by permuting the reduced-model residuals ({m}; Freedman & Lane, "
            "1983; Winkler et al., 2014). The randomisation makes the test exact "
            "regardless of autocorrelation; the covariate only serves to restore "
            "the power lost to the learning drift.")
    else:
        s.append(f"The condition effect is tested by a {test}.")
    if grouped:
        s.append(
            "At the group level, the patient serves as a block: the labels are permuted "
            "within each patient (stratified test)"
            + (", after within-patient standardisation (z-score) to make the patients "
               "comparable" if standardize else "") + ".")
    if "heterogeneity" in stats_used:
        s.append(
            "Between-patient heterogeneity of the effect was assessed by a permutation "
            "test of the patient x condition interaction and by Cochran's Q / I^2 "
            "(Higgins & Thompson, 2002).")
    if "posthoc" in stats_used:
        s.append(
            "In case of an overall effect, pairwise contrasts (all combinations) "
            "were estimated (Hedges' g + bootstrap CI; Hedges, 1981) with Holm "
            "(FWER) and Benjamini-Hochberg (FDR) multiplicity correction.")
    if "trend" in stats_used:
        dose_txt = f" (doses {list(dose)})" if dose is not None else ""
        s.append(
            f"The conditions being ordered levels of a dose{dose_txt}, a monotone "
            "dose-response trend was tested IN ADDITION (slope on the dose and "
            "Jonckheere-Terpstra; Terpstra, 1952; Jonckheere, 1954), to be interpreted "
            "with caution because it is not validated for alternating designs "
            "(Michiels et al., 2017) and pre-specified a priori (Manolov et al., 2021).")
    s.append(
        "The nonoverlap effect sizes (NAP, Parker & Vannest, 2009; bounded Tau-U, "
        "Brossard et al., 2018 / Tarlow, 2016) complement the test descriptively. "
        "The retained improvement direction is '" + improvement + "'.")
    return " ".join(s)


def perm_boot_slope_band(y, design, effect_col, units, predict_grid, *,
                         n_perm=2000, n_boot=2000, seed=0):
    """Design-based inference for a linear-model SLOPE, resampling-consistent (no Wald / normal
    approximation) - the pair a SCED figure needs : a permutation p AND a matching confidence band.

      p    : within-unit FREEDMAN-LANE permutation of the slope. Fit the reduced model (``design``
             without ``effect_col``), permute its residuals WITHIN each unit, refit the full model and
             read the slope ; the two-sided p is the share of |permuted slope| >= |observed|. Exact
             under the null exchangeability of the within-unit residuals (Freedman & Lane, 1983).
      band : within-unit RESIDUAL BOOTSTRAP of the group-mean prediction over ``predict_grid``. Resample
             the full-model residuals WITH replacement within each unit, refit, predict ; the 95% band
             is the 2.5-97.5 percentiles across resamples (Efron & Tibshirani, 1993). Full prediction
             band (level + slope uncertainty) -> the classic hourglass with a NON-ZERO centre, unlike a
             slope-only band. Both null and band condition on the observed units (fixed-effect /
             conditional inference), so they are mutually consistent.

    Parameters
    ----------
    y : (n,) outcome.
    design : (n, p) full model design matrix (e.g. [unit dummies | effect | nuisance...]).
    effect_col : int, the column of ``design`` holding the tested slope covariate.
    units : (n,) unit (patient) label per row - defines the within-unit exchangeability blocks.
    predict_grid : (g, p) design rows at which to predict the GROUP-MEAN line (e.g. unit dummies set to
        1/n_units for the average intercept, the effect column swept over the axis, nuisance at ref/mean).
    n_perm, n_boot : permutation / bootstrap resample counts. seed : RNG seed.

    Returns
    -------
    dict : ``{"slope", "p", "band_lo", "band_hi", "yhat", "n_perm", "n_boot"}`` where ``yhat`` is the
    observed group-mean line over ``predict_grid`` and ``band_lo``/``band_hi`` are the (g,) bootstrap
    percentiles.

    References: Freedman & Lane (1983) - residual permutation of a nuisance-adjusted slope - catalogued
    and benchmarked against the other GLM-coefficient permutation schemes (Draper-Stoneman, Huh-Jhun,
    ter Braak...) by Helwig (2019, doi:10.1016/j.neuroimage.2019.116030). The band is a percentile
    bootstrap CI (Bruce, Bruce & Gedeck, Practical Statistics for Data Scientists, 2020) ; the
    within-unit residual variant is the standard extension for clustered data. R equivalent:
    permuco::lmperm (freedman_lane) for the p ; boot::boot for the band.
    """
    y = np.asarray(y, float); X = np.asarray(design, float); units = np.asarray(units)
    G = np.asarray(predict_grid, float)
    ix = int(effect_col)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); slope = float(b[ix])
    fit_full = X @ b; resid_full = y - fit_full
    keep = [j for j in range(X.shape[1]) if j != ix]; Xr = X[:, keep]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None); fitr = Xr @ br; residr = y - fitr
    uu = list(dict.fromkeys(units.tolist())); idx = [np.where(units == u)[0] for u in uu]
    rng = np.random.RandomState(seed)
    # Freedman-Lane permutation p (within-unit residual permutation)
    cnt = 1
    for _ in range(int(n_perm)):
        rp = residr.copy()
        for gi in idx:
            pp = gi.copy(); rng.shuffle(pp); rp[gi] = residr[pp]
        bp, *_ = np.linalg.lstsq(X, fitr + rp, rcond=None)
        if abs(float(bp[ix])) >= abs(slope) - 1e-15:
            cnt += 1
    p = cnt / (int(n_perm) + 1)
    # residual bootstrap band (within-unit resampling with replacement). Also the bootstrap CI of EVERY
    # coefficient (forest bars), the resampling sibling of the band - the whole design in one pass.
    # NB: raw OLS residuals are slightly shrunk (by 1-h_ii), so the band is mildly anti-conservative
    # (conditional CI). A leverage rescaling r_i/sqrt(1-h_ii) (Wu 1986; Davidson & Flachaire 2008) would
    # correct this - deliberately NOT applied, to keep the method within the wiki's referenced bootstrap.
    preds = np.empty((int(n_boot), G.shape[0]))
    coefs = np.empty((int(n_boot), X.shape[1]))
    for bI in range(int(n_boot)):
        rb = resid_full.copy()
        for gi in idx:
            rb[gi] = resid_full[rng.choice(gi, size=gi.size, replace=True)]
        bb, *_ = np.linalg.lstsq(X, fit_full + rb, rcond=None)
        preds[bI] = G @ bb; coefs[bI] = bb
    return {"slope": slope, "p": p, "band_lo": np.percentile(preds, 2.5, axis=0),
            "band_hi": np.percentile(preds, 97.5, axis=0), "yhat": G @ b,
            "coef": b, "coef_lo": np.percentile(coefs, 2.5, axis=0),
            "coef_hi": np.percentile(coefs, 97.5, axis=0), "n_perm": int(n_perm), "n_boot": int(n_boot)}
