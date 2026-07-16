"""
SCED - Multivariate randomization test for alternating / N-of-1 designs
=======================================================================

Several outcomes measured per session (e.g. accuracy + reaction time + fatigue).
Classical MANOVA (Hotelling/Wilks) is infeasible at n=1 (needs n > p and an invertible
covariance); the **design-based randomization** route is not - it never inverts a
covariance, so it works even when p > n.

Implementation = **PERMANOVA** (Anderson 2001): a pseudo-F on a (Euclidean) distance
matrix, evaluated by permuting the condition labels. With Euclidean distance the
pseudo-F equals the trace-based MANOVA F, so it is computed directly from total/within
sums of squares summed over the standardized, residualized outcomes (no NxN matrix).
The time trend (and, for a group, the unit means) are removed first by residualization;
permutation is over the schedule (single) or **within unit** (group). An omnibus p is
reported, with **per-outcome univariate follow-ups** (Holm/FDR) to see which outcomes
drive the effect.

Reference: Anderson, M. J. (2001). A new method for non-parametric multivariate
analysis of variance. Austral Ecology, 26(1), 32-46.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from functions.sced.core import (
    alternating_scheme,
    condition_permutation_test,
    stratified_condition_permutation_test,
    _group_nuisance,
    _nuisance_basis,
)


def permanova_condition_test(df, *, outcome_cols, session_col, condition_col,
                             unit_col=None, conditions=None, time_covariate="none",
                             detrend=None, standardize=True, n_perm=5000,
                             max_consecutive=None, correction="all", random_state=0):
    """
    Multivariate (PERMANOVA-style) randomization test of a condition effect on several
    outcomes at once, plus per-outcome follow-ups. The observed pseudo-F is verified ==
    R ``vegan::adonis2`` (Euclidean, ``standardize=False``, no time covariate).

    ``outcome_cols`` : list of outcome columns. ``unit_col`` given -> group (permute
    within unit, unit means removed); else single series. ``detrend`` in {none, linear,
    log} residualizes each outcome on the time trend first; ``standardize`` z-scores
    each (recommended, so scales don't dominate the Euclidean distance). Returns
    ``{pseudo_F, p_value, df, n_perm, n_outcomes, n_units, per_outcome}`` where
    ``per_outcome`` is a DataFrame (one row per outcome) with the univariate
    permutation p and Holm/FDR-adjusted p.

    Missing data: rows with ANY missing outcome or session are dropped (**complete-case**,
    as a multivariate distance needs every dimension); the count is returned as
    ``n_dropped_missing``. If the observed pseudo-F is still undefined, ``p_value`` is NaN
    (never a spurious 1/(B+1)).

    References: Anderson (2001) (PERMANOVA / pseudo-F on distances); Holm (1979) and Benjamini and Hochberg (1995) (per-outcome multiplicity control).
    R equivalent: vegan::adonis2 (Euclidean PERMANOVA, numerically verified); stats::p.adjust for the Holm / FDR follow-ups.
    """
    # `detrend` renamed to `time_covariate` (backward-compatible alias)
    detrend = detrend if detrend is not None else time_covariate
    # "auto" (time diagnostic) applies only to the univariate engines; PERMANOVA has no
    # trend diagnostic -> "auto" is resolved to "none" (the canonical default).
    if detrend == "auto":
        detrend = "none"
    if conditions is None:
        conditions = sorted(df[condition_col].dropna().unique().tolist(), key=str)
    conditions = list(conditions)
    sub = df[df[condition_col].isin(conditions)]
    grouped = unit_col is not None
    # Complete-case: a multivariate test (on distances) requires ALL outcomes present;
    # a single NaN otherwise propagates a nan pseudo-F and a spurious p (comparisons with
    # nan are always false -> ge=0 -> p=1/(B+1)). Rows with a missing outcome or session
    # are therefore dropped (the univariate path already drops per column).
    Yall = np.column_stack([pd.to_numeric(sub[c], errors="coerce").to_numpy()
                            for c in outcome_cols]).astype(float)
    sess_all = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
    ok = np.isfinite(Yall).all(axis=1) & np.isfinite(sess_all)
    n_dropped = int((~ok).sum())
    sub = sub[ok]
    labels = sub[condition_col].to_numpy()
    sess = sess_all[ok]
    units = sub[unit_col].to_numpy() if grouped else np.zeros(int(ok.sum()))
    Y = Yall[ok]
    N, p = Y.shape

    # residualize each outcome on the nuisance (unit + time), then standardize
    if detrend != "none":
        Z = _group_nuisance(units, sess, detrend) if grouped else _nuisance_basis(sess, detrend)
    elif grouped:
        Z = np.column_stack([(units == u).astype(float) for u in np.unique(units)])
    else:
        Z = np.ones((N, 1))
    R = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
    if standardize:
        sd = R.std(axis=0, ddof=0); sd[sd == 0] = 1.0
        R = R / sd

    a = len(conditions)
    df_num, df_den = a - 1, N - a

    def pseudo_F(lab):
        sst = float(((R - R.mean(0)) ** 2).sum())
        ssw = 0.0
        for c in conditions:
            m = lab == c
            if m.any():
                ssw += float(((R[m] - R[m].mean(0)) ** 2).sum())
        if ssw <= 0 or df_den <= 0:
            return np.inf
        return ((sst - ssw) / df_num) / (ssw / df_den)

    F_obs = pseudo_F(labels)
    if np.isnan(F_obs):                       # guard: never return a spurious p
        return {"pseudo_F": np.nan, "p_value": np.nan, "df": (df_num, df_den),
                "n_perm": int(n_perm), "n_outcomes": p,
                "n_units": int(len(np.unique(units))) if grouped else 1,
                "per_outcome": pd.DataFrame(), "n_dropped_missing": n_dropped,
                "note": "pseudo-F undefined (degenerate data) - p undefined."}
    rng = np.random.RandomState(random_state)
    scheme = alternating_scheme(max_consecutive=max_consecutive)
    uniq_u = np.unique(units)
    idx = {u: np.where(units == u)[0] for u in uniq_u}
    ge = 0
    for _ in range(n_perm):
        if grouped:
            lab = labels.copy()
            for u in uniq_u:
                lab[idx[u]] = scheme(labels[idx[u]], rng)
        else:
            lab = scheme(labels, rng)
        if pseudo_F(lab) >= F_obs - 1e-12:
            ge += 1
    p_omni = (1 + ge) / (1 + n_perm)

    # per-outcome univariate follow-ups (same engine as the main analysis)
    rows, praw = [], []
    for c in outcome_cols:
        y = pd.to_numeric(sub[c], errors="coerce").to_numpy()
        if grouped:
            r = stratified_condition_permutation_test(
                y, labels, units, sess, detrend=detrend, method="freedman-lane",
                n_perm=n_perm, max_consecutive=max_consecutive, random_state=random_state)
        else:
            r = condition_permutation_test(
                y, labels, sess, detrend=detrend, method="freedman-lane",
                n_perm=n_perm, max_consecutive=max_consecutive, random_state=random_state)
        rows.append({"Outcome": c, "F": r["observed_F"], "p (perm)": round(r["p_value"], 4)})
        praw.append(r["p_value"])
    per = pd.DataFrame(rows)
    if praw:
        if correction in ("holm", "all"):
            per["p Holm"] = multipletests(praw, method="holm")[1].round(4)
        if correction in ("fdr_bh", "all"):
            per["p FDR"] = multipletests(praw, method="fdr_bh")[1].round(4)

    return {"pseudo_F": round(float(F_obs), 4), "p_value": p_omni,
            "df": (df_num, df_den), "n_perm": int(n_perm), "n_outcomes": p,
            "n_units": int(len(uniq_u)) if grouped else 1, "per_outcome": per,
            "n_dropped_missing": n_dropped}
