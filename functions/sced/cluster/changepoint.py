"""Change-point (relu / broken-line) primitives - DOMAIN-NEUTRAL, pure numpy.

A hinge basis relu_k(t) = max(0, t - k) tested for an UNKNOWN onset k by a sup over candidate
onsets, adjusting for an ARBITRARY reduced model Z (intercept + any covariates : a linear time
term, factor dummies...). No SCED, no cluster, no permutation here - the caller (relu_run /
scalar_relu) wraps these with a permutation null and an adjacency. So the general method lives on
its own and only the integration is SCED-specific.

References : segmented / broken-line regression (Muggeo 2003 ; Hinkley 1969) ; sup-statistic for a
breakpoint present only under the alternative (Davies 1977, 1987 ; Andrews 1993).
"""
import numpy as np


def candidate_onsets(time, onsets):
    """Default candidate onsets = the interior distinct time values (both segments non-empty)."""
    tvec = np.asarray(time, dtype=float)
    if onsets is None:
        u = np.unique(tvec)
        onsets = u[1:-1] if u.size >= 3 else u
    onsets = np.asarray(onsets, dtype=float)
    if onsets.size == 0:
        raise ValueError("no candidate onset (need >= 3 distinct time values)")
    return onsets


def hinge_fits(time, onsets, Z):
    """Per-onset full-model fits for the hinge, adjusting for the reduced model Z (n x q ; MUST
    include the intercept and every covariate to adjust for). Returns (Z, pZ, df2, fits, onsets)
    with fits[k] = (D_k, pinv(D_k), tcol_k), D_k = [Z, relu_k], relu_k(t) = max(0, t - onset_k).
    df2 = n - (q + 1) : the full model always adds exactly one hinge column."""
    tvec = np.asarray(time, dtype=float)
    onsets = candidate_onsets(tvec, onsets)
    Z = np.asarray(Z, dtype=float)
    pZ = np.linalg.pinv(Z)
    df2 = Z.shape[0] - (Z.shape[1] + 1)
    if df2 <= 0:
        raise ValueError("not enough observations for the hinge model")
    fits = []
    for k in onsets:
        relu = np.maximum(0.0, tvec - k)[:, None]
        D = np.hstack([Z, relu])
        fits.append((D, np.linalg.pinv(D), D.shape[1] - 1))
    return Z, pZ, df2, fits, onsets


def hinge_sup(Z, pZ, df2, fits, tail):
    """Return (statmap, t_per_k). statmap(Yv) = per-column SUP over onsets of the signed partial t
    of the hinge column beyond Z (F with 1 numerator df, t = sign(beta) sqrt(F)). tail selects the
    aggregation over onsets : 'pos' max, 'neg' min, 'both' largest magnitude."""
    def t_per_k(Yv, D, pD, tcol):
        rss_f = np.sum((Yv - D @ (pD @ Yv)) ** 2, axis=0)
        rss_r = np.sum((Yv - Z @ (pZ @ Yv)) ** 2, axis=0)
        F = (rss_r - rss_f) / np.maximum(rss_f / df2, 1e-12)   # p_x = 1 -> F = t^2
        beta = (pD @ Yv)[tcol]
        return np.sign(beta) * np.sqrt(np.maximum(F, 0.0))

    def statmap(Yv):
        ts = np.stack([t_per_k(Yv, D, pD, tcol) for D, pD, tcol in fits], axis=0)  # (K, E)
        if tail == "pos":
            return ts.max(axis=0)
        if tail == "neg":
            return ts.min(axis=0)
        idx = np.argmax(np.abs(ts), axis=0)             # largest magnitude for 'both'
        return np.take_along_axis(ts, idx[None, :], axis=0)[0]

    return statmap, t_per_k


def hinge_betas(Y, fits):
    """Per-onset hinge slope (beta of the relu column) for each element : (K, E). The slope is the
    post-onset rate of change (or the CHANGE of slope when Z carries a linear time term)."""
    return np.stack([(pD @ Y)[tcol] for _D, pD, tcol in fits], axis=0)
