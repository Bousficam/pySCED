"""
SCED - Multiple Baseline Design (family 2): randomization test on the intervention start
=========================================================================================

In a multiple-baseline design the intervention is introduced at **staggered** moments
across several *tiers* (participants, behaviours or settings): tier 1 starts treatment
early, tier 2 later, tier 3 later still. An effect is shown when each tier's outcome
changes *at* its own intervention point while the still-in-baseline tiers do not - this
rules out a shared time/maturation confound.

Primary inference = **randomization test on the intervention start points** (Edgington &
Onghena 2007; Marascuilo & Busk 1988): the experimenter pre-specifies, for each tier, a
set of *candidate* start sessions (the window the start could have been drawn from); the
start is then randomised within that set. The test reuses the actual randomisation: it
re-draws a start per tier from each tier's candidate set, recomputes a statistic
(average per-tier mean(B) - mean(A), oriented by ``improvement``) and compares the
observed value to that distribution. Enumerated exactly when the candidate space is
small, else Monte-Carlo.

Data: long, one row per (tier, session): ``tier | session | outcome`` (+ an optional
``phase`` A/B, or explicit ``intervention_starts``). Non-overlap effect sizes (Tau-U,
NAP) are reported per tier in support.
"""
import itertools
import math

import numpy as np
import pandas as pd

from functions.sced.core import tau_u, nap


def _fmt_p(p, digits=4):
    """Format a p-value for display: never shows 0 (a p is always > 0). Returns '<0.0001'
    for a positive p below the display resolution (parametric underflow OR the randomization
    floor 1/(1+B)), the rounded float otherwise, and None if missing."""
    if p is None:
        return None
    try:
        v = float(p)
    except (TypeError, ValueError):
        return p
    if not np.isfinite(v):
        return None
    thr = 10.0 ** (-digits)
    return f"<{thr:g}" if 0 <= v < thr else round(v, digits)


def _tier_series(df, tier_col, session_col, outcome_col, phase_col=None, keep_phases=None):
    """Per-tier (session, outcome) arrays, session-sorted. Inputs: long frame plus column names;
    ``keep_phases`` restricts to those phase labels when ``phase_col`` is given. Output: dict
    ``{tier: (session_array, outcome_array)}``; tiers with no finite outcome are dropped."""
    out = {}
    for t, g in df.groupby(tier_col):
        g = g.sort_values(session_col)
        if keep_phases is not None and phase_col is not None:
            g = g[g[phase_col].isin(keep_phases)]
        s = pd.to_numeric(g[session_col], errors="coerce").to_numpy()
        y = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
        if not np.isfinite(y).any():           # tier with NO data at all for this outcome -> excluded
            continue                           # (otherwise: phantom tiers -> inflated randomization space)
        out[t] = (s, y)
    return out


def _auto_candidates(series, min_baseline, min_treatment):
    """Per-tier candidate start sessions: a start at the i-th ordered session is allowed
    if >=min_baseline sessions precede it and >=min_treatment follow (incl. it)."""
    cand = {}
    for t, (s, _) in series.items():
        opts = [int(s[i]) for i in range(len(s))
                if i >= min_baseline and (len(s) - i) >= min_treatment]
        cand[t] = opts if opts else [int(s[len(s) // 2])]
    return cand


def _phase_slope(s, y):
    """Ordinary least-squares slope of ``y`` on session ``s`` within one phase. Returns 0.0 when
    fewer than two distinct finite points make the slope undefined."""
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    return float(np.polyfit(s, y, 1)[0]) if (len(s) >= 2 and len(np.unique(s)) >= 2) else 0.0


def _combined_change(s, y, st):
    """Proportional variance explained by a LEVEL + SLOPE change at ``st`` beyond a plain
    time trend (interrupted-time-series extra sum of squares, in [0,1]; non-directional - 
    captures any immediate and/or gradual change)."""
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok].astype(float), y[ok]
    if len(s) < 4 or len(np.unique(s)) < 2:
        return 0.0
    post = (s >= st).astype(float)
    one = np.ones_like(s)

    def rss(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return float(r @ r)
    rr = rss(np.column_stack([one, s]))                    # time only
    rf = rss(np.column_stack([one, s, post, post * s]))    # + level + slope change
    return (rr - rf) / rr if rr > 0 else 0.0


def _tier_value(s, y, st, sign, statistic):
    """ONE tier's ORIENTED contribution at start ``st`` - the value averaged across tiers in
    ``_stat``. ``level`` = mean(B)-mean(A) ; ``slope`` = slope(B)-slope(A) ; ``combined`` =
    ITS extra-SS of a level+slope change (non-directional) ; ``itei`` = mean(first 3 B)-mean
    (last 3 A) (Michiels 2018 transition index) ; ``tau_u`` = per-tier Tau-U (auto
    baseline-trend corrected) ; ``nap`` = per-tier Non-overlap of All Pairs. ``level`` /
    ``slope`` / ``itei`` are oriented here by ``sign`` ; ``combined`` / ``tau_u`` / ``nap``
    are already non-directional / oriented, so ``sign`` is not re-applied. Returns ``np.nan``
    when the tier cannot contribute (too few points) - the caller then drops it."""
    ma, mb = s < st, s >= st
    sa, ya = s[ma], y[ma]
    sb, yb = s[mb], y[mb]
    ya2, yb2 = ya[np.isfinite(ya)], yb[np.isfinite(yb)]
    if statistic == "level":
        return sign * (yb2.mean() - ya2.mean()) if len(ya2) and len(yb2) else np.nan
    if statistic == "slope":
        if len(ya2) >= 2 and len(yb2) >= 2:
            return sign * (_phase_slope(sb, yb) - _phase_slope(sa, ya))
        return np.nan
    if statistic == "combined":
        return _combined_change(s, y, st)                  # non-directional
    if statistic == "itei":                                # Michiels 2018: transition (last 3 A vs first 3 B)
        if len(ya2) and len(yb2):
            k = 3
            return sign * (yb2[:k].mean() - ya2[-k:].mean())
        return np.nan
    if statistic == "tau_u":                               # per-tier non-overlap (auto trend-corrected, already oriented)
        if len(ya2) and len(yb2):
            imp = "increase" if sign >= 0 else "decrease"
            tv = tau_u(ya2, yb2, improvement=imp).get("tau_u")
            return tv if np.isfinite(tv) else np.nan
        return np.nan
    if statistic == "nap":                                 # per-tier Non-overlap of All Pairs (already oriented)
        if len(ya2) and len(yb2):
            imp = "increase" if sign >= 0 else "decrease"
            nv = nap(ya2, yb2, improvement=imp)
            return nv if np.isfinite(nv) else np.nan
        return np.nan
    raise ValueError("statistic must be 'level' / 'slope' / 'combined' / 'itei' / 'tau_u' / 'nap'")


def _stat(series, starts, sign, statistic="level"):
    """Average across tiers of the per-tier oriented change (see ``_tier_value``). Tiers that
    cannot contribute at their start (NaN value) are dropped from the average."""
    statistic = {"contrast": "level", "md": "level"}.get(statistic, statistic)   # alias (= phase_design "contrast")
    vals = [_tier_value(s, y, starts[t], sign, statistic) for t, (s, y) in series.items()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _exact_additive_p(value_lists, obs_sum, *, tol=1e-9, ndigits=9, max_states=20_000_000):
    """Exact upper-tail randomization p for an **additive** statistic - a SUM of independent
    per-tier values, one drawn from each tier's candidate set - by successive **convolution**
    of the per-tier value distributions. This is the exact p over the full prod k_i product space
    WITHOUT enumerating it (Monte-Carlo-free). Applicable when the statistic is a mean of
    per-tier values, the scheme is independent (per-tier draws), and no cross-tier constraint
    couples the tiers. ``value_lists`` = one list of candidate values per tier. Returns
    ``(p, total)`` (total = prod len(list)), or ``(None, None)`` if the running support exceeds
    ``max_states`` (memory guard -> caller falls back to Monte-Carlo)."""
    dist = {0.0: 1}                                        # sum -> count of assignments
    for vals in value_lists:
        nd = {}
        for cur, cnt in dist.items():
            for v in vals:
                k = round(cur + v, ndigits)
                nd[k] = nd.get(k, 0) + cnt
        dist = nd
        if len(dist) > max_states:
            return None, None
    total = 1
    for vals in value_lists:
        total *= len(vals)
    ge = sum(cnt for sm, cnt in dist.items() if sm >= obs_sum - tol)
    return ge / total, total


def _gap_ok(assignment, min_gap):
    """All consecutive (sorted) intervention starts at least ``min_gap`` apart - the
    Koehler-Levin regulated-stagger constraint."""
    if not min_gap:
        return True
    v = sorted(assignment.values())
    return all(v[i + 1] - v[i] >= min_gap for i in range(len(v) - 1))


def _assignments(tiers, cand, scheme, pool, tier_order, rng, enumerate_cap, n_perm, min_gap):
    """Return ``(assignments, exact, n_eval)``. Schemes: independent (per-tier windows,
    with repetition; Marascuilo-Busk), distinct (shared pool without replacement;
    Wampold-Worsham/Revusky), ordered (shared pool, increasing along the staggering;
    Koehler-Levin). ``min_gap`` filters to a minimum stagger between consecutive starts.
    When exact, ``assignments`` is the materialised list of valid assignments."""
    k = len(tiers)
    if scheme == "independent":
        total = int(np.prod([len(cand[t]) for t in tiers]))
        raw = (dict(zip(tiers, c)) for c in itertools.product(*[cand[t] for t in tiers]))
        sampler = lambda: dict(zip(tiers, [int(rng.choice(cand[t])) for t in tiers]))
    elif scheme == "distinct":
        total = math.perm(len(pool), k)
        raw = (dict(zip(tiers, p)) for p in itertools.permutations(pool, k))
        sampler = lambda: dict(zip(tiers, list(rng.permutation(pool))[:k]))
    elif scheme == "ordered":
        total = math.comb(len(pool), k)
        spool = sorted(pool)
        raw = (dict(zip(tier_order, c)) for c in itertools.combinations(spool, k))
        sampler = lambda: dict(zip(tier_order, sorted(rng.choice(pool, k, replace=False))))
    else:
        raise ValueError("scheme must be 'independent' / 'distinct' / 'ordered'")

    if total <= enumerate_cap:
        valid = [a for a in raw if _gap_ok(a, min_gap)]
        return valid, True, len(valid), len(valid)         # exact: total = number of valid assignments

    def _mc():
        for _ in range(n_perm):
            a = sampler()
            for _try in range(50):                          # rejection-sample the gap constraint
                if _gap_ok(a, min_gap):
                    break
                a = sampler()
            yield a
    return _mc(), False, n_perm, int(total)                # Monte-Carlo: total = ACTUAL size of the space


def multiple_baseline_test(df, *, tier_col, session_col, outcome_col, phase_col=None,
                           baseline_phase=None, treatment_phase=None,
                           intervention_starts=None, candidates=None, entry_points=None,
                           scheme="independent", statistic="level", improvement="increase",
                           min_baseline=3, min_treatment=3, min_gap=None, n_perm=5000,
                           enumerate_cap=200000, conv_cap=20_000_000, random_state=0):
    """
    Randomization test for a multiple-baseline design.

    The intervention start per tier is taken from ``intervention_starts`` (dict
    ``{tier: start_session}``) or derived from ``phase_col``. The null re-draws the
    starts and recomputes the statistic. Inference is **exact** whenever possible:
    direct enumeration when the space <= ``enumerate_cap``; for a larger space under the
    independent (MB) scheme with no ``min_gap`` and an additive statistic, the exact p is
    still obtained by **convolution** of the per-tier candidate-value distributions (up to
    ``conv_cap`` combinations, memory-guarded) - no Monte-Carlo error. Only when neither
    exact route applies does it fall back to Monte-Carlo (``n_perm`` draws, p = (1+#)/(1+B)).

    Verified against R ``scan::rand_test`` (exact match where the admissible space is small
    enough to enumerate; within Monte-Carlo error otherwise) and against direct full
    enumeration for the convolution path (|deltap| < 1e-12).

    ``scheme`` - the randomisation set, with the canonical names of the Levin et al.
    (2016) taxonomy (distribution size in parentheses):
      - ``"MB"`` / ``"independent"`` - **Marascuilo & Busk 1988**: start-point WITH
        replacement; each tier draws from its window ``candidates`` (auto from
        ``min_baseline``/``min_treatment`` if omitted). Size prod k_i (= k^N if equal).
      - ``"WW"`` - **Wampold-Worsham 1986**: case permutation; a pool ``entry_points`` of
        EXACTLY N points assigned to the N tiers. Size N!.
      - ``"MB-R"`` / ``"distinct"`` - Restricted MB: start-point WITHOUT replacement; a
        pool of k > N points assigned bijectively. Size P(k,N) = k!/(k-N)!.
      - ``"ordered"`` - a *non-canonical* regulated-stagger constraint: increasing
        subsets, C(k,N). (Not Koehler-Levin - see note.) ``min_gap`` enforces a minimum
        stagger between consecutive starts.

    NOT yet implemented (would need the casexstart-point product): **KL** (Koehler-Levin
    1998, N!*prodk_i) and the stepwise **Revusky** variants (Rev / Rev-M). The ``"WW"`` /
    ``"MB"`` / ``"MB-R"`` schemes assume **equal series** (all cases observed through the
    final start interval); ``MB-R`` tolerates early-ending cases.

    ``statistic`` - ``"level"`` = average per-tier mean(B)-mean(A) (immediate change);
    ``"slope"`` = average per-tier slope(B)-slope(A) (progressive/trend change). Both
    oriented by ``improvement``.

    **More than two phases** (e.g. A=no-TMS, B=sham, C=real): give ``baseline_phase`` and
    ``treatment_phase`` to test exactly that transition. The series is restricted to those
    two phases (others, like a run-in A, are dropped), the start = first
    ``treatment_phase`` session, and the test compares ``treatment_phase`` vs
    ``baseline_phase`` with the transition randomised within their combined window - e.g.
    ``baseline_phase="B", treatment_phase="C"`` isolates real TMS over and above sham.

    Returns ``{observed, p_value, n_eval, exact, scheme, statistic, starts, per_tier}``.

    References: Edgington & Onghena 2007; Marascuilo & Busk 1988; Wampold & Worsham 1986;
    Levin, Ferron & Gafurov 2016; Phipson & Smyth 2010 (start-point randomization test, MBD).
    R equivalent: SCRT (single-case randomization tests) / scan::rand_test - potential
    equivalent, to test (no single R function reproduces the MB/WW/MB-R schemes + convolution path).
    """
    sign = 1.0 if improvement == "increase" else -1.0
    keep = ({baseline_phase, treatment_phase}
            if (baseline_phase is not None and treatment_phase is not None) else None)
    series = _tier_series(df, tier_col, session_col, outcome_col, phase_col, keep)
    tiers = list(series)

    if intervention_starts is not None:
        starts = {t: int(intervention_starts[t]) for t in tiers}
    elif phase_col is not None:
        starts = {}
        for t, g in df.groupby(tier_col):
            g = g.sort_values(session_col)
            if treatment_phase is not None:
                tr = g[g[phase_col] == treatment_phase]
            else:
                tr = g[g[phase_col] != g[phase_col].iloc[0]]      # first non-baseline phase
            starts[t] = int(pd.to_numeric(tr[session_col]).min()) if len(tr) \
                else int(pd.to_numeric(g[session_col]).max()) + 1
    else:
        raise ValueError("Provide phase_col or intervention_starts.")

    cand = ({t: [int(x) for x in candidates[t]] for t in tiers} if candidates is not None
            else _auto_candidates(series, min_baseline, min_treatment))
    for t in tiers:                                          # observed start must be a candidate
        if starts[t] not in cand[t]:
            cand[t] = sorted(set(cand[t]) | {starts[t]})

    # resolve canonical scheme names (Levin et al. 2016 taxonomy)
    requested = str(scheme).lower()
    canon = {"mb": "independent", "marascuilo-busk": "independent", "independent": "independent",
             "ww": "distinct", "wampold-worsham": "distinct",
             "mb-r": "distinct", "mbr": "distinct", "distinct": "distinct",
             "ordered": "ordered"}.get(requested)
    if canon is None:
        raise ValueError("unknown scheme - use MB / WW / MB-R (or independent/distinct) "
                         "or ordered.")
    scheme = canon

    # ``entry_points`` = FLAT pool of candidate starts (MB-R/WW: draw without replacement from a
    # shared pool). Tolerates a PER-TIER dict {tier:[starts]} (what START_WINDOW/
    # BASELINE_WINDOW produce) by flattening it into a sorted union - otherwise ``for x in dict``
    # would iterate the KEYS (tier names) and ``int('P1')`` would crash.
    if isinstance(entry_points, dict):
        entry_points = sorted({int(v) for vals in entry_points.values() for v in vals})
    pool = ([int(x) for x in entry_points] if entry_points is not None
            else sorted(set(starts.values())))
    n = len(tiers)
    if scheme in ("distinct", "ordered"):
        if len(pool) < n:
            raise ValueError(f"scheme='{requested}' requires >= {n} entry points "
                             f"(entry_points), received {len(pool)}.")
        if requested in ("ww", "wampold-worsham") and len(pool) != n:
            raise ValueError(f"WW requires exactly N={n} entry points (otherwise MB-R).")
        missing = [t for t in tiers if starts[t] not in pool]
        if missing:
            raise ValueError(f"The observed starts of tiers {missing} are not in "
                             "entry_points.")
        if scheme == "distinct" and len(set(starts.values())) != n:
            raise ValueError("Scheme without replacement: the observed starts must be distinct.")
    tier_order = sorted(tiers, key=lambda t: starts[t])      # staggering order (for 'ordered')
    # canonical procedure label + distribution-size formula (Levin et al. 2016)
    if scheme == "independent":
        procedure = "MB - Marascuilo & Busk 1988 (start-point, with replacement, prod k_i)"
    elif scheme == "distinct":
        procedure = ("WW - Wampold-Worsham 1986 (case permutation, N!)" if len(pool) == n
                     else "MB-R - Restricted MB (start-point, without replacement, P(k,N))")
    else:
        procedure = "regulated stagger (increasing subsets, C(k,N)) - non-canonical constraint"

    obs = _stat(series, starts, sign, statistic)
    rng = np.random.RandomState(random_state)
    statc = {"contrast": "level", "md": "level"}.get(statistic, statistic)

    # --- exact via convolution for an ADDITIVE statistic under independent start-point
    # randomisation (MB scheme, no cross-tier gap constraint). The statistic is a mean of
    # independent per-tier values, so the exact null distribution of their sum is the
    # convolution of the per-tier candidate-value distributions - the exact p over the full
    # prod k_i space without enumerating it (replaces Monte-Carlo when the space is too large to
    # enumerate combo-by-combo). Requires every candidate to yield a finite per-tier value
    # (else the mean's denominator would vary across assignments -> not additive). ---
    conv_p = conv_total = None
    total_indep = int(np.prod([len(cand[t]) for t in tiers]))
    if (scheme == "independent" and not min_gap
            and total_indep > enumerate_cap and total_indep <= conv_cap):
        value_lists, additive = [], True
        for t in tiers:
            s, y = series[t]
            vt = [_tier_value(s, y, c, sign, statc) for c in cand[t]]
            if not all(np.isfinite(v) for v in vt):
                additive = False
                break
            value_lists.append(vt)
        if additive:
            obs_sum = float(np.sum([_tier_value(*series[t], starts[t], sign, statc) for t in tiers]))
            conv_p, conv_total = _exact_additive_p(value_lists, obs_sum)

    if conv_p is not None:
        p, exact, n_eval, n_total = conv_p, True, conv_total, conv_total
    else:
        gen, exact, n_eval, n_total = _assignments(tiers, cand, scheme, pool, tier_order,
                                                   rng, enumerate_cap, n_perm, min_gap)
        ge = 0
        for st in gen:
            v = _stat(series, st, sign, statistic)
            if np.isfinite(v) and v >= obs - 1e-12:
                ge += 1
        p = ge / n_eval if exact else (1 + ge) / (1 + n_eval)

    from functions.sced.core import hedges_g_within
    rows = []
    for t in tiers:
        s, y = series[t]
        st = starts[t]
        a, b = y[s < st], y[s >= st]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        tu = tau_u(a, b, improvement=improvement) if len(a) and len(b) else {"tau_u": np.nan}
        hg = hedges_g_within(a, b, improvement=improvement) if len(a) and len(b) else {"g": np.nan}
        rows.append({"Tier": str(t), "Start": st, "n_A": len(a), "n_B": len(b),
                     "mean_A": round(float(a.mean()), 3) if len(a) else np.nan,
                     "mean_B": round(float(b.mean()), 3) if len(b) else np.nan,
                     "diff": round(float(b.mean() - a.mean()), 3) if len(a) and len(b) else np.nan,
                     "Tau-U": tu["tau_u"], "Hedges g (within)": hg["g"],
                     "NAP": round(nap(a, b, improvement=improvement), 4) if len(a) and len(b) else np.nan})
    # --- randomization space + minimum ATTAINABLE p (design resolution) ---
    n_cand = {str(t): int(len(cand[t])) for t in tiers}            # candidate starts per tier
    # n_total = ACTUAL size of the space (product of starts; even if sampled by MC)
    min_p_group = (1.0 / n_total) if n_total else np.nan          # smallest possible test p (resolution)
    min_p_indiv = {t: (1.0 / c if c else np.nan) for t, c in n_cand.items()}   # 1/(tier starts)
    return {"observed": round(float(obs), 4), "p_value": round(float(p), 4),
            "n_eval": int(n_eval), "exact": bool(exact), "scheme": scheme,
            "procedure": procedure, "statistic": statistic, "starts": starts,
            "n_candidates": n_cand, "n_combinations": int(n_total),
            "min_p_group": min_p_group, "min_p_indiv": min_p_indiv,
            "per_tier": pd.DataFrame(rows)}


def _per_tier_table(df, tier_col, session_col, outcome_col, starts, improvement,
                    phase_col=None, keep=None):
    """Per-tier descriptive table (start, n, means, diff, Tau-U, NAP, within-case Hedges' g)."""
    from functions.sced.core import hedges_g_within
    series = _tier_series(df, tier_col, session_col, outcome_col, phase_col, keep)
    rows = []
    for t, (s, y) in series.items():
        st = starts[t]
        a, b = y[s < st], y[s >= st]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        tu = tau_u(a, b, improvement=improvement) if len(a) and len(b) else {"tau_u": np.nan}
        hg = hedges_g_within(a, b, improvement=improvement) if len(a) and len(b) else {"g": np.nan}
        rows.append({"Tier": str(t), "Start": st, "n_A": len(a), "n_B": len(b),
                     "mean_A": round(float(a.mean()), 3) if len(a) else np.nan,
                     "mean_B": round(float(b.mean()), 3) if len(b) else np.nan,
                     "diff": round(float(b.mean() - a.mean()), 3) if len(a) and len(b) else np.nan,
                     "Tau-U": tu["tau_u"], "Hedges g (within)": hg["g"],
                     "NAP": round(nap(a, b, improvement=improvement), 4) if len(a) and len(b) else np.nan})
    return pd.DataFrame(rows)


def pipeline_sced_multiple_baseline(df, *, tier_col, session_col, outcome_col, phase_col=None,
                                    baseline_phase=None, treatment_phase=None,
                                    intervention_starts=None, windows=None,
                                    case_randomization=False, start_point_randomization=True,
                                    replacement=False, comparison="within", statistic="level",
                                    multilevel=False, treatment_trajectory="continuous",
                                    ramp=3, model_selection=False, outcome_type="continuous",
                                    improvement="increase", min_baseline=3, min_treatment=3,
                                    min_gap=None, n_perm=5000, random_state=0,
                                    plot_path=None, pooled_plot_path=None, pooled_title=None,
                                    style=None, impute_col=None,
                                    bayesian=False, bayes_hypothesis="cumulative_trend",
                                    bayes_family="gaussian", bayes_bounds=None, bayes_draws=1000,
                                    bayes_tune=1000, bayes_chains=2, verbose=True):
    """
    End-to-end multiple-baseline pipeline. The randomization procedure is selected by its
    **features** (Levin et al. 2017), not a name:
    ``case_randomization`` (permute cases over positions), ``start_point_randomization``
    (randomise the start within a window), ``replacement`` (start-points with/without
    replacement), ``comparison``  in  {within, between (stepwise Revusky)}. The combination
    maps to WW / MB / MB-R / KL / Rev / Rev-M (see ``mbd_randomization_test``). Supply
    ``windows`` when ``start_point_randomization`` is on (per-case dict for MB, per-position
    lists for KL/Rev-M, flat pool for MB-R). Produces per-tier effect sizes and an optional
    staggered figure. Returns ``(model_info, result)``. The consolidated, styled report is
    assembled one level up by ``report_sced_multiple_baseline`` (``functions.sced.mbd.report``).

    Pooled event-time figures (when ``plot_path`` is set): ``multilevel=True`` adds the
    **inferential** pooled plot (model b1/b2/b3 over the realigned cases); ``bayesian=True``
    additionally fits ``bayes_hier_sced`` (hypothesis ``bayes_hypothesis``, family
    ``bayes_family``, ``bayes_bounds``/draws/tune/chains) and adds the **Bayesian** pooled
    plot with a 95% HDI band - saved as ``*_mbd_pooled_inferentiel.png`` /
    ``*_mbd_pooled_bayesien.png``. ``bayesian`` requires ``phase_col`` and is skipped (with a
    note) if the fit fails.

    References: Levin, Ferron & Gafurov 2016 (feature-specified MBD randomization procedures);
    Moeyaert, Ferron, Beretvas & Van den Noortgate 2014 (multilevel pooled fit).
    R equivalent: SCRT / scan::rand_test (randomization); scan::hplm (multilevel) - potential
    equivalent, to test.
    """
    import os
    from functions.sced.core import missing_data_warnings
    from functions.sced.mbd.procedures import mbd_randomization_test, _derive_starts

    needed = [tier_col, session_col, outcome_col]
    miss_cols = [c for c in needed if c not in df.columns]
    if miss_cols:
        raise ValueError(f"Missing columns: {miss_cols}. LONG format expected - one row "
                         "per (tier, session): [tier, session, outcome].")
    if pd.to_numeric(df[outcome_col], errors="coerce").notna().sum() == 0:
        raise ValueError(f"The outcome '{outcome_col}' is not numeric.")
    dup = int(df.duplicated(subset=[tier_col, session_col]).sum())
    if dup:
        raise ValueError(f"{dup} row(s) with a duplicate (tier, session) - a single measurement "
                         "per occasion is expected (aggregate first otherwise).")
    warns = missing_data_warnings(df, outcome_col=outcome_col, session_col=session_col,
                                  unit_col=tier_col)
    if warns and verbose:
        for w in warns:
            print("Warning:", w)

    starts = _derive_starts(df, tier_col, session_col, phase_col, intervention_starts, treatment_phase)
    res = mbd_randomization_test(
        df, case_col=tier_col, session_col=session_col, outcome_col=outcome_col,
        phase_col=phase_col, intervention_starts=intervention_starts, treatment_phase=treatment_phase,
        case_randomization=case_randomization, start_point_randomization=start_point_randomization,
        replacement=replacement, comparison=comparison, windows=windows, statistic=statistic,
        improvement=improvement, n_perm=n_perm, random_state=random_state)
    keep = ({baseline_phase, treatment_phase}
            if (baseline_phase is not None and treatment_phase is not None) else None)
    per_tier = _per_tier_table(df, tier_col, session_col, outcome_col, starts, improvement,
                               phase_col=phase_col, keep=keep)

    p_value = res["p_value"]
    sig = p_value < 0.05
    contrast = (f"{treatment_phase} vs {baseline_phase}"
                if (baseline_phase and treatment_phase) else "B vs A")
    model_info = {
        "Design": "Multiple baseline (staggered intervention)",
        "Data warnings": warns if warns else "none",
        "Outcome": outcome_col, "Tiers": list(per_tier["Tier"]),
        "Improvement direction": improvement,
        "Primary inference": "Randomization test on the intervention start (Levin et al. 2017)",
        "Contrast": contrast, "Procedure": res.get("procedure", "?"),
        "Randomisation features": (f"case={case_randomization}, start-point={start_point_randomization}, "
                                   f"replacement={replacement}, comparison={comparison}"),
        "Statistic": statistic,
        "Randomisation": f"{res.get('n_eval', '?')} assignments"
                         + (" (exact)" if res.get("exact") else ""),
        # randomization space + minimum ATTAINABLE p (design resolution)
        "Rando - combinations (group)": res.get("n_combinations"),
        "Rando - min attainable p (group)": res.get("min_p_group"),
        "Rando - candidate starts / tier": res.get("n_candidates"),
        "Rando - min attainable p (indiv.)": res.get("min_p_indiv"),
        "Intervention starts": starts,
        "Observed statistic": res.get("observed", res.get("observed_ranksum")),
        "p-value": p_value,
        "Verdict": ("Effect demonstrated: staggered change concordant with the intervention."
                    if sig else "No concordant staggered effect detected."),
    }

    multilevel_df = None
    selection_df = None
    msum = None
    if multilevel:
        from functions.sced.mbd.multilevel import multilevel_mbd_model
        msum, multilevel_df = multilevel_mbd_model(
            df, tier_col=tier_col, session_col=session_col, outcome_col=outcome_col,
            phase_col=phase_col, intervention_starts=intervention_starts,
            treatment_phase=treatment_phase, outcome_type=outcome_type, improvement=improvement,
            treatment_trajectory=treatment_trajectory, ramp=ramp,
            min_baseline=min_baseline, min_treatment=min_treatment, n_perm=n_perm,
            random_state=random_state)
        model_info["Multilevel - trajectory"] = treatment_trajectory
        model_info["Multilevel - baseline trend b1"] = msum.get("baseline_trend_b1")
        model_info["Multilevel - level change b2"] = msum.get("level_change_b2")
        model_info["Multilevel - slope change b3"] = msum.get("slope_change_b3")
        model_info["Multilevel - ICC"] = msum.get("icc")
        model_info["Multilevel - p level / slope"] = (msum["p_level_change_perm"],
                                                      msum["p_slope_change_perm"])
        # CANONICAL inference (Moeyaert): Kenward-Roger p of the fixed effects (via R lmerTest)
        if msum.get("canonical_ddf"):
            model_info["Multilevel canonical - df"] = msum.get("canonical_ddf")
            model_info["Multilevel canonical - p (b1/b2/b3)"] = (
                msum.get("p_b1_kr"), msum.get("p_b2_kr"), msum.get("p_b3_kr"))
    if model_selection:
        from functions.sced.mbd.multilevel import mbd_model_selection
        selection_df, sel_sum = mbd_model_selection(
            df, tier_col=tier_col, session_col=session_col, outcome_col=outcome_col,
            phase_col=phase_col, intervention_starts=intervention_starts,
            treatment_phase=treatment_phase, ramp=ramp)
        model_info["Model selection - AIC preferred"] = sel_sum.get("aic_preferred")
        model_info["Model selection - BIC preferred"] = sel_sum.get("bic_preferred")
        model_info["Model selection - effect sign stable"] = sel_sum.get("effect_sign_stable")

    if plot_path is not None:
        from functions.sced.plots.panels import plot_multiple_baseline
        ppool = pooled_plot_path or plot_path          # pooled (model-dependent) -> dedicated sub-folder if provided
        # if imputed points: TWO versions (with/without markers), indicated in the title
        _has_imp = bool(impute_col and impute_col in df.columns and df[impute_col].any())
        _mb_var = ([("_marker", " (imputed data marked)", impute_col),
                    ("_nomarker", " (imputed data not marked)", None)]
                   if _has_imp else [("", "", None)])
        for _suff, _ttag, _impv in _mb_var:
            plot_multiple_baseline(df, tier_col=tier_col, session_col=session_col,
                                   outcome_col=outcome_col, starts=starts, improvement=improvement,
                                   title=f"Multiple baseline (staggered intervention){_ttag}",
                                   name=f"{outcome_col}{_suff}",
                                   save_path=plot_path, style=style, impute_col=_impv)
        model_info["Figure"] = os.path.join(plot_path, f"{outcome_col}_multiple_baseline_plot.png")
        if msum is not None:                          # INFERENTIAL event-time pooled view (b1/b2/b3)
            from functions.sced.plots.panels import plot_mbd_pooled_fit, ml_band_draws
            import matplotlib.pyplot as _plt
            plot_mbd_pooled_fit(df, tier_col=tier_col, session_col=session_col,
                                outcome_col=outcome_col, starts=starts, fit=msum, per_case=True,
                                band_draws=ml_band_draws(msum),   # 95% confidence band (CI)
                                title=pooled_title or f"MBD pooled - {outcome_col} (inferential ML, 95% CI)",
                                save_path=ppool, fname=f"{outcome_col}_mbd_pooled_inferentiel", style=style)
            _plt.close("all")
            model_info["Figure (pooled inferential)"] = os.path.join(
                ppool, f"{outcome_col}_mbd_pooled_inferentiel.png")
        if bayesian and phase_col is not None:        # BAYESIAN variant: posterior curve + HDI band
            try:
                from functions.sced.mbd.bayesian import bayes_hier_sced
                from functions.sced.plots.panels import plot_mbd_pooled_fit, mbd_bayes_band_draws
                import matplotlib.pyplot as _plt
                _res = bayes_hier_sced(df, tier_col=tier_col, session_col=session_col,
                                       phase_col=phase_col, outcome_col=outcome_col,
                                       baseline=baseline_phase, treatment=treatment_phase,
                                       improvement=improvement, hypothesis=bayes_hypothesis,
                                       family=bayes_family, bounds=bayes_bounds, ar=False,
                                       draws=bayes_draws, tune=bayes_tune, chains=bayes_chains,
                                       rope=None, progressbar=False)
                _fitb, _band = mbd_bayes_band_draws(_res["_idata"], improvement=improvement)
                plot_mbd_pooled_fit(df, tier_col=tier_col, session_col=session_col,
                                    outcome_col=outcome_col, starts=starts, fit=_fitb,
                                    band_draws=_band, per_case=True,
                                    title=pooled_title or f"MBD pooled - {outcome_col} (Bayesian, 95% HDI)",
                                    save_path=ppool, fname=f"{outcome_col}_mbd_pooled_bayesien", style=style)
                _plt.close("all")
                model_info["Figure (pooled Bayesian)"] = os.path.join(
                    ppool, f"{outcome_col}_mbd_pooled_bayesien.png")
                model_info["Bayesian - R-hat max"] = _res.get("rhat_max")
            except Exception as _exc:
                if verbose:
                    print(f"[pooled Bayesian skipped] {type(_exc).__name__}: {_exc}")

    return model_info, res
