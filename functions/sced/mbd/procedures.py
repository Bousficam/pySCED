"""
SCED - Canonical MBD randomization procedures: Koehler-Levin & Revusky (between-case)
=====================================================================================

Completes the Levin, Ferron & Gafurov (2017, *Dev. Neurorehabil.* 21(5):290-311)
taxonomy of multiple-baseline randomization tests. The within-case start-point/case
schemes (MB, WW, MB-R) live in ``SCED_multiple_baseline``; this module adds:

- **KL** - Koehler & Levin (1998) "regulated randomization": *case + start-point*. Each
  staggered position has a NON-OVERLAPPING window of k_i candidate start points; the test
  randomises both the case->position assignment (N!) and the start within each position
  window (prod k_i) -> N!*prod k_i outcomes. Statistic = average per-case mean(B)-mean(A).
- **Rev** - Revusky (1967), *between-case, stepwise*: at step j the case intervened at
  position j is compared with the still-baseline cases using a truncated B phase
  (start_j ... start_{j+1}-1); the cases are ranked and the intervened case's rank is
  summed across steps. Distribution = N! (case->position permutations). Smaller rank sum
  = stronger effect.
- **Rev-M** - Modified Revusky (present article): the stepwise B-A means (not ranks)
  averaged across steps, with a window of start points per position -> N!*prod k_i.

Data: long, one row per (case, session): ``case | session | outcome``; ``starts`` /
``windows`` give the observed start per case and the per-position candidate sets.
Validated against the worked classroom example in the paper.
"""
import itertools
import math

import numpy as np
import pandas as pd


def _series(df, case_col, session_col, outcome_col):
    """Time-ordered outcome array per case. Inputs: long frame plus column names. Output: dict
    ``{case: outcome_array}`` sorted by session, coerced to float (non-numeric -> NaN)."""
    out = {}
    for c, g in df.groupby(case_col):
        g = g.sort_values(session_col)
        out[c] = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
    return out


def _ba_full(y, start):
    """mean(B) - mean(A) with the FULL B phase; ``start`` is the 1-based observation index
    of the first B session (A = O1..O[start-1], B = O[start]..end)."""
    a, b = y[:start - 1], y[start - 1:]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (b.mean() - a.mean()) if len(a) and len(b) else np.nan


def _ba_step(y, start, start_next):
    """Stepwise mean(B) - mean(A): B truncated to O[start]..O[start_next-1] (or end)."""
    a = y[:start - 1]
    b = y[start - 1:(start_next - 1) if start_next else None]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (b.mean() - a.mean()) if len(a) and len(b) else np.nan


def _pos_of_observed(case_ids, observed_starts, windows):
    """Map each case to the position whose window contains its observed start."""
    order = [None] * len(windows)
    for c in case_ids:
        for j, W in enumerate(windows):
            if observed_starts[c] in W:
                order[j] = c
                break
    if any(o is None for o in order):
        raise ValueError("Each observed start must belong to exactly one position window.")
    return order


def koehler_levin_test(df, *, case_col, session_col, outcome_col, windows, observed_starts,
                       improvement="increase", n_perm=5000, enumerate_cap=500000, random_state=0):
    """KL regulated randomization (N!*prod k_i). ``windows`` = list of per-position candidate
    start-point lists (non-overlapping, staggering order); ``observed_starts`` = dict
    case->start. Returns ``{observed, p_value, n_eval, exact, procedure}``.

    References: Koehler & Levin 1998; Levin, Ferron & Gafurov 2016 (KL regulated randomization).
    R equivalent: SCRT - potential equivalent, to test (no turnkey R function for the KL scheme)."""
    sign = 1.0 if improvement == "increase" else -1.0
    cases = _series(df, case_col, session_col, outcome_col)
    case_ids = list(cases)
    N = len(case_ids)
    obs_order = _pos_of_observed(case_ids, observed_starts, windows)

    def stat(order, starts):
        return sign * np.mean([_ba_full(cases[order[j]], starts[j]) for j in range(N)])
    obs = stat(obs_order, [observed_starts[obs_order[j]] for j in range(N)])
    total = math.factorial(N) * int(np.prod([len(W) for W in windows]))
    exact = total <= enumerate_cap
    ge = 0
    if exact:
        for order in itertools.permutations(case_ids):
            for starts in itertools.product(*windows):
                if stat(order, starts) >= obs - 1e-12:
                    ge += 1
        p = ge / total
        n_eval = total
    else:
        rng = np.random.RandomState(random_state)
        for _ in range(n_perm):
            order = list(rng.permutation(case_ids))
            starts = [int(rng.choice(W)) for W in windows]
            if stat(order, starts) >= obs - 1e-12:
                ge += 1
        p = (1 + ge) / (1 + n_perm)
        n_eval = n_perm
    return {"observed": round(float(obs), 4), "p_value": round(float(p), 4),
            "n_eval": int(n_eval), "exact": bool(exact),
            "procedure": "KL - Koehler-Levin 1998 (case + start-point, N!*prodk_i)"}


def revusky_test(df, *, case_col, session_col, outcome_col, position_starts, observed_order,
                 improvement="increase"):
    """Revusky (1967) between-case stepwise rank test (N!). ``position_starts`` = the fixed
    staggered start per position (increasing); ``observed_order`` = the cases in their
    actual position order. Statistic = sum of the intervened cases' ranks (smaller =
    stronger). Returns ``{observed_ranksum, p_value, n_eval, procedure}``.

    References: Revusky 1967; Levin, Ferron & Gafurov 2016 (between-case stepwise rank test).
    R equivalent: no direct R equivalent (SCRT covers other MBD schemes, not the Revusky ranks)."""
    sign = 1.0 if improvement == "increase" else -1.0
    cases = _series(df, case_col, session_col, outcome_col)
    case_ids = list(cases)
    N = len(position_starts)

    def ranksum(order):
        total = 0.0
        for j in range(N):
            s_j = position_starts[j]
            s_next = position_starts[j + 1] if j + 1 < N else None
            remaining = order[j:]
            ba = {c: sign * _ba_step(cases[c], s_j, s_next) for c in remaining}
            vals = np.array([ba[c] for c in remaining], float)
            # rank 1 = largest BA; average ranks for ties
            order_idx = (-vals).argsort()
            ranks = np.empty(len(vals)); ranks[order_idx] = np.arange(1, len(vals) + 1)
            # midranks for ties
            for v in np.unique(vals):
                m = vals == v
                if m.sum() > 1:
                    ranks[m] = ranks[m].mean()
            total += ranks[list(remaining).index(order[j])]
        return float(total)

    obs = ranksum(list(observed_order))
    sums = [ranksum(list(p)) for p in itertools.permutations(case_ids)]
    p = float(np.mean([s <= obs + 1e-9 for s in sums]))    # smaller rank sum = more extreme
    return {"observed_ranksum": round(obs, 4), "p_value": round(p, 4),
            "n_eval": math.factorial(N),
            "procedure": "Rev - Revusky 1967 (between-case, stepwise ranks, N!)"}


def revusky_modified_test(df, *, case_col, session_col, outcome_col, windows, observed_starts,
                          improvement="increase", n_perm=5000, enumerate_cap=500000, random_state=0):
    """Modified Revusky (present article): between-case stepwise B-A means (not ranks),
    averaged across steps, randomised over case->position (N!) x start windows (prod k_i).
    ``windows`` per-position candidate lists; ``observed_starts`` dict case->start.

    References: Levin, Ferron & Gafurov 2016 (modified Revusky, case + start-point stepwise).
    R equivalent: no direct R equivalent."""
    sign = 1.0 if improvement == "increase" else -1.0
    cases = _series(df, case_col, session_col, outcome_col)
    case_ids = list(cases)
    N = len(case_ids)
    obs_order = _pos_of_observed(case_ids, observed_starts, windows)

    def stat(order, starts):
        vals = []
        for j in range(N):
            s_next = starts[j + 1] if j + 1 < N else None
            vals.append(_ba_step(cases[order[j]], starts[j], s_next))
        return sign * np.mean(vals)
    obs = stat(obs_order, [observed_starts[obs_order[j]] for j in range(N)])
    total = math.factorial(N) * int(np.prod([len(W) for W in windows]))
    exact = total <= enumerate_cap
    ge = 0
    if exact:
        for order in itertools.permutations(case_ids):
            for starts in itertools.product(*windows):
                if stat(order, starts) >= obs - 1e-12:
                    ge += 1
        p = ge / total
        n_eval = total
    else:
        rng = np.random.RandomState(random_state)
        for _ in range(n_perm):
            order = list(rng.permutation(case_ids))
            starts = [int(rng.choice(W)) for W in windows]
            if stat(order, starts) >= obs - 1e-12:
                ge += 1
        p = (1 + ge) / (1 + n_perm)
        n_eval = n_perm
    return {"observed": round(float(obs), 4), "p_value": round(float(p), 4),
            "n_eval": int(n_eval), "exact": bool(exact),
            "procedure": "Rev-M - Modified Revusky (case + start-point, stepwise, N!*prodk_i)"}


def choose_mbd_procedure(*, n_cases, equal_series, n_start_points=1,
                         effect_type="immediate", require_stagger=True):
    """
    Recommend an MBD randomization procedure (Levin, Ferron & Gafurov 2017).

    Decision drivers (from the paper):
      - ``n_start_points`` (k per case): 1 = fixed start (only case randomisation
        available) -> WW / Rev; >1 = randomised start window -> start-point procedures.
      - ``equal_series``: can every case be observed through the FINAL staggered start
        interval? WW / MB / KL require it; MB-R / Rev / Rev-M tolerate early-ending cases.
      - ``effect_type``: for an *immediate abrupt* effect, KL and MB-R are the most
        powerful. (For a *gradual* effect, keep the procedure but use statistic
        "slope"/"combined".)
      - ``require_stagger``: a true MBD forbids two cases sharing a start - MB (with
        replacement) does not guarantee that, so it is only a fallback.

    Returns ``{recommended, rationale, alternatives, note}``.

    References: Levin, Ferron & Gafurov 2016 (procedure selection for MBD randomization tests).
    R equivalent: no direct R equivalent (decision heuristic, not a statistical estimator).
    """
    k = int(n_start_points)
    alt = []
    if k <= 1:
        rec = "WW" if equal_series else "Rev"
        rationale = ("A single start point per case -> no randomization of the moment; "
                     "only CASE randomization is possible.")
        alt = ["Rev (between-case)"] if equal_series else ["WW (if equal series)"]
    else:
        if equal_series:
            rec = "KL"
            rationale = ("Randomized start windows + equal series -> KL (case + "
                         "start-point): powerful for an immediate effect and guarantees the stagger.")
            alt = ["MB-R (without replacement, equally powerful)", "WW (case only)"]
        else:
            rec = "MB-R" if effect_type == "immediate" else "Rev-M"
            rationale = ("Unequal series (early-ending cases) -> tolerant procedures: "
                         "MB-R (within-case) or Rev-M (between-case, stepwise).")
            alt = ["Rev-M (between-case)"] if rec == "MB-R" else ["MB-R (within-case)"]
    note = []
    if not require_stagger:
        note.append("MB (with replacement) is possible but is NOT a true MBD (starts may coincide).")
    if effect_type == "gradual":
        note.append("Gradual effect -> keep the procedure but statistic='slope'/'combined'.")
    if k > 1 and equal_series and effect_type == "immediate":
        note.append("KL and MB-R are the most powerful for an immediate abrupt effect (Levin et al. 2017).")
    return {"recommended": rec, "rationale": rationale, "alternatives": alt,
            "note": " ".join(note) if note else " - "}


def _derive_starts(df, case_col, session_col, phase_col, intervention_starts, treatment_phase=None):
    """Per-case intervention start session. Uses ``intervention_starts`` when given, otherwise the
    first ``treatment_phase`` session (or the first non-baseline phase) read from ``phase_col``.
    Inputs: long frame plus column names. Output: dict ``{case: start_session}`` (falls back to the
    last session + 1 for a case that never enters treatment)."""
    case_ids = list(pd.unique(df[case_col]))
    if intervention_starts is not None:
        return {c: int(intervention_starts[c]) for c in case_ids}
    if phase_col is None:
        raise ValueError("Provide phase_col or intervention_starts.")
    starts = {}
    for c, g in df.groupby(case_col):
        g = g.sort_values(session_col)
        tr = (g[g[phase_col] == treatment_phase] if treatment_phase
              else g[g[phase_col] != g[phase_col].iloc[0]])
        starts[c] = int(pd.to_numeric(tr[session_col]).min()) if len(tr) \
            else int(pd.to_numeric(g[session_col]).max()) + 1
    return starts


def mbd_randomization_test(df, *, case_col, session_col, outcome_col, phase_col=None,
                           intervention_starts=None, treatment_phase=None,
                           case_randomization, start_point_randomization, replacement=False,
                           comparison="within", windows=None, improvement="increase",
                           statistic="level", n_perm=5000, random_state=0):
    """
    Unified MBD randomization test specified by the **randomization features** rather than
    a procedure name (Levin et al. 2017). The feature combination identifies the canonical
    procedure and dispatches:

    | case_rand | start_rand | replacement | comparison | -> |
    |-----------|-----------|-------------|------------|---|
    |   True    |   False    | - |  within    | WW   |
    |   False   |   True     |    True     |  within    | MB   |
    |   False   |   True     |    False    |  within    | MB-R |
    |   True    |   True     | - |  within    | KL   |
    |   True    |   False    | - |  between   | Rev  |
    |   True    |   True     | - |  between   | Rev-M|

    Inputs by feature: with ``start_point_randomization`` you must give ``windows`` - 
    a **dict** ``{case: [starts]}`` for MB (per-case), a **list of per-position lists** for
    KL / Rev-M, or a **flat list** (shared pool) for MB-R. Without it the fixed/observed
    starts (from ``phase_col`` / ``intervention_starts``) are used (WW, Rev). The observed
    starts and the staggering order are derived from the data. Returns the dispatched
    test's result dict (always carries ``procedure`` and ``p_value``).

    References: Levin, Ferron & Gafurov 2016 (feature-specified MBD randomization dispatcher).
    R equivalent: SCRT / scan::rand_test - potential equivalent, to test.
    """
    from functions.sced.mbd.baseline import multiple_baseline_test
    starts = _derive_starts(df, case_col, session_col, phase_col, intervention_starts, treatment_phase)
    case_ids = list(starts)
    order = sorted(case_ids, key=lambda c: starts[c])               # staggering order

    if comparison == "between":
        if not case_randomization:
            raise ValueError("A between-case comparison (Revusky) requires case_randomization=True.")
        if start_point_randomization:                              # Rev-M
            return revusky_modified_test(df, case_col=case_col, session_col=session_col,
                                         outcome_col=outcome_col, windows=windows,
                                         observed_starts=starts, improvement=improvement,
                                         n_perm=n_perm, random_state=random_state)
        return revusky_test(df, case_col=case_col, session_col=session_col,           # Rev
                            outcome_col=outcome_col, position_starts=sorted(starts.values()),
                            observed_order=order, improvement=improvement)

    # within-case
    if case_randomization and start_point_randomization:           # KL
        return koehler_levin_test(df, case_col=case_col, session_col=session_col,
                                  outcome_col=outcome_col, windows=windows, observed_starts=starts,
                                  improvement=improvement, n_perm=n_perm, random_state=random_state)
    common = dict(tier_col=case_col, session_col=session_col, outcome_col=outcome_col,
                  phase_col=phase_col, intervention_starts=intervention_starts,
                  improvement=improvement, statistic=statistic, n_perm=n_perm,
                  random_state=random_state)
    if case_randomization and not start_point_randomization:       # WW
        return multiple_baseline_test(df, scheme="WW", entry_points=sorted(starts.values()), **common)
    if (not case_randomization) and start_point_randomization:     # MB / MB-R
        if replacement:                                            # MB
            return multiple_baseline_test(df, scheme="MB", candidates=windows, **common)
        return multiple_baseline_test(df, scheme="MB-R", entry_points=windows, **common)   # MB-R
    raise ValueError("No randomization: case_randomization and start_point_randomization "
                     "cannot both be False.")
