"""
SCED - Power analysis for alternating-treatments / N-of-1 designs (single & group)
==================================================================================

Randomization tests have **no closed-form power**, so power is obtained by
**Monte-Carlo simulation** that reuses the project's *actual* test engines
(``condition_permutation_test`` for n=1, ``stratified_condition_permutation_test``
for a group): datasets are simulated under a specified effect and design, the real
test is run on each, and power = proportion with p < alpha.

Three honest read-outs (see the note on "post-hoc power"):

- ``power_sced_alternating``  : **a-priori** power for a planned effect + design.
- ``mdes_sced_alternating``   : **minimum detectable effect** (raw + Cohen's d units)
                                reaching a target power - the rigorous sensitivity
                                analysis, more useful than a single power number.
- ``retrospective_power``     : power for a **user-specified** effect computed from the
                                *realised* design and noise of an observed dataset.

NB - what we deliberately do NOT provide: "observed power" (plugging the *observed*
effect back in). It is a monotone function of the p-value and is uninformative
(Hoenig & Heisey 2001, The American Statistician 55(1), 19-24); ``retrospective_power``
asks instead "what power did this design have for an effect I care about?", which is
legitimate.
"""
import numpy as np
import pandas as pd
from scipy.stats import nct, t as _tdist

from functions.sced.core import (
    validate_sced,
    alternating_scheme,
    condition_permutation_test,
    stratified_condition_permutation_test,
    _group_nuisance,
    _nuisance_basis,
)


def _as_effect_dict(effect, conditions):
    """Normalise ``effect`` into a per-condition shift dict. A scalar shifts the LAST
    condition (vs the others at 0); a dict is used as-is (missing -> 0)."""
    if isinstance(effect, dict):
        return {c: float(effect.get(c, 0.0)) for c in conditions}
    d = {c: 0.0 for c in conditions}
    d[conditions[-1]] = float(effect)
    return d


def _simulate_p(effect_dict, conditions, n_sessions, sd, learning, n_units,
                detrend, method, max_consecutive, standardize, n_perm, rng):
    """Simulate one dataset under the specified effect/design and return the test p."""
    levels = list(conditions)
    reps = n_sessions // len(levels)
    scheme = alternating_scheme(max_consecutive=max_consecutive)
    V, L, U, S = [], [], [], []
    for u in range(n_units):
        order = scheme(np.repeat(levels, reps), rng)          # randomised schedule
        sess = np.arange(1, len(order) + 1)
        y = (np.array([effect_dict[c] for c in order])
             + learning * sess + rng.normal(0, sd, len(order)))
        V += list(y); L += list(order); U += [u] * len(order); S += list(sess)
    V, L, U, S = np.array(V), np.array(L, dtype=object), np.array(U), np.array(S)
    if n_units == 1:
        return condition_permutation_test(V, L, S, detrend=detrend, method=method,
                                          n_perm=n_perm, max_consecutive=max_consecutive,
                                          random_state=int(rng.randint(1 << 30)))["p_value"]
    return stratified_condition_permutation_test(
        V, L, U, S, detrend=detrend, method=method, n_perm=n_perm,
        max_consecutive=max_consecutive, standardize=standardize,
        random_state=int(rng.randint(1 << 30)))["p_value"]


def power_sced_alternating(*, effect, sd, n_sessions, conditions=("A", "B"),
                           n_units=1, detrend="none", learning=0.0,
                           method="freedman-lane", max_consecutive=None,
                           standardize=True, alpha=0.05, n_sims=400, n_perm=400,
                           random_state=0):
    """
    **A-priori power** by simulation for an alternating / N-of-1 design.

    Simulate ``n_sims`` datasets with the given ``effect`` (raw outcome units; scalar =
    shift of the last condition vs the others, or a ``{condition: shift}`` dict),
    residual ``sd``, ``n_sessions`` per unit (balanced across ``conditions``), optional
    ``learning`` slope and ``n_units`` (1 = single series, >1 = stratified group). Each
    dataset is analysed with the **real** test engine, so the estimate reflects the
    test actually used. Power = proportion of p < ``alpha``.

    What drives power is the standardised effect ``effect/sd`` (~ Cohen's d), the number
    of sessions and (for a group) the number of units. Returns ``{power, mc_se, alpha,
    cohens_d, n_sims, n_perm, ...}``. Increase ``n_sims``/``n_perm`` to tighten the
    Monte-Carlo error (reported as ``mc_se``).

    References: Michiels, Heyvaert and Onghena (2017); Bouwmeester and Jongerling (2020) (simulation-based power for single-case randomization tests).
    R equivalent: no direct R equivalent (Monte-Carlo power reuses the study's own permutation test); SCRT power tools - potential equivalent, to test.
    """
    conditions = tuple(conditions)
    if n_sessions % len(conditions) != 0:
        raise ValueError("n_sessions must be divisible by the number of conditions.")
    eff = _as_effect_dict(effect, conditions)
    rng = np.random.RandomState(random_state)
    hits = sum(_simulate_p(eff, conditions, n_sessions, sd, learning, n_units, detrend,
                           method, max_consecutive, standardize, n_perm, rng) < alpha
               for _ in range(n_sims))
    power = hits / n_sims
    d_spread = (np.std([eff[c] for c in conditions], ddof=0) / sd if sd > 0 else np.nan)
    return {"power": round(power, 3), "mc_se": round(np.sqrt(power * (1 - power) / n_sims), 3),
            "alpha": alpha, "cohens_d_spread": round(float(d_spread), 3),
            "n_units": n_units, "n_sessions": n_sessions, "conditions": list(conditions),
            "n_sims": n_sims, "n_perm": n_perm}


def mdes_sced_alternating(*, sd, n_sessions, conditions=("A", "B"), n_units=1,
                          target_power=0.80, detrend="none", learning=0.0,
                          method="freedman-lane", max_consecutive=None,
                          standardize=True, alpha=0.05, d_grid=(0.2, 0.4, 0.6, 0.8,
                          1.0, 1.2, 1.5), n_sims=300, n_perm=300, random_state=0):
    """
    **Minimum Detectable Effect Size** (sensitivity analysis): the smallest effect the
    design can detect at ``target_power``. Sweeps a grid of standardised effects
    ``d_grid`` (the active condition shifted by ``d.sd``), estimates power at each, and
    linearly interpolates the crossing of ``target_power``. Returns the MDES in Cohen's
    d and raw units, plus the full power curve. This is the rigorous way to read a
    design "after the fact": what could it have caught? (cf. Hoenig & Heisey 2001).

    References: Michiels, Heyvaert and Onghena (2017) (conditional power of randomization tests); Hoenig and Heisey (2001) (sensitivity vs observed-power fallacy).
    R equivalent: no direct R equivalent (simulation-based MDES on the study's own permutation test).
    """
    curve = []
    for d in d_grid:
        p = power_sced_alternating(effect=d * sd, sd=sd, n_sessions=n_sessions,
                                   conditions=conditions, n_units=n_units, detrend=detrend,
                                   learning=learning, method=method,
                                   max_consecutive=max_consecutive, standardize=standardize,
                                   alpha=alpha, n_sims=n_sims, n_perm=n_perm,
                                   random_state=random_state)["power"]
        curve.append((float(d), p))
    mdes_d = None
    for (d0, p0), (d1, p1) in zip(curve, curve[1:]):
        if p0 < target_power <= p1:
            mdes_d = d0 + (target_power - p0) * (d1 - d0) / (p1 - p0)
            break
    if mdes_d is None:
        mdes_d = (curve[0][0] if curve[0][1] >= target_power else np.inf)
    return {"mdes_cohens_d": (round(mdes_d, 3) if np.isfinite(mdes_d) else ">= grid max"),
            "mdes_raw": (round(mdes_d * sd, 3) if np.isfinite(mdes_d) else ">= grid max"),
            "target_power": target_power, "alpha": alpha,
            "power_curve": [(d, round(p, 3)) for d, p in curve],
            "n_units": n_units, "n_sessions": n_sessions}


def _estimate_design(df, *, session_col, condition_col, outcome_col, unit_col,
                     conditions, detrend):
    """From an observed dataset, read the realised design (conditions, n_units, sessions
    per unit) and estimate the residual SD (after removing unit means and, if
    detrending, the per-unit time trend). Shared by retrospective_power / power_report."""
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, unit_col=unit_col)
    if conditions is None:
        conditions = tuple(report["Phases/conditions"])
    conditions = tuple(conditions)
    sub = clean[clean[condition_col].isin(conditions)]
    y = pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy()
    sess = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
    units = sub[unit_col].to_numpy() if unit_col else np.zeros(len(y))
    n_units = len(np.unique(units))
    if detrend != "none":
        Z = (_group_nuisance(units, sess, detrend) if unit_col else _nuisance_basis(sess, detrend))
    else:
        Z = np.column_stack([(units == u).astype(float) for u in np.unique(units)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    rank = np.linalg.matrix_rank(Z)
    sd = float(np.std(resid, ddof=rank) if len(y) > rank else np.std(resid, ddof=0))
    n_sessions = int(round(len(y) / n_units))
    n_sessions -= n_sessions % len(conditions)            # make divisible
    return conditions, n_units, n_sessions, sd


def retrospective_power(df, *, session_col, condition_col, outcome_col, effect,
                        unit_col=None, conditions=None, detrend="none",
                        method="freedman-lane", max_consecutive=None, standardize=True,
                        alpha=0.05, n_sims=400, n_perm=400, random_state=0):
    """
    Power the **realised design** had for a **user-specified** effect of interest
    (e.g. a clinically meaningful difference) - the defensible "retrospective" power.

    It reads the design from ``df`` (number of sessions, conditions, units) and the
    residual ``sd`` (after removing unit means and, if detrending, the time trend),
    then calls :func:`power_sced_alternating` for the *specified* ``effect``. It does
    **not** use the observed effect (that would be the discredited observed-power
    fallacy; Hoenig & Heisey 2001). Returns the power dict plus the estimated ``sd``.

    References: Hoenig and Heisey (2001) (power for a specified effect, not the observed one); Michiels, Heyvaert and Onghena (2017).
    R equivalent: no direct R equivalent (simulation-based retrospective power).
    """
    conditions, n_units, n_sessions, sd = _estimate_design(
        df, session_col=session_col, condition_col=condition_col, outcome_col=outcome_col,
        unit_col=unit_col, conditions=conditions, detrend=detrend)
    out = power_sced_alternating(effect=effect, sd=sd, n_sessions=n_sessions,
                                 conditions=conditions, n_units=n_units, detrend=detrend,
                                 method=method, max_consecutive=max_consecutive,
                                 standardize=standardize, alpha=alpha, n_sims=n_sims,
                                 n_perm=n_perm, random_state=random_state)
    out["estimated_residual_sd"] = round(sd, 3)
    out["effect_specified"] = effect
    out["note"] = ("Power for a SPECIFIED effect (not the observed effect) - "
                   "avoids the observed-power fallacy (Hoenig & Heisey 2001).")
    return out


# ---------------------------------------------------------------------------
# Closed-form power for (AB)^k designs (Hedges, Shadish & Natesan Batley 2022)
# ---------------------------------------------------------------------------
# For the *phase-design* family (AB, ABAB, ... = (AB)^k repeated across m cases)
# there IS a closed-form power, based on the design-comparable standardised mean
# difference delta = (mu^T - mu^C)/sqrt(sigma2+tau2) (Cohen's d metric; Hedges et al.
# 2012). The test statistic is a noncentral t; power follows directly. This complements
# the simulation-based power above (which serves the *randomization* test on
# alternating designs) and is much faster for planning an (AB)^k study.
#
# Validated to the digit against the paper's worked example (k=2, n=3, m=3,
# phi=rho=0.5, delta=0.75 -> a=0.1670, c*=0.4571, h=5.95, lambda=1.982, power=0.38) - see
# tests/test_sced.py::TestHedgesABkPower.


def _abk_constants(k, n, m, phi, rho):
    """Hedges-2022 constants for a balanced (AB)^k design with m cases, n
    observations/phase, autocorrelation ``phi`` and ICC ``rho`` = tau2/(sigma2+tau2).

    Returns ``(a, b_star, c_star, h)`` where (appendix A4/A9/A10):
      - ``a`` = Var(Dbar)/sigma2 (variance of the mean within-case contrast, sigma2 units),
      - ``b_star`` = E[S2]/(sigma2+tau2), ``c_star`` = Var[S2]/(sigma2+tau2)2,
      - ``h`` = 2.b*2/c* the Satterthwaite df of the pooled variance S2.
    """
    N = 2 * k * n
    w = np.tile(np.concatenate([np.ones(n), -np.ones(n)]), k)   # k x (1_n, -1_n)
    idx = np.arange(N)
    Pc = phi ** np.abs(idx[:, None] - idx[None, :])             # AR(1) correlation
    # a in sigma2 units (tau2 cancels: w sums to zero within each phase pair)
    a = (w @ (Pc / (1 - phi ** 2)) @ w) / (m * (k * n) ** 2)
    # timepoint covariance T (total var = 1): tau2 + sigma2.phi^|i-j|/(1-phi2)
    sigma2, tau2 = 1 - rho, rho
    T = tau2 + sigma2 / (1 - phi ** 2) * Pc
    b_star = float(np.trace(T)) / N
    c_star = 2 * float((T ** 2).sum()) * (m - 1) / (N * (m - 1)) ** 2
    h = 2 * b_star ** 2 / c_star
    return a, b_star, c_star, h


def power_abk_design(*, delta, k=1, n=3, m, phi=0.5, rho=0.5, alpha=0.05, tails=2):
    """
    **Closed-form a-priori power** for an (AB)^k single-case design with replication
    (Hedges, Shadish & Natesan Batley 2022, *Behav. Res. Methods*).

    Parameters (all design knobs the planner controls or must assume):
      - ``delta`` : design-comparable effect delta = (mu^T-mu^C)/sqrt(sigma2+tau2) (Cohen's d metric).
      - ``k``     : number of AB *pairs* (k=1 -> AB, k=2 -> ABAB; WWC wants k>=2).
      - ``n``     : observations per phase (balanced; WWC wants n>=3, ideally >=5).
      - ``m``     : number of cases/participants (power needs m>1; WWC MBD wants m>=3).
      - ``phi``   : within-case first-order autocorrelation (conservative default 0.5).
      - ``rho``   : ICC tau2/(sigma2+tau2), between-case share of variance (default 0.5).
      - ``tails`` : 1 or 2 (default two-tailed at ``alpha``).

    Power = P(reject H0: delta=0). The statistic is noncentral t with df ``h`` and
    noncentrality ``lambda = sqrt(b*/a).delta``; power = tail mass beyond the central-t
    critical value. The dominant lever is delta, then m, then k (Hedges 2022, Table 1).
    Returns a dict with ``power`` and the constants ``a, b_star, c_star, h, lambda``.

    NB the df ``nu`` for the *critical value* under H0 is, per the paper, a more complex
    function of phi than ``h``; ``h`` is used for both (Satterthwaite), which reproduces
    the paper's worked example exactly and its headline designs (m=5,n=3->0.65 ;
    m=6,n=5->0.80); expect <=0.04 drift only at large n with very small m.

    References: Hedges, Shadish and Natesan Batley (2022) (closed-form power for (AB)k designs); Hedges, Pustejovsky and Shadish (2012) (design-comparable SMD metric).
    R equivalent: no direct CRAN equivalent (Hedges et al. 2022 provide standalone R code for (AB)k power) - potential equivalent, to test.
    """
    if m < 2:
        raise ValueError("(AB)^k power requires m >= 2 cases (Hedges 2022).")
    if not (0 <= rho < 1) or not (-1 < phi < 1):
        raise ValueError("Requires 0 <= rho < 1 and -1 < phi < 1.")
    a, b_star, c_star, h = _abk_constants(k, n, m, phi, rho)
    lam = np.sqrt(b_star / a) * delta
    if tails == 2:
        crit = _tdist.ppf(1 - alpha / 2, h)
        power = 1 - nct.cdf(crit, h, lam) + nct.cdf(-crit, h, lam)
    else:
        crit = _tdist.ppf(1 - alpha, h)
        power = 1 - nct.cdf(crit, h, lam)
    return {"power": round(float(power), 3), "delta": delta, "k": k, "n": n, "m": m,
            "phi": phi, "rho": rho, "alpha": alpha, "tails": tails,
            "a": round(a, 4), "b_star": round(b_star, 4), "c_star": round(c_star, 4),
            "h_df": round(h, 2), "lambda": round(float(lam), 3)}


def mdes_abk_design(*, k=1, n=3, m, phi=0.5, rho=0.5, alpha=0.05, tails=2,
                    target_power=0.80):
    """
    **Minimum detectable effect** (in delta / Cohen's d units) for an (AB)^k design at
    ``target_power`` - the closed-form sensitivity read-out. Bisects delta (power is
    monotone increasing in delta). Returns ``{mdes_delta, target_power, ...}``.

    References: Hedges, Shadish and Natesan Batley (2022) (closed-form (AB)k power / sensitivity).
    R equivalent: no direct CRAN equivalent (Hedges et al. 2022 R code) - potential equivalent, to test.
    """
    lo, hi = 0.0, 5.0
    p_hi = power_abk_design(delta=hi, k=k, n=n, m=m, phi=phi, rho=rho, alpha=alpha,
                            tails=tails)["power"]
    if p_hi < target_power:
        return {"mdes_delta": "> 5", "target_power": target_power, "k": k, "n": n, "m": m}
    for _ in range(40):
        mid = (lo + hi) / 2
        p = power_abk_design(delta=mid, k=k, n=n, m=m, phi=phi, rho=rho, alpha=alpha,
                             tails=tails)["power"]
        if p < target_power:
            lo = mid
        else:
            hi = mid
    return {"mdes_delta": round((lo + hi) / 2, 3), "target_power": target_power,
            "k": k, "n": n, "m": m, "phi": phi, "rho": rho, "alpha": alpha}


def wwc_design_check(*, k, n_per_phase, m, design="phase"):
    """
    Check a planned design against the **What Works Clearinghouse** SCED pilot
    standards (Kratochwill et al. 2010, as read for power by Hedges 2022).

    For a phase design (AB)^k: **k >= 2** (>=3 reversal opportunities -> ABAB),
    **n >= 3** per phase to *meet standards with reservations*, **n >= 5** to *meet
    standards without reservations*. For a multiple-baseline design: at least
    **m >= 3** cases (and the same n thresholds per phase). Returns ``(summary, sheet)``
    with an ``Explanation`` / ``Interpretation`` column per criterion.

    References: Kratochwill et al. (2010) (What Works Clearinghouse SCED standards); Hedges, Shadish and Natesan Batley (2022) (power reading of the standards).
    R equivalent: no direct R equivalent (design-standards checklist).
    """
    rows, ok = [], True
    def add(crit, val, passed, expl, interp):
        nonlocal ok
        ok = ok and passed
        rows.append({"Criterion": crit, "Value": val, "OK": "yes" if passed else "no",
                     "Explanation": expl, "Interpretation": interp})

    if design == "phase":
        add("k (AB pairs)", k, k >= 2,
            "WWC: at least one reintroduction (ABAB) for 3 demonstration occasions.",
            "k>=2: power drops sharply at k=1 (Hedges 2022)." if k >= 2
            else "k=1 (simple AB): very underpowered, below standard - aim for ABAB.")
    else:
        add("m (number of cases)", m, m >= 3,
            "WWC: >=3 tiers/cases to demonstrate the effect at different moments.",
            "m>=3: meets the replication threshold." if m >= 3
            else "m<3: below the MBD threshold (>=3 cases required).")
    add("n (obs/phase) - with reservations", n_per_phase, n_per_phase >= 3,
        "WWC: >=3 points per phase to *meet standards with reservations*.",
        ">=3: minimal threshold reached." if n_per_phase >= 3 else "<3: below the minimal threshold.")
    add("n (obs/phase) - without reservations", n_per_phase, n_per_phase >= 5,
        "WWC: >=5 points per phase to *meet standards without reservations*.",
        ">=5: most demanding standard reached." if n_per_phase >= 5
        else "3-4: acceptable *with reservations* only.")
    add("m (power)", m, m >= 2,
        "The closed-form (AB)^k power requires at least 2 cases (Hedges 2022).",
        "m>=2: power computation possible." if m >= 2 else "m=1: power undefined.")
    sheet = pd.DataFrame(rows, columns=["Criterion", "Value", "OK", "Explanation", "Interpretation"])
    summary = {"design": design, "k": k, "n_per_phase": n_per_phase, "m": m,
               "meets_with_reservations": bool(n_per_phase >= 3 and (m >= 3 if design != "phase" else k >= 2)),
               "meets_without_reservations": bool(n_per_phase >= 5 and (m >= 3 if design != "phase" else k >= 2)),
               "all_pass": bool(ok)}
    return summary, sheet


def power_report(df, *, session_col, condition_col, outcome_col, unit_col=None,
                 conditions=None, effect_of_interest=None, target_power=0.80,
                 detrend="none", method="freedman-lane", max_consecutive=None,
                 standardize=True, alpha=0.05, d_grid=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5),
                 n_sims=200, n_perm=200, random_state=0):
    """
    Power read-out for the **realised design** of an observed dataset, ready to drop
    into a report. Always returns the **MDES** (minimum detectable effect at
    ``target_power``) and, if ``effect_of_interest`` (raw units) is given, the
    **retrospective power** for that effect - both simulated through the real test with
    the residual SD estimated from the data. Returns ``(summary_dict, sheet_df)`` where
    ``sheet_df`` already carries ``Explanation`` / ``Interpretation`` columns.

    References: Hoenig and Heisey (2001); Michiels, Heyvaert and Onghena (2017) (MDES + retrospective power for randomization tests).
    R equivalent: no direct R equivalent (simulation-based power read-out).
    """
    conditions, n_units, n_sessions, sd = _estimate_design(
        df, session_col=session_col, condition_col=condition_col, outcome_col=outcome_col,
        unit_col=unit_col, conditions=conditions, detrend=detrend)
    common = dict(sd=sd, n_sessions=n_sessions, conditions=conditions, n_units=n_units,
                  detrend=detrend, method=method, max_consecutive=max_consecutive,
                  standardize=standardize, alpha=alpha, n_sims=n_sims, n_perm=n_perm,
                  random_state=random_state)
    md = mdes_sced_alternating(target_power=target_power, d_grid=d_grid, **common)
    rows = [{"Analysis": "MDES (minimum detectable effect)",
             "Value": f"d={md['mdes_cohens_d']} (raw={md['mdes_raw']})",
             "Explanation": "Smallest effect detectable at the target power threshold (sensitivity analysis).",
             "Interpretation": f"The design detects an effect >= d={md['mdes_cohens_d']} at "
                               f"{int(target_power*100)}% (alpha={alpha})."}]
    summary = {"estimated_residual_sd": round(sd, 3), "n_sessions": n_sessions,
               "n_units": n_units, "mdes_cohens_d": md["mdes_cohens_d"],
               "mdes_raw": md["mdes_raw"], "power_curve": md["power_curve"]}
    if effect_of_interest is not None:
        pr = power_sced_alternating(effect=effect_of_interest, **common)["power"]
        d_eff = round(abs(effect_of_interest) / sd, 3) if sd > 0 else float("nan")
        rows.append({"Analysis": f"Retrospective power (effect={effect_of_interest})",
                     "Value": pr,
                     "Explanation": "Power of the realized design for the SPECIFIED effect "
                                    "(not the observed effect - Hoenig & Heisey 2001).",
                     "Interpretation": (f"Underpowered for d~{d_eff} (power={pr})."
                                        if pr < 0.8 else f"Sufficient power (={pr}, d~{d_eff}).")})
        summary["retrospective_power"] = pr
    sheet = pd.DataFrame(rows, columns=["Analysis", "Value", "Explanation", "Interpretation"])
    return summary, sheet
