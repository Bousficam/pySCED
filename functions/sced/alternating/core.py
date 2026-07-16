"""
SCED - Alternating Treatments Design / N-of-1 trial (single unit)
=================================================================

Family 1 of the SCED pipelines: conditions are RANDOMLY ASSIGNED to sessions/
blocks (rapid-alternating ATD) up to a formal, often blinded, N-of-1 trial. The
maths are the same along that continuum; the N-of-1 simply adds rigour
(randomisation + washout + blinding).

Primary inference (per the chosen convention) is the **randomization test**: the
condition labels are reshuffled according to the assignment schedule that could
have been used (``alternating_scheme``, with an optional max-consecutive
constraint), and a difference-in-means (2 conditions) or between-condition mean
variance (>=3) serves as the statistic. **Tau-U / NAP** are reported alongside as
descriptive nonoverlap effect sizes. A **carryover / order diagnostic** flags a
session-order trend that would threaten the no-carryover assumption.

Outcome scales: continuous, count and binary (the statistics are distribution-
free; the difference of means doubles as a rate/proportion difference).

Same house contract as the other pipelines::

    model_info, rand = pipeline_sced_alternating(
        df, session_col="session", condition_col="cond", outcome_col="y",
        conditions=("A", "B"), outcome_type="continuous", save_path=save_path)
"""
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from functions.sced.core import (
    validate_sced,
    sced_data_warnings,
    missing_data_warnings,
    describe_by_phase,
    nap,
    tau_u,
    pnd,
    pem,
    randomization_test,
    alternating_scheme,
    block_scheme,
    diff_in_means,
    omnibus_variance,
    make_ordered_trend_stat,
    _validate_condition_order,
    diagnose_series,
    make_condition_ss_stat,
    condition_permutation_test,
    choose_permutation_method,
    sced_test_name,
    sced_methods_text,
    nap_magnitude,
    tau_magnitude,
    pnd_magnitude,
    pem_magnitude,
    _nuisance_basis,
)


def _carryover_diagnostic(df, session_col, condition_col, outcome_col):
    """
    Order/carryover check: Spearman correlation between the outcome and the
    session index (within the alternation). A strong session trend suggests
    carryover or a time confound - the ATD/N-of-1 assumes none. Returns a dict.

    References: Spearman 1904 (rank correlation as an order/carryover diagnostic).
    R equivalent: stats::cor.test(method="spearman").
    """
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    s = pd.to_numeric(df[session_col], errors="coerce")
    ok = y.notna() & s.notna()
    if ok.sum() < 4:
        return {"Carryover/order check": " - ",
                "Comment": "Too few sessions to assess an order effect."}
    rho, p = stats.spearmanr(s[ok], y[ok])
    flag = abs(rho) > 0.5 and p < 0.05
    return {"Order trend (Spearman rho)": round(float(rho), 3),
            "Order trend p": round(float(p), 4),
            "Carryover/order warning": (
                "Marked session trend: risk of carryover / temporal confound."
                if flag else "No marked order trend.")}


def _effect_sizes_table(df, condition_col, outcome_col, conditions, improvement):
    """Nonoverlap effect sizes for **every pair of conditions** (Tau-U, NAP, PND, PEM)
    plus their interpretation band. Previously only ``conditions[1:] vs ref`` were
    computed, which omitted non-reference pairs (e.g. C vs B); now all C(k,2) pairs are
    returned so no combination is missing. Bands: NAP (Parker & Vannest 2009), Tau-U
    (Vannest & Ninci 2015 - indicative, cf. Critical Tau-U Fingerhut 2021), PND (Scruggs &
    Mastropieri 1998), PEM (Ma 2006).

    References: Parker et al. 2011 (Tau-U); Parker & Vannest 2009 (NAP) - the
    interpretation bands are cited inline above.
    R equivalent: SingleCaseES (NAP + CI); scan::tau_u / scan::corrected_tau.
    """
    rows = []
    for ref, c in itertools.combinations(conditions, 2):
        base = pd.to_numeric(df[df[condition_col] == ref][outcome_col], errors="coerce").dropna()
        trt = pd.to_numeric(df[df[condition_col] == c][outcome_col], errors="coerce").dropna()
        if len(base) == 0 or len(trt) == 0:
            continue
        tu = tau_u(base, trt, improvement=improvement)
        nap_v = round(nap(base, trt, improvement=improvement), 4)
        pnd_v = round(pnd(base, trt, improvement=improvement), 1)
        pem_v = round(pem(base, trt, improvement=improvement), 1)
        se = tu["se"]
        nap_ci = ((round(max(0.0, nap_v - 1.96 * se / 2), 3), round(min(1.0, nap_v + 1.96 * se / 2), 3))
                  if np.isfinite(se) else (np.nan, np.nan))   # SE(NAP)=SE(tau)/2 since NAP=(tau+1)/2
        rows.append({
            "Comparison": f"{c} vs {ref}",
            "Tau-U": tu["tau_u"],
            # which variant was actually applied (default 'auto' only corrects the baseline
            # trend when one is detected, else returns the raw Tau) - surfaced so a fixed
            # 'baseline-corrected' label does not mislead when no correction ran.
            "Tau-U variant": tu["variant"],
            "Tau-U CI95": str(tu["ci"]), "Tau-U critical (a.05)": tu["critical"],
            "Tau-U p": tu["p_value"], "Tau (raw)": tu["tau"],
            "NAP": nap_v, "NAP CI95": str(nap_ci), "PND": pnd_v, "PEM": pem_v,
            "Strength NAP": nap_magnitude(nap_v), "Strength Tau-U": tau_magnitude(tu["tau_u"]),
            "Strength PND": pnd_magnitude(pnd_v), "Strength PEM": pem_magnitude(pem_v),
        })
    return pd.DataFrame(rows, columns=["Comparison", "Tau-U", "Tau-U variant", "Tau-U CI95",
                                       "Tau-U critical (a.05)", "Tau-U p", "Tau (raw)", "NAP",
                                       "NAP CI95", "PND", "PEM", "Strength NAP", "Strength Tau-U",
                                       "Strength PND", "Strength PEM"])


def pipeline_sced_alternating(df, *, session_col, condition_col, outcome_col,
                              conditions=None, outcome_type="continuous",
                              improvement="increase", max_consecutive=None, block_size=None,
                              time_covariate="none", detrend=None, perm_method="freedman-lane",
                              covariate_cols=None,
                              trend_test=True, test_time_trend=True, condition_order=None, dose=None,
                              power_analysis=False, power_effect=None, power_target=0.80,
                              power_n_sims=200, power_n_perm=200,
                              n_perm=5000, random_state=0,
                              save_path=None, plot_path=None, verbose=True):
    """
    End-to-end pipeline for an alternating-treatments / N-of-1 design (single unit).

    Steps: validate structure; **diagnose the series** (Mann-Kendall+Hamed-Rao
    trend, Sen's slope, autocorrelation) to flag a learning trend; describe each
    condition; run the **randomization test** as primary inference (difference in
    means for 2 conditions / between-condition variance for >=3 when
    ``detrend="none"``; otherwise the partial F for condition adjusting for the time
    trend, via ``condition_permutation_test`` with the chosen ``perm_method``);
    Tau-U/NAP/PND/PEM; flag carryover/order. With ``save_path`` set, writes a report
    (sheets ``Setup Info``, ``Diagnostics``, ``By Phase``, ``Effect Sizes``,
    ``Randomization Test``).

    ``detrend`` in {none, linear, log, **auto**}: remove a learning/time trend before
    testing. ``"auto"`` adopts the diagnostic's suggestion (``Suggested detrend``);
    this stays exact because the choice is made ONCE, before permuting, and the time
    basis ``Z`` depends only on the sessions - the same fixed statistic is applied to
    the observed data and every permutation. With an explicit value the analyst keeps
    full control: ``log`` when learning decelerates (typical practice curve),
    ``linear`` for a steady drift, ``none`` when the diagnostic shows no trend.

    ``perm_method`` in {**freedman-lane** (default), draper-stoneman, auto}: permutation
    scheme used when the (resolved) detrend != "none" (ignored otherwise). Freedman-Lane
    (permute reduced-model residuals) is robust across regimes and never does worse
    (Winkler et al. 2014). Draper-Stoneman (permute the condition labels) is exact for
    a randomised design with condition independent of time but the least robust under
    temporal collinearity. ``"auto"`` delegates to ``choose_permutation_method`` (DS when
    condition is independent of time, FL otherwise). The report always includes that
    recommendation.

    ``outcome_type`` in {continuous, count, binary} (descriptive formatting only;
    the inferential statistics are distribution-free). ``improvement`` in
    {increase, decrease} sets the beneficial direction for the nonoverlap indices.

    **Randomisation schedule provision** - ``max_consecutive`` (no more than k identical
    conditions in a row) and ``block_size`` (randomized-block: permute within consecutive
    blocks, e.g. ``block_size=len(conditions)`` for one of each per block) constrain the
    permutation set to match the schedule actually used. With neither, the default permutes
    the labels freely while preserving each condition's count (the exact reference set for
    a count-balanced design). The permutation distribution MUST mirror how conditions were
    really assigned, else the p-value is not exact (Edgington & Onghena 2007).

    **Ordered-trend test (dose-response)** - when ``trend_test`` and >=3 conditions, an
    ORDERED-condition test is reported IN ADDITION to the omnibus: is the outcome monotone
    across the ordered conditions (e.g. MI time 4/5/6 s)? Pass ``condition_order`` (low to
    high; defaults to ``conditions``) and optionally ``dose`` (numeric levels, e.g. (4,5,6))
    so the slope reads per dose unit. Two statistics: OLS slope on the dose (linear
    polynomial contrast) and Jonckheere-Terpstra (Terpstra 1952; Jonckheere 1954). Justified
    within the RT framework (any statistic is admissible; Edgington & Onghena 2007) but NOT
    power-validated for SCED alternation designs (Michiels et al. 2017 benchmarked only
    MD/NAP) - hence reported alongside, not instead, and to be pre-specified a priori
    (Manolov et al. 2021). When detrending, the trend test runs on the time-detrended
    residuals for consistency with the primary analysis.

    Returns ``(model_info: dict, randomization_result: dict)``.

    References: Edgington & Onghena 2007 (randomization test); Winkler et al. 2014
    (Freedman-Lane partial F); Terpstra 1952, Jonckheere 1954 (ordered-alternatives trend).
    R equivalent: SCRT (design-based randomization); permuco::lmperm (Freedman-Lane
    partial F) - no single-function equivalent for the full pipeline.
    """
    # `detrend` renamed to `time_covariate` (it puts the time slope in the model, it is not
    # a separate pre-detrend step). `detrend` kept as a backward-compatible alias.
    detrend = detrend if detrend is not None else time_covariate
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col)
    if conditions is None:
        conditions = tuple(report["Phases/conditions"])
    conditions = tuple(conditions)
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required for an ATD/N-of-1.")
    if len(clean) == 0:
        raise ValueError("No valid row after cleaning (non-numeric outcome "
                         "or entirely missing?).")
    missing = [c for c in conditions if (clean[condition_col] == c).sum() == 0]
    if missing:
        raise ValueError(f"Condition(s) absent from the data: {missing}.")

    sub = clean[clean[condition_col].isin(conditions)]
    warnings = (missing_data_warnings(df, outcome_col=outcome_col, session_col=session_col)
                + sced_data_warnings(sub, session_col=session_col, condition_col=condition_col,
                                     outcome_col=outcome_col, conditions=conditions))
    if warnings and verbose:
        print("Warning - data quality issues:")
        for msg in warnings:
            print(f"   - {msg}")
    bytime_df = describe_by_phase(sub, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, outcome_type=outcome_type)
    es_df = _effect_sizes_table(sub, condition_col, outcome_col, conditions, improvement)
    diag = diagnose_series(sub, session_col=session_col, outcome_col=outcome_col,
                           condition_col=condition_col)

    labels = sub[condition_col].to_numpy()
    values = pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy()
    sessions = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
    cov = None
    if covariate_cols:
        cov = sub[list(covariate_cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
    # resolve detrend="auto" ONCE, before permuting (keeps the test exact)
    detrend_used = diag["Suggested detrend"] if detrend == "auto" else detrend
    method_rec = None
    method_for_name = None
    if detrend_used != "none" or cov is not None:
        # partial F for condition adjusting for the time trend (+ covariates), chosen scheme
        method_rec = choose_permutation_method(labels, sessions, detrend=detrend_used)
        if block_size:
            # a block-randomized schedule's exact reference set is within-block label
            # permutation (Draper-Stoneman); Freedman-Lane permutes residuals freely and
            # cannot honour the block schedule, so force DS to keep the test exact.
            method_used = "draper-stoneman"
        else:
            method_used = method_rec["recommended_method"] if perm_method == "auto" else perm_method
        res = condition_permutation_test(values, labels, sessions, detrend=detrend_used,
                                         method=method_used, n_perm=n_perm,
                                         max_consecutive=max_consecutive, block_size=block_size,
                                         covariates=cov, random_state=random_state)
        method_for_name = res["method"]
        adj = ("time " + detrend_used) if detrend_used != "none" else "covariate(s)"
        if covariate_cols:
            adj += f" + {list(covariate_cols)}"
        stat_desc = f"Partial F (condition | {adj})"
        rand = {"observed": res["observed_F"], "p_value": res["p_value"],
                "n_perm": res["n_perm"], "two_sided": False, "statistic": stat_desc}
    else:
        scheme = (block_scheme(block_size) if block_size
                  else alternating_scheme(max_consecutive=max_consecutive))
        if len(conditions) == 2:
            stat = lambda lab, val: diff_in_means(lab, val, target=conditions[1],
                                                  reference=conditions[0])
            stat_desc = f"Difference of means ({conditions[1]} - {conditions[0]})"
            two_sided = True
        else:
            stat = omnibus_variance
            stat_desc = "Variance of between-condition means"
            two_sided = False
        rand = randomization_test(labels, values, statistic=stat, scheme=scheme,
                                  n_perm=n_perm, random_state=random_state, two_sided=two_sided)
        rand["statistic"] = stat_desc
    # NAME of the test (not just its statistic)
    test_name = sced_test_name(n_conditions=len(conditions), detrend=detrend_used,
                               grouped=False, method=method_for_name)

    # --- Ordered-trend (dose-response) test, reported IN ADDITION to the omnibus ---
    # Justification: an ordered/trend statistic is admissible in the randomization-test
    # framework (validity from the randomisation, not the statistic; Edgington & Onghena
    # 2007). The OLS slope on the dose is the linear polynomial contrast; Jonckheere-
    # Terpstra is the canonical nonparametric ordered-alternatives test (Terpstra 1952;
    # Jonckheere 1954). NOT validated for SCED alternation designs (Michiels et al. 2017
    # benchmarked only MD/NAP) -> kept alongside the omnibus, to be pre-specified a priori
    # (Manolov et al. 2021). The permutation reuses the schedule scheme (max_consecutive /
    # block_size) so it stays exact for the design.
    trend = None
    if trend_test and len(conditions) >= 3:
        order = tuple(condition_order) if condition_order else conditions
        _validate_condition_order(order, labels)
        if dose is not None and len(dose) != len(order):
            raise ValueError(f"dose ({list(dose)}) must have the same length as "
                             f"the condition order ({list(order)}).")
        # consistency with the primary analysis: detrend the values first when detrending
        tvals = values
        if detrend_used != "none":
            Z = _nuisance_basis(sessions, detrend_used)
            beta, *_ = np.linalg.lstsq(Z, values, rcond=None)
            tvals = values - Z @ beta
        scheme_t = (block_scheme(block_size) if block_size
                    else alternating_scheme(max_consecutive=max_consecutive))
        r_slope = randomization_test(
            labels, tvals, two_sided=False, n_perm=n_perm, random_state=random_state,
            statistic=make_ordered_trend_stat(order, kind="slope", dose=dose,
                                              improvement=improvement), scheme=scheme_t)
        r_jt = randomization_test(
            labels, tvals, two_sided=False, n_perm=n_perm, random_state=random_state,
            statistic=make_ordered_trend_stat(order, kind="jonckheere", dose=dose,
                                              improvement=improvement), scheme=scheme_t)
        dose_used = list(dose) if dose is not None else list(range(len(order)))
        trend = {"order": list(order), "dose": dose_used,
                 "slope_observed": round(float(r_slope["observed"]), 4),
                 "slope_p": round(float(r_slope["p_value"]), 4),
                 "jonckheere_observed": round(float(r_jt["observed"]), 4),
                 "jonckheere_p": round(float(r_jt["p_value"]), 4)}

    # --- Time-trend axis : TEST the session slope as a FIXED covariate adjusting for condition
    # (+ covariates), the scalar sibling of the cluster templates' trend axis. Exact-exchangeability
    # Huh-Jhun at small n (recommend_scheme decides HJ vs FL, and SCED lives in the small-n regime
    # where HJ is exact where FL/DS are only approximate). Reported IN ADDITION ; the condition
    # effect above stays the PRIMARY randomized inference. HJ is for a fixed covariate only - it is
    # NOT applied to the randomized condition factor.
    time_trend = None
    if test_time_trend and np.unique(sessions[np.isfinite(sessions)]).size >= 4:
        from ..core import huh_jhun_test, recommend_scheme
        cond_codes = pd.Categorical(labels, categories=list(conditions)).codes.astype(float)
        nuis = [cond_codes] + ([cov[:, j] for j in range(cov.shape[1])] if cov is not None else [])
        nkind = ["discrete"] + (["continuous"] * (cov.shape[1] if cov is not None else 0))
        rec = recommend_scheme(role="fixed", kind="continuous", n=int(np.asarray(sessions).size))
        hj = huh_jhun_test(values, effect=sessions, nuisance=nuis, nuisance_kind=nkind,
                           stat="W", n_perm=n_perm, seed=random_state)
        time_trend = {"slope": round(float(hj["slope"]), 5), "p": round(float(hj["p"]), 4),
                      "stat": round(float(hj["stat"]), 4), "scheme": hj["scheme"],
                      "recommended": rec["primary"]}

    carry = _carryover_diagnostic(sub, session_col, condition_col, outcome_col)

    model_info = {
        "Design": "Alternating treatments / N-of-1 (single unit)",
        "Rows dropped (missing)": report["Sessions dropped (missing)"],
        "Data warnings": warnings if warnings else "none",
        "Covariate adjustment": (f"{list(covariate_cols)} (ANCOVA) - assumed not "
                                 "affected by the condition" if covariate_cols else "none"),
        "Outcome type": outcome_type, "Outcome": outcome_col,
        "Conditions": list(conditions), "Improvement direction": improvement,
        "Primary inference (test)": test_name,
        "Test statistic": stat_desc,
        "Detrend": detrend_used + (" (auto)" if detrend == "auto" else ""),
        "Permutation method": (res["method"] + (" (auto)" if perm_method == "auto" else "")
                               if detrend_used != "none" else "n/a (no nuisance)"),
        "Permutations": n_perm,
        "Max consecutive constraint": max_consecutive if max_consecutive else "none",
        "Randomization p-value": round(rand["p_value"], 4),
        "Observed statistic": round(rand["observed"], 4),
    }
    model_info.update(report)
    model_info.update(diag)
    if method_rec is not None:
        model_info["Method recommendation"] = method_rec["recommended_method"]
        model_info["Condition-time R2"] = method_rec["condition_time_R2"]
        model_info["Method rationale"] = method_rec["rationale"]
    model_info.update(carry)

    if time_trend is not None:                          # HJ (small-n exact) time-trend axis
        model_info["Time-trend slope (/session, adj. condition)"] = time_trend["slope"]
        model_info["Time-trend slope p"] = time_trend["p"]
        model_info["Time-trend scheme"] = (f"{time_trend['scheme']} (W robust ; "
                                           f"recommended {time_trend['recommended']})")

    # ordered-trend results into model_info + a dedicated sheet (Explanation/Interpretation)
    trend_df = None
    if trend is not None:
        def _sig(p):
            return "significant" if p < 0.05 else "not significant"
        model_info["Ordered-trend test"] = (
            f"IN ADDITION to the omnibus - monotone dose-response over the order {trend['order']} "
            f"(dose {trend['dose']}). RT framework (Edgington & Onghena 2007); NOT validated "
            "for SCED alternation (Michiels et al. 2017) -> pre-specify (Manolov et al. 2021).")
        model_info["Ordered-trend slope (/dose unit)"] = trend["slope_observed"]
        model_info["Ordered-trend slope p"] = trend["slope_p"]
        model_info["Ordered-trend Jonckheere p"] = trend["jonckheere_p"]
        trend_df = pd.DataFrame([
            {"Statistic": "Slope on the dose (linear contrast)",
             "Observed": trend["slope_observed"], "p (randomization)": trend["slope_p"],
             "Explanation": "OLS slope of the outcome on the ordered dose - change in the "
                            "outcome per dose step (linear-trend contrast, classic "
                            "experimental design).",
             "Interpretation": f"Linear dose-response {_sig(trend['slope_p'])} (p="
                               f"{trend['slope_p']}). Oriented: >0 = beneficial with the dose."},
            {"Statistic": "Jonckheere-Terpstra (ordered alternative)",
             "Observed": trend["jonckheere_observed"], "p (randomization)": trend["jonckheere_p"],
             "Explanation": "Nonparametric ordered-alternatives test based on ranks "
                            "(Terpstra 1952; Jonckheere 1954) - robust, no linearity "
                            "assumption.",
             "Interpretation": f"Monotone trend {_sig(trend['jonckheere_p'])} (p="
                               f"{trend['jonckheere_p']})."},
            {"Statistic": " - Methodological note - ", "Observed": "", "p (randomization)": "",
             "Explanation": "Complements the omnibus (which ignores the condition order). RT "
                            "framework: any statistic is admissible (Edgington & Onghena 2007).",
             "Interpretation": "NOT validated for SCED alternation (Michiels et al. 2017: only "
                               "MD/NAP are); pre-specify a priori (Manolov et al. 2021)."},
        ], columns=["Statistic", "Observed", "p (randomization)", "Explanation", "Interpretation"])

    # power analysis on the realised design (MDES + retrospective; opt-in, slow)
    power_df = None
    if power_analysis:
        from functions.sced.power import power_report
        psum, power_df = power_report(
            sub, session_col=session_col, condition_col=condition_col, outcome_col=outcome_col,
            conditions=conditions, effect_of_interest=power_effect, target_power=power_target,
            detrend=(diag["Suggested detrend"] if detrend == "auto" else detrend),
            method=(perm_method if perm_method != "auto" else "freedman-lane"),
            max_consecutive=max_consecutive, n_sims=power_n_sims, n_perm=power_n_perm,
            random_state=random_state)
        model_info["Power - MDES (Cohen d @target)"] = psum["mdes_cohens_d"]
        if power_effect is not None:
            model_info["Power - retrospective (specified effect)"] = psum["retrospective_power"]

    rand_df = pd.DataFrame([{
        "Test": test_name, "Statistic": stat_desc, "Observed": round(rand["observed"], 4),
        "P-value (randomization)": round(rand["p_value"], 4),
        "Permutations": rand["n_perm"], "Two-sided": rand["two_sided"]}])
    diag_df = pd.DataFrame([{"Diagnostic": k, "Value": str(v)} for k, v in diag.items()],
                           columns=["Diagnostic", "Value"])

    # Methods paragraph (scientific-article style) for the report
    stats_used = (["trend"] if trend is not None else []) + \
                 (["posthoc"] if len(conditions) >= 3 else [])
    methods_txt = sced_methods_text(
        design="ATD / N-of-1", n_units=1, conditions=list(conditions), outcome=outcome_col,
        detrend=detrend_used, method=method_for_name, n_perm=n_perm,
        improvement=improvement, statistics_used=stats_used, dose=dose)
    methods_df = pd.DataFrame([{"Methods": methods_txt}])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)               # report goes straight in save_path (the
        full = os.path.join(save_path, f"{outcome_col}_sced_alternating_report.xlsx")  # Analyse dir)
        info_df = pd.DataFrame([{"Parameter": k,
                                 "Value": (", ".join(map(str, v)) if isinstance(v, list)
                                           else (str(v) if isinstance(v, dict) else v))}
                                for k, v in model_info.items()], columns=["Parameter", "Value"])
        from functions.sced.glossary import interpretation_glossary, ascii_sanitize_df as S, data_recap_df
        recap = data_recap_df(df, tier_col=None, session_col=session_col, phase_col=condition_col,
                              outcomes=[outcome_col], design="ATD / N-of-1 (randomised alternation)")
        # Model-based ADJUSTED EFFECTS (complement the omnibus + nonoverlap ES) : the time slope via
        # Freedman-Lane (fixed covariate) and the condition contrasts via Draper-Stoneman (randomized
        # dose). Notably fills the missing TIME-slope estimate. Best-effort (keeps the report robust).
        adj_df = None
        try:
            from functions.sced.cluster.report import scalar_adjusted_effects
            adj_df = scalar_adjusted_effects(values, sessions, labels, list(conditions),
                                             perm_phase="draper-stoneman", block_size=block_size,
                                             max_consecutive=max_consecutive, n_perm=n_perm)
        except Exception as e:                              # keep the report robust, but never SILENT
            adj_df = None
            if verbose:
                print(f"Adjusted effects sheet skipped: {type(e).__name__}: {e}")
        with pd.ExcelWriter(full, engine="xlsxwriter") as writer:
            S(recap).to_excel(writer, sheet_name="Data & design", index=False)   # first sheet
            S(info_df).to_excel(writer, sheet_name="Setup Info", index=False)
            S(diag_df).to_excel(writer, sheet_name="Diagnostics", index=False)
            S(rand_df).to_excel(writer, sheet_name="Randomization Test", index=False)
            if adj_df is not None:
                S(adj_df).to_excel(writer, sheet_name="Adjusted effects", index=False)
            if trend_df is not None:
                S(trend_df).to_excel(writer, sheet_name="Ordered Trend", index=False)
            S(es_df).to_excel(writer, sheet_name="Effect Sizes", index=False)
            S(bytime_df).to_excel(writer, sheet_name="By Phase", index=False)
            if power_df is not None:
                S(power_df).to_excel(writer, sheet_name="Power", index=False)
            S(methods_df).to_excel(writer, sheet_name="Methods", index=False)
            S(interpretation_glossary("rando")).to_excel(
                writer, sheet_name="Interpretation guide", index=False)
        if verbose:
            print(f"Report exported to {full}")
    elif verbose:
        print("Report not exported")

    if plot_path is not None:
        from functions.sced.plots.panels import plot_sced_alternating
        plot_sced_alternating(sub, session_col=session_col, condition_col=condition_col,
                              outcome_col=outcome_col, conditions=list(conditions),
                              save_path=plot_path)
        model_info["Figure"] = os.path.join(plot_path, f"{outcome_col}_sced_alternating_plot.png")

    return model_info, rand
