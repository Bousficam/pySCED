"""
SCED - Within-visit multilevel (3 levels): repeated measures within a visit.

Structure ``patient > visit > measures``. When a visit carries several raw measures
(e.g. 8 trials), this module estimates the 3-level variance decomposition and tests
the condition effect, the dispersion and the within-visit slope. Inference relies on
the visit-level randomization test (unit of assignment), consistent with the rest of
the toolbox. The reshaping functions (measures -> visit) are defined in
:mod:`functions.sced.prep` (aggregate_visits, aggregate_runs, collapse_long_measures,
unstack_outcomes).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from functions.sced.core import (
    validate_sced,
    condition_permutation_test,
    stratified_condition_permutation_test,
)
from functions.sced.prep import aggregate_visits


def dispersion_test(df, *, unit_col, session_col, condition_col, outcome_col,
                    measure="sd", conditions=None, detrend="none", method="freedman-lane",
                    n_perm=5000, max_consecutive=None, standardize=True, random_state=0):
    """
    Test whether the condition affects the **within-visit dispersion** (consistency):
    collapse each visit to its within-visit ``measure`` in {sd, cv}, then run the usual
    randomization test on that dispersion outcome (stratified by unit if ``unit_col``,
    else single-series). Returns ``(result_dict, visit_level_df)``.

    References: Baek et al. (2014); Ferron et al. (2009) (3-level SCED; within-visit structure).
    R equivalent: no direct R equivalent (visit-level stratified permutation on a within-visit dispersion outcome); SCRT for the permutation part - potential equivalent, to test.
    """
    if measure not in ("sd", "cv"):
        raise ValueError("measure must be 'sd' or 'cv'")
    vis = aggregate_visits(df, unit_col=unit_col, session_col=session_col,
                           condition_col=condition_col, outcome_col=outcome_col, stat=measure)
    vis = vis.dropna(subset=[outcome_col])
    labels = vis[condition_col].to_numpy()
    values = vis[outcome_col].to_numpy()
    sess = pd.to_numeric(vis[session_col], errors="coerce").to_numpy()
    grouped = unit_col is not None and vis[unit_col].nunique() > 1
    if grouped:
        res = stratified_condition_permutation_test(
            values, labels, vis[unit_col].to_numpy(), sess, detrend=detrend, method=method,
            n_perm=n_perm, max_consecutive=max_consecutive, standardize=standardize,
            random_state=random_state)
    else:
        res = condition_permutation_test(values, labels, sess, detrend=detrend, method=method,
                                         n_perm=n_perm, max_consecutive=max_consecutive,
                                         random_state=random_state)
    res["dispersion_measure"] = measure
    res["note"] = ("Outcome = within-visit dispersion: tests whether the condition affects "
                   "CONSISTENCY (not the mean).")
    return res, vis


def within_visit_slope_test(df, *, unit_col, session_col, condition_col, outcome_col,
                            trial_col, conditions=None, detrend="none", method="freedman-lane",
                            n_perm=5000, max_consecutive=None, standardize=True, random_state=0):
    """
    Test whether the condition affects the **within-visit slope** - the trajectory across
    a visit's ordered repeated measures (``trial_col`` = the measure order, e.g. 1..8;
    captures within-session fatigue / warm-up / practice).

    Two-stage: per visit, the OLS slope of ``outcome ~ trial_col`` is computed (one slope
    per visit), then the usual randomization test runs on that slope as the visit-level
    outcome (stratified by unit if ``unit_col`` has >=2 units, else single-series). A
    within-visit order is **required** - without it the measures are exchangeable and no
    slope is defined. Returns ``(result_dict, visit_level_df)``.

    References: Baek et al. (2014); Ferron et al. (2009) (3-level SCED; within-visit trajectory).
    R equivalent: no direct R equivalent (per-visit OLS slope + SCRT visit-level permutation) - potential equivalent, to test.
    """
    if conditions is not None:
        df = df[df[condition_col].isin(list(conditions))]
    rows = []
    for (u, v), g in df.groupby([unit_col, session_col]):
        x = pd.to_numeric(g[trial_col], errors="coerce").to_numpy()
        y = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        slope = float(np.polyfit(x, y, 1)[0]) if (len(np.unique(x)) >= 2 and len(y) >= 2) else np.nan
        rows.append({unit_col: u, session_col: v,
                     condition_col: g[condition_col].iloc[0], "_slope": slope})
    vis = pd.DataFrame(rows).dropna(subset=["_slope"])
    labels = vis[condition_col].to_numpy()
    values = vis["_slope"].to_numpy()
    sess = pd.to_numeric(vis[session_col], errors="coerce").to_numpy()
    grouped = unit_col is not None and vis[unit_col].nunique() > 1
    if grouped:
        res = stratified_condition_permutation_test(
            values, labels, vis[unit_col].to_numpy(), sess, detrend=detrend, method=method,
            n_perm=n_perm, max_consecutive=max_consecutive, standardize=standardize,
            random_state=random_state)
    else:
        res = condition_permutation_test(values, labels, sess, detrend=detrend, method=method,
                                         n_perm=n_perm, max_consecutive=max_consecutive,
                                         random_state=random_state)
    # ESTIMATION (not only the test): MEAN within-visit slope per condition, and overall -
    # to read the effect (e.g. steeper downward trajectory = fatigue) per condition.
    by_cond = vis.groupby(condition_col)["_slope"].mean().round(4)
    res["slope_by_condition"] = {str(k): float(v) for k, v in by_cond.items()}
    res["slope_overall"] = round(float(vis["_slope"].mean()), 4) if len(vis) else np.nan
    # PER PATIENT: mean within-visit slope per patient, and per patient x condition
    res["slope_by_patient"] = {str(u): round(float(s), 4)
                               for u, s in vis.groupby(unit_col)["_slope"].mean().items()}
    res["slope_by_patient_condition"] = {
        f"{u}|{c}": round(float(s), 4)
        for (u, c), s in vis.groupby([unit_col, condition_col])["_slope"].mean().items()}
    res["note"] = ("Outcome = within-visit slope (outcome ~ trial per visit): tests whether the "
                   "condition changes the TRAJECTORY within a visit (slope_by_condition = mean "
                   "slope per condition; <0 = decline/fatigue, >0 = rise/warm-up).")
    return res, vis.rename(columns={"_slope": f"{outcome_col}_within_slope"})


def variance_decomposition_3level(df, *, unit_col, session_col, condition_col, outcome_col,
                                  conditions=None, detrend="none", method="freedman-lane",
                                  n_perm=5000, max_consecutive=None, standardize=True,
                                  random_state=0):
    """
    3-level variance decomposition (measures > visit > patient) on the RAW measures,
    with a visit-level permutation p for the condition effect.

    The mixed model ``outcome ~ condition (+ time) + (1|patient) + (1|visit-in-patient)``
    splits the variance into **between-patient**, **between-visit (within patient)** and
    **within-visit (residual = measurement noise)**. The condition effect is estimated
    there, but its p comes from the stratified randomization test on the **visit means**
    (the visit is the unit of assignment). Returns ``(summary, sheet)`` with the variance
    components, their share, ICCs and the permutation p. Gaussian outcomes.

    References: Baek, Moeyaert, Petit-Bois, Beretvas, Van den Noortgate and Ferron (2014); Moeyaert, Ferron, Beretvas and Van den Noortgate (2014) (3-level SCED variance decomposition).
    R equivalent: lme4::lmer(outcome ~ condition + (1|patient/visit)) or nlme::lme (nested random effects); performance::icc for the ICCs. Visit-level permutation p - potential equivalent, to test.
    """
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, unit_col=unit_col)
    if conditions is None:
        conditions = tuple(report["Phases/conditions"])
    conditions = tuple(conditions)
    if report["Number of units"] < 2:
        raise ValueError("The 3-level decomposition requires at least two patients.")
    sub = clean[clean[condition_col].isin(conditions)].copy()

    # p-value: permutation at the VISIT level (on visit means) - visit = unit of assignment
    means = aggregate_visits(sub, unit_col=unit_col, session_col=session_col,
                             condition_col=condition_col, outcome_col=outcome_col, stat="mean")
    means = means.dropna(subset=[outcome_col])
    p_common = stratified_condition_permutation_test(
        means[outcome_col].to_numpy(), means[condition_col].to_numpy(),
        means[unit_col].to_numpy(),
        pd.to_numeric(means[session_col], errors="coerce").to_numpy(),
        detrend=detrend, method=method, n_perm=n_perm, max_consecutive=max_consecutive,
        standardize=standardize, random_state=random_state)["p_value"]

    # 3-level mixed model on RAW measures for the variance components
    d = sub.rename(columns={outcome_col: "_y", condition_col: "_c", unit_col: "_u"})
    d["_y"] = pd.to_numeric(d["_y"], errors="coerce")
    d["_visit"] = d["_u"].astype(str) + "_" + sub[session_col].astype(str)   # unique per patient
    fixed = "_y ~ C(_c)"
    if detrend in ("linear", "log"):
        t = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
        d["_t"] = t if detrend == "linear" else np.log(t - np.nanmin(t) + 1.0)
        fixed += " + _t"
    trt = conditions[1]
    slope_name = f"C(_c)[T.{trt}]"

    summary = {"conditions": list(conditions), "n_units": report["Number of units"],
               "p_common_effect_perm": round(p_common, 4)}
    note = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(fixed, d, groups=d["_u"], re_formula="~1",
                              vc_formula={"visit": "0 + C(_visit)"}).fit()
        v_pat = float(res.cov_re.iloc[0, 0])
        v_vis = float(np.asarray(res.vcomp)[0])
        v_res = float(res.scale)
        tot = v_pat + v_vis + v_res
        eff = float(res.fe_params[slope_name]) if slope_name in res.fe_params.index else np.nan
        summary.update({
            "var_patient": round(v_pat, 4), "var_visit_within_patient": round(v_vis, 4),
            "var_within_visit": round(v_res, 4),
            "pct_patient": round(100 * v_pat / tot, 1), "pct_visit": round(100 * v_vis / tot, 1),
            "pct_within_visit": round(100 * v_res / tot, 1),
            "icc_patient": round(v_pat / tot, 4), "condition_effect": round(eff, 4)})
    except Exception as exc:
        note = f"3-level model not fitted ({type(exc).__name__}) - permutation p only."
        summary.update({"var_patient": "n/a", "var_visit_within_patient": "n/a",
                        "var_within_visit": "n/a", "condition_effect": "n/a"})

    rows = [
        {"Quantity": "Between-patient variance", "Value": summary["var_patient"],
         "Explanation": "Level differences between patients.",
         "Interpretation": f"{summary.get('pct_patient', 'n/a')}% of the total variance."},
        {"Quantity": "Between-visit variance (within patient)", "Value": summary["var_visit_within_patient"],
         "Explanation": "Fluctuation from one visit to another within the same patient.",
         "Interpretation": f"{summary.get('pct_visit', 'n/a')}% of the total variance."},
        {"Quantity": "Within-visit variance (measure)", "Value": summary["var_within_visit"],
         "Explanation": "Measurement noise between the repetitions of the same visit.",
         "Interpretation": f"{summary.get('pct_within_visit', 'n/a')}% of the total variance."},
        {"Quantity": f"Condition effect ({trt} vs {conditions[0]})", "Value": summary["condition_effect"],
         "Explanation": "Estimated mean effect (3-level model).",
         "Interpretation": (f"{'significant' if p_common < 0.05 else 'not significant'} effect "
                            f"(visit-level permutation p = {summary['p_common_effect_perm']}).")},
    ]
    if note:
        rows.append({"Quantity": "Note", "Value": " - ", "Explanation": note,
                     "Interpretation": "permutation p valid."})
    sheet = pd.DataFrame(rows, columns=["Quantity", "Value", "Explanation", "Interpretation"])
    return summary, sheet
