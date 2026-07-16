"""
SCED - Hierarchical (multilevel) model for replicated alternating designs (GROUP)
=================================================================================

Hierarchical = sessions nested in patients. This is a **group-only** concept (it makes
no sense for a single unit). A linear mixed model
``outcome ~ condition (+ time) + (intercept + condition slope | patient)`` adds what the
fixed-effects tools don't: **partial-pooling (shrunk) per-patient effects**, the
**random-slope variance** (the parametric analogue of the heterogeneity test) and the
**ICC**.

Key choice (rigorous with few clusters): the mixed model supplies the *estimates*, but
the **p-values come from permutation**, not the model's asymptotic p - with ~5 patients
the latter is not trustworthy. The fixed condition effect's p is the design-based
**stratified randomization** p; the random-slope (heterogeneity) p is the
**interaction permutation** p. Both reuse the project's exact tests.

References: Laird & Ware (1982) random-effects models; for replicated single-case
multilevel analysis see Moeyaert et al. (2014).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from functions.sced.core import (
    validate_sced,
    stratified_condition_permutation_test,
    heterogeneity_test,
    icc_band, i2_band, ICC_BANDS, I2_BANDS, G_BANDS,
)


def hierarchical_group_model(df, *, unit_col, session_col, condition_col, outcome_col,
                             conditions=None, outcome_type="continuous", detrend="none",
                             method="freedman-lane", n_perm=5000, max_consecutive=None,
                             standardize=True, random_state=0):
    """
    Fit a multilevel model to a replicated alternating design and pair it with
    permutation p-values.

    Returns ``(summary: dict, sheet: DataFrame)``. ``summary`` holds the population
    condition effect(s), the variance components (random intercept, random condition
    slope, residual), the ICC, the shrunk per-patient effect for the primary contrast,
    and the two permutation p-values (common effect; heterogeneity). ``sheet`` is a
    tidy table with ``Explanation`` / ``Interpretation`` columns. If the mixed model
    fails to converge it degrades gracefully to the permutation p-values + a note.

    References: Laird and Ware (1982) (random-effects models); Moeyaert, Ferron, Beretvas and Van den Noortgate (2014) (multilevel analysis of replicated single-case designs).
    R equivalent: nlme::lme or lme4::lmer (lme4::glmer for binary/count); design-based p via SCRT / scan randomization - potential equivalent, to test.
    """
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, unit_col=unit_col)
    if conditions is None:
        conditions = tuple(report["Phases/conditions"])
    conditions = tuple(conditions)
    if report["Number of units"] < 2:
        raise ValueError("The hierarchical model requires at least two units.")
    sub = clean[clean[condition_col].isin(conditions)].copy()

    labels = sub[condition_col].to_numpy()
    values = pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy()
    sessions = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
    units = sub[unit_col].to_numpy()

    # permutation p-values (the trustworthy ones with few clusters)
    p_common = stratified_condition_permutation_test(
        values, labels, units, sessions, detrend=detrend, method=method, n_perm=n_perm,
        max_consecutive=max_consecutive, standardize=standardize,
        random_state=random_state)["p_value"]
    het = heterogeneity_test(values, labels, units, sessions, conditions=conditions,
                             detrend=detrend, n_perm=n_perm, max_consecutive=max_consecutive,
                             standardize=standardize, random_state=random_state)
    p_hetero = het["interaction_p"]

    # model for the ESTIMATES (the p-values stay from permutation above).
    # continuous -> linear mixed model ; binary/count -> GLMM (logistic/Poisson mixed).
    d = sub.rename(columns={outcome_col: "_y", condition_col: "_c", unit_col: "_u"})
    d["_y"] = pd.to_numeric(d["_y"], errors="coerce")
    fixed = "_y ~ C(_c)"
    if detrend == "linear":
        d["_t"] = sessions; fixed += " + _t"
    elif detrend == "log":
        d["_t"] = np.log(sessions - np.nanmin(sessions) + 1.0); fixed += " + _t"
    ref, trt = conditions[0], conditions[1]
    slope_name = f"C(_c)[T.{trt}]"

    summary = {"conditions": list(conditions), "n_units": report["Number of units"],
               "outcome_type": outcome_type,
               "p_common_effect_perm": round(p_common, 4),
               "p_heterogeneity_perm": (round(p_hetero, 4) if np.isfinite(p_hetero) else "n/a"),
               "I2_percent": het["I2_percent"]}
    mlm_note, scale_note = "", ""
    try:
        if outcome_type == "continuous":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(fixed, d, groups=d["_u"], re_formula="~C(_c)").fit(reml=True)
            fe, cov_re = res.fe_params, res.cov_re
            var_int = float(cov_re.iloc[0, 0]) if cov_re.shape[0] else np.nan
            var_slope = (float(cov_re.loc[slope_name, slope_name])
                         if slope_name in cov_re.index else np.nan)
            resid = float(res.scale)
            icc = var_int / (var_int + resid) if np.isfinite(var_int) and (var_int + resid) > 0 else np.nan
            pop_effect = float(fe[slope_name]) if slope_name in fe.index else np.nan
            per = {str(u): round(pop_effect + float(re_u.get(slope_name, 0.0)), 3)
                   for u, re_u in res.random_effects.items()}
            summary.update({
                "population_effect": round(pop_effect, 4) if np.isfinite(pop_effect) else "n/a",
                "var_random_intercept": round(var_int, 4) if np.isfinite(var_int) else "n/a",
                "sd_random_slope": round(float(np.sqrt(var_slope)), 4) if np.isfinite(var_slope) else "n/a",
                "residual_var": round(resid, 4), "icc": round(icc, 4) if np.isfinite(icc) else "n/a",
                "per_patient_effect_shrunk": per})
        else:                                          # binary / count -> GLMM (Bayes VB)
            from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
                                                            PoissonBayesMixedGLM)
            GLMM = BinomialBayesMixedGLM if outcome_type == "binary" else PoissonBayesMixedGLM
            scale_note = "logit scale" if outcome_type == "binary" else "log scale"
            vc = {"unit": "0 + C(_u)", "unitcond": "0 + C(_u):C(_c)"}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gr = GLMM.from_formula(fixed, vc, d).fit_vb()
            fe = dict(zip(gr.model.exog_names, np.asarray(gr.fe_mean)))
            sds = dict(zip(gr.model.vcp_names, np.exp(np.asarray(gr.vcp_mean))))
            pop_effect = float(fe.get(slope_name, np.nan))
            sd_int, sd_slope_v = sds.get("unit", np.nan), sds.get("unitcond", np.nan)
            icc = (sd_int ** 2 / (sd_int ** 2 + np.pi ** 2 / 3)        # latent-scale (logit)
                   if outcome_type == "binary" and np.isfinite(sd_int) else np.nan)
            summary.update({
                "population_effect": round(pop_effect, 4) if np.isfinite(pop_effect) else "n/a",
                "var_random_intercept": round(float(sd_int ** 2), 4) if np.isfinite(sd_int) else "n/a",
                "sd_random_slope": round(float(sd_slope_v), 4) if np.isfinite(sd_slope_v) else "n/a",
                "icc": round(float(icc), 4) if np.isfinite(icc) else "n/a (non-linear scale)",
                "per_patient_effect_shrunk": {}})
    except Exception as exc:                       # convergence / singular fit
        mlm_note = f"Model not fitted ({type(exc).__name__}) - permutation p only."
        summary.update({"population_effect": "n/a", "sd_random_slope": "n/a", "icc": "n/a"})

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    sd_slope = summary.get("sd_random_slope", "n/a")
    hetero_sig = np.isfinite(p_hetero) and p_hetero < 0.05
    eff_unit = f" ({scale_note})" if scale_note else ""
    icc_v, i2_v = _f(summary.get("icc")), _f(het["I2_percent"])
    pop_v, var_int_v = _f(summary.get("population_effect")), _f(summary.get("var_random_intercept"))
    resid_v, sd_slope_v = _f(summary.get("residual_var")), _f(sd_slope)
    tot_v = var_int_v + resid_v
    # standardized population effect (~ Cohen's d) if variances available, as a benchmark
    d_like = pop_v / np.sqrt(tot_v) if np.isfinite(pop_v) and np.isfinite(tot_v) and tot_v > 0 else np.nan
    d_band = ("small" if abs(d_like) < 0.5 else "medium" if abs(d_like) < 0.8 else "large") \
        if np.isfinite(d_like) else "n/a"

    rows = [
        {"Quantity": f"Population effect ({trt} - {ref}){eff_unit}",
         "Value": summary["population_effect"],
         "Explanation": f"Mean condition effect, partial pooling (mixed model{eff_unit}). "
                        "The p comes from permutation (the mixed model's asymptotic p is not "
                        "trustworthy at ~5 patients).",
         "Interpretation": (
             f"Common effect {'significant' if p_common < 0.05 else 'not significant'} "
             f"(permutation p={summary['p_common_effect_perm']})"
             + (f"; standardized d~{round(d_like, 2)} ({d_band} - {G_BANDS})."
                if np.isfinite(d_like) else "."))},
        {"Quantity": "Between-patient variance (random intercept)",
         "Value": summary.get("var_random_intercept", "n/a"),
         "Explanation": "Dispersion of the baseline LEVEL from one patient to another.",
         "Interpretation": (f"{round(100 * var_int_v / tot_v)}% of the variance comes from "
                            f"level differences between patients."
                            if np.isfinite(var_int_v) and np.isfinite(tot_v) and tot_v > 0 else " - ")},
        {"Quantity": "Residual variance (within patient)",
         "Value": summary.get("residual_var", "n/a"),
         "Explanation": "Remaining variability within a patient (session to session / noise).",
         "Interpretation": (f"{round(100 * resid_v / tot_v)}% of the variance is within-patient."
                            if np.isfinite(resid_v) and np.isfinite(tot_v) and tot_v > 0 else " - ")},
        {"Quantity": "ICC (within-patient correlation)",
         "Value": summary.get("icc", "n/a"),
         "Explanation": "Share of the variance due to BETWEEN-patient differences (vs within). "
                        f"Benchmark: {ICC_BANDS}.",
         "Interpretation": (
             f"ICC={round(icc_v, 2)} -> {icc_band(icc_v)}: {round(100 * icc_v)}% of the variance "
             "is between patients; two measures of the same patient resemble each other that much. "
             "High ICC = very different patients -> the group analysis must neutralize these "
             "gaps (hence the within-patient standardization)." if np.isfinite(icc_v) else " - ")},
        {"Quantity": "Heterogeneity - random slope SD",
         "Value": sd_slope,
         "Explanation": "Standard deviation of the condition EFFECT between patients (individual "
                        f"effects spread ~ effect +/- 2.SD). I2 benchmark: {I2_BANDS}.",
         "Interpretation": (
             (f"{'HETEROGENEOUS' if hetero_sig else 'homogeneous'} effects between patients "
              f"(p={summary['p_heterogeneity_perm']}, I2={het['I2_percent']}% -> {i2_band(i2_v)})")
             + (f"; individual effects ~ {round(pop_v, 2)} +/- {round(2 * sd_slope_v, 2)} = "
                f"[{round(pop_v - 2 * sd_slope_v, 2)}, {round(pop_v + 2 * sd_slope_v, 2)}]."
                if np.isfinite(pop_v) and np.isfinite(sd_slope_v) else "."))},
    ]
    # individual (shrunk / partial pooling) effects - one per patient
    per = summary.get("per_patient_effect_shrunk", {}) or {}
    for u, v in per.items():
        rows.append({"Quantity": f"Shrunk individual effect - {u}", "Value": v,
                     "Explanation": "Patient's condition effect, pulled toward the group mean "
                                    "(partial pooling) - more stable than a raw effect.",
                     "Interpretation": (f"{'positive' if _f(v) > 0 else 'negative'} direction; to "
                                        "compare with the corrected individual test (Per Unit sheet).")})
    # plain-language SYNTHESIS line (how to read the group hierarchy)
    a = ("present" if p_common < 0.05 else "ABSENT")
    b = (f"very different (ICC={round(icc_v, 2)} {icc_band(icc_v)})"
         if np.isfinite(icc_v) and icc_v >= 0.75 else
         f"moderately different (ICC={round(icc_v, 2)} {icc_band(icc_v)})"
         if np.isfinite(icc_v) and icc_v >= 0.5 else
         f"close (ICC={round(icc_v, 2)} {icc_band(icc_v)})" if np.isfinite(icc_v) else "?")
    cc = ("HETEROGENEOUS" if hetero_sig else "homogeneous") + \
         (f" (I2={het['I2_percent']}% {i2_band(i2_v)})" if np.isfinite(i2_v) else "")
    concl = ("Mean condition effect absent; what dominates is the LEVEL differences between "
             "patients (not their response to the condition)."
             if p_common >= 0.05 and not hetero_sig else
             "Effect varies across patients: look at the individual effects (Per Unit / rows "
             "above)." if hetero_sig else
             "Condition effect common to the patients.")
    rows.append({
        "Quantity": "Overall reading (group hierarchy)", "Value": " - ",
        "Explanation": "How to read: (a) is there a condition effect ON AVERAGE? (population "
                       "effect / p); (b) do patients differ in LEVEL? (ICC); "
                       "(c) does the effect differ BETWEEN patients? (slope SD / I2).",
        "Interpretation": f"(a) group effect {a} (p={summary['p_common_effect_perm']}); "
                          f"(b) patients {b}; (c) effect {cc}. -> {concl}"})
    if mlm_note:
        rows.append({"Quantity": "Note", "Value": " - ",
                     "Explanation": mlm_note, "Interpretation": "permutation p valid."})
    sheet = pd.DataFrame(rows, columns=["Quantity", "Value", "Explanation", "Interpretation"])
    return summary, sheet
