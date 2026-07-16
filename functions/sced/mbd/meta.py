"""
SCED - Bayesian meta-analysis (stage 2 of a two-stage approach)
===============================================================

Two ways to synthesize several cases (tiers / patients) in SCED:

  - ONE stage - multilevel / hierarchical model (``SCED_mbd_bayesian`` with
    ``pooling='partial'``): everything is estimated jointly, the per-case effects are
    shrunk toward the population. Efficient, but with few cases the between-case tau is
    poorly estimated and the heterogeneity of effect *type* (one "level" case, another
    "slope" case) is averaged and then masked.

  - TWO stages - meta-analysis:
      stage 1 = INDEPENDENT per-case fit (``pooling='none'``) -> raw effect_i + SE_i;
      stage 2 = THIS function = random-effects meta-analysis on (effect_i, SE_i) -> mu, tau,
      I^2, and posterior-shrunk theta_i. The stage-1 per-case effects are NOT
      contaminated by the group; shrinkage happens only at stage 2, transparently
      (raw-vs-shrunk forest plot).

DECISION RULE (what to choose):
  - few points/cases (< ~8-10 per phase), homogeneous effect, POPULATION statement primary
        -> one-stage hierarchical (borrowing of strength, stable).
  - enough points/cases (>= ~10-12), heterogeneity of effect TYPE, PER-CASE decision
    (responder identification) primary
        -> two-stage meta-analysis (uncontaminated per-case + clean mu/tau/I^2).
  - in all cases: DETECTION is anchored on the randomization test (trend-robust);
    the Bayesian part serves MAGNITUDE + credibility. If one-stage and two-stage diverge
    strongly on mu, that is a heterogeneity signal to report, not a bug.

IDENTIFYING RESPONDERS (binarizing patients) -> look at the RAW, not the shrunk:
  - binarize on the RAW value (stage-1 HDI/pd), ideally:
      * threshold pre-specified vs an MCID, not only vs 0;
      * backed by the PER-CASE randomization test (exact, design-based, trend-robust) as the
        backbone of detection - the raw Bayesian value then gives the magnitude + credibility;
      * accounting for multiplicity (k patients) and wide uncertainty (few points).
  - the SHRUNK value serves: the GROUP statement (mu), a stabilized ranking (better MSE under
    exchangeability), a regularized estimate - NOT to decide responder / non-responder
    (it pulls the theta_i toward mu: false positives near mu, false negatives on the
    idiosyncratic responders; its pd partly reflects the group evidence, not the patient's).
    -> Raw = who responded; shrunk/mu = what the population says.

Model (stage 2), NON-CENTERED parameterization (avoids the funnel/divergences):
    mu      ~ Normal(0, prior_mu_sd)       # population effect
    tau     ~ HalfNormal(prior_tau_sd)     # between-case standard deviation (heterogeneity)
    theta_i = mu + tau * z_i ,  z_i ~ Normal(0, 1)
    eff_i   ~ Normal(theta_i, SE_i)        # measurement error (stage-1 uncertainty)

References (meta-analysis & SCED):
  - Van den Noortgate & Onghena (2003) Behavior Research Methods 35(1):1-10 - multilevel
    models for integrating effects in single-case research.
  - Van den Noortgate & Onghena (2008) Evidence-Based Comm. Assessment & Intervention 2(3) -
    multilevel meta-analysis of SCED studies.
  - Moeyaert, Ugille, Ferron, Beretvas & Van den Noortgate (2014) J. School Psychology 52 -
    from single-level analysis to the three-level multilevel model.
  - Pustejovsky, Hedges & Shadish (2014) JEBS 39(5):368-393 - design-comparable effects
    (BC-SMD) in multiple-baseline designs (R package ``scdhlm``).
  - Hedges, Pustejovsky & Shadish (2012, 2013) Research Synthesis Methods - d-statistic
    for AB and multiple-baseline designs.
  - Rindskopf (2014) - Bayesian estimation of SCED models (stabilizes tau with few cases).
  - Burke, Ensor & Riley (2017) Statistics in Medicine 36 - one-stage vs two-stage in
    meta-analysis (general framework: agree when well done, diverge on sparse data).
  - Higgins & Thompson (2002) Statistics in Medicine 21 - I^2 heterogeneity measure.
"""
import numpy as np
import pymc as pm
import arviz as az


def _pd(x):
    """Probability of direction: the larger tail mass of ``x`` on one side of 0, in [0.5, 1].
    Input: a draw array. Output: float.

    References: Makowski, Ben-Shachar & Ludecke 2019 (probability of direction).
    R equivalent: bayestestR::p_direction."""
    x = np.asarray(x).ravel()
    return float(max((x > 0).mean(), (x < 0).mean()))


def _hdi(x, prob=0.95):
    """Shortest HDI (no ArviZ dependency; robust to small sizes).

    References: Kruschke 2018 (highest density interval).
    R equivalent: bayestestR::hdi."""
    x = np.sort(np.asarray(x).ravel())
    n = len(x)
    k = int(np.floor(prob * n))
    if k < 1 or k >= n:
        return (float(x.min()), float(x.max()))
    widths = x[k:] - x[:n - k]
    i = int(np.argmin(widths))
    return (float(x[i]), float(x[i + k]))


def _summ(x):
    """Posterior summary of a draw array: median, mean, 95% HDI bounds and probability of
    direction. Input: draws. Output: dict."""
    x = np.asarray(x).ravel()
    lo, hi = _hdi(x)
    return {"median": float(np.median(x)), "mean": float(np.mean(x)),
            "hdi_low": lo, "hdi_high": hi, "pd": _pd(x)}


def bayes_meta_analysis(effects, ses, *, labels=None, improvement="increase",
                        draws=2000, tune=2000, chains=4, seed=42, target_accept=0.95,
                        prior_mu_sd=None, prior_tau_sd=None, progressbar=False):
    """Bayesian random-effects meta-analysis (normal-normal, non-centered) on PER-CASE effects
    and their standard errors. The effects are assumed ALREADY oriented (positive = improvement;
    this holds for the stage-1 ``effect_end_pts`` Deterministics). See the module docstring.

    Parameters
    ----------
    effects : array (k,) - observed effect per case (original scale, e.g. points).
    ses     : array (k,) - standard error per case (= stage-1 posterior SD).
    labels  : case names (else case0..k-1).
    improvement : 'increase' | 'decrease' - clinical direction (does NOT orient here; informational).
    prior_mu_sd / prior_tau_sd : prior standard deviations (auto = data scale if None).

    Returns: dict {population, heterogeneity{tau, I2, prediction_interval}, per_case, k,
                   improvement, rhat_max, diverging, _idata}.

    References: Van den Noortgate & Onghena 2003, 2008; Higgins & Thompson 2002 (I^2); Burke,
    Ensor & Riley 2017 (two-stage random-effects Bayesian meta-analysis).
    R equivalent: brms (brm(effect | se(se) ~ 1 + (1|case))) / bayesmeta; metafor::rma (frequentist).
    """
    eff = np.asarray(effects, float).ravel()
    se = np.asarray(ses, float).ravel()
    if eff.shape != se.shape:
        raise ValueError("effects and ses must have the same length.")
    k = len(eff)
    if k < 2:
        raise ValueError("Meta-analysis: at least 2 cases required.")
    if labels is None:
        labels = [f"case{i}" for i in range(k)]
    labels = [str(x) for x in labels]
    pos = se[np.isfinite(se) & (se > 0)]
    se = np.where(np.isfinite(se) & (se > 0), se, np.median(pos) if pos.size else 1.0)

    scale = max(float(np.nanstd(eff)), float(np.median(se)), 1e-6)
    mu_sd = float(prior_mu_sd) if prior_mu_sd is not None else 10.0 * scale
    tau_sd = float(prior_tau_sd) if prior_tau_sd is not None else scale

    with pm.Model(coords={"case": labels}):
        mu = pm.Normal("mu", 0.0, mu_sd)
        tau = pm.HalfNormal("tau", tau_sd)
        z = pm.Normal("z", 0.0, 1.0, dims="case")                  # non-centered
        theta = pm.Deterministic("theta", mu + tau * z, dims="case")
        pm.Normal("obs", mu=theta, sigma=se, observed=eff, dims="case")
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=progressbar,
                          compute_convergence_checks=False)

    post = idata.posterior
    mu_s = post["mu"].values.ravel()
    tau_s = post["tau"].values.ravel()

    # I^2 (Higgins-Thompson): share of variance due to true heterogeneity.
    w = 1.0 / se ** 2
    denom = (np.sum(w) ** 2 - np.sum(w ** 2))
    s2_typ = (k - 1) * np.sum(w) / denom if denom > 0 else float("nan")   # "typical" within-study variance
    tau2 = float(np.mean(tau_s ** 2))
    I2 = float(tau2 / (tau2 + s2_typ)) if np.isfinite(s2_typ) and (tau2 + s2_typ) > 0 else float("nan")

    # PREDICTION interval: plausible range of the effect for a NEW case (mu + tau noise).
    rng = np.random.default_rng(seed)
    pred = mu_s + tau_s * rng.standard_normal(len(mu_s))

    th = post["theta"]
    per = [{"case": lab, "effect_raw": float(eff[i]), "se": float(se[i]),
            "shrunk": _summ(th.isel(case=i).values.ravel())} for i, lab in enumerate(labels)]

    return {"population": _summ(mu_s),
            "heterogeneity": {"tau": _summ(tau_s), "I2": I2,
                              "prediction_interval": tuple(_hdi(pred))},
            "per_case": per, "k": k, "improvement": improvement,
            "rhat_max": float(az.summary(idata)["r_hat"].max()),
            "diverging": int(idata.sample_stats["diverging"].values.sum()),
            "_idata": idata}


def meta_from_idata(idata, *, estimand="effect_end", labels=None, **kwargs):
    """Meta-analysis (stage 2) directly from a stage-1 ``pooling='none'`` model.

    estimand : PER-CASE Deterministic to synthesize. Default ``'effect_end'`` =
    ``effect_end_pts`` = TOTAL end-of-phase effect (b2 + b3*T_B), the best identified.
    Others: any per-case variable name present in the posterior (dims ending in 'case').
    Posterior mean + SD per case are extracted, then ``bayes_meta_analysis`` is called.

    References: Van den Noortgate & Onghena 2008 (two-stage multilevel meta-analysis of SCED).
    R equivalent: brms - potential equivalent, to test.
    """
    post = idata.posterior
    key = {"effect_end": "effect_end_pts", "level": "level_pts",
           "slope": "slope_pts", "trend": "trend_pts"}.get(estimand, estimand)
    if key not in post.data_vars:
        raise KeyError(f"'{key}' absent from the model. Available per-case Deterministics: "
                       f"{[v for v in post.data_vars if post[v].ndim == 3]}.")
    arr = post[key]                                  # dims (chain, draw, case)
    case_dim = arr.dims[-1]
    cases = list(arr.coords[case_dim].values) if labels is None else labels
    flat = arr.values.reshape(-1, arr.shape[-1])     # (samples, case)
    return bayes_meta_analysis(flat.mean(axis=0), flat.std(axis=0), labels=cases, **kwargs)


def meta_summary_rows(result, *, label="effect"):
    """Report-ready rows (list of dicts): 1 population row + 1 per case."""
    pop = result["population"]; het = result["heterogeneity"]
    rows = [{"Level": "POPULATION", "Case": f"(mu, {result['k']} cases)",
             f"{label} (median)": round(pop["median"], 2),
             "HDI95": f"[{pop['hdi_low']:.2f};{pop['hdi_high']:.2f}]",
             "pd": round(pop["pd"], 3),
             "tau (heterog.)": round(het["tau"]["median"], 2),
             "I2": round(het["I2"], 3) if np.isfinite(het["I2"]) else np.nan,
             "Prediction interval": f"[{het['prediction_interval'][0]:.2f};"
                                    f"{het['prediction_interval'][1]:.2f}]"}]
    for pc in result["per_case"]:
        s = pc["shrunk"]
        rows.append({"Level": "case", "Case": pc["case"],
                     f"{label} raw (stage 1)": round(pc["effect_raw"], 2),
                     "SE": round(pc["se"], 2),
                     f"{label} shrunk (stage 2)": round(s["median"], 2),
                     "HDI95": f"[{s['hdi_low']:.2f};{s['hdi_high']:.2f}]",
                     "pd": round(s["pd"], 3)})
    return rows


def write_meta_report(results, save_path, *, unit="pts", estimand_label="Total end-B effect"):
    """COMPLETE two-stage meta-analysis Excel report, IN THE SAME FORMAT as the partial
    hierarchical report: separate sheets **Population (group / meta)** and **Per case** (all outcomes
    stacked), + **Data & model**. This is NOT a comparison with the hierarchical model: it is the
    standalone rendering of the two-stage synthesis (stage 1 = independent ``pooling='none'`` fits;
    stage 2 = random-effects). HDI excluding 0 -> green; credible pd -> green/amber; Rhat/divergences -> red.
    ``results`` = dict ``{outcome: bayes_meta_analysis_result}``. Returns the written path."""
    import os
    import numpy as np
    import pandas as pd
    from functions.sced.glossary import write_stacked, hdi_str_fill, ascii_sanitize_df, style_cells

    def _pd_fill(v):
        try:
            a = float(v)
        except (TypeError, ValueError):
            return (False, None)
        return (True, "C6EFCE") if a >= 0.95 else ((True, "FFE699") if a >= 0.90 else (False, "F2F2F2"))

    def _rhat_fill(v):
        try:
            return (True, "FFC7CE") if float(v) > 1.01 else (False, None)
        except (TypeError, ValueError):
            return (False, None)

    def _div_fill(v):
        try:
            return (True, "FFC7CE") if float(v) > 0 else (False, None)
        except (TypeError, ValueError):
            return (False, None)

    def _i2_fill(v):                                    # heterogeneity: >0.75 red, >0.5 amber
        try:
            a = float(v)
        except (TypeError, ValueError):
            return (False, None)
        return (True, "FFC7CE") if a > 0.75 else ((False, "FFE699") if a > 0.50 else (False, None))

    pop_rows, case_rows = [], []
    for oc, res in results.items():
        pop = res["population"]; het = res["heterogeneity"]
        pop_rows.append({"Outcome": oc,
                         f"{estimand_label} (mu, {unit})": round(pop["median"], 2),
                         "HDI95": f"[{pop['hdi_low']:.2f};{pop['hdi_high']:.2f}]",
                         "pd": round(pop["pd"], 3),
                         "tau (heterog.)": round(het["tau"]["median"], 2),
                         "I2": round(het["I2"], 3) if np.isfinite(het["I2"]) else np.nan,
                         "Prediction interval": f"[{het['prediction_interval'][0]:.2f};"
                                                f"{het['prediction_interval'][1]:.2f}]",
                         "k cases": res["k"], "Rhat": round(res["rhat_max"], 3), "diverg.": res["diverging"]})
        for pc in res["per_case"]:
            s = pc["shrunk"]
            case_rows.append({"Outcome": oc, "Case": pc["case"],
                              f"{estimand_label} (shrunk, {unit})": round(s["median"], 2),
                              "HDI95": f"[{s['hdi_low']:.2f};{s['hdi_high']:.2f}]",
                              "pd": round(s["pd"], 3),
                              f"raw stage 1 ({unit})": round(pc["effect_raw"], 2),
                              "SE": round(pc["se"], 2)})
    pop_df = pd.DataFrame(pop_rows); case_df = pd.DataFrame(case_rows)

    recap = pd.DataFrame({"Element": [
        "Design", "Synthesis", "Estimand", "Stage 1", "Stage 2", "Heterogeneity",
        "Outcomes", "Orientation", "References"], "Value": [
        "SCED - two-stage Bayesian meta-analysis (group + per case)",
        "Population (mu) + per-case SHRUNK a posteriori (not a comparison with the hierarchical model)",
        f"{estimand_label} = b2 + b3*T_B (in {unit})",
        "INDEPENDENT per-case fits (pooling='none') -> effect_i + SE_i",
        "random-effects non-centered: mu ~ N ; tau ~ HalfNormal ; theta_i = mu + tau*z_i ; eff_i ~ N(theta_i, SE_i)",
        "tau (between-case SD) + I2 (Higgins-Thompson) + prediction interval",
        ", ".join(str(o) for o in results), "improvement: positive = improvement",
        "Van den Noortgate & Onghena 2003/2008 ; Moeyaert 2017 ; Burke-Ensor-Riley 2017 ; Higgins-Thompson 2002"]})

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    S = ascii_sanitize_df
    pop_rules = {"pd": _pd_fill, "HDI95": hdi_str_fill, "I2": _i2_fill,
                 "Rhat": _rhat_fill, "diverg.": _div_fill}
    case_rules = {"pd": _pd_fill, "HDI95": hdi_str_fill}
    with pd.ExcelWriter(save_path, engine="openpyxl") as w:
        write_stacked(w, "Data & model", [("TWO-STAGE META-ANALYSIS", recap)])
        S(pop_df).to_excel(w, sheet_name="Population (meta)", index=False)
        style_cells(w.sheets["Population (meta)"], pop_df, pop_rules)
        S(case_df).to_excel(w, sheet_name="Per case", index=False)
        style_cells(w.sheets["Per case"], case_df, case_rules)
    return save_path
