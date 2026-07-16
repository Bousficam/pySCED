"""
SCED - Bayesian estimation of the condition effect (single unit & multilevel)
=============================================================================

A **Bayesian complement** to the design-based randomization tests: instead of a
p-value it returns the POSTERIOR of the condition effect - a credible interval, the
posterior probability that the effect is beneficial, and a **Bayes factor (BF10)**
quantifying the evidence *for vs against* an effect. A non-significant randomization
test is NOT evidence of no effect, whereas a Bayes factor can support the null (BF10<1).

The model is a generalized linear (mixed) model with an **additive linear predictor**,
built brick by brick so it mirrors the options of the alternating pipeline:

    g(E[y]) = a + sum_c delta_c*1[cond=c]  (+ time trend)  (+ covariates)  (+ random effects)

  - **Family / link** (``outcome_type``): ``continuous`` -> Normal (identity, outcome
    standardized so delta is in Cohen's-d units); ``robust`` -> Student-t (heavy-tailed, resists
    outliers); ``binary`` -> Bernoulli (logit, delta = log-odds -> odds ratio); ``count`` ->
    Poisson (log, delta = log-rate -> rate ratio).
  - **Priors** (``effect_prior``, ``intercept_prior``): the effect prior is configurable -
    ``("cauchy", scale)`` (JZS default, scale 0.707 "medium") or ``("normal", mean, sd)``
    for an INFORMATIVE prior (e.g. a previous study's posterior -> sequential updating).
  - **Hierarchy** (``group_cols``): one random intercept per grouping column (e.g.
    ``["patient"]``, or nested ``["site", "patient"]``); ``random_slope=True`` adds a random
    treatment effect across the first grouping level (the effect varies by patient).
  - **Nuisance**: ``detrend`` (none/linear/log) and ``covariate_cols`` (ANCOVA-style),
    as in the alternating pipeline.

The reported effect is the contrast ``target - reference``. Its **Bayes factor** for
``delta_target != 0`` is the **Savage-Dickey density ratio** (Wagenmakers, Lodewyckx, Kuriyal &
Grasman 2010): ``BF01 = p(delta=0|data)/p(delta=0|prior)`` - posterior density at 0 (Gaussian KDE)
over the analytic prior density at 0.

Estimation by MCMC (PyMC NUTS); convergence (R-hat, ESS) is checked and surfaced. This
**complements - does not replace - ** the randomization test (primary inference; Edgington &
Onghena 2007). Bayesian SCED with weakly-informative priors: Rouder et al. (2009); Baek,
Beretvas, Van den Noortgate & Ferron (2020); Natesan Batley & Hedges (2021).
"""
import numpy as np
import pandas as pd

from functions.sced.core import validate_sced, _nuisance_basis


def _bf_label(bf10):
    """Jeffreys evidence categories as refined by Lee & Wagenmakers (2013).

    References: Jeffreys 1961; Lee & Wagenmakers 2013 (Bayes-factor evidence scale).
    R equivalent: effectsize::interpret_bf - potential equivalent, to test.
    """
    b = bf10 if bf10 >= 1 else 1.0 / bf10
    side = "for the effect" if bf10 >= 1 else "for the absence of effect (H0)"
    if b < 3:        strength = "anecdotal"
    elif b < 10:     strength = "moderate"
    elif b < 30:     strength = "strong"
    elif b < 100:    strength = "very strong"
    else:            strength = "extreme"
    return f"{strength} evidence {side}"


def _fmt_bf(bf10):
    """Readable Bayes-factor display: scientific notation for extremes, else 3 decimals."""
    if not np.isfinite(bf10):
        return "inf"
    if bf10 >= 1000 or (0 < bf10 < 0.001):
        return f"{bf10:.2e}"
    return round(float(bf10), 3)


def _resolve_effect_prior(outcome_type, prior_mean, prior_sd, prior_scale, y_sd):
    """Turn the user-facing prior into an INTERNAL-scale spec + a readable description.

    The user gives, for an INFORMATIVE prior, only ``prior_mean`` / ``prior_sd`` **in the
    natural units of the effect**, and the values are translated to the model's parameter
    scale by family:
      - continuous / robust : raw outcome units -> standardized (/ outcome SD), since the
        model works on the z-scored outcome (effect = Cohen's d). -> Normal(m/sd, s/sd).
      - binary / count : ``prior_mean``/``prior_sd`` describe the **odds / rate ratio**
        (the reported effect size). Translate to the log-odds / log-rate scale the model
        uses: mean = log(ratio); SD via the delta method SD(log r) ~ SD(r)/r. -> Normal(...).
    Without ``prior_mean``/``prior_sd`` a weakly-informative default is used: JZS Cauchy(0,
    0.707) on the standardized effect (continuous/robust), or Normal(0, 1.5) on the
    logit/log scale (binary/count); ``prior_scale`` overrides that default scale.

    Returns ``(spec, description)`` where ``spec`` is ``('normal', mu, sigma)`` or
    ``('cauchy', scale)`` on the internal scale (used both for the PyMC prior and the
    Savage-Dickey density at 0).

    References: Rouder et al. 2009 (JZS Cauchy default prior); delta method for the
    log-scale SD translation of an odds/rate ratio.
    R equivalent: BayesFactor (JZS medium prior, scale 0.707) - potential equivalent, to test.
    """
    informative = prior_mean is not None and prior_sd is not None
    if (prior_mean is None) ^ (prior_sd is None):
        raise ValueError("Informative prior: provide prior_mean AND prior_sd (or neither).")
    if informative:
        if float(prior_sd) <= 0:
            raise ValueError("prior_sd must be > 0.")
        if outcome_type in ("continuous", "robust"):
            mu, sg = float(prior_mean) / y_sd, float(prior_sd) / y_sd
            desc = f"informative Normal(mean={prior_mean}, sd={prior_sd} raw units -> d)"
        else:
            if float(prior_mean) <= 0:
                raise ValueError("For binary/count, prior_mean is an odds/rate ratio > 0.")
            mu, sg = float(np.log(prior_mean)), float(prior_sd) / float(prior_mean)
            ratio = "odds ratio" if outcome_type == "binary" else "rate ratio"
            desc = f"informative (on log {ratio}; mean={prior_mean}, sd={prior_sd})"
        return ("normal", mu, sg), desc
    if outcome_type in ("continuous", "robust"):
        scale = prior_scale if prior_scale is not None else 0.707
        return ("cauchy", float(scale)), f"non-informative Cauchy(0, {scale}) [JZS]"
    scale = prior_scale if prior_scale is not None else 1.5
    return ("normal", 0.0, float(scale)), f"non-informative Normal(0, {scale}) [logit/log scale]"


def _build_effect_dist(pm, name, spec, *, shape=None, dims=None):
    """Instantiate the effect prior from an internal spec: ('cauchy', scale) or
    ('normal', mean, sd). ``dims`` names the contrast dimension so every ArviZ plot
    labels the components by their contrast (e.g. 'B - A', 'C - A')."""
    kw = {"dims": dims} if dims is not None else {"shape": shape}
    kind = spec[0].lower()
    if kind == "cauchy":
        return pm.Cauchy(name, 0.0, float(spec[1]), **kw)
    return pm.Normal(name, float(spec[1]), float(spec[2]), **kw)


def _prior_density_at_0(spec):
    """Analytic prior density at delta=0 for the Savage-Dickey Bayes factor.

    References: Wagenmakers et al. 2010 (Savage-Dickey density ratio).
    R equivalent: no direct R equivalent (prior ordinate; cf. BayesFactor).
    """
    from scipy.stats import cauchy, norm
    if spec[0].lower() == "cauchy":
        return float(cauchy.pdf(0.0, loc=0.0, scale=float(spec[1])))
    return float(norm.pdf(0.0, loc=float(spec[1]), scale=float(spec[2])))


def _hdi(samples, prob):
    """Highest-density interval of ``samples`` at mass ``prob`` (narrowest interval that
    contains that fraction - the Bayesian default interval; Kruschke 2015).

    References: Kruschke 2015 (highest-density interval).
    R equivalent: bayestestR::hdi; HDInterval::hdi.
    """
    s = np.sort(np.asarray(samples, float))
    n = len(s)
    k = max(1, int(np.floor(prob * n)))
    if k >= n:
        return float(s[0]), float(s[-1])
    widths = s[k:] - s[:n - k]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k])


def _contrast_effect(name, draws_c, sgn, outcome_type, y_sd, prior_spec, hdi_prob, rope):
    """Posterior summary of ONE contrast (a non-reference condition vs the reference):
    family-appropriate effect size + **HDI** (at ``hdi_prob``), P(beneficial), the
    Savage-Dickey Bayes factor, and - if ``rope`` is given - the **ROPE** decision
    (Kruschke 2018: HDI vs a Region Of Practical Equivalence). ``rope`` is expressed in
    the EFFECT-SIZE units (d for continuous/robust ; odds/rate ratio for binary/count).

    References: Kruschke 2018 (ROPE); Makowski et al. 2019 (probability of direction);
    Wagenmakers et al. 2010 (Savage-Dickey Bayes factor).
    R equivalent: bayestestR::hdi / p_direction / rope; BayesFactor for the Savage-Dickey BF.
    """
    from scipy.stats import gaussian_kde
    mean = float(draws_c.mean())
    lo, hi = _hdi(draws_c, hdi_prob)                    # HDI on the model (delta) scale
    p_ben = float((sgn * draws_c > 0).mean())
    try:
        bf10 = _prior_density_at_0(prior_spec) / float(gaussian_kde(draws_c)(0.0)[0])
    except Exception:
        bf10 = np.nan
    if outcome_type in ("continuous", "robust"):
        es_name, es = "Cohen's d", round(mean, 4)
        es_ci = (round(lo, 4), round(hi, 4))
        raw_name, raw_val = "Effect (raw units)", round(mean * y_sd, 4)
        raw_ci = (round(lo * y_sd, 4), round(hi * y_sd, 4))
        nullval, rope_delta = "0", (rope if rope is not None else None)
    else:
        ratio = "Odds ratio (exp delta)" if outcome_type == "binary" else "Rate ratio (exp delta)"
        es_name, es = ratio, round(float(np.exp(mean)), 4)
        es_ci = (round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4))
        raw_name = "Effect (log-odds)" if outcome_type == "binary" else "Effect (log-rate)"
        raw_val, raw_ci = round(mean, 4), (round(lo, 4), round(hi, 4))
        nullval = "1"
        rope_delta = ((float(np.log(rope[0])), float(np.log(rope[1])))
                      if rope is not None else None)

    rope_res = None
    if rope_delta is not None:
        rlo, rhi = sorted(rope_delta)
        pct_in = float(((draws_c >= rlo) & (draws_c <= rhi)).mean())
        if lo >= rlo and hi <= rhi:
            decision = "HDI within ROPE -> practical equivalence to null (negligible effect)"
        elif hi < rlo or lo > rhi:
            decision = "HDI outside ROPE -> practically significant effect"
        else:
            decision = "HDI overlaps ROPE -> undecided (more data needed)"
        rope_res = {"rope": tuple(rope), "pct_in_rope": round(pct_in, 4), "decision": decision}

    return {"name": name, "mean": mean, "es_name": es_name, "es": es, "es_ci": es_ci,
            "raw_name": raw_name, "raw_val": raw_val, "raw_ci": raw_ci,
            "p_benefit": round(p_ben, 4), "bf10_num": bf10, "bf10": _fmt_bf(bf10),
            "nullval": nullval, "excludes0": bool(not (lo <= 0 <= hi)), "rope": rope_res}


def bayesian_condition_effect(df, *, session_col, condition_col, outcome_col,
                              conditions=None, reference=None, target=None,
                              outcome_type="continuous", improvement="increase",
                              time_covariate="none", covariate_cols=None,
                              group_cols=None, unit_col=None, random_slope=False,
                              prior_mean=None, prior_sd=None, prior_scale=None,
                              intercept_prior=("normal", 0.0, 10.0),
                              hdi_prob=0.95, rope=None,
                              draws=2000, tune=1000, chains=4,
                              target_accept=0.9, random_state=0, return_idata=False):
    """
    Bayesian estimate of a condition effect for alternating / N-of-1 and multilevel SCED.

    Returns ``(summary, sheet)``. Every non-reference condition is modelled and **all
    contrasts vs the reference are reported** (``summary["contrasts"]`` + one sheet block
    each). The scalar headline keys (``effect_size``, ``p_benefit``, ``bf10`` ...) describe
    the ``target`` contrast - by default the **last** condition; set ``target`` to focus
    on a specific one. Each contrast carries an interpretable effect size (Cohen's d /
    odds ratio / rate ratio), a 95% credible interval, ``p_benefit`` = P(effect
    beneficial), and a Savage-Dickey ``bf10``; convergence (max R-hat, min ESS) is global.

    Key options (mirroring the alternating pipeline):
      - ``outcome_type`` : ``continuous`` | ``robust`` (Student-t) | ``binary`` | ``count``.
      - **Priors** : leave them out for a weakly-informative default (JZS Cauchy on the
        standardized effect; ``prior_scale`` tunes it). For an INFORMATIVE prior give only
        ``prior_mean`` / ``prior_sd`` **in natural units** - raw outcome units for
        continuous/robust (translated to the Cohen's-d scale / outcome SD), or the
        odds/rate ratio for binary/count (translated to the log scale) - and the right
        distribution is built automatically.
      - ``group_cols`` : list of grouping columns -> one random intercept each (e.g.
        ``["patient"]`` or nested ``["site", "patient"]``). ``unit_col`` is a single-level
        alias. ``random_slope=True`` adds a random treatment effect over the first level.
      - ``time_covariate`` (none/linear/log) and ``covariate_cols`` : nuisance adjustment.
      - ``reference`` / ``target`` : the reference baseline / the focused contrast
        (defaults: first condition / last condition, all contrasts reported either way).

    References: Rouder et al. 2009; Baek et al. 2020; Natesan Batley & Hedges 2021
    (Bayesian SCED); Wagenmakers et al. 2010 (Savage-Dickey Bayes factor).
    R equivalent: brms (condition + time + random effects) + bayestestR (HDI / pd / ROPE);
    BayesFactor for the Savage-Dickey BF.
    """
    import logging
    import warnings as _warnings
    import pymc as pm
    import arviz as az
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    if outcome_type not in ("continuous", "robust", "binary", "count"):
        raise ValueError("outcome_type must be in {continuous, robust, binary, count}.")
    group_cols = list(group_cols) if group_cols else ([unit_col] if unit_col else [])

    unit_for_validate = group_cols[0] if group_cols else None
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, unit_col=unit_for_validate)
    if conditions is not None:
        clean = clean[clean[condition_col].isin(conditions)]
        conds = list(conditions)
    else:
        conds = list(report["Phases/conditions"])
    if len(conds) < 2:
        raise ValueError("At least 2 conditions are required.")
    ref = reference if reference is not None else conds[0]
    tgt = target if target is not None else conds[-1]      # headline contrast (all are reported)
    if ref not in conds or tgt not in conds:
        raise ValueError(f"reference/target not among the present conditions {conds}.")
    if ref == tgt:
        raise ValueError("reference and target must differ.")

    y = pd.to_numeric(clean[outcome_col], errors="coerce").to_numpy(float)
    cond = clean[condition_col].to_numpy()
    sess = pd.to_numeric(clean[session_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(sess)
    cov = None
    if covariate_cols:
        cov = clean[list(covariate_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = ok & np.isfinite(cov).all(axis=1)
    g_raw = [clean[gc].to_numpy() for gc in group_cols]
    y, cond, sess = y[ok], cond[ok], sess[ok]
    if cov is not None:
        cov = cov[ok]
    g_raw = [gv[ok] for gv in g_raw]
    if len(y) < 3:
        raise ValueError("Too few valid observations (< 3).")

    # family-specific outcome handling
    if outcome_type == "binary":
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("binary outcome: values 0/1 expected.")
        y_model, y_sd = y, 1.0
    elif outcome_type == "count":
        if np.any(y < 0) or np.any(y != np.round(y)):
            raise ValueError("count outcome: integers >= 0 expected.")
        y_model, y_sd = y, 1.0
    else:                                                   # continuous / robust -> standardize
        y_sd = float(y.std(ddof=1))
        if y_sd == 0:
            raise ValueError("Zero-variance outcome - effect not identifiable.")
        y_model = (y - y.mean()) / y_sd

    # resolve the prior on the model's internal scale now that y_sd is known
    eff_spec, prior_desc = _resolve_effect_prior(outcome_type, prior_mean, prior_sd,
                                                 prior_scale, y_sd)

    non_ref = [c for c in conds if c != ref]
    tgt_j = non_ref.index(tgt)
    X = np.column_stack([(cond == c).astype(float) for c in non_ref])

    Z = None
    if time_covariate != "none":
        Zb = _nuisance_basis(sess, time_covariate)[:, 1:]
        Z = (Zb - Zb.mean(0)) / (Zb.std(0) + 1e-12)
    if cov is not None:
        cov = (cov - cov.mean(0)) / (cov.std(0) + 1e-12)

    groups = []                                            # (name, idx, n_levels)
    for gc, gv in zip(group_cols, g_raw):
        uniq = pd.unique(gv)
        gmap = {u: i for i, u in enumerate(uniq)}
        groups.append((gc, np.array([gmap[v] for v in gv]), len(uniq)))

    contrast_labels = [f"{c} - {ref}" for c in non_ref]
    with pm.Model(coords={"contrast": contrast_labels}):
        ip = intercept_prior
        a = pm.Normal("a", float(ip[1]), float(ip[2]))
        delta = _build_effect_dist(pm, "delta", eff_spec, dims="contrast")
        mu = a + pm.math.dot(X, delta)
        if Z is not None:
            gamma = pm.Normal("gamma", 0.0, 10.0, shape=Z.shape[1])
            mu = mu + pm.math.dot(Z, gamma)
        if cov is not None:
            beta_c = pm.Normal("beta_cov", 0.0, 10.0, shape=cov.shape[1])
            mu = mu + pm.math.dot(cov, beta_c)
        for gc, idx, n_lev in groups:                      # random intercept per level
            tau_g = pm.HalfNormal(f"tau_{gc}", 1.0)
            u_g = pm.Normal(f"u_{gc}", 0.0, tau_g, shape=n_lev)
            mu = mu + u_g[idx]
        if random_slope and groups:                        # random treatment effect (1st level)
            gc0, idx0, n0 = groups[0]
            tau_b = pm.HalfNormal("tau_slope", 1.0)
            b = pm.Normal("b_slope", 0.0, tau_b, shape=n0)
            mu = mu + b[idx0] * X[:, tgt_j]

        if outcome_type == "continuous":
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_model)
        elif outcome_type == "robust":
            sigma = pm.HalfNormal("sigma", 1.0)
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
            pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_model)
        elif outcome_type == "binary":
            pm.Bernoulli("obs", logit_p=mu, observed=y_model)
        else:                                              # count
            pm.Poisson("obs", mu=pm.math.exp(mu), observed=y_model)

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                              target_accept=target_accept, random_seed=random_state,
                              progressbar=False, idata_kwargs={"log_likelihood": False})
            if return_idata:                          # posterior predictive for the PPC plot
                idata.extend(pm.sample_posterior_predictive(
                    idata, progressbar=False, random_seed=random_state))

    draws_delta = idata.posterior["delta"].values.reshape(-1, X.shape[1])
    sgn = 1.0 if improvement == "increase" else -1.0
    contrasts = [_contrast_effect(f"{c} - {ref}", draws_delta[:, j], sgn, outcome_type,
                                  y_sd, eff_spec, hdi_prob, rope)
                 for j, c in enumerate(non_ref)]
    head = contrasts[tgt_j]
    pct = int(round(hdi_prob * 100))

    rhat = float(np.nanmax(az.rhat(idata, var_names=["delta"])["delta"].values))
    ess = float(np.nanmin(az.ess(idata, var_names=["delta"])["delta"].values))
    converged = np.isfinite(rhat) and rhat < 1.01 and ess > 400

    nu_stats = None
    if outcome_type == "robust":
        nu_draws = idata.posterior["nu"].values.reshape(-1)
        nu_lo, nu_hi = (float(v) for v in np.percentile(nu_draws, [2.5, 97.5]))
        nu_stats = {"mean": round(float(nu_draws.mean()), 2),
                    "hdi": tuple(round(v, 2) for v in _hdi(nu_draws, hdi_prob))}

    summary = {
        "outcome_type": outcome_type, "contrast": head["name"],
        "design": ("single unit" if not groups else
                   "multilevel: " + " / ".join(f"{g[0]}({g[2]})" for g in groups)
                   + (" + random slope" if (random_slope and groups) else "")),
        "n_obs": int(len(y)), "levels": {g[0]: g[2] for g in groups},
        "effect_posterior_mean": round(head["mean"], 4),
        "effect_size_name": head["es_name"], "effect_size": head["es"],
        "effect_size_hdi": head["es_ci"], "hdi_prob": hdi_prob,
        "p_benefit": head["p_benefit"], "bf10": head["bf10"], "rope": head["rope"],
        "contrasts": [{"contrast": c["name"], "effect_size_name": c["es_name"],
                       "effect_size": c["es"], "hdi": c["es_ci"], "p_benefit": c["p_benefit"],
                       "bf10": c["bf10"], "rope": c["rope"]} for c in contrasts],
        "effect_prior": prior_desc, "time_covariate": time_covariate,
        "covariates": list(covariate_cols) if covariate_cols else None,
        "rhat_max": round(rhat, 4), "ess_min": round(ess, 1), "converged": bool(converged),
    }
    if nu_stats is not None:
        summary["nu_robust"] = nu_stats["mean"]
        summary["nu_robust_hdi"] = nu_stats["hdi"]

    es_expl = {"continuous": "Posterior mean of the standardized effect (standard deviations).",
               "robust": "Standardized effect (Student-t likelihood, robust to outliers).",
               "binary": "Posterior odds ratio (exp of the logit effect).",
               "count": "Posterior rate ratio (exp of the log effect)."}[outcome_type]
    h = head
    bf_txt = (_bf_label(h["bf10_num"]) if np.isfinite(h["bf10_num"]) and h["bf10_num"] > 0
              else "decisive evidence for the effect")
    rows = [
        {"Quantity": f"Effect {h['name']} - {h['es_name']}", "Value": h["es"],
         "Explanation": es_expl,
         "Interpretation": (f"HDI {pct}% = [{h['es_ci'][0]}, {h['es_ci'][1]}]; "
                            + (f"excludes {h['nullval']} -> credible effect." if h["excludes0"]
                               else f"includes {h['nullval']}."))},
        {"Quantity": h["raw_name"], "Value": h["raw_val"],
         "Explanation": "Same effect on the natural scale of the parameter.",
         "Interpretation": f"HDI {pct}% = [{h['raw_ci'][0]}, {h['raw_ci'][1]}]."},
        {"Quantity": "P(beneficial effect)", "Value": h["p_benefit"],
         "Explanation": f"Posterior probability that the effect goes in the '{improvement}' direction.",
         "Interpretation": ("Very likely." if h["p_benefit"] > 0.95 else
                            "Likely." if h["p_benefit"] > 0.8 else "Uncertain.")},
        {"Quantity": "Bayes factor BF10", "Value": h["bf10"],
         "Explanation": "Evidence for an effect vs its absence (Savage-Dickey; Rouder 2009, "
                        "Wagenmakers 2010). BF10>1 favors the effect, <1 favors H0.",
         "Interpretation": bf_txt + " (Lee & Wagenmakers 2013 scale)."},
    ]
    if h["rope"] is not None:
        rows.append({"Quantity": f"ROPE {h['rope']['rope']}",
                     "Value": f"{round(h['rope']['pct_in_rope'] * 100, 1)}% of the posterior inside",
                     "Explanation": "Region of practical equivalence to null: HDI vs ROPE decision "
                                    "(Kruschke 2018).",
                     "Interpretation": h["rope"]["decision"]})
    if len(contrasts) > 1:                              # >=3 conditions -> list every contrast
        for c in contrasts:
            rope_txt = ("; " + c["rope"]["decision"].split(" -> ")[-1]) if c["rope"] else ""
            rows.append({"Quantity": f"Contrast {c['name']} - {c['es_name']}", "Value": c["es"],
                         "Explanation": "Effect of this condition vs the reference (all reported).",
                         "Interpretation": (f"HDI {pct}% = [{c['es_ci'][0]}, {c['es_ci'][1]}]; "
                                            f"P(benef)={c['p_benefit']}; BF10={c['bf10']}" + rope_txt)})
    rows.append({"Quantity": "Model", "Value": summary["design"],
                 "Explanation": f"Family = {outcome_type}; effect prior = {prior_desc}.",
                 "Interpretation": ("Hierarchical (random effects)." if groups else "Single case.")})
    if nu_stats is not None:
        nu_m = nu_stats["mean"]
        rows.append({
            "Quantity": "nu (Student-t degrees of freedom)", "Value": nu_m,
            "Explanation": "Tail thickness, estimated from the data: small nu = heavy "
                           "tails (outliers dampened); large nu = ~ Gaussian.",
            "Interpretation": (f"HDI = [{nu_stats['hdi'][0]}, {nu_stats['hdi'][1]}]; "
                               + ("low nu -> heavy tails DETECTED, robustness active."
                                  if nu_m < 7 else
                                  "moderate nu -> slightly heavy tails." if nu_m < 30 else
                                  "high nu -> no marked outlier, ~ Gaussian model."))})
    rows.append({"Quantity": "Convergence (Rhat max / ESS min)",
                 "Value": f"{summary['rhat_max']} / {summary['ess_min']}",
                 "Explanation": "MCMC diagnostics: Rhat ~ 1 and high ESS = reliable chains.",
                 "Interpretation": ("OK." if converged else
                                    "WARNING: not converged (Rhat >= 1.01 or low ESS) - increase draws/tune.")})
    sheet = pd.DataFrame(rows, columns=["Quantity", "Value", "Explanation", "Interpretation"])
    if return_idata:
        return summary, sheet, idata
    return summary, sheet


_DIAG_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 110, "savefig.bbox": "tight",
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 7,
    "axes.titlecolor": "#22313f",
    "axes.labelsize": 10.5, "axes.labelcolor": "#22313f", "text.color": "#22313f",
    "axes.edgecolor": "#9aa7b0", "axes.linewidth": 0.9,
    "xtick.color": "#46606e", "ytick.color": "#46606e",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": "#cdd6db", "grid.alpha": 0.55, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
}
_ACCENT = "#1f6f8b"        # teal - point estimate / posterior
_ACCENT2 = "#e07b39"       # warm - observed / mean overlays


def plot_bayesian_diagnostics(idata, *, save_path=".", var_names=("delta",), rope=None,
                              hdi_prob=0.95, prefix="bayesian", title=None, dpi=200,
                              dashboard=True, fmt="png"):
    """
    **Publication-quality diagnostic & model-quality plots** for the fitted Bayesian model,
    following the recommended Bayesian workflow (Gabry, Simpson, Vehtari, Betancourt &
    Gelman 2019, *JRSS-A*; Vehtari, Gelman, Simpson, Carpenter & Burkner 2021, *Bayesian
    Analysis*).

    Pass the ``idata`` returned by :func:`bayesian_condition_effect` with
    ``return_idata=True``. Applies a clean article-style theme and saves one high-DPI PNG
    per diagnostic in ``save_path`` (plus a combined ``dashboard``), returning a
    ``{name: path}`` dict:

      - ``trace``     : chains + marginal posteriors - eyeball mixing/stationarity.
      - ``rank``      : rank plots - the modern convergence check; bars ~uniform across
                        chains (better than trace; Vehtari 2021).
      - ``forest``    : posterior intervals with **R-hat and ESS** annotated per parameter.
      - ``posterior`` : marginal posterior with the **HDI**, a reference line at 0 and the
                        **ROPE** band if supplied - the inference plot (Kruschke).
      - ``energy``    : NUTS energy / BFMI - flags sampler pathologies.
      - ``ppc``       : **posterior predictive check** - observed vs replicated data (only
                        if the posterior predictive was drawn, ``return_idata=True``).
      - ``dashboard`` : all of the above tiled into one labelled figure (``dashboard=True``).

    ``rope`` is a ``(lo, hi)`` tuple on the delta / model scale (drawn on the posterior plot);
    ``title`` sets the dashboard's super-title; ``dpi`` the export resolution.

    References: Gabry et al. 2019 (Bayesian workflow visualization); Vehtari et al. 2021
    (rank plots, R-hat / ESS).
    R equivalent: bayesplot (mcmc_trace / mcmc_rank_bars / ppc_dens_overlay); loo for PSIS.
    """
    import os
    import warnings as _warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import arviz as az

    os.makedirs(save_path, exist_ok=True)
    paths = {}
    # number of contrast components (>=3 conditions -> a vector delta = one panel each)
    _da = idata.posterior[list(var_names)[0]]
    k = int(np.prod([s for d, s in zip(_da.dims, _da.shape)
                     if d not in ("chain", "draw")])) or 1
    titles = {
        "trace": "Trace & posterior densities",
        "rank": "Rank plot - chain convergence",
        "forest": "Forest - intervals, Rhat and ESS",
        "posterior": "Effect posterior - HDI" + (" + ROPE" if rope is not None else "")
                     + " (ref. 0)",
        "energy": "Energy / BFMI - sampler health",
        "ppc": "Posterior predictive check - model adequacy",
    }
    rope_arg = ({list(var_names)[0]: [{"rope": tuple(rope)}]} if rope is not None else None)

    def _style_axes(fig):
        for ax in fig.get_axes():
            ax.title.set_color("#22313f")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    def _finish(name):
        fig = plt.gcf()
        _style_axes(fig)
        fig.suptitle(titles[name], fontsize=13.5, fontweight="bold", color="#22313f", y=1.02)
        p = os.path.join(save_path, f"{prefix}_{name}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close("all")
        paths[name] = p

    panels = [
        ("trace", lambda: az.plot_trace(idata, var_names=var_names, compact=True,
                                        figsize=(9, 3.2))),       # compact -> stays 2 panels
        ("rank", lambda: az.plot_rank(idata, var_names=var_names, kind="bars",
                                      figsize=(max(9, 4.6 * k), 3.2))),
        ("forest", lambda: az.plot_forest(idata, var_names=var_names, r_hat=True, ess=True,
                                          hdi_prob=hdi_prob, combined=True, colors=_ACCENT,
                                          figsize=(8, max(3.0, 1.3 * k + 1.6)))),
        ("posterior", lambda: az.plot_posterior(idata, var_names=var_names, ref_val=0,
                                                rope=rope_arg, hdi_prob=hdi_prob,
                                                point_estimate="mean", color=_ACCENT,
                                                figsize=(max(4.5, 3.8 * k), 3.6))),
        ("energy", lambda: az.plot_energy(idata, fill_alpha=(0.75, 0.35),
                                          fill_color=(_ACCENT, _ACCENT2), figsize=(7, 3.6))),
    ]
    if "posterior_predictive" in idata.groups():
        panels.append(("ppc", lambda: az.plot_ppc(idata, num_pp_samples=80, mean=True,
                                                  figsize=(8, 3.6))))

    with plt.rc_context(_DIAG_RC):
        for name, fn in panels:
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    fn()
                _finish(name)
            except Exception:
                plt.close("all")

        if dashboard and paths:
            order = [n for n in ("posterior", "ppc", "forest", "rank", "energy", "trace")
                     if n in paths]
            ncols = 2 if len(order) <= 4 else 3
            nrows = int(np.ceil(len(order) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.2 * nrows),
                                     facecolor="white")
            axes = np.atleast_1d(axes).ravel()
            for i, name in enumerate(order):
                ax = axes[i]
                ax.imshow(mpimg.imread(paths[name]))
                ax.axis("off")
                ax.set_title(f"({chr(97 + i)})", loc="left", fontsize=13,
                             fontweight="bold", color="#22313f")
            for j in range(len(order), len(axes)):
                axes[j].axis("off")
            fig.suptitle(title or "Bayesian model diagnostics", fontsize=16,
                         fontweight="bold", color="#22313f")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            dp = os.path.join(save_path, f"{prefix}_dashboard.{fmt}")
            fig.savefig(dp, dpi=dpi, facecolor="white", bbox_inches="tight")
            plt.close("all")
            paths["dashboard"] = dp
    return paths


def report_sced_bayesian_condition(
        df, *, session_col="session", condition_col="condition", outcome_col="score",
        conditions=("A", "B"), reference=None, target=None, outcome_type="continuous",
        improvement="increase", prior_mean=None, prior_sd=None, prior_scale=None,
        group_cols=None, random_slope=False, time_covariate="none", covariate_cols=None,
        hdi_prob=0.95, rope=None, draws=2000, tune=1000, chains=4, random_state=0,
        save_path=None, output_dir=None, diagnostics=False, verbose=True):
    """Report-level Bayesian SCED pipeline for a CONDITION effect (alternating / N-of-1 /
    multilevel): calls ``bayesian_condition_effect`` (effect + HDI + BF10 + ROPE), writes
    the styled xlsx summary (+ provenance) if ``save_path``/``output_dir``, and - if
    ``diagnostics`` or ``output_dir`` - the MCMC diagnostics in ``<output>/diagnostics/``.
    The template only calls. Returns the summary dict.

    References: Rouder et al. 2009; Baek et al. 2020; Natesan Batley & Hedges 2021
    (Bayesian SCED condition effect).
    R equivalent: brms + bayestestR (report layer).
    """
    import os
    if output_dir is not None and save_path is None:        # per-OUTCOME layout: <oc>/Analyse/bayes/condition/
        from functions.sced.glossary import std_layout
        save_path = std_layout(output_dir, outcome_col, "bayes/condition")["analyse"]
    want_diag = bool(diagnostics or output_dir)
    out = bayesian_condition_effect(
        df, session_col=session_col, condition_col=condition_col, outcome_col=outcome_col,
        conditions=conditions, reference=reference, target=target, outcome_type=outcome_type,
        improvement=improvement, prior_mean=prior_mean, prior_sd=prior_sd, prior_scale=prior_scale,
        group_cols=group_cols, random_slope=random_slope, time_covariate=time_covariate,
        covariate_cols=covariate_cols, hdi_prob=hdi_prob, rope=rope,
        draws=draws, tune=tune, chains=chains, random_state=random_state, return_idata=want_diag)
    summ, sheet = out[0], out[1]
    if verbose:
        print("== BAYESIAN SCED (condition effect) ==")
        for k in ["design", "outcome_type", "contrast", "effect_size_name", "effect_size",
                  "effect_size_hdi", "p_benefit", "bf10", "rope", "effect_prior",
                  "rhat_max", "ess_min", "converged"]:
            print(f"  {k}: {summ.get(k)}")
        print("\n" + sheet.to_string(index=False))
        if not summ.get("converged", True):
            print("\nWARNING: chains not converged (Rhat >= 1.01 or low ESS) - increase DRAWS/TUNE.")

    if save_path:
        from functions.sced.glossary import (ascii_sanitize_df, write_stacked, provenance_df,
                                              interpretation_glossary)
        os.makedirs(save_path, exist_ok=True)
        outp = os.path.join(save_path, "bayes_condition_summary.xlsx")
        S = ascii_sanitize_df
        try:
            with pd.ExcelWriter(outp, engine="openpyxl") as w:
                S(sheet).to_excel(w, sheet_name="Condition effect", index=False)
                write_stacked(w, "Interpretation & provenance",
                              [("INTERPRETATION GUIDE", interpretation_glossary("bayes")),
                               ("PROVENANCE", provenance_df(seed=random_state))])
            if verbose:
                print(f"\nReport written: {outp}")
        except Exception as e:
            if verbose:
                print(f"\n[not written] {type(e).__name__}: {e}")

    if want_diag and len(out) > 2:                          # MCMC diagnostics co-located (not in plots/)
        ddir = os.path.join(save_path, "diagnostics") if save_path else "."
        plot_rope = rope if (rope is not None and outcome_type in ("continuous", "robust")) else None
        try:
            paths = plot_bayesian_diagnostics(out[2], save_path=ddir, rope=plot_rope, hdi_prob=hdi_prob)
            if verbose:
                print("\nMCMC diagnostics:", ", ".join(f"{k}" for k in paths) + f" -> {ddir}/")
        except Exception as e:
            if verbose:
                print(f"[diagnostics skipped] {type(e).__name__}: {e}")
    return summ
