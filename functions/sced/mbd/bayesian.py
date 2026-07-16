"""
SCED - Per-case Bayesian models: simple, BITS, BUCP (Natesan Batley et al. 2020)
================================================================================

Three Bayesian time-series models for ONE AB series (one patient x one outcome),
from simplest to most complete (Natesan Batley, Contractor & Caldas 2020,
doi:10.1002/jts.22614; Natesan & Hedges 2017):

  - ``simple`` - per-phase intercept, i.i.d. Gaussian noise (WITHOUT autocorrelation). The most
    basic: an A->B level jump + standardized effect size.
  - ``bits`` - *Bayesian Interrupted Time-Series*: per-phase intercept + explicitly modelled
    lag-1 autocorrelation rho. yhat_t = beta_phase; e_t = rho*e_{t-1} + eps; the first point
    uses the marginal standard deviation sigma/sqrt(1-rho^2). This is the reference model for
    short autocorrelated series (>= ~8 points/phase, expected d >= 3 for good estimation).
  - ``bucp`` - *Bayesian Unknown Change-Point*: BITS + unknown change point CP
    (integer in [4, T-3], categorical prior). The CP posterior quantifies immediacy
    (concentrated near the phase boundary = immediate effect; diffuse = uncertain/delayed effect).

Effect size: ``es = (beta_baseline - beta_treatment)/sigma`` (standardized difference of
intercepts; small-sample correction included - Natesan Batley 2020 p. 1149). Here oriented by
``improvement``: ``es`` > 0 = improvement. Inference by HDI + ROPE + pd (Kruschke 2013).

Priors (Natesan Batley 2020, Table Parameters): beta ~ Normal(mu, 1), mu ~ Normal(prior_mean, sqrt(20));
sigma ~ Uniform(0.1, sigma_max); rho ~ Uniform(-1, 1); CP ~ Categorical uniform on [4, T-3].

Estimation: PyMC NUTS (+ automatic Metropolis for the discrete CP of BUCP). R-hat / ESS
reported. Note: PyMC implementation of the tutorial's JAGS model (reference: github.com/prathiba-stat/BITS-BUCP).
"""
import logging

import numpy as np
import pandas as pd

from functions.sced.bayes.condition import _hdi


def _summarise(draws, prob=0.95):
    """Posterior summary of a draw array: median, mean, SD, HDI bounds and probability of
    direction. Input: draws + HDI mass. Output: dict."""
    d = np.asarray(draws, float)
    lo, hi = _hdi(d, prob)
    pd_val = float(max(np.mean(d > 0), np.mean(d < 0)))
    return {"median": float(np.median(d)), "mean": float(np.mean(d)), "sd": float(np.std(d)),
            "hdi_low": lo, "hdi_high": hi, "pd": pd_val}


def _rope_stats(draws, rope, prob=0.95):
    """ROPE indices (Makowski 2019 / bayestestR) + HDI vs ROPE decision (Kruschke 2018).
    ROPE = [-rope, +rope] (effect in original unit / MCID). Returns % inside the ROPE, P(>ROPE)
    (oriented utility), and the 3-way verdict: effect (HDI outside ROPE) / equivalence (HDI inside
    ROPE) / undecided (overlaps).

    References: Kruschke 2018; Makowski, Ben-Shachar & Ludecke 2019 (ROPE + HDI decision rule).
    R equivalent: bayestestR::rope / bayestestR::equivalence_test."""
    d = np.asarray(draws, float)
    lo, hi = _hdi(d, prob)
    if hi < -rope or lo > rope:
        dec = "effect (HDI outside ROPE)"                   # rejects H0 (effect practically non-zero)
    elif lo >= -rope and hi <= rope:
        dec = "equivalence (HDI inside ROPE)"               # accepts H0 (negligible effect)
    else:
        dec = "undecided (HDI overlaps ROPE)"
    return {"pct_in_rope": round(float(np.mean(np.abs(d) < rope)), 3),
            "p_gt_rope": round(float(np.mean(d > rope)), 3),
            "rope_decision": dec, "rope": round(float(rope), 4)}


def bayes_phase_model(y, *, start_index, model="bits", improvement="increase",
                      rope=0.5, trend=False, baseline_trend=False, prior_mean=None,
                      sigma_max=None, draws=2000, tune=2000, chains=4, seed=42,
                      progressbar=False, cache_dir=None, force_refit=False):
    """
    Fits an AB Bayesian model on ONE time-ordered series ``y``. ``start_index``
    = 0-based index of the first observation of phase B (treatment). ``model`` in
    {``simple``, ``bits``, ``bucp``}. ``rope`` = lower practical-equivalence bound on ``es``
    (P(es > rope) is reported). Returns a dict:
    ``{model, n, n_a, n_b, es, level_change, rho, cp (bucp), p_es_gt_rope, rhat_max, ess_min,
       diverging, ...}`` where ``es``/``level_change``/``rho``/``cp`` are summary sub-dicts.

    References: Natesan Batley, Contractor & Caldas 2020; Natesan & Hedges 2017; Kruschke 2018
    (per-case Bayesian simple / BITS / BUCP models).
    R equivalent: brms (simple / bits: brm(y ~ phase (+ time), autocor=~ar(...))); BUCP has no
    turnkey R equivalent (original JAGS/Stan; a brms change-point form - potential equivalent, to test).
    """
    import pymc as pm
    import pytensor.tensor as pt
    logging.getLogger("pymc").setLevel(logging.ERROR)

    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    ok = np.isfinite(y)
    y = y[ok]
    T = len(y)
    if T < 6:
        raise ValueError(f"Series too short (T={T}) for a Bayesian model (>=6, ideally >=8/phase).")
    sign = 1.0 if improvement == "increase" else -1.0
    pm_mean = float(np.mean(y)) if prior_mean is None else float(prior_mean)
    s_max = float(max(5.0, 3.0 * np.std(y))) if sigma_max is None else float(sigma_max)
    phase_known = np.zeros(T)
    phase_known[start_index:] = 1.0            # 0 = baseline (A), 1 = treatment (B)

    # --- baseline trend test (Tarlow): DESCRIPTIVE DIAGNOSTIC reported (no auto selection) ---
    from functions.sced.core import tau_u
    ba_tau = ba_p = np.nan
    try:
        _bt = tau_u(y[:start_index], y[start_index:], improvement=improvement)
        ba_tau, ba_p = _bt.get("baseline_trend_tau"), _bt.get("baseline_trend_p")
    except Exception:
        pass
    include_b1 = (baseline_trend is True)   # baseline-slope inclusion = PRE-SPECIFIED (True/False)
    time_c_np = (np.arange(T) - (T - 1) / 2.0).astype(float)   # centered session (trend)

    with pm.Model() as m:
        mu_h = pm.Normal("mu_h", mu=pm_mean, sigma=np.sqrt(20.0), shape=2)
        beta = pm.Normal("beta", mu=mu_h, sigma=1.0, shape=2)   # [intercept A, intercept B]
        sigma = pm.Uniform("sigma", 0.1, s_max)

        if model == "bucp":                    # unknown change point (integer)
            lo, hi = 3, T - 4                   # 0-based: >=4 in A (0..3) and >=3 in B
            if hi < lo:
                raise ValueError(f"Series too short (T={T}) for BUCP (needs >=7).")
            ncp = hi - lo + 1
            cp = pm.Categorical("cp_idx", p=np.ones(ncp) / ncp)  # 0..ncp-1
            cp_pos = pm.Deterministic("cp", cp + lo)             # index of the LAST A point
            idx = pt.arange(T)
            phase = pt.cast(idx > cp_pos, "float64")
            first_B = cp_pos + 1.0
            t_since = pt.maximum(0.0, idx - first_B)             # cumulative dose in B (0,1,2,...)
            dur = T - 1.0 - first_B                              # duration of B (number of B - 1)
        else:
            phase = phase_known
            t_since = np.maximum(0.0, np.arange(T) - start_index).astype(float)
            dur = float(T - 1 - start_index)

        yhat = beta[0] * (1 - phase) + beta[1] * phase
        if include_b1:                         # baseline trend (spontaneous recovery) -> net b2/b3
            b1 = pm.Normal("b1", mu=0.0, sigma=1.0)
            time_c = (pt.arange(T) - (T - 1) / 2.0) if model == "bucp" else time_c_np
            yhat = yhat + b1 * time_c
        if trend:                              # CUMULATIVE EFFECT: slope in B (derived from the number of B)
            b3 = pm.Normal("b3", mu=0.0, sigma=1.0)
            yhat = yhat + b3 * t_since
        if model == "simple":                  # i.i.d. noise (no autocorrelation)
            rho_det = pm.Deterministic("rho", pt.zeros(()))
            pm.Normal("y_obs", mu=yhat, sigma=sigma, observed=y)
        else:                                  # BITS / BUCP: lag-1 AR1 on the residuals
            rho = pm.Uniform("rho", -1.0, 1.0)
            resid_prev = y[:-1] - yhat[:-1]
            mu_t = yhat[1:] + rho * resid_prev
            sd0 = sigma / pt.sqrt(1.0 - rho ** 2)
            pm.Normal("y0", mu=yhat[0], sigma=sd0, observed=y[0])
            pm.Normal("yt", mu=mu_t, sigma=sigma, observed=y[1:])

        # standardized effect size, oriented (es>0 = improvement)
        pm.Deterministic("es", sign * (beta[1] - beta[0]) / sigma)
        pm.Deterministic("level_change", sign * (beta[1] - beta[0]))
        if include_b1:
            pm.Deterministic("baseline_trend_b1", sign * b1)   # modelled baseline slope
        if trend:                              # end-of-phase effect = jump + slope*duration
            pm.Deterministic("slope_change", sign * b3)
            pm.Deterministic("effect_end", sign * ((beta[1] - beta[0]) + b3 * dur))
            pm.Deterministic("es_end", sign * ((beta[1] - beta[0]) + b3 * dur) / sigma)

        def _sample():
            return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                             progressbar=progressbar, compute_convergence_checks=False,
                             target_accept=0.95)
        from functions.sced.model_cache import fit_or_load, array_hash
        _spec = {"engine": "per_case", "model": model, "improvement": improvement, "trend": trend,
                 "baseline_trend": baseline_trend, "prior_mean": prior_mean, "sigma_max": sigma_max,
                 "draws": draws, "tune": tune, "chains": chains, "seed": seed,
                 "start_index": int(start_index), "data": array_hash(y, [start_index])}
        _prefix = f"percase_{model}{'+trend' if trend else ''}"
        idata, _ = fit_or_load(cache_dir, _prefix, _spec, _sample, force_refit=force_refit)

    post = idata.posterior
    es = post["es"].values.ravel()
    lvl = post["level_change"].values.ravel()
    rho_draws = post["rho"].values.ravel()
    import arviz as az
    diag_vars = ["es", "beta", "sigma"] + (["rho"] if model != "simple" else []) \
        + (["b3"] if trend else []) + (["b1"] if include_b1 else [])
    summ = az.summary(idata, var_names=diag_vars, kind="diagnostics")
    out = {"model": model, "trend": bool(trend), "n": int(T), "n_a": int(start_index),
           "n_b": int(T - start_index),
           "es": _summarise(es), "level_change": _summarise(lvl), "rho": _summarise(rho_draws),
           "p_es_gt_rope": float(np.mean(es > rope)), "rope": rope,
           # baseline trend test (Tarlow) + correction applied or not
           "baseline_trend_tau": (round(float(ba_tau), 3) if ba_tau is not None
                                  and np.isfinite(ba_tau) else None),
           "baseline_trend_p": (round(float(ba_p), 4) if ba_p is not None
                                and np.isfinite(ba_p) else None),
           "baseline_corrected": bool(include_b1),
           "rhat_max": float(np.nanmax(summ["r_hat"].values)) if len(summ) else np.nan,
           "ess_min": float(np.nanmin(summ["ess_bulk"].values)) if len(summ) else np.nan,
           "diverging": int(idata.sample_stats["diverging"].values.sum())}
    if include_b1:
        out["baseline_trend_b1"] = _summarise(post["baseline_trend_b1"].values.ravel())
    if trend:                                  # cumulative / late effect
        es_end = post["es_end"].values.ravel()
        out["slope_change"] = _summarise(post["slope_change"].values.ravel())
        out["effect_end"] = _summarise(post["effect_end"].values.ravel())
        out["es_end"] = _summarise(es_end)
        out["p_es_end_gt_rope"] = float(np.mean(es_end > rope))
    if model == "bucp":
        cp_draws = post["cp"].values.ravel()
        vals, counts = np.unique(cp_draws, return_counts=True)
        mode = int(vals[np.argmax(counts)])
        lo_cp, hi_cp = _hdi(cp_draws.astype(float), 0.95)
        out["cp"] = {"median": float(np.median(cp_draws)), "mode_lastA": mode,
                     "start_B_mode": mode + 1, "hdi_low": lo_cp, "hdi_high": hi_cp,
                     # concentration = immediacy: share of the posterior at the mode
                     "concentration": float(counts.max() / counts.sum())}
    out["_idata"] = idata
    return out


def bayes_hier_sced(df, *, tier_col, session_col, phase_col, outcome_col,
                    baseline=None, treatment=None, improvement="increase",
                    hypothesis="immediate_flat", terms=None, pooling="partial",
                    family="gaussian", bounds=None, n_trials=None,
                    ar=True, rope=0.5, draws=2000, tune=2000, chains=4, seed=42,
                    target_accept=0.95, progressbar=False, cache_dir=None, force_refit=False):
    """
    Hierarchical Bayesian SCED model (one model, all cases): within-case AR1 +
    between-case random effects (partial pooling) -> INDIVIDUAL (shrunk) AND
    POPULATION evidence, without selection or asymptotic df (Rindskopf 2014; Van den Noortgate &
    Onghena; Moeyaert).

        y_it = b0_i + b1_i*time_c + b2_i*phase_B + b3_i*time_B + AR1
        (b0,b1,b2,b3)_i ~ Normal(mu, diag(tau))        # per-case random effects

    ``hypothesis`` fixes the terms (pre-specified, not data-driven):
      - ``immediate_flat``   : b2 (jump), flat baseline         [hierarchical BITS]
      - ``cumulative_flat``  : b2 + b3 (slope), flat baseline
      - ``immediate_trend``  : b1 (trend) + b2
      - ``cumulative_trend`` : b1 + b2 + b3
    Priors: soft sceptical b2 effect ``Normal(0, sd_y)``; regularizing b1/b3 nuisances
    (scale ``sd_y/n_sessions``); AR1 ``rho~Uniform(-1,1)``; residual sigma ``HalfNormal(sd_y)``
    (well identified -> stable); between-case SD tau ``HalfNormal(sd_y)``.

    Reported effect: POPULATION level in original units + HDI + pd (+ P(effect>ROPE) if
    ``rope`` given, ROPE in original units / MCID). Everything is oriented (``>0`` = improvement).
    For a design-comparable effect size (BC-SMD g_AB), use the R package scdhlm
    via ``bc_smd_scdhlm`` (canonical REML estimator, Pustejovsky-Hedges-Shadish).

    Returns ``{population, heterogeneity, per_case, rhat_max, ess_min, diverging, ...}``.

    References: Van den Noortgate & Onghena 2003; Moeyaert, Ferron, Beretvas & Van den Noortgate
    2014; Rindskopf 2014 (hierarchical Bayesian SCED, partial pooling).
    R equivalent: brms (multilevel brm with per-case random effects + ar(p=1)); bayestestR (HDI/pd/ROPE).
    """
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    logging.getLogger("pymc").setLevel(logging.ERROR)

    # model terms: ``terms`` (explicit set among trend/level/slope) takes precedence over
    # ``hypothesis`` (backward-compatible preset, b2 always present).
    if terms is not None:
        terms = set(terms)
        use_b1, use_b2, use_b3 = "trend" in terms, "level" in terms, "slope" in terms
    else:
        use_b1 = hypothesis in ("immediate_trend", "cumulative_trend")
        use_b2 = True
        use_b3 = hypothesis in ("cumulative_flat", "cumulative_trend")
    if not (use_b1 or use_b2 or use_b3):
        raise ValueError("At least one term required among {'trend','level','slope'}.")
    if pooling not in ("partial", "random_intercept", "none", "correlated"):
        raise ValueError("pooling in {'partial','random_intercept','none','correlated'}.")
    d = df[[tier_col, session_col, phase_col, outcome_col]].copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d[session_col] = pd.to_numeric(d[session_col], errors="coerce")
    d = d.dropna(subset=[outcome_col, session_col])
    phs = d[phase_col].astype(str)
    baseline = baseline if baseline is not None else sorted(phs.unique())[0]
    treatment = treatment if treatment is not None else [p for p in sorted(phs.unique())
                                                         if p != str(baseline)][-1]
    d = d[phs.isin([str(baseline), str(treatment)])].copy()
    d = d.sort_values([tier_col, session_col]).reset_index(drop=True)
    cases = sorted(d[tier_col].astype(str).unique().tolist())
    if len(cases) < 2:
        raise ValueError("The hierarchical model requires >= 2 cases.")
    cidx = d[tier_col].astype(str).map({c: i for i, c in enumerate(cases)}).to_numpy()
    y = d[outcome_col].to_numpy(float)
    phase01 = (d[phase_col].astype(str) == str(treatment)).to_numpy(float)
    # centered time (trend) + dose in B (0,1,2...), PER case; "previous" AR1 index
    time_c = np.zeros(len(d)); tsince = np.zeros(len(d))
    prev_idx = np.arange(len(d)); is_first = np.ones(len(d))
    for i, c in enumerate(cases):
        m = np.where(cidx == i)[0]
        s = d[session_col].to_numpy()[m]
        time_c[m] = s - s.mean()
        ph = phase01[m]
        tb = np.where(ph == 1, np.maximum(0.0, np.arange(len(m)) - int(np.argmax(ph == 1))), 0.0)
        tsince[m] = tb
        is_first[m[0]] = 1.0
        for j in range(1, len(m)):
            prev_idx[m[j]] = m[j - 1]; is_first[m[j]] = 0.0
    sign = 1.0 if improvement == "increase" else -1.0
    mean_y, sd_y = float(np.mean(y)), float(max(np.std(y), 1.0))
    if rope == "auto":                                  # ROPE = 0.1*SD (Kruschke/bayestestR convention)
        rope = round(0.1 * float(np.std(y)), 4)
    typ_n = max(float(np.median(np.bincount(cidx))), 2.0)
    nC = len(cases)
    logit_fam = family in ("beta", "binomial")      # logit link, effect in POINTS, no AR1
    student_fam = (family == "student")
    if logit_fam:                                   # BOUNDED outcome (Beta) / COUNT k/N (Binomial)
        import scipy.special as _sp
        lo, hi = bounds if bounds is not None else (
            (0.0, 100.0) if (y.min() >= 0 and y.max() <= 100) else (float(y.min()), float(y.max())))
        rng_ = max(hi - lo, 1e-6)
        n_ = len(y)
        prop = np.clip((y - lo) / rng_, 0.0, 1.0)
        y01 = np.clip((prop * (n_ - 1) + 0.5) / n_, 1e-4, 1 - 1e-4)   # Smithson-Verkuilen (beta)
        if family == "binomial":
            if n_trials is None:
                raise ValueError("family='binomial' requires n_trials (number of items N per measurement).")
            k = np.clip(np.round(prop * float(n_trials)), 0, n_trials).astype(int)
        c0, s0, s2, slp = float(_sp.logit(np.clip(y01.mean(), 1e-3, 1 - 1e-3))), 1.5, 1.5, 1.5 / typ_n
        sd0_pr = sd2_pr = 1.0; slp_sd = 1.5 / typ_n
    else:                                           # gaussian / student: original scale
        c0, s0, s2, slp = mean_y, 2 * sd_y, sd_y, sd_y / typ_n
        sd0_pr = sd2_pr = sd_y; slp_sd = sd_y / typ_n

    if logit_fam and pooling not in ("partial", "random_intercept", "none"):
        raise ValueError("beta/binomial families: pooling 'partial', 'random_intercept' or 'none' required.")
    # pooling='none' + beta/binomial = INDEPENDENT per-case Beta regressions (no borrowing from
    # the group). Valid as stage 1 of a two-stage meta-analysis; wider per-case posteriors.
    if pooling == "correlated" and logit_fam:
        raise ValueError("pooling='correlated': continuous families (gaussian/student) only.")

    with pm.Model(coords={"case": [str(c) for c in cases],
                          "obs": np.arange(len(d))}) as m:           # cases named by patient; obs indexed
        # phase (0=baseline,1=treatment) and case of EACH observation -> grouped PPC from the .nc
        pm.ConstantData("obs_phase", phase01, dims="obs")
        pm.ConstantData("obs_case", cidx.astype("int64"), dims="obs")
        # spec of included coefficients (intercept ALWAYS): (name, center, sd_prior_mu, sd_prior_tau)
        spec = [("b0", c0, s0, sd0_pr)]
        if use_b1: spec.append(("b1", 0.0, slp, slp_sd))
        if use_b2: spec.append(("b2", 0.0, s2, sd2_pr))
        if use_b3: spec.append(("b3", 0.0, slp, slp_sd))
        coefs, mus, sds = {}, {}, {}
        re_corr_b2 = None                               # intercept<->effect correlation (ceiling), if correlated

        if pooling == "correlated":                     # CORRELATED random effects (LKJ covariance)
            D = len(spec)
            mu_vec = pm.Normal("mu_vec", mu=np.array([c for _, c, _, _ in spec]),
                               sigma=np.array([m_ for _, _, m_, _ in spec]), shape=D)
            sd_dist = pm.HalfNormal.dist(sigma=np.array([t for _, _, _, t in spec]), shape=D)
            chol, corr, stds = pm.LKJCholeskyCov("chol_re", n=D, eta=2.0, sd_dist=sd_dist,
                                                 compute_corr=True)
            re = pt.dot(pm.Normal("z_re", 0.0, 1.0, shape=(nC, D)), chol.T)   # (nC, D)
            for i, (name, _, _, _) in enumerate(spec):
                coefs[name] = pm.Deterministic(name, mu_vec[i] + re[:, i], dims="case")
                mus[name] = mu_vec[i]; sds[name] = stds[i]
            if use_b2:                                  # corr(b0, b2): baseline level <-> effect
                j = [n for n, _, _, _ in spec].index("b2")
                re_corr_b2 = pm.Deterministic("re_corr_b2", corr[0, j])
        else:                                           # independent: partial / random_intercept / none
            def _mk(name, center, mu_sd, sd_prior, is_intercept=False):
                if pooling == "none":
                    return pm.Normal(name, center, mu_sd, dims="case"), None, None
                mu = pm.Normal("mu_" + name, center, mu_sd)
                if pooling == "partial" or is_intercept:        # random slope / random intercept
                    sd = pm.HalfNormal("sd_" + name, sd_prior)
                    return (pm.Deterministic(name, mu + sd * pm.Normal("z_" + name, 0, 1, shape=nC),
                                             dims="case"), mu, sd)
                return pm.Deterministic(name, mu * pt.ones(nC), dims="case"), mu, None  # common slope (complete)
            for name, center, mu_sd, sd_prior in spec:
                coefs[name], mus[name], sds[name] = _mk(name, center, mu_sd, sd_prior,
                                                        is_intercept=(name == "b0"))

        b0, mu_b0, sd_b0 = coefs["b0"], mus["b0"], sds["b0"]
        b0_pop = mu_b0 if mu_b0 is not None else pt.mean(b0)   # population intercept (fallback if pooling='none')
        eta = b0[cidx]
        mu_b2 = sd_b2 = None
        if use_b2:
            b2, mu_b2, sd_b2 = coefs["b2"], mus["b2"], sds["b2"]
            eta = eta + b2[cidx] * phase01
        mu_b1 = sd_b1 = None
        if use_b1:
            b1, mu_b1, sd_b1 = coefs["b1"], mus["b1"], sds["b1"]
            eta = eta + b1[cidx] * time_c
        if use_b3:
            b3, mu_b3, sd_b3 = coefs["b3"], mus["b3"], sds["b3"]
            eta = eta + b3[cidx] * tsince

        if logit_fam:                               # Beta / Binomial (logit link) - bounded / count
            mu01 = pm.math.invlogit(eta)
            if family == "binomial":
                pm.Binomial("y_obs", n=int(n_trials), p=mu01, observed=k, dims="obs")
            else:
                phi = pm.HalfNormal("phi", sigma=50.0)
                pm.Beta("y_obs", alpha=mu01 * phi, beta=(1.0 - mu01) * phi, observed=y01, dims="obs")
            if use_b2:                              # level effect only if the term is present
                lvl_pop = mu_b2 if mu_b2 is not None else pt.mean(b2)   # fallback pooling='none'
                pA = pm.math.invlogit(b0_pop); pB = pm.math.invlogit(b0_pop + lvl_pop)
                pm.Deterministic("pop_level", sign * (pB - pA) * rng_)   # effect in POINTS (original scale)
                pm.Deterministic("logit_b2", sign * lvl_pop)
                pm.Deterministic("tau_level", sd_b2 if sd_b2 is not None else pt.zeros(()))
        else:                                       # Gaussian / Student-t (+ AR1 if ar=True) + BC-SMD
            sigma = pm.HalfNormal("sigma", sigma=sd_y)   # residual sigma well identified -> HalfNormal (stable)
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1) if student_fam else None   # degrees of freedom (robustness)
            if ar:
                rho = pm.Uniform("rho", -0.99, 0.99)
                resid_prev = y[prev_idx] - eta[prev_idx]
                mu_obs = eta + (1.0 - is_first) * rho * resid_prev
                sd_obs = pt.where(pt.eq(pt.as_tensor(is_first), 1.0),
                                  sigma / pt.sqrt(1.0 - rho ** 2), sigma)
            else:
                mu_obs, sd_obs = eta, sigma
            if student_fam:                         # heavy tails: robust to outliers
                pm.StudentT("y_obs", nu=nu, mu=mu_obs, sigma=sd_obs, observed=y, dims="obs")
            else:
                pm.Normal("y_obs", mu=mu_obs, sigma=sd_obs, observed=y, dims="obs")
            if use_b2:
                lvl = mu_b2 if mu_b2 is not None else pt.mean(b2)
                tau_lvl = sd_b2 if sd_b2 is not None else pt.std(b2)   # 0 if random_intercept (common slope)
                pm.Deterministic("pop_level", sign * lvl)             # POPULATION effect in original units
                pm.Deterministic("tau_level", tau_lvl)
        if use_b3:
            slp_pop = mu_b3 if mu_b3 is not None else pt.mean(b3)
            tau_slp = sd_b3 if sd_b3 is not None else pt.std(b3)
            pm.Deterministic("pop_slope", sign * slp_pop)
            pm.Deterministic("tau_slope", tau_slp)
            Tb = float(np.max(tsince))                          # longest horizon of phase B
            if logit_fam:        # Beta/Binomial slope: conversion to POINTS (non-linear logit link)
                p0 = pm.math.invlogit(b0_pop)
                pm.Deterministic("pop_slope_pts_ps",           # points / session (at the start of B)
                                 sign * (pm.math.invlogit(b0_pop + slp_pop) - p0) * rng_)
                pm.Deterministic("pop_slope_pts_cum",          # points accumulated over the whole B phase
                                 sign * (pm.math.invlogit(b0_pop + slp_pop * Tb) - p0) * rng_)
            else:                                              # gaussian/student: slope already in original units
                pm.Deterministic("pop_slope_pts_ps", sign * slp_pop)
                pm.Deterministic("pop_slope_pts_cum", sign * slp_pop * Tb)
        if use_b1:                                             # baseline trend (nuisance, but reported)
            trd_pop = mu_b1 if mu_b1 is not None else pt.mean(b1)
            tau_trd = sd_b1 if sd_b1 is not None else pt.std(b1)
            pm.Deterministic("pop_trend", sign * trd_pop)      # raw coefficient (logit/session in beta)
            pm.Deterministic("tau_trend", tau_trd)
            if logit_fam:        # Beta/Binomial trend: also in POINTS/session (marginal at mid-series)
                p0t = pm.math.invlogit(b0_pop)
                pm.Deterministic("pop_trend_pts_ps", sign * (pm.math.invlogit(b0_pop + trd_pop) - p0t) * rng_)
            else:                                              # gaussian/student: already in points/session
                pm.Deterministic("pop_trend_pts_ps", sign * trd_pop)
        # --- TOTAL EFFECT AT END OF PHASE B = b2 + b3*T_B (jump + cumulative slope), IN POINTS ---
        # The best-identified estimand: the level<->slope trade-off cancels in the sum.
        # Reported per case AND population, for ALL poolings (partial/RI/none) and families.
        if use_b2 or use_b3:
            Tb_all = float(np.max(tsince)) if tsince.size else 0.0
            Tb_case = np.array([float(tsince[cidx == i].max()) if np.any(cidx == i) else 0.0
                                for i in range(nC)])           # B horizon specific to each case
            shift_case = pt.zeros(nC); shift_pop = 0.0
            if use_b2:
                shift_case = shift_case + b2
                shift_pop = shift_pop + (mu_b2 if mu_b2 is not None else pt.mean(b2))
            if use_b3:
                shift_case = shift_case + b3 * pt.as_tensor(Tb_case)
                shift_pop = shift_pop + (mu_b3 if mu_b3 is not None else pt.mean(b3)) * Tb_all
            if logit_fam:                                      # logit -> points (at each case's baseline level)
                eff_case = (pm.math.invlogit(b0 + shift_case) - pm.math.invlogit(b0)) * rng_
                eff_pop = (pm.math.invlogit(b0_pop + shift_pop) - pm.math.invlogit(b0_pop)) * rng_
            else:                                              # gaussian/student: already in points
                eff_case = shift_case; eff_pop = shift_pop
            pm.Deterministic("effect_end_pts", sign * eff_case, dims="case")
            pm.Deterministic("pop_effect_end_pts", sign * eff_pop)
        # per case IN POINTS (oriented) -> inputs of the two-stage meta-analysis PER PARAMETER
        if logit_fam:
            _b0p = pm.math.invlogit(b0)
            if use_b1:
                pm.Deterministic("trend_pts", sign * (pm.math.invlogit(b0 + b1) - _b0p) * rng_, dims="case")
            if use_b2:
                pm.Deterministic("level_pts", sign * (pm.math.invlogit(b0 + b2) - _b0p) * rng_, dims="case")
            if use_b3:
                pm.Deterministic("slope_pts", sign * (pm.math.invlogit(b0 + b3) - _b0p) * rng_, dims="case")
        else:
            if use_b1:
                pm.Deterministic("trend_pts", sign * b1, dims="case")
            if use_b2:
                pm.Deterministic("level_pts", sign * b2, dims="case")
            if use_b3:
                pm.Deterministic("slope_pts", sign * b3, dims="case")
        def _sample():                                  # MCMC + PPC (called only if not cached)
            _id = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                            target_accept=target_accept, progressbar=progressbar,
                            idata_kwargs={"log_likelihood": True},     # required for LOO
                            compute_convergence_checks=False)
            try:                                        # replicates for PPC + LOO-PIT calibration (QQ)
                pm.sample_posterior_predictive(_id, extend_inferencedata=True,
                                               random_seed=seed, progressbar=False)
            except Exception:
                pass
            return _id
        from functions.sced.model_cache import fit_or_load, data_hash
        _spec = {"engine": "hier", "terms": sorted([t for t, u in (("trend", use_b1),
                 ("level", use_b2), ("slope", use_b3)) if u]), "family": family, "pooling": pooling,
                 "ar": ar, "bounds": bounds, "n_trials": n_trials, "baseline": baseline,
                 "treatment": treatment, "improvement": improvement,
                 "draws": draws, "tune": tune, "chains": chains, "seed": seed,
                 "target_accept": target_accept,
                 "data": data_hash(d, [tier_col, session_col, phase_col, outcome_col])}
        _prefix = f"{outcome_col}__{'+'.join(_spec['terms']) or 'b0'}__{family}__{pooling}"
        idata, _ = fit_or_load(cache_dir, _prefix, _spec, _sample, force_refit=force_refit)

    post = idata.posterior
    summ = az.summary(idata, kind="diagnostics")   # all parameters -> worst-case R-hat/ESS robust
    out = {"hypothesis": hypothesis, "terms": sorted([t for t, u in
                (("trend", use_b1), ("level", use_b2), ("slope", use_b3)) if u]),
           "pooling": pooling, "family": family, "n_cases": nC, "n_obs": int(len(d)),
           "population": {}, "heterogeneity": {},
           "rhat_max": float(np.nanmax(summ["r_hat"].values)) if len(summ) else np.nan,
           "ess_min": float(np.nanmin(summ["ess_bulk"].values)) if len(summ) else np.nan,
           "diverging": int(idata.sample_stats["diverging"].values.sum())}
    if use_b1:
        out["population"]["trend"] = _summarise(post["pop_trend"].values.ravel())
        if "pop_trend_pts_ps" in post:               # baseline trend readable in points/session
            out["population"]["trend_points_per_session"] = _summarise(post["pop_trend_pts_ps"].values.ravel())
        out["heterogeneity"]["sd_trend"] = _summarise(post["tau_trend"].values.ravel())
    if use_b2:
        lvl_s = _summarise(post["pop_level"].values.ravel())
        if rope is not None:                         # ROPE (MCID): %in + P(>ROPE) + Kruschke 2018 decision
            lvl_s.update(_rope_stats(post["pop_level"].values.ravel(), rope))
        out["population"]["level"] = lvl_s
        out["heterogeneity"]["sd_level"] = _summarise(post["tau_level"].values.ravel())
    if use_b3:
        out["population"]["slope"] = _summarise(post["pop_slope"].values.ravel())
        out["heterogeneity"]["sd_slope"] = _summarise(post["tau_slope"].values.ravel())
        if "pop_slope_pts_cum" in post:              # slope in POINTS (readable): /session + cumulative over B
            out["population"]["slope_points_per_session"] = _summarise(post["pop_slope_pts_ps"].values.ravel())
            out["population"]["slope_points_cumulative"] = _summarise(post["pop_slope_pts_cum"].values.ravel())
            out["population"]["slope_horizon_B"] = float(np.max(tsince))
    if "pop_effect_end_pts" in post:                 # TOTAL end-of-phase effect (b2 + b3*T_B), in points
        ee_s = _summarise(post["pop_effect_end_pts"].values.ravel())
        if rope is not None:                         # ROPE on the cumulative effect (the clinical estimand)
            ee_s.update(_rope_stats(post["pop_effect_end_pts"].values.ravel(), rope))
        out["population"]["effect_end"] = ee_s
    if pooling == "correlated" and "re_corr_b2" in post:   # corr(baseline level, effect) - EXPLORATORY
        out["re_corr_level"] = _summarise(post["re_corr_b2"].values.ravel())
    if logit_fam and use_b2:
        out["logit_b2"] = {**_summarise(post["logit_b2"].values.ravel())}
    elif not logit_fam and use_b2:
        if ar:
            out["rho"] = _summarise(post["rho"].values.ravel())
        if student_fam:
            out["nu"] = _summarise(post["nu"].values.ravel())
    # INDIVIDUAL shrunk evidence (partial pooling) - effect in POINTS (interpretable)
    per = [{"case": c} for c in cases]
    if use_b1:                                      # PER-CASE baseline trend (pts/session; logit->pts in beta)
        if logit_fam:
            from scipy.special import expit
            b0c = post["b0"].values.reshape(-1, nC); b1r = post["b1"].values.reshape(-1, nC)
            b1c = np.stack([sign * (expit(b0c[:, i] + b1r[:, i]) - expit(b0c[:, i])) * rng_
                            for i in range(nC)], axis=1)
        else:
            b1c = post["b1"].values.reshape(-1, nC) * sign
        for i in range(nC):
            per[i]["trend"] = _summarise(b1c[:, i])
    if use_b2:
        if logit_fam:                               # Beta/Binomial: convert b2 (logit) to delta points per case
            from scipy.special import expit
            b0c = post["b0"].values.reshape(-1, nC); b2r = post["b2"].values.reshape(-1, nC)
            b2c = np.stack([sign * (expit(b0c[:, i] + b2r[:, i]) - expit(b0c[:, i])) * rng_
                            for i in range(nC)], axis=1)
        else:
            b2c = post["b2"].values.reshape(-1, nC) * sign   # already in original units
        for i in range(nC):
            per[i]["level"] = _summarise(b2c[:, i])
    if use_b3:
        if logit_fam:                               # Beta/Binomial: convert b3 (logit) to delta points/session per case
            from scipy.special import expit
            b0c = post["b0"].values.reshape(-1, nC); b3r = post["b3"].values.reshape(-1, nC)
            b3c = np.stack([sign * (expit(b0c[:, i] + b3r[:, i]) - expit(b0c[:, i])) * rng_
                            for i in range(nC)], axis=1)
        else:
            b3c = post["b3"].values.reshape(-1, nC) * sign   # already in original units (points/session)
        for i in range(nC):
            per[i]["slope"] = _summarise(b3c[:, i])
    if "effect_end_pts" in post:                    # per-case total end-of-phase effect (points, oriented)
        ee = post["effect_end_pts"].values.reshape(-1, nC)
        for i in range(nC):
            per[i]["effect_end"] = _summarise(ee[:, i])
    out["per_case"] = per
    if "phi" in post:                                # Beta precision (residual dispersion) -> report
        out["phi"] = _summarise(post["phi"].values.ravel())
    # --- Autocorrelation PPC (BARG): does the conditional-independence structure reproduce
    #     the observed serial autocorrelation? p~0.5 = adequate; near 0/1 = autocorr poorly captured ---
    try:
        cidx = d[tier_col].astype(str).to_numpy()
        yobs = pd.to_numeric(d[outcome_col], errors="coerce").to_numpy(dtype=float)

        def _pooled_ac1(vals):                       # POOLED within-case lag-1 autocorr (centered by case)
            num = den = 0.0
            for c in np.unique(cidx):
                v = vals[cidx == c]; v = v[np.isfinite(v)]
                if len(v) < 3:
                    continue
                v = v - v.mean()
                num += float(np.sum(v[:-1] * v[1:])); den += float(np.sum(v * v))
            return num / den if den > 0 else np.nan
        obs_ac = _pooled_ac1(yobs)
        pp = idata.posterior_predictive["y_obs"].values            # (chain, draw, obs)
        rep = pp.reshape(-1, pp.shape[-1])
        sel = np.linspace(0, len(rep) - 1, min(300, len(rep))).astype(int)
        reps = np.array([_pooled_ac1(rep[i]) for i in sel]); reps = reps[np.isfinite(reps)]
        if np.isfinite(obs_ac) and len(reps):
            out["autocorr_obs"] = round(float(obs_ac), 3)
            out["autocorr_ppc_p"] = round(float(np.mean(reps >= obs_ac)), 3)
    except Exception:
        pass
    out["_idata"] = idata
    return out


def hier_param_count(terms, n_cases, pooling="partial", family="gaussian", ar=False):
    """NOMINAL count of free parameters of a hierarchical SCED model (intercept ALWAYS
    included). Serves as a STRUCTURAL complexity index (to compare with n_cases / n_obs). Note: the
    EFFECTIVE number of parameters (shrunk by the hierarchy) is rather the comparator's ``p_loo``."""
    D = 1 + sum(t in set(terms) for t in ("trend", "level", "slope"))   # +intercept
    if pooling == "none":
        re = n_cases * D; pop = 0; tau = 0; corr = 0
    elif pooling == "random_intercept":
        re = n_cases; pop = D; tau = 1; corr = 0
    elif pooling == "correlated":
        re = n_cases * D; pop = D; tau = D; corr = D * (D - 1) // 2
    else:                                               # partial (random slopes)
        re = n_cases * D; pop = D; tau = D; corr = 0
    resid = {"gaussian": 1, "student": 2, "beta": 1, "binomial": 0}.get(family, 1)
    resid += 1 if (ar and family in ("gaussian", "student")) else 0
    return int(pop + tau + corr + re + resid)


# default comparison set: MEAN structure (the only compared axis)
_DEFAULT_MODELS = {
    "M0 trend":              {"trend"},                    # trend only = "no effect"
    "Mi trend+level":        {"trend", "level"},           # immediate effect (jump)
    "Mg trend+slope":        {"trend", "slope"},           # gradual effect (slope), without jump
    "Mf trend+level+slope":  {"trend", "level", "slope"},  # both
}


def compare_sced_models(df, *, tier_col, session_col, phase_col, outcome_col,
                        models=None, baseline=None, treatment=None, improvement="increase",
                        family="gaussian", bounds=None, n_trials=None, pooling="partial",
                        ar=False, rope=0.5, draws=2000, tune=2000, chains=4, seed=42,
                        target_accept=0.99, se_mult=2.0, progressbar=False,
                        cache_dir=None, force_refit=False):
    """Compares nested MEAN structures with several canonical Bayesian comparators (ArviZ):
    PSIS-LOO and WAIC (Vehtari et al. 2017; Watanabe 2010), with stacking weights (Yao et al.
    2018) AND pseudo-BMA+ (Bayesian bootstrap).

    Only the mean structure is compared (``models`` = dict name->set of terms among
    {'trend','level','slope'}). The error structure (``ar``) and the ``pooling`` are FIXED
    (sensitivity axes, not compared). Default = the 4 models of ``_DEFAULT_MODELS``.

    Decision rule (parsimony): the best (LOO) is kept as decisive only if the
    ``elpd_diff`` gap to the 2nd exceeds ``se_mult``*``dse``; otherwise -> "indistinguishable"
    (prefer the simplest / a model-averaged effect via the stacking weights). AR=False by
    default because per-observation LOO/WAIC assumes conditional independence.

    Returns ``{compare (LOO), compare_waic, weights (stacking), weights_pbma, best, decisive,
    decision, models{name: result}}``.

    References: Vehtari, Gelman & Gabry 2017 (PSIS-LOO); Watanabe 2010 (WAIC); Yao, Vehtari,
    Simpson & Gelman 2018 (stacking); Manolov & Moeyaert 2025 (model comparison for SCED).
    R equivalent: loo::loo / loo::waic / loo::loo_compare / loo::loo_model_weights.
    """
    import arviz as az
    models = models or _DEFAULT_MODELS
    fitted, idatas = {}, {}
    for name, tset in models.items():
        r = bayes_hier_sced(df, tier_col=tier_col, session_col=session_col, phase_col=phase_col,
                            outcome_col=outcome_col, terms=tset, pooling=pooling, baseline=baseline,
                            treatment=treatment, improvement=improvement, family=family, bounds=bounds,
                            n_trials=n_trials, ar=ar, rope=rope, draws=draws, tune=tune, chains=chains,
                            seed=seed, target_accept=target_accept, progressbar=progressbar,
                            cache_dir=cache_dir, force_refit=force_refit)
        fitted[name] = r; idatas[name] = r["_idata"]

    def _cmp(ic, weight_method="stacking"):
        c = az.compare(idatas, ic=ic, method=weight_method)
        keep = {"loo": ["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"],
                "waic": ["rank", "elpd_waic", "p_waic", "elpd_diff", "dse", "weight"]}[ic]
        return c[[k for k in keep if k in c.columns]]

    cmp = _cmp("loo", "stacking")                              # main comparator: LOO + stacking
    cmp_waic = _cmp("waic", "stacking")                        # canonical comparator #2: WAIC
    pbma = az.compare(idatas, ic="loo", method="BB-pseudo-BMA")  # alternative weights (pseudo-BMA+)
    # worst-case Pareto-k per model (LOO reliability)
    kbad = {}
    for name, idata in idatas.items():
        loo = az.loo(idata, pointwise=True)
        kbad[name] = int((np.asarray(loo.pareto_k) > 0.7).sum())
    cmp["k>0.7"] = [kbad.get(i, np.nan) for i in cmp.index]
    # COMPLEXITY index relative to the dataset: nominal n_params + p_loo (effective) / data
    nC = next(iter(fitted.values()))["n_cases"]; nO = next(iter(fitted.values()))["n_obs"]
    cmp["n_params"] = [hier_param_count(fitted[m]["terms"], nC, pooling, family, ar) for m in cmp.index]
    cmp["n_par/n_cas"] = (cmp["n_params"] / max(nC, 1)).round(2)
    if "p_loo" in cmp.columns:
        cmp["p_eff/N_obs"] = (cmp["p_loo"] / max(nO, 1)).round(3)
    best = cmp.index[cmp["rank"] == 0][0]
    runner = cmp.index[cmp["rank"] == 1][0] if (cmp["rank"] == 1).any() else None
    decisive = bool(runner is not None and
                    cmp.loc[runner, "elpd_diff"] > se_mult * cmp.loc[runner, "dse"])
    if runner is None:
        decision = f"{best} (single model)"
    elif decisive:
        decision = f"{best} preferred (Delta-elpd {cmp.loc[runner,'elpd_diff']:.1f} > {se_mult}*dse)"
    else:
        decision = (f"INDISTINGUISHABLE (Delta-elpd {cmp.loc[runner,'elpd_diff']:.1f} <= {se_mult}xdse "
                    f"{se_mult*cmp.loc[runner,'dse']:.1f}) -> keep the simplest / model-averaged effect")
    waic_best = cmp_waic.index[cmp_waic["rank"] == 0][0]
    return {"compare": cmp, "compare_waic": cmp_waic,
            "weights": {i: float(cmp.loc[i, "weight"]) for i in cmp.index},
            "weights_pbma": {i: float(pbma.loc[i, "weight"]) for i in pbma.index},
            "best": best, "best_waic": waic_best, "loo_waic_agree": bool(best == waic_best),
            "decisive": decisive, "decision": decision, "models": fitted}


def select_percase_models(df, *, tier_col, session_col, phase_col, outcome_col,
                          models=None, baseline=None, treatment=None, improvement="increase",
                          family="beta", bounds=None, n_trials=None, rope=0.5,
                          draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99,
                          se_mult=2.0, cache_dir=None, force_refit=False, progressbar=False):
    """PER-PATIENT model selection by PSIS-LOO, REUSING ``bayes_hier_sced`` with
    ``pooling='none'`` (independent per-case regressions).

    Each model in ``models`` (mean structures; default ``_DEFAULT_MODELS``) is fitted
    ONCE over all patients (independent per-case parameters). Since with
    ``pooling='none'`` a patient's likelihood depends ONLY on its own parameters, a
    PER-PATIENT LOO is obtained by summing its pointwise ``elpd_loo`` - hence a ranking of the
    models specific to each patient (no additional fit). The per-patient estimands are
    already provided by ``bayes_hier_sced`` (``per_case``); the report only has to PICK those of
    each patient's best model.

    Returns ``{per_patient: {case: {best, runner, decisive, decision, table}}, fitted, cases,
    model_names}``. ``table`` = DataFrame (model, n_obs, elpd_loo, rank, elpd_diff, k_worst).

    References: Vehtari, Gelman & Gabry 2017 (per-observation PSIS-LOO, summed within case).
    R equivalent: loo::loo (pointwise) - potential equivalent, to test (the per-case sum is manual).
    """
    import arviz as az
    models = models or _DEFAULT_MODELS
    fitted = {}
    for name, tset in models.items():
        fitted[name] = bayes_hier_sced(
            df, tier_col=tier_col, session_col=session_col, phase_col=phase_col,
            outcome_col=outcome_col, terms=tset, pooling="none", family=family, bounds=bounds,
            n_trials=n_trials, baseline=baseline, treatment=treatment, improvement=improvement,
            rope=rope, draws=draws, tune=tune, chains=chains, seed=seed,
            target_accept=target_accept, progressbar=progressbar, cache_dir=cache_dir,
            force_refit=force_refit)
    model_names = list(fitted.keys())
    first = fitted[model_names[0]]["_idata"]
    cases = [str(c) for c in first.posterior.coords["case"].values]      # order of cases (identical across models)
    # pointwise LOO per model (same observation order everywhere: same df) + map obs -> case
    loo_i, obs_case = {}, {}
    for name in model_names:
        idata = fitted[name]["_idata"]
        lo = az.loo(idata, pointwise=True)
        loo_i[name] = (np.asarray(lo.loo_i).ravel(), np.asarray(lo.pareto_k).ravel())
        obs_case[name] = np.asarray(idata.constant_data["obs_case"].values).ravel().astype(int)

    per_patient = {}
    for ci, case in enumerate(cases):
        rows = []
        for name in model_names:
            elpd_pw, pk = loo_i[name]
            mask = obs_case[name] == ci
            rows.append({"model": name, "n_obs": int(mask.sum()),
                         "elpd_loo": float(np.sum(elpd_pw[mask])),
                         "k_worst": float(np.max(pk[mask])) if mask.any() else np.nan})
        tab = (pd.DataFrame(rows).sort_values("elpd_loo", ascending=False)
               .reset_index(drop=True))
        tab["rank"] = range(len(tab))
        tab["elpd_diff"] = round(tab["elpd_loo"].iloc[0], 4) - tab["elpd_loo"]
        tab["elpd_loo"] = tab["elpd_loo"].round(3); tab["elpd_diff"] = tab["elpd_diff"].round(3)
        tab["k_worst"] = tab["k_worst"].round(2)
        best = tab.loc[0, "model"]; runner = tab.loc[1, "model"] if len(tab) > 1 else None
        decisive, dse = False, np.nan
        if runner is not None:                          # SE of the pointwise difference best vs runner
            mask = obs_case[best] == ci                 # same mask (same df) for the two models
            diff = loo_i[best][0][mask] - loo_i[runner][0][mask]
            if diff.size > 1:
                dse = float(np.std(diff, ddof=1) * np.sqrt(diff.size))
                decisive = bool(np.isfinite(dse) and tab.loc[1, "elpd_diff"] > se_mult * dse)
        if runner is None:
            decision = f"{best} (single model)"
        elif decisive:
            decision = f"{best} preferred (Delta-elpd {tab.loc[1,'elpd_diff']:.1f} > {se_mult}*dse {se_mult*dse:.1f})"
        else:
            decision = (f"indistinguishable (Delta-elpd {tab.loc[1,'elpd_diff']:.1f} <= {se_mult}*dse "
                        f"{se_mult*dse:.1f}) -> keep the simplest")
        per_patient[case] = {"best": best, "runner": runner, "decisive": decisive,
                             "decision": decision, "dse": dse, "table": tab}
    return {"per_patient": per_patient, "fitted": fitted, "cases": cases,
            "model_names": model_names}
