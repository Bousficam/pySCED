"""LONGITUDINAL multilevel models (general, non-SCED) - Phase 1: within-cluster COUPLING.

Coupling = effect of a focal time-varying covariate (``x``, e.g. EVA fatigue) on an
outcome (``y``, e.g. MI accuracy), AT EQUAL TIME (temporal confounder ``time``/block as a
covariate). Target = EFFECT ESTIMATION (pilot, small n of patients), not group NHST.

Four engines:
  - :func:`rmcorr` - repeated-measures correlation (Bakdash & Marusich 2017), simple, primary.
  - :func:`slope_vs_slope` - 2-step: per-cluster slopes then correlation of the slopes (dynamics).
  - :func:`fit_growth_lmm` - frequentist mixed model (statsmodels) + Satterthwaite/KR df via R (lmerTest).
  - :func:`fit_growth_bayes` - Bayesian mixed model (PyMC): partial pooling, gaussian/beta, pd/HDI/ROPE.

Phase 2 - frequentist GROWTH-CURVE (shape of the trajectory):
  - :func:`growth_terms` - time terms: discrete | linear | poly2 | spline.
  - :func:`add_within_between` - within/between decomposition of a time-varying covariate.
  - :func:`fit_growth_curve` - trajectory LMM (random intercept/slope, 2/3 levels) + ICC + R2.
  - :func:`compare_growth_models` - AIC/BIC/LRT selection (shapes x random structures).
  - :func:`marginal_trajectory` - marginal trajectory (emmeans-like) over a grid + CI (delta method).
  - :func:`blup_table` - per-subject random effects (BLUP) (caterpillar).

Phase 3 - BAYESIAN GROWTH-CURVE (PyMC), parity with the frequentist side:
  - :func:`fit_growth_bayes_curve` - Bayesian trajectory (poly/spline, random int/slope, 2/3
    levels, auto-scale priors, LOO/WAIC, pd/HDI/ROPE); fit-or-load cache + SCED diagnostics.
  - :func:`posterior_trajectory` - posterior marginal trajectory (median + HDI band).

Phase 4 - GAMM / GLMM / forecasting / power:
  - ``fit_growth_bayes_curve(shape="pspline")`` - GAMM (spline basis + smoothing penalty).
  - ``fit_growth_bayes_curve(family=...)`` - Bayesian GLMM: poisson | binomial | ordinal.
  - :func:`forecast_individual` - predicted trajectory of one subject (fixed effects + its random effects) + HDI.
  - :func:`power_growth` - simulation-based power (slope / interaction x time).

Reuses: Collinearity_Analysis.compute_vif_table (VIF), SCED_mbd_bayesian (_summarise/_rope_stats/_hdi),
SCED_model_cache.fit_or_load (fit-or-load cache).
"""
import numpy as np
import pandas as pd


# =========================================================================== #
#  A. rmcorr - repeated-measures correlation (Bakdash & Marusich 2017)
# =========================================================================== #
def rmcorr(df, *, x, y, subject):
    """Repeated-measures correlation: COMMON within-``subject`` slope of ``y`` on ``x``
    (ANCOVA ``y ~ x + C(subject)``), avoids pseudoreplication (vs stacked Pearson) and Simpson's
    paradox. ``subject`` = grouping unit of the coupling (here the SESSION).

    r_rm = sign(b_x)*sqrt(SS_x / (SS_x + SS_residual)); df = N - k - 1 (k = number of subjects);
    p = t-test of the ``x`` coefficient; 95% CI via Fisher transformation. Returns a dict.

    Note: ignores any higher-level nesting (e.g. sessions nested in patients): this is a primary
    estimate; the mixed model (1|patient/session) is its rigorous version.

    References: Bakdash & Marusich 2017 (repeated-measures correlation).
    R equivalent: rmcorr::rmcorr."""
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from scipy import stats
    d = df[[x, y, subject]].dropna().copy()
    d.columns = ["_x", "_y", "_s"]
    k = d["_s"].nunique()
    n = len(d)
    if k < 2 or n < k + 2:
        return {"available": False, "message": f"rmcorr: >= 2 subjects and N > k+1 required (n={n}, k={k})."}
    d["_s"] = d["_s"].astype("category")
    model = smf.ols("_y ~ C(_s) + _x", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=1)                  # SS type I: subject THEN x
    ss_x = float(anova.loc["_x", "sum_sq"])
    ss_res = float(anova.loc["Residual", "sum_sq"])
    df_res = int(model.df_resid)                             # = N - k - 1
    b_x = float(model.params["_x"])
    b_ci = model.conf_int().loc["_x"]
    r = float(np.sign(b_x) * np.sqrt(ss_x / (ss_x + ss_res))) if (ss_x + ss_res) > 0 else np.nan
    p = float(model.pvalues["_x"])
    # Fisher 95% CI (df = residual df)
    if np.isfinite(r) and abs(r) < 1 and df_res > 1:
        z = np.arctanh(r); se = 1.0 / np.sqrt(df_res - 1)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = np.nan
    return {"available": True, "r_rm": round(r, 4), "ci95": (round(lo, 4), round(hi, 4)),
            "df": df_res, "p": p, "slope": round(b_x, 5),
            "slope_ci": (round(float(b_ci[0]), 5), round(float(b_ci[1]), 5)),
            "n_obs": n, "k_subjects": k}


# =========================================================================== #
#  B. slope-vs-slope (2-step) - DYNAMICS coupling
# =========================================================================== #
def slope_vs_slope(df, *, x, y, time, cluster, min_points=3):
    """Per ``cluster`` (session): OLS slope of ``x`` on ``time`` and of ``y`` on ``time``;
    then correlation of the slopes ACROSS clusters (1 point/cluster -> independence OK).

    First checks the VARIANCE of the ``x`` slopes: if near zero (fatigue ~ monotone everywhere),
    the slope correlation is uninterpretable -> the analysis falls back on two one-sample t-tests
    (slope x != 0? and slope y != 0?). Returns a dict (+ table of the slopes).

    References: Matthews et al 1990 (two-stage / summary-measures analysis of serial data).
    R equivalent: no direct equivalent - per-cluster stats::lm then stats::cor.test on the slopes."""
    from scipy import stats
    rows = []
    for c, g in df[[x, y, time, cluster]].dropna().groupby(cluster):
        if g[time].nunique() < min_points:
            continue
        t = g[time].astype(float).to_numpy()
        sx = np.polyfit(t, g[x].astype(float).to_numpy(), 1)[0]
        sy = np.polyfit(t, g[y].astype(float).to_numpy(), 1)[0]
        rows.append({"cluster": c, "slope_x": sx, "slope_y": sy})
    tab = pd.DataFrame(rows)
    n = len(tab)
    out = {"available": n >= 3, "n_clusters": n, "slope_table": tab}
    if n < 3:
        out["message"] = "slope-vs-slope: >= 3 clusters with >= min_points timepoints required."
        return out
    var_sx = float(tab["slope_x"].var(ddof=1))
    out["var_slope_x"] = round(var_sx, 6)
    out["mean_slope_x"] = round(float(tab["slope_x"].mean()), 5)
    out["mean_slope_y"] = round(float(tab["slope_y"].mean()), 5)
    # one-sample t-tests (always reported: x increases? y decreases?)
    out["t_slope_x"] = {k: round(float(v), 5) for k, v in
                        zip(("t", "p"), stats.ttest_1samp(tab["slope_x"], 0.0))}
    out["t_slope_y"] = {k: round(float(v), 5) for k, v in
                        zip(("t", "p"), stats.ttest_1samp(tab["slope_y"], 0.0))}
    rng_x = float(tab["slope_x"].abs().mean()) or 1.0
    if var_sx < (0.05 * rng_x) ** 2:                        # variance of the x slopes too low
        out["warn"] = ("Variance of the x slopes near zero (fatigue ~ monotone) -> slope "
                       "correlation uninterpretable; rely on the one-sample t-tests.")
        out["r_slopes"] = np.nan
        return out
    r, p = stats.pearsonr(tab["slope_x"], tab["slope_y"])
    z = np.arctanh(r); se = 1.0 / np.sqrt(max(n - 3, 1))
    out.update({"r_slopes": round(float(r), 4), "p": round(float(p), 5),
                "ci95": (round(float(np.tanh(z - 1.96 * se)), 4),
                         round(float(np.tanh(z + 1.96 * se)), 4))})
    return out


# =========================================================================== #
#  shared helpers
# =========================================================================== #
def vif_table(df, terms):
    """VIF of the focal terms (reuses Collinearity_Analysis.compute_vif_table).

    References: Fox & Monette 1992 (generalized collinearity diagnostics / VIF).
    R equivalent: car::vif."""
    from functions.general.collinearity import compute_vif_table
    cols = [c for c in terms if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return pd.DataFrame({"Variable": cols, "VIF": [1.0] * len(cols)})
    vif = compute_vif_table(df[cols].dropna(), cols)
    return vif[vif["Variable"] != "const"].reset_index(drop=True)   # const = statsmodels artefact


def _r2_nakagawa(var_fixed, var_random, var_resid):
    """Marginal R2 (fixed effects only) and conditional R2 (fixed + random) - Nakagawa & Schielzeth 2013.

    References: Nakagawa & Schielzeth 2013 (marginal / conditional R2 for mixed models).
    R equivalent: MuMIn::r.squaredGLMM / performance::r2_nakagawa."""
    tot = var_fixed + var_random + var_resid
    if tot <= 0:
        return np.nan, np.nan
    return round(var_fixed / tot, 3), round((var_fixed + var_random) / tot, 3)


# =========================================================================== #
#  C. FREQUENTIST mixed model (statsmodels) + Satterthwaite/KR df via R
# =========================================================================== #
def fit_growth_lmm(df, *, outcome, fixed, group, group_l3=None, fixed_l3_threshold=6,
                   kr=True, vif_terms=None):
    """Gaussian linear mixed model (``statsmodels.MixedLM``): ``outcome ~ fixed + (1|group)``.
    ``fixed`` = list of patsy terms (e.g. ``["eva", "C(bloc)"]``). ``group`` = random-intercept
    cluster (e.g. session). ``group_l3`` (e.g. patient) enters as a FIXED EFFECT ``C(group_l3)`` if
    its number of levels < ``fixed_l3_threshold`` (variance unreliable at small n) - otherwise
    ignored here (3-level frequentist = Phase 2). Returns ``(res, summary)``.

    summary: fixed effects (coef, CI, Wald p; Satterthwaite/KR p if R+lmerTest), ICC, marginal/
    conditional R2 (Nakagawa), VIF of the focal terms (``vif_terms``).

    References: Laird & Ware 1982 (LMM); Satterthwaite 1946 / Kenward & Roger 1997 (df);
    Nakagawa & Schielzeth 2013 (R2).
    R equivalent: lme4::lmer + lmerTest / nlme::lme."""
    import statsmodels.formula.api as smf
    import warnings
    d = df.copy()
    terms = list(fixed)
    if group_l3 is not None and d[group_l3].nunique() < fixed_l3_threshold:
        terms.append(f"C({group_l3})")                      # patient as FIXED (small n)
    formula = f"{outcome} ~ " + " + ".join(terms)
    summary = {"Model": "Gaussian LMM (MixedLM)", "Formula": formula,
               "Random": f"(1 | {group})"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(formula, d, groups=d[group]).fit(reml=True)
    except Exception as exc:
        summary["Warning"] = f"MixedLM not fitted ({type(exc).__name__}: {exc})."
        return None, summary
    # fixed effects (Wald)
    ci = res.conf_int()
    fe = []
    for name in res.fe_params.index:
        fe.append({"term": name, "coef": round(float(res.fe_params[name]), 5),
                   "CI95": (round(float(ci.loc[name, 0]), 5), round(float(ci.loc[name, 1]), 5)),
                   "p_wald": round(float(res.pvalues.get(name, np.nan)), 5)})
    summary["fixed"] = fe
    # variance components + ICC + Nakagawa R2
    g_var = float(res.cov_re.iloc[0, 0]) if getattr(res, "cov_re", None) is not None else np.nan
    r_var = float(res.scale)
    summary["var_group"] = round(g_var, 5)
    summary["var_resid"] = round(r_var, 5)
    summary["ICC"] = round(g_var / (g_var + r_var), 3) if (g_var + r_var) > 0 else np.nan
    var_fixed = float(np.var(res.fittedvalues - res.resid, ddof=0))   # variance of the fixed-effect prediction
    summary["R2_marginal"], summary["R2_conditional"] = _r2_nakagawa(var_fixed, g_var, r_var)
    # VIF of the focal terms
    if vif_terms:
        summary["VIF"] = vif_table(d, vif_terms)
    # Satterthwaite/KR df via R (lmerTest) if available
    if kr:
        summary["KR"] = _r_lmer_kr(d, outcome=outcome, fixed_terms=terms, group=group)
    return res, summary


_R_LMER = r"""
args <- commandArgs(trailingOnly=TRUE)
csv <- args[1]; form <- args[2]
ok <- requireNamespace("lmerTest", quietly=TRUE)
if (!ok) { cat("RESULT_NA lmerTest absent\n"); quit(status=0) }
suppressMessages(library(lmerTest))
dat <- read.csv(csv)
m <- tryCatch(lmerTest::lmer(as.formula(form), data=dat, REML=TRUE),
              error=function(e){cat(paste0("RESULT_NA ", conditionMessage(e), "\n")); quit(status=0)})
co <- summary(m)$coefficients
df_method <- "Satterthwaite"
out <- list()
for (rn in rownames(co)) {
  out[[rn]] <- list(estimate=unname(co[rn,"Estimate"]),
                    df=unname(co[rn,"df"]),
                    p=unname(co[rn,"Pr(>|t|)"]))
}
cat(paste0("RESULT ", jsonlite::toJSON(list(method=df_method, coef=out), auto_unbox=TRUE), "\n"))
"""


def _r_lmer_kr(df, *, outcome, fixed_terms, group, timeout=120):
    """R bridge (lmerTest): Satterthwaite df + fixed-effect p for ``outcome ~ fixed + (1|group)``.
    Clean fallback ``{available: False, message}`` if Rscript/lmerTest/jsonlite are absent.

    References: Satterthwaite 1946 (approximate df); Kenward & Roger 1997 (KR df).
    R equivalent: lmerTest::lmer (Satterthwaite) / pbkrtest::KRmodcomp (Kenward-Roger)."""
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    if shutil.which("Rscript") is None:
        return {"available": False, "message": "Rscript not found - install R + lmerTest + jsonlite."}
    formula = f"{outcome} ~ " + " + ".join(fixed_terms) + f" + (1|{group})"
    tmpd = tempfile.mkdtemp()
    csv = os.path.join(tmpd, "d.csv"); df.to_csv(csv, index=False)
    rs = os.path.join(tmpd, "lmer.R")
    with open(rs, "w") as f:
        f.write(_R_LMER)
    try:
        res = subprocess.run(["Rscript", "--vanilla", rs, csv, formula],
                             capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return {"available": False, "message": f"R failed: {type(e).__name__}: {e}"}
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT")]
    if not lines or lines[-1].startswith("RESULT_NA"):
        msg = lines[-1][len("RESULT_NA "):] if lines else (res.stderr.strip()[:200] or "failure")
        return {"available": False, "message": f"lmerTest: {msg}"}
    try:
        r = json.loads(lines[-1][len("RESULT "):])
        r["available"] = True
        return r
    except Exception:
        return {"available": False, "message": "lmerTest: unreadable output."}


# =========================================================================== #
#  D. BAYESIAN mixed model (PyMC) - partial pooling, gaussian/beta, pd/HDI/ROPE
# =========================================================================== #
def fit_growth_bayes(df, *, outcome, x, cat=None, group, group_l3=None, family="gaussian",
                     bounds=None, rope="auto", draws=2000, tune=2000, chains=4, seed=42,
                     target_accept=0.95, cache_dir=None, force_refit=False, progressbar=False):
    """Bayesian COUPLING mixed model (PyMC):
    ``outcome ~ a + b_x*x + sum effects(cat) + u[group_l3] + u[group]`` (= (1|patient/session)).

    ``x`` = numeric focal covariate (eva). ``cat`` = categorical covariates (e.g. ["bloc"],
    reference coding). Random intercept effects on ``group`` (session) and ``group_l3``
    (patient) with prior SD ``student_t(3,0, 0.1*SD(outcome))`` (ON THE SCALE of the outcome).

    ``family``: ``"gaussian"`` (main) or ``"beta"`` (sensitivity, bounded outcome; ``bounds``
    = (lo,hi), default observed min/max). ``rope`` = ROPE on b_x in original units (gaussian),
    ``"auto"`` = 0.1*SD(outcome)/SD(x). Returns a dict (b_x: pd/HDI/ROPE; variances; idata).

    References: Gelman & Hill 2007 (Bayesian multilevel / partial pooling); Kruschke 2018 (ROPE).
    R equivalent: brms::brm / rstanarm::stan_glmer (+ bayestestR for pd/ROPE)."""
    import pymc as pm
    from functions.sced.mbd.bayesian import _summarise, _rope_stats
    from functions.sced.model_cache import fit_or_load, data_hash

    cat = list(cat or [])
    keep = [outcome, x, group] + cat + ([group_l3] if group_l3 else [])
    d = df[keep].dropna().copy()
    y_raw = d[outcome].astype(float).to_numpy()
    xv = d[x].astype(float).to_numpy()
    sd_y, sd_x = float(np.std(y_raw)), float(np.std(xv)) or 1.0
    sess_idx, sess_u = pd.factorize(d[group].astype(str))
    if group_l3:
        pat_idx, pat_u = pd.factorize(d[group_l3].astype(str))
    # categorical design (reference = first modality), per cat covariate
    cat_cols, cat_labels = [], []
    for c in cat:
        dummies = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True)
        cat_cols.append(dummies.to_numpy(dtype=float))
        cat_labels += list(dummies.columns)
    X_cat = np.hstack(cat_cols) if cat_cols else np.zeros((len(d), 0))

    is_beta = (family == "beta")
    if is_beta:
        lo, hi = bounds if bounds else (float(np.min(y_raw)), float(np.max(y_raw)))
        span = (hi - lo) or 1.0
        y = np.clip((y_raw - lo) / span, 1e-4, 1 - 1e-4)
    else:
        y = y_raw
    rope_val = (round(0.1 * sd_y / sd_x, 5) if rope == "auto" else rope) if rope is not None else None

    def _fit():
        with pm.Model() as _m:
            a = pm.Normal("a", 0.0, 2.5 * (1.0 if is_beta else sd_y))
            b_x = pm.Normal("b_x", 0.0, (2.5 if is_beta else 2.5 * sd_y / sd_x))
            mu = a + b_x * xv
            if X_cat.shape[1]:
                b_cat = pm.Normal("b_cat", 0.0, (2.5 if is_beta else 2.5 * sd_y),
                                  shape=X_cat.shape[1])
                mu = mu + pm.math.dot(X_cat, b_cat)
            tau_sess = pm.HalfStudentT("tau_session", nu=3, sigma=0.1 * (1.0 if is_beta else sd_y))
            z_sess = pm.Normal("z_session", 0.0, 1.0, shape=len(sess_u))
            mu = mu + (z_sess * tau_sess)[sess_idx]
            if group_l3:
                tau_pat = pm.HalfStudentT("tau_patient", nu=3, sigma=0.1 * (1.0 if is_beta else sd_y))
                z_pat = pm.Normal("z_patient", 0.0, 1.0, shape=len(pat_u))
                mu = mu + (z_pat * tau_pat)[pat_idx]
            if is_beta:
                phi = pm.HalfNormal("phi", 50.0)
                p = pm.math.invlogit(mu)
                pm.Beta("y_obs", alpha=p * phi, beta=(1 - p) * phi, observed=y)
            else:
                sigma = pm.HalfNormal("sigma", sd_y)
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                              target_accept=target_accept, progressbar=progressbar,
                              idata_kwargs={"log_likelihood": True})
        return idata

    spec = {"family": family, "x": x, "cat": "+".join(cat), "group": group, "l3": group_l3 or "",
            "draws": draws, "tune": tune, "chains": chains, "seed": seed,
            "dhash": data_hash(d, keep)}
    prefix = f"coupling__{outcome}__{x}__{family}"
    idata, _hit = fit_or_load(cache_dir, prefix, spec, _fit, force_refit=force_refit)

    post = idata.posterior
    bx = np.asarray(post["b_x"]).reshape(-1)
    res = {"family": family, "scale": ("logit" if is_beta else "original"),
           "b_x": {**_summarise(bx), "rope": rope_val}, "_idata": idata,
           "rhat_max": None}
    if rope_val is not None and not is_beta:
        res["b_x"].update(_rope_stats(bx, rope_val))
    if "b_cat" in post:
        bc = np.asarray(post["b_cat"])                      # (chain,draw,k)
        res["b_cat"] = {cat_labels[j]: _summarise(bc[..., j].reshape(-1)) for j in range(len(cat_labels))}
    res["variance"] = {"tau_session": _summarise(np.asarray(post["tau_session"]).reshape(-1))}
    if group_l3 and "tau_patient" in post:
        res["variance"]["tau_patient"] = _summarise(np.asarray(post["tau_patient"]).reshape(-1))
    try:
        import arviz as az
        res["rhat_max"] = float(az.rhat(idata).to_array().max())
        res["loo"] = float(az.loo(idata).elpd_loo)
    except Exception:
        pass
    return res


# =========================================================================== #
#  PHASE 2 - frequentist GROWTH-CURVE (shape of the trajectory)
# =========================================================================== #
def growth_terms(time, shape="linear", knots=4):
    """List of patsy terms for the SHAPE of time:
    ``"discrete"`` -> [C(time)]; ``"linear"`` -> [time]; ``"poly2"`` -> [time, I(time**2)];
    ``"spline"`` -> [bs(time, df=knots)] (cubic regression B-spline). Returns (terms, label).

    R equivalent: splines::bs (spline) / stats::poly (polynomial time terms)."""
    if shape == "discrete":
        return [f"C({time})"], "discrete time (factor)"
    if shape == "linear":
        return [time], "linear"
    if shape == "poly2":
        return [time, f"I({time}**2)"], "quadratic (poly2)"
    if shape == "spline":
        return [f"bs({time}, df={int(knots)})"], f"B-spline (df={int(knots)})"
    if shape == "pspline":                                   # spline basis + penalty (smoothing) on the Bayesian side
        return [f"bs({time}, df={int(knots)})"], f"P-spline (GAMM, df={int(knots)})"
    raise ValueError(f"unknown shape '{shape}' (discrete|linear|poly2|spline|pspline).")


def add_within_between(df, var, subject):
    """WITHIN/BETWEEN decomposition of a time-varying covariate (per-subject centering):
    ``{var}_between`` = per-subject mean (between-subject, cross-sectional effect); ``{var}_within`` =
    deviation from that mean (within-subject, purely longitudinal effect). Returns a COPY of df.

    References: Neuhaus & Kalbfleisch 1998 (between- vs within-cluster covariate effects).
    R equivalent: datawizard::demean / parameters::demean."""
    d = df.copy()
    m = d.groupby(subject)[var].transform("mean")
    d[f"{var}_between"] = m
    d[f"{var}_within"] = d[var] - m
    return d


def _re_formula(random, time):
    """Map a random-effects keyword to the patsy re_formula string.

    ``random`` in {intercept, int_slope, int_slope_corr} -> ``None`` (random intercept only)
    or ``"~{time}"`` (add a random slope on time). Returns the re_formula string or ``None``."""
    return {"intercept": None, "int_slope": f"~{time}", "int_slope_corr": f"~{time}"}.get(random, None)


def _n_params(res):
    """Number of estimated parameters (fixed effects + (co)variance components + scale)."""
    k_fe = len(res.fe_params)
    k_re = int(res.cov_re.shape[0] * (res.cov_re.shape[0] + 1) / 2) if getattr(res, "cov_re", None) is not None else 0
    _vc = getattr(res, "vcomp", None)
    k_vc = len(_vc) if _vc is not None else 0
    return k_fe + k_re + k_vc + 1


def fit_growth_curve(df, *, outcome, time, shape="linear", knots=4, by=None, predictors=None,
                     group, group_l3=None, random="int_slope", reml=True, kr=False):
    """TRAJECTORY LMM: ``outcome ~ f(time) [+ by*f(time)] [+ predictors] + (random | group)``.

    ``shape``: shape of time (see :func:`growth_terms`). ``by``: time-invariant predictor (e.g. arm)
    -> main effect + interactions with every time term (differential trajectories).
    ``predictors``: additional time-invariant covariates (main effects). ``random``:
    ``intercept`` | ``int_slope`` (random slope on ``time``, correlated with the intercept). ``group`` =
    level-2 (subject). ``group_l3`` (cluster): additional level-3 random intercept (via
    vc_formula; no level-3 random slope here). ``reml``: REML (False to compare SHAPES, see
    :func:`compare_growth_models`). Returns ``(res, summary)``.

    References: Laird & Ware 1982 (growth-curve mixed model); Nakagawa & Schielzeth 2013 (R2).
    R equivalent: nlme::lme / lme4::lmer."""
    import statsmodels.formula.api as smf
    import warnings
    tterms, tlabel = growth_terms(time, shape, knots)
    terms = list(tterms)
    if by is not None:
        terms.append(f"C({by})")
        terms += [f"C({by}):{t}" for t in tterms]
    for p in (predictors or []):
        terms.append(p)
    formula = f"{outcome} ~ " + " + ".join(terms)
    re_formula = _re_formula(random, time)
    summary = {"Model": "Growth LMM", "Formula": formula, "shape": shape, "shape_label": tlabel,
               "Random": f"({'1' if re_formula is None else '1 + ' + time} | {group})"
                         + (f" + (1 | {group_l3})" if group_l3 else "")}
    kw = {}
    if group_l3 is not None:
        kw["vc_formula"] = {group: f"0 + C({group})"}        # subject intercept nested in the cluster
        groups = df[group_l3]
        summary["Random"] = f"(1 | {group_l3}) + (1 | {group_l3}:{group})"
    else:
        groups = df[group]
        if re_formula is not None:
            kw["re_formula"] = re_formula
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(formula, df, groups=groups, **kw).fit(reml=reml)
    except Exception as exc:
        summary["Warning"] = f"MixedLM not fitted ({type(exc).__name__}: {exc})."
        return None, summary
    summary["fixed"] = [{"term": n, "coef": round(float(res.fe_params[n]), 5),
                         "CI95": (round(float(res.conf_int().loc[n, 0]), 5),
                                  round(float(res.conf_int().loc[n, 1]), 5)),
                         "p_wald": round(float(res.pvalues.get(n, np.nan)), 5)}
                        for n in res.fe_params.index]
    g_var = float(res.cov_re.iloc[0, 0]) if getattr(res, "cov_re", None) is not None and res.cov_re.size else 0.0
    _vc = getattr(res, "vcomp", None)
    vc_var = float(np.sum(_vc)) if (_vc is not None and len(_vc)) else 0.0
    r_var = float(res.scale)
    re_tot = g_var + vc_var
    summary.update({"var_random": round(re_tot, 5), "var_resid": round(r_var, 5),
                    "ICC": round(re_tot / (re_tot + r_var), 3) if (re_tot + r_var) > 0 else np.nan})
    var_fixed = float(np.var(res.fittedvalues - res.resid, ddof=0))
    summary["R2_marginal"], summary["R2_conditional"] = _r2_nakagawa(var_fixed, re_tot, r_var)
    summary["llf"] = float(res.llf); summary["k_params"] = _n_params(res); summary["n_obs"] = int(res.nobs)
    summary["AIC"] = round(-2 * res.llf + 2 * summary["k_params"], 2)
    summary["BIC"] = round(-2 * res.llf + summary["k_params"] * np.log(res.nobs), 2)
    summary["reml"] = reml
    if kr:
        rand = "1" if re_formula is None else f"{time}"
        summary["KR"] = _r_lmer_kr(df, outcome=outcome, fixed_terms=terms, group=group, random=rand)
    return res, summary


def compare_growth_models(df, *, outcome, time, group, group_l3=None,
                          shapes=("linear", "poly2", "spline"), random="int_slope",
                          by=None, predictors=None, knots=4):
    """Selection of the SHAPE of time: fits each ``shape`` in **ML** (REML not comparable on
    fixed effects) and returns an AIC/BIC/logLik/df table + **LRT** vs the previous model (nested
    linear subset of poly2). The best AIC is flagged. Returns a DataFrame sorted by AIC.

    References: Akaike 1974 (AIC); Schwarz 1978 (BIC); Wilks 1938 (likelihood-ratio test).
    R equivalent: stats::AIC / stats::BIC / stats::anova (LRT on nested lme4 fits)."""
    from scipy import stats
    rows, prev = [], None
    for sh in shapes:
        res, s = fit_growth_curve(df, outcome=outcome, time=time, shape=sh, knots=knots, by=by,
                                  predictors=predictors, group=group, group_l3=group_l3,
                                  random=random, reml=False)
        if res is None:
            rows.append({"shape": sh, "AIC": np.nan, "BIC": np.nan, "logLik": np.nan,
                         "k": np.nan, "note": s.get("Warning", "failure")})
            continue
        row = {"shape": s["shape_label"], "AIC": s["AIC"], "BIC": s["BIC"],
               "logLik": round(s["llf"], 2), "k": s["k_params"]}
        if prev is not None and prev["k"] < s["k_params"]:               # nested LRT
            lr = 2 * (s["llf"] - prev["llf"]); ddf = s["k_params"] - prev["k"]
            row["LRT_p"] = round(float(stats.chi2.sf(max(lr, 0), ddf)), 5) if ddf > 0 else np.nan
        rows.append(row); prev = {"k": s["k_params"], "llf": s["llf"]}
    tab = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    if tab["AIC"].notna().any():
        tab["best"] = (tab["AIC"] == tab["AIC"].min())
    return tab


def marginal_trajectory(res, df, *, time, shape, by=None, knots=4, grid_n=60):
    """Marginal trajectory (population mean, emmeans-like): prediction of the FIXED effects over
    a grid of ``time`` (x levels of ``by``), 95% CI via the delta method (X*cov_fe*X^T).
    The other covariates are set to their mean (numeric) / first modality (categorical).
    Returns a DataFrame ``time[, by], pred, lo, hi``.

    References: Searle, Speed & Milliken 1980 (population marginal / estimated marginal means).
    R equivalent: emmeans::emmeans / ggeffects::ggpredict."""
    import patsy
    di = res.model.data.design_info
    if shape == "discrete":
        tg = np.sort(df[time].dropna().unique())
    else:
        tg = np.linspace(float(df[time].min()), float(df[time].max()), grid_n)
    by_levels = sorted(df[by].dropna().unique(), key=str) if by is not None else [None]
    # reference values of the other columns present in the design
    base = {}
    for c in df.columns:
        if c in (time, by):
            continue
        base[c] = df[c].mean() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]
    fe = res.fe_params.values
    cov = np.asarray(res.cov_params())[:len(fe), :len(fe)]
    rows = []
    for lev in by_levels:
        g = pd.DataFrame({time: tg})
        for c, v in base.items():
            g[c] = v
        if by is not None:
            g[by] = lev
        X = np.asarray(patsy.build_design_matrices([di], g, return_type="matrix")[0])
        pred = X @ fe
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, cov, X), 0.0))
        for i, t in enumerate(tg):
            r = {time: float(t), "pred": float(pred[i]),
                 "lo": float(pred[i] - 1.96 * se[i]), "hi": float(pred[i] + 1.96 * se[i])}
            if by is not None:
                r[by] = lev
            rows.append(r)
    return pd.DataFrame(rows)


def blup_table(res, *, group, term="intercept"):
    """BLUP (conditional random effects) per subject, sorted - for the caterpillar.
    ``term`` = label of the random-effect column ('intercept' = first column, or slope name).
    Returns a DataFrame ``group, blup`` sorted by blup.

    References: Robinson 1991 (BLUP / estimation of random effects).
    R equivalent: lme4::ranef / nlme::ranef."""
    re = res.random_effects
    rows = []
    for g, vals in re.items():
        if term == "intercept":
            v = float(vals.iloc[0])
        else:
            v = float(vals.get(term, np.nan)) if hasattr(vals, "get") else float(vals[term])
        rows.append({"group": g, "blup": v})
    return pd.DataFrame(rows).sort_values("blup").reset_index(drop=True)


# =========================================================================== #
#  PHASE 3 - BAYESIAN GROWTH-CURVE (PyMC): parity with the frequentist side
# =========================================================================== #
def _growth_formula_rhs(time, shape, knots, by, predictors):
    """Shared patsy RHS (freq/bayes): time terms [+ by*time] [+ predictors], WITHOUT intercept marker."""
    tterms, tlabel = growth_terms(time, shape, knots)
    terms = list(tterms)
    if by is not None:
        terms.append(f"C({by})")
        terms += [f"C({by}):{t}" for t in tterms]
    terms += list(predictors or [])
    return " + ".join(terms), tlabel


def fit_growth_bayes_curve(df, *, outcome, time, shape="linear", knots=4, by=None, predictors=None,
                           group, group_l3=None, random="int_slope", family="gaussian", n_trials=None,
                           rope="auto", draws=2000, tune=2000, chains=4, seed=42, target_accept=0.95,
                           cache_dir=None, force_refit=False, progressbar=False):
    """BAYESIAN growth curve (PyMC) - parity with :func:`fit_growth_curve`:
    ``outcome ~ f(time) [+ by*f(time)] [+ predictors] + (1[+time] | subject) [+ (1|cluster)]``.

    ``shape``: ... | ``"pspline"`` = **GAMM** (spline basis + smoothing PENALTY: shared SD
    ``tau_smooth`` learned over the basis coefficients). ``family``: ``gaussian`` | ``poisson`` |
    ``binomial`` (``n_trials`` required) | ``ordinal`` (ordered logistic, cutpoints; no intercept).
    AUTO-SCALE priors; random intercept effects (+ slope on centered time) + cluster.
    Returns a dict (fixed effects pd/HDI[/ROPE], variances, LOO/WAIC, idata, design_info).

    References: Gelman & Hill 2007 (Bayesian multilevel); Eilers & Marx 1996 (P-splines / GAMM).
    R equivalent: brms::brm / rstanarm::stan_glmer (brms::s() for the P-spline term)."""
    import patsy
    import pymc as pm
    import pytensor.tensor as pt
    from functions.sced.mbd.bayesian import _summarise, _rope_stats
    from functions.sced.model_cache import fit_or_load, data_hash

    rhs, tlabel = _growth_formula_rhs(time, shape, knots, by, predictors)
    keep = list({outcome, time, group} | ({by} if by else set()) | ({group_l3} if group_l3 else set())
                | set(c for c in df.columns if c in rhs))
    d = df.dropna(subset=[outcome, time, group]).copy()
    is_ordinal = (family == "ordinal")
    Xdf = patsy.dmatrix(("0 + " + rhs) if is_ordinal else rhs, d, return_type="dataframe")
    design_info = Xdf.design_info                             # ordinal: no intercept (the cutpoints absorb it)
    colnames = list(Xdf.columns)
    X = Xdf.to_numpy()
    yv = d[outcome].astype(float).to_numpy()
    sd_y, mean_y = float(np.std(yv)) or 1.0, float(np.mean(yv))
    tc = d[time].astype(float).to_numpy(); tc = tc - tc.mean()    # centered time (random slope)
    subj_idx, subj_u = pd.factorize(d[group].astype(str))
    has_slope = (random == "int_slope")
    if group_l3:
        cl_idx, cl_u = pd.factorize(d[group_l3].astype(str))
    spline_idx = np.array([j for j, c in enumerate(colnames) if c.startswith("bs(")], dtype=int)
    col_sd = X.std(axis=0); col_sd[col_sd < 1e-8] = 1.0
    is_intercept = np.array([c.lower().startswith("intercept") for c in colnames])
    # prior scale by family (gaussian = outcome unit; otherwise = link scale)
    if family == "gaussian":
        prior_sd = np.where(is_intercept, 5.0 * sd_y, 2.5 * sd_y / col_sd)
        prior_mu = np.where(is_intercept, mean_y, 0.0)
        re_scale = 0.1 * sd_y
    else:
        if family == "poisson":
            link0 = float(np.log(max(mean_y, 0.5)))
        elif family == "binomial":
            nt = (d[n_trials].to_numpy() if isinstance(n_trials, str) else
                  np.full(len(d), float(n_trials if n_trials else 1)))
            link0 = float(np.log((mean_y + 0.5) / (np.mean(nt) - mean_y + 0.5)))
        else:
            link0 = 0.0
        prior_sd = np.where(is_intercept, 3.0, 2.5 / col_sd)
        prior_mu = np.where(is_intercept, link0, 0.0)
        re_scale = 0.5
    if is_ordinal:
        y_int, y_cats = pd.factorize(d[outcome])             # 0..K-1
        n_cat = len(y_cats)

    def _fit():
        with pm.Model() as _m:
            sig = pt.as_tensor_variable(prior_sd)
            if shape == "pspline" and len(spline_idx):       # GAMM: smoothing penalty
                tau_s = pm.HalfStudentT("tau_smooth", nu=3, sigma=(2.5 * sd_y if family == "gaussian" else 2.5))
                sig = pt.set_subtensor(sig[spline_idx], tau_s)
            beta = pm.Normal("beta", mu=prior_mu, sigma=sig, shape=X.shape[1])
            eta = pm.math.dot(X, beta)
            tau0 = pm.HalfStudentT("tau_subject", nu=3, sigma=re_scale)
            eta = eta + (pm.Normal("z0", 0.0, 1.0, shape=len(subj_u)) * tau0)[subj_idx]
            if has_slope:
                tau1 = pm.HalfStudentT("tau_slope", nu=3, sigma=re_scale)
                eta = eta + (pm.Normal("z1", 0.0, 1.0, shape=len(subj_u)) * tau1)[subj_idx] * tc
            if group_l3:
                tauC = pm.HalfStudentT("tau_cluster", nu=3, sigma=re_scale)
                eta = eta + (pm.Normal("zC", 0.0, 1.0, shape=len(cl_u)) * tauC)[cl_idx]
            if family == "gaussian":
                sigma = pm.HalfNormal("sigma", sd_y)
                pm.Normal("y_obs", mu=eta, sigma=sigma, observed=yv)
            elif family == "poisson":
                pm.Poisson("y_obs", mu=pm.math.exp(eta), observed=yv)
            elif family == "binomial":
                pm.Binomial("y_obs", n=nt, p=pm.math.invlogit(eta), observed=yv)
            elif family == "ordinal":
                cut = pm.Normal("cutpoints", 0.0, 5.0, shape=n_cat - 1,
                                transform=pm.distributions.transforms.ordered,
                                initval=np.linspace(-2, 2, n_cat - 1))
                pm.OrderedLogistic("y_obs", eta=eta, cutpoints=cut, observed=y_int, compute_p=False)
            else:
                raise ValueError(f"unknown family '{family}'.")
            return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                             target_accept=target_accept, progressbar=progressbar,
                             idata_kwargs={"log_likelihood": not is_ordinal})  # OrderedLogistic: log_lik not supported here

    spec = {"rhs": rhs, "random": random, "group": group, "l3": group_l3 or "", "shape": shape,
            "family": family, "draws": draws, "tune": tune, "chains": chains, "seed": seed,
            "dhash": data_hash(d, keep)}
    idata, _hit = fit_or_load(cache_dir, f"growthbayes__{outcome}__{shape}__{family}", spec, _fit,
                              force_refit=force_refit)

    post = idata.posterior
    bdraws = np.asarray(post["beta"]).reshape(-1, X.shape[1])
    rope_val = (round(0.1 * sd_y, 4) if rope == "auto" else rope) if (rope is not None and family == "gaussian") else None
    fixed = {}
    for j, c in enumerate(colnames):
        s = _summarise(bdraws[:, j])
        if rope_val is not None and (time in c) and ("Intercept" not in c):
            s.update(_rope_stats(bdraws[:, j], rope_val)); s["rope"] = rope_val
        fixed[c] = s
    out = {"family": family, "shape": shape, "shape_label": tlabel, "fixed": fixed,
           "variance": {"tau_subject": _summarise(np.asarray(post["tau_subject"]).reshape(-1))},
           "_idata": idata, "_design_info": design_info, "_colnames": colnames,
           "_time": time, "_by": by, "_shape": shape, "_knots": knots, "_ref": d,
           "_group": group, "_subj_u": list(subj_u), "_has_slope": has_slope,
           "_tc_mean": float(d[time].astype(float).mean())}
    if family == "gaussian":
        out["variance"]["sigma"] = _summarise(np.asarray(post["sigma"]).reshape(-1))
    if shape == "pspline" and "tau_smooth" in post:
        out["variance"]["tau_smooth"] = _summarise(np.asarray(post["tau_smooth"]).reshape(-1))
    if has_slope:
        out["variance"]["tau_slope"] = _summarise(np.asarray(post["tau_slope"]).reshape(-1))
    if group_l3:
        out["variance"]["tau_cluster"] = _summarise(np.asarray(post["tau_cluster"]).reshape(-1))
    try:
        import arviz as az
        out["rhat_max"] = float(az.rhat(idata).to_array().max())
        out["loo"] = float(az.loo(idata).elpd_loo)
        out["waic"] = float(az.waic(idata).elpd_waic)
    except Exception:
        out["rhat_max"] = None
    return out


def posterior_trajectory(bayes_res, *, grid_n=60, hdi=0.95):
    """POSTERIOR marginal trajectory (population mean, random effects integrated out at 0):
    median + HDI band per point of the time grid x levels of ``by``. Reuses the stored patsy
    ``design_info`` -> splines/poly reconstructed identically. Returns a DataFrame
    ``time[, by], pred, lo, hi``.

    R equivalent: brms::posterior_epred / tidybayes::add_epred_draws."""
    import patsy
    from functions.sced.mbd.bayesian import _hdi
    di = bayes_res["_design_info"]; time = bayes_res["_time"]; by = bayes_res["_by"]
    shape = bayes_res["_shape"]; d = bayes_res["_ref"]
    bdraws = np.asarray(bayes_res["_idata"].posterior["beta"]).reshape(-1, len(bayes_res["_colnames"]))
    if shape == "discrete":
        tg = np.sort(d[time].dropna().unique())
    else:
        tg = np.linspace(float(d[time].min()), float(d[time].max()), grid_n)
    base = {c: (d[c].mean() if pd.api.types.is_numeric_dtype(d[c]) else d[c].mode().iloc[0])
            for c in d.columns if c not in (time, by)}
    by_levels = sorted(d[by].dropna().unique(), key=str) if by is not None else [None]
    rows = []
    for lev in by_levels:
        g = pd.DataFrame({time: tg})
        for c, v in base.items():
            g[c] = v
        if by is not None:
            g[by] = lev
        Xg = np.asarray(patsy.build_design_matrices([di], g, return_type="matrix")[0])
        Yg = Xg @ bdraws.T                                   # (n_grid, n_draws)
        med = np.median(Yg, axis=1)
        for i, t in enumerate(tg):
            lo, hii = _hdi(Yg[i], hdi)
            r = {time: float(t), "pred": float(med[i]), "lo": float(lo), "hi": float(hii)}
            if by is not None:
                r[by] = lev
            rows.append(r)
    return pd.DataFrame(rows)


# =========================================================================== #
#  PHASE 4 - individual forecasting + power planning
# =========================================================================== #
def _inv_link(eta, family):
    """Apply the inverse link of ``family`` to a linear predictor ``eta``.

    poisson -> exp(eta), binomial -> logistic(eta), otherwise identity (gaussian).
    Returns the response-scale array."""
    if family == "poisson":
        return np.exp(eta)
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    return eta


def forecast_individual(bayes_res, *, subject, horizon=None, grid_n=40, hdi=0.95, response=True):
    """PREDICTED trajectory of one subject (fixed effects + ITS random effects) over a time grid
    extended up to ``horizon`` (default = observed max). Median + HDI band (epistemic uncertainty).
    ``response=True`` applies the inverse link (outcome scale). Returns a DataFrame
    ``time, pred, lo, hi`` (on the response scale if ``response``).

    R equivalent: brms::posterior_epred(re_formula = NULL) / tidybayes::add_epred_draws."""
    import patsy
    from functions.sced.mbd.bayesian import _hdi
    di = bayes_res["_design_info"]; time = bayes_res["_time"]; by = bayes_res["_by"]
    d = bayes_res["_ref"]; group = bayes_res["_group"]; subj_u = bayes_res["_subj_u"]
    fam = bayes_res["family"]; post = bayes_res["_idata"].posterior
    if str(subject) not in [str(s) for s in subj_u]:
        raise ValueError(f"subject '{subject}' absent ({len(subj_u)} subjects).")
    sidx = [str(s) for s in subj_u].index(str(subject))
    sub = d[d[group].astype(str) == str(subject)]
    t_max = float(d[time].max())
    t0, t1 = float(d[time].min()), float(horizon if horizon is not None else t_max)
    if bayes_res["_shape"] in ("spline", "pspline", "discrete") and t1 > t_max:
        t1 = t_max                                           # splines/discrete do not extrapolate out of range
    tg = np.linspace(t0, t1, grid_n)
    base = {c: (sub[c].iloc[0] if c in sub and len(sub) else
                (d[c].mean() if pd.api.types.is_numeric_dtype(d[c]) else d[c].mode().iloc[0]))
            for c in d.columns if c not in (time,)}
    g = pd.DataFrame({time: tg})
    for c, v in base.items():
        g[c] = v
    if by is not None and by in sub:
        g[by] = sub[by].iloc[0]
    Xg = np.asarray(patsy.build_design_matrices([di], g, return_type="matrix")[0])
    beta = np.asarray(post["beta"]).reshape(-1, Xg.shape[1])              # (draws, ncol)
    eta = Xg @ beta.T                                                     # (grid, draws)
    tau0 = np.asarray(post["tau_subject"]).reshape(-1)
    z0 = np.asarray(post["z0"]).reshape(-1, len(subj_u))[:, sidx]
    eta = eta + (z0 * tau0)[None, :]
    if bayes_res["_has_slope"] and "z1" in post:
        tau1 = np.asarray(post["tau_slope"]).reshape(-1)
        z1 = np.asarray(post["z1"]).reshape(-1, len(subj_u))[:, sidx]
        eta = eta + ((z1 * tau1)[None, :]) * (tg[:, None] - bayes_res["_tc_mean"])
    vals = _inv_link(eta, fam) if response else eta
    rows = []
    for i, t in enumerate(tg):
        lo, hii = _hdi(vals[i], hdi)
        rows.append({time: float(t), "pred": float(np.median(vals[i])), "lo": float(lo), "hi": float(hii)})
    return pd.DataFrame(rows)


def power_growth(*, n_subj, n_times, beta_time, sd_resid=1.0, sd_intercept=1.0, sd_slope=0.0,
                 by_effect=None, n_sim=200, alpha=0.05, seed=0):
    """SIMULATION-based power for a LINEAR growth curve. Generates ``n_sim`` datasets
    (``n_subj`` subjects x ``n_times`` timepoints 0..T-1) under: y = b0_i + (``beta_time`` + b1_i)*t
    [+ ``by_effect``*arm*t] + eps, with b0_i~N(0,``sd_intercept``), b1_i~N(0,``sd_slope``),
    eps~N(0,``sd_resid``); fits ``fit_growth_curve`` (linear, random int+slope) and counts the
    replications where the target effect (``time`` slope or ``arm:time`` interaction) is significant
    (Wald CI excludes 0). Returns ``{power, target_effect, n_sim, ...}``.

    References: Green & MacLeod 2016 (simulation-based power for GLMMs, SIMR).
    R equivalent: simr::powerSim / simr::powerCurve."""
    rng = np.random.RandomState(seed)
    target = "arm:t" if by_effect is not None else "t"
    hits = 0; ok = 0
    for _ in range(n_sim):
        rows = []
        for s in range(n_subj):
            b0 = rng.normal(0, sd_intercept); b1 = rng.normal(0, sd_slope)
            arm = s % 2 if by_effect is not None else 0
            for t in range(n_times):
                y = b0 + (beta_time + b1) * t + (by_effect or 0.0) * arm * t + rng.normal(0, sd_resid)
                row = {"subj": f"S{s}", "t": t, "y": y}
                if by_effect is not None:
                    row["arm"] = f"arm{arm}"
                rows.append(row)
        df = pd.DataFrame(rows)
        res, summ = fit_growth_curve(df, outcome="y", time="t", shape="linear",
                                     by=("arm" if by_effect is not None else None),
                                     group="subj", random="int_slope", reml=True)
        if res is None:
            continue
        ok += 1
        fe = {f["term"]: f for f in summ["fixed"]}
        key = next((k for k in fe if (target == "t" and k == "t") or (target == "arm:t" and ":" in k and "t" in k
                    and ("arm" in k.lower()))), None)
        if key is not None:
            lo, hi = fe[key]["CI95"]
            if lo > 0 or hi < 0:
                hits += 1
    return {"power": round(hits / ok, 3) if ok else np.nan, "target_effect": target,
            "beta_time": beta_time, "by_effect": by_effect, "n_subj": n_subj, "n_times": n_times,
            "n_sim_ok": ok}
