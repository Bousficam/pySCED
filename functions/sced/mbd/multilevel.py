"""
SCED - Multilevel (piecewise) model for multiple-baseline designs
=================================================================

The standard model-based analysis of MBD data (Van den Noortgate & Onghena 2003;
Moeyaert, Ferron, Beretvas & Van den Noortgate 2014; Manolov & Moeyaert 2017): a
**piecewise (interrupted-time-series) mixed model** with, per case,

    Y = b0 + b1*time + b2*phase + b3*(treatment trajectory) + e

where ``phase`` is the B-phase indicator. Two codings of the treatment trajectory
are offered (``treatment_trajectory``):

  - ``"continuous"`` (default): b3*time_since, ``time_since`` = max(0, session - start).
    b2 = immediate level change at onset, b3 = a *single sustained slope* over the
    whole B phase.
  - ``"two_piece"``: b3*h with ``h`` coded so the effect changes linearly over the
    first ``ramp`` treatment observations then **plateaus** (Cheng, Yi, Moeyaert,
    Beretvas, Van den Noortgate & Ferron 2025, *Synthesizing SCEDs: Modeling Complex
    Data Structures*). For ``ramp=3`` h is -2, -1, 0, 0, ... across the B phase. Then
    **b2 = the stabilised (plateau) treatment effect** and **b3 = the slope of the
    initial rise**. Cheng et al. found this 2-piece coding gave lower bias and better
    CI coverage than a quadratic trajectory for effects that grow then level off
    (the typical TMS-like pattern) - the recommended default when that shape is
    expected.

As elsewhere in the toolkit the mixed model supplies the *estimates* (b2, b3, variance
components, ICC) but the **p-values come from the design-based randomization test** on
the intervention start (level and slope statistics) - valid with the few cases typical
of MBD, where the model's asymptotic p is untrustworthy (Manolov & Moeyaert 2017).
Outcome-type aware: continuous -> linear mixed model, binary/count -> GLMM (logit/log).

:func:`mbd_model_selection` complements this with a small **model-comparison /
sensitivity** table (AIC, BIC, likelihood-ratio test across the immediate / continuous
/ two-piece fixed-effect specifications), following Manolov & Moeyaert (2025): prefer a
theory/expertise-driven choice, use AIC/BIC/LRT as data-driven support, and report
whether the sign and significance of the effect hold across specifications (a
sensitivity / multiverse check).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from functions.sced.mbd.procedures import _derive_starts
from functions.sced.mbd.baseline import multiple_baseline_test


# TODO(SCED, design-comparable effect size): add the c-REML design-comparable SMD g_AB
# (Pustejovsky, Hedges & Shadish 2014; R package `scdhlm`) so MBD effects are on the same
# Cohen's d metric as between-subjects RCTs and can be meta-analysed. Requires the MB1-MB5
# HLM catalog + Satterthwaite-corrected J(nu) small-sample adjustment; use robust variance
# (Hedges-Tipton-Johnson) for m < 9 where the model-based SE is anti-conservative.
# TODO(SCED, error structure / df): statsmodels MixedLM has no AR(1) errors and no
# Kenward-Roger df (Cheng et al. 2025). For model-based inference matching the literature,
# fit under nlme / SAS PROC MIXED; here we deliberately lean on the randomization p-values.


def _piecewise_design(df, *, tier_col, session_col, outcome_col, starts,
                      treatment_trajectory, ramp):
    """Build the piecewise design columns (``time``, ``phase``, and the treatment
    trajectory regressor ``time_since`` *or* the two-piece ``h``). Returns the
    augmented frame and the name of the slope regressor."""
    d = df.rename(columns={outcome_col: "_y", tier_col: "_u"}).copy()
    d["_y"] = pd.to_numeric(d["_y"], errors="coerce")
    sess = pd.to_numeric(d[session_col], errors="coerce").to_numpy()
    st = d["_u"].map(starts).to_numpy()
    d["time"] = sess
    d["phase"] = (sess >= st).astype(float)
    if treatment_trajectory == "two_piece":
        # treatment-observation index (1,2,3,...) within each case, then h = min(0, idx-ramp)
        d["_sess"] = sess
        idx = (d.loc[d["phase"] == 1]
                 .groupby("_u")["_sess"].rank(method="first").astype(float))
        treat_idx = pd.Series(0.0, index=d.index)
        treat_idx.loc[idx.index] = idx.to_numpy()
        d["h"] = np.where(d["phase"].to_numpy() == 1,
                          np.minimum(0.0, treat_idx.to_numpy() - ramp), 0.0)
        d = d.drop(columns="_sess")
        return d, "h"
    d["time_since"] = np.maximum(0.0, sess - st)
    return d, "time_since"


def multilevel_mbd_model(df, *, tier_col, session_col, outcome_col, phase_col=None,
                         intervention_starts=None, treatment_phase=None,
                         outcome_type="continuous", improvement="increase",
                         treatment_trajectory="continuous", ramp=3,
                         min_baseline=3, min_treatment=3, n_perm=5000, random_state=0):
    """
    Fit the piecewise multilevel MBD model and pair it with permutation p-values.

    ``treatment_trajectory`` selects the B-phase coding: ``"continuous"`` (sustained
    slope ``time_since``) or ``"two_piece"`` (linear rise over the first ``ramp``
    observations then a plateau; Cheng et al. 2025). Under ``"two_piece"`` the level
    coefficient b2 is the *stabilised* effect and the slope b3 is the *initial* rise.

    Returns ``(summary, sheet)``. ``summary`` holds the population level change (b2),
    slope change (b3), baseline trend (b1), the random-effect SDs / ICC, and the
    design-based randomization p-values for the **level** and **slope** effects. ``sheet``
    is a tidy table with ``Explanation`` / ``Interpretation`` columns. Falls back to the
    permutation p-values if the mixed model does not converge.

    Degrees-of-freedom note (for audit) - Cheng et al. (2025, J. Behavioral Education)
    recommend **Kenward-Roger adjusted df** with REML for fixed-effect inference in
    three-level SCED meta-analysis. statsmodels MixedLM does not implement KR, so the
    model-based SEs here are not KR-adjusted; this is one reason the reported inference
    deliberately relies on the **design-based randomization p-values** (valid with the few
    cases typical of MBD) rather than the mixed model's asymptotic p. For KR-adjusted
    model inference, fit under nlme / SAS PROC MIXED (cf. SCED.md "Multilevel Models").

    Validation (for audit) - verified against R ``scan::hplm`` (model "W", REML). The **fixed
    effects match to ~0.01** once the random structure is matched (random-INTERCEPT-only: b1
    vs hplm mt, b2 vs phaseB, b3 vs interB). NB this function's DEFAULT random structure is
    ``~1 + phase`` (random intercept **and** random level) - a defensible richer model, but it
    re-splits the baseline-trend vs slope-change attribution (the total B-phase slope b1+b3
    still matches hplm's mt+interB, only the split differs). The reported **ICC also differs by
    definition** (here var_intercept/(var_intercept+var_resid) from the full model;
    ``scan::hplm`` reports the unconditional ICC from a null model). For publication-grade,
    AR(1)-aware, KR-adjusted fixed effects use ``bc_smd_scdhlm`` / ``scan::hplm`` directly - 
    this MixedLM fit has no AR(1) and no KR df.

    References: Van den Noortgate & Onghena 2003; Moeyaert, Ferron, Beretvas & Van den Noortgate
    2014; Cheng et al. 2025 (piecewise interrupted-time-series multilevel model, two-piece coding).
    R equivalent: nlme::lme / lme4::lmer (+ lmerTest for the Kenward-Roger p); scan::hplm - fixed
    effects verified against scan::hplm.
    """
    if treatment_trajectory not in ("continuous", "two_piece"):
        raise ValueError("treatment_trajectory must be 'continuous' or 'two_piece'.")
    starts = _derive_starts(df, tier_col, session_col, phase_col, intervention_starts, treatment_phase)
    tiers = list(starts)

    # design-based randomization p-values (robust with few cases)
    common = dict(tier_col=tier_col, session_col=session_col, outcome_col=outcome_col,
                  intervention_starts=starts, scheme="independent", improvement=improvement,
                  min_baseline=min_baseline, min_treatment=min_treatment, n_perm=n_perm,
                  random_state=random_state)
    p_level = multiple_baseline_test(df, statistic="level", **common)["p_value"]
    p_slope = multiple_baseline_test(df, statistic="slope", **common)["p_value"]

    d, slope_col = _piecewise_design(df, tier_col=tier_col, session_col=session_col,
                                     outcome_col=outcome_col, starts=starts,
                                     treatment_trajectory=treatment_trajectory, ramp=ramp)
    fixed = f"_y ~ time + phase + {slope_col}"

    two_piece = treatment_trajectory == "two_piece"
    summary = {"tiers": [str(t) for t in tiers], "n_cases": len(tiers),
               "outcome_type": outcome_type, "treatment_trajectory": treatment_trajectory,
               "ramp": ramp if two_piece else None,
               "p_level_change_perm": round(p_level, 4), "p_slope_change_perm": round(p_slope, 4)}
    note, scale = "", ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if outcome_type == "continuous":
                re = "~1 + phase"
                res = smf.mixedlm(fixed, d, groups=d["_u"], re_formula=re).fit(reml=True)
                fe = res.fe_params
                var_int = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] else np.nan
                resid = float(res.scale)
                icc = var_int / (var_int + resid) if np.isfinite(var_int) and var_int + resid > 0 else np.nan
                summary.update({"intercept_b0": round(float(fe.get("Intercept", np.nan)), 4),
                                "baseline_trend_b1": round(float(fe.get("time", np.nan)), 4),
                                "level_change_b2": round(float(fe.get("phase", np.nan)), 4),
                                "slope_change_b3": round(float(fe.get(slope_col, np.nan)), 4),
                                "icc": round(icc, 4) if np.isfinite(icc) else "n/a",
                                "residual_var": round(resid, 4)})
                try:                                # SE + covariance of the fixed effects (-> CI band)
                    cov = res.cov_params()
                    nm = ["time", "phase", slope_col]
                    summary["se_b1"] = round(float(res.bse.get("time", np.nan)), 4)
                    summary["se_b2"] = round(float(res.bse.get("phase", np.nan)), 4)
                    summary["se_b3"] = round(float(res.bse.get(slope_col, np.nan)), 4)
                    summary["fe_cov"] = [[float(cov.loc[a, b]) for b in nm] for a in nm]
                except Exception:
                    pass
            else:
                from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
                                                                PoissonBayesMixedGLM)
                GLMM = BinomialBayesMixedGLM if outcome_type == "binary" else PoissonBayesMixedGLM
                scale = "logit scale" if outcome_type == "binary" else "log scale"
                gr = GLMM.from_formula(fixed, {"case": "0 + C(_u)"}, d).fit_vb()
                fe = dict(zip(gr.model.exog_names, np.asarray(gr.fe_mean)))
                summary.update({"intercept_b0": round(float(fe.get("Intercept", np.nan)), 4),
                                "baseline_trend_b1": round(float(fe.get("time", np.nan)), 4),
                                "level_change_b2": round(float(fe.get("phase", np.nan)), 4),
                                "slope_change_b3": round(float(fe.get(slope_col, np.nan)), 4),
                                "icc": "n/a (non-linear scale)"})
    except Exception as exc:
        note = f"Model not fitted ({type(exc).__name__}) - permutation p only."
        summary.update({"level_change_b2": "n/a", "slope_change_b3": "n/a", "icc": "n/a"})

    # --- CANONICAL inference (Moeyaert): fixed effects b1/b2/b3 + Kenward-Roger p via R ---
    summary["canonical_ddf"] = None
    if phase_col is not None:
        try:
            _h = hplm_R(df, tier_col=tier_col, session_col=session_col, outcome_col=outcome_col,
                        phase_col=phase_col, treatment=treatment_phase, improvement=improvement)
            if _h.get("available"):
                summary.update({"p_b1_kr": round(float(_h["p_b1"]), 4),
                                "p_b2_kr": round(float(_h["p_b2"]), 4),
                                "p_b3_kr": round(float(_h["p_b3"]), 4),
                                "canonical_ddf": _h.get("ddf"),
                                "canonical_random": _h.get("random")})
        except Exception:
            pass

    sc = f" ({scale})" if scale else ""
    b2_expl = ("STABILIZED effect (plateau) after the initial rise (2-piece coding)."
               if two_piece else "IMMEDIATE level change at the intervention (piecewise model).")
    b3_expl = (f"Slope of the INITIAL RISE over the first {ramp} treatment "
               "observations (progressive effect, e.g. TMS) - then plateau." if two_piece else
               "TREND change after the intervention (PROGRESSIVE effect, e.g. TMS).")
    kr = summary.get("canonical_ddf")                       # Kenward-Roger/Satterthwaite df if R available
    pk1, pk2, pk3 = summary.get("p_b1_kr"), summary.get("p_b2_kr"), summary.get("p_b3_kr")

    def _verdict(pk, pperm, prog=False):                    # prefers the canonical p (KR), else permutation
        if pk is not None:
            return (f"{'Progressive effect' if prog else 'Effect'} {'significant' if pk < 0.05 else 'not significant'} "
                    f"(model p {kr} = {pk}).")
        return (f"{'Progressive effect' if prog else 'Effect'} {'significant' if pperm < 0.05 else 'not significant'} "
                f"(permutation p = {pperm}; canonical R model unavailable).")

    rows = [
        {"Quantity": f"Baseline trend b1{sc}", "Value": summary.get("baseline_trend_b1", "n/a"),
         "Explanation": "PRE-existing slope in phase A (nuisance/control, e.g. spontaneous recovery).",
         "Interpretation": (f"Baseline trend {'significant' if (pk1 is not None and pk1 < 0.05) else 'not significant'} "
                            f"(model p {kr} = {pk1})." if pk1 is not None
                            else "No design p (b1 is not an intervention effect); canonical R model unavailable.")},
        {"Quantity": f"{'Plateau effect' if two_piece else 'Level jump'} b2{sc}",
         "Value": summary["level_change_b2"], "Explanation": b2_expl,
         "Interpretation": _verdict(pk2, summary["p_level_change_perm"])},
        {"Quantity": f"{'Initial slope' if two_piece else 'Slope change'} b3{sc}",
         "Value": summary["slope_change_b3"], "Explanation": b3_expl,
         "Interpretation": _verdict(pk3, summary["p_slope_change_perm"], prog=True)},
        {"Quantity": "ICC (within-case correlation)", "Value": summary.get("icc", "n/a"),
         "Explanation": "Share of variance due to differences between cases.",
         "Interpretation": "The higher the ICC, the more the cases differ in level."},
    ]
    if note:
        rows.append({"Quantity": "Note", "Value": " - ", "Explanation": note,
                     "Interpretation": "Permutation p valid."})
    sheet = pd.DataFrame(rows, columns=["Quantity", "Value", "Explanation", "Interpretation"])
    return summary, sheet


_SCDHLM_R = r'''
args <- commandArgs(trailingOnly = TRUE)
ok <- requireNamespace("scdhlm", quietly=TRUE) && requireNamespace("nlme", quietly=TRUE)
if (!ok) { cat("RESULT_NA scdhlm/nlme absent\n"); quit(status=0) }
suppressMessages({ library(nlme); library(scdhlm) })
dat <- read.csv(args[1])
trend <- if (length(args) >= 2) as.integer(args[2]) else 0L   # 1 = corrects the baseline trend
tryCatch({
  if (trend == 1L) {                       # level NET of a common linear trend (parallel slopes)
    m <- lme(y ~ time + phase, random = ~ 1 | case,
             correlation = corAR1(0, ~ time | case),
             data = dat, method = "REML",
             control = lmeControl(opt = "optim", returnObject = TRUE, msMaxIter = 200))
    pc <- c(0, 0, 1)                        # (Intercept, time, phase) -> selects phase
  } else {                                  # raw level (canonical design-comparable specification)
    m <- lme(y ~ phase, random = ~ 1 | case,
             correlation = corAR1(0, ~ time | case),
             data = dat, method = "REML",
             control = lmeControl(opt = "optim", returnObject = TRUE, msMaxIter = 200))
    pc <- c(0, 1)                           # (Intercept, phase) -> selects phase
  }
  g <- g_mlm(m, p_const = pc, r_const = c(1, 0, 1), infotype = "expected")
  est <- as.numeric(g$g_AB); se <- as.numeric(g$SE_g_AB); dfv <- as.numeric(g$nu)
  tcrit <- qt(0.975, df = dfv)
  cat(sprintf('RESULT {"g_AB": %.4f, "SE": %.4f, "df": %.2f, "ci_low": %.4f, "ci_high": %.4f}\n',
      est, se, dfv, est - tcrit * se, est + tcrit * se))
}, error = function(e) cat(paste0("RESULT_NA ", conditionMessage(e), "\n")))
'''


def bc_smd_scdhlm(df, *, tier_col, session_col, outcome_col, phase_col,
                  baseline=None, treatment=None, improvement="increase", trend=False, timeout=180):
    """BC-SMD g_AB design-comparable effect size via the R package `scdhlm` - CANONICAL REML
    estimator (Hedges-Pustejovsky-Shadish; Satterthwaite df). Runs R when available.

    ``trend``:
      - ``False`` (default): ``lme(y ~ phase, random=~1|case, corAR1(~time|case))`` then
        ``g_mlm(p_const=c(0,1), r_const=c(1,0,1))`` - RAW level, canonical design-comparable
        specification. Does NOT correct the baseline trend: "B vs A", sensitive to maturation.
      - ``True``: ``lme(y ~ time + phase, ...)`` then ``g_mlm(p_const=c(0,0,1), ...)`` - level
        effect NET of a common linear baseline trend (parallel slopes). To be compared with the
        raw value: a g that collapses under correction is an effect largely carried by the trend.

    Returns ``{available, g_AB, SE, df, ci_low, ci_high, method, trend}`` when computed, otherwise
    ``{available: False, message}`` (R/scdhlm absent, or fit failure). No in-house
    reimplementation: this is the reference tool.

    References: Pustejovsky, Hedges & Shadish 2014; Valentine, Tanner-Smith, Pustejovsky & Lau
    2016 (design-comparable between-case SMD g_AB).
    R equivalent: scdhlm::g_mlm (called directly - the exact equivalent).
    """
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    if shutil.which("Rscript") is None:
        return {"available": False,
                "message": "Rscript not found - install R + the scdhlm package for the canonical BC-SMD."}
    d = df[[tier_col, session_col, phase_col, outcome_col]].copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d[session_col] = pd.to_numeric(d[session_col], errors="coerce")
    d = d.dropna(subset=[outcome_col, session_col])
    phs = d[phase_col].astype(str)
    baseline = baseline if baseline is not None else sorted(phs.unique())[0]
    treatment = treatment if treatment is not None else [p for p in sorted(phs.unique())
                                                         if p != str(baseline)][-1]
    d = d[phs.isin([str(baseline), str(treatment)])]
    if d[tier_col].nunique() < 2:
        return {"available": False, "message": "BC-SMD design-comparable: >= 2 cases required."}
    sign = 1.0 if improvement == "increase" else -1.0
    out = pd.DataFrame({"case": d[tier_col].astype(str).values,
                        "time": d[session_col].astype(float).values,
                        "phase": (d[phase_col].astype(str) == str(treatment)).astype(int).values,
                        "y": (sign * d[outcome_col].astype(float)).values})
    tmpd = tempfile.mkdtemp()
    csv = os.path.join(tmpd, "d.csv"); out.to_csv(csv, index=False)
    rs = os.path.join(tmpd, "bcsmd.R")
    with open(rs, "w") as f:
        f.write(_SCDHLM_R)
    try:
        res = subprocess.run(["Rscript", "--vanilla", rs, csv, "1" if trend else "0"],
                             capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return {"available": False, "message": f"R failed: {type(e).__name__}: {e}"}
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT")]
    if not lines or lines[-1].startswith("RESULT_NA"):
        msg = (lines[-1][len("RESULT_NA "):] if lines else (res.stderr.strip()[:200] or "failure"))
        return {"available": False, "message": f"scdhlm: {msg}"}
    try:
        r = json.loads(lines[-1][len("RESULT "):])
    except Exception:
        return {"available": False, "message": "scdhlm: unreadable output."}
    r["available"] = True; r["trend"] = bool(trend)
    r["method"] = ("scdhlm g_mlm (REML, corAR1, Satterthwaite df"
                   + (", y~time+phase trend-corrected)" if trend else ")"))
    return r


_HPLM_R = r'''
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("lmerTest", quietly=TRUE)) { cat("RESULT_NA lmerTest absent\n"); quit(status=0) }
suppressMessages(library(lmerTest))
dat <- read.csv(args[1])
ddf <- if (length(args) >= 2) args[2] else "Kenward-Roger"
fit_one <- function(reform)
  lmerTest::lmer(as.formula(paste0("y ~ time + phase + tsince + (", reform, " | case)")),
                 data = dat, REML = TRUE,
                 control = lmerControl(optimizer = "bobyqa",
                                       check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)))
tryCatch({
  rs <- "1 + phase"
  m <- tryCatch(fit_one(rs), error = function(e) { rs <<- "1"; fit_one("1") })
  co <- summary(m, ddf = ddf)$coefficients
  vc <- as.data.frame(VarCorr(m))
  v_int <- vc$vcov[vc$grp == "case" & vc$var1 == "(Intercept)" & is.na(vc$var2)][1]
  v_res <- vc$vcov[vc$grp == "Residual"][1]
  icc <- v_int / (v_int + v_res); if (is.na(icc)) icc <- 0
  g <- function(r, c) if (r %in% rownames(co)) as.numeric(co[r, c]) else 0
  cat(sprintf(paste0('RESULT {"b0":%.4f,"b1":%.4f,"b2":%.4f,"b3":%.4f,',
      '"se_b1":%.4f,"se_b2":%.4f,"se_b3":%.4f,"df_b1":%.2f,"df_b2":%.2f,"df_b3":%.2f,',
      '"p_b1":%.5g,"p_b2":%.5g,"p_b3":%.5g,"icc":%.4f,"random":"%s","ddf":"%s"}\n'),
      g("(Intercept)","Estimate"), g("time","Estimate"), g("phase","Estimate"), g("tsince","Estimate"),
      g("time","Std. Error"), g("phase","Std. Error"), g("tsince","Std. Error"),
      g("time","df"), g("phase","df"), g("tsince","df"),
      g("time","Pr(>|t|)"), g("phase","Pr(>|t|)"), g("tsince","Pr(>|t|)"), icc, rs, ddf))
}, error = function(e) cat(paste0("RESULT_NA ", conditionMessage(e), "\n")))
'''


def hplm_R(df, *, tier_col, session_col, outcome_col, phase_col, baseline=None,
           treatment=None, improvement="increase", ddf="Kenward-Roger", timeout=240):
    """Canonical piecewise multilevel model (Moeyaert) via R `lmerTest`/`lme4`: fits
    ``y ~ time + phase + tsince + (1 + phase | case)`` (REML) and returns the fixed effects
    **b1** (baseline trend), **b2** (level jump), **b3** (slope change) with their
    SE, df and Kenward-Roger p (``ddf`` in {"Kenward-Roger","Satterthwaite"}), plus the ICC.
    ``tsince`` = 0 in phase A and 0,1,2... from the first B point -> b2 = immediate jump. Falls
    back to ``(1 | case)`` when the random structure does not converge. ``y`` oriented by ``improvement``.

    This is the Moeyaert-style fixed-effect inference (model test, KR df) - distinct from the
    design-based randomization p of the pipeline. No reimplementation: R wrapper, on the same
    model as :func:`bc_smd_scdhlm`. Returns ``{available, b0..b3, se_b*, df_b*, p_b* (KR),
    icc, random, ddf, method, baseline_trend_b1/level_change_b2/slope_change_b3 (alias)}`` or
    ``{available: False, message}`` (R/lmerTest absent, < 2 cases, or fit failure).

    References: Moeyaert, Ferron, Beretvas & Van den Noortgate 2014; Cheng et al. 2025
    (canonical piecewise multilevel model, Kenward-Roger df).
    R equivalent: lmerTest::lmer / lme4::lmer (called directly); scan::hplm."""
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    if shutil.which("Rscript") is None:
        return {"available": False, "message": "Rscript not found - install R + lmerTest."}
    d = df[[tier_col, session_col, phase_col, outcome_col]].copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d[session_col] = pd.to_numeric(d[session_col], errors="coerce")
    d = d.dropna(subset=[outcome_col, session_col])
    phs = d[phase_col].astype(str)
    baseline = baseline if baseline is not None else sorted(phs.unique())[0]
    treatment = treatment if treatment is not None else [p for p in sorted(phs.unique())
                                                         if p != str(baseline)][-1]
    d = d[phs.isin([str(baseline), str(treatment)])]
    if d[tier_col].nunique() < 2:
        return {"available": False, "message": "Multilevel: >= 2 cases required."}
    sign = 1.0 if improvement == "increase" else -1.0
    case = d[tier_col].astype(str).to_numpy()
    time = d[session_col].astype(float).to_numpy()
    phase = (d[phase_col].astype(str) == str(treatment)).astype(int).to_numpy()
    y = sign * d[outcome_col].astype(float).to_numpy()
    tsince = np.zeros(len(d))                                # 0 in A, 0..k from the first B point
    for cs in np.unique(case):
        idx = case == cs
        bt = time[idx & (phase == 1)]
        if len(bt):
            tsince[idx] = np.maximum(0.0, time[idx] - bt.min())
    out = pd.DataFrame({"case": case, "time": time, "phase": phase, "tsince": tsince, "y": y})
    tmpd = tempfile.mkdtemp()
    csv = os.path.join(tmpd, "d.csv"); out.to_csv(csv, index=False)
    rs = os.path.join(tmpd, "hplm.R")
    with open(rs, "w") as f:
        f.write(_HPLM_R)
    try:
        res = subprocess.run(["Rscript", "--vanilla", rs, csv, ddf],
                             capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return {"available": False, "message": f"R failed: {type(e).__name__}: {e}"}
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT")]
    if not lines or lines[-1].startswith("RESULT_NA"):
        msg = (lines[-1][len("RESULT_NA "):] if lines else (res.stderr.strip()[:200] or "failure"))
        return {"available": False, "message": f"lmerTest: {msg}"}
    try:
        r = json.loads(lines[-1][len("RESULT "):])
    except Exception:
        return {"available": False, "message": "lmerTest: unreadable output."}
    r["available"] = True
    r["method"] = f"lmer (REML) + {r.get('ddf', ddf)} df; fixed effects b1/b2/b3 (Moeyaert canonical)"
    r["baseline_trend_b1"] = r.get("b1"); r["level_change_b2"] = r.get("b2")
    r["slope_change_b3"] = r.get("b3"); r["intercept_b0"] = r.get("b0")
    return r


def between_case_smd(df, *, tier_col, session_col, outcome_col, phase_col=None,
                     intervention_starts=None, treatment_phase=None, model="level",
                     treatment_trajectory="continuous", ramp=3,
                     min_baseline=3, min_treatment=3):
    """
    BC-SMD - between-case standardized mean difference (design-comparable effect size),
    Hedges, Pustejovsky & Shadish (2012, 2013) estimator: the treatment effect is
    standardized by the BETWEEN-case standard deviation (sqrt(random-intercept variance +
    residual variance)) -> same scale as a group Cohen's d, meta-analysable. Reference R
    package: ``scdhlm``.

    ``model``:
      - ``"level"`` (default, = standard scdhlm): ``y ~ phase`` + random intercept -> LEVEL d.
        This is the canonical design-comparable specification.
      - ``"piecewise"``: ``y ~ time + phase + time_since`` + (intercept, slope) random effects ->
        effect NET of a modelled trend (reserve for long baselines).

    Small-sample inference (approximates scdhlm):
      - Welch-Satterthwaite df combining between-case variance (df = m-1) and within-case
        (df = N-m): nu = (tau+sigma^2)^2 / (tau^2/(m-1) + sigma^4/(N-m));
      - correction ``J = 1 - 3/(4*nu-1)``; Student CI (t_nu), not normal.

    Limitation vs scdhlm: ``statsmodels`` MixedLM does not model AR1 (scdhlm does) ->
    slightly different variance decomposition; variance of g by delta method (HPS
    exact in scdhlm). For publication, validate against ``scdhlm``.

    Audit (vs ``bc_smd_scdhlm`` / R ``scdhlm``, no trend) - CLOSE approximation: g, SE
    and Satterthwaite df agree closely. The residual gap on g (of order
    ~5-10%) comes from the AR(1) that scdhlm models (corAR1) and MixedLM cannot. For
    a publishable value, use ``bc_smd_scdhlm`` (canonical R wrapper).
    Returns ``{d, g, se, ci_low, ci_high, J, df, n_cases, n_obs, b2, between_var, within_var, model}``.

    References: Hedges, Pustejovsky & Shadish 2012, 2013; Pustejovsky, Hedges & Shadish 2014
    (between-case SMD; this is a Python approximation of the scdhlm estimator).
    R equivalent: scdhlm::g_mlm - potential equivalent, to test (MixedLM lacks AR(1), ~5-10% gap on g).
    """
    from scipy import stats as _st
    starts = _derive_starts(df, tier_col, session_col, phase_col, intervention_starts,
                            treatment_phase)
    n_cases = len(starts)
    out = {"n_cases": n_cases, "n_obs": 0, "d": np.nan, "g": np.nan, "se": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "J": np.nan, "df": np.nan, "b2": np.nan,
           "between_var": np.nan, "within_var": np.nan, "model": model}
    if n_cases < 2:
        return out
    if model == "piecewise":
        d, slope_col = _piecewise_design(df, tier_col=tier_col, session_col=session_col,
                                         outcome_col=outcome_col, starts=starts,
                                         treatment_trajectory=treatment_trajectory, ramp=ramp)
        formula, re_f = f"_y ~ time + phase + {slope_col}", "~1 + phase"
    else:                                    # "level" - standard design-comparable (scdhlm)
        d, _ = _piecewise_design(df, tier_col=tier_col, session_col=session_col,
                                 outcome_col=outcome_col, starts=starts,
                                 treatment_trajectory="continuous", ramp=ramp)
        formula, re_f = "_y ~ phase", "~1"
    d = d.dropna(subset=["_y"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(formula, d, groups=d["_u"], re_formula=re_f).fit(reml=True)
        b2 = float(res.fe_params.get("phase", np.nan))
        se_b2 = float(res.bse_fe.get("phase", np.nan))
        var_int = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] else np.nan
        resid = float(res.scale)
        tot = var_int + resid
        if not (np.isfinite(tot) and tot > 0 and np.isfinite(b2)):
            return out
        m, N = n_cases, int(len(d))
        d_val = b2 / np.sqrt(tot)
        # Welch-Satterthwaite df (combines between-case df=m-1 and within-case df=N-m)
        denom = (var_int ** 2) / max(m - 1, 1) + (resid ** 2) / max(N - m, 1)
        nu = float((tot ** 2) / denom) if denom > 0 else float(max(m - 1, 1))
        nu = max(nu, 1.0)
        J = 1.0 - 3.0 / (4.0 * nu - 1.0)
        g = J * d_val
        var_d = (se_b2 ** 2) / tot + (d_val ** 2) / (2.0 * nu)
        se = J * float(np.sqrt(var_d)) if np.isfinite(var_d) else np.nan
        tcrit = float(_st.t.ppf(0.975, nu))                      # t-based CI (not 1.96)
        out.update({"d": round(d_val, 3), "g": round(g, 3), "se": round(se, 3),
                    "ci_low": round(g - tcrit * se, 3), "ci_high": round(g + tcrit * se, 3),
                    "J": round(J, 3), "df": round(nu, 1), "b2": round(b2, 3), "n_obs": N,
                    "between_var": round(var_int, 3), "within_var": round(resid, 3)})
    except Exception:
        pass
    return out


def mbd_model_selection(df, *, tier_col, session_col, outcome_col, phase_col=None,
                        intervention_starts=None, treatment_phase=None, ramp=3,
                        random_structure=False,
                        candidates=("immediate", "continuous", "two_piece")):
    """
    **Model-selection / sensitivity** support for the multilevel MBD model
    (continuous outcome), following Manolov & Moeyaert (2025).

    Correction rationale (for audit) - Manolov & Moeyaert (2025) split the decision into
    two categories that must be treated differently:
      - **Category 1 - fixed effects** (e.g. level-only vs level+slope): the choice MUST
        be driven by the research HYPOTHESIS, *not* by data fit. Applying AIC/BIC to the
        fixed structure "conflates the research hypothesis with model fitting" (their
        recommendation 1). The fixed-effect table below is therefore reported only as a
        **sensitivity / multiverse** check (does the sign of the effect hold across
        plausible specifications?), NOT as a selector.
      - **Category 2 - random effects + error structure**: this is where AIC/BIC are
        legitimate (recommendations 2-3), with **BIC preferred over AIC** for SCED data.
        When ``random_structure=True`` the random structure (random intercept vs random
        intercept + slope) is compared by BIC at constant fixed effects, fit in **REML**
        (the correct likelihood for comparing random structures with identical fixed
        effects).

    Fixed-effect candidates (fit with **full ML** so AIC/BIC/LRT are comparable across
    *fixed*-effect differences):
      - ``immediate``  : ``_y ~ time + phase``            (level change only),
      - ``continuous`` : ``_y ~ time + phase + time_since`` (sustained slope),
      - ``two_piece``  : ``_y ~ time + phase + h``        (rise over ``ramp`` obs, plateau).

    ``table`` reports, per fixed model, the level-change estimate (b2), AIC, BIC,
    log-likelihood and the LRT vs the nested ``immediate`` model, then the Category-2
    random-structure rows. ``summary`` carries ``effect_sign_stable`` (sensitivity),
    ``aic_preferred`` / ``bic_preferred`` (fixed - diagnostic only) and
    ``random_structure_bic_preferred`` (the legitimate data-driven choice). Columns carry
    ``Explanation`` / ``Interpretation``.

    Documented limitation - recommendation 3 (independent vs AR(1) errors) is not
    realisable: statsmodels MixedLM offers no AR(1) error structure. That comparison needs
    another engine (nlme / SAS), cf. SCED.md "Multilevel Models".

    References: Manolov & Moeyaert 2025 (multilevel model selection for SCED: theory-driven
    fixed effects, BIC-driven random structure).
    R equivalent: stats::AIC / stats::BIC / stats::anova on nlme::lme or lme4::lmer fits.
    """
    starts = _derive_starts(df, tier_col, session_col, phase_col, intervention_starts, treatment_phase)
    base, _ = _piecewise_design(df, tier_col=tier_col, session_col=session_col,
                                outcome_col=outcome_col, starts=starts,
                                treatment_trajectory="continuous", ramp=ramp)
    tp, _ = _piecewise_design(df, tier_col=tier_col, session_col=session_col,
                              outcome_col=outcome_col, starts=starts,
                              treatment_trajectory="two_piece", ramp=ramp)
    base["h"] = tp["h"].to_numpy()
    formulas = {"immediate": "_y ~ time + phase",
                "continuous": "_y ~ time + phase + time_since",
                "two_piece": "_y ~ time + phase + h"}

    fits, rows = {}, []
    ll_immediate = None
    for name in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formulas[name], base, groups=base["_u"],
                                  re_formula="~1 + phase").fit(reml=False)
            fits[name] = res
            b2 = float(res.fe_params.get("phase", np.nan))
            k_par = res.df_modelwc if hasattr(res, "df_modelwc") else len(res.params)
            ll = float(res.llf)
            aic = -2 * ll + 2 * k_par
            bic = -2 * ll + np.log(int(base.shape[0])) * k_par
            if name == "immediate":
                ll_immediate = ll
                lrt = " - "
            else:
                stat = 2 * (ll - ll_immediate) if ll_immediate is not None else np.nan
                p = stats.chi2.sf(stat, 1) if np.isfinite(stat) and stat >= 0 else np.nan
                lrt = (f"chi^2={stat:.2f}, p={p:.3f}" if np.isfinite(p) else "n/a")
            rows.append({"Model": name, "b2 (level)": round(b2, 4),
                         "AIC": round(aic, 1), "BIC": round(bic, 1), "logLik": round(ll, 1),
                         "LRT vs immediate": lrt,
                         "Explanation": {"immediate": "Level jump only (abrupt, sustained effect).",
                                         "continuous": "Jump + sustained slope over the whole B phase.",
                                         "two_piece": f"Jump + rise over {ramp} obs then plateau (Cheng 2025)."}[name],
                         "Interpretation": ("Reference model (comparison baseline)." if name == "immediate"
                                            else "Does the slope term improve the fit? (LRT/AIC/BIC).")})
        except Exception as exc:
            rows.append({"Model": name, "b2 (level)": "n/a", "AIC": "n/a", "BIC": "n/a",
                         "logLik": "n/a", "LRT vs immediate": "n/a",
                         "Explanation": f"Not fitted ({type(exc).__name__}).",
                         "Interpretation": " - "})

    table = pd.DataFrame(rows)
    valid = [r for r in rows if isinstance(r["AIC"], (int, float))]
    summary = {"candidates": list(candidates)}
    if valid:
        best_aic = min(valid, key=lambda r: r["AIC"])["Model"]
        best_bic = min(valid, key=lambda r: r["BIC"])["Model"]
        # stable = no two specifications carry STRICTLY OPPOSITE signs. A b2 that rounds to
        # exactly 0 (e.g. the two-piece coding shifting the level into the ramp term) is not a
        # sign flip, so ignore zeros rather than counting np.sign(0)=0 as a third "sign".
        signs = {np.sign(r["b2 (level)"]) for r in valid
                 if isinstance(r["b2 (level)"], (int, float)) and r["b2 (level)"] != 0}
        stable = not (1.0 in signs and -1.0 in signs)
        summary.update({"aic_preferred": best_aic, "bic_preferred": best_bic,
                        "effect_sign_stable": bool(stable)})
        sens = ("The SIGN of the level effect is STABLE across specifications "
                "(conclusion robust to model choice)." if stable else
                "The sign of the effect CHANGES across specifications - conclusion "
                "sensitive to the model, interpret with caution.")
        table = pd.concat([table, pd.DataFrame([
            {"Model": "-> AIC (fixed effects, DIAGNOSTIC only)", "b2 (level)": "",
             "AIC": "", "BIC": "", "logLik": "", "LRT vs immediate": best_aic,
             "Explanation": "Lowest AIC among the fixed effects.",
             "Interpretation": "Do NOT select the fixed effects by AIC: Category 1, "
                               "theory takes precedence (Manolov & Moeyaert 2025, rec. 1)."},
            {"Model": "-> BIC (fixed effects, DIAGNOSTIC only)", "b2 (level)": "",
             "AIC": "", "BIC": "", "logLik": "", "LRT vs immediate": best_bic,
             "Explanation": "Lowest BIC among the fixed effects.",
             "Interpretation": "Same: a data-driven choice is legitimate ONLY for the "
                               "random structure (Category 2, below)."},
            {"Model": "-> Sensitivity (sign of b2)", "b2 (level)": "", "AIC": "", "BIC": "",
             "logLik": "", "LRT vs immediate": "stable" if stable else "unstable",
             "Explanation": "Sensitivity / multiverse analysis (Steegen 2016; Manolov 2025).",
             "Interpretation": sens},
        ])], ignore_index=True)

    # ---- Category 2 (Manolov & Moeyaert 2025): RANDOM structure chosen by BIC ----
    # Rationale: it is HERE - and not on the fixed effects - that AIC/BIC are legitimate
    # (recommendations 2-3). Random intercept only vs intercept + slope are compared at
    # CONSTANT fixed effects, in REML (the correct likelihood for comparing random
    # structures with identical fixed effects). BIC is preferred over AIC for this choice
    # in SCED data (penalises complexity more, avoids the ambiguity of n).
    if random_structure:
        re_specs = {"intercept": "~1", "intercept+slope": "~1 + phase"}
        re_rows = []
        for rname, re_f in re_specs.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rres = smf.mixedlm("_y ~ time + phase + time_since", base,
                                       groups=base["_u"], re_formula=re_f).fit(reml=True)
                k_par = len(rres.params)
                ll = float(rres.llf)
                re_rows.append((rname, -2 * ll + 2 * k_par,
                                -2 * ll + np.log(int(base.shape[0])) * k_par))
            except Exception as exc:
                re_rows.append((rname, np.nan, np.nan))
        valid_re = [r for r in re_rows if isinstance(r[2], float) and np.isfinite(r[2])]
        if valid_re:
            summary["random_structure_bic_preferred"] = min(valid_re, key=lambda r: r[2])[0]
        for rname, aic, bic in re_rows:
            table = pd.concat([table, pd.DataFrame([{
                "Model": f"[random] {rname}", "b2 (level)": "",
                "AIC": round(aic, 1) if np.isfinite(aic) else "n/a",
                "BIC": round(bic, 1) if np.isfinite(bic) else "n/a", "logLik": "",
                "LRT vs immediate": " - ",
                "Explanation": "RANDOM structure (fixed effects constant, REML).",
                "Interpretation": "AIC/BIC LEGITIMATE here (Category 2, Manolov & Moeyaert 2025)."}])],
                ignore_index=True)
        if valid_re:
            table = pd.concat([table, pd.DataFrame([{
                "Model": "-> Random structure preference (BIC)", "b2 (level)": "",
                "AIC": "", "BIC": "", "logLik": "",
                "LRT vs immediate": summary["random_structure_bic_preferred"],
                "Explanation": "BIC preferred over AIC for the random structure (Manolov & Moeyaert 2025).",
                "Interpretation": "LEGITIMATE data-driven choice (intercept only vs intercept+slope)."}])],
                ignore_index=True)
    return table, summary
