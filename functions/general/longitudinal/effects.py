"""
Longitudinal extensions: count outcomes + effect sizes
======================================================

Additive companion to ``Longitudinal_Analysis`` (no new dependency), bringing two
things reviewers routinely ask for on repeated/paired designs:

1. **Count outcome pipeline** (``pipeline_paired_count``) - events repeated within
   subject (seizures, admissions, lesions...). Marginal Poisson / negative-binomial
   GEE clustered on the subject; overdispersion is detected (Pearson dispersion of
   the Poisson fit) and the model auto-upgrades to negative binomial when needed.
   Effects are exported as **rate ratios (IRR)**.

2. **Effect sizes** (``paired_effect_sizes``) - the standardized magnitudes that
   p-values do not convey:
     - continuous, 2 times : Cohen's **dz** (paired) + **Hedges' g** small-sample
       correction, and the Wilcoxon **rank-biserial** correlation;
     - continuous, >2 times: **η²** / partial η² of the time factor (from the
       repeated-measures ANOVA);
     - count: the **rate ratio** is the effect size (reported by the pipeline).

Everything reuses the same ``(model_info: dict, model)`` contract and the same
wide<->long handling as the rest of the package.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

from functions.general.longitudinal.analysis import ensure_long, describe_by_time, prepare_repeated


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cohen_dz(pre, post):
    """
    Paired Cohen's dz = mean(diff) / sd(diff), with Hedges' small-sample
    correction g = dz * (1 - 3/(4*n - 5)).

    Returns ``(dz, g, n_pairs)``. dz is the effect size that matches a paired
    t-test (it standardises by the SD of the within-subject differences, not the
    pooled SD), so it is the right magnitude to report next to that test.

    References: Cohen 1988 (paired dz); Hedges 1981 (small-sample g correction).
    R equivalent: effectsize::cohens_d(paired = TRUE) / effsize::cohen.d.
    """
    pre = pd.to_numeric(pd.Series(pre), errors="coerce")
    post = pd.to_numeric(pd.Series(post), errors="coerce")
    diff = (post - pre).dropna()
    n = len(diff)
    if n < 2 or diff.std(ddof=1) == 0:
        return np.nan, np.nan, n
    dz = diff.mean() / diff.std(ddof=1)
    g = dz * (1 - 3 / (4 * n - 5)) if (4 * n - 5) != 0 else dz
    return float(dz), float(g), n


def rank_biserial_wilcoxon(pre, post):
    """
    Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-), the effect
    size that accompanies the Wilcoxon signed-rank test. ``|r|`` near 1 means the
    differences are consistently of one sign. Returns ``(r, n_nonzero_pairs)``.

    References: Cureton 1956; Kerby 2014 (matched-pairs rank-biserial).
    R equivalent: effectsize::rank_biserial / rcompanion::wilcoxonPairedR.
    """
    pre = pd.to_numeric(pd.Series(pre), errors="coerce")
    post = pd.to_numeric(pd.Series(post), errors="coerce")
    diff = (post - pre).dropna()
    diff = diff[diff != 0]
    n = len(diff)
    if n < 1:
        return np.nan, 0
    ranks = stats.rankdata(diff.abs())
    w_pos = ranks[diff.to_numpy() > 0].sum()
    w_neg = ranks[diff.to_numpy() < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return np.nan, n
    return float((w_pos - w_neg) / total), n


def eta_squared_rm(db_long, id_col, time_col, outcome):
    """
    η² and partial η² of the time factor from a repeated-measures ANOVA
    (complete cases). Returns a dict; NaN-safe if the design is too small.

    η²        = SS_time / SS_total
    partial η² = SS_time / (SS_time + SS_error)   (the within-subjects analogue)

    References: Olejnik & Algina 2003 (generalized / partial eta-squared).
    R equivalent: effectsize::eta_squared / ez::ezANOVA (reports eta-squared).
    """
    wide = db_long.pivot_table(index=id_col, columns=time_col,
                               values=outcome, aggfunc="mean").dropna()
    times = list(wide.columns)
    if len(times) < 2 or len(wide) < 3:
        return {"eta_squared": np.nan, "partial_eta_squared": np.nan, "N": len(wide)}
    long_cc = wide.reset_index().melt(id_vars=id_col, var_name=time_col, value_name=outcome)
    try:
        aov = AnovaRM(long_cc, depvar=outcome, subject=id_col, within=[time_col]).fit()
        tbl = aov.anova_table
        f = float(tbl["F Value"].iloc[0])
        df_t = float(tbl["Num DF"].iloc[0])
        df_e = float(tbl["Den DF"].iloc[0])
        # partial η² from F: (F*df_t) / (F*df_t + df_e)
        peta = (f * df_t) / (f * df_t + df_e)
        # η² needs the SS decomposition: rebuild from grand/marginal means
        grand = long_cc[outcome].mean()
        ss_total = float(((long_cc[outcome] - grand) ** 2).sum())
        cell = long_cc.groupby(time_col)[outcome].mean()
        n_subj = len(wide)
        ss_time = float((((cell - grand) ** 2) * n_subj).sum())
        eta = ss_time / ss_total if ss_total > 0 else np.nan
        return {"eta_squared": float(eta), "partial_eta_squared": float(peta), "N": n_subj}
    except Exception:
        return {"eta_squared": np.nan, "partial_eta_squared": np.nan, "N": len(wide)}


def paired_effect_sizes(db_long, id_col, time_col, outcome, *, group_col=None):
    """
    Tidy table of effect sizes for a continuous repeated outcome, overall and per
    group. For 2 timepoints: dz, Hedges' g, rank-biserial. For >2: η² / partial η²
    of the time factor. Returns a DataFrame.

    References: Cohen 1988; Hedges 1981; Cureton 1956; Olejnik & Algina 2003.
    R equivalent: effectsize (cohens_d / rank_biserial / eta_squared).
    """
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    groups = ([("All", db_long)] if group_col is None
              else [("All", db_long)] +
                   [(str(g), db_long[db_long[group_col] == g])
                    for g in sorted(db_long[group_col].dropna().unique().tolist(), key=str)])
    rows = []
    for label, sub in groups:
        wide = sub.pivot_table(index=id_col, columns=time_col,
                               values=outcome, aggfunc="mean")
        if len(times) == 2 and set(times).issubset(wide.columns):
            dz, g, n = cohen_dz(wide[times[0]], wide[times[1]])
            rbc, n_rb = rank_biserial_wilcoxon(wide[times[0]], wide[times[1]])
            rows.append({"Group": label, "Effect size": "Cohen's dz (paired)",
                         "Value": round(dz, 3) if pd.notna(dz) else np.nan, "N": n})
            rows.append({"Group": label, "Effect size": "Hedges' g",
                         "Value": round(g, 3) if pd.notna(g) else np.nan, "N": n})
            rows.append({"Group": label, "Effect size": "Rank-biserial (Wilcoxon)",
                         "Value": round(rbc, 3) if pd.notna(rbc) else np.nan, "N": n_rb})
        else:
            es = eta_squared_rm(sub, id_col, time_col, outcome)
            rows.append({"Group": label, "Effect size": "η² (time)",
                         "Value": round(es["eta_squared"], 3) if pd.notna(es["eta_squared"]) else np.nan,
                         "N": es["N"]})
            rows.append({"Group": label, "Effect size": "partial η² (time)",
                         "Value": round(es["partial_eta_squared"], 3) if pd.notna(es["partial_eta_squared"]) else np.nan,
                         "N": es["N"]})
    return pd.DataFrame(rows, columns=["Group", "Effect size", "Value", "N"])


# --------------------------------------------------------------------------- #
# Count outcome: Poisson / negative-binomial GEE
# --------------------------------------------------------------------------- #
def _pearson_dispersion(res):
    """Pearson dispersion phi = sum(pearson_resid^2) / df_resid. phi >> 1 => overdispersion.

    References: McCullagh & Nelder 1989 (Pearson dispersion / overdispersion).
    R equivalent: performance::check_overdispersion - potential equivalent, to test."""
    try:
        pr = np.asarray(res.resid_pearson, dtype=float)
    except Exception:
        return np.nan
    df_resid = len(pr) - len(res.params)
    if df_resid <= 0:
        return np.nan
    return float(np.nansum(pr ** 2) / df_resid)


def _estimate_nb_alpha(db_long, formula):
    """Rough negative-binomial dispersion α from a pooled NB GLM (ignores
    clustering - only used to parameterise the NB working family of the GEE).

    References: Cameron & Trivedi 1998 (negative-binomial dispersion).
    R equivalent: MASS::glm.nb (theta by MLE, not method-of-moments) - potential equivalent, to test."""
    try:
        m = smf.glm(formula, data=db_long,
                    family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        # method-of-moments refinement from Pearson dispersion of a Poisson fit
        pois = smf.glm(formula, data=db_long, family=sm.families.Poisson()).fit()
        mu = np.asarray(pois.fittedvalues, dtype=float)
        y = np.asarray(pois.model.endog, dtype=float)
        denom = np.sum(mu ** 2)
        if denom <= 0:
            return 1.0
        alpha = np.sum((y - mu) ** 2 - mu) / denom
        return float(max(alpha, 1e-3))
    except Exception:
        return 1.0


def fit_gee_count(db_long, id_col, time_col, outcome, *, group_col=None,
                  time_as="factor", family="auto", cov_struct="exchangeable",
                  overdispersion_threshold=1.5):
    """
    Marginal count model via GEE clustered on the subject.

    ``family="auto"`` fits Poisson first, measures Pearson overdispersion, and
    refits with a negative-binomial working family when φ exceeds
    ``overdispersion_threshold``. Force a family with ``"poisson"`` /
    ``"negativebinomial"``. Returns ``(results | None, info)``.

    References: Liang & Zeger 1986 (GEE); McCullagh & Nelder 1989 (Poisson/NB overdispersion).
    R equivalent: geepack::geeglm(family = poisson) / MASS::negative.binomial.
    """
    time_term = f"C({time_col})" if time_as == "factor" else time_col
    rhs = time_term if group_col is None else f"{time_term} * C({group_col})"
    formula = f"{outcome} ~ {rhs}"
    cs = {"exchangeable": sm.cov_struct.Exchangeable(),
          "ar1": sm.cov_struct.Autoregressive(),
          "independence": sm.cov_struct.Independence()}.get(cov_struct.lower())
    if cs is None:
        raise ValueError("cov_struct must be exchangeable / ar1 / independence")
    fam = family.lower()
    if fam not in {"auto", "poisson", "negativebinomial", "nb"}:
        raise ValueError("family must be auto / poisson / negativebinomial")

    info = {"Model": "Marginal count GEE", "Formula": formula,
            "Cluster": id_col, "Working correlation": cov_struct.lower()}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pois = smf.gee(formula, groups=id_col, data=db_long,
                           family=sm.families.Poisson(), cov_struct=cs).fit()
            phi = _pearson_dispersion(pois)
            info["Poisson Pearson dispersion (>1.5 => overdispersed)"] = (
                round(phi, 3) if pd.notna(phi) else np.nan)
            use_nb = fam in {"negativebinomial", "nb"} or (
                fam == "auto" and pd.notna(phi) and phi > overdispersion_threshold)
            if use_nb:
                alpha = _estimate_nb_alpha(db_long, formula)
                info["Family"] = "Negative binomial"
                info["NB alpha (dispersion)"] = round(alpha, 4)
                res = smf.gee(formula, groups=id_col, data=db_long,
                              family=sm.families.NegativeBinomial(alpha=alpha),
                              cov_struct=cs).fit()
            else:
                info["Family"] = "Poisson"
                res = pois
        info["Converged"] = bool(getattr(res, "converged", True))
    except Exception as exc:
        info["Warning"] = f"Count GEE not fitted ({type(exc).__name__}: {exc})."
        return None, info
    return res, info


def _irr_table(res):
    """Rate ratios IRR = exp(β) with 95% CI and p, skipping the intercept."""
    cols = ["Term", "IRR (95% CI)", "P"]
    if res is None:
        return pd.DataFrame(columns=cols)
    params, conf, pv = res.params, res.conf_int(), res.pvalues
    rows = []
    for term in params.index:
        if term == "Intercept":
            continue
        lo, hi = conf.loc[term]
        rows.append({"Term": term,
                     "IRR (95% CI)": f"{np.exp(params[term]):.3f} ({np.exp(lo):.3f} to {np.exp(hi):.3f})",
                     "P": f"{pv[term]:.4f}"})
    return pd.DataFrame(rows, columns=cols)


def _count_summary(serie):
    """median [IQR] + mean for a count cell."""
    s = pd.to_numeric(pd.Series(serie), errors="coerce").dropna()
    if len(s) == 0:
        return ""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{s.median():.0f} [{q1:.0f} ; {q3:.0f}] (mean {s.mean():.2f})"


def count_paired_tests(db_long, id_col, time_col, outcome):
    """Nonparametric omnibus on counts: Wilcoxon (2 times) / Friedman (>2).
    Returns a one/few-row DataFrame.

    References: Wilcoxon 1945 (signed-rank, 2 times); Friedman 1937 (>2 times).
    R equivalent: stats::wilcox.test(paired = TRUE) / stats::friedman.test."""
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    wide = db_long.pivot_table(index=id_col, columns=time_col,
                               values=outcome, aggfunc="mean").dropna()
    n = len(wide)
    if n < 3 or len(times) < 2:
        return pd.DataFrame([{"Test": " - ", "Statistic": np.nan, "P": np.nan,
                              "N pairs": n, "Comment": "Insufficient sample size."}])
    if len(times) == 2:
        try:
            w, p = stats.wilcoxon(wide[times[0]], wide[times[1]])
            return pd.DataFrame([{"Test": "Wilcoxon signed-rank", "Statistic": float(w),
                                  "P": float(p), "N pairs": n,
                                  "Comment": "Paired counts, nonparametric."}])
        except Exception as exc:
            return pd.DataFrame([{"Test": "Wilcoxon signed-rank", "Statistic": np.nan,
                                  "P": np.nan, "N pairs": n, "Comment": str(exc)}])
    chi2, p = stats.friedmanchisquare(*[wide[t].to_numpy(float) for t in times])
    return pd.DataFrame([{"Test": "Friedman", "Statistic": float(chi2), "P": float(p),
                          "N pairs": n, "Comment": "Repeated counts (>2 timepoints)."}])


def pipeline_paired_count(db, *, id_col, outcome, time_col=None, time_map=None,
                          group_col=None, covariates=None, time_as="factor",
                          family="auto", cov_struct="exchangeable",
                          impute="none", n_iter=10,
                          save_path=None, verbose=True):
    """
    End-to-end paired / repeated-measures pipeline for a COUNT outcome.

    Reshapes if needed, audits completeness and optionally imputes (``impute`` ∈
    {"none", "mice", "locf"}), summarises counts by timepoint, runs a
    nonparametric omnibus (Wilcoxon / Friedman), fits a Poisson-or-NB GEE
    (``outcome ~ time [+ group + time:group]``, auto overdispersion handling) and
    exports rate ratios. Writes a report (``Setup Info``, ``Rate Ratios``,
    ``Tests``, ``By Time``, ``Missingness``) when ``save_path`` is given. The
    report states whether an imputation was performed.

    Returns ``(model_info: dict, gee_results | None)``.

    References: Liang & Zeger 1986 (marginal count GEE, rate ratio = exp(beta)).
    R equivalent: geepack::geeglm - potential pipeline equivalent, to test.
    """
    db_long, time_col, report, miss_df, _ = prepare_repeated(
        db, id_col, outcome, time_col=time_col, time_map=time_map,
        group_col=group_col, covariates=covariates, impute=impute,
        n_iter=n_iter, binary=False)

    # by-time count summary (overall + per group)
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    groups = ([None] if group_col is None
              else sorted(db_long[group_col].dropna().unique().tolist(), key=str))
    bt_rows = []
    for g in groups:
        sub = db_long if g is None else db_long[db_long[group_col] == g]
        for t in times:
            cell = sub[sub[time_col] == t][outcome]
            bt_rows.append({"Group": "All" if g is None else str(g), "Time": str(t),
                            "N": int(pd.to_numeric(cell, errors="coerce").notna().sum()),
                            "Summary": _count_summary(cell)})
    bytime_df = pd.DataFrame(bt_rows, columns=["Group", "Time", "N", "Summary"])

    tests_df = count_paired_tests(db_long, id_col, time_col, outcome)
    res, fit_info = fit_gee_count(db_long, id_col, time_col, outcome,
                                  group_col=group_col, time_as=time_as,
                                  family=family, cov_struct=cov_struct)
    irr_df = _irr_table(res)

    model_info = {"Outcome type": "Count", "Outcome": outcome, "Subject id": id_col,
                  "Time": time_col, "Group": group_col if group_col else " - ",
                  "Time encoding": time_as}
    model_info.update(report)
    model_info.update(fit_info)

    if save_path is not None:
        sub_dir = os.path.join(save_path, "Longitudinal models", "Paired_Count")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        full = os.path.join(sub_dir, f"{outcome}_paired_count_report.xlsx")
        info_df = pd.DataFrame([{"Parameter": k,
                                 "Value": (", ".join(map(str, v)) if isinstance(v, list) else v)}
                                for k, v in model_info.items()], columns=["Parameter", "Value"])
        with pd.ExcelWriter(full, engine="xlsxwriter") as writer:
            info_df.to_excel(writer, sheet_name="Setup Info", index=False)
            irr_df.to_excel(writer, sheet_name="Rate Ratios", index=False)
            tests_df.to_excel(writer, sheet_name="Tests", index=False)
            if miss_df is not None and not miss_df.empty:
                miss_df.to_excel(writer, sheet_name="Missingness", index=False)
            bytime_df.to_excel(writer, sheet_name="By Time", index=False)
        if verbose:
            print(f"Report exported to {full}")
    elif verbose:
        print("Report not exported")

    return model_info, res
