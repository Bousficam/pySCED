"""
Longitudinal / paired-measures pipeline
=======================================

Counterpart of the univariate/multivariate pipelines for PAIRED or
REPEATED-MEASURES designs - a pre/post (2 timepoints) study or a follow-up
with ``>2`` timepoints, with or without a between-subject group factor (a
treatment arm, etc.).

Unlike a screening pipeline, there is *no variable selection* here: the
estimand (``outcome ~ time [+ group + time:group]``) is pre-specified, and the
job is to run, in one go, every step a paired analysis needs and to drop a
ready-to-use multi-sheet Excel report.

One pipeline per outcome type, mirroring the ``ols`` / ``logit`` split, with the
same call/return contract as the other pipelines
(``(model_info: dict, model)``)::

    model_info, model = pipeline_paired_ols(
        db, id_col="subject", outcome="score",
        time_map={"pre": "score_V1", "post": "score_V2"},   # wide -> long
        group_col="arm", save_path=save_path)

    model_info, model = pipeline_paired_logit(
        db_long, id_col="subject", outcome="success",
        time_col="visit", save_path=save_path)              # already long

Steps (all automatic):
  1. Reshape to long format if needed (``to_long`` / ``ensure_long`` /
     ``prepare_repeated``), audit completeness (``missingness_report``) and
     optionally impute (``impute`` ∈ {"none", "mice", "locf"}); the report states
     whether an imputation was performed and its caveat.
  2. Descriptive summary BY TIMEPOINT (+ within-subject delta for 2 times),
     split by group when one is provided.
  3. Omnibus paired test(s):
       continuous : 2 times  -> paired t / Wilcoxon (normality of the diffs)
                    >2 times  -> repeated-measures ANOVA (+ Greenhouse-Geisser
                                 sphericity correction) / Friedman
                    any       -> paired PERMANOVA (within-subject permutation)
       binary     : 2 times  -> McNemar
                    >2 times  -> Cochran's Q
  4. Model accounting for the within-subject correlation:
       continuous : statsmodels ``MixedLM`` (random intercept [+ slope] / subject)
       binary     : statsmodels ``GEE`` (logit link, ``cov_struct`` working corr.)
  5. Model diagnostics (continuous only): Shapiro-Wilk normality of the residuals
     and a within-subject Durbin-Watson autocorrelation check; plus a QIC
     comparison of exchangeable vs AR(1) working correlation
     (``compare_corr_structures``, ≥3 timepoints).
  6. Excel report with sheets ``Setup Info``, ``Model Effects``, ``Diagnostics``
     (OLS), ``Missingness``, ``Correlation Structure``, ``Paired Tests`` and
     ``By Time``.

No new dependency: everything sits on top of ``scipy`` and ``statsmodels``
(``MixedLM``, ``GEE``, ``AnovaRM``, ``mcnemar``, ``cochrans_q``).
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q


# --------------------------------------------------------------------------- #
# Reshape & validation
# --------------------------------------------------------------------------- #
def to_long(db, id_col, time_map, outcome="value", *, group_col=None, covariates=None):
    """
    Reshape a WIDE table (one row per subject, one column per timepoint) into the
    long format expected by the rest of the module (one row per subject x time).

    Parameters
    ----------
    db : pd.DataFrame
    id_col : str
        Subject identifier column.
    time_map : dict[str, str] | list[str]
        Mapping ``{time_label: wide_column}`` (e.g. ``{"pre": "score_V1",
        "post": "score_V2"}``) or a plain list of wide columns - in which case the
        column name itself is used as the time label.
    outcome : str
        Name to give the stacked value column in the long output.
    group_col : str | None
        Between-subject factor carried over unchanged (constant per subject).
    covariates : list[str] | None
        Extra (subject-level) columns carried over unchanged.

    Returns
    -------
    pd.DataFrame with columns ``[id_col, "time", outcome, (group_col), *covariates]``.

    R equivalent: tidyr::pivot_longer / reshape2::melt.
    """
    if isinstance(time_map, dict):
        items = list(time_map.items())
    else:
        items = [(c, c) for c in time_map]

    missing = [col for _, col in items if col not in db.columns]
    if missing:
        raise KeyError(f"Time columns missing from the DataFrame: {missing}")

    carry = [id_col]
    if group_col is not None:
        carry.append(group_col)
    if covariates:
        carry += [c for c in covariates if c not in carry]
    carry_missing = [c for c in carry if c not in db.columns]
    if carry_missing:
        raise KeyError(f"Columns missing from the DataFrame: {carry_missing}")

    frames = []
    for label, col in items:
        chunk = db[carry].copy()
        chunk["time"] = label
        chunk[outcome] = pd.to_numeric(db[col], errors="coerce")
        frames.append(chunk)
    long = pd.concat(frames, axis=0, ignore_index=True)
    ordered = [id_col, "time", outcome] + [c for c in carry if c != id_col]
    return long[ordered]


def validate_long(db_long, id_col, time_col, outcome, *, group_col=None):
    """
    Sanity-check a long table and return a clean copy (rows with a missing
    outcome dropped) plus a small report dict.

    Drops rows whose outcome is NaN, coerces the outcome to numeric, and reports
    the number of timepoints, subjects and whether the design is balanced (each
    subject seen at every timepoint exactly once).
    """
    for col in [id_col, time_col, outcome] + ([group_col] if group_col else []):
        if col not in db_long.columns:
            raise KeyError(f"Column '{col}' missing from the long DataFrame.")

    df = db_long.copy()
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[outcome])
    times = sorted(df[time_col].dropna().unique().tolist(), key=str)

    counts = df.groupby([id_col, time_col]).size()
    balanced = bool((counts == 1).all()
                    and df.groupby(id_col)[time_col].nunique().eq(len(times)).all())

    report = {
        "Number of timepoints": len(times),
        "Timepoints": times,
        "Number of subjects": int(df[id_col].nunique()),
        "Observations dropped (missing outcome)": n_before - len(df),
        "Balanced design": balanced,
    }
    return df, report


def ensure_long(db, id_col, outcome, *, time_col=None, time_map=None,
                group_col=None, covariates=None):
    """
    Return ``(db_long, time_col, report)`` whatever the input shape.

    - If ``time_map`` is given -> the input is WIDE and is reshaped via ``to_long``
      (the resulting time column is named ``"time"``).
    - Else ``time_col`` must be given -> the input is already LONG and is
      validated as-is.
    """
    if time_map is not None:
        db_long = to_long(db, id_col, time_map, outcome=outcome,
                          group_col=group_col, covariates=covariates)
        time_col = "time"
    elif time_col is not None:
        db_long = db
    else:
        raise ValueError("Provide either time_map (wide format) or time_col (long format).")
    db_long, report = validate_long(db_long, id_col, time_col, outcome, group_col=group_col)
    return db_long, time_col, report


# --------------------------------------------------------------------------- #
# Missingness & imputation
# --------------------------------------------------------------------------- #
def missingness_report(db_long_raw, id_col, time_col, outcome, *, group_col=None):
    """
    Completeness audit of a (pre-drop) long table. Returns ``(info: dict, df)``.

    Reports, overall and per group:
      - per-timepoint count of observed vs missing outcome,
      - number of subjects seen at every timepoint (``complete``) vs partially,
      - the share of subjects a complete-case analysis would discard.

    This matters because the paired tests and the PERMANOVA drop any subject not
    seen at all timepoints; a large complete-case loss silently biases results
    (under MAR) and costs power. The linear mixed model, by contrast, stays valid
    under MAR without imputation - surface this number so the choice is informed.

    R equivalent: naniar::miss_var_summary / mice::md.pattern.
    """
    df = db_long_raw.copy()
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    times = sorted(df[time_col].dropna().unique().tolist(), key=str)
    n_times = len(times)

    def _block(sub, label):
        rows = []
        for t in times:
            cell = sub[sub[time_col] == t][outcome]
            n_obs = int(cell.notna().sum())
            n_mis = int(cell.isna().sum())
            tot = n_obs + n_mis
            rows.append({"Group": label, "Time": str(t), "Observed": n_obs,
                         "Missing": n_mis,
                         "% missing": round(100 * n_mis / tot, 1) if tot else np.nan})
        seen = sub.dropna(subset=[outcome]).groupby(id_col)[time_col].nunique()
        n_subj = int(sub[id_col].nunique())
        n_complete = int((seen == n_times).sum())
        rows.append({"Group": label, "Time": " - subjects complete / total - ",
                     "Observed": n_complete, "Missing": n_subj - n_complete,
                     "% missing": round(100 * (n_subj - n_complete) / n_subj, 1) if n_subj else np.nan})
        return rows, n_subj, n_complete

    all_rows, n_subj, n_complete = _block(df, "All")
    rows = list(all_rows)
    if group_col is not None:
        for g in sorted(df[group_col].dropna().unique().tolist(), key=str):
            grows, _, _ = _block(df[df[group_col] == g], str(g))
            rows += grows

    pct_dropped = round(100 * (n_subj - n_complete) / n_subj, 1) if n_subj else 0.0
    info = {
        "Subjects total": n_subj,
        "Subjects complete (all timepoints)": n_complete,
        "Complete-case subjects dropped (%)": pct_dropped,
    }
    if pct_dropped >= 20:
        info["Missingness warning"] = (
            f"Complete-case analysis discards {pct_dropped}% of subjects: consider the "
            f"mixed model (valid under MAR) or an imputation (impute='mice').")
    return info, pd.DataFrame(rows, columns=["Group", "Time", "Observed", "Missing", "% missing"])


def impute_long(db_long_raw, id_col, time_col, outcome, *, method="mice",
                n_iter=10, random_state=0, group_col=None, binary=False):
    """
    Impute the missing outcome of a long table. Returns ``(db_imputed, info)``.

    method='mice' : statsmodels ``MICEData`` on the subject × time matrix (each
      timepoint is a predictor of the others), a SINGLE completed dataset drawn
      after ``n_iter`` iterations. ``info`` flags that the resulting standard
      errors do NOT incorporate between-imputation uncertainty (no Rubin pooling)
       - single imputation, to be read with that caveat.
    method='locf' : last-observation-carried-forward then backward fill along the
      time order; simple but biased, flagged as such.

    For a binary outcome, MICE draws are rounded back to 0/1.

    References: van Buuren & Groothuis-Oudshoorn 2011 (MICE / chained equations).
    R equivalent: mice::mice (MICE); zoo::na.locf / imputeTS (LOCF).
    """
    df = db_long_raw.copy()
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    times = sorted(df[time_col].dropna().unique().tolist(), key=str)
    wide = df.pivot_table(index=id_col, columns=time_col, values=outcome, aggfunc="mean")
    wide = wide.reindex(columns=times)
    n_missing = int(wide.isna().sum().sum())
    info = {"Imputation": method, "Missing cells imputed": n_missing}

    if n_missing == 0:
        info["Imputation note"] = "No missing data: imputation has no effect."
        return df, info

    if method == "locf":
        filled = wide.ffill(axis=1).bfill(axis=1)
        info["Imputation note"] = ("LOCF/BOCF: last observed value carried forward "
                                   " - simple but BIASED (variance underestimated).")
    elif method == "mice":
        from statsmodels.imputation.mice import MICEData
        cols = [str(t) for t in times]
        work = wide.copy()
        work.columns = cols
        mid = MICEData(work.reset_index(drop=True))
        mid.update_all(n_iter)
        filled = mid.data.copy()
        filled.columns = times
        filled.index = wide.index
        if binary:
            filled = filled.round().clip(0, 1)
        info["Imputation note"] = (
            f"MICE (single completed dataset after {n_iter} iterations). "
            f"Standard errors DO NOT incorporate the imputation uncertainty "
            f"(no Rubin rules): interpret the CIs with caution.")
    else:
        raise ValueError("method must be 'mice' or 'locf'")

    long_filled = filled.reset_index().melt(id_vars=id_col, var_name=time_col,
                                            value_name=outcome)
    # carry back group/covariates (subject-level, constant)
    carry = [c for c in df.columns if c not in (time_col, outcome)]
    meta = df[carry].drop_duplicates(subset=[id_col])
    out = long_filled.merge(meta, on=id_col, how="left")
    return out[df.columns.tolist()], info


def prepare_repeated(db, id_col, outcome, *, time_col=None, time_map=None,
                     group_col=None, covariates=None, impute="none",
                     n_iter=10, random_state=0, binary=False):
    """
    Full data-preparation step shared by the pipelines.

    Builds the long frame, audits completeness (``missingness_report``), optionally
    imputes (``impute`` ∈ {"none", "mice", "locf"}), then validates. Returns
    ``(db_long, time_col, report, missingness_df, impute_info)`` where
    ``report`` already carries the missingness summary and, when an imputation was
    run, an ``"Imputation"`` entry so the Excel report states it explicitly.
    """
    # 1) to long (without dropping yet, to audit missingness honestly)
    if time_map is not None:
        long_raw = to_long(db, id_col, time_map, outcome=outcome,
                           group_col=group_col, covariates=covariates)
        time_col = "time"
    elif time_col is not None:
        long_raw = db.copy()
    else:
        raise ValueError("Provide either time_map (wide format) or time_col (long format).")

    miss_info, miss_df = missingness_report(long_raw, id_col, time_col, outcome,
                                            group_col=group_col)

    impute_info = {"Imputation": "none"}
    if impute and impute.lower() != "none":
        long_raw, impute_info = impute_long(long_raw, id_col, time_col, outcome,
                                            method=impute.lower(), n_iter=n_iter,
                                            random_state=random_state,
                                            group_col=group_col, binary=binary)

    db_long, report = validate_long(long_raw, id_col, time_col, outcome, group_col=group_col)
    report.update(miss_info)
    report["Imputation"] = impute_info.get("Imputation", "none")
    if "Imputation note" in impute_info:
        report["Imputation note"] = impute_info["Imputation note"]
    if "Missing cells imputed" in impute_info:
        report["Missing cells imputed"] = impute_info["Missing cells imputed"]
    return db_long, time_col, report, miss_df, impute_info


# --------------------------------------------------------------------------- #
# Descriptive summary by timepoint
# --------------------------------------------------------------------------- #
def _summarise_series(serie, continuous=True):
    """One-cell summary: mean ± sd / median [IQR] for continuous, n (%) for binary."""
    serie = pd.to_numeric(serie, errors="coerce").dropna()
    n = len(serie)
    if n == 0:
        return ""
    if continuous:
        normal = n >= 3 and stats.shapiro(serie)[1] > 0.05
        if normal:
            return f"{serie.mean():.2f} ± {serie.std(ddof=1):.2f}"
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        return f"{serie.median():.2f} [{q1:.2f} ; {q3:.2f}]"
    # binary
    k = int((serie != 0).sum())
    return f"{k}/{n} ({k / n * 100:.1f}%)"


def describe_by_time(db_long, id_col, time_col, outcome, *, group_col=None, continuous=True):
    """
    Descriptive summary of the outcome at each timepoint (and per group when a
    group factor is given), plus the within-subject delta for a 2-timepoint
    design. Returns a tidy DataFrame.
    """
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    groups = ([None] if group_col is None
              else sorted(db_long[group_col].dropna().unique().tolist(), key=str))

    rows = []
    for g in groups:
        sub = db_long if g is None else db_long[db_long[group_col] == g]
        for t in times:
            cell = sub[sub[time_col] == t][outcome]
            rows.append({
                "Group": "All" if g is None else str(g),
                "Time": str(t),
                "N": int(pd.to_numeric(cell, errors="coerce").notna().sum()),
                "Summary": _summarise_series(cell, continuous=continuous),
            })
        # within-subject delta when exactly two timepoints (continuous only:
        # a numeric pre/post change is meaningful for scores, not for a 0/1 state)
        if continuous and len(times) == 2:
            wide = (sub.pivot_table(index=id_col, columns=time_col,
                                    values=outcome, aggfunc="mean"))
            if set(times).issubset(wide.columns):
                delta = (wide[times[1]] - wide[times[0]]).dropna()
                rows.append({
                    "Group": "All" if g is None else str(g),
                    "Time": f"Δ ({times[1]}-{times[0]})",
                    "N": int(len(delta)),
                    "Summary": _summarise_series(delta, continuous=True),
                })
    return pd.DataFrame(rows, columns=["Group", "Time", "N", "Summary"])


def _paired_wide(db_long, id_col, time_col, outcome, times):
    """Subject x time matrix (mean over any duplicates), columns ordered as `times`."""
    wide = db_long.pivot_table(index=id_col, columns=time_col,
                               values=outcome, aggfunc="mean")
    cols = [t for t in times if t in wide.columns]
    return wide[cols]


# --------------------------------------------------------------------------- #
# Omnibus paired tests
# --------------------------------------------------------------------------- #
def _continuous_paired_test(wide, times):
    """Run the right paired test on a subject x time matrix (complete cases).

    References: Student 1908 (paired t); Wilcoxon 1945 (signed-rank); Friedman 1937
    (>2 times); Shapiro & Wilk 1965 (normality gate on the differences).
    R equivalent: stats::t.test(paired = TRUE) / stats::wilcox.test / stats::friedman.test."""
    cc = wide.dropna()
    n = len(cc)
    if n < 3:
        return {"Test": " - ", "Statistic": np.nan, "P": np.nan, "N pairs": n,
                "Comment": "Insufficient paired sample size (n<3)."}
    if len(times) == 2:
        diff = cc[times[1]] - cc[times[0]]
        if diff.nunique() <= 1:
            return {"Test": " - ", "Statistic": np.nan, "P": np.nan, "N pairs": n,
                    "Comment": "Constant differences: no test applicable."}
        normal = stats.shapiro(diff)[1] > 0.05
        if normal:
            t, p = stats.ttest_rel(cc[times[1]], cc[times[0]])
            return {"Test": "Paired t-test", "Statistic": float(t), "P": float(p),
                    "N pairs": n, "Comment": "Normality of the differences not rejected (Shapiro)."}
        w, p = stats.wilcoxon(cc[times[1]], cc[times[0]])
        return {"Test": "Wilcoxon signed-rank", "Statistic": float(w), "P": float(p),
                "N pairs": n, "Comment": "Differences not normal -> nonparametric test."}
    # >2 timepoints
    samples = [cc[t].to_numpy(float) for t in times]
    chi2, p = stats.friedmanchisquare(*samples)
    return {"Test": "Friedman", "Statistic": float(chi2), "P": float(p),
            "N pairs": n, "Comment": "Repeated measures (>2 timepoints), nonparametric test."}


def _greenhouse_geisser_epsilon(wide_cc):
    """
    Greenhouse-Geisser sphericity-correction factor ε from a complete-case
    subject × time matrix.

    Sphericity is the assumption a repeated-measures ANOVA relies on: the
    variances of *all pairwise differences between timepoints* are equal
    (equivalently, the within-subject covariance matrix is "compound-symmetric
    up to rotation"). In a longitudinal study this rarely holds - measurements
    close in time correlate more than distant ones (AR-like structure), so
    Var(t2−t0) > Var(t1−t0), and sphericity is violated. The unbounded
    consequence: the ANOVA F uses too-optimistic degrees of freedom and the test
    becomes anti-conservative (p-values too small, type-I inflation).

    ε ∈ [1/(k−1), 1] quantifies the departure (1 = perfect sphericity). It is
    used to shrink the ANOVA degrees of freedom (df × ε), which raises the
    p-value back to an honest level. We apply it unconditionally for k ≥ 3 rather
    than gating on Mauchly's test (under-powered in small samples, over-sensitive
    in large ones): when sphericity does hold, ε ≈ 1 and the correction vanishes.

    Returns ``1.0`` for k < 3 (sphericity is trivial with a single difference).

    References: Greenhouse & Geisser 1959 (sphericity-correction epsilon).
    R equivalent: ez::ezANOVA / afex::aov_ez (report the GG epsilon).
    """
    X = np.asarray(wide_cc, dtype=float)
    n, k = X.shape
    if k < 3 or n < 2:
        return 1.0
    S = np.cov(X, rowvar=False, ddof=1)          # k×k covariance across subjects
    sbar = S.mean()                               # grand mean of all entries
    diag_mean = np.trace(S) / k                   # mean of the diagonal
    num = k ** 2 * (diag_mean - sbar) ** 2
    den = (k - 1) * (np.sum(S ** 2)
                     - 2 * k * np.sum(S.mean(axis=1) ** 2)
                     + k ** 2 * sbar ** 2)
    if den <= 0:
        return 1.0
    eps = num / den
    return float(min(1.0, max(1.0 / (k - 1), eps)))


# --------------------------------------------------------------------------- #
# PERMANOVA - distance-based permutation test for paired / repeated series
# --------------------------------------------------------------------------- #
def _within_ss(D2, labels, levels):
    """Within-group sum of squares from a squared-distance matrix (Anderson 2001).

    For each level, the within sum of squared distances divided by its size; using
    the identity Σ_{i<j} d²_ij = ½·Σ_all d²_ij avoids building triangular indices.

    References: Anderson 2001 (PERMANOVA sum-of-squares decomposition).
    R equivalent: vegan::adonis2 (within-group SS component).
    """
    w = 0.0
    for lev in levels:
        idx = labels == lev
        n_g = int(idx.sum())
        if n_g > 0:
            w += float(D2[np.ix_(idx, idx)].sum()) / (2.0 * n_g)
    return w


def _restricted_permute(labels, subjects, subj_index, rng):
    """Permute the time labels WITHIN each subject block (paired permutation).

    Under H0 (no time effect) the response is exchangeable across timepoints
    inside a subject, so reshuffling labels within each subject - never across
    subjects - gives the correct null for a paired/repeated design.

    References: Anderson & ter Braak 2003 (restricted permutation schemes).
    R equivalent: permute::how / vegan::adonis2 with restricted permutations.
    """
    out = labels.copy()
    for idx in subj_index:
        out[idx] = rng.permutation(labels[idx])
    return out


def permanova_paired(db_long, id_col, time_col, outcomes, *, n_perm=4999,
                     distance="euclidean", standardize=True, random_state=0):
    """
    PERMANOVA (Anderson 2001) for a PAIRED / repeated-measures design, testing the
    within-subject TIME effect with **restricted (within-subject) permutation**.

    Distance-based and **multivariate-capable**: pass several columns in
    ``outcomes`` to test a vector response (where t-test / Wilcoxon / RM-ANOVA
    cannot go). The pseudo-F partitions the squared-distance matrix between vs
    within timepoints; its null distribution is built by permuting the time
    labels inside each subject (the pairing is respected - a free permutation
    would be invalid here because within-subject observations are not
    exchangeable).

    Parameters
    ----------
    outcomes : str | list[str]
        One column (univariate) or several (multivariate response).
    n_perm : int
        Number of restricted permutations (p = (1 + #{F* ≥ F}) / (1 + n_perm)).
    distance : {"euclidean", "manhattan"}
        Dissimilarity between observations.
    standardize : bool
        z-score each outcome before the distance (recommended when the columns
        have different scales, so none dominates the distance).

    Returns
    -------
    dict with ``Test``, ``Pseudo-F``, ``P``, ``R2`` (= SS_between / SS_total),
    sample sizes and a comment. Only subjects observed at *every* timepoint are
    used (a balanced set is required for within-subject permutation).

    References: Anderson 2001 (PERMANOVA); Anderson & ter Braak 2003 (restricted permutation).
    R equivalent: vegan::adonis2 with permute::how (within-subject restriction).
    """
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    df = db_long[[id_col, time_col] + outcomes].copy()
    for c in outcomes:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=outcomes)
    # one row per subject x time (average any duplicates)
    df = df.groupby([id_col, time_col], as_index=False)[outcomes].mean()
    times = sorted(df[time_col].dropna().unique().tolist(), key=str)

    info = {"Test": "PERMANOVA (within-subject permutation)",
            "Outcomes": list(outcomes), "Permutations": int(n_perm),
            "Distance": distance}

    # keep only subjects present at every timepoint (balanced -> permutable)
    complete = df.groupby(id_col)[time_col].nunique()
    complete = complete[complete == len(times)].index
    df = df[df[id_col].isin(complete)].sort_values([id_col, time_col])
    n_subj = int(df[id_col].nunique())
    info.update({"N subjects": n_subj, "N timepoints": len(times)})

    if n_subj < 2 or len(times) < 2:
        info.update({"Pseudo-F": np.nan, "P": np.nan, "R2": np.nan,
                     "Comment": "Insufficient sample size / timepoints for the PERMANOVA."})
        return info

    X = df[outcomes].to_numpy(float)
    if standardize:
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd

    metric = {"euclidean": "euclidean", "manhattan": "cityblock",
              "cityblock": "cityblock"}.get(distance)
    if metric is None:
        raise ValueError("distance must be 'euclidean' or 'manhattan'")
    D2 = squareform(pdist(X, metric=metric)) ** 2

    labels = df[time_col].to_numpy()
    subjects = df[id_col].to_numpy()
    subj_index = [np.where(subjects == s)[0] for s in pd.unique(subjects)]
    levels = times
    N = len(labels)
    a = len(levels)
    df_a, df_r = a - 1, N - a
    total = float(D2.sum()) / (2.0 * N)

    def _pseudo_f(lab):
        within = _within_ss(D2, lab, levels)
        if within <= 0 or df_a <= 0 or df_r <= 0:
            return np.nan, within
        return ((total - within) / df_a) / (within / df_r), within

    f_obs, within_obs = _pseudo_f(labels)
    if not np.isfinite(f_obs):
        info.update({"Pseudo-F": np.nan, "P": np.nan, "R2": np.nan,
                     "Comment": "Pseudo-F not computable (zero within dispersion)."})
        return info

    rng = np.random.RandomState(random_state)
    ge = 0
    for _ in range(n_perm):
        f_p, _ = _pseudo_f(_restricted_permute(labels, subjects, subj_index, rng))
        if np.isfinite(f_p) and f_p >= f_obs - 1e-12:
            ge += 1
    p = (1 + ge) / (1 + n_perm)
    info.update({
        "Pseudo-F": float(f_obs), "P": float(p),
        "R2": float((total - within_obs) / total) if total > 0 else np.nan,
        "Comment": ("Within-subject permutation of the timepoints (pairing preserved)"
                    + ("; multivariate response." if len(outcomes) > 1 else ".")),
    })
    return info


def paired_tests_continuous(db_long, id_col, time_col, outcome, *, group_col=None,
                            permanova=True, permanova_n_perm=999, random_state=0):
    """
    Omnibus within-subject test(s) for a continuous outcome.

    Always reports the overall paired test (pooling groups). When a group factor
    is present, also reports the test within each group and adds a
    repeated-measures ANOVA where applicable (the formal between/within
    interaction is left to the mixed model). When ``permanova=True`` (default), a
    distance-based permutation test with restricted within-subject permutation is
    also reported (see ``permanova_paired``). Returns a DataFrame.

    References: repeated-measures ANOVA; Greenhouse & Geisser 1959 (sphericity);
    Friedman 1937 (nonparametric >2 times); Anderson 2001 (PERMANOVA arm).
    R equivalent: afex::aov_ez / ez::ezANOVA (RM-ANOVA + GG); vegan::adonis2 (PERMANOVA).
    """
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    rows = []

    def _permanova_row(sub_df, group_label):
        info = permanova_paired(sub_df, id_col, time_col, outcome,
                                n_perm=permanova_n_perm, random_state=random_state)
        return {"Group": group_label, "Test": info["Test"],
                "Statistic": info.get("Pseudo-F", np.nan), "P": info.get("P", np.nan),
                "N pairs": info.get("N subjects", np.nan), "Comment": info["Comment"]}

    overall = _continuous_paired_test(_paired_wide(db_long, id_col, time_col, outcome, times), times)
    overall["Group"] = "All"
    rows.append(overall)
    if permanova:
        rows.append(_permanova_row(db_long, "All"))

    # Repeated-measures ANOVA (within-only design, i.e. no group factor).
    if group_col is None and len(times) >= 2:
        cc = _paired_wide(db_long, id_col, time_col, outcome, times).dropna()
        if len(cc) >= 3:
            long_cc = cc.reset_index().melt(id_vars=id_col, var_name=time_col, value_name=outcome)
            try:
                aov = AnovaRM(long_cc, depvar=outcome, subject=id_col, within=[time_col]).fit()
                tbl = aov.anova_table
                f_val = float(tbl["F Value"].iloc[0])
                df_num = float(tbl["Num DF"].iloc[0])
                df_den = float(tbl["Den DF"].iloc[0])
                rows.append({
                    "Group": "All", "Test": "Repeated-measures ANOVA",
                    "Statistic": f_val,
                    "P": float(tbl["Pr > F"].iloc[0]),
                    "N pairs": len(cc),
                    "Comment": "Parametric alternative (sphericity not corrected).",
                })
                # Greenhouse-Geisser: same F, but the degrees of freedom are shrunk
                # (df x epsilon) -> honest p.
                if len(times) >= 3:
                    eps = _greenhouse_geisser_epsilon(cc)
                    p_gg = float(stats.f.sf(f_val, df_num * eps, df_den * eps))
                    rows.append({
                        "Group": "All", "Test": "RM-ANOVA (Greenhouse-Geisser)",
                        "Statistic": f_val, "P": p_gg, "N pairs": len(cc),
                        "Comment": (f"epsilon={eps:.3f} ({'sphericity nearly met' if eps > 0.95 else 'sphericity violated -> correction applied'}); "
                                    f"corrected df {df_num * eps:.2f}, {df_den * eps:.2f}."),
                    })
            except Exception as exc:  # pragma: no cover - numerical safeguard
                rows.append({"Group": "All", "Test": "Repeated-measures ANOVA",
                             "Statistic": np.nan, "P": np.nan, "N pairs": len(cc),
                             "Comment": f"Not computed ({type(exc).__name__})."})

    if group_col is not None:
        for g in sorted(db_long[group_col].dropna().unique().tolist(), key=str):
            sub = db_long[db_long[group_col] == g]
            res = _continuous_paired_test(_paired_wide(sub, id_col, time_col, outcome, times), times)
            res["Group"] = str(g)
            rows.append(res)
            if permanova:
                rows.append(_permanova_row(sub, str(g)))

    cols = ["Group", "Test", "Statistic", "P", "N pairs", "Comment"]
    return pd.DataFrame(rows)[cols]


def _binary_paired_test(wide, times):
    """McNemar (2 times) / Cochran's Q (>2 times) on a 0/1 subject x time matrix.

    References: McNemar 1947 (2 times); Cochran 1950 (Q test, >2 times).
    R equivalent: stats::mcnemar.test / DescTools::CochranQTest."""
    cc = wide.dropna()
    n = len(cc)
    mat = (cc.to_numpy() != 0).astype(int)
    if n < 1:
        return {"Test": " - ", "Statistic": np.nan, "P": np.nan, "N pairs": n,
                "Comment": "No complete pairs."}
    if len(times) == 2:
        table = pd.crosstab(mat[:, 0], mat[:, 1]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        b, c = table.loc[0, 1], table.loc[1, 0]
        # Exact test when few discordant pairs, asymptotic (with continuity corr.) otherwise.
        exact = (b + c) < 25
        res = mcnemar(table.to_numpy(), exact=exact, correction=not exact)
        return {"Test": "McNemar (exact)" if exact else "McNemar (asymptotic)",
                "Statistic": float(res.statistic), "P": float(res.pvalue), "N pairs": n,
                "Comment": f"Discordant pairs: {int(b)} / {int(c)}."}
    res = cochrans_q(mat)
    return {"Test": "Cochran's Q", "Statistic": float(res.statistic), "P": float(res.pvalue),
            "N pairs": n, "Comment": "Repeated binary measures (>2 timepoints)."}


def paired_tests_binary(db_long, id_col, time_col, outcome, *, group_col=None):
    """Omnibus within-subject test(s) for a binary outcome. Returns a DataFrame.

    References: McNemar 1947 (2 times); Cochran 1950 (Q test, >2 times).
    R equivalent: stats::mcnemar.test / DescTools::CochranQTest."""
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)
    rows = []
    overall = _binary_paired_test(_paired_wide(db_long, id_col, time_col, outcome, times), times)
    overall["Group"] = "All"
    rows.append(overall)
    if group_col is not None:
        for g in sorted(db_long[group_col].dropna().unique().tolist(), key=str):
            sub = db_long[db_long[group_col] == g]
            res = _binary_paired_test(_paired_wide(sub, id_col, time_col, outcome, times), times)
            res["Group"] = str(g)
            rows.append(res)
    cols = ["Group", "Test", "Statistic", "P", "N pairs", "Comment"]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# Models accounting for within-subject correlation
# --------------------------------------------------------------------------- #
def _build_formula(outcome, time_col, group_col, time_as):
    """outcome ~ time [+ group + time:group], time as factor or numeric trend."""
    time_term = f"C({time_col})" if time_as == "factor" else time_col
    if group_col is None:
        rhs = time_term
    else:
        rhs = f"{time_term} * C({group_col})"
    return f"{outcome} ~ {rhs}"


def fit_mixed_ols(db_long, id_col, time_col, outcome, *, group_col=None,
                  time_as="factor", random_slope=False):
    """
    Linear mixed model (``statsmodels.MixedLM``) for a continuous outcome.

    Random intercept by subject by default; a random slope on time is added when
    ``random_slope=True`` (only meaningful with numeric time). Returns the fitted
    results object (or ``None`` if the fit fails) plus a small info dict.

    References: Laird & Ware 1982 (random-effects models for longitudinal data).
    R equivalent: nlme::lme / lme4::lmer.
    """
    formula = _build_formula(outcome, time_col, group_col, time_as)
    re_formula = None
    if random_slope and time_as != "factor":
        re_formula = f"~{time_col}"
    info = {"Model": "Linear mixed model (MixedLM)", "Formula": formula,
            "Random effects": f"({'1 + ' + time_col if re_formula else '1'} | {id_col})"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, db_long, groups=db_long[id_col], re_formula=re_formula)
            res = md.fit(reml=True)
        info["Converged"] = bool(getattr(res, "converged", True))
        info["Log-Likelihood"] = float(res.llf) if res.llf is not None else np.nan
    except Exception as exc:
        info["Warning"] = f"MixedLM not fitted ({type(exc).__name__}: {exc})."
        return None, info
    return res, info


def diagnostics_mixed_ols(res, db_long, id_col, time_col):
    """
    Post-fit diagnostics for the gaussian mixed model.

    Two assumptions that matter for a longitudinal linear model:

    - **Normality of residuals** - Shapiro-Wilk on the model residuals
      (``p > 0.05`` = normality not rejected).
    - **Within-subject autocorrelation** - a Durbin-Watson statistic computed on
      the residuals ordered by time *inside each subject* (consecutive pairs that
      cross a subject boundary are excluded). ``DW ≈ 2`` means no autocorrelation,
      ``DW < 2`` positive autocorrelation (the exchangeable random-intercept
      structure is then too restrictive - an AR(1) error structure would fit
      better), ``DW > 2`` negative autocorrelation. Most informative with
      ``>2`` timepoints.

    Returns a ``(diag_info: dict, diag_df: DataFrame)`` pair; ``diag_df`` is the
    ``Metric / Value / Comment`` table written to the report.

    References: Shapiro & Wilk 1965 (normality); Durbin & Watson 1950, 1951 (autocorrelation).
    R equivalent: stats::shapiro.test / car::durbinWatsonTest (lmtest::dwtest).
    """
    rows = []
    info = {}
    if res is None:
        return info, pd.DataFrame(columns=["Metric", "Value", "Comment"])

    resid = pd.Series(np.asarray(res.resid, dtype=float), index=res.resid.index)

    # --- Residual normality (Shapiro-Wilk) ---
    if len(resid) >= 3:
        w, p = stats.shapiro(resid.to_numpy())
        info["Residual normality (Shapiro W)"] = round(float(w), 4)
        info["Residual normality p (>0.05 = ok)"] = float(p)
        rows.append({"Metric": "Residual normality (Shapiro-Wilk)",
                     "Value": f"W={w:.4f}, p={p:.4f}",
                     "Comment": "Normality not rejected." if p > 0.05
                     else "Normality rejected (p<0.05): interpret the p-values with caution."})
    else:
        rows.append({"Metric": "Residual normality (Shapiro-Wilk)", "Value": " - ",
                     "Comment": "Insufficient sample size (n<3)."})

    # --- Within-subject autocorrelation (Durbin-Watson) ---
    # Not relevant with 2 timepoints: a single gap per subject -> the statistic
    # is not informative. It is computed only from 3 timepoints onward.
    n_times = int(db_long.loc[resid.index, time_col].nunique())
    if n_times <= 2:
        rows.append({"Metric": "Within-subject autocorrelation (Durbin-Watson)",
                     "Value": " - ",
                     "Comment": "Not relevant with 2 timepoints (a single gap per subject)."})
        return info, pd.DataFrame(rows, columns=["Metric", "Value", "Comment"])

    diag = pd.DataFrame({"resid": resid,
                         id_col: db_long.loc[resid.index, id_col].values,
                         time_col: db_long.loc[resid.index, time_col].values})
    num = den = 0.0
    for _, g in diag.sort_values([id_col, time_col]).groupby(id_col, sort=False):
        e = g["resid"].to_numpy(float)
        if len(e) >= 2:
            num += float(np.sum(np.diff(e) ** 2))
        den += float(np.sum(e ** 2))
    if den > 0 and num > 0:
        dw = num / den
        info["Within-subject Durbin-Watson (~2 = no autocorr.)"] = round(dw, 3)
        if dw < 1.5:
            comment = "Positive autocorrelation: consider an AR(1) structure (see GEE ar1)."
        elif dw > 2.5:
            comment = "Negative autocorrelation."
        else:
            comment = "No notable within-subject autocorrelation."
        rows.append({"Metric": "Within-subject autocorrelation (Durbin-Watson)",
                     "Value": f"{dw:.3f}", "Comment": comment})
    else:
        rows.append({"Metric": "Within-subject autocorrelation (Durbin-Watson)",
                     "Value": " - ",
                     "Comment": "Not computable (a single timepoint per subject or null residuals)."})

    return info, pd.DataFrame(rows, columns=["Metric", "Value", "Comment"])



def compare_corr_structures(db_long, id_col, time_col, outcome, *, group_col=None,
                            time_as="factor", family="gaussian"):
    """
    Compare an EXCHANGEABLE vs an AR(1) working correlation by GEE QIC, and
    recommend one. Returns ``(info: dict, df)``.

    This turns the qualitative Durbin-Watson autocorrelation diagnostic into a
    quantitative decision: exchangeable assumes every pair of timepoints is
    equally correlated, AR(1) lets the correlation decay with the time gap
    (corr ∝ ρ^|i−j|) - usually more realistic for a temporal follow-up. The lower
    QIC wins. Most informative with ≥3 timepoints.

    References: Liang & Zeger 1986 (GEE working correlations); Pan 2001 (QIC).
    R equivalent: geepack::geeglm + geepack::QIC / MuMIn::QIC.
    """
    fam = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
           "poisson": sm.families.Poisson()}.get(family.lower())
    if fam is None:
        raise ValueError("family must be gaussian / binomial / poisson")
    formula = _build_formula(outcome, time_col, group_col, time_as)
    structs = {"exchangeable": sm.cov_struct.Exchangeable(),
               "ar1": sm.cov_struct.Autoregressive()}
    rows, qics = [], {}
    for name, cs in structs.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.gee(formula, groups=id_col, data=db_long,
                              family=fam, cov_struct=cs).fit()
            q = res.qic()
            qic = float(q[0] if isinstance(q, (tuple, list, np.ndarray)) else q)
        except Exception as exc:  # pragma: no cover - numerical safeguard
            qic = np.nan
            rows.append({"Working correlation": name, "QIC": np.nan,
                         "Comment": f"Not fitted ({type(exc).__name__})."})
            continue
        qics[name] = qic
        rows.append({"Working correlation": name, "QIC": round(qic, 2), "Comment": ""})
    info = {}
    valid = {k: v for k, v in qics.items() if np.isfinite(v)}
    if valid:
        best = min(valid, key=valid.get)
        info["Recommended working correlation (lower QIC)"] = best
        if {"exchangeable", "ar1"} <= set(valid):
            delta = valid["exchangeable"] - valid["ar1"]
            info["QIC gain AR1 vs exchangeable"] = round(delta, 2)
            info["Correlation note"] = (
                "AR(1) preferable (lower QIC): decaying temporal correlation."
                if best == "ar1" else
                "Exchangeable sufficient: no QIC gain for AR(1).")
    return info, pd.DataFrame(rows, columns=["Working correlation", "QIC", "Comment"])


def fit_gee_logit(db_long, id_col, time_col, outcome, *, group_col=None,
                  time_as="factor", cov_struct="exchangeable"):
    """
    Marginal logistic model for a binary outcome via GEE
    (``statsmodels.GEE``), clustering on the subject id with an exchangeable (or
    AR1 / independence) working correlation. Returns ``(results | None, info)``.

    References: Liang & Zeger 1986 (GEE, logit link).
    R equivalent: geepack::geeglm(family = binomial) / gee::gee.
    """
    formula = _build_formula(outcome, time_col, group_col, time_as)
    cs = {"exchangeable": sm.cov_struct.Exchangeable(),
          "ar1": sm.cov_struct.Autoregressive(),
          "independence": sm.cov_struct.Independence()}.get(cov_struct.lower())
    if cs is None:
        raise ValueError("cov_struct must be exchangeable / ar1 / independence")
    info = {"Model": "Marginal logistic GEE", "Formula": formula,
            "Cluster": id_col, "Working correlation": cov_struct.lower()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.gee(formula, groups=id_col, data=db_long,
                         family=sm.families.Binomial(), cov_struct=cs)
            res = md.fit()
        info["Converged"] = bool(getattr(res, "converged", True))
    except Exception as exc:
        info["Warning"] = f"GEE not fitted ({type(exc).__name__}: {exc})."
        return None, info
    return res, info


# --------------------------------------------------------------------------- #
# Effects export
# --------------------------------------------------------------------------- #
def _effects_table(res, *, exponentiate):
    """β (95% CI) + p, or OR (95% CI) + p when exponentiate=True. Skips intercept/group var."""
    if res is None:
        cols = (["Term", "OR (95% CI)", "P"] if exponentiate
                else ["Term", "Beta (95% CI)", "P"])
        return pd.DataFrame(columns=cols)
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues
    rows = []
    for term in params.index:
        if term in ("Intercept", "Group Var") or term.endswith("Var"):
            continue
        coef = params[term]
        lo, hi = conf.loc[term]
        if exponentiate:
            rows.append({"Term": term,
                         "OR (95% CI)": f"{np.exp(coef):.3f} ({np.exp(lo):.3f} to {np.exp(hi):.3f})",
                         "P": f"{pvals[term]:.4f}"})
        else:
            rows.append({"Term": term,
                         "Beta (95% CI)": f"{coef:.3f} ({lo:.3f} to {hi:.3f})",
                         "P": f"{pvals[term]:.4f}"})
    cols = (["Term", "OR (95% CI)", "P"] if exponentiate
            else ["Term", "Beta (95% CI)", "P"])
    return pd.DataFrame(rows, columns=cols)


def _write_report(save_path, filename, info, bytime_df, tests_df, effects_df, verbose,
                  diag_df=None, extra_sheets=None):
    """Write the Excel report (adds a Diagnostics sheet when diag_df is given, and
    any ``extra_sheets`` {name: df}); mkdir as needed. Returns the full path."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full = os.path.join(save_path, filename)
    info_df = pd.DataFrame([{"Parameter": k, "Value": (", ".join(map(str, v)) if isinstance(v, list) else v)}
                            for k, v in info.items()], columns=["Parameter", "Value"])
    with pd.ExcelWriter(full, engine="xlsxwriter") as writer:
        info_df.to_excel(writer, sheet_name="Setup Info", index=False)
        effects_df.to_excel(writer, sheet_name="Model Effects", index=False)
        if diag_df is not None:
            diag_df.to_excel(writer, sheet_name="Diagnostics", index=False)
        for name, df in (extra_sheets or {}).items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
        tests_df.to_excel(writer, sheet_name="Paired Tests", index=False)
        bytime_df.to_excel(writer, sheet_name="By Time", index=False)
    if verbose:
        print(f"Report exported to {full}")
    return full


# --------------------------------------------------------------------------- #
# Pipelines (one per outcome type)
# --------------------------------------------------------------------------- #
def pipeline_paired_ols(db, *, id_col, outcome, time_col=None, time_map=None,
                        group_col=None, covariates=None, time_as="factor",
                        random_slope=False, impute="none", n_iter=10,
                        compare_ar1=True, save_path=None, verbose=True):
    """
    End-to-end paired / repeated-measures pipeline for a CONTINUOUS outcome.

    Accepts wide (``time_map``) or long (``time_col``) input, audits completeness,
    optionally imputes (``impute`` ∈ {"none", "mice", "locf"}), summarises by
    timepoint, runs the omnibus paired test(s) (paired t / Wilcoxon for 2 times,
    RM-ANOVA / Friedman for >2), fits a linear mixed model
    (``outcome ~ time [+ group + time:group]`` with a random intercept [+ slope]
    by subject), runs model diagnostics (Shapiro-Wilk normality + within-subject
    Durbin-Watson) and - when ``compare_ar1`` - a QIC comparison of exchangeable
    vs AR(1) working correlation. With ``save_path`` set, writes the Excel report
    (sheets incl. ``Missingness`` and ``Correlation Structure``).

    Returns ``(model_info: dict, mixed_results | None)``; missingness, imputation
    flag, diagnostics and the AR1 recommendation are folded into ``model_info``.

    References: Laird & Ware 1982 (mixed model core); omnibus tests as above.
    R equivalent: nlme::lme / lme4::lmer (no single pipeline equivalent - to test).
    """
    db_long, time_col, report, miss_df, _ = prepare_repeated(
        db, id_col, outcome, time_col=time_col, time_map=time_map,
        group_col=group_col, covariates=covariates, impute=impute,
        n_iter=n_iter, binary=False)

    bytime_df = describe_by_time(db_long, id_col, time_col, outcome,
                                 group_col=group_col, continuous=True)
    tests_df = paired_tests_continuous(db_long, id_col, time_col, outcome, group_col=group_col)
    res, model_info_fit = fit_mixed_ols(db_long, id_col, time_col, outcome,
                                        group_col=group_col, time_as=time_as,
                                        random_slope=random_slope)
    effects_df = _effects_table(res, exponentiate=False)
    diag_info, diag_df = diagnostics_mixed_ols(res, db_long, id_col, time_col)

    model_info = {"Outcome type": "Continuous", "Outcome": outcome,
                  "Subject id": id_col, "Time": time_col,
                  "Group": group_col if group_col else " - ", "Time encoding": time_as}
    model_info.update(report)
    model_info.update(model_info_fit)
    model_info.update(diag_info)

    corr_df = None
    if compare_ar1 and len(report.get("Timepoints", [])) >= 3:
        corr_info, corr_df = compare_corr_structures(
            db_long, id_col, time_col, outcome, group_col=group_col,
            time_as=time_as, family="gaussian")
        model_info.update(corr_info)

    if save_path is not None:
        sub = os.path.join(save_path, "Longitudinal models", "Paired_OLS")
        _write_report(sub, f"{outcome}_paired_ols_report.xlsx",
                      model_info, bytime_df, tests_df, effects_df, verbose,
                      diag_df=diag_df,
                      extra_sheets={"Missingness": miss_df,
                                    "Correlation Structure": corr_df})
    elif verbose:
        print("Report not exported")
    return model_info, res


def pipeline_paired_logit(db, *, id_col, outcome, time_col=None, time_map=None,
                          group_col=None, covariates=None, time_as="factor",
                          cov_struct="exchangeable", impute="none", n_iter=10,
                          compare_ar1=True, save_path=None, verbose=True):
    """
    End-to-end paired / repeated-measures pipeline for a BINARY outcome.

    Same flow as ``pipeline_paired_ols`` (completeness audit, optional MICE/LOCF
    imputation, AR1-vs-exchangeable QIC comparison) but with McNemar (2 times) /
    Cochran's Q (>2 times) as the omnibus test and a marginal logistic GEE
    (``cov_struct`` working correlation, clustered on subject) as the model;
    effects are exported as odds ratios.

    Returns ``(model_info: dict, gee_results | None)``.

    References: Liang & Zeger 1986 (marginal logistic GEE); McNemar 1947 / Cochran 1950 (omnibus).
    R equivalent: geepack::geeglm(family = binomial) - potential pipeline equivalent, to test.
    """
    db_long, time_col, report, miss_df, _ = prepare_repeated(
        db, id_col, outcome, time_col=time_col, time_map=time_map,
        group_col=group_col, covariates=covariates, impute=impute,
        n_iter=n_iter, binary=True)

    bytime_df = describe_by_time(db_long, id_col, time_col, outcome,
                                 group_col=group_col, continuous=False)
    tests_df = paired_tests_binary(db_long, id_col, time_col, outcome, group_col=group_col)
    res, model_info_fit = fit_gee_logit(db_long, id_col, time_col, outcome,
                                        group_col=group_col, time_as=time_as,
                                        cov_struct=cov_struct)
    effects_df = _effects_table(res, exponentiate=True)

    model_info = {"Outcome type": "Binary", "Outcome": outcome,
                  "Subject id": id_col, "Time": time_col,
                  "Group": group_col if group_col else " - ", "Time encoding": time_as}
    model_info.update(report)
    model_info.update(model_info_fit)

    corr_df = None
    if compare_ar1 and len(report.get("Timepoints", [])) >= 3:
        corr_info, corr_df = compare_corr_structures(
            db_long, id_col, time_col, outcome, group_col=group_col,
            time_as=time_as, family="binomial")
        model_info.update(corr_info)

    if save_path is not None:
        sub = os.path.join(save_path, "Longitudinal models", "Paired_Logit")
        _write_report(sub, f"{outcome}_paired_logit_report.xlsx",
                      model_info, bytime_df, tests_df, effects_df, verbose,
                      extra_sheets={"Missingness": miss_df,
                                    "Correlation Structure": corr_df})
    elif verbose:
        print("Report not exported")
    return model_info, res


# --------------------------------------------------------------------------- #
# Multivariate repeated measures (several outcomes measured at each timepoint)
# --------------------------------------------------------------------------- #
def to_long_multi(db, id_col, outcome_maps, *, group_col=None, covariates=None):
    """
    Reshape a WIDE table into long form for a MULTIVARIATE repeated-measures
    design - several outcomes, each measured at every timepoint.

    Parameters
    ----------
    outcome_maps : dict[str, dict[str, str]]
        ``{outcome_name: {time_label: wide_column}}``. Every outcome must declare
        the **same set of time labels** (they index the same visits), e.g.::

            {"hr":  {"pre": "hr_V1",  "post": "hr_V2"},
             "sbp": {"pre": "sbp_V1", "post": "sbp_V2"}}

    Returns
    -------
    pd.DataFrame with columns ``[id_col, "time", *outcomes, (group_col), *covariates]``.

    R equivalent: tidyr::pivot_longer (names_to / values_to on several outcomes).
    """
    if not isinstance(outcome_maps, dict) or not outcome_maps:
        raise ValueError("outcome_maps must be a non-empty dict {outcome: {time: column}}.")
    time_sets = [tuple(sorted(m, key=str)) for m in outcome_maps.values()]
    if len(set(time_sets)) != 1:
        raise ValueError("All outcomes must share the same time labels.")
    time_labels = list(dict.fromkeys(next(iter(outcome_maps.values())).keys()))

    base = None
    for name, tmap in outcome_maps.items():
        long_one = to_long(db, id_col, tmap, outcome=name,
                           group_col=group_col, covariates=covariates)
        keys = [id_col, "time"] + ([group_col] if group_col else []) + (covariates or [])
        keys = list(dict.fromkeys(keys))
        base = long_one if base is None else base.merge(long_one, on=keys, how="outer")
    ordered = [id_col, "time"] + list(outcome_maps) + \
        ([group_col] if group_col else []) + (covariates or [])
    ordered = list(dict.fromkeys(ordered))
    # honour the declared chronological order of the timepoints
    base["time"] = pd.Categorical(base["time"], categories=time_labels, ordered=True)
    return base.sort_values([id_col, "time"])[ordered].reset_index(drop=True)


def pipeline_paired_manova(db, *, id_col, outcomes, time_col=None, outcome_maps=None,
                           group_col=None, covariates=None, n_perm=4999,
                           distance="euclidean", standardize=True,
                           per_outcome_followup=True, random_state=0,
                           save_path=None, verbose=True):
    """
    End-to-end paired / repeated-measures pipeline for a MULTIVARIATE outcome
    (several continuous measures recorded at each timepoint).

    The omnibus test is a PERMANOVA on the joint response with restricted
    within-subject permutation (``permanova_paired``) - the multivariate analogue
    of the RM-ANOVA, with no normality/sphericity assumption. As a follow-up
    (``per_outcome_followup``) each outcome is also run through the usual
    univariate paired test, with a Holm-adjusted p-value to control the
    family-wise error across outcomes.

    Accepts wide (``outcome_maps``, see ``to_long_multi``) or already-long
    (``time_col`` + several ``outcomes`` columns) input. With ``save_path`` set,
    writes a report (sheets ``Setup Info``, ``PERMANOVA``, ``Per-Outcome Tests``,
    ``By Time``).

    Returns ``(model_info: dict, permanova_info: dict)``.

    References: Anderson 2001 (PERMANOVA); Holm 1979 (family-wise correction).
    R equivalent: vegan::adonis2 + stats::p.adjust(method = "holm").
    """
    if isinstance(outcomes, str):
        outcomes = [outcomes]

    # --- reshape / validate (long, multivariate) ---
    if outcome_maps is not None:
        db_long = to_long_multi(db, id_col, outcome_maps,
                                group_col=group_col, covariates=covariates)
        time_col = "time"
        outcomes = list(outcome_maps)
    elif time_col is not None:
        db_long = db
    else:
        raise ValueError("Provide either outcome_maps (wide) or time_col (long).")
    missing = [c for c in outcomes if c not in db_long.columns]
    if missing:
        raise KeyError(f"Outcome columns missing: {missing}")
    times = sorted(db_long[time_col].dropna().unique().tolist(), key=str)

    # --- descriptive per timepoint, stacked over outcomes ---
    bytime_parts = []
    for y in outcomes:
        d = describe_by_time(db_long, id_col, time_col, y,
                             group_col=group_col, continuous=True)
        d.insert(0, "Outcome", y)
        bytime_parts.append(d)
    bytime_df = pd.concat(bytime_parts, ignore_index=True)

    # --- omnibus multivariate PERMANOVA (overall + per group) ---
    perm_rows = []
    overall = permanova_paired(db_long, id_col, time_col, outcomes, n_perm=n_perm,
                               distance=distance, standardize=standardize,
                               random_state=random_state)
    perm_rows.append({"Group": "All", "Pseudo-F": overall.get("Pseudo-F", np.nan),
                      "P": overall.get("P", np.nan), "R2": overall.get("R2", np.nan),
                      "N subjects": overall.get("N subjects", np.nan),
                      "Comment": overall["Comment"]})
    if group_col is not None:
        for g in sorted(db_long[group_col].dropna().unique().tolist(), key=str):
            sub = db_long[db_long[group_col] == g]
            gi = permanova_paired(sub, id_col, time_col, outcomes, n_perm=n_perm,
                                  distance=distance, standardize=standardize,
                                  random_state=random_state)
            perm_rows.append({"Group": str(g), "Pseudo-F": gi.get("Pseudo-F", np.nan),
                              "P": gi.get("P", np.nan), "R2": gi.get("R2", np.nan),
                              "N subjects": gi.get("N subjects", np.nan),
                              "Comment": gi["Comment"]})
    permanova_df = pd.DataFrame(perm_rows,
                                columns=["Group", "Pseudo-F", "P", "R2", "N subjects", "Comment"])

    # --- per-outcome univariate follow-up (Holm-adjusted across outcomes) ---
    followup_df = pd.DataFrame()
    if per_outcome_followup:
        rows = []
        for y in outcomes:
            res = _continuous_paired_test(
                _paired_wide(db_long, id_col, time_col, y, times), times)
            rows.append({"Outcome": y, "Test": res["Test"],
                         "Statistic": res["Statistic"], "P": res["P"],
                         "N pairs": res["N pairs"]})
        followup_df = pd.DataFrame(rows)
        followup_df["P (Holm)"] = _holm_adjust(followup_df["P"].to_numpy(float))

    model_info = {"Outcome type": "Multivariate (continuous)",
                  "Outcomes": list(outcomes), "Subject id": id_col, "Time": time_col,
                  "Group": group_col if group_col else " - ",
                  "Number of timepoints": len(times),
                  "Number of subjects": int(db_long[id_col].nunique()),
                  "Omnibus test": overall["Test"], "Permutations": int(n_perm),
                  "Distance": distance, "Standardized": bool(standardize)}
    model_info["PERMANOVA Pseudo-F (overall)"] = overall.get("Pseudo-F", np.nan)
    model_info["PERMANOVA P (overall)"] = overall.get("P", np.nan)

    if save_path is not None:
        sub = os.path.join(save_path, "Longitudinal models", "Paired_MANOVA")
        if not os.path.exists(sub):
            os.makedirs(sub)
        tag = "_".join(outcomes)[:40]
        full = os.path.join(sub, f"{tag}_paired_manova_report.xlsx")
        info_df = pd.DataFrame([{"Parameter": k,
                                 "Value": (", ".join(map(str, v)) if isinstance(v, list) else v)}
                                for k, v in model_info.items()], columns=["Parameter", "Value"])
        with pd.ExcelWriter(full, engine="xlsxwriter") as writer:
            info_df.to_excel(writer, sheet_name="Setup Info", index=False)
            permanova_df.to_excel(writer, sheet_name="PERMANOVA", index=False)
            if not followup_df.empty:
                followup_df.to_excel(writer, sheet_name="Per-Outcome Tests", index=False)
            bytime_df.to_excel(writer, sheet_name="By Time", index=False)
        if verbose:
            print(f"Report exported to {full}")
    elif verbose:
        print("Report not exported")

    return model_info, overall


def _holm_adjust(pvals):
    """Holm step-down family-wise correction. Returns adjusted p-values (NaN-safe).

    References: Holm 1979 (step-down Bonferroni).
    R equivalent: stats::p.adjust(method = "holm")."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = idx.size
    running = 0.0
    for rank, j in enumerate(order):
        val = (m - rank) * p[j]
        running = max(running, val)          # enforce monotonicity
        out[j] = min(1.0, running)
    return out
