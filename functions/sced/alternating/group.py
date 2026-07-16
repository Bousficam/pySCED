"""
SCED - Alternating Treatments / N-of-1, GROUP level (replicated single-case)
============================================================================

Group counterpart of ``SCED_alternating`` (family 1): several units each run their
own randomised alternating-treatments series, and a single inference about a COMMON
condition effect is wanted. The primary inference stays a **randomization test**,
now **stratified** with the unit as a block (Edgington & Onghena 2007): each unit
was randomised independently, so under the sharp null the joint randomisation is the
product of the per-unit schedules - the condition labels are permuted **within each
unit** and one pooled model is refit.

The pooled model is ``outcome ~ [unit intercepts (+ unit-specific time slope if
detrend)] + condition``; the shared condition block is tested by its partial F
(Winkler et al. 2014). Outcomes are z-scored within unit by default so units
contribute comparably (permutation-invariant, hence still exact). Per-unit and
aggregated **Tau-U / NAP** are reported alongside as descriptive effect sizes.

House contract::

    model_info, rand = pipeline_sced_alternating_group(
        df, unit_col="patient", session_col="session", condition_col="cond",
        outcome_col="y", conditions=("A", "B"), save_path=save_path)
"""
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.anova import AnovaRM

from functions.sced.core import (
    validate_sced,
    sced_data_warnings,
    missing_data_warnings,
    nap,
    tau_u,
    pnd,
    pem,
    diagnose_series,
    condition_permutation_test,
    stratified_condition_permutation_test,
    stratified_ordered_trend_test,
    choose_permutation_method_group,
    heterogeneity_test,
    pairwise_contrasts,
    lag1_autocorr,
    durbin_watson,
    sens_slope,
    mann_kendall,
    sced_test_name,
    sced_methods_text,
    nap_magnitude,
    tau_magnitude,
    pnd_magnitude,
    pem_magnitude,
    i2_band,
    I2_BANDS,
    NAP_BANDS,
    TAU_BANDS,
    PND_BANDS,
    PEM_BANDS,
    _group_nuisance,
)
from functions.sced.alternating.core import _effect_sizes_table
from functions.sced.power import power_report


def _mauchly_sphericity(Y):
    """Mauchly's test of sphericity on an n x k repeated-measures matrix, plus the
    Greenhouse-Geisser epsilon. Returns NaNs for k<3 (sphericity is trivial) or n<=k.

    References: Mauchly 1940 (sphericity test); Greenhouse & Geisser 1959 (epsilon).
    R equivalent: stats::mauchly.test.
    """
    n, k = Y.shape
    if k < 3 or n <= k:
        return {"W": np.nan, "p": np.nan, "gg_epsilon": np.nan}
    S = np.cov(Y, rowvar=False, ddof=1)
    C = np.linalg.svd(np.eye(k) - np.ones((k, k)) / k)[2][:k - 1]   # orthonormal contrasts
    eig = np.linalg.eigvalsh(C @ S @ C.T)
    eig = eig[eig > 1e-12]
    if len(eig) < k - 1:
        return {"W": np.nan, "p": np.nan, "gg_epsilon": np.nan}
    W = float(np.prod(eig) / np.mean(eig) ** (k - 1))
    dfm = k * (k - 1) // 2 - 1
    corr = 1 - (2 * (k - 1) ** 2 + (k - 1) + 2) / (6 * (k - 1) * (n - 1))
    chi = -(n - 1) * corr * np.log(W)
    return {"W": W, "p": float(sps.chi2.sf(chi, dfm)),
            "gg_epsilon": float(eig.sum() ** 2 / ((k - 1) * (eig ** 2).sum()))}


def classical_rm_checks(df, *, unit_col, session_col, condition_col, outcome_col,
                        conditions=None):
    """
    Familiar **classical repeated-measures** cross-checks on the per-unit x condition
    means, with their assumption diagnostics - a secondary, reviewer-friendly view.

    Reports **Friedman** (rank RM-ANOVA, k>=3) and **RM-ANOVA** (F), each on the
    collapsed means, then the assumptions these tests rely on but the SCED structure
    tends to break: **normality** of residuals (Shapiro-Wilk), **sphericity**
    (Mauchly's W + Greenhouse-Geisser epsilon), **autocorrelation** (lag-1 / Durbin-
    Watson, within unit) and the **temporal slope** (Sen + Mann-Kendall). The last two
    are the reason these tests are only a cross-check here: collapsing to means and
    ignoring the time trend / serial dependence is exactly what the primary
    randomization test avoids. Returns a tidy DataFrame.

    References: Friedman 1937; Shapiro & Wilk 1965; Mauchly 1940 (sphericity);
    Durbin & Watson 1950; Mann 1945, Kendall 1938 (trend); Sen 1968 (slope).
    R equivalent: stats::friedman.test; afex::aov_ez (RM-ANOVA); stats::shapiro.test;
    stats::mauchly.test.
    """
    if conditions is None:
        conditions = sorted(df[condition_col].dropna().unique().tolist(), key=str)
    conditions = list(conditions)
    d = df[df[condition_col].isin(conditions)].copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    means = (d.pivot_table(index=unit_col, columns=condition_col, values=outcome_col,
                           aggfunc="mean").reindex(columns=conditions).dropna())
    n_units, k = means.shape
    rows = []  # (Test, Statistic, p-value, Interpretation, Note)

    def _sig(p, yes, no):
        return yes if (np.isfinite(p) and p < 0.05) else no

    if k >= 3 and n_units >= 2:
        chi2v, pf = sps.friedmanchisquare(*[means[c].to_numpy() for c in conditions])
        rows.append(("Friedman chi2", round(float(chi2v), 4), round(float(pf), 4),
                     _sig(pf, "The conditions differ (on ranks).",
                          "No difference between conditions detected."),
                     f"Nonparametric RM-ANOVA on means, k={k}, n={n_units}"))
    else:
        rows.append(("Friedman chi2", "n/a", "n/a", "Not applicable.",
                     "requires k>=3 and >=2 units"))

    try:
        long = means.reset_index().melt(id_vars=unit_col, var_name=condition_col,
                                        value_name=outcome_col)
        tab = AnovaRM(long, depvar=outcome_col, subject=unit_col,
                      within=[condition_col]).fit().anova_table
        pa = float(tab["Pr > F"].iloc[0])
        rows.append(("RM-ANOVA F", round(float(tab["F Value"].iloc[0]), 4), round(pa, 4),
                     _sig(pa, "Significant condition effect (parametric).",
                          "No condition effect detected."),
                     f"df=({tab['Num DF'].iloc[0]:.0f},{tab['Den DF'].iloc[0]:.0f}) on means"))
        gm = means.to_numpy().mean()
        resid = (means.to_numpy() - means.mean(axis=1).to_numpy()[:, None]
                 - means.mean(axis=0).to_numpy()[None, :] + gm).ravel()
        if len(resid) >= 3:
            Wn, pn = sps.shapiro(resid)
            rows.append(("Residual normality (Shapiro-Wilk)", round(float(Wn), 4), round(float(pn), 4),
                         _sig(pn, "Residuals NOT normal: RM-ANOVA fragile -> prefer permutation.",
                              "Normality plausible."),
                         "RM-ANOVA assumption"))
    except Exception:
        rows.append(("RM-ANOVA F", "n/a", "n/a", "Computation failed.",
                     "missing patient x condition cells?"))

    mau = _mauchly_sphericity(means.to_numpy())
    if np.isfinite(mau["p"]):
        rows.append(("Sphericity (Mauchly W)", round(mau["W"], 4), round(mau["p"], 4),
                     _sig(mau["p"],
                          f"Sphericity VIOLATED: correct with Greenhouse-Geisser (epsilon={mau['gg_epsilon']:.2f}).",
                          f"Sphericity plausible (epsilon={mau['gg_epsilon']:.2f})."),
                     "RM-ANOVA assumption (k>=3)"))
    else:
        rows.append(("Sphericity (Mauchly W)", "n/a", "n/a", "Trivial (k=2) or n<=k.",
                     " - "))

    r1, dw, sl, mk = [], [], [], []
    for _, g in d.groupby(unit_col):
        g = g.sort_values(session_col)
        y = g[outcome_col].to_numpy()
        s = pd.to_numeric(g[session_col], errors="coerce").to_numpy()
        r1.append(lag1_autocorr(y - np.nanmean(y))); dw.append(durbin_watson(y - np.nanmean(y)))
        sl.append(sens_slope(y, x=s)); mk.append(mann_kendall(y)["p_value"])
    mr1, mdw, msl, mmk = (float(np.nanmean(v)) for v in (r1, dw, sl, mk))
    rows.append(("Lag-1 autocorrelation (mean within-unit)", round(mr1, 4), " - ",
                 ("Marked serial dependence: RM-ANOVA independence violated -> permutation preferable."
                  if abs(mr1) > 0.2 else "Weak autocorrelation."),
                 "!=0 violates the assumed independence"))
    rows.append(("Durbin-Watson (mean within-unit)", round(mdw, 4), " - ",
                 ("Autocorrelation (DW departs from 2)." if (mdw < 1.5 or mdw > 2.5)
                  else "No marked autocorrelation."),
                 "~2 = independence"))
    rows.append(("Sen temporal slope (mean within-unit)", round(msl, 4), round(mmk, 4),
                 _sig(mmk,
                      "SIGNIFICANT learning: Friedman/RM-ANOVA confound it with the condition -> detrend justified (primary test).",
                      "No clear temporal trend."),
                 "!=0 = learning ignored by Friedman/RM-ANOVA"))
    out = pd.DataFrame(rows, columns=["Test / Check", "Statistic", "p-value",
                                      "Interpretation", "Explanation"])
    return out[["Test / Check", "Statistic", "p-value", "Explanation", "Interpretation"]]


# --------------------------------------------------------------------------- #
# Report annotation: add an "Explanation" (what the stat is) and "Interpretation"
# (verdict for THIS data) column to every results sheet.
# --------------------------------------------------------------------------- #
def _g_mag(g):
    """Verbal magnitude band for a standardised effect g (Cohen cutoffs 0.2/0.5/0.8)."""
    a = abs(g)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _nap_mag(nap):
    """Verbal magnitude band for a NAP nonoverlap value (0.66 / 0.92 cutoffs)."""
    return "strong" if nap > 0.92 else "medium" if nap >= 0.66 else "weak/none"


def _sig_txt(p):
    """Short significance verdict string for a p-value (0.05 threshold)."""
    return f"significant (p={p:.3g})" if (np.isfinite(p) and p < 0.05) else f"not significant (p={p:.3g})"


_SETUP_EXPLAIN = {
    "Design": "Analysed design.", "Outcome type": "Measurement scale.",
    "Outcome": "Measured variable.", "Number of units": "Number of patients/units.",
    "Conditions": "Compared conditions (reference first).",
    "Improvement direction": "Direction of improvement (for Tau-U/NAP).",
    "Primary inference (test)": "NAME of the primary test (permutation repeated-measures ANOVA/ANCOVA).",
    "Test statistic": "Permuted statistic (partial F / difference of means).",
    "Detrend": "Learning-trend removal (none/linear/log).",
    "Per-unit detrend suggestion": "Suggested trend per patient.",
    "Permutation method": "Permutation scheme (FL/DS).",
    "Within-unit standardization": "Within-patient z-score (comparability).",
    "Permutations": "Number of resamples.",
    "Max consecutive constraint": "Assignment-schedule constraint.",
    "Degrees of freedom (num, den)": "df (k-1; residual).",
    "Randomization p-value": "p of the primary test (common effect).",
    "Observed statistic": "Observed partial F.",
    "Data warnings": "Data-quality alerts.",
}


def _annotate_randomization(df, p):
    """Add Explanation/Interpretation columns to the stratified randomization sheet;
    the verdict reflects whether the common condition effect is significant at 0.05."""
    df = df.copy()
    df["Explanation"] = ("Stratified randomization test: permutes conditions within "
                         "patient; partial F = COMMON condition effect beyond time.")
    df["Interpretation"] = ("COMMON group-level condition effect - " + _sig_txt(p) + "."
                            if p < 0.05 else "No common effect detected - " + _sig_txt(p) + ".")
    return df


def _annotate_heterogeneity(df, het=None):
    """One row per read-out (all-conditions interaction / Q-I2 per contrast / dose-
    response slope); per-row verdict derived from p and I2."""
    df = df.copy()
    df["Explanation"] = (
        "Question: does the effect vary BETWEEN PATIENTS? (!= Post-hoc, which asks WHICH "
        "conditions differ). Three read-outs: patient x condition interaction (permutation, "
        "all conditions); Cochran's Q / I2 per contrast; Q/I2 on the within-patient dose-"
        "response slope. Complements the COMMON effect: a mean effect can mask unequal "
        f"(even opposite) responders. I2 bands - {I2_BANDS}. Ref. Cochran 1954; Higgins & "
        "Thompson 2002.")

    def verdict(r):
        p, i2 = r.get("p"), r.get("I2 (%)")
        p_ok = isinstance(p, (int, float)) and np.isfinite(p) and p < 0.05
        i2v = i2 if isinstance(i2, (int, float)) and np.isfinite(i2) else np.nan
        band = i2_band(i2v) if np.isfinite(i2v) else "n/a"
        i2_ok = np.isfinite(i2v) and i2v > 50
        kind = "HETEROGENEOUS" if (p_ok or i2_ok) else "homogeneous"
        return f"{kind} (I2={i2 if i2 != '' else 'n/a'}% -> {band}, p={p})"
    df["Interpretation"] = df.apply(verdict, axis=1)
    return df


def _annotate_posthoc(df):
    """Add Explanation/Interpretation columns to the pairwise post-hoc sheet; the
    verdict uses the Holm-corrected p (falling back to the raw permutation p)."""
    df = df.copy()
    df["Explanation"] = (
        "Question: WHICH conditions differ? (!= Heterogeneity, which asks whether the effect "
        "varies between patients). After a significant global test, each pair of conditions "
        "is compared: Hedges' g (standardised effect; Cohen 0.2 small / 0.5 medium / 0.8 large) "
        "+ bootstrap CI, permutation p corrected Holm (FWER) / FDR. Read the CORRECTED p.")
    pcol = "p Holm" if "p Holm" in df.columns else "p (perm)"

    def verdict(r):
        g = r["Hedges g"]; p = r[pcol]
        sens = "+" if g > 0 else "-"
        return (f"{'Significant' if (np.isfinite(p) and p < 0.05) else 'NS'} after correction, "
                f"{_g_mag(g)} effect ({sens}).")
    df["Interpretation"] = df.apply(verdict, axis=1)
    return df


def _annotate_effect_sizes(df):
    """Add Explanation/Interpretation columns to the aggregated effect-size sheet; the
    verdict summarises the NAP band per comparison."""
    df = df.copy()
    df["Explanation"] = ("Pairwise nonoverlap (averaged across patients). Strength-column "
                         f"bands - NAP: {NAP_BANDS}; Tau-U: {TAU_BANDS}; "
                         f"PND: {PND_BANDS}; PEM: {PEM_BANDS}.")
    df["Interpretation"] = df.apply(
        lambda r: f"{_nap_mag(r['NAP'])} effect (NAP={r['NAP']}, Tau-U={r['Tau-U']}).",
        axis=1)
    return df


def _annotate_per_unit(df, conditions):
    """Add Explanation/Interpretation columns to the per-patient sheet; the verdict
    combines the Holm-corrected individual test, nonoverlap and temporal trend."""
    df = df.copy()
    ref, trt = conditions[0], conditions[1]
    napcol = f"NAP[{trt}v{ref}]"
    df["Explanation"] = (
        "Per patient (n-of-1): n and mean per condition; Tau-U/NAP nonoverlap for EACH "
        "pair; within-patient temporal slope (OLS + Sen, conditions centred); Mann-Kendall"
        "+Hamed-Rao trend; INDIVIDUAL randomization test (individual p, + Holm/FDR corrected "
        "across all patients). "
        f"Bands - NAP: {NAP_BANDS}; Tau-U: {TAU_BANDS}. "
        "NB: the 'Tau-U critical' depends only on the number of measures per condition; it is "
        "therefore IDENTICAL across patients when they have the same number of sessions per "
        "condition (this is not an error).")

    def verdict(r):
        parts = []
        p = r.get("Individual p", np.nan)
        ph = r.get("Individual p Holm", np.nan)
        if np.isfinite(p):
            # the verdict relies on the CORRECTED p (Holm) to avoid over-interpreting the screening
            sig = np.isfinite(ph) and ph < 0.05
            parts.append(f"individual effect {'significant' if sig else 'NS'} after Holm "
                         f"(raw p={p}, Holm={ph})")
        napv = r.get(napcol, np.nan)
        tauv = r.get(f"Tau-U[{trt}v{ref}]", np.nan)
        if np.isfinite(napv):
            parts.append(f"nonoverlap {nap_magnitude(napv)} (NAP={napv}, Tau-U "
                         f"{tau_magnitude(tauv)}={tauv})")
        tp = r.get("Trend p (MK+HR)", np.nan)
        if np.isfinite(tp) and tp < 0.05:
            parts.append("temporal trend present")
        return " ; ".join(parts) if parts else " - "
    df["Interpretation"] = df.apply(verdict, axis=1)
    return df


def _annotate_setup(df):
    """Add an Explanation column to the Setup Info sheet by mapping each Parameter to
    its short gloss; the Interpretation column is left blank (setup rows have no verdict)."""
    df = df.copy()
    df["Explanation"] = df["Parameter"].map(_SETUP_EXPLAIN).fillna(" - ")
    df["Interpretation"] = " - "
    return df


def _group_detrend_suggestion(clean, *, unit_col, session_col, outcome_col, condition_col):
    """Aggregate the per-unit ``diagnose_series`` suggestions into one detrend type:
    if at least half the units show a trend, take the most frequent non-"none" basis,
    else "none". Returns ``(suggested, per_unit_dict)``."""
    per_unit = {}
    for u, g in clean.groupby(unit_col):
        if g[session_col].notna().sum() >= 4:
            per_unit[str(u)] = diagnose_series(g, session_col=session_col,
                                               outcome_col=outcome_col,
                                               condition_col=condition_col)["Suggested detrend"]
    suggs = list(per_unit.values())
    trended = [s for s in suggs if s != "none"]
    if suggs and len(trended) >= len(suggs) / 2:
        suggested = max(set(trended), key=trended.count)
    else:
        suggested = "none"
    return suggested, per_unit


def _per_unit_table(clean, *, unit_col, session_col, condition_col, outcome_col, conditions,
                    improvement, detrend="none", method="freedman-lane", max_consecutive=None,
                    n_perm=2000, random_state=0):
    """
    Rich per-patient summary - each patient is its own n-of-1 experiment, so INDIVIDUAL
    statistics are reported, not only group-level ones:

    - **count + mean per condition** (n[A], mean[A], ...) ;
    - **all pairwise** non-overlap (Tau-U + NAP) - every C(k,2) combination, not just the
      primary contrast (so no pair is missing) ;
    - **per-patient temporal slope** isolating the learning trend (condition means removed
      first): an **OLS** slope (least squares) AND a **Sen/Theil-Sen** slope (robust,
      median of pairwise slopes - resistant to outliers, hence preferred as the default
      trend magnitude for short noisy SCED series) ;
    - **per-patient trend test** Mann-Kendall + Hamed-Rao (autocorrelation-corrected) ;
    - **per-patient INDIVIDUAL inference**: an n-of-1 randomization test of the condition
      effect (partial F by permutation, same engine/detrend/scheme as the group) - the
      individual analogue of the group's primary test, beyond the descriptive overlap.

    Interpretation bands are added by ``_annotate_per_unit``. ``n_perm`` here is the
    per-patient permutation budget (kept modest; 5 patients x n_perm).

    References: Sen 1968 (Theil-Sen slope); Hamed & Rao 1998 (autocorrelation-
    corrected Mann-Kendall); Edgington & Onghena 2007 (per-case randomization);
    Holm 1979, Benjamini & Hochberg 1995 (multiplicity).
    R equivalent: no single equivalent - trend::sens.slope + modifiedmk::mmkh
    (Hamed-Rao) + stats::p.adjust (Holm/BH); per-case permutation via SCRT.
    """
    rows = []
    for u, g in clean.groupby(unit_col):
        g = g.sort_values(session_col)
        yc = pd.to_numeric(g[outcome_col], errors="coerce")
        sc = pd.to_numeric(g[session_col], errors="coerce")
        lab = g[condition_col].astype(object).to_numpy()
        row = {"Unit": str(u), "N": int(yc.notna().sum())}
        # count + mean per condition
        for c in conditions:
            vals = pd.to_numeric(g[g[condition_col] == c][outcome_col], errors="coerce").dropna()
            row[f"n[{c}]"] = int(len(vals))
            row[f"mean[{c}]"] = round(float(vals.mean()), 3) if len(vals) else np.nan
            row[f"sd[{c}]"] = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan
        # all pairwise non-overlap (+ CI95 / Critical Tau-U / NAP CI95 on the primary contrast)
        prim_ref, prim_trt = conditions[0], conditions[1]
        for ref, c in itertools.combinations(conditions, 2):
            a = pd.to_numeric(g[g[condition_col] == ref][outcome_col], errors="coerce").dropna()
            b = pd.to_numeric(g[g[condition_col] == c][outcome_col], errors="coerce").dropna()
            if len(a) and len(b):
                tu = tau_u(a, b, improvement=improvement)
                nap_v = round(nap(a, b, improvement=improvement), 3)
                row[f"Tau-U[{c}v{ref}]"] = tu["tau_u"]
                row[f"NAP[{c}v{ref}]"] = nap_v
                if ref == prim_ref and c == prim_trt:        # CI/Critical on the primary contrast
                    se = tu["se"]
                    row["Tau-U CI95 (primary)"] = str(tu["ci"])
                    row["Tau-U critical (primary)"] = tu["critical"]
                    row["NAP CI95 (primary)"] = str(
                        (round(max(0.0, nap_v - 1.96 * se / 2), 3),
                         round(min(1.0, nap_v + 1.96 * se / 2), 3)) if np.isfinite(se) else (np.nan, np.nan))
            else:
                row[f"Tau-U[{c}v{ref}]"], row[f"NAP[{c}v{ref}]"] = np.nan, np.nan
        # per-patient temporal slope (learning trend) = condition-centered, OLS + Sen + MK
        ok = yc.notna() & sc.notna()
        yy, ss, ll = yc[ok].to_numpy(float), sc[ok].to_numpy(float), lab[ok.to_numpy()]
        resid = yy.copy()
        for c in conditions:                       # remove per-condition means -> isolate time
            m = ll == c
            if m.any():
                resid[m] = yy[m] - yy[m].mean()
        row["OLS slope/session"] = (round(float(np.polyfit(ss, resid, 1)[0]), 4)
                                    if len(ss) >= 2 and len(np.unique(ss)) >= 2 else np.nan)
        sl = sens_slope(resid, x=ss)
        row["Sen slope/session"] = round(float(sl), 4) if np.isfinite(sl) else np.nan
        row["Trend p (MK+HR)"] = mann_kendall(resid)["p_value"]
        # per-patient INDIVIDUAL randomization test (n-of-1), same engine as the group
        try:
            r1 = condition_permutation_test(yy, ll, ss, detrend=detrend, method=method,
                                            n_perm=n_perm, max_consecutive=max_consecutive,
                                            random_state=random_state)
            row["Individual F"] = r1["observed_F"]
            row["Individual p"] = round(r1["p_value"], 4)
        except Exception:
            row["Individual F"], row["Individual p"] = np.nan, np.nan
        rows.append(row)

    # Multiplicity correction on the INDIVIDUAL tests (one per patient): scanning patients
    # for responders = a family of tests -> Holm (FWER) + Benjamini-Hochberg (FDR). The raw p
    # stays displayed (legitimate in per-case single-case logic).
    from statsmodels.stats.multitest import multipletests
    praw = [r.get("Individual p", np.nan) for r in rows]
    fin = [i for i, p in enumerate(praw) if isinstance(p, (int, float)) and np.isfinite(p)]
    holm = {i: np.nan for i in range(len(rows))}
    fdr = {i: np.nan for i in range(len(rows))}
    if len(fin) >= 2:
        pvals = [praw[i] for i in fin]
        h = multipletests(pvals, method="holm")[1]
        f = multipletests(pvals, method="fdr_bh")[1]
        for j, i in enumerate(fin):
            holm[i], fdr[i] = round(float(h[j]), 4), round(float(f[j]), 4)
    elif len(fin) == 1:
        holm[fin[0]] = fdr[fin[0]] = praw[fin[0]]
    for i, r in enumerate(rows):
        r["Individual p Holm"] = holm[i]
        r["Individual p FDR"] = fdr[i]
    return pd.DataFrame(rows)


def _group_by_condition(sub, *, unit_col, condition_col, outcome_col, conditions):
    """GROUP-level description per condition: pooled mean and SD (all obs) AND the
    between-patient SD (standard deviation of per-patient means) - to distinguish within-
    from between-patient variability."""
    rows = []
    for c in conditions:
        m = sub[condition_col] == c
        vals = pd.to_numeric(sub.loc[m, outcome_col], errors="coerce").dropna()
        pmeans = (sub[m].groupby(unit_col)[outcome_col]
                  .apply(lambda s: pd.to_numeric(s, errors="coerce").mean()).dropna())
        sd_in = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan
        sd_btw = round(float(pmeans.std(ddof=1)), 3) if len(pmeans) > 1 else np.nan
        rows.append({
            "Condition": str(c), "N (obs)": int(len(vals)), "Patients": int(len(pmeans)),
            "Mean (group)": round(float(vals.mean()), 3) if len(vals) else np.nan,
            "SD (pooled)": sd_in,
            "Mean of patient means": round(float(pmeans.mean()), 3) if len(pmeans) else np.nan,
            "SD between patients": sd_btw,
            "Explanation": "Per condition: mean and SD over all observations (pooled) "
                           "+ mean of per-patient means and SD BETWEEN patients (dispersion "
                           "of individual levels).",
            "Interpretation": (
                f"variability mostly {'BETWEEN patients' if (np.isfinite(sd_btw) and np.isfinite(sd_in) and sd_btw >= sd_in) else 'WITHIN patient'} "
                f"(SD between={sd_btw} vs SD pooled={sd_in})" if np.isfinite(sd_btw) else " - ")})
    return pd.DataFrame(rows, columns=["Condition", "N (obs)", "Patients", "Mean (group)",
                                       "SD (pooled)", "Mean of patient means",
                                       "SD between patients", "Explanation", "Interpretation"])


# ``block_size`` is wired into BOTH the GROUP ordered-trend test and the PRIMARY stratified
# partial-F: when a block_size is given the primary test forces within-block Draper-Stoneman
# label permutation (the exact reference set for a block-randomized schedule), since
# Freedman-Lane permutes residuals freely and cannot reproduce that schedule.
def pipeline_sced_alternating_group(df, *, unit_col, session_col, condition_col,
                                    outcome_col, conditions=None,
                                    outcome_type="continuous", improvement="increase",
                                    max_consecutive=None, block_size=None,
                                    time_covariate="none", detrend=None,
                                    perm_method="freedman-lane", standardize=True,
                                    test_heterogeneity=True, posthoc="auto",
                                    trend_test=True, condition_order=None, dose=None,
                                    classical_checks=True, hierarchical=False,
                                    covariate_cols=None, power_analysis=False,
                                    power_effect=None, power_target=0.80,
                                    power_n_sims=200, power_n_perm=200,
                                    n_perm=5000, random_state=0,
                                    save_path=None, plot_path=None, verbose=True):
    """
    End-to-end GROUP pipeline for replicated alternating-treatments / N-of-1 designs.

    Steps: validate structure (multi-unit); per-unit trend diagnostics aggregated to a
    suggested detrend; per-unit and aggregated effect sizes (Tau-U/NAP); the
    **stratified randomization test** as primary inference (partial F for a common
    condition effect, unit as block, labels permuted within unit via
    ``stratified_condition_permutation_test``). With ``save_path`` set, writes a report
    (sheets ``Setup Info``, ``Per Unit``, ``Effect Sizes``, ``Randomization Test``,
    ``Heterogeneity``).

    ``test_heterogeneity`` (default True) adds the unit x condition interaction test
    (Freedman-Lane, approximate) plus Cochran's Q / I2 on the per-unit effect - the
    common-effect test alone can be significant despite a non-responder, or null while
    strong opposite effects cancel, so this read-out guards against both.

    ``posthoc`` (default ``"auto"`` = on when >=3 conditions) adds all pairwise
    contrasts via ``pairwise_contrasts``: Hedges' g + bootstrap CI and Holm/FDR-adjusted
    permutation p - the omnibus says *some* conditions differ, this says *which*.

    ``classical_checks`` (default True) adds a secondary "Classical RM" sheet with the
    familiar Friedman / RM-ANOVA on the per-unit x condition means plus their assumption
    diagnostics (normality, Mauchly sphericity + GG epsilon, autocorrelation, temporal
    slope) - a reviewer-friendly cross-check; the randomization test stays primary because
    these classical tests ignore the time trend and serial dependence.

    ``covariate_cols`` (list of session-level columns) adds ANCOVA-style adjustment to
    the primary common-effect test: the condition is tested *beyond* time and those
    covariates. Adjust only for covariates **not affected by the condition** (exogenous);
    a mediator would bias the effect.

    ``hierarchical`` (default False) adds a "Hierarchical" sheet: a mixed model
    (random intercept + random condition slope) giving the population effect, the
    between-patient slope SD, the ICC and shrunk per-patient effects - with p-values
    taken from permutation (the model's asymptotic p is untrustworthy at few clusters).

    ``power_analysis`` (default False - it is a slow Monte-Carlo) adds a "Power" sheet
    on the **realised** design: the MDES (minimum detectable effect at ``power_target``)
    and, if ``power_effect`` (raw units) is given, the retrospective power for that
    effect - never the discredited observed power (Hoenig & Heisey 2001). Tune cost via
    ``power_n_sims`` / ``power_n_perm``.

    ``detrend`` in {none, linear, log, auto} - ``"auto"`` adopts the aggregated per-unit
    suggestion (resolved once, before permuting; the time basis depends only on the
    sessions, so the test stays exact). ``perm_method`` in {freedman-lane (default),
    draper-stoneman, auto}; ``"auto"`` uses ``choose_permutation_method_group``.
    ``standardize`` z-scores the outcome within each unit (default True). ``conditions``
    takes the reference first; >=3 conditions give an omnibus F.

    **Ordered-trend test (dose-response), group level** - when ``trend_test`` and >=3
    conditions, a stratified ORDERED-condition test is reported IN ADDITION to the omnibus
    (``stratified_ordered_trend_test``): is the outcome monotone across the ordered
    conditions across units (e.g. MI 4/5/6 s over the 5 pilot patients)? Pass
    ``condition_order`` (low to high; defaults to ``conditions``) and optionally ``dose``
    (numeric levels, e.g. (4,5,6)). Labels are permuted **within unit** (exact stratified
    reference set); values are within-unit z-scored, so the pooled slope reads per dose unit
    on the z-scale. Justified within the RT framework but NOT power-validated for SCED
    alternation designs (Michiels et al. 2017) -> alongside, not instead; pre-specify a
    priori (Manolov et al. 2021).

    **Randomisation schedule provision** - ``max_consecutive`` and ``block_size``
    (randomized-block within unit) constrain the permutation to match the schedule used.
    ``block_size`` currently applies to the ordered-trend test; the primary stratified F
    uses within-unit count-preserving permutation (+ ``max_consecutive``).

    Returns ``(model_info: dict, randomization_result: dict)``.

    References: Edgington & Onghena 2007 (stratified randomization); Winkler et al.
    2014 (Freedman-Lane partial F); Higgins & Thompson 2002 (I2 heterogeneity).
    R equivalent: permuco::lmperm with within-unit blocks / FSL PALM (stratified
    Freedman-Lane) - potential equivalent, to test; no single-function equivalent.
    """
    # `detrend` renamed to `time_covariate` (time slope IN the model, not a separate
    # pre-detrend); `detrend` kept as a backward-compatible alias.
    detrend = detrend if detrend is not None else time_covariate
    clean, report = validate_sced(df, session_col=session_col, phase_col=condition_col,
                                  outcome_col=outcome_col, unit_col=unit_col)
    if conditions is None:
        conditions = tuple(report["Phases/conditions"])
    conditions = tuple(conditions)
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required.")
    if report["Number of units"] < 2:
        raise ValueError("The group pipeline requires at least two units "
                         "(use pipeline_sced_alternating for n=1).")
    if len(clean) == 0:
        raise ValueError("No valid row after cleaning (non-numeric outcome "
                         "or entirely missing?).")
    missing = [c for c in conditions if (clean[condition_col] == c).sum() == 0]
    if missing:
        raise ValueError(f"Condition(s) absent from the data: {missing}.")

    sub = clean[clean[condition_col].isin(conditions)]

    warnings = (missing_data_warnings(df, outcome_col=outcome_col, session_col=session_col,
                                      unit_col=unit_col)
                + sced_data_warnings(sub, session_col=session_col, condition_col=condition_col,
                                     outcome_col=outcome_col, unit_col=unit_col,
                                     conditions=conditions))
    if warnings and verbose:
        print("Warning - data quality issues:")
        for msg in warnings:
            print(f"   - {msg}")

    # group description per condition (pooled mean/SD + between-patient SD)
    bycond_df = _group_by_condition(sub, unit_col=unit_col, condition_col=condition_col,
                                    outcome_col=outcome_col, conditions=conditions)

    # aggregated effect sizes (the rich per-patient table is built later, once the
    # detrend/scheme are resolved, so the per-patient individual test matches the group)
    es_parts = [_effect_sizes_table(g, condition_col, outcome_col, conditions, improvement)
                for _, g in sub.groupby(unit_col)]
    es_concat = pd.concat(es_parts, ignore_index=True)
    es_df = (es_concat.groupby("Comparison", as_index=False)
             .mean(numeric_only=True).round(4))
    # the Tau-U CI / Critical / p are specific to EACH patient (cf. Per Unit); their
    # cross-patient mean is meaningless -> not kept in the group aggregate.
    es_df = es_df.drop(columns=["Tau-U critical (a.05)", "Tau-U p"], errors="ignore")
    es_df.insert(0, "Aggregation", "mean across units")
    # re-attach the interpretation bands (lost by numeric aggregation) on the aggregated
    # values, so they are reported as in n-of-1
    es_df["Strength NAP"] = es_df["NAP"].map(nap_magnitude)
    es_df["Strength Tau-U"] = es_df["Tau-U"].map(tau_magnitude)
    es_df["Strength PND"] = es_df["PND"].map(pnd_magnitude)
    es_df["Strength PEM"] = es_df["PEM"].map(pem_magnitude)

    # detrend resolution
    suggested, per_unit_detrend = _group_detrend_suggestion(
        sub, unit_col=unit_col, session_col=session_col,
        outcome_col=outcome_col, condition_col=condition_col)
    detrend_used = suggested if detrend == "auto" else detrend

    labels = sub[condition_col].to_numpy()
    values = pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy()
    sessions = pd.to_numeric(sub[session_col], errors="coerce").to_numpy()
    units = sub[unit_col].to_numpy()
    cov = None
    if covariate_cols:
        cov = sub[list(covariate_cols)].apply(pd.to_numeric, errors="coerce").to_numpy()

    if block_size:
        # a block-randomized schedule's exact reference set is WITHIN-BLOCK label
        # permutation (design-based, Draper-Stoneman); Freedman-Lane permutes residuals
        # freely and cannot reproduce that schedule, so force DS to keep the primary test
        # exact for the design (Edgington & Onghena 2007).
        method_rec = None
        method_used = "draper-stoneman"
    elif perm_method == "auto" and detrend_used != "none":
        method_rec = choose_permutation_method_group(labels, units, sessions,
                                                     detrend=detrend_used)
        method_used = method_rec["recommended_method"]
    else:
        method_rec = None
        method_used = perm_method if perm_method != "auto" else "freedman-lane"

    res = stratified_condition_permutation_test(
        values, labels, units, sessions, detrend=detrend_used, method=method_used,
        n_perm=n_perm, max_consecutive=max_consecutive, block_size=block_size,
        standardize=standardize, covariates=cov, random_state=random_state)
    # NAME of the test (not just its statistic) - patient = within-subject block
    test_name = sced_test_name(n_conditions=len(conditions), detrend=detrend_used,
                               grouped=True, method=res["method"])
    stat_desc = ("Stratified partial F (condition | patient"
                 + (f" + time {detrend_used}" if detrend_used != "none" else "")
                 + (f" + {list(covariate_cols)}" if covariate_cols else "") + ")")
    rand = {"observed": res["observed_F"], "p_value": res["p_value"],
            "n_perm": res["n_perm"], "two_sided": False, "statistic": stat_desc}

    # rich per-patient table (now that detrend/scheme are known -> individual n-of-1 test
    # uses the SAME engine/detrend/scheme as the group)
    per_unit_df = _per_unit_table(
        sub, unit_col=unit_col, session_col=session_col, condition_col=condition_col,
        outcome_col=outcome_col, conditions=conditions, improvement=improvement,
        detrend=detrend_used, method=method_used, max_consecutive=max_consecutive,
        n_perm=min(n_perm, 2000), random_state=random_state)

    model_info = {
        "Design": "Alternating treatments / N-of-1 - GROUP (replicated single-case)",
        "Rows dropped (missing)": report["Sessions dropped (missing)"],
        "Data warnings": warnings if warnings else "none",
        "Covariate adjustment": (f"{list(covariate_cols)} (ANCOVA) - assumed not "
                                 "affected by the condition" if covariate_cols else "none"),
        "Outcome type": outcome_type, "Outcome": outcome_col,
        "Number of units": report["Number of units"],
        "Conditions": list(conditions), "Improvement direction": improvement,
        "Primary inference (test)": test_name,
        "Test statistic": stat_desc,
        "Detrend": detrend_used + (" (auto)" if detrend == "auto" else ""),
        "Per-unit detrend suggestion": per_unit_detrend,
        "Permutation method": res["method"] + (" (auto)" if perm_method == "auto" else ""),
        "Within-unit standardization": standardize,
        "Permutations": n_perm,
        "Max consecutive constraint": max_consecutive if max_consecutive else "none",
        "Degrees of freedom (num, den)": res["df"],
        "Randomization p-value": round(rand["p_value"], 4),
        "Observed statistic": round(rand["observed"], 4),
    }
    if method_rec is not None:
        model_info["Method recommendation"] = method_rec["recommended_method"]
        model_info["Condition-time R2 (mean within-unit)"] = method_rec["condition_time_R2"]

    # heterogeneity: does the effect vary across units? (interaction FL + Cochran Q/I2)
    het_df = None
    if test_heterogeneity:
        het = heterogeneity_test(values, labels, units, sessions, conditions=conditions,
                                 condition_order=condition_order, dose=dose,
                                 detrend=detrend_used, n_perm=n_perm,
                                 max_consecutive=max_consecutive, standardize=standardize,
                                 random_state=random_state)
        ip = het["interaction_p"]
        ds = het["dose_slope"]
        het_sig = (np.isfinite(ip) and ip < 0.05) or \
                  (isinstance(ds["I2_percent"], (int, float)) and np.isfinite(ds["I2_percent"])
                   and ds["I2_percent"] > 50)
        model_info["Heterogeneity - interaction p"] = (round(ip, 4) if np.isfinite(ip) else "n/a")
        model_info["Heterogeneity - I2 (%) primary"] = het["I2_percent"]
        model_info["Heterogeneity - dose slope I2 (%)"] = ds["I2_percent"]
        model_info["Heterogeneity - verdict"] = (
            "HETEROGENEOUS effect between patients: the common effect masks different "
            "responses (see Per Unit)." if het_sig else
            "No marked heterogeneity: the common effect summarises the patients well.")
        # one row for the interaction (all conditions) + one per contrast + the dose slope
        rows = [{"Element": "Patient x condition interaction (all conditions)",
                 "Statistic": f"F={het['interaction_F']}",
                 "p": model_info["Heterogeneity - interaction p"],
                 "I2 (%)": "", "Pooled effect (z)": "", "Note": het["interaction_note"]}]
        for pc in het["per_contrast"]:
            rows.append({"Element": f"Contrast {pc['contrast']}", "Statistic": f"Q={pc['Q']}",
                         "p": pc["Q_p"], "I2 (%)": pc["I2_percent"],
                         "Pooled effect (z)": pc["pooled_effect"], "Note": "Cochran Q/I2 (per pair)"})
        rows.append({"Element": "Dose-response slope (per patient)", "Statistic": f"Q={ds['Q']}",
                     "p": ds["Q_p"], "I2 (%)": ds["I2_percent"],
                     "Pooled effect (z)": ds["pooled_effect"],
                     "Note": "Q/I2 on within-patient dose-response slopes (z/dose)"})
        het_df = pd.DataFrame(rows, columns=["Element", "Statistic", "p", "I2 (%)",
                                             "Pooled effect (z)", "Note"])

    # post-hoc pairwise contrasts (which conditions differ?)
    posthoc_df = None
    do_posthoc = posthoc is True or (posthoc == "auto" and len(conditions) >= 3)
    if do_posthoc:
        posthoc_df = pairwise_contrasts(
            values, labels, units=units, sessions=sessions, detrend=detrend_used,
            method=res["method"], improvement=improvement, n_perm=n_perm,
            max_consecutive=max_consecutive, standardize=standardize,
            random_state=random_state)
        model_info["Post-hoc pairwise"] = f"{len(posthoc_df)} contrasts (Holm + FDR)"

    # ordered-trend (dose-response) at the GROUP level, IN ADDITION to the omnibus.
    # Justification + caveat: see stratified_ordered_trend_test / make_ordered_trend_stat
    # (RT admits any statistic - Edgington & Onghena 2007; not validated for SCED
    # alternation designs - Michiels et al. 2017; pre-specify - Manolov et al. 2021).
    trend_df = None
    if trend_test and len(conditions) >= 3:
        order = tuple(condition_order) if condition_order else conditions
        if dose is not None and len(dose) != len(order):
            raise ValueError(f"dose ({list(dose)}) must have the same length as "
                             f"the condition order ({list(order)}).")
        # consistency with the primary analysis: residualise the per-unit time trend first
        tvals = values
        if detrend_used != "none":
            Z = _group_nuisance(units, sessions, detrend_used)
            beta, *_ = np.linalg.lstsq(Z, values, rcond=None)
            tvals = values - Z @ beta
        common_t = dict(condition_order=order, dose=dose, improvement=improvement,
                        standardize=standardize, max_consecutive=max_consecutive,
                        block_size=block_size, n_perm=n_perm, random_state=random_state)
        r_slope = stratified_ordered_trend_test(tvals, labels, units, kind="slope", **common_t)
        r_jt = stratified_ordered_trend_test(tvals, labels, units, kind="jonckheere", **common_t)
        dose_used = list(dose) if dose is not None else list(range(len(order)))

        def _sig(p):
            return "significant" if p < 0.05 else "not significant"
        model_info["Ordered-trend test (group)"] = (
            f"IN ADDITION to the omnibus - monotone dose-response over the order {list(order)} "
            f"(dose {dose_used}), within-unit permutation. RT framework (Edgington & Onghena "
            "2007); NOT validated for SCED alternation (Michiels et al. 2017) -> pre-specify "
            "(Manolov et al. 2021).")
        model_info["Ordered-trend slope p (group)"] = round(r_slope["p_value"], 4)
        model_info["Ordered-trend Jonckheere p (group)"] = round(r_jt["p_value"], 4)
        trend_df = pd.DataFrame([
            {"Statistic": "Slope on the dose (linear contrast, within-unit z)",
             "Observed": r_slope["observed"], "p (randomization)": round(r_slope["p_value"], 4),
             "Explanation": "OLS slope of the within-unit z-scored outcome on the ordered "
                            "dose, pooled across units - change per dose step.",
             "Interpretation": f"Linear dose-response {_sig(r_slope['p_value'])} "
                               f"(p={round(r_slope['p_value'],4)}); >0 = beneficial with the dose."},
            {"Statistic": "Jonckheere-Terpstra (ordered alternative)",
             "Observed": r_jt["observed"], "p (randomization)": round(r_jt["p_value"], 4),
             "Explanation": "Nonparametric ordered-alternatives test (ranks; Terpstra "
                            "1952; Jonckheere 1954) - robust, no linearity assumption.",
             "Interpretation": f"Monotone trend {_sig(r_jt['p_value'])} "
                               f"(p={round(r_jt['p_value'],4)})."},
            {"Statistic": " - Methodological note - ", "Observed": "", "p (randomization)": "",
             "Explanation": "Complements the stratified omnibus (which ignores the order). "
                            "Within-unit permutation = exact reference set (Edgington & Onghena 2007).",
             "Interpretation": "NOT validated for SCED alternation (Michiels et al. 2017: only "
                               "MD/NAP); pre-specify a priori (Manolov et al. 2021)."},
        ], columns=["Statistic", "Observed", "p (randomization)", "Explanation", "Interpretation"])

    # classical repeated-measures cross-check (Friedman / RM-ANOVA + assumptions)
    classical_df = None
    if classical_checks:
        classical_df = classical_rm_checks(sub, unit_col=unit_col, session_col=session_col,
                                           condition_col=condition_col, outcome_col=outcome_col,
                                           conditions=conditions)
        model_info["Classical RM cross-check"] = "see 'Classical RM' sheet (secondary)"

    # hierarchical (mixed model) estimates with permutation p-values (opt-in)
    hier_df = None
    if hierarchical:
        from functions.sced.hierarchical import hierarchical_group_model
        hsum, hier_df = hierarchical_group_model(
            sub, unit_col=unit_col, session_col=session_col, condition_col=condition_col,
            outcome_col=outcome_col, conditions=conditions, outcome_type=outcome_type,
            detrend=detrend_used, method=res["method"], n_perm=n_perm,
            max_consecutive=max_consecutive, standardize=standardize, random_state=random_state)
        model_info["Hierarchical - population effect"] = hsum.get("population_effect")
        model_info["Hierarchical - SD random slope"] = hsum.get("sd_random_slope")
        model_info["Hierarchical - ICC"] = hsum.get("icc")

    # power analysis on the realised design (MDES + retrospective; opt-in, slow)
    power_df = None
    if power_analysis:
        psum, power_df = power_report(
            sub, session_col=session_col, condition_col=condition_col, outcome_col=outcome_col,
            unit_col=unit_col, conditions=conditions, effect_of_interest=power_effect,
            target_power=power_target, detrend=detrend_used, method=res["method"],
            max_consecutive=max_consecutive, standardize=standardize, n_sims=power_n_sims,
            n_perm=power_n_perm, random_state=random_state)
        model_info["Power - MDES (Cohen d @target)"] = psum["mdes_cohens_d"]
        if power_effect is not None:
            model_info["Power - retrospective (specified effect)"] = psum["retrospective_power"]

    rand_df = pd.DataFrame([{
        "Test": test_name, "Statistic": stat_desc, "Observed": round(rand["observed"], 4),
        "P-value (randomization)": round(rand["p_value"], 4),
        "Permutations": rand["n_perm"], "Two-sided": rand["two_sided"],
        "Units": res["n_units"]}])

    # Methods paragraph (scientific-article style) for the report
    stats_used = (["heterogeneity"] if test_heterogeneity else []) \
        + (["posthoc"] if (posthoc is True or (posthoc == "auto" and len(conditions) >= 3)) else []) \
        + (["trend"] if trend_df is not None else [])
    methods_txt = sced_methods_text(
        design="ATD / N-of-1 replicated", n_units=report["Number of units"],
        conditions=list(conditions), outcome=outcome_col, detrend=detrend_used,
        method=res["method"], n_perm=n_perm, standardize=standardize,
        improvement=improvement, statistics_used=stats_used, dose=dose)
    methods_df = pd.DataFrame([{"Methods": methods_txt}])

    # LOG: the messages usually printed to the console (quality alerts, detrend/scheme
    # choices) - so they appear IN the report, not only at the terminal.
    jrows = [{"Type": "Data alert", "Message": w} for w in (warnings or [])]
    jrows.append({"Type": "Detrend", "Message": f"per-patient suggestion: {per_unit_detrend}; "
                  f"applied: {detrend_used}"})
    if method_rec is not None:
        jrows.append({"Type": "Permutation scheme", "Message": method_rec.get("rationale", "")})
    jrows.append({"Type": "Inference", "Message": f"{test_name}; statistic: {stat_desc}."})
    journal_df = pd.DataFrame(jrows or [{"Type": " - ", "Message": "No alert."}],
                             columns=["Type", "Message"])

    # GLOSSARY: definitions of the cited methods (including Holm/FDR) + report-specific markers.
    from functions.sced.glossary import explain
    gl = explain("randomization test", "p-value", "partial F", "stratified test",
                 "omnibus variance", "ordered trend", "jonckheere", "tau-u", "nap", "pnd", "pem",
                 "hedges g", "pairwise contrasts", "holm", "fdr", "heterogeneity", "i2",
                 "cochran q", "detrend", "freedman-lane", "draper-stoneman", as_dict=True)
    extra = {
        "Critical Tau-U": "Threshold the Tau-U must exceed to be significant (~1.96*SE). "
                          "Depends ONLY on n (same counts per condition -> same threshold for "
                          "all patients) - which is why it is identical in Per Unit.",
        "ICC": "Intraclass correlation: share of between-patient variance. Koo & Li 2016: "
               "<.50 poor, .50-.75 moderate, .75-.90 good, >=.90 excellent.",
        "partial pooling": "The mixed model pulls each patient's effect toward the group "
                           "mean -> more stable (shrunk) individual effects.",
        "Holm": gl.get("holm", ""), "FDR": gl.get("fdr", "")}
    glossaire_df = pd.DataFrame(
        [{"Term": k, "Definition": v} for k, v in {**gl, **extra}.items()],
        columns=["Term", "Definition"])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)               # report goes straight in save_path (the
        full = os.path.join(save_path, f"{outcome_col}_sced_alternating_group_report.xlsx")  # Analyse dir)
        info_df = pd.DataFrame([{"Parameter": k,
                                 "Value": (", ".join(map(str, v)) if isinstance(v, list)
                                           else (str(v) if isinstance(v, (dict, tuple)) else v))}
                                for k, v in model_info.items()], columns=["Parameter", "Value"])
        from functions.sced.glossary import (interpretation_glossary, ascii_sanitize_df as S,
                                             data_recap_df, style_cells, _to_float)
        recap = data_recap_df(df, tier_col=unit_col, session_col=session_col, phase_col=condition_col,
                              outcomes=[outcome_col], design="Group ATD (randomised alternation)")

        # --- colour rules to HIGHLIGHT THE RESULTS (green = notable, amber = intermediate,
        # grey = not significant) - detected by column name, applied uniformly to all data
        # sheets. Colour code from style_cells. ---
        def _p_fill(v):                                  # p-value: green if < .05
            s = str(v).strip()
            if s.startswith("<"):
                return (True, "C6EFCE")                  # e.g. "<0.0001" -> significant
            a = _to_float(v)
            if a is None or not np.isfinite(a):
                return (False, None)
            return (True, "C6EFCE") if a < 0.05 else (False, "F2F2F2")

        def _nap_fill(v):                                # NAP: >=.93 strong, >=.66 moderate
            a = _to_float(v)
            if a is None or not np.isfinite(a):
                return (False, None)
            return (True, "C6EFCE") if a >= 0.93 else ((False, "FFE699") if a >= 0.66 else (False, None))

        def _absmag_fill(v, hi, mid):                    # |effect|: green >= hi, amber >= mid
            a = _to_float(v)
            if a is None or not np.isfinite(a):
                return (False, None)
            a = abs(a)
            return (True, "C6EFCE") if a >= hi else ((False, "FFE699") if a >= mid else (False, None))

        def _result_rules(cols):
            rules = {}
            for c in cols:
                n = str(c).strip().lower()
                if ("p-value" in n or "p_value" in n or n == "p" or n.endswith(" p")
                        or n.startswith("p ") or "p (" in n or n.endswith("_p") or "p value" in n):
                    rules[c] = _p_fill
                elif n == "nap":
                    rules[c] = _nap_fill
                elif "tau-u" in n or "tau_u" in n or n == "tau":
                    rules[c] = lambda v: _absmag_fill(v, 0.60, 0.20)     # nonoverlap
                elif "hedges" in n or n == "g" or "smd" in n or n == "d":
                    rules[c] = lambda v: _absmag_fill(v, 0.80, 0.50)     # Cohen's d/g
                elif n.startswith("i2") or n == "i2" or "i2" in n:
                    rules[c] = lambda v: _absmag_fill(v, 75.0, 50.0)     # I2 heterogeneity
            return rules

        with pd.ExcelWriter(full, engine="openpyxl") as writer:
            S(recap).to_excel(writer, sheet_name="Data & design", index=False)   # first sheet
            # The generic "Explanation" column is REMOVED from each data sheet and grouped
            # into a dedicated "Explanations" sheet; "Interpretation" (data-specific verdict)
            # stays in the sheet.
            legend = []

            def W(d, sheet, key_col=None):
                d = d.copy()
                if "Explanation" in d.columns:
                    if d["Explanation"].nunique(dropna=True) <= 1:
                        legend.append({"Sheet": sheet, "Element": "",
                                       "Explanation": (d["Explanation"].iloc[0] if len(d) else "")})
                    else:
                        for _, r in d.iterrows():
                            legend.append({"Sheet": sheet,
                                           "Element": (str(r[key_col]) if key_col and key_col in d.columns else ""),
                                           "Explanation": r["Explanation"]})
                    d = d.drop(columns="Explanation")
                sd = S(d)
                sd.to_excel(writer, sheet_name=sheet, index=False)   # ASCII guaranteed
                style_cells(writer.sheets[sheet], sd, _result_rules(sd.columns))  # result colours

            W(_annotate_setup(info_df), "Setup Info", key_col="Parameter")
            W(bycond_df, "By Condition", key_col="Condition")
            W(_annotate_per_unit(per_unit_df, conditions), "Per Unit")
            W(_annotate_effect_sizes(es_df), "Effect Sizes")
            W(_annotate_randomization(rand_df, rand["p_value"]), "Randomization Test")
            if het_df is not None:
                W(_annotate_heterogeneity(het_df), "Heterogeneity", key_col="Element")
            if posthoc_df is not None:
                W(_annotate_posthoc(posthoc_df).astype(str), "Post-hoc")
            if trend_df is not None:
                W(trend_df, "Ordered Trend", key_col="Statistic")
            if classical_df is not None:
                W(classical_df, "Classical RM")
            if hier_df is not None:
                W(hier_df, "Hierarchical", key_col="Quantity")
            if power_df is not None:
                W(power_df, "Power", key_col="Analysis")
            S(pd.DataFrame(legend, columns=["Sheet", "Element", "Explanation"])).to_excel(
                writer, sheet_name="Explanations", index=False)
            S(glossaire_df).to_excel(writer, sheet_name="Glossary", index=False)
            S(journal_df).to_excel(writer, sheet_name="Log", index=False)
            S(methods_df).to_excel(writer, sheet_name="Methods", index=False)
            S(interpretation_glossary("rando")).to_excel(
                writer, sheet_name="Interpretation guide", index=False)   # reading thresholds
        if verbose:
            print(f"Report exported to {full}")
    elif verbose:
        print("Report not exported")

    if plot_path is not None:
        from functions.sced.plots.panels import plot_sced_alternating
        plot_sced_alternating(sub, session_col=session_col, condition_col=condition_col,
                              outcome_col=outcome_col, unit_col=unit_col,
                              conditions=list(conditions), save_path=plot_path)
        model_info["Figure"] = os.path.join(plot_path, f"{outcome_col}_sced_alternating_plot.png")

    return model_info, rand
