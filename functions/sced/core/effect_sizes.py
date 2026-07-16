"""
SCED core - nonoverlap and standardized effect sizes.

NAP (+ CI), the bounded Tau-U family, within-case and between-group Hedges' g,
PND / PEM, their interpretation bands and magnitude labels. Imports the shared
Kendall S / Theil-Sen leaf helpers from the nuisance base layer.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .nuisance import _kendall_s, sens_slope


def nap(baseline, treatment, *, improvement="increase"):
    """
    Nonoverlap of All Pairs (NAP): share of baseline-treatment pairs in the
    improving direction (ties = 0.5). NAP = U / (n_a*n_b); 0.5 = chance, 1 = full
    separation. Preferred over PND (less sensitive to single extreme points).
    Returns the point estimate only; use ``nap_ci`` for the SE + confidence interval.

    Verified against scan 0.68.1 and SingleCaseES 0.7.4 (exact match).

    References: Parker & Vannest 2009 (Nonoverlap of All Pairs).
    R equivalent: SingleCaseES::NAP; scan::nap.
    """
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = b[:, None] - a[None, :]
    if improvement == "decrease":
        diff = -diff
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(wins / (len(a) * len(b)))


def nap_ci(baseline, treatment, *, improvement="increase", se_method="unbiased",
           confidence=0.95):
    """NAP with its **standard error** and **confidence interval**, following the
    SingleCaseES (Pustejovsky) implementation exactly.

    NAP = mean of the placement matrix Q[i,j] = 1{b_j>a_i} + 0.5*1{b_j=a_i}. The SE uses
    the structural-component (placement) variance:
      - ``"unbiased"`` (default, scES default; Mee 1990 / Sen 1967 U-statistic variance)
        V = (p(1-p) + n*Q1 + m*Q2 - 2X)/((m-1)(n-1)), where m=n_A, n=n_B, Q1/Q2 are the
        row/column placement variances and X the total; p truncated to [0.5/mn, 1-0.5/mn];
      - ``"hanley"`` (Hanley & McNeil 1982);
      - ``"null"`` (variance under H0, (m+n+1)/(12mn)).
    The CI is the score interval of Newcombe (2006) solved by root-finding (the scES
    default), bounded to [0,1]. Returns ``{nap, se, ci_lower, ci_upper, method,
    n_baseline, n_treatment}``.

    Verified against SingleCaseES 0.7.4 (SE and CI match to 1e-4).

    References: Parker & Vannest 2009 (NAP); Newcombe 2006 (score CI); Mee 1990 / Sen 1967
    (unbiased U-statistic variance); Hanley & McNeil 1982 (hanley variance).
    R equivalent: SingleCaseES::NAP.
    """
    from scipy.optimize import brentq
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy(dtype=float)
    out = {"nap": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
           "method": se_method, "n_baseline": len(a), "n_treatment": len(b)}
    if len(a) == 0 or len(b) == 0:
        return out
    if improvement == "decrease":
        a, b = -a, -b
    m, n = len(a), len(b)                                   # m = n_A (rows), n = n_B (cols)
    Q = (b[None, :] > a[:, None]).astype(float) + 0.5 * (b[None, :] == a[:, None])
    p = float(Q.mean())
    out["nap"] = p
    sm = str(se_method).lower()
    # row/column placement variances (centred), per scES calc_NAP
    Q1 = float(np.sum((Q - p).sum(axis=1) ** 2)) / (m * n * n)
    Q2 = float(np.sum((Q - p).sum(axis=0) ** 2)) / (m * m * n)
    X = float(np.sum((Q - p) ** 2)) / (m * n)
    trunc = 0.5 / (m * n)
    p_t = min(max(p, trunc), 1 - trunc)
    if sm == "unbiased":
        V = (p_t * (1 - p_t) + n * Q1 + m * Q2 - 2 * X) / ((m - 1) * (n - 1)) if m > 1 and n > 1 else np.nan
    elif sm == "hanley":
        V = (p_t * (1 - p_t) + (n - 1) * Q1 + (m - 1) * Q2) / (m * n)
    elif sm == "null":
        V = (m + n + 1) / (12.0 * m * n)
    else:
        raise ValueError("se_method must be 'unbiased' / 'hanley' / 'null'")
    out["se"] = float(np.sqrt(V)) if np.isfinite(V) and V >= 0 else np.nan
    # Newcombe (2006) score CI (scES default), solved by root-finding
    if confidence is not None:
        h = (m + n) / 2.0 - 1
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        def f(x):
            return m * n * (p - x) ** 2 * (2 - x) * (1 + x) - z ** 2 * x * (1 - x) * (2 + h + (1 + 2 * h) * x * (1 - x))
        out["ci_lower"] = float(brentq(f, 0, p)) if p > 0 else 0.0
        out["ci_upper"] = float(brentq(f, p, 1)) if p < 1 else 1.0
    return out


def _tau_significance(tau_val, n, two_sided=True):
    """p-value for a **whole-series** Kendall tau (e.g. the baseline-trend diagnostic Tau_A)
    via the normal approximation, using the single-series Kendall variance -> SE =
    sqrt(2(2n+5)/(9n(n-1))) (Vannest & Ninci 2015; Brossard et al. 2018 p. 21). ``n`` = number
    of points in the series. ``two_sided`` (default) tests H1: tau != 0 (p = 2*Phi(-|z|));
    ``two_sided=False`` is the one-sided upper-tail test H1: tau > 0 (p = Phi(-z)). Returns
    ``(se, p)``; ``(nan, nan)`` if n < 3.

    NB - for the **A-vs-B contrast** (the Tau-U coefficient itself) use
    ``_tau_contrast_significance``: that statistic is a Mann-Whitney-type comparison over the
    n_A x n_B cross-pairs, whose exact null variance differs from this whole-series formula.
    Applying this series SE to the contrast (the old behaviour, and the singlecaseresearch.org
    convention) is anti-conservative.

    References: Vannest & Ninci 2015; Brossart et al. 2018 (whole-series Kendall SE).
    R equivalent: Kendall::MannKendall (whole-series tau test); no direct SCED equivalent."""
    if n < 3:
        return np.nan, np.nan
    se = np.sqrt((2.0 * (2 * n + 5)) / (9.0 * n * (n - 1)))
    if se == 0 or not np.isfinite(tau_val):
        return float(se), np.nan
    z = tau_val / se
    p = 2 * stats.norm.sf(abs(z)) if two_sided else stats.norm.sf(z)
    return float(se), float(p)


def _tau_contrast_significance(tau_ab, na, nb, two_sided=True):
    """p-value and SE for an **A-vs-B Tau** (the Tau-U coefficient tau = S/(n_A*n_B), S =
    #concordant - #discordant over the n_A x n_B cross-phase pairs) via the variance used by
    the reference module ``scan`` (Wilbert & Luke): the whole-series Kendall-S null variance
    VAR(S) = n(n-1)(2n+5)/18 (n = n_A+n_B, no tie correction), scaled by the contrast
    denominator -> SE = sqrt(n(n-1)(2n+5)/18) / (n_A*n_B), z = tau/SE.

    This is the **dimensionally consistent** SE for tau = S/(n_A*n_B) - the same VAR(S) as the
    applied Tau-U calculators (Vannest & Ninci 2015) but divided by n_A*n_B (the contrast's
    maximum |S|) rather than by n(n-1)/2 (a full series' maximum |S|). The Vannest-Ninci form
    SE = sqrt(2(2n+5)/(9n(n-1))) divides by the larger n(n-1)/2, giving a SE ~2x too small and
    an anti-conservative p that flags a moderate Tau-U (~0.4) as significant. ``two_sided``
    (default) -> p = 2*Phi(-|z|); else one-sided upper tail p = Phi(-z). Returns ``(se, p)``;
    ``(nan, nan)`` if n_A < 1, n_B < 1 or n_A+n_B < 3. Verified against ``scan`` (|delta| < 0.001).

    References: Vannest & Ninci 2015 (Kendall-S variance); scan (Wilbert & Lueke) contrast scaling.
    R equivalent: scan::tau_u (internal contrast SE)."""
    n = na + nb
    if na < 1 or nb < 1 or n < 3:
        return np.nan, np.nan
    var_s = n * (n - 1) * (2 * n + 5) / 18.0           # whole-series Kendall-S variance (no ties)
    se = np.sqrt(var_s) / (na * nb)                     # scaled by the contrast denominator (scan)
    if se == 0 or not np.isfinite(tau_ab):
        return float(se), np.nan
    z = tau_ab / se
    p = 2 * stats.norm.sf(abs(z)) if two_sided else stats.norm.sf(z)
    return float(se), float(p)


def hedges_g_within(baseline, treatment, *, improvement="increase"):
    """**Within-case Hedges' g**: d = (M_B - M_A) / SD_within-pooled, small-sample corrected
    (Hedges 1981 factor J), oriented by ``improvement``. SD_pooled = sqrt(((n_A-1)s_A^2 +
    (n_B-1)s_B^2)/(n_A+n_B-2)).

    WARNING: standardizes by a **within-case** SD -> magnitude **not comparable** across cases/studies
    (often inflated when the within-case SD is small); this is NOT the design-comparable BC-SMD (which
    standardizes by the between-case SD, cf. bc_smd_scdhlm). Reserved for a descriptive single-case reading.
    Returns ``{d, g, J, sd_within, n_baseline, n_treatment}`` (nan if data insufficient).

    References: Hedges 1981 (small-sample-corrected standardized mean difference J).
    R equivalent: SingleCaseES::SMD (within-case) - potential equivalent, to test."""
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy()
    na, nb = len(a), len(b)
    out = {"d": np.nan, "g": np.nan, "J": np.nan, "sd_within": np.nan,
           "n_baseline": na, "n_treatment": nb}
    if na < 2 or nb < 2:
        return out
    df = na + nb - 2
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / df)
    if not (np.isfinite(sp) and sp > 0):
        return out
    sign = 1.0 if improvement == "increase" else -1.0
    d = sign * (b.mean() - a.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * df - 1.0)               # small-sample correction (Hedges 1981)
    out.update({"d": round(float(d), 3), "g": round(float(J * d), 3), "J": round(float(J), 3),
                "sd_within": round(float(sp), 3)})
    return out


def tau_u(baseline, treatment, *, correct_baseline_trend=True, improvement="increase",
          method="auto", min_baseline_for_adj=7, alternative="two-sided"):
    """
    Tau-U family of nonparametric SCED effect sizes (Parker, Vannest, Davis & Sauber
    2011), built on Kendall's S - BOUNDED in [-1, +1] by construction.

    Correction rationale (the reason this function was rewritten) - the original Parker
    (2011a) hand-calculation of the baseline-corrected coefficient divides (S_AB - S_A) by
    n_a*n_b, a denominator that does NOT keep the result in [-1, +1]; values |tau|>1 are
    common when the baseline trend opposes the treatment shift (Brossard et al. 2018,
    Cogent Psychology, p. 9-10; Tarlow 2016, Behavior Modification, doi:10.1177/
    0145445516676750, found 17% of values >1 across 65 published contrasts). The previous
    implementation used exactly that n_a*n_b denominator and could return tau_u = 1.4. This
    version NEVER uses it for the corrected coefficient and offers two bounded variants,
    selected per the Fingerhut, Xu & Moeyaert (2021) decision flowchart:

      - ``"adj"``     : Tarlow (2016) Baseline Corrected Tau - remove the baseline
                        monotonic trend with a Theil-Sen slope from the whole series, then
                        compute **Kendall's tau-b** (tie-corrected rank correlation with the
                        phase dummy) on the residuals. Always bounded; preferred when the
                        baseline is long enough (Tarlow shows trend control holds only for
                        n_A >= 7). This is the **faithful tau-b form** of Tarlow's calculator - 
                        **verified == R ``scan::corrected_tau`` on both the coefficient and its
                        p-value** (they coincide except on a degenerate constant baseline,
                        e.g. [0,0,2,2,2,2], where identical Theil-Sen slopes can disagree by
                        float tie-resolution on a value that is null either way). The p-value
                        is the Kendall tau-b test. (The other variants here are tau-a; ``adj``
                        is tau-b because that is Tarlow's definition.)
      - ``"trend_a"`` : bounded Parker coefficient with the Brossard (2018) denominator
                        (S_AB - S_A)/(n_AB + n_A); fallback when n_A < 7 (Theil-Sen slope
                        unreliable). Acknowledged limitation: less effective trend control
                        than ``adj``. **Verified identical to ``scan::tau_u`` "A vs. B -
                        Trend A" (tau_method="a").**

    ``method="auto"`` (default) applies the flowchart: report **Tau (A vs B)** when the
    baseline trend is not significant (Brossard rule: trend p >= 0.05 AND |Tau_trend_A| <
    0.40), else ``"adj"`` if n_A >= ``min_baseline_for_adj`` else ``"trend_a"``. ``method``
    may be forced to ``adj`` / ``trend_a`` / ``none``. ``correct_baseline_trend=False``
    forces ``none`` (plain Tau A vs B) for backward compatibility.

    ``alternative`` - ``"two-sided"`` (default; H1: tau != 0, critical Tau-U = 1.96*SE) or
    ``"greater"`` (one-sided upper tail; H1: tau > 0 in the oriented improvement direction,
    critical Tau-U = 1.645*SE). Use ``"greater"`` only when the improvement direction is
    pre-specified a priori. The baseline-trend diagnostic stays two-sided regardless.

    Returns ``{tau, tau_u, variant, bounded, alternative, baseline_trend_tau,
    baseline_trend_p, se, p_value, ci, critical, n_baseline, n_treatment}``. ``tau`` is the
    uncorrected A-vs-B contrast (= 2*NAP - 1, always bounded; value verified == ``scan``
    tau_method="a" "A vs. B" and SingleCaseES ``Tau()``); ``tau_u`` is the
    selected coefficient. ``p_value`` is the normal-approximation significance test using the
    contrast-scaled Kendall-S variance SE = sqrt(n(n-1)(2n+5)/18)/(n_A*n_B) (``scan`` convention;
    see ``_tau_contrast_significance``), not the anti-conservative Vannest-Ninci SE. The
    Critical Tau-U interpretation of Fingerhut et al. (2021, Appendix A) is the recommended
    benchmark but is not reproduced here (table not in the source).

    References: Parker, Vannest, Davis & Sauber 2011 (Tau-U); Brossart et al. 2018 (bounded
    denominator); Tarlow 2016 (baseline-corrected tau-b); Fingerhut, Xu & Moeyaert 2021
    (variant flowchart, Critical Tau-U).
    R equivalent: scan::tau_u and scan::corrected_tau (numerically verified).
    """
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy()
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"tau": np.nan, "tau_u": np.nan, "variant": "n/a", "bounded": True,
                "baseline_trend_tau": np.nan, "baseline_trend_p": np.nan,
                "se": np.nan, "p_value": np.nan, "n_baseline": na, "n_treatment": nb}

    sign = 1.0 if improvement == "increase" else -1.0
    a_dir, b_dir = sign * a, sign * b
    diff = b_dir[:, None] - a_dir[None, :]
    s_ab = float(np.sum(diff > 0) - np.sum(diff < 0))
    tau = s_ab / (na * nb)                              # = 2*NAP - 1, bounded [-1, +1]

    # baseline trend diagnostic (Tau_trend_A) - always computed/reported first, per the
    # Brossard (2018, p. 19) and Fingerhut (2021) rule that it decides whether to correct.
    s_aa = _kendall_s(a_dir)
    n_pairs_a = na * (na - 1) / 2.0
    tau_trend_a = (s_aa / n_pairs_a) if n_pairs_a > 0 else 0.0
    _, p_a = _tau_significance(tau_trend_a, na)

    m = "none" if not correct_baseline_trend else str(method).lower()
    if m == "auto":
        trend_present = (np.isfinite(p_a) and p_a < 0.05) or abs(tau_trend_a) >= 0.40
        m = ("adj" if na >= min_baseline_for_adj else "trend_a") if trend_present else "none"

    if m in ("none", "ab", "a_vs_b"):
        tau_u_val = tau
        variant = "Tau-U A vs B (no trend correction)"
    elif m == "trend_a":
        # Brossard (2018) bounded denominator (S_AB - S_A)/(n_AB + n_A) - replaces the
        # unbounded Parker 2011a n_a*n_b denominator.
        tau_u_val = (s_ab - s_aa) / (na * nb + n_pairs_a)
        variant = "Tau-U (A vs B - trend A), Brossard 2018 bounded denominator"
    elif m == "adj":
        # Tarlow (2016) Baseline Corrected Tau, FAITHFUL tau-b form (== R scan::corrected_tau):
        # detrend the WHOLE series by the baseline Theil-Sen slope (continuous time index),
        # then Kendall's tau-b (tie-corrected rank correlation) between the detrended scores
        # and the phase dummy (A=0, B=1). Always bounded. The tau-b normalisation (not the
        # tau-a nonoverlap S/(n_A*n_B)) is what Tarlow's calculator and scan report.
        t_a = np.arange(1, na + 1, dtype=float)
        t_b = np.arange(na + 1, na + nb + 1, dtype=float)
        slope = sens_slope(a_dir, x=t_a)
        slope = slope if np.isfinite(slope) else 0.0
        ra, rb = a_dir - slope * t_a, b_dir - slope * t_b
        resid = np.concatenate([ra, rb])
        phase_dummy = np.concatenate([np.zeros(na), np.ones(nb)])
        tb = stats.kendalltau(phase_dummy, resid)      # tau-b + two-sided p (Tarlow/scan)
        tau_u_val = float(tb.statistic) if np.isfinite(tb.statistic) else 0.0
        adj_pvalue = float(tb.pvalue) if np.isfinite(tb.pvalue) else np.nan
        variant = "Tau-U adj (Tarlow 2016 baseline-corrected, tau-b == scan::corrected_tau)"
        if na < min_baseline_for_adj:                  # Tarlow 2016: control fails for n_A < 7
            variant += f" [WARNING n_A={na}<7: Tarlow 2016 shows trend control unreliable]"
    else:
        raise ValueError("method must be auto / adj / trend_a / none")

    two_sided = str(alternative).lower().replace("_", "-") in ("two-sided", "two", "2", "ts")
    if m == "adj":
        # tau-b coefficient -> use the Kendall tau-b test p (scipy, == Tarlow/scan); derive a
        # consistent SE from that p so the critical value / CI stay coherent.
        p = adj_pvalue
        if not two_sided and np.isfinite(p):           # one-sided 'greater'
            p = (p / 2) if tau_u_val >= 0 else (1 - p / 2)
        if np.isfinite(p) and 0 < p < 1 and tau_u_val != 0:
            se = abs(tau_u_val) / stats.norm.isf(p / 2 if two_sided else p)
        else:
            se = np.nan
    else:
        # A-vs-B contrast significance uses the EXACT Mann-Whitney/Kendall-S variance (scan
        # convention) - principled for the two-phase contrast and not anti-conservative, unlike
        # the whole-series SE. For trend_a it is an approximation on the same n_A*n_B scale.
        se, p = _tau_contrast_significance(tau_u_val, na, nb, two_sided=two_sided)
    # 95% CI (normal approx, always two-sided for reporting) bounded to [-1, 1], and the
    # Critical Tau-U = the value the estimate must exceed to be significant at alpha=.05
    # (z*SE ; Fingerhut-style threshold). z = 1.96 two-sided, 1.645 one-sided (greater).
    z_crit = 1.959963984540054 if two_sided else 1.6448536269514722
    if np.isfinite(se):
        ci = (round(max(-1.0, tau_u_val - 1.96 * se), 3), round(min(1.0, tau_u_val + 1.96 * se), 3))
        critical = round(z_crit * se, 3)
    else:
        ci, critical = (np.nan, np.nan), np.nan
    return {"tau": round(tau, 4), "tau_u": round(float(tau_u_val), 4),
            "variant": variant, "bounded": bool(abs(tau_u_val) <= 1.0 + 1e-9),
            "alternative": ("two-sided" if two_sided else "greater"),
            "baseline_trend_tau": round(tau_trend_a, 4),
            "baseline_trend_p": round(p_a, 4) if np.isfinite(p_a) else np.nan,
            "se": round(se, 4) if np.isfinite(se) else np.nan,
            "p_value": round(p, 4) if np.isfinite(p) else np.nan,
            "ci": ci, "critical": critical,
            "n_baseline": na, "n_treatment": nb}


def hedges_g(treatment, reference):
    """Standardized mean difference (treatment - reference) with the small-sample
    bias correction (Hedges 1981): ``g = J * (m_t - m_r)/s_pooled``. Returns NaN if a
    group has <2 points or zero pooled SD.

    References: Hedges 1981 (bias-corrected standardized mean difference).
    R equivalent: effectsize::hedges_g; metafor::escalc(measure="SMD")."""
    a = np.asarray(reference, float); b = np.asarray(treatment, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    d = (b.mean() - a.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)         # small-sample correction
    return float(J * d)


def pnd(baseline, treatment, *, improvement="increase"):
    """Percentage of Non-overlapping Data: % of treatment points beyond the best
    baseline point. Simple but biased (one extreme baseline point dominates) - 
    reported for comparability only. Not recommended as a primary effect size: WWC V1.0
    (Kratochwill et al. 2012, p. 7) rejects PND for its "undesirable statistical
    properties"; Michiels et al. (2017) show it as an RT statistic loses ~16% power vs
    MD/NAP. Prefer NAP / Tau-U (bounded, trend-aware).

    Returned in **percent** (0-100). Verified against SingleCaseES 0.7.4 (which reports the
    0-1 proportion): identical up to the x100 convention.

    References: Scruggs, Mastropieri & Casto 1987 (PND); Kratochwill et al. 2012 (WWC
    rejection); Michiels et al. 2017 (power).
    R equivalent: SingleCaseES::PND; scan::pnd."""
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    thr = a.max() if improvement == "increase" else a.min()
    beyond = np.sum(b > thr) if improvement == "increase" else np.sum(b < thr)
    return float(100 * beyond / len(b))


def pem(baseline, treatment, *, improvement="increase"):
    """Percentage Exceeding the Median: % of treatment points beyond the baseline
    median. Less extreme-sensitive than PND. Returned in **percent** (0-100); verified
    against SingleCaseES 0.7.4 (0-1 proportion) up to the x100 convention.

    References: Ma 2006 (Percentage Exceeding the Median).
    R equivalent: SingleCaseES::PEM; scan::pem."""
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(treatment), errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    med = np.median(a)
    beyond = np.sum(b > med) if improvement == "increase" else np.sum(b < med)
    return float(100 * beyond / len(b))


def nap_magnitude(x):
    """NAP bands (Parker & Vannest 2009): .50-.65 weak, .66-.92 medium, .93-1 strong."""
    if not np.isfinite(x):
        return "n/a"
    return "strong" if x > 0.92 else "medium" if x >= 0.66 else "weak/none"


def tau_magnitude(x):
    """Tau-U bands (Vannest & Ninci 2015): <.20 negligible, .20-.60 moderate, .60-.80 strong,
    >.80 very strong. NB: Fingerhut et al. (2021) recommend the **Critical Tau-U** rather
    than these thresholds - to be used as an indicative benchmark only."""
    if not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return ("negligible" if a < 0.20 else "moderate" if a < 0.60
            else "strong" if a <= 0.80 else "very strong")


def pnd_magnitude(x):
    """PND bands, % (Scruggs & Mastropieri 1998): >90 very effective, 70-90 effective,
    50-70 questionable, <50 ineffective. (PND not recommended as a primary ES - WWC 2012.)"""
    if not np.isfinite(x):
        return "n/a"
    return ("very effective" if x > 90 else "effective" if x >= 70
            else "questionable" if x >= 50 else "ineffective")


def pem_magnitude(x):
    """PEM bands, % (Ma 2006): >90 strong, 70-90 moderate, <70 weak/questionable."""
    if not np.isfinite(x):
        return "n/a"
    return "strong" if x > 90 else "moderate" if x >= 70 else "weak/questionable"


ICC_BANDS = "Koo & Li 2016: <0.50 weak, 0.50-0.75 moderate, 0.75-0.90 good, >=0.90 excellent"


I2_BANDS = "Higgins & Thompson 2003: <25% weak, 25-50% moderate, 50-75% strong, >=75% very strong"


G_BANDS = "Cohen: 0.2 small, 0.5 medium, 0.8 large"


NAP_BANDS = "Parker & Vannest 2009: .50-.65 weak, .66-.92 medium, .93-1 strong"


TAU_BANDS = ("Vannest & Ninci 2015 (indicative; prefer the Critical Tau-U): <.20 negligible, "
             ".20-.60 moderate, .60-.80 strong, >.80 very strong")


PND_BANDS = "Scruggs & Mastropieri 1998: >90 very effective, 70-90 effective, 50-70 questionable, <50 ineffective"


PEM_BANDS = "Ma 2006: >90 strong, 70-90 moderate, <70 weak/questionable"


def icc_band(x):
    """ICC interpretation band (Koo & Li 2016)."""
    if not np.isfinite(x):
        return "n/a"
    return "weak" if x < 0.50 else "moderate" if x < 0.75 else "good" if x < 0.90 else "excellent"


def i2_band(x):
    """I^2 heterogeneity band, in % (Higgins & Thompson 2003)."""
    if not np.isfinite(x):
        return "n/a"
    return "weak" if x < 25 else "moderate" if x < 50 else "strong" if x < 75 else "very strong"
