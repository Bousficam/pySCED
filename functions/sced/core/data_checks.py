"""
SCED core - data validation and descriptive checks.

Structure validation, plain-language pre-flight warnings, long-format checks and
the per-phase descriptive summary. Depends only on numpy / pandas.
"""
import numpy as np
import pandas as pd


def validate_sced(df, *, session_col, phase_col, outcome_col, unit_col=None):
    """
    Validate a long SCED table and return ``(clean_df, report)``.

    Coerces the outcome to numeric, drops rows with a missing outcome, sorts by
    (unit, session), and reports per-unit phase counts. ``session`` must be an
    orderable integer index of measurement occasions.

    Missing-data note (for audit) - this function uses **listwise deletion** (rows with a
    missing outcome are dropped). For a *randomization* test on missing single-case data,
    De, Michiels, Tanious & Onghena (2020, Behavior Research Methods) show the
    **randomized-marker** method (missing values carried as markers and permuted with the
    schedule) is the recommended default - equal or better power than imputation, ~2-10x
    faster, and Type-I-error-valid. The randomized-marker method is **not** implemented
    here; with non-trivial missingness, minimise it at the design stage (the dominant
    driver of power loss is the % missing, not the handling method; De et al. 2020).

    TODO(SCED, missing data): implement the **randomized-marker** RT (De et al. 2020) - 
    carry NA as a marker exchangeable with the schedule rather than dropping the row.
    ARIMA single-imputation and MICE are documented alternatives but lower-power / slower.
    """
    needed = [session_col, phase_col, outcome_col] + ([unit_col] if unit_col else [])
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' absent from the SCED DataFrame.")
    out = df.copy()
    out[outcome_col] = pd.to_numeric(out[outcome_col], errors="coerce")
    out[session_col] = pd.to_numeric(out[session_col], errors="coerce")
    n_before = len(out)
    out = out.dropna(subset=[outcome_col, session_col])
    sort_cols = ([unit_col] if unit_col else []) + [session_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    phases = sorted(out[phase_col].dropna().unique().tolist(), key=str)
    if unit_col:
        units = sorted(out[unit_col].dropna().unique().tolist(), key=str)
    else:
        units = ["all"]
    report = {
        "Number of units": len(units),
        "Phases/conditions": phases,
        "Number of phases": len(phases),
        "Total sessions": int(len(out)),
        "Sessions dropped (missing)": n_before - len(out),
        "Sessions per phase": {str(p): int((out[phase_col] == p).sum()) for p in phases},
    }
    return out, report


def sced_data_warnings(df, *, session_col, condition_col, outcome_col, unit_col=None,
                       conditions=None, min_per_condition=3, imbalance_ratio=2.0):
    """
    Friendly pre-flight checks on a SCED table - returns a list of plain-language
    warning strings (empty list = nothing to flag). Catches the issues that quietly
    weaken or break an analysis, so the user sees them up front rather than as a
    cryptic error or an over-confident result:

    - a condition entirely **missing** in a unit, or with **< min_per_condition**
      observations (effect sizes / per-unit estimates get unreliable);
    - **fewer than 2** observations in a condition (variance / Hedges' g undefined);
    - **non-integer or duplicated** session indices within a unit;
    - strong **count imbalance** between conditions (max/min > imbalance_ratio);
    - a group analysis with **< 2 units**.

    Purely advisory: it never raises and never alters the data.
    """
    w = []
    d = df.copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    if conditions is None:
        conditions = sorted(d[condition_col].dropna().unique().tolist(), key=str)
    units = (sorted(d[unit_col].dropna().unique().tolist(), key=str)
             if unit_col else [None])
    if unit_col and len(units) < 2:
        w.append(f"Group analysis requested but {len(units)} unit(s): "
                 "use the n=1 pipeline instead.")

    counts = {}
    for u in units:
        sub = d if u is None else d[d[unit_col] == u]
        tag = "" if u is None else f"[{u}] "
        sess = pd.to_numeric(sub[session_col], errors="coerce")
        if sess.notna().any() and not np.allclose(sess.dropna() % 1, 0):
            w.append(f"{tag}non-integer sessions - the time order must be an integer.")
        if sess.dropna().duplicated().any():
            w.append(f"{tag}duplicated session indices - one row per occasion expected.")
        for c in conditions:
            n = int(((sub[condition_col] == c) & sub[outcome_col].notna()).sum())
            counts.setdefault(c, []).append(n)
            if n == 0:
                w.append(f"{tag}condition '{c}' absent.")
            elif n < 2:
                w.append(f"{tag}condition '{c}': n={n} (<2) - variance/effect not computable.")
            elif n < min_per_condition:
                w.append(f"{tag}condition '{c}': n={n} (<{min_per_condition}) - estimates unreliable.")

    totals = {c: sum(v) for c, v in counts.items() if sum(v) > 0}
    if len(totals) >= 2 and min(totals.values()) > 0:
        if max(totals.values()) / min(totals.values()) > imbalance_ratio:
            w.append(f"Count imbalance between conditions {totals} "
                     f"(ratio > {imbalance_ratio}) - unequal power across conditions.")
    return w


def missing_data_warnings(df, *, outcome_col, session_col, unit_col=None):
    """Flag MISSING data on the raw table (run before validation, which drops it):
    counts of missing/non-numeric outcomes and missing sessions, the rows that will be
    removed, and the units most affected. Returns a list of warning strings (empty =
    nothing missing). Purely advisory."""
    w = []
    n = len(df)
    if n == 0:
        return ["Empty table."]
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    n_miss = int(y.isna().sum())
    if n_miss:
        w.append(f"{n_miss}/{n} ({100 * n_miss / n:.1f}%) missing or non-numeric "
                 "outcome value(s) -> removed from the analysis.")
        if unit_col and unit_col in df.columns:
            per = df.assign(_m=y.isna().to_numpy()).groupby(unit_col)["_m"].mean()
            worst = per[per > 0].sort_values(ascending=False)
            if len(worst):
                top = ", ".join(f"{u}:{100 * v:.0f}%" for u, v in worst.head(3).items())
                w.append(f"   missing per unit (top): {top}")
    s = pd.to_numeric(df[session_col], errors="coerce")
    n_s = int(s.isna().sum())
    if n_s:
        w.append(f"{n_s} non-numeric/missing session(s) -> row(s) removed.")
    return w


def check_long_format(df, *, session_col, condition_col, outcome_col, unit_col=None,
                      allow_repeated_measures=False):
    """
    Check that ``df`` is in the expected **long** SCED layout (one row per session:
    ``[unit,] session, condition, outcome``) and return a list of blocking problems
    (empty = OK). Catches the format mistakes that would otherwise crash or mislead the
    pipeline - missing columns, a non-numeric outcome, and the common **wide** layout
    (one column per condition) - with a hint to reshape via ``pandas.melt``. ``outcome_col``
    may be a string or a list (multivariate).

    ``allow_repeated_measures=True`` accepts **several rows per visit** (multiple raw
    measures / runs of the same ``[unit,] session``) - a legitimate input that the
    pipeline aggregates per visit (and can decompose at a 3rd level); the duplicate-visit
    check is then skipped.
    """
    out_cols = [outcome_col] if isinstance(outcome_col, str) else list(outcome_col)
    needed = ([unit_col] if unit_col else []) + [session_col, condition_col] + out_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [f"Absent columns: {missing}. LONG format expected - one row per session: "
                f"[{'unit, ' if unit_col else ''}session, condition, outcome]. "
                "If the data are in WIDE format (one column per condition), "
                "reshape to long with pandas.melt(...)."]
    problems = []
    for oc in out_cols:
        if pd.to_numeric(df[oc], errors="coerce").notna().sum() == 0:
            problems.append(f"The outcome '{oc}' is not numeric.")
    nlev = df[condition_col].nunique(dropna=True)
    if nlev < 2:
        problems.append(f"The condition column '{condition_col}' has only {nlev} level(s) "
                        "(>=2 expected) - data possibly in WIDE format? Reshape with pandas.melt.")
    elif nlev > 0.8 * len(df):
        problems.append(f"'{condition_col}' has almost as many levels as rows: "
                        "it is probably not a condition column.")
    levels = set(map(str, df[condition_col].dropna().unique()))
    cols_as_levels = [c for c in df.columns if str(c) in levels and c != condition_col]
    if cols_as_levels:
        problems.append(f"Some columns carry condition names {cols_as_levels}: "
                        "format probably WIDE -> pandas.melt to the long format.")
    if not allow_repeated_measures:
        keys = ([unit_col] if unit_col else []) + [session_col]
        dup = int(df.duplicated(subset=keys).sum())
        if dup:
            problems.append(f"{dup} row(s) with the same {keys}: a single row per session is "
                            "expected in long format (otherwise several measures per occasion -> "
                            "reshaping required, or pass allow_repeated_measures=True / MEASURES).")
    return problems


def _ols_slope(x, y):
    """Least-squares slope of y on x (NaN if <2 points or no x-variance)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.nanstd(x) == 0:
        return np.nan
    b = np.polyfit(x, y, 1)
    return float(b[0])


def describe_by_phase(df, *, session_col, phase_col, outcome_col, outcome_type="continuous"):
    """
    Per-phase summary: N, level (mean / median for skew or count, % for binary),
    within-phase trend (OLS slope vs session) and variability (SD). Returns a
    tidy DataFrame - the descriptive backbone of any SCED report.

    R equivalent: psych::describeBy - potential equivalent, to test.
    """
    phases = sorted(df[phase_col].dropna().unique().tolist(), key=str)
    rows = []
    for p in phases:
        sub = df[df[phase_col] == p]
        y = pd.to_numeric(sub[outcome_col], errors="coerce").dropna()
        n = len(y)
        if outcome_type == "binary":
            level = f"{int((y != 0).sum())}/{n} ({(y != 0).mean() * 100:.1f}%)" if n else ""
        elif outcome_type == "count":
            level = f"{y.median():.1f} (mean {y.mean():.2f})" if n else ""
        else:
            level = f"{y.mean():.2f} +/- {y.std(ddof=1):.2f}" if n > 1 else (f"{y.mean():.2f}" if n else "")
        rows.append({
            "Phase": str(p), "N": n, "Level": level,
            "Trend (slope/session)": round(_ols_slope(sub[session_col], sub[outcome_col]), 4) if n >= 2 else np.nan,
            "Variability (SD)": round(float(y.std(ddof=1)), 3) if n > 1 else np.nan,
        })
    return pd.DataFrame(rows, columns=["Phase", "N", "Level", "Trend (slope/session)", "Variability (SD)"])
