"""
SCED - Input preprocessing: case/whitespace + temporal columns to sessions
==========================================================================

Two frequent issues before analysis, independent of the statistical engines:

1) **Case / whitespace** - hand-entered condition labels ("A", "a", " A ",
   "Phase A") or column names ("Session" vs "session") break exact matches
   (``isin(conditions)``, ``COL[...]``). ``resolve_columns`` finds columns
   without regard to case; ``harmonize_conditions`` maps condition values back
   to canonical labels (case/whitespace insensitive).

2) **Dates** - the SCED session must be an **integer temporal order**. If the
   temporal column is a date, ``pd.to_numeric`` turns it into ``NaN`` (sorting/
   aggregation corrupted, rows dropped). ``dates_to_sessions`` converts dates
   into an ordered session index 1..n (per patient when grouped), with an
   optional "S_1", "S_2"... label.
"""
import numpy as np
import pandas as pd


def resolve_columns(df, mapping):
    """
    Resolve column names **case-insensitively** (and ignoring edge whitespace).
    ``mapping`` = dict ``role -> requested_name`` (``None`` values are kept).
    Returns ``role -> actual_name_in_df`` (the requested name is kept as-is when
    there is no match, so downstream checks can report the absence).
    """
    lut = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for role, name in mapping.items():
        if name is None:
            out[role] = None
        else:
            out[role] = lut.get(str(name).strip().lower(), name)
    return out


def harmonize_conditions(df, condition_col, *, conditions=None, case="upper", strip=True):
    """
    Harmonize the values of ``condition_col`` (case/whitespace). If ``conditions``
    (canonical labels) is supplied, each value is mapped back to the matching
    canonical label case/whitespace-insensitively ("a"/" A " -> "A"); values with
    no match are left as-is. Otherwise, plain ``upper``/``lower`` normalization +
    strip. Returns a COPY of ``df``.
    """
    d = df.copy()
    raw = d[condition_col].astype("string")
    if strip:
        raw = raw.str.strip()
    if conditions is not None:
        canon = {str(c).strip().lower(): c for c in conditions}
        d[condition_col] = raw.map(
            lambda v: canon.get(str(v).strip().lower(), v) if not pd.isna(v) else v)
    else:
        d[condition_col] = raw.str.upper() if case == "upper" else raw.str.lower()
    return d


def harmonize_values(df, columns, *, case="upper", strip=True):
    """Normalize the case/whitespace of free categorical columns (e.g. ``outcome_name``
    in tidy format) to make matching reliable. Returns a COPY of ``df``."""
    d = df.copy()
    for col in (columns if isinstance(columns, (list, tuple)) else [columns]):
        if col in d.columns:
            s = d[col].astype("string")
            if strip:
                s = s.str.strip()
            d[col] = s.str.upper() if case == "upper" else s.str.lower()
    return d


def dates_to_sessions(df, date_col, *, unit_col=None, new_col="session", label=False,
                      label_col=None, label_prefix="S_", dayfirst=False, keep_date=True):
    """
    Convert a **date** column into an ordered integer **session** 1..n (the temporal
    order expected by the SCED engines). The rank is computed PER PATIENT when
    ``unit_col`` is supplied (otherwise global). Identical dates -> same session
    number (dense rank). With ``label=True``, adds a text label "S_1", "S_2"...
    (column ``label_col`` or ``f"{new_col}_label"``). ``dayfirst=True`` for
    DD/MM/YYYY dates. Returns a COPY.
    """
    d = df.copy()
    dt = pd.to_datetime(d[date_col], errors="coerce", dayfirst=dayfirst)
    if dt.notna().sum() == 0:
        raise ValueError(f"No interpretable date in '{date_col}' "
                         "(try dayfirst=True for the DD/MM/YYYY format).")
    d["_dt_"] = dt
    if unit_col:
        rank = d.groupby(unit_col)["_dt_"].rank(method="dense")
    else:
        rank = d["_dt_"].rank(method="dense")
    d[new_col] = rank.astype("Int64")
    if label:
        lc = label_col or f"{new_col}_label"
        d[lc] = label_prefix + d[new_col].astype("string")
    return d.drop(columns="_dt_") if keep_date else d.drop(columns=["_dt_", date_col])


# ---------------------------------------------------------------------------
# Reshaping measures -> visit (repeated within-visit measures)
# These functions collapse repeated measures within a visit into one value per
# visit (or into a long measure-per-row format). They prepare the data for the
# 3-level variance decomposition (functions.sced.intravisit) as well as for the
# standard visit-level pipelines.
# ---------------------------------------------------------------------------

def aggregate_visits(df, *, unit_col, session_col, condition_col, outcome_col, stat="mean"):
    """Collapse measure-level rows to one row per (unit, visit) with ``stat`` in
    {mean, median, sd, cv, n}. The condition (constant within a visit) is carried over,
    and ``_n`` (number of non-missing measures) is added for optional weighting."""
    def agg(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return np.nan
        if stat == "mean":
            return float(s.mean())
        if stat == "median":
            return float(s.median())
        if stat == "sd":
            return float(s.std(ddof=1)) if len(s) > 1 else np.nan
        if stat == "cv":
            m = s.mean()
            return float(s.std(ddof=1) / m) if (len(s) > 1 and m != 0) else np.nan
        if stat == "n":
            return float(len(s))
        raise ValueError("stat must be mean / median / sd / cv / n")

    rows = []
    for (u, v), g in df.groupby([unit_col, session_col]):
        rows.append({unit_col: u, session_col: v,
                     condition_col: g[condition_col].iloc[0],
                     outcome_col: agg(g[outcome_col]),
                     "_n": int(pd.to_numeric(g[outcome_col], errors="coerce").notna().sum())})
    return pd.DataFrame(rows).sort_values([unit_col, session_col]).reset_index(drop=True)


def aggregate_runs(df, *, session_col, condition_col, run_cols, unit_col=None, agg="mean"):
    """
    Harmonize a **WIDE** input where each visit carries several raw measures in
    columns (``run_1, run_2, ... run_n``): computes the per-visit aggregate
    (mean/median) AND builds the **LONG** measure-per-row version required by the
    third hierarchical level.

    Allows supplying either a value already aggregated per visit (classic path,
    without ``run_cols``) or the raw runs (here) - hence the option to include or
    not the 3-level decomposition (measures > visit > patient) without changing
    pipeline.

    Returns ``(df_visit, long_runs, out_col)``:
      - ``df_visit`` : copy of ``df`` + a column ``out_col`` = per-visit aggregate;
      - ``long_runs`` : long format (one row per measure, column ``_measure``) for
        ``variance_decomposition_3level`` / ``dispersion_test``;
      - ``out_col`` : name of the aggregated column (to pass as outcome to the engine).
    """
    run_cols = list(run_cols)
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")
    out = f"_visit_{agg}"
    d = df.copy()
    runs = d[run_cols].apply(pd.to_numeric, errors="coerce")
    d[out] = runs.median(axis=1) if agg == "median" else runs.mean(axis=1)
    keep = ([unit_col] if unit_col else []) + [session_col, condition_col]
    long = d.melt(id_vars=keep, value_vars=run_cols, var_name="_run", value_name="_measure")
    long["_measure"] = pd.to_numeric(long["_measure"], errors="coerce")
    long["_trial"] = long["_run"].map({c: i + 1 for i, c in enumerate(run_cols)})  # within-visit order
    return d, long, out


def collapse_long_measures(df, *, session_col, condition_col, value_col, unit_col=None,
                           agg="mean", trial_col=None, keep_cols=None):
    """
    **LONG** format: when several rows share the same visit
    (``[unit,] session, condition``), they are **repeated measures** of that visit.
    This function aggregates them to one value per visit (mean/median) AND returns
    the measure-per-row detail - the "long" equivalent of :func:`aggregate_runs`
    (which instead starts from ``run_1...run_n`` columns). No column to name: the
    structure is inferred from the duplicated visit keys. ``trial_col`` (optional)
    gives the order of the measure within the visit (for the within-visit slope);
    otherwise the row order is used. ``keep_cols`` = ancillary per-visit columns
    (covariates) carried over to the visit-level frame (assumed constant within a
    visit, first value kept). Returns ``(df_visit, long_measures, out_col)``.
    """
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")
    keep = ([unit_col] if unit_col else []) + [session_col, condition_col]
    extra = [c for c in (keep_cols or []) if c in df.columns and c not in keep]   # covariates (per visit)
    cols = keep + [value_col] + ([trial_col] if trial_col else []) + extra
    long = df[cols].rename(columns={value_col: "_measure"}).copy()
    long["_measure"] = pd.to_numeric(long["_measure"], errors="coerce")
    if trial_col:
        long = long.rename(columns={trial_col: "_trial"})
    else:
        long["_trial"] = long.groupby(keep).cumcount() + 1          # within-visit row order
    out = f"_visit_{agg}"
    g = (long.groupby(keep, as_index=False)["_measure"]
             .agg(agg).rename(columns={"_measure": out}))
    if extra:                                                       # re-attach the per-visit covariates
        g = g.merge(long.groupby(keep, as_index=False)[extra].first(), on=keep, how="left")
    return g, long, out


def unstack_outcomes(df, *, outcome_name_col, value_col, id_cols, trial_col=None,
                     keep_cols=None):
    """
    **TIDY / outcome-stacked** format: a column ``outcome_name_col`` names the outcome,
    a column ``value_col`` carries the score, **one row per score**. Pivots to the
    wide format expected by the engine (one column per outcome).

    ``id_cols`` = visit keys ([unit,] session, condition). ``trial_col`` (optional) =
    order of the repeated measure within the visit: kept as a column to preserve the
    repeated measures (otherwise, if several scores of the SAME outcome share a visit,
    an occurrence index ``_occ`` is created automatically so nothing is overwritten).
    ``keep_cols`` = ancillary columns to carry over (covariates) - assumed constant
    per visit. Returns ``(df_wide, outcome_names)``.
    """
    id_cols = [c for c in id_cols if c in df.columns]
    extra = [c for c in (keep_cols or []) if c in df.columns and c not in id_cols]
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    if trial_col and trial_col in tmp.columns:
        index = id_cols + extra + [trial_col]
    elif tmp.duplicated(subset=id_cols + [outcome_name_col]).any():
        tmp["_occ"] = tmp.groupby(id_cols + [outcome_name_col]).cumcount() + 1
        index = id_cols + extra + ["_occ"]
    else:
        index = id_cols + extra
    wide = tmp.pivot_table(index=index, columns=outcome_name_col, values=value_col,
                           aggfunc="first").reset_index()
    wide.columns.name = None
    names = [c for c in wide.columns if c not in index]
    if not names:
        raise ValueError("unstack_outcomes: no outcome reconstructed - check "
                         f"outcome_name_col={outcome_name_col!r} / value_col={value_col!r}.")
    return wide, names
