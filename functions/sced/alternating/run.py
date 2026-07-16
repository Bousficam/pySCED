"""
SCED - Unified dispatcher for alternating designs (n-of-1 / group / multivariate)
=================================================================================

A single entry point, ``run_sced_alternating``, that **chooses the right engine**
according to the shape of the data - instead of calling the correct ``pipeline_...`` by
hand:

    | outcome(s) | unit_col | engine called                         | function              |
    |------------|----------|---------------------------------------|-----------------------|
    | 1          | None     | n-of-1 (one patient)                  | pipeline_sced_alternating        |
    | 1          | provided | group (stratified, patient = block)   | pipeline_sced_alternating_group  |
    | >=2        | None/provided | multivariate (PERMANOVA)         | permanova_condition_test         |

``explain_statistic_choice`` documents - and is callable from a template - the logic for
selecting the engine AND the test statistic, and how it works.

The engines themselves stay distinct (different maths: permutation partial F vs PERMANOVA
pseudo-F); only the *choice* is centralised here.
"""
import fnmatch
import inspect
import os

import numpy as np
import pandas as pd


def _level3_to_df(level3):
    """Tidy the ``level3`` dict (variance decomposition + dispersion + within-visit slope,
    per condition AND per patient) into a report sheet."""
    s = level3["summary"]
    rows = [
        {"Quantity": "Variance between-patient (%)", "Value": s.get("pct_patient"),
         "Interpretation": "LEVEL differences between patients"},
        {"Quantity": "Variance between-visit (%)", "Value": s.get("pct_visit"),
         "Interpretation": "fluctuation from one visit to the next (within-patient)"},
        {"Quantity": "Variance within-visit (%)", "Value": s.get("pct_within_visit"),
         "Interpretation": "measurement noise between runs of the same visit"},
        {"Quantity": "ICC patient", "Value": s.get("icc_patient"),
         "Interpretation": "share of between-patient variance (cf. Hierarchical sheet)"},
        {"Quantity": "Condition effect - p (visit level)", "Value": s.get("p_common_effect_perm"),
         "Interpretation": "permutation at the visit level"},
    ]
    d = level3.get("dispersion")
    if d:
        rows.append({"Quantity": f"Within-visit dispersion ({d.get('dispersion_measure')}) - p",
                     "Value": round(d["p_value"], 4),
                     "Interpretation": "does the condition change the within-visit REGULARITY?"})
    w = level3.get("within_visit_slope")
    if w:
        rows += [
            {"Quantity": "Within-visit slope - p", "Value": round(w["p_value"], 4),
             "Interpretation": "does the condition change the within-session TRAJECTORY?"},
            {"Quantity": "Within-visit slope per condition", "Value": str(w.get("slope_by_condition")),
             "Interpretation": "<0 = decline/fatigue, >0 = rise/warm-up"},
            {"Quantity": "Within-visit slope per patient", "Value": str(w.get("slope_by_patient")),
             "Interpretation": "within-session trajectory of each patient"},
        ]
    return pd.DataFrame(rows, columns=["Quantity", "Value", "Interpretation"])


def _append_level3_sheet(save_path, model_info, level3):
    """Add a "Level 3" sheet to the already-written group report (otherwise the 3rd level
    appears only in the console), with p-value colouring (green = significant)."""
    import numpy as np
    import pandas as pd
    outcome = model_info.get("Outcome")
    path = os.path.join(save_path, f"{outcome}_sced_alternating_group_report.xlsx")  # flat (see group.py)
    if not os.path.exists(path):
        return
    from functions.sced.glossary import ascii_sanitize_df as S, style_cells, _to_float
    sd = S(_level3_to_df(level3))

    def _l3_valeur_fill(row):                            # colour the Value cell according to the Quantity
        q = str(row.get("Quantity", "")).lower()
        if "p (" in q or "- p" in q or q.endswith(" p"):
            a = _to_float(row.get("Value"))
            if a is not None and np.isfinite(a):
                return (True, "C6EFCE") if a < 0.05 else (False, "F2F2F2")
        return (False, None)

    try:                                                 # silent if the file is locked/open
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            sd.to_excel(w, sheet_name="Level 3", index=False)
            style_cells(w.sheets["Level 3"], sd, {}, row_rules={"Value": _l3_valeur_fill})
    except Exception:
        pass


def _write_permanova_report(save_path, result, outcomes, *, grouped):
    """Write the .xlsx report of the MULTIVARIATE engine (PERMANOVA) - otherwise this engine
    produced no file (>=2 outcomes). Sheets: Summary, By outcome, Method, Explanations."""
    os.makedirs(save_path, exist_ok=True)               # flat, like the n-of-1 / group reports
    fname = "-".join(str(o) for o in outcomes)[:80] + "_permanova_report.xlsx"
    path = os.path.join(save_path, fname)
    dfnum, dfden = result.get("df", (np.nan, np.nan)) if isinstance(result.get("df"), tuple) else (np.nan, np.nan)
    resume = pd.DataFrame([
        {"Quantity": "observed pseudo-F", "Value": result.get("pseudo_F"),
         "Interpretation": "multivariate gap between conditions (distances); larger = clearer"},
        {"Quantity": "p (permutation)", "Value": result.get("p_value"),
         "Interpretation": "p=(1+#>=obs)/(1+B); <0.05 = joint profile shifted by the condition"},
        {"Quantity": "Outcomes (dimensions)", "Value": result.get("n_outcomes"),
         "Interpretation": "number of response variables analysed JOINTLY"},
        {"Quantity": "Units (patients)", "Value": result.get("n_units"),
         "Interpretation": "within-patient stratified permutation" if grouped else "single case"},
        {"Quantity": "Permutations", "Value": result.get("n_perm"), "Interpretation": "B"},
        {"Quantity": "df (num, den)", "Value": f"{dfnum}, {dfden}", "Interpretation": "degrees of freedom of the pseudo-F"},
        {"Quantity": "Rows dropped (NaN)", "Value": result.get("n_dropped_missing"),
         "Interpretation": "incomplete cases removed: a multivariate distance requires ALL dimensions"},
    ], columns=["Quantity", "Value", "Interpretation"])
    methode = pd.DataFrame({"Method (PERMANOVA - multivariate response)": [
        "Several outcomes = multivariate response: tests whether the condition shifts the JOINT",
        "PROFILE of the outcomes, via PERMANOVA (pseudo-F on a Euclidean distance matrix,",
        "p by permutation; Anderson 2001, doi:10.1111/j.1442-9993.2001.01070.x). Valid even",
        "if p>n. At group level, the permutation is stratified within patient (Edgington & Onghena 2007).",
        "Univariate follow-ups per outcome (permutation F) with Holm (FWER) and BH (FDR) correction",
        "to locate WHICH outcome carries the effect. NB: incomplete cases are dropped (complete-case)",
        "because a multivariate distance requires all dimensions - see 'Rows dropped'.",
    ]})
    explic = pd.DataFrame({
        "Column (By outcome)": ["Outcome", "F", "p (perm)", "p Holm", "p FDR"],
        "Explanation": [
            "outcome name (response dimension)",
            "permutation F statistic of the univariate follow-up",
            "raw permutation p for this outcome",
            "Holm-corrected p (controls the risk of >=1 false positive over the set - FWER)",
            "Benjamini-Hochberg-corrected p (controls the expected proportion of false positives - FDR)"]})
    from functions.sced.glossary import interpretation_glossary, write_report
    return write_report(path, {                       # ASCII guaranteed per sheet
        "Summary": resume,
        "By outcome": result.get("per_outcome", pd.DataFrame()),
        "Method": methode,
        "Explanations": explic,
        "Interpretation guide": interpretation_glossary("mv"),
    })


def _resolve_measure_cols(df, spec, exclude):
    """Resolve ``measures`` into a list of columns: explicit list, OR a **prefix** (all
    columns starting with the string), OR a **glob** (if the string contains * ? [).
    ``exclude`` = structural columns never to take (keys, outcome, covariates)."""
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, str):
        is_glob = any(ch in spec for ch in "*?[")
        cols = [c for c in df.columns if c not in exclude and (
            fnmatch.fnmatch(str(c), spec) if is_glob else str(c).startswith(spec))]
        if not cols:
            raise ValueError(f"No column matches measures={spec!r}.")
        return cols
    raise ValueError("measures must be a list, a prefix/glob (str), or None.")


def _call_filtered(fn, df, **kwargs):
    """Call ``fn(df, **kwargs)`` keeping only the kwargs RECOGNISED by ``fn`` - each engine
    thus ignores the parameters that do not concern it (e.g. PERMANOVA ignores
    ``trend_test``/``dose``)."""
    ok = set(inspect.signature(fn).parameters)
    return fn(df, **{k: v for k, v in kwargs.items() if k in ok})


def run_sced_alternating(df, *, session_col, condition_col, outcome=None, measures=None,
                         run_cols=None, unit_col=None, level3=False, agg="mean",
                         dispersion="sd", trial_col=None, within_visit_slope=False,
                         outcome_name_col=None, value_col=None, **kwargs):
    """
    Dispatcher: runs the right alternating-SCED engine according to the data.

    **Flexible visit input** - three ways to provide the measure, without naming anything
    specific:
      - ``outcome`` = value already aggregated per visit (str; or list >=2 -> multivariate);
      - ``measures`` = raw per-visit measures, in **columns**: explicit list, OR a
        **prefix** ("run" takes run_1, run_2...), OR a **glob** ("trial_*");
      - **LONG format**: pass nothing special - if several rows share the same visit
        (``[unit,] session, condition``), they are detected as repeated measures and
        aggregated automatically.
      - **TIDY format (stacked outcome)**: ``outcome_name_col`` (column naming the outcome) +
        ``value_col`` (score column), one row per score. Unpivoted to wide: 1 distinct
        outcome -> univariate; >=2 -> multivariate (PERMANOVA). Pass ``outcome`` to analyse
        only a subset of the stacked outcomes.
    In the last two cases the ``agg`` aggregate (mean/median) per visit is computed for the
    standard analysis and the detail is kept for the 3rd level. (``run_cols`` = former name
    of ``measures``, kept as an alias.)

    **Hierarchical 3rd level (optional)** - ``level3=True`` (with ``run_cols`` and
    ``unit_col``) adds the variance decomposition **measures > visit > patient**
    (between-patient / between-visit / within-visit = measurement noise), visit-level p
    by permutation. No separate pipeline needed: it is the same call.

    ``unit_col``: None = one patient; provided = replicated group. The other settings
    (conditions, time_covariate, perm_method, max_consecutive, block_size, trend_test,
    condition_order, dose, covariate_cols, hierarchical, n_perm, save_path, ...) are
    forwarded to the selected engine if it accepts them.

    Returns ``{engine, model_info, result}`` (+ ``level3`` if computed).

    References: Anderson 2001 (PERMANOVA, multivariate branch); Edgington & Onghena
    2007 (randomization, univariate branches).
    R equivalent: vegan::adonis2 (PERMANOVA branch); SCRT (randomization branches) -
    the dispatch itself has no single equivalent.
    """
    from functions.sced.alternating.core import pipeline_sced_alternating
    from functions.sced.alternating.group import pipeline_sced_alternating_group

    # TIDY format: outcome(s) stacked in rows (1 column = name, 1 column = score)
    # -> unpivot to wide (1 column per outcome) before everything else.
    if outcome_name_col is not None and value_col is not None:
        from functions.sced.prep import unstack_outcomes
        ids = ([unit_col] if unit_col else []) + [session_col, condition_col]
        df, names = unstack_outcomes(
            df, outcome_name_col=outcome_name_col, value_col=value_col, id_cols=ids,
            trial_col=trial_col, keep_cols=kwargs.get("covariate_cols"))
        if outcome is None:                                  # otherwise: subset requested explicitly
            outcome = names if len(names) >= 2 else names[0]
        trial_col = trial_col if trial_col else ("_occ" if "_occ" in df.columns else None)

    measures = measures if measures is not None else run_cols   # backward-compat alias
    keys = ([unit_col] if unit_col else []) + [session_col, condition_col]
    long_runs = None
    if measures is not None:                                 # measures in COLUMNS (list/prefix/glob)
        from functions.sced.prep import aggregate_runs
        excl = set(keys) | {outcome} | set(o for o in (
            [] if outcome is None else ([outcome] if isinstance(outcome, str) else list(outcome))))
        excl |= set(kwargs.get("covariate_cols") or [])
        cols = _resolve_measure_cols(df, measures, excl)
        df, long_runs, outcome = aggregate_runs(
            df, session_col=session_col, condition_col=condition_col,
            run_cols=cols, unit_col=unit_col, agg=agg)
    elif (isinstance(outcome, str) and outcome in df.columns
          and df.duplicated(subset=keys).any()):            # LONG format: repeated measures per visit
        from functions.sced.prep import collapse_long_measures
        df, long_runs, outcome = collapse_long_measures(
            df, session_col=session_col, condition_col=condition_col,
            value_col=outcome, unit_col=unit_col, agg=agg, trial_col=trial_col,
            keep_cols=kwargs.get("covariate_cols"))
    if outcome is None:
        raise ValueError("Provide 'outcome' (per-visit value) or 'measures' (raw measures).")

    outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
    common = dict(session_col=session_col, condition_col=condition_col, **kwargs)

    if len(outcomes) >= 2:                                   # >=2 outcomes -> multivariate
        from functions.sced.multivariate import permanova_condition_test
        res = _call_filtered(permanova_condition_test, df, outcome_cols=outcomes,
                             unit_col=unit_col, **common)
        out = {"engine": "multivariate - PERMANOVA (pseudo-F by permutation)",
               "model_info": None, "result": res}
        if kwargs.get("save_path"):                          # the multivariate engine also writes a report
            rep = _write_permanova_report(kwargs["save_path"], res, outcomes,
                                          grouped=unit_col is not None)
            out["report_path"] = rep
    elif unit_col is not None:                               # 1 outcome, >=2 patients -> group
        mi, res = _call_filtered(pipeline_sced_alternating_group, df, unit_col=unit_col,
                                 outcome_col=outcomes[0], **common)
        out = {"engine": "group - stratified test (patient = block)",
               "model_info": mi, "result": res}
    else:                                                    # 1 outcome, 1 patient -> n-of-1
        mi, res = _call_filtered(pipeline_sced_alternating, df,
                                 outcome_col=outcomes[0], **common)
        out = {"engine": "n-of-1 - single patient", "model_info": mi, "result": res}

    # 3rd level (measures > visit > patient), optional - same call, no separate pipeline
    if level3:
        if long_runs is None:
            out["level3_note"] = "level3 requested without run_cols (raw measures) - ignored."
        elif unit_col is None:
            out["level3_note"] = "level3 requires >=2 patients (unit_col) - ignored."
        else:
            from functions.sced.intravisit import variance_decomposition_3level, dispersion_test, within_visit_slope_test
            tc = kwargs.get("time_covariate", kwargs.get("detrend", "none")) or "none"
            tc = "none" if tc == "auto" else tc
            l3common = dict(unit_col=unit_col, session_col=session_col,
                            condition_col=condition_col, outcome_col="_measure",
                            detrend=tc, n_perm=kwargs.get("n_perm", 5000),
                            random_state=kwargs.get("random_state", 0))
            l3sum, l3sheet = variance_decomposition_3level(long_runs, **l3common)
            level3 = {"summary": l3sum, "sheet": l3sheet}
            # does the condition affect the within-visit REGULARITY? (SD/CV per visit)
            if dispersion in ("sd", "cv"):
                dres, dvis = dispersion_test(long_runs, measure=dispersion, **l3common)
                level3["dispersion"] = dres
            # slope WITHIN the visit (fatigue/warm-up) - secondary/OPTIONAL (off by default)
            if within_visit_slope and "_trial" in long_runs.columns:
                sres, svis = within_visit_slope_test(long_runs, trial_col="_trial", **l3common)
                level3["within_visit_slope"] = sres
            out["level3"] = level3
            # write the 3rd level INTO the xlsx report (otherwise console only)
            if kwargs.get("save_path") and out.get("model_info") is not None:
                _append_level3_sheet(kwargs["save_path"], out["model_info"], level3)
    return out


def run_sced_by_group(df, *, group_col, session_col, condition_col, **kwargs):
    """
    Analyses several **SCED groups** SEPARATELY (cohorts - e.g. acute vs chronic patients).
    ``group_col`` is a level ABOVE the patient: not to be confused with ``unit_col`` (the
    patient) or ``condition_col`` (the randomised dose/condition). For each value of
    ``group_col``, the subset is passed to :func:`run_sced_alternating` (which chooses the
    engine - n-of-1 / group / multivariate - according to that subset).

    Returns a dict ``{group_value: result}`` where each result is either the dict returned
    by ``run_sced_alternating``, or ``{"error": message}`` if that group could not be
    analysed (insufficient sample size, etc.) - a failing group does not interrupt the
    others. ``save_path`` (if provided) is suffixed by a sub-folder named after the group
    so the reports are not overwritten.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' absent.")
    base_save = kwargs.pop("save_path", None)
    out = {}
    for gval, gdf in df.groupby(group_col):
        sp = (os.path.join(base_save, str(gval)) if base_save else None)
        try:
            out[gval] = run_sced_alternating(gdf, session_col=session_col,
                                             condition_col=condition_col, save_path=sp, **kwargs)
        except Exception as e:                                # a failing group does not stop the others
            out[gval] = {"error": f"{type(e).__name__}: {e}"}
    return out


def explain_statistic_choice(*, n_conditions=None, ordered=False, multivariate=False,
                             grouped=False, has_covariate=False):
    """
    Explains HOW the engine and the test statistic are chosen, and how they work - text
    ready to print in a template. The arguments highlight the relevant path; with no
    argument, the full tree is returned.
    """
    head = (
        "ENGINE CHOICE (run_sced_alternating)\n"
        "  - >=2 outcomes             -> MULTIVARIATE: PERMANOVA (pseudo-F on distances, by\n"
        "                               permutation; Anderson 2001). Works even if p>n.\n"
        "  - 1 outcome + several patients -> GROUP: stratified test (patient = block,\n"
        "                               within-patient permutation; Edgington & Onghena 2007).\n"
        "  - 1 outcome + 1 patient    -> N-OF-1: simple randomization test.\n")
    stat = (
        "\nSTATISTIC CHOICE (within a univariate engine)\n"
        "  1) Is time put in as a COVARIATE (time_covariate != none; former 'detrend')?\n"
        "     - YES -> linear model with time as covariate -> ANCOVA; the p comes from\n"
        "             permuting the RESIDUALS of the reduced model (Freedman-Lane; Freedman &\n"
        "             Lane 1983; Winkler 2014). Statistic = PARTIAL F (condition effect\n"
        "             beyond time).\n"
        "     - NO -> ANOVA: the condition LABELS are permuted directly.\n"
        "  2) How many conditions?\n"
        "     - 2  -> statistic = DIFFERENCE OF MEANS (two-sided test).\n"
        "     - >=3 -> statistic = VARIANCE OF between-condition MEANS (ANOVA, one-sided)\n"
        "             [= partial F if covariate]. Tests 'at least one condition differs'.\n"
        "  3) ORDERED conditions (dose, e.g. MI 4/5/6 s)? -> IN ADDITION: dose-response trend\n"
        "     (slope on the dose + Jonckheere-Terpstra). To pre-specify; not validated for\n"
        "     alternating designs (Michiels 2017; Manolov 2021).\n"
        "  4) Post-hoc (>=3 conditions): contrasts of ALL pairs (Hedges' g + CI,\n"
        "     Holm/FDR-corrected p) = WHICH conditions differ.\n")
    grp = (
        "\nGROUP LEVEL (in addition)\n"
        "  - The COMMON effect is tested by the same partial F, stratified (within-patient\n"
        "    permutation), within-patient z-scored values.\n"
        "  - HETEROGENEITY: does the effect vary between patients? (interaction + Cochran Q/I2).\n"
        "  - Per patient: INDIVIDUAL n-of-1 test + temporal slope (OLS & Sen) + Mann-\n"
        "    Kendall/Hamed-Rao.\n")
    why_permanova = (
        "\nWHY PERMANOVA IN THE MULTIVARIATE CASE\n"
        "  Several outcomes = multivariate response: tests whether the condition shifts the\n"
        "  JOINT PROFILE (not a nuisance variable - that would be a covariate). Univariate\n"
        "  follow-ups per outcome (Holm/FDR) to see WHICH outcome carries the effect.\n")
    parts = [head, stat]
    if grouped:
        parts.append(grp)
    if multivariate:
        parts.append(why_permanova)
    if not grouped and not multivariate:
        parts.append(grp); parts.append(why_permanova)
    return "".join(parts)


def report_sced_alternating(
        df, *, unit_col=None, session_col="session", condition_col="condition",
        outcome_name_col=None, value_col=None, outcomes=("score",), outcome_mode="joint",
        case_insensitive=True, date_col=None, date_dayfirst=False, group_col=None,
        measures=None, agg="mean", level3=False, dispersion=None, within_visit_slope=False,
        trial_col=None, covariates=None, conditions=("A", "B", "C"), outcome_type="continuous",
        improvement="increase", max_consecutive=None, block_size=None, trend_test=True,
        test_time_trend=True, condition_order=("A", "B", "C"), dose=(4, 5, 6), time_covariate="auto",
        perm_method="freedman-lane", standardize=True, hierarchical=False, n_perm=5000,
        save_path=None, plot_path=".", output_dir=None, plot_kind="series", plot_by="session",
        plot_center="mean", plot_points=True, plot_cond_line=True, plot_vaior=False,
        vaior_ref="A", vaior_pooled=False, plot_brinley=True, brinley_mode="classic",
        bounds=None, y_percent=False, row_height=1.7, plot_panels=True, style=None, verbose=True):
    """REPORT-LEVEL alternating SCED pipeline: prep (case/dates/tidy), engine choice,
    cohort loop (x outcomes if separate), figures (dashboard / VAIOR / Brinley) and call to
    ``run_sced_alternating`` (styled .xlsx report). The template only calls; all the
    orchestration lives here. ``output_dir`` = single root (standard layout built into the
    pipeline): xlsx -> ``Analyse/Inferentielle/``; figures -> ``Plot/``."""
    import os
    from functions.sced.core import sced_data_warnings, missing_data_warnings, check_long_format
    from functions.sced.plots.panels import plot_sced_alternating, plot_sced_panels
    from functions.sced.glossary import explain, std_layout, slug_token
    uc, sc, cc = unit_col, session_col, condition_col
    on, vc = outcome_name_col, value_col

    # --- STEP 0a: case-insensitivity (columns + conditions) ---
    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        res = resolve_columns(df, {"unit": uc, "session": sc, "condition": cc, "date": date_col,
                                   "group": group_col, "outcome_name": on, "value": vc})
        uc, sc, cc, date_col, group_col = (res["unit"], res["session"], res["condition"],
                                           res["date"], res["group"])
        on, vc = res["outcome_name"], res["value"]
        df = harmonize_conditions(df, cc, conditions=conditions)

    # --- STEP 0b: dates -> ordered session 1..n (per patient if group) ---
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=uc, new_col=sc, dayfirst=date_dayfirst)
        if verbose:
            print(f"STEP 0 - Dates '{date_col}' -> ordered session 1..n"
                  + (" (per patient)" if uc else "") + f" in '{sc}'")

    outcomes = list(outcomes) if outcomes else []
    # --- TIDY format: unpivot stacked outcome(s) -> 1 column per outcome ---
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        ids = ([uc] if uc else []) + [sc, cc]
        keep = (list(covariates) if covariates else []) + ([group_col] if group_col else [])
        df, _names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=ids,
                                      trial_col=trial_col, keep_cols=keep)
        missing = [o for o in outcomes if o not in _names]
        if outcomes and missing:
            if verbose:
                print(f"  Warning: OUTCOMES {missing} absent from the stacked outcomes {_names} -> analysing ALL.")
            outcomes = _names
        elif not outcomes:
            outcomes = _names
        if verbose:
            print(f"STEP 0 - Tidy format unpivoted: outcomes = {outcomes}")

    if output_dir is not None:                              # PER-OUTCOME layout: <oc>/Analyse + <oc>/Plot
        _otag = slug_token(outcomes[0]) if len(outcomes) == 1 else "multivariate"
        L = std_layout(output_dir, _otag)                  # report straight in Analyse/ (no deep subtree)
        if save_path is None:
            save_path = L["analyse"]
        if plot_path is None or plot_path == ".":
            plot_path = L["plot"]
    multivariate = (not measures) and len(outcomes) >= 2
    repeated = (bool(measures) or bool(trial_col) or level3
                or df.duplicated(subset=([uc] if uc else []) + [sc, cc]).any())

    if verbose:
        print("\nSTEP 1 - Engine/statistic choice:")
        print(explain_statistic_choice(n_conditions=len(conditions), ordered=trend_test,
                                       multivariate=(multivariate and outcome_mode == "joint"),
                                       grouped=uc is not None, has_covariate=covariates is not None))

    def _bnd(oc):
        return bounds.get(oc) if isinstance(bounds, dict) else bounds

    def _analyze(gdf, label, pp, sp, oc_list):
        """Full analysis of ONE (sub)set: format -> plots -> test -> read-out."""
        mv = (not measures) and len(oc_list) >= 2
        for oc in oc_list:
            fmt = check_long_format(gdf, session_col=sc, condition_col=cc, outcome_col=oc,
                                    unit_col=uc, allow_repeated_measures=repeated)
            if fmt:
                if verbose:
                    print(f"INVALID FORMAT{label}:", *fmt, sep="\n  - ")
                return None
        warns = (missing_data_warnings(gdf, outcome_col=oc_list[0], session_col=sc, unit_col=uc)
                 + sced_data_warnings(gdf, session_col=sc, condition_col=cc, outcome_col=oc_list[0],
                                      unit_col=uc, conditions=conditions))
        if verbose:
            print(f"STEP 0 - Data check{label}:", "OK" if not warns else "")
            for w in warns:
                print("  Warning:", w)

        # --- STEP 2: visualisation ---
        if pp and plot_panels:
            plot_sced_panels(gdf, session_col=sc, condition_col=cc, outcome_col=oc_list[0],
                             unit_col=uc, conditions=conditions, reference=vaior_ref,
                             improvement=improvement, kind=plot_kind, by=plot_by, center=plot_center,
                             show_points=plot_points, condition_line=plot_cond_line,
                             vaior=plot_vaior, brinley=plot_brinley, brinley_mode=brinley_mode,
                             bounds=_bnd(oc_list[0]), y_percent=y_percent, row_height=row_height,
                             title=f"SCED - dashboard ({oc_list[0]}){label}", save_path=pp, style=style)
            if plot_vaior and uc is not None and not vaior_pooled:
                from functions.sced.plots.vaior import plot_vaior_grid
                _, tbl = plot_vaior_grid(gdf, unit_col=uc, session_col=sc, condition_col=cc,
                                         outcome_col=oc_list[0], reference=vaior_ref,
                                         conditions=conditions, improvement=improvement, save_path=pp, style=style)
                if verbose:
                    for comp, sub_t in tbl.groupby("compared"):
                        npos = int(sub_t["decision_overall"].sum())
                        print(f"  VAIOR {comp} vs {vaior_ref} (per patient){label}: {npos}/{len(sub_t)} positive")
        elif pp:
            plot_sced_alternating(gdf, session_col=sc, condition_col=cc, outcome_col=oc_list[0],
                                  unit_col=uc, title=f"SCED alternating{label}", save_path=pp,
                                  kind=plot_kind, by=plot_by, center=plot_center,
                                  show_points=plot_points, condition_line=plot_cond_line,
                                  bounds=_bnd(oc_list[0]), style=style)
            if plot_vaior or plot_brinley:
                from functions.sced.plots.vaior import plot_vaior as _pv, plot_brinley as _pb, vaior_by_unit
                others = [c for c in conditions if c != vaior_ref]
                if plot_vaior:
                    for comp in others:
                        if uc is not None and not vaior_pooled:
                            vaior_by_unit(gdf, unit_col=uc, session_col=sc, condition_col=cc,
                                          outcome_col=oc_list[0], reference=vaior_ref, compared=comp,
                                          improvement=improvement, save_path=pp, style=style)
                        else:
                            _pv(gdf, session_col=sc, condition_col=cc, outcome_col=oc_list[0],
                                reference=vaior_ref, compared=comp, improvement=improvement,
                                save_path=pp, style=style)
                if plot_brinley:
                    for comp in others:
                        _pb(gdf, condition_col=cc, outcome_col=oc_list[0], cond_x=vaior_ref,
                            cond_y=comp, session_col=sc, unit_col=uc, mode=brinley_mode,
                            bounds=_bnd(oc_list[0]), save_path=pp, style=style)

        # --- STEP 3: analysis (engine chosen automatically) ---
        out = run_sced_alternating(
            gdf, session_col=sc, condition_col=cc,
            outcome=(None if measures else (oc_list if mv else oc_list[0])),
            measures=measures, level3=level3, agg=agg, dispersion=dispersion, trial_col=trial_col,
            within_visit_slope=within_visit_slope, unit_col=uc, conditions=conditions,
            outcome_type=outcome_type, improvement=improvement, max_consecutive=max_consecutive,
            block_size=block_size, time_covariate=time_covariate, perm_method=perm_method,
            standardize=standardize, hierarchical=hierarchical, covariate_cols=covariates,
            trend_test=trend_test, test_time_trend=test_time_trend,
            condition_order=condition_order, dose=dose,
            n_perm=n_perm, save_path=sp, verbose=verbose)
        if verbose:
            print(f"\n== RESULT{label} - engine: {out['engine']} ==")
            if out["model_info"] is not None:
                info = out["model_info"]
                for k in ["Primary inference (test)", "Test statistic", "Detrend",
                          "Permutation method", "Randomization p-value",
                          "Time-trend slope p", "Time-trend scheme", "Ordered-trend slope p",
                          "Ordered-trend slope p (group)", "Heterogeneity - verdict",
                          "Hierarchical - population effect", "Hierarchical - SD random slope",
                          "Hierarchical - ICC"]:
                    if k in info:
                        print(f"  {k}: {info.get(k)}")
            else:
                r = out["result"]
                print(f"  pseudo-F = {r['pseudo_F']}; p = {r['p_value']:.4f}; outcomes = {r['n_outcomes']}")
                print("  follow-ups per outcome:"); print(r["per_outcome"].to_string(index=False))
        return out

    # --- SCED GROUP LOOP: cohorts analysed SEPARATELY ---
    if group_col:
        cohorts = [(g, gdf) for g, gdf in df.groupby(group_col)]
        if verbose:
            print(f"\n=== {len(cohorts)} SCED group(s) '{group_col}': {[g for g, _ in cohorts]} ===")
    else:
        cohorts = [(None, df)]
    results = []
    for gname, gdf in cohorts:
        label = "" if gname is None else f" [{group_col}={gname}]"
        pp = plot_path if (gname is None or not plot_path) else os.path.join(plot_path, str(gname))
        sp = save_path if (gname is None or not save_path) else os.path.join(save_path, str(gname))
        if gname is not None and verbose:
            n = gdf[uc].nunique() if uc else len(gdf)
            print(f"\n################  GROUP {group_col} = {gname}  (n={n})  ################")
        if outcome_mode == "separate" and len(outcomes) >= 2:
            for oc in outcomes:
                if verbose:
                    print(f"\n--------  OUTCOME: {oc}{label}  --------")
                opp = os.path.join(pp, str(oc)) if pp else None
                osp = os.path.join(sp, str(oc)) if sp else None
                results.append(_analyze(gdf, f"{label}[{oc}]", opp, osp, [oc]))
        else:
            results.append(_analyze(gdf, label, pp, sp, outcomes))
    if verbose:
        print("\nHelp:\n" + explain("randomization test", "p-value", "ordered trend", "tau-u"))
    return results
