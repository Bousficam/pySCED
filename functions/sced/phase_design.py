"""
SCED - Phase-design randomization test (AB / ABA / ABAB), randomized change points
==================================================================================

The canonical single-case **phase design** (Onghena 1992; Edgington & Onghena 2007): a
case alternates phases A (baseline) and B (treatment) - AB, ABA, ABAB... - and the
**moments of phase change** are randomised within pre-specified windows (respecting the
phase order and a minimum length per phase). The randomization test re-draws the change
points from the admissible set and recomputes a statistic.

Two statistics:
  - ``"contrast"`` : mean(B phases) - mean(A phases), oriented by ``improvement`` (does
    treatment differ from baseline?), one-sided;
  - ``"omnibus"`` : between-phase sum of squares (does ANY phase differ from the others?),
    one-sided, non-directional - the legitimate omnibus here because the phase moments
    are what is randomised.

Data: long, one row per session for ONE case: ``session | outcome`` (phases are defined
by the change points). For replicated phase designs pass ``unit_col`` - the change
points are drawn per unit and the statistic averaged across units (Edgington combined
test). Admissible change-point sets are enumerated exactly when small, else Monte-Carlo.
"""
import itertools

import numpy as np
import pandas as pd


# TODO(SCED, new family - Changing-Criterion Design): add a dedicated CCD module with the
# two randomization procedures of Manolov & Tanious (2024, Behavior Research Methods):
#   - PCM (Phase Change Moment): randomise the moment each criterion changes within a
#     min phase length; test statistic = MAD (mean absolute deviation to the criterion
#     levels); systematic reference distribution when n <= 36 else Monte-Carlo. Caveat:
#     Type-I-error inflation for triangular (short-long-short) phase-length patterns - 
#     mitigated by including a reversal.
#   - BAC (Blocked Alternating Criterion): randomise the order of the two criteria within
#     each block of two phases (2^B reference distribution; always systematic).
# This is the design of the BCINET pilot arm (MI time demanded as a moving criterion), so
# it is the highest-value gap to fill. R reference code: https://osf.io/dbfxk/.
#
# TODO(SCED, Bayesian): the phase_design_test below estimates change-point effects with a
# *fixed* known knot. Add the Bayesian unknown-change-point models (Aerts 2015;
# Natesan & Hedges 2017 BUCP) that treat the knot as an estimated parameter and report
# HDI + ROPE - they give independent confirmatory evidence of immediacy. Needs a JAGS/PyMC
# backend (not currently a project dependency).
def _phase_labels(n, cuts, labels):
    """Phase label per ordered session given the ordered change-point indices ``cuts``
    (0-based positions where a new phase starts) and the phase ``labels`` sequence."""
    out = np.empty(n, dtype=object)
    bounds = [0] + list(cuts) + [n]
    for k in range(len(labels)):
        out[bounds[k]:bounds[k + 1]] = labels[k]
    return out


def _admissible_cuts(n, n_phases, min_len):
    """All increasing change-point tuples (positions 1..n-1) splitting n sessions into
    ``n_phases`` blocks each of length >= ``min_len``."""
    need = n_phases - 1
    valid = []
    for cuts in itertools.combinations(range(min_len, n - min_len + 1), need):
        bounds = [0] + list(cuts) + [n]
        if all((bounds[k + 1] - bounds[k]) >= min_len for k in range(n_phases)):
            valid.append(cuts)
    return valid


def _series_stat(y, cuts, labels, baseline, sign, statistic):
    """Compute the phase-design statistic for one series given the change-point indices
    ``cuts``, the phase ``labels`` sequence, the ``baseline`` label set, orientation
    ``sign`` (+1/-1) and the requested ``statistic``. Returns a float (NaN if a required
    phase group is empty). ``statistic`` in {contrast (B vs A mean, oriented), itei
    (immediate treatment effect index), omnibus (between-phase sum of squares)}."""
    statistic = {"level": "contrast", "md": "contrast"}.get(statistic, statistic)   # alias (= MBD "level")
    lab = _phase_labels(len(y), cuts, labels)
    base = np.array(list(baseline), dtype=object)
    if statistic == "contrast":
        in_base = np.isin(lab, base)
        a = y[in_base]
        b = y[~in_base]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        return sign * (b.mean() - a.mean()) if len(a) and len(b) else np.nan
    if statistic == "omnibus":
        gm = np.nanmean(y)
        ss = 0.0
        for lv in set(labels):
            g = y[lab == lv]
            g = g[np.isfinite(g)]
            if len(g):
                ss += len(g) * (g.mean() - gm) ** 2
        return ss
    if statistic == "itei":                 # Immediate Treatment Effect Index (Michiels 2018):
        in_base = np.isin(lab, base)        # mean of first 3 B - mean of last 3 A (ORIENTED) at the switch
        trans = next((i for i in range(1, len(lab)) if in_base[i - 1] and not in_base[i]), None)
        if trans is None:
            return np.nan
        a_tail = y[:trans][in_base[:trans]]; a_tail = a_tail[np.isfinite(a_tail)][-3:]
        b_head = y[trans:][~in_base[trans:]]; b_head = b_head[np.isfinite(b_head)][:3]
        return sign * (b_head.mean() - a_tail.mean()) if len(a_tail) and len(b_head) else np.nan
    raise ValueError("statistic must be 'contrast', 'itei' or 'omnibus'")


def _runs(labels):
    """Ordered phase sequence and the change-point positions from per-session labels."""
    seq, cuts = [labels[0]], []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            seq.append(labels[i]); cuts.append(i)
    return seq, tuple(cuts)


def phase_design_test(df, *, session_col, outcome_col, phase_col, unit_col=None,
                      baseline=None, min_len=3, statistic="contrast",
                      improvement="increase", n_perm=5000, enumerate_cap=200000, random_state=0):
    """
    Randomization test for an AB / ABA / ABAB phase design with randomized change points.

    The actual phase sequence and the realised change points are read from ``phase_col``
    (per session); ``baseline`` = labels treated as baseline for the contrast (default =
    the first phase). Each unit's sessions are re-split into the same number of blocks of
    length >= ``min_len`` at every admissible set of change points; the observed statistic
    (at the real change points) is compared to that distribution. ``statistic`` in
    {contrast (B-phases vs A-phases mean, oriented), omnibus (between-phase SS,
    non-directional)}. ``unit_col`` -> replicated phase design (per-unit change points,
    statistic averaged; Edgington combined test).

    Returns ``{observed, p_value, n_eval, exact, statistic, phases, n_units}``.

    References: Onghena (1992); Edgington and Onghena (2007) (phase-design change-point randomization); Michiels and Onghena (2018) (ITEI statistic).
    R equivalent: SCRT (single-case randomization tests) - phase-design change-point test; scan::rand_test - potential equivalent, to test.
    """
    sign = 1.0 if improvement == "increase" else -1.0
    units = [None] if unit_col is None else list(pd.unique(df[unit_col]))
    series, obs_cuts, phase_seq = {}, {}, None
    for u in units:
        g = df if unit_col is None else df[df[unit_col] == u]
        g = g.sort_values(session_col)
        series[u] = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
        seq, cuts = _runs(g[phase_col].to_numpy())
        obs_cuts[u] = cuts
        phase_seq = seq if phase_seq is None else phase_seq
    phases = phase_seq
    if baseline is None:
        baseline = (phases[0],)

    adm = {u: _admissible_cuts(len(series[u]), len(phases), min_len) for u in units}
    for u in units:
        if obs_cuts[u] not in adm[u]:
            adm[u] = sorted(set(adm[u]) | {obs_cuts[u]})    # the realised split is admissible
        if not adm[u]:
            raise ValueError(f"No admissible split (unit={u}).")

    def combined_stat(choice):
        vals = [_series_stat(series[u], choice[u], phases, set(baseline), sign, statistic)
                for u in units]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    obs = combined_stat(obs_cuts)

    total = int(np.prod([len(adm[u]) for u in units]))
    exact = total <= enumerate_cap
    rng = np.random.RandomState(random_state)
    ge = 0
    if exact:
        for combo in itertools.product(*[adm[u] for u in units]):
            if combined_stat(dict(zip(units, combo))) >= obs - 1e-12:
                ge += 1
        p = ge / total
        n_eval = total
    else:
        for _ in range(n_perm):
            choice = {u: adm[u][rng.randint(len(adm[u]))] for u in units}
            if combined_stat(choice) >= obs - 1e-12:
                ge += 1
        p = (1 + ge) / (1 + n_perm)
        n_eval = n_perm

    return {"observed": round(float(obs), 4), "p_value": round(float(p), 4),
            "n_eval": int(n_eval), "exact": bool(exact), "statistic": statistic,
            "phases": list(phases), "n_units": len(units)}


def phase_glossary():
    """Glossary of the phase-design report ('Glossary' sheet)."""
    g = [
        ("Phase design", "AB / ABA / ABAB: baseline (A) and treatment (B) phases in blocks, randomized change moments."),
        ("Randomization test", "The sessions are re-split at every admissible change and the observed statistic is compared. Exact if few splits, otherwise Monte-Carlo."),
        ("statistic = contrast", "mean(B) - mean(A) = MD, oriented (improvement)."),
        ("statistic = itei", "Immediate Treatment Effect Index: |mean of last 3 A - first 3 B|. More powerful than MD under TREND for AB-phase designs (Michiels 2018)."),
        ("statistic = omnibus", "Between-phase variance: does the phase have ANY effect (useful in ABAB)."),
        ("MIN_LEN", "Minimum length of each phase = constraint defining the admissible splits."),
        ("p-value", "p=(1+#>=obs)/(1+B). <0.05 = phase profile beyond the chance of the change moments."),
        ("Tau-U (trend-corr.)", "Rank non-overlap baseline vs treatment, corrected for the baseline trend (Tarlow 2016). Bounded -1..1."),
        ("NAP", "Non-overlap of All Pairs: P(B point > A point). 0.5 = chance, 1 = total separation."),
        ("n_units", "Replicated design (UNIT_COL) = combined test over the units; otherwise 1 case."),
    ]
    return pd.DataFrame(g, columns=["Term", "Definition"])


def report_sced_phase_design(
        df, *, unit_col=None, session_col="session", phase_col="phase",
        outcome_name_col=None, value_col=None, outcomes=("y",),
        case_insensitive=True, date_col=None, date_dayfirst=False, group_col=None,
        phases=("A", "B"), baseline=None, min_len=3, statistic="contrast",
        improvement="increase", n_perm=5000, save_path=None, output_dir=None, verbose=True):
    """REPORT-LEVEL phase-design pipeline: loops over cohorts x outcomes (randomization
    test on the change moments + Tau-U/NAP), then writes the consolidated STYLED summary
    ``phase_design_summary.xlsx`` (if ``save_path``). Returns the summary DataFrame (or
    None). All the logic lives here; the template only calls it.
    ``output_dir`` = single root (standard layout built into the pipeline):
    xlsx -> ``Analyse/Inferentielle/``.

    References: Onghena (1992); Edgington and Onghena (2007) (phase-design change-point randomization); Tarlow (2016) (baseline-corrected Tau-U); Parker and Vannest (2009) (NAP).
    R equivalent: SCRT / scan::rand_test - potential equivalent, to test; scan::tau_u and SingleCaseES for the effect sizes.
    """
    import os
    from functions.sced.glossary import std_layout          # PER-OUTCOME tree: <oc>/Analyse/phase_design/
    from functions.sced.core import tau_u, nap
    sc, pc, uc = session_col, phase_col, unit_col
    on, vc = outcome_name_col, value_col
    baseline = baseline if baseline is not None else phases[0]

    # --- prep: case, dates -> sessions, tidy ---
    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        r = resolve_columns(df, {"tier": uc, "session": sc, "phase": pc, "date": date_col,
                                 "group": group_col, "outcome_name": on, "value": vc})
        uc, sc, pc, date_col, group_col, on, vc = (r["tier"], r["session"], r["phase"], r["date"],
                                                   r["group"], r["outcome_name"], r["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=phases)
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=(uc or None), new_col=sc, dayfirst=date_dayfirst)
        if verbose:
            print(f"STEP 0 - Dates '{date_col}' -> session 1..n in '{sc}'")
    outcomes = list(outcomes) if outcomes else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        ids = [c for c in (uc, sc, pc) if c]
        keep = [group_col] if group_col else []
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=ids, keep_cols=keep)
        outcomes = outcomes or names
        if verbose:
            print(f"STEP 0 - tidy: outcomes = {outcomes}")
    if not outcomes:
        if verbose:
            print("No outcome.")
        return None

    cohorts = ([(g, gd) for g, gd in df.groupby(group_col)] if group_col else [(None, df)])
    if group_col and len(cohorts) > 1:
        cohorts = cohorts + [("(all)", df)]

    rows = []
    if verbose:
        print(f"\nPHASE DESIGN: statistic={statistic} | {n_perm} permutations\n")
    for gname, gd in cohorts:
        gtag = "(all)" if (gname is None or gname == "(all)") else str(gname)
        for oc in outcomes:
            d = gd.dropna(subset=[oc])
            if d.empty:
                continue
            try:
                res = phase_design_test(d, session_col=sc, outcome_col=oc, phase_col=pc, unit_col=uc,
                                        baseline=baseline, min_len=min_len, statistic=statistic,
                                        improvement=improvement, n_perm=n_perm)
            except Exception as e:
                if verbose:
                    print(f"  [skipped] {gtag}/{oc}: {type(e).__name__}: {e}")
                continue
            # non-overlap effect sizes: baseline (all A phases) vs treatment (the rest)
            phs = d[pc].astype(str)
            a = pd.to_numeric(d[oc][phs == str(baseline)], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(d[oc][phs != str(baseline)], errors="coerce").dropna().to_numpy()
            tu = tau_u(a, b, improvement=improvement) if (len(a) and len(b)) else {}
            np_ = nap(a, b, improvement=improvement) if (len(a) and len(b)) else np.nan
            p = res["p_value"]
            verdict = ("Phase effect demonstrated (p<0.05)" if (p is not None and p < 0.05)
                       else "No phase effect demonstrated")
            if verbose:
                print(f"  [{gtag}/{oc}] phases={res['phases']} stat={res['statistic']} "
                      f"obs={res['observed']:.3f} p={p} | Tau-U={tu.get('tau_u', float('nan')):.2f} "
                      f"NAP={np_:.2f} ({res['n_eval']} splits{'' if res['exact'] else ', MC'})")
            rows.append({"Group": gtag, "Outcome": oc, "Phases": "".join(map(str, res["phases"])),
                         "Statistic": res["statistic"], "Observed": round(float(res["observed"]), 3),
                         "p-value": p, "Splits": res["n_eval"],
                         "Exact": res["exact"], "n_units": res["n_units"],
                         "Tau-U (trend-corr.)": (round(tu.get("tau_u"), 3) if tu else np.nan),
                         "Tau-U p": (round(tu.get("p_value"), 4) if tu else np.nan),
                         "NAP": (round(float(np_), 3) if np.isfinite(np_) else np.nan),
                         "Verdict": verdict})

    if not rows:
        if verbose:
            print("\nNo analysis.")
        return None
    summary = pd.DataFrame(rows)
    if verbose:
        print("\n================  SUMMARY  ================")
        print(summary.to_string(index=False))
        print("\n-> p<0.05: the phase profile exceeds what would be obtained by randomly moving the "
              "change moments. Tau-U/NAP = magnitude of the non-overlap baseline vs treatment.")

    if save_path or output_dir:
        from functions.sced.glossary import (interpretation_glossary, ascii_sanitize_df,
                                             dedup_pooled_rows, data_recap_df, style_cells, cohen_fill,
                                             slug_token)
        S = ascii_sanitize_df

        def _p_fill(p):                             # significant -> green+bold; otherwise gray
            if isinstance(p, str) and p.strip().startswith("<"):
                return (True, "C6EFCE")
            try:
                return (True, "C6EFCE") if float(p) < 0.05 else (False, "F2F2F2")
            except (TypeError, ValueError):
                return (False, None)

        def _nap_fill(v):                          # NAP: <0.66 gray / <0.93 amber / >=0.93 green
            try:
                a = float(v)
            except (TypeError, ValueError):
                return (False, None)
            return (False, "F2F2F2") if a < 0.66 else ((True, "FFE699") if a < 0.93 else (True, "C6EFCE"))

        recap_rules = {"p-value": _p_fill, "Tau-U p": _p_fill, "NAP": _nap_fill,
                       "Tau-U (trend-corr.)": lambda v: cohen_fill(v, bands=(0.20, 0.60, 0.80))}

        def _write_book(out_path, ocs):
            sub = summary[summary["Outcome"].isin(ocs)] if "Outcome" in summary.columns else summary
            if sub.empty:
                return
            recap = data_recap_df(df, tier_col=uc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                                  group_col=group_col, baseline=baseline,
                                  design=f"Phase design (randomization test, statistic={statistic})",
                                  extra=[("Min phase length (MIN_LEN)", min_len), ("Permutations", n_perm)])
            recapitulatif = dedup_pooled_rows(sub, "Outcome")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            try:
                with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                    S(recap).to_excel(w, sheet_name="Data & design", index=False)
                    S(recapitulatif).to_excel(w, sheet_name="Summary", index=False)
                    style_cells(w.sheets["Summary"], recapitulatif, recap_rules)
                    S(phase_glossary()).to_excel(w, sheet_name="Glossary", index=False)
                    S(interpretation_glossary("mbd")).to_excel(w, sheet_name="Interpretation guide", index=False)
                if verbose:
                    print(f"\nSummary written: {out_path}")
            except Exception as e:
                if verbose:
                    print(f"\n[not written] {type(e).__name__}: {e}")

        if output_dir is not None:
            for oc in outcomes:
                ad = std_layout(output_dir, oc, "phase_design")["analyse"]
                _write_book(os.path.join(ad, "phase_design_summary.xlsx"), [oc])
        else:
            os.makedirs(save_path, exist_ok=True)
            _otag = slug_token(outcomes[0]) if len(outcomes) == 1 else "multi"
            _write_book(os.path.join(save_path, f"phase_design_{_otag}_summary.xlsx"), list(outcomes))
    return summary
