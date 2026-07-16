"""
Consolidated report - SCED multiple baseline design
===================================================

Orchestration layer above the ``pipeline_sced_multiple_baseline`` engine:
loops over (cohort x outcome), design-comparable effect sizes (BC-SMD via scdhlm),
multilevel model, per-tier detail (individual p, trend-corrected Tau-U),
figures (dashboard, VAIOR, pooled), then WRITES the consolidated, styled
``permutation_test_summary.xlsx`` (5 sheets: Permutation [design + scheme
+ combinations + test result], Multilevel [ML + BC-SMD], By tier,
Glossary, References).

The template only: loads the data + calls
``report_sced_multiple_baseline``. All the report logic lives here (versioned,
reusable), not in the template.
"""
import os

import numpy as np
import pandas as pd

from functions.sced.mbd.baseline import pipeline_sced_multiple_baseline, _fmt_p


def _correction_label(variant):
    """Short label of the baseline-trend correction applied per tier:
    None (none) / Tarlow (Theil-Sen, n_A>=7) / Brossard (bounded denominator, n_A<7)."""
    v = (variant or "").lower()
    if "tarlow" in v or "adj" in v:
        return "Tarlow"
    if "brossard" in v or "brossart" in v or "trend a" in v or "trend_a" in v:
        return "Brossard"
    return "None"


def mbd_glossary():
    """Glossary (DataFrame Term/Definition) explaining the columns and concepts of the
    MBD report - written as the \"Glossary\" sheet. Citations in short form
    (first author et al., year); full bibliography + DOI in the \"References\" sheet."""
    g = [
        (" - Identification - ", ""),
        ("Group", "Analysed cohort (e.g. acute/chronic profile). \"(all)\" = all tiers together (more power)."),
        ("Outcome", "Response variable analysed (each outcome is treated SEPARATELY)."),
        ("Tiers (n)", "Number of tiers (patients) that actually have data for this outcome."),
        ("Starts", "Intervention introduction session (phase B) specific to each tier (staggered)."),
        ("Admissible starts (randomization)", "Possible start-points: sessions where the intervention COULD have been introduced per tier (baseline window). Defines the randomization space of the permutation test."),
        (" - \"Permutation\" sheet: randomization test - ", ""),
        ("Multiple baseline (MBD)", "Design where the intervention is introduced at STAGGERED moments across tiers. Effect demonstrated if each tier changes AT ITS OWN moment while the others remain stable."),
        ("Permutation test", "Primary inference: the observed effect is compared with those obtained by re-assigning the introduction MOMENTS according to the actually possible draw. Exact if the starts were randomized; assumes no distribution (Edgington & Onghena, 2007; Levin et al., 2016)."),
        ("Procedure (scheme)", "Randomization scheme (Levin et al., 2016): WW = case permutation; MB = start-point with replacement; MB-R = without replacement; KL = Koehler-Levin; Rev/Rev-M = Revusky (stepwise)."),
        ("Statistic", "Quantity tested by the randomization. STANDARD: level = A->B level jump (mean difference, MD). Optional: slope = slope change; combined = both (ITS, non-directional)."),
        ("Candidate starts / tier", "Number of admissible introduction moments for THIS tier (baseline window). Determines the resolution of the test."),
        ("Number of combinations (group)", "Size of the randomization space = product of the candidate starts across tiers (with-replacement scheme, prod k_i; Marascuilo & Busk, 1988)."),
        ("Enumeration", "EXACT (all combinations enumerated, or exact p by convolution) or MONTE-CARLO (B random draws) depending on the size of the space."),
        ("p-value", "Randomization p: proportion of assignments as/more extreme than the observed one - exact if enumerated, otherwise p=(1+#)/(1+B). <0.05 = effect concordant with the introduction, not with time."),
        ("Min attainable p (group)", "Smallest possible test p = 1/(number of combinations) when exact (otherwise 1/(1+B)). If > 0.05, NO result can be significant - a design limitation, not the effect's."),
        ("Min attainable p (indiv.)", "Same per patient = 1/(tier candidate starts). Often ~0.1-0.2 on a short baseline -> a non-significant (indiv.) p may just reflect this floor."),
        ("Verdict", "Automatic reading: staggered effect demonstrated (p<0.05) or not."),
        (" - \"Multilevel\" sheet: ML + BC-SMD (MODEL inference) - ", ""),
        ("ML (piecewise multilevel)", "Mixed-effects regression fitting ALL patient series together (partial pooling). Gives quantified effect sizes independently of the randomization (Moeyaert et al., 2014)."),
        ("ML jump b2", "IMMEDIATE A->B LEVEL change, NET of time. Main effect size (outcome unit)."),
        ("ML slope b3", "SLOPE change after the introduction (does the intervention accelerate progression?)."),
        ("ML ICC", "Share of BETWEEN-patient variance (0..1). High = patients very different in level (Koo & Li, 2016: <0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent)."),
        ("ML p b1/b2/b3 (KR)", "Multilevel model p: fixed effects trend b1 / jump b2 / slope b3, Kenward-Roger df (R lmerTest; Moeyaert et al., 2014). Distinct from the randomization p. Empty if R/lmerTest unavailable."),
        ("Rando vs ML", "Both are read together: rando = \"due to the luck of the draw?\" (exact, low power); ML = \"by how much, and homogeneous?\" (b2 + ICC)."),
        ("d (BC-SMD)", "Between-case SMD g_AB: jump standardized by the BETWEEN-case standard deviation -> comparable to an RCT Cohen's d, meta-analysable (Pustejovsky et al., 2014). Computed by the R package scdhlm (REML, corAR1, Satterthwaite df; Valentine et al., 2016). Cohen benchmarks: 0.2/0.5/0.8."),
        ("BC-SMD CI95", "95% confidence interval of g_AB (scdhlm). Wide/unstable with few cases."),
        ("d (BC-SMD trend-corr.)", "Same g_AB under lme(y ~ time + phase): level effect NET of a common linear baseline trend. A g that collapses after correction = effect largely carried by the trend (maturation/recovery), not by the treatment."),
        ("BC-SMD trend CI95", "CI95 of the trend-corrected g_AB (scdhlm)."),
        (" - \"By tier\" sheet - ", ""),
        ("n_A / n_B", "Number of observations in phase A (baseline) and B (intervention) for this tier."),
        ("mean_A / mean_B / diff", "Phase means and their difference (mean_B - mean_A), in the outcome unit."),
        ("NAP", "Non-overlap of All Pairs: P(B point > A point), 0.5=chance, 1=total separation (Parker & Vannest, 2009)."),
        ("NAP CI95", "95% confidence interval of NAP (Newcombe method on the AUC)."),
        ("Hedges g (within)", "WITHIN-case SMD (M_B-M_A)/pooled within SD, J-corrected. Single-case descriptive; standardized by a within-case SD -> often INFLATED and NOT comparable across cases (prefer the BC-SMD for comparison)."),
        ("Tau-U raw", "A-vs-B Tau-U UNCORRECTED (= 2*NAP-1). Confounded with the baseline trend when it exists (Parker et al., 2011)."),
        ("Baseline trend (Tau-A)", "MONOTONIC baseline trend (Kendall on phase A) in [-1,+1]; large |Tau-A| = strong drift before treatment."),
        ("Correction", "Baseline-trend correction actually applied: None (no trend detected); Tarlow (Theil-Sen, n_A>=7; Tarlow, 2017); Brossard (bounded denominator, n_A<7; Brossart et al., 2018). Triggered if a trend is present (Fingerhut et al., 2021 rule: p<0.05 OR |Tau-A|>=0.40)."),
        ("Tau-U trend-corr.", "FINAL Tau-U, corrected for the baseline trend if a correction was applied, otherwise = raw."),
        ("CI95 trend-corr.", "95% confidence interval of the final Tau-U."),
        ("Tau-U p", "OWN significance of the per-tier Tau-U (two-sided test based on the variance of Kendall's S; Parker et al., 2011), vs its critical Tau-U. NOT to be confused with the design randomization test p."),
        ("Perm. stat (indiv.)", "Observed statistic of the SINGLE-CASE randomization test (tier A->B contrast) compared with its permutation distribution."),
        ("p (indiv.)", "Associated single-case randomization p. DESCRIPTIVE (spot responders); min ~ 1/(number of admissible cut points) -> weakly powered alone. Do NOT combine (the omnibus is the MBD test)."),
        ("autocorr lag-1", "Lag-1 autocorrelation of the within-phase residuals - diagnostic of serial dependence."),
        (" - Appraisal benchmarks (magnitude) - ", ""),
        ("NAP (magnitude)", "0.50 = chance. Parker & Vannest (2009) benchmarks: 0.50-0.65 weak; 0.66-0.92 medium; 0.93-1.0 strong."),
        ("Tau-U (magnitude)", "0 = no effect. Vannest & Ninci (2015) benchmarks: |Tau| <0.20 negligible; 0.20-0.60 moderate; 0.60-0.80 large; >0.80 very large."),
        ("Tau-A (appraising the trend)", "NOT an effect: the baseline drift. |Tau-A| <0.20 flat; 0.20-0.40 slight; >=0.40 SUBSTANTIAL (correction threshold). Read the SIGN vs the effect direction: same direction = confounds the effect (spontaneous recovery); opposite direction = reinforces it."),
        ("BC-SMD (magnitude)", "Cohen benchmarks: 0.2 small; 0.5 medium; 0.8 large. Compare raw vs trend-corrected (a g that collapses = carried by the trend)."),
        ("diff (magnitude)", "Mean difference in POINTS (original scale): no universal benchmark -> judge vs the outcome's minimal clinically important difference (MCID)."),
        ("Short-baseline caveat", "On 4-7 points, Tau-U / NAP / Tau-A are discrete and noisy -> INDICATIVE benchmarks, underpowered p; the primary inference remains the randomization test."),
        (" - Visual aid - ", ""),
        ("VAIOR", "Manolov & Vannest (2019): baseline trend (Theil-Sen) +/- projected MAD; B points classed green (outside band) / yellow (outside trend) / red (neither). Visual aid, NOT a test."),
        ("Staggered figure", "One panel per tier, vertical line at the STAGGERED introduction moment, A/B means. Standard SCED visual reading."),
    ]
    return pd.DataFrame(g, columns=["Term", "Definition"])


def mbd_references():
    """Bibliography of the methods cited in the MBD report (\"References\" sheet).
    DataFrame Reference (APA) / DOI; DOIs verified against the sources wiki."""
    refs = [
        ("Brossart, D. F., Laird, V. C., & Armstrong, T. W. (2018). Interpreting Kendall's Tau and Tau-U for single-case experimental designs. Cogent Psychology, 5(1), 1518358.",
         "10.1080/23311908.2018.1518358"),
        ("Edgington, E. S., & Onghena, P. (2007). Randomization tests (4th ed.). Chapman & Hall/CRC.",
         " - "),
        ("Fingerhut, J., Xu, X., & Moeyaert, M. (2021). Selecting the proper Tau-U measure for single-case experimental designs: Development and application of a decision flowchart. Evidence-Based Communication Assessment and Intervention.",
         "10.1080/17489539.2021.1937851"),
        ("Hedges, L. V., Pustejovsky, J. E., & Shadish, W. R. (2012). A standardized mean difference effect size for single case designs. Research Synthesis Methods, 3(3), 224-239.",
         "10.1002/jrsm.1052"),
        ("Hedges, L. V., Pustejovsky, J. E., & Shadish, W. R. (2013). A standardized mean difference effect size for multiple baseline designs across individuals. Research Synthesis Methods, 4(4), 324-341.",
         "10.1002/jrsm.1086"),
        ("Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine, 15(2), 155-163.",
         "10.1016/j.jcm.2016.02.012"),
        ("Levin, J. R., Ferron, J. M., & Gafurov, B. S. (2016). Comparison of randomization-test procedures for single-case multiple-baseline designs. Developmental Neurorehabilitation.",
         "10.1080/17518423.2016.1197708"),
        ("Manolov, R., & Vannest, K. J. (2019). A visual aid and objective rule encompassing the data features of visual analysis. Behavior Modification, 47(5), 1259-1284.",
         "10.1177/0145445519854323"),
        ("Marascuilo, L. A., & Busk, P. L. (1988). Combining statistics for multiple-baseline AB and replicated ABAB designs across subjects. Behavioral Assessment, 10, 1-28.",
         " - "),
        ("Moeyaert, M., Ugille, M., Ferron, J. M., Beretvas, S. N., & Van den Noortgate, W. (2014). Three-level analysis of single-case experimental data: Empirical validation. Journal of School Psychology, 52(2), 191-211.",
         "10.1016/j.jsp.2013.11.003"),
        ("Parker, R. I., & Vannest, K. J. (2009). An improved effect size for single-case research: Nonoverlap of all pairs. Behavior Therapy, 40(4), 357-367.",
         "10.1016/j.beth.2008.10.004"),
        ("Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011). Combining nonoverlap and trend for single-case research: Tau-U. Behavior Therapy, 42(2), 284-299.",
         "10.1016/j.beth.2010.08.006"),
        ("Pustejovsky, J. E., Hedges, L. V., & Shadish, W. R. (2014). Design-comparable effect sizes in multiple baseline designs: A general modeling framework. Journal of Educational and Behavioral Statistics, 39(5), 368-393.",
         "10.3102/1076998614547577"),
        ("Tarlow, K. R. (2017). An improved rank correlation effect size statistic for single-case designs: Baseline corrected Tau. Behavior Modification, 41(4), 427-467.",
         "10.1177/0145445516676750"),
        ("Valentine, J. C., Tanner-Smith, E. E., Pustejovsky, J. E., & Lau, T. S. (2016). Between-case standardized mean difference effect sizes for single-case designs. The Campbell Collaboration.",
         "10.4073/cmdp.2016.1"),
        ("Vannest, K. J., & Ninci, J. (2015). Evaluating intervention effects in single-case research designs. Journal of Counseling & Development, 93(4), 403-411.",
         "10.1002/jcad.12038"),
    ]
    return pd.DataFrame(refs, columns=["Reference", "DOI"])


def report_sced_multiple_baseline(
        df, *, tier_col="tier", session_col="session", phase_col="phase",
        outcome_name_col=None, value_col=None, outcomes=("accuracy",),
        case_insensitive=True, date_col=None, date_dayfirst=False,
        group_col=None, include_pooled=True,
        phases=("A", "B"), baseline_phase=None, treatment_phase=None,
        improvement="increase", statistic="level",
        case_randomization=False, start_point_randomization=True,
        replacement=False, comparison="within",
        start_window=None, baseline_window=None, windows=None,
        intervention_starts=None, n_perm=5000,
        save_path=None, plot_path=".", pooled_plot_path=None, pooled_title=None, style=None,
        multilevel=True, treatment_trajectory="continuous", ramp=3,
        model_selection=False, bc_smd=True, bc_smd_trend=True,
        plot_vaior=True, vaior_pooled=False, plot_panels=True,
        brinley_mode="classic", bounds=None,
        bayesian=False, bayes_hypothesis="cumulative_trend", bayes_family="gaussian",
        bayes_bounds=None, bayes_draws=1000, bayes_tune=1000, bayes_chains=2,
        mcid=None, output_dir=None, impute_col=None, verbose=True):
    """End-to-end MBD pipeline AT REPORT LEVEL: loops over cohorts x outcomes,
    effect sizes, figures, then writes the consolidated styled summary
    ``MBD_summary.xlsx`` (if ``save_path``). Returns the summary DataFrame (or None).

    LONG format expected: one row per (tier, session) - [tier, session, phase, outcome].
    See the glossary (\"Glossary & interpretation\" sheet) for each column.

    ``output_dir`` = single root (standard layout built into the pipeline), ORGANIZED PER
    OUTCOME: <output_dir>/<outcome>/Analyse/permutation_test/ (xlsx); <outcome>/Plot/
    (descriptive); <outcome>/Plot/poole/ (model-dependent pooled). ``save_path``/``plot_path``
    remain the advanced mode (single folder, consolidated multi-outcome report).

    References: Levin, Ferron & Gafurov 2016 (randomization test); Moeyaert, Ferron, Beretvas &
    Van den Noortgate 2014 (multilevel); Pustejovsky, Hedges & Shadish 2014 (BC-SMD).
    R equivalent: scan::rand_test + scan::hplm + scdhlm::g_mlm - potential equivalent, to test.
    """
    from functions.sced.glossary import std_layout
    _ANALYSIS = "permutation_test"                          # name of the analysis sub-folder (formerly "MBD")
    tc, sc, pc = tier_col, session_col, phase_col
    on, vc = outcome_name_col, value_col
    baseline = baseline_phase if baseline_phase is not None else phases[0]    # VAIOR reference
    treatment = treatment_phase if treatment_phase is not None else phases[-1]

    # --- STEP 0a: case-insensitivity (columns + phases) ---
    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        res = resolve_columns(df, {"tier": tc, "session": sc, "phase": pc, "date": date_col,
                                   "group": group_col, "outcome_name": on, "value": vc})
        tc, sc, pc, date_col, group_col, on, vc = (res["tier"], res["session"], res["phase"],
                                                   res["date"], res["group"], res["outcome_name"],
                                                   res["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=phases)

    # --- imputation column resolved CASE-INSENSITIVELY (None if absent) ---
    if impute_col:
        impute_col = next((c for c in df.columns
                           if str(c).strip().lower() == str(impute_col).strip().lower()), None)

    # --- STEP 0b: dates -> ordered session 1..n (per tier) ---
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=tc, new_col=sc, dayfirst=date_dayfirst)
        if verbose:
            print(f"STEP 0 - Dates '{date_col}' -> ordered session 1..n (per tier) in '{sc}'")

    # --- STEP 0c: TIDY format -> 1 column per outcome ---
    outcomes = list(outcomes) if outcomes else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        keep = [group_col] if group_col else []
        imp_tidy = (df[[tc, sc, pc, on, impute_col]].copy()       # imputation flag BEFORE unpivoting
                    if (impute_col and impute_col in df.columns) else None)
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=[tc, sc, pc],
                                     keep_cols=keep)
        if imp_tidy is not None:                       # re-attach a PER-outcome flag (_imp_<outcome>)
            for nm in names:
                sub = (imp_tidy[imp_tidy[on] == nm][[tc, sc, pc, impute_col]]
                       .rename(columns={impute_col: f"_imp_{nm}"}))
                df = df.merge(sub, on=[tc, sc, pc], how="left")
                df[f"_imp_{nm}"] = df[f"_imp_{nm}"].fillna(False).astype(bool)
        missing = [o for o in outcomes if o not in names]
        if outcomes and missing:
            if verbose:
                print(f"  Warning: OUTCOMES {missing} absent from the stacked outcomes {names} -> analysing ALL.")
            outcomes = names
        elif not outcomes:
            outcomes = names
        if verbose:
            print(f"STEP 0 - Tidy format unpivoted: outcomes = {outcomes}")
    if not outcomes:
        if verbose:
            print("No outcome to analyse (set outcomes or outcome_name_col/value_col).")
        return None

    summary_rows, pertier_rows, rando_rows = [], [], []   # summary + randomization space (1st sheet)

    def _mean(per, col):
        return round(float(pd.to_numeric(per[col], errors="coerce").mean()), 3) if col in per else None

    def _analyze(gdf, gtag, label, outcome, pp, ppp=None):
        """Complete MBD analysis of one (cohort x outcome): randomization test + figure + VAIOR."""
        win = windows
        if start_window is not None and baseline_window is not None:
            raise ValueError("Set EITHER start_window (start), OR baseline_window (number of A points), not both.")
        start_win = start_window
        if baseline_window is not None:                       # number of A points -> start = n_A + 1
            start_win = (int(baseline_window[0]) + 1, int(baseline_window[1]) + 1)
        if start_win is not None:                             # start window -> {tier:[lo..hi]}
            lo, hi = start_win
            win = {t: list(range(int(lo), int(hi) + 1)) for t in gdf[tc].dropna().unique()}
        bfam = bayes_family.get(outcome, "gaussian") if isinstance(bayes_family, dict) else bayes_family
        bbnd = bayes_bounds.get(outcome) if isinstance(bayes_bounds, dict) else bayes_bounds
        _imp = None                                           # imputation flag column for THIS outcome
        if impute_col:
            _imp = (f"_imp_{outcome}" if f"_imp_{outcome}" in gdf.columns
                    else (impute_col if impute_col in gdf.columns else None))
        try:
            info, result = pipeline_sced_multiple_baseline(
                gdf, tier_col=tc, session_col=sc, outcome_col=outcome, phase_col=pc,
                baseline_phase=baseline_phase, treatment_phase=treatment_phase,
                intervention_starts=intervention_starts, windows=win,
                case_randomization=case_randomization,
                start_point_randomization=start_point_randomization, replacement=replacement,
                comparison=comparison, statistic=statistic, multilevel=multilevel,
                treatment_trajectory=treatment_trajectory, ramp=ramp, model_selection=model_selection,
                improvement=improvement, n_perm=n_perm, plot_path=pp, pooled_plot_path=ppp,
                pooled_title=pooled_title, style=style, impute_col=_imp,
                bayesian=bayesian, bayes_hypothesis=bayes_hypothesis, bayes_family=bfam,
                bayes_bounds=bbnd, bayes_draws=bayes_draws, bayes_tune=bayes_tune,
                bayes_chains=bayes_chains, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [skipped]{label}[{outcome}]: {type(e).__name__}: {e}")
            return None

        if verbose:
            print(f"\n== MULTIPLE BASELINE{label} - outcome: {outcome} ==")
            for k in ["Tiers", "Contrast", "Intervention starts", "Procedure", "Statistic",
                      "Randomisation", "Observed statistic", "p-value", "Verdict"]:
                print(f"  {k}: {info.get(k)}")
        per = result["per_tier"]
        cols = [c for c in ["Tier", "Start", "mean_A", "mean_B", "diff", "Tau-U", "Hedges g (within)", "NAP"]
                if c in per.columns]
        if verbose:
            print("  -- by tier --"); print(per[cols].to_string(index=False))

        # --- BC-SMD (design-comparable effect size, single-study meta-analytic) ---
        # BC-SMD design-comparable (g_AB) via the R package scdhlm - CANONICAL REML estimator.
        bc = {"g": None, "ci_low": None, "ci_high": None}
        bc_t = {"g": None, "ci_low": None, "ci_high": None}
        if bc_smd:
            from functions.sced.mbd.multilevel import bc_smd_scdhlm
            r_bc = bc_smd_scdhlm(gdf, tier_col=tc, session_col=sc, outcome_col=outcome, phase_col=pc,
                                 baseline=baseline_phase, treatment=treatment_phase)
            if r_bc.get("available"):
                bc = {"g": round(r_bc["g_AB"], 3), "ci_low": round(r_bc["ci_low"], 3),
                      "ci_high": round(r_bc["ci_high"], 3)}
            elif verbose:
                print(f"  [BC-SMD] {r_bc.get('message')}")     # R/scdhlm absent -> not computed
            if bc_smd_trend:                                   # BC-SMD corrected for the baseline trend (y~time+phase)
                r_bt = bc_smd_scdhlm(gdf, tier_col=tc, session_col=sc, outcome_col=outcome, phase_col=pc,
                                     baseline=baseline_phase, treatment=treatment_phase, trend=True)
                if r_bt.get("available"):
                    bc_t = {"g": round(r_bt["g_AB"], 3), "ci_low": round(r_bt["ci_low"], 3),
                            "ci_high": round(r_bt["ci_high"], 3)}

        # --- collection for the consolidated summary ---
        gname_disp = "(all)" if gtag is None else str(gtag)
        # --- randomization space (conditions + number of combinations + min attainable p) ---
        mpi = info.get("Rando - min attainable p (indiv.)") or {}
        ncand = info.get("Rando - candidate starts / tier") or {}
        mpg = info.get("Rando - min attainable p (group)")
        _fmtp = lambda x: (None if x is None else (f"{x:.1e}" if x < 1e-3 else round(float(x), 4)))
        rando_rows.append({
            "Group": gname_disp, "Outcome": outcome,
            "Procedure": info.get("Procedure"),
            "Conditions": info.get("Randomisation features"),
            "Statistic": info.get("Statistic"),
            "Candidate starts / tier": ", ".join(f"{k}:{v}" for k, v in ncand.items()) or None,
            "Combinations (group)": info.get("Rando - combinations (group)"),
            "Enumeration": info.get("Randomisation"),
            "Min attainable p (group)": _fmtp(mpg),
            "Min attainable p (indiv.)": (f"{min(mpi.values()):.3f}-{max(mpi.values()):.3f}"
                                          if mpi else None)})
        ml_kr = info.get("Multilevel canonical - p (b1/b2/b3)")   # Kenward-Roger p (R lmerTest)
        summary_rows.append({
            "Group": gname_disp, "Outcome": outcome,
            "Tiers (n)": int(gdf.dropna(subset=[outcome])[tc].nunique()),
            "Procedure": info.get("Procedure"),
            "Statistic": info.get("Statistic"), "Observed stat": info.get("Observed statistic"),
            "p-value": _fmt_p(info.get("p-value")), "Verdict": info.get("Verdict"),
            # MULTILEVEL effect size (does not depend on the randomization)
            "ML jump b2": info.get("Multilevel - level change b2"),
            "ML slope b3": info.get("Multilevel - slope change b3"),
            "ML ICC": info.get("Multilevel - ICC"),
            # CANONICAL p (Moeyaert): fixed-effect test, Kenward-Roger df (via R lmerTest)
            "ML p b1 (KR)": (_fmt_p(ml_kr[0]) if isinstance(ml_kr, (list, tuple)) else None),
            "ML p b2 (KR)": (_fmt_p(ml_kr[1]) if isinstance(ml_kr, (list, tuple)) and len(ml_kr) > 1 else None),
            "ML p b3 (KR)": (_fmt_p(ml_kr[2]) if isinstance(ml_kr, (list, tuple)) and len(ml_kr) > 2 else None),
            # BC-SMD (design-comparable, comparable to an RCT d) - raw + trend-corrected
            "d (BC-SMD)": bc.get("g"),
            "BC-SMD CI95": (f"[{bc['ci_low']};{bc['ci_high']}]"
                            if bc.get("ci_low") is not None and pd.notna(bc.get("ci_low")) else None),
            "d (BC-SMD trend-corr.)": bc_t.get("g"),
            "BC-SMD trend CI95": (f"[{bc_t['ci_low']};{bc_t['ci_high']}]"
                                  if bc_t.get("ci_low") is not None and pd.notna(bc_t.get("ci_low")) else None),
            "Starts": info.get("Intervention starts")})

        # --- per-tier detail: + individual p (single-case) + Tau-U CI95 ---
        from functions.sced.phase_design import phase_design_test
        from functions.sced.core import tau_u
        for _, r in per.iterrows():
            row = {"Group": gname_disp, "Outcome": outcome, **{c: r[c] for c in cols}}
            sub = gdf[gdf[tc].astype(str) == str(r["Tier"])]   # robust match (tier IDs int OR str)
            _pmi = mpi.get(str(r["Tier"]))
            row["Min attainable p (indiv.)"] = round(_pmi, 3) if _pmi is not None else None   # 1/number of tier starts
            try:                                              # single-case rando: observed stat + p
                pr = phase_design_test(sub, session_col=sc, outcome_col=outcome, phase_col=pc,
                                       baseline=baseline, improvement=improvement,
                                       statistic="contrast")
                row["Perm. stat (indiv.)"] = round(float(pr["observed"]), 3)
                row["p (indiv.)"] = _fmt_p(pr["p_value"])
            except Exception:
                row["Perm. stat (indiv.)"] = None
                row["p (indiv.)"] = None
            try:        # RAW Tau-U (level) vs BASELINE-CORRECTED (net of trend, Tarlow 2016)
                a = pd.to_numeric(sub[sub[pc].astype(str) == str(baseline)][outcome],
                                  errors="coerce").dropna().to_numpy()
                b = pd.to_numeric(sub[sub[pc].astype(str) == str(treatment)][outcome],
                                  errors="coerce").dropna().to_numpy()
                # per-phase descriptives (n + SD) + NAP CI95 (Newcombe) - reliability/context
                row["n_A"] = int(len(a)); row["n_B"] = int(len(b))
                row["SD_A"] = round(float(np.std(a, ddof=1)), 2) if len(a) > 1 else None
                row["SD_B"] = round(float(np.std(b, ddof=1)), 2) if len(b) > 1 else None
                # lag-1 autocorrelation of residuals (y - phase mean) - SCED diagnostic (#12)
                _sub2 = sub.sort_values(sc)
                _y = pd.to_numeric(_sub2[outcome], errors="coerce").to_numpy(dtype=float)
                _ph = _sub2[pc].astype(str).to_numpy()
                for _lv in set(_ph[np.isfinite(_y)]):
                    _msk = (_ph == _lv) & np.isfinite(_y)
                    _y[_msk] = _y[_msk] - np.nanmean(_y[_msk])
                _fin = np.isfinite(_y)
                if int(_fin.sum()) > 3:
                    _r1 = np.corrcoef(_y[_fin][:-1], _y[_fin][1:])[0, 1]
                    row["autocorr lag-1"] = round(float(_r1), 2) if np.isfinite(_r1) else None
                from functions.sced.core import nap_ci
                _nci = nap_ci(a, b, improvement=improvement)
                if np.isfinite(_nci.get("ci_lower", np.nan)):
                    row["NAP CI95"] = f"[{_nci['ci_lower']:.2f};{_nci['ci_upper']:.2f}]"
                raw = tau_u(a, b, improvement=improvement, correct_baseline_trend=False)
                cor = tau_u(a, b, improvement=improvement, correct_baseline_trend=True)
                cr, cc2 = raw.get("ci"), cor.get("ci")
                # --- per-case BASELINE trend (effect + significance; Tarlow 2016) ---
                bt, bp = raw.get("baseline_trend_tau"), raw.get("baseline_trend_p")
                row["Baseline trend (Tau-A)"] = round(bt, 3) if bt is not None else None
                row["p trend"] = _fmt_p(bp)
                row["Tau-U raw"] = round(raw.get("tau"), 3)                 # uncorrected A-vs-B contrast
                row["CI95 raw"] = f"[{cr[0]:.2f};{cr[1]:.2f}]" if cr else None
                row["Tau-U trend-corr."] = round(cor.get("tau_u"), 3)       # corrected coefficient (Tarlow/Brossard) - NOT 'tau'
                row["Corr. variant"] = cor.get("variant")
                row["Correction"] = _correction_label(cor.get("variant"))   # None / Tarlow / Brossard
                row["CI95 trend-corr."] = f"[{cc2[0]:.2f};{cc2[1]:.2f}]" if cc2 else None
                row["Tau-U crit."] = cor.get("critical")
                # OWN significance of the per-tier Tau-U (variance of S; NOT a randomization test)
                row["Tau-U p"] = _fmt_p(cor.get("p_value"))
                # --- verdict ALIGNED with the correction actually applied (Fingerhut/Brossard rule:
                # trend if p<0.05 OR |Tau-A|>=0.40), not on the p alone (short baseline = underpowered) ---
                tend = "no trend correction" not in (cor.get("variant") or "")   # was the correction applied?
                cor_net = bool(cc2 and cc2[0] > 0); raw_net = bool(cr and cr[0] > 0)
                if not tend:
                    row["Verdict (level/trend)"] = ("no baseline trend -> raw Tau-U reliable"
                                                    if raw_net else "no trend; uncertain effect")
                else:
                    row["Verdict (level/trend)"] = ("baseline trend (magnitude/p) -> ROBUST effect (corrected Tau-U)"
                                                    if cor_net else
                                                    "baseline trend (magnitude/p) -> effect CARRIED by the trend")
                # --- per-case RESPONDER: CREDIBLE non-overlap on the SAME branch as the verdict
                #     (raw Tau-U if no trend, corrected if trend); tier 'probable' if p<0.10;
                #     AND, if MCID provided, magnitude |diff| >= MCID ---
                _mc = mcid.get(outcome) if isinstance(mcid, dict) else mcid
                _diff = r.get("diff")
                mag_ok = (_mc is None) or (pd.notna(_diff) and abs(float(_diff)) >= float(_mc))
                credible = (cor_net if tend else raw_net)        # CI95 (raw/corrected depending on trend) excludes 0
                _tup = cor.get("p_value") if tend else raw.get("p_value")
                _tau = (cor.get("tau_u") if tend else raw.get("tau"))
                probable = (not credible) and bool((_tau or 0) > 0) and (_tup is not None and _tup < 0.10)
                base = "yes" if credible else ("probable" if probable else "no")
                row["Responder"] = base if (base == "no" or mag_ok) else f"{base} (sub-MCID)"
            except Exception:
                for k in ("Baseline trend (Tau-A)", "p trend", "Tau-U raw", "CI95 raw",
                          "Tau-U trend-corr.", "Corr. variant", "Correction", "CI95 trend-corr.",
                          "Tau-U crit.", "Tau-U p", "Verdict (level/trend)", "Responder"):
                    row[k] = None
            pertier_rows.append(row)

        # --- unified MBD dashboard (annotated staggered + Brinley + distribution) ---
        # if imputed points exist: TWO versions (with/without markers), indicated in the title
        if pp and plot_panels and pc is not None:
            from functions.sced.plots.panels import plot_mbd_panels
            try:
                starts = {str(r["Tier"]): r["Start"] for _, r in per.iterrows()
                          if pd.notna(r.get("Start"))}
                _bnd = bounds.get(outcome) if isinstance(bounds, dict) else bounds
                _has_imp = bool(_imp and _imp in gdf.columns and gdf[_imp].any())
                variants = ([("_marker", " (imputed data marked)", _imp),
                             ("_nomarker", " (imputed data not marked)", None)]
                            if _has_imp else [("", "", None)])
                for _suff, _ttag, _impv in variants:
                    plot_mbd_panels(gdf, tier_col=tc, session_col=sc, outcome_col=outcome,
                                    starts=starts, phase_col=pc, baseline=baseline, treatment=treatment,
                                    improvement=improvement, per_tier=per, brinley_mode=brinley_mode,
                                    bounds=_bnd, title=f"MBD: {outcome}{label}{_ttag}",
                                    name=f"{outcome}{_suff}", save_path=pp, style=style, impute_col=_impv)
            except Exception as e:
                if verbose:
                    print(f"  [panels skipped] {type(e).__name__}: {e}")

        # --- VAIOR per tier (baseline vs intervention) ---
        if pp and plot_vaior and pc is not None:
            from functions.sced.plots.vaior import plot_vaior_grid
            from functions.sced.plots.vaior import plot_vaior as _plot_vaior_pooled
            try:
                if vaior_pooled:
                    _plot_vaior_pooled(gdf, session_col=sc, condition_col=pc, outcome_col=outcome,
                                       reference=baseline, compared=treatment, improvement=improvement,
                                       save_path=pp, style=style)
                else:
                    _, vt = plot_vaior_grid(gdf, unit_col=tc, session_col=sc, condition_col=pc,
                                            outcome_col=outcome, reference=baseline,
                                            compared=treatment, improvement=improvement,
                                            save_path=pp, style=style)
                    npos = int(vt["decision_overall"].sum())
                    if verbose:
                        print(f"  VAIOR {treatment} vs {baseline} (per tier): {npos}/{len(vt)} positive")
            except Exception as e:
                if verbose:
                    print(f"  [VAIOR skipped] {type(e).__name__}: {e}")
        return info

    # --- LOOP cohorts x outcomes (each combination analysed SEPARATELY) ---
    if group_col:
        cohorts = [(g, gdf) for g, gdf in df.groupby(group_col)]
        if include_pooled:                                   # + all tiers together
            cohorts.append(("(all)", df))
        if verbose:
            print(f"\n=== cohorts '{group_col}': {[g for g, _ in cohorts]} ===")
    else:
        cohorts = [(None, df)]
    for gname, gdf in cohorts:
        gsub = None if gname == "(all)" else gname            # folder "(all)" but gtag -> None
        glabel = "" if gname is None else f" [{group_col}={gname}]"
        if gname is not None and verbose:
            print(f"\n################  {('ALL' if gname == '(all)' else group_col + ' = ' + str(gname))}"
                  f"  (n={gdf[tc].nunique()})  ################")
        for oc in outcomes:
            if output_dir is not None:                        # per-outcome tree: <oc>/Plot[/pooled]
                Lc = std_layout(output_dir, oc, _ANALYSIS)
                pp = Lc["plot"] if gname is None else os.path.join(Lc["plot"], str(gname))
                ppp = Lc["pooled"] if gname is None else os.path.join(Lc["pooled"], str(gname))
            else:
                sub = [] if gname is None else [str(gname)]
                pp = plot_path if not plot_path else os.path.join(plot_path, *sub, str(oc))
                ppp = pooled_plot_path if not pooled_plot_path else os.path.join(pooled_plot_path, *sub, str(oc))
            if (len(outcomes) > 1 or gname is not None) and verbose:
                print(f"\n--------  OUTCOME: {oc}{glabel}  --------")
            _analyze(gdf, gsub if gname != "(all)" else None, glabel, oc, pp, ppp)

    # --- CONSOLIDATED SUMMARY REPORT (1 row per group x outcome) ---
    summary = None
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        if "Outcome" in summary.columns:                       # results grouped/sorted by outcome
            summary = summary.sort_values("Outcome", kind="stable").reset_index(drop=True)
        if verbose:
            print("\n================  PERMUTATION  ================")
            show = ["Group", "Outcome", "Tiers (n)", "Statistic", "Observed stat",
                    "p-value", "Verdict"]
            print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
            print("\n================  MULTILEVEL (ML + BC-SMD)  ================")
            showm = ["Group", "Outcome", "Tiers (n)", "ML jump b2", "ML p b2 (KR)",
                     "ML slope b3", "ML p b3 (KR)", "ML ICC", "d (BC-SMD)", "BC-SMD CI95",
                     "d (BC-SMD trend-corr.)", "BC-SMD trend CI95"]
            print(summary[[c for c in showm if c in summary.columns]].to_string(index=False))
        if save_path or output_dir:
            from functions.sced.glossary import (interpretation_glossary, ascii_sanitize_df,
                                                 dedup_pooled_rows, data_recap_df, write_stacked,
                                                 style_cells, cohen_fill, provenance_df, data_quality_df,
                                                 slug_token)
            S = ascii_sanitize_df                       # ASCII guaranteed on each sheet

            def _p_fill(p):                             # significant -> green+bold; else grey
                if isinstance(p, str) and p.strip().startswith("<"):
                    return (True, "C6EFCE")             # e.g. "<0.0001"
                try:
                    return (True, "C6EFCE") if float(p) < 0.05 else (False, "F2F2F2")
                except (TypeError, ValueError):
                    return (False, None)

            def _nap_fill(v):                          # NAP: <0.66 grey / <0.93 amber / >=0.93 green
                try:
                    a = float(v)
                except (TypeError, ValueError):
                    return (False, None)
                return (False, "F2F2F2") if a < 0.66 else ((True, "FFE699") if a < 0.93 else (True, "C6EFCE"))

            _tau_fill = lambda v: cohen_fill(v, bands=(0.20, 0.60, 0.80))
            # --- "Permutation" sheet: test result (per group x outcome) ---
            perm_cols = ["Group", "Outcome", "Tiers (n)", "Statistic", "Observed stat",
                         "p-value", "Verdict"]
            perm_result = summary[[c for c in perm_cols if c in summary.columns]]
            # --- "Multilevel" sheet: ML + BC-SMD ---
            multi_cols = ["Group", "Outcome", "Tiers (n)", "ML jump b2", "ML p b1 (KR)",
                          "ML p b2 (KR)", "ML slope b3", "ML p b3 (KR)", "ML ICC",
                          "d (BC-SMD)", "BC-SMD CI95", "d (BC-SMD trend-corr.)", "BC-SMD trend CI95"]
            multi_df = summary[[c for c in multi_cols if c in summary.columns]]
            multi_rules = {"ML p b1 (KR)": _p_fill, "ML p b2 (KR)": _p_fill, "ML p b3 (KR)": _p_fill,
                           "d (BC-SMD)": cohen_fill, "d (BC-SMD trend-corr.)": cohen_fill}

            pertier = pd.DataFrame(pertier_rows)
            statcols = [c for c in ("Tau-U", "Hedges g (within)", "mean_A", "mean_B", "NAP", "diff")
                        if c in pertier.columns]
            if statcols:                                # split outcome: remove tiers WITHOUT data
                pertier = pertier[~pertier[statcols].isna().all(axis=1)]
            pertier = dedup_pooled_rows(pertier, "Tier")  # identical Tau-U cohort/(all) -> once
            if "Outcome" in pertier.columns:            # individual results sorted by outcome
                pertier = pertier.sort_values("Outcome", kind="stable").reset_index(drop=True)

            # === SINGLE "By tier" sheet (requested order) =============================
            _par = ["Group", "Outcome", "Tier", "n_A", "n_B", "mean_A", "mean_B", "diff",
                    "NAP", "NAP CI95", "Hedges g (within)", "Tau-U raw",
                    "Baseline trend (Tau-A)", "Correction", "Tau-U trend-corr.",
                    "CI95 trend-corr.", "Tau-U p", "Perm. stat (indiv.)", "p (indiv.)",
                    "autocorr lag-1"]
            pertier_view = pertier[[c for c in _par if c in pertier.columns]]
            par_rules = {"NAP": _nap_fill, "Tau-U raw": _tau_fill, "Tau-U trend-corr.": _tau_fill,
                         "Tau-U p": _p_fill, "p (indiv.)": _p_fill}
            # ============================================================================
            prov = provenance_df()                                                       # provenance (#7)
            rando_full = pd.DataFrame(rando_rows)
            refs = mbd_references()

            def _write_book(out_path, ocs):
                """Writes ONE permutation_test workbook for the outcome subset ``ocs``.
                Sheets: Permutation - Multilevel - By tier - Glossary - References."""
                def _sel(d):
                    return d[d["Outcome"].isin(ocs)] if (len(ocs) and "Outcome" in getattr(d, "columns", [])) else d
                permf, multif, parf, randof = (_sel(perm_result), _sel(multi_df),
                                               _sel(pertier_view), _sel(rando_full))
                if not len(permf):
                    return
                # ADMISSIBLE intervention starts (start-point possibilities = randomization window)
                if windows:
                    _starts = "; ".join(f"{k}: {list(v)}" for k, v in windows.items())
                else:
                    _sw = start_window
                    if baseline_window is not None:                       # number of A points -> start = n_A + 1
                        _sw = (int(baseline_window[0]) + 1, int(baseline_window[1]) + 1)
                    if _sw is not None:
                        _lst = list(range(int(_sw[0]), int(_sw[1]) + 1))
                        _starts = f"sessions {_sw[0]}-{_sw[1]} per tier ({len(_lst)} candidate starts: {_lst})"
                    elif intervention_starts:
                        _starts = f"imposed starts: {intervention_starts}"
                    else:
                        _starts = "not randomized (observed starts)"
                recapf = data_recap_df(df, tier_col=tc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                                       group_col=group_col, baseline=baseline, treatment=treatment,
                                       design="Multiple-baseline (permutation test + effect sizes)",
                                       extra=[("Admissible starts (randomization)", _starts)])
                qualityf = data_quality_df(df, tier_col=tc, session_col=sc, phase_col=pc,    # quality (#3)
                                           outcome_col=ocs[0] if len(ocs) == 1 else None,
                                           baseline=baseline, treatment=treatment) if len(ocs) == 1 else None
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                try:
                    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                        # 1) DATA & DESIGN: all the initial info (recap + quality + provenance)
                        dblocks = [("DATA & DESIGN", recapf)]
                        if qualityf is not None:
                            dblocks.append(("DATA QUALITY (per case)", qualityf))
                        dblocks.append(("PROVENANCE", prov))
                        write_stacked(w, "Data & design", dblocks)
                        # 2) PERMUTATION: scheme/combinations + test result
                        write_stacked(w, "Permutation",
                                      [("RANDOMIZATION SCHEME & COMBINATIONS", randof),
                                       ("PERMUTATION TEST RESULT", permf)],
                                      styles=[None, {"rules": {"p-value": _p_fill}}])
                        # 3) MULTILEVEL: ML + BC-SMD
                        S(multif).to_excel(w, sheet_name="Multilevel", index=False)
                        style_cells(w.sheets["Multilevel"], multif, multi_rules)
                        # 4) BY TIER (single sheet, sorted by outcome)
                        S(parf).to_excel(w, sheet_name="By tier", index=False)
                        style_cells(w.sheets["By tier"], parf, par_rules)
                        # 5) GLOSSARY & interpretation
                        write_stacked(w, "Glossary",
                                      [("GLOSSARY", mbd_glossary()),
                                       ("INTERPRETATION GUIDE", interpretation_glossary("mbd"))])
                        # 6) REFERENCES (bibliography + DOI)
                        S(refs).to_excel(w, sheet_name="References", index=False)
                    if verbose:
                        print(f"\nSummary written: {out_path}")
                except Exception as e:
                    if verbose:
                        print(f"\n[summary not written] {type(e).__name__}: {e}")

            if output_dir is not None:                       # per-outcome tree: <oc>/Analyse/permutation_test/
                for oc in outcomes:
                    _ad = std_layout(output_dir, oc, _ANALYSIS)["analyse"]
                    _write_book(os.path.join(_ad, "permutation_test_summary.xlsx"), [oc])
            else:                                            # consolidated mode: 1 workbook, all outcomes
                _fname = (f"permutation_test_{slug_token(outcomes[0])}_summary.xlsx"
                          if len(outcomes) == 1 else "permutation_test_summary.xlsx")
                _write_book(os.path.join(save_path, _fname), list(outcomes))

    if verbose:
        print("\n-> p<0.05: the staggered change is concordant with the introduction of "
              "the intervention (and not with the mere passage of time).")
    return summary
