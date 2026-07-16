"""
Consolidated reports - Bayesian SCED pipelines
==============================================

Orchestration layer above the Bayesian engines: loops over (cohort x outcome),
then WRITES the consolidated and STYLED .xlsx reports (color-coded results).
The templates only load the data and call the appropriate function.

- ``report_sced_model_compare``  : model comparison (PSIS-LOO + WAIC).
- ``report_sced_bayesian_percase`` : per-PATIENT model selection (LOO per patient), beta.
"""
import os

import numpy as np
import pandas as pd


def _gtok(g):
    """Clean cohort token for FILE NAMES: '(all)'/None -> 'all',
    otherwise alnum/_/-. Avoids ugly '_all___' in models/*.nc and figures."""
    s = "all" if (g is None or str(g).strip() in ("(all)", "all", "")) else str(g)
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


_FOREST_SPEC = {     # forest estimand -> per-case key, population keys (fallback), unit, suffix, title
    "effect_end": dict(case="effect_end", pop=("effect_end",), unit="pts", tag="endB",
                       label="cumulative effect end-B"),
    "slope": dict(case="slope", pop=("slope_points_per_session", "slope"), unit="pts/session", tag="b3",
                  label="effect slope b3"),
    "level": dict(case="level", pop=("level",), unit="pts", tag="b2", label="level jump b2"),
}


def _forest_spec(forest_estimand):
    """Return the forest spec dict for one estimand (falls back to 'effect_end')."""
    return _FOREST_SPEC.get(forest_estimand, _FOREST_SPEC["effect_end"])


def _resolve_forest_estimands(forest_estimand, terms):
    """Return the LIST of estimands to forest, CALIBRATED on the model terms.
    'auto': level only -> [level]; slope only (b1+b3) -> [slope]; level+slope (full) ->
    [level, slope, effect_end] (all 3: jump, slope, AND total effect). Explicit choice -> [that one]."""
    if forest_estimand and forest_estimand != "auto":
        return [forest_estimand]
    t = set(terms or [])
    ests = []
    if "level" in t:
        ests.append("level")
    if "slope" in t:
        ests.append("slope")
    if "level" in t and "slope" in t:
        ests.append("effect_end")                       # the total effect in addition to the jump and the slope
    return ests or ["effect_end"]


def _b_terms(terms):
    """UNIFORM b1/b2/b3 notation for a model from its terms (trend=b1, level=b2, slope=b3)."""
    t = set(terms)
    lab = "+".join(b for k, b in (("trend", "b1"), ("level", "b2"), ("slope", "b3")) if k in t)
    return lab or "b0"


def _b_map(models):
    """{model name -> b notation} to relabel the LOO/WAIC/Decisions tables."""
    from functions.sced.mbd.bayesian import _DEFAULT_MODELS
    return {name: _b_terms(terms) for name, terms in (models or _DEFAULT_MODELS).items()}


def _model_terms_df(models):
    """Sheet 'Compared models': name -> included terms (what was fitted)."""
    from functions.sced.mbd.bayesian import _DEFAULT_MODELS
    mods = models or _DEFAULT_MODELS
    rows = []
    for name, terms in mods.items():
        t = set(terms)
        rows.append({"Model": name, "Terms (b)": _b_terms(terms),
                     "trend b1": "yes" if "trend" in t else "-",
                     "level jump b2": "yes" if "level" in t else "-",
                     "slope b3": "yes" if "slope" in t else "-"})
    return pd.DataFrame(rows)


def _compare_glossary(se_mult):
    """Glossary of the comparison report columns/terms (ASCII)."""
    rows = [
        ("elpd_loo / elpd_waic", "Estimated out-of-sample predictive accuracy (PSIS-LOO; WAIC). HIGHER = better."),
        ("rank", "Model ranking (0 = best on this comparator)."),
        ("elpd_diff", "elpd gap to the best model. dse = standard error of that gap."),
        ("Decision (parsimony)", f"DECISIVE only if elpd_diff > {se_mult} x dse. Otherwise INDISTINGUISHABLE -> keep the simplest OR use the model-averaged effect (weights)."),
        ("weight (stacking)", "Weights of the optimal prediction combination (Yao et al. 2018). Sum=1."),
        ("weight pseudo-BMA+", "Alternative weights (Bayesian bootstrap). Compare to the stacking weights: robustness."),
        ("p_loo / p_waic", "Number of EFFECTIVE parameters (shrunk by the hierarchy). If it approaches N obs -> very flexible model."),
        ("n_params", "NOMINAL number of free parameters (structural). Grows with n_cases x number of terms (random slopes) and the pooling."),
        ("n_par/n_cas", "Structural complexity relative to the number of patients. High -> rich model vs data."),
        ("p_eff/N_obs", "EFFECTIVE parameters per observation. > ~0.2-0.3 -> overfitting risk / low parsimony."),
        ("k>0.7", "Number of points where PSIS-LOO is unreliable (Pareto-k>0.7). If high -> treat LOO with caution."),
        ("LOO vs WAIC", "Two canonical comparators (Vehtari 2017; Watanabe 2010). Agreement = robustness; disagreement = fragile signal."),
        ("Power", "On short series / small N, elpd_diff is often < threshold -> 'indistinguishable' = LACK OF POWER, not proof of absence of effect."),
    ]
    return pd.DataFrame(rows, columns=["Term", "Definition"])


def _pd_fill(v):                                  # CREDIBLE pd: >=0.95 green+bold / >=0.90 amber / otherwise grey
    """Cell color rule for a probability-of-direction value: value -> (bold, hex_fill) for style_cells."""
    try:
        a = float(v)
    except (TypeError, ValueError):
        return (False, None)
    return (True, "C6EFCE") if a >= 0.95 else ((True, "FFE699") if a >= 0.90 else (False, "F2F2F2"))


def _bool_fill(v):                                # boolean "decisive"/"agreement": TRUE -> green; FALSE -> grey
    """Cell color rule for a boolean value: value -> (bold, hex_fill) for style_cells (TRUE green, FALSE grey)."""
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return (True, "C6EFCE")
    if s in ("false", "no", "0"):
        return (False, "F2F2F2")
    return (False, None)


def _responder(median, pd_, mcid):
    """Per-case responder verdict: 'yes' if the direction is CREDIBLE (pd>=0.95), 'probable' if
    0.90<=pd<0.95, otherwise 'no'. If MCID is provided, also requires |effect|>=MCID (otherwise the
    suffix ' (sub-MCID)'). Reflects pd as continuous evidence (Makowski 2019), not a hard threshold.

    References: Makowski et al. 2019 (probability of direction).
    R equivalent: no direct R equivalent (responder rule on bayestestR::p_direction + MCID).
    """
    try:
        p = float(pd_)
    except (TypeError, ValueError):
        return None
    if p < 0.90:
        return "no"
    base = "yes" if p >= 0.95 else "probable"
    if mcid is None:
        return base
    try:
        return base if abs(float(median)) >= float(mcid) else f"{base} (sub-MCID)"
    except (TypeError, ValueError):
        return base


def _loo_row(idata):
    """elpd_loo + p_loo + number of Pareto-k>0.7 (PSIS-LOO predictive fit; Vehtari 2017).
    Returns {} if LOO fails.

    References: Vehtari, Gelman & Gabry 2017 (PSIS-LOO).
    R equivalent: loo::loo.
    """
    import arviz as az
    import numpy as _np
    try:
        lo = az.loo(idata)
        k = getattr(lo, "pareto_k", None)
        nk = int((_np.asarray(k) > 0.7).sum()) if k is not None else 0
        return {"elpd_loo": round(float(lo.elpd_loo), 1), "p_loo": round(float(lo.p_loo), 1),
                "k>0.7": nk}
    except Exception:
        return {}


def _emit_diagnostics(models, save_path, *, style=None, verbose=True):
    """MCMC diagnostics (trace/rank/posterior/PPC/panel) written NEXT TO the models, in
    ``<save_path>/diagnostics/`` - NEVER in a results folder. ``models`` = dict
    {key: idata}. ``style`` (PlotStyle/dict) drives fmt/dpi/font of the technical figures."""
    if not save_path:
        return
    from functions.sced.plots.panels import plot_bayesian_diag, plot_bayesian_panel
    dd = os.path.join(save_path, "diagnostics"); os.makedirs(dd, exist_ok=True)
    try:
        from functions.sced.glossary import bayesian_diag_legend
        bayesian_diag_legend().to_csv(os.path.join(dd, "_diagnostics_legend.csv"), index=False)
    except Exception:
        pass
    n = 0
    for key, idata in models.items():
        if idata is None:
            continue
        name = "__".join(_gtok(k) for k in (key if isinstance(key, tuple) else (key,)))
        try:
            plot_bayesian_diag(idata, save_path=dd, name=name, style=style)
            plot_bayesian_panel(idata, save_path=dd, name=name, style=style)
            n += 1
        except Exception as e:
            if verbose:
                print(f"  [diag skipped] {name}: {type(e).__name__}: {e}")
    if verbose and n:
        print(f"MCMC diagnostics: {n} models -> {dd}/")


def report_sced_model_compare(
        df, *, tier_col="tier", session_col="session", phase_col="phase",
        outcome_name_col=None, value_col=None, outcomes=("accuracy",),
        case_insensitive=True, date_col=None, date_dayfirst=False, group_col=None,
        phases=("A", "B"), baseline_phase=None, treatment_phase=None,
        improvement="increase", family="gaussian", bounds=(0, 100), n_trials=None,
        models=None, pooling="partial", ar=False, rope=0.5, se_mult=2.0,
        draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99,
        save_path=None, output_dir=None, diagnostics=False, cache_dir=None,
        force_refit=False, per_model_reports=True, pooled_title=None, style=None, verbose=True):
    """Report-level SCED model-comparison pipeline: loops over cohorts x outcomes
    (PSIS-LOO + WAIC + decisions), then writes the consolidated STYLED summary
    ``model_compare_summary.xlsx`` + saves the models to netCDF (if ``save_path``).
    Returns the decisions DataFrame (or None). The template only calls.

    References: Vehtari, Gelman & Gabry 2017 (PSIS-LOO); Watanabe 2010 (WAIC);
    Yao et al. 2018 (stacking).
    R equivalent: loo::loo / waic / loo_compare / loo_model_weights.
    """
    from functions.sced.mbd.bayesian import compare_sced_models, _DEFAULT_MODELS
    from functions.sced.glossary import (ascii_sanitize_df, data_recap_df, save_idata,
                                         interpretation_glossary, style_cells, write_stacked, provenance_df,
                                         std_layout)
    _family = "meta" if str(pooling) == "meta" else "hier"  # comparison filed under bayes/<hier|meta>/
    _analysis = f"bayes/{_family}"
    # output_dir: per-OUTCOME layout. The comparison goes in <oc>/Analyse/bayes/<fam>/model_compare_summary.xlsx;
    # the PER-MODEL reports + the cache + the forests are produced by report_sced_bayesian_mbd
    # (same output_dir) -> <oc>/Analyse/bayes/<fam>/<bN...>/ and <oc>/Plot/forest/<bN...>/.
    df_orig, outcomes_orig = df, outcomes                    # raw inputs for the per-model reports
    tc, sc, pc = tier_col, session_col, phase_col
    on, vc = outcome_name_col, value_col
    baseline = baseline_phase if baseline_phase is not None else phases[0]
    treatment = treatment_phase if treatment_phase is not None else phases[-1]

    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        r = resolve_columns(df, {"tier": tc, "session": sc, "phase": pc, "date": date_col,
                                 "group": group_col, "outcome_name": on, "value": vc})
        tc, sc, pc, date_col, group_col, on, vc = (r["tier"], r["session"], r["phase"], r["date"],
                                                   r["group"], r["outcome_name"], r["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=phases)
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=tc, new_col=sc, dayfirst=date_dayfirst)

    outcomes = list(outcomes) if outcomes else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        keep = [group_col] if group_col else []
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=[tc, sc, pc],
                                     keep_cols=keep)
        outcomes = outcomes or names
    if not outcomes:
        if verbose:
            print("No outcome.")
        return None

    def _fam(oc):
        return family.get(oc, "gaussian") if isinstance(family, dict) else family

    def _bounds(oc):
        return bounds.get(oc) if isinstance(bounds, dict) else bounds

    def _ntrials(oc):
        return n_trials.get(oc) if isinstance(n_trials, dict) else n_trials

    cohorts = ([(g, gd) for g, gd in df.groupby(group_col)] if group_col else [(None, df)])
    if group_col and len(cohorts) > 1:
        cohorts = cohorts + [("(all)", df)]

    name2b = _b_map(models)                                 # model name -> uniform b1/b2/b3 notation
    decisions, cmp_sheets, fitted = [], {}, {}
    for gname, gd in cohorts:
        gtag = "(all)" if (gname is None or gname == "(all)") else str(gname)
        for oc in outcomes:
            if gd[tc].nunique() < 2:
                if verbose:
                    print(f"[skipped] {gtag}/{oc}: <2 cases")
                continue
            _cd = cache_dir or (std_layout(output_dir, oc, _analysis)["models"]   # cache PAR OUTCOME
                                if output_dir is not None else None)
            try:
                res = compare_sced_models(
                    gd, tier_col=tc, session_col=sc, phase_col=pc, outcome_col=oc, models=models,
                    baseline=baseline, treatment=treatment, improvement=improvement, family=_fam(oc),
                    bounds=_bounds(oc), n_trials=_ntrials(oc), pooling=pooling, ar=ar, rope=rope,
                    draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept,
                    se_mult=se_mult, cache_dir=_cd, force_refit=force_refit)
            except Exception as e:
                if verbose:
                    print(f"[skipped] {gtag}/{oc}: {type(e).__name__}: {e}")
                continue
            tag = f"{oc}_{_gtok(gtag)}"
            loo = res["compare"].reset_index().rename(columns={"index": "Model"})
            loo.insert(1, "Terms (b)", loo["Model"].map(name2b))   # uniform b1/b2/b3 notation
            loo["w_pseudoBMA"] = [round(res["weights_pbma"].get(m, np.nan), 3) for m in loo["Model"]]
            cmp_sheets[f"LOO {tag}"] = loo
            waic = res["compare_waic"].reset_index().rename(columns={"index": "Model"})
            waic.insert(1, "Terms (b)", waic["Model"].map(name2b))
            cmp_sheets[f"WAIC {tag}"] = waic
            if verbose:
                print(f"\n######## {oc} / {gtag} (n={gd[tc].nunique()}) ########")
                print(loo.round(2).to_string(index=False))
                print(f"-> {res['decision']}  | WAIC best={res['best_waic']} (LOO/WAIC agreement: {res['loo_waic_agree']})")
            best = res["models"][res["best"]]
            lvl = best["population"].get("level"); slp = best["population"].get("slope")
            decisions.append({
                "Outcome": oc, "Cohort": gtag, "n_cases": int(best.get("n_cases", gd[tc].nunique())),
                "Selected model (LOO)": res["best"], "Selected terms (b)": name2b.get(res["best"]),
                "Decisive": res["decisive"],
                "Selected model (WAIC)": res["best_waic"], "LOO/WAIC agreement": res["loo_waic_agree"],
                "Decision": res["decision"],
                "stacking weight": round(res["weights"].get(res["best"], np.nan), 2),
                "level effect (pts)": (round(lvl["median"], 2) if lvl else None),
                "level pd": (round(lvl["pd"], 3) if lvl else None),
                "slope effect/session": (round(slp["median"], 2) if slp else None),
                "slope pd": (round(slp["pd"], 3) if slp else None)})
            for mname, mr in res["models"].items():
                fitted[(gtag, oc, mname)] = mr.get("_idata")

    if not decisions:
        if verbose:
            print("\nNo comparison.")
        return None
    decisions_df = pd.DataFrame(decisions)

    if not (save_path or output_dir):
        return decisions_df
    S = ascii_sanitize_df
    dec_rules = {"level pd": _pd_fill, "slope pd": _pd_fill,
                 "Decisive": _bool_fill, "LOO/WAIC agreement": _bool_fill}

    def _write_compare(out_path, ocs):
        decf = decisions_df[decisions_df["Outcome"].isin(ocs)] if "Outcome" in decisions_df.columns else decisions_df
        if decf.empty:
            return
        cmpf = {k: v for k, v in cmp_sheets.items()
                if any(k.startswith(f"LOO {oc}_") or k.startswith(f"WAIC {oc}_") for oc in ocs)}
        recap = data_recap_df(df, tier_col=tc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                              group_col=group_col, baseline=baseline, treatment=treatment,
                              design="SCED model comparison (LOO + WAIC)",
                              extra=[("Families", family), ("Pooling", pooling), ("AR1", ar),
                                     ("MCMC", f"{chains}x{draws} (warmup {tune}), target_accept={target_accept}"),
                                     ("Decision rule", f"decisive if Delta-elpd > {se_mult} x dse")])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                write_stacked(w, "Data & design", [("DATA & DESIGN", recap),
                                                   ("PROVENANCE", provenance_df(seed=seed))])
                S(_model_terms_df(models)).to_excel(w, sheet_name="Compared models", index=False)
                S(decf).to_excel(w, sheet_name="Decisions", index=False)
                style_cells(w.sheets["Decisions"], decf, dec_rules)
                for name, cmp in cmpf.items():
                    S(cmp).to_excel(w, sheet_name=name[:31], index=False)
                S(_compare_glossary(se_mult)).to_excel(w, sheet_name="Glossary", index=False)
                S(interpretation_glossary("bayes")).to_excel(w, sheet_name="Interpretation guide", index=False)
            if verbose:
                print(f"\nReport written: {out_path}")
        except Exception as e:
            if verbose:
                print(f"\n[not written] {type(e).__name__}: {e}")

    def _one_model_report(terms, *, out_dir_arg=None, save_arg=None, forest_arg=None):
        try:
            report_sced_bayesian_mbd(
                df_orig, pooling=pooling, onset="known", ar=ar, terms=set(terms), analysis=_analysis,
                tier_col=tier_col, session_col=session_col, phase_col=phase_col,
                outcome_name_col=outcome_name_col, value_col=value_col, outcomes=outcomes_orig,
                case_insensitive=case_insensitive, date_col=date_col, group_col=group_col,
                phases=phases, baseline_phase=baseline_phase, treatment_phase=treatment_phase,
                improvement=improvement, family=family, bounds=bounds, n_trials=n_trials, rope=rope,
                draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept,
                output_dir=out_dir_arg, save_path=save_arg, forest_path=forest_arg,
                cache_dir=cache_dir, diagnostics=diagnostics, pooled_title=pooled_title,
                style=style, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  [model report {_b_terms(terms)} skipped] {type(e).__name__}: {e}")

    if output_dir is not None:                              # per-OUTCOME layout
        for oc in outcomes:
            ad = std_layout(output_dir, oc, _analysis)["analyse"]
            _write_compare(os.path.join(ad, "model_compare_summary.xlsx"), [oc])
        if per_model_reports:                               # per-model reports + cache + forests + pooled
            for name, terms in (models or _DEFAULT_MODELS).items():
                _one_model_report(terms, out_dir_arg=output_dir)
            if verbose:
                print(f"Per-model reports -> <oc>/Analyse/{_analysis}/<model>/ (reuse the cache)")
    else:                                                   # advanced mode: single folder
        os.makedirs(save_path, exist_ok=True)
        _write_compare(os.path.join(save_path, "model_compare_summary.xlsx"), list(outcomes))
        if cache_dir:
            if verbose:
                print(f"Models in shared cache: {cache_dir}/")
        else:
            nsaved = sum(save_idata(idata, os.path.join(save_path, "models"), f"{_gtok(g)}__{oc}__{m}") is not None
                         for (g, oc, m), idata in fitted.items())
            if nsaved and verbose:
                print(f"Bayesian models saved: {nsaved} -> {os.path.join(save_path, 'models')}/")
        if diagnostics:
            _emit_diagnostics(fitted, save_path, style=style, verbose=verbose)
        if per_model_reports:
            for name, terms in (models or _DEFAULT_MODELS).items():
                _bt = _b_terms(terms).replace("+", "")
                _one_model_report(terms, save_arg=os.path.join(save_path, _bt),
                                  forest_arg=os.path.join(save_path, _bt, "plots"))
    return decisions_df


# =========================================================================== #
#  bayesian_mbd : hierarchical / meta two-stage / per-case reports
# =========================================================================== #
def _rhat_fill(v):
    """Cell color rule for an R-hat value: value -> (bold, hex_fill), red when R-hat > 1.01 (non-convergence)."""
    try:
        return (True, "FFC7CE") if float(v) > 1.01 else (False, None)
    except (TypeError, ValueError):
        return (False, None)


def _div_fill(v):
    """Cell color rule for a divergences count: value -> (bold, hex_fill), red when any divergence (> 0)."""
    try:
        return (True, "FFC7CE") if float(v) > 0 else (False, None)
    except (TypeError, ValueError):
        return (False, None)


def _ppc_fill(v):                                      # PPC autocorr: extreme (<0.05 or >0.95) = misfit
    """Cell color rule for a PPC autocorrelation p-value: value -> (bold, hex_fill), red at extremes, amber near them."""
    try:
        a = float(v)
    except (TypeError, ValueError):
        return (False, None)
    return (True, "FFC7CE") if (a < 0.05 or a > 0.95) else ((False, "FFE699") if (a < 0.1 or a > 0.9)
                                                            else (False, None))


def _i2_fill(v):                                       # meta heterogeneity: >0.75 red, >0.5 amber
    """Cell color rule for a meta I2 value: value -> (bold, hex_fill), red > 0.75, amber > 0.50."""
    try:
        a = float(v)
    except (TypeError, ValueError):
        return (False, None)
    return (True, "FFC7CE") if a > 0.75 else ((False, "FFE699") if a > 0.50 else (False, None))


def _bayes_appreciation():
    """Benchmarks to APPRAISE each value of the Bayesian report (magnitude / reliability)."""
    rows = [
        ("pd (probability of direction)", "0.50 = no direction. Makowski 2019 benchmarks: 0.90 uncertain; 0.95 evidence; 0.975 strong; 0.99 very strong. NO 'significant' in Bayesian terms."),
        ("HDI 95 %", "Includes 0 -> UNCERTAIN effect (!= absence of effect); excludes 0 -> credible direction. The WIDTH = the precision."),
        ("Level / slope effect (points)", "In original units: judge vs the MINIMUM CLINICALLY IMPORTANT DIFFERENCE (MCID) of the outcome; no universal threshold."),
        ("tau (inter-case heterogeneity)", "SD of the effect across patients. To compare TO THE EFFECT: large tau vs effect = HETEROGENEOUS response (responders and non-responders); tau~0 (random_intercept) = effect assumed COMMON."),
        ("P(effect>ROPE)", "Posterior probability of a PRACTICALLY useful effect (beyond the ROPE); set the ROPE to the MCID. >0.95 = credibly useful."),
        ("phi (Beta)", "Precision of the Beta distribution = the model 'noise' (var = mu(1-mu)/(1+phi)). LARGE phi = low residual dispersion (points tight around the mean); small phi = high variability. It is the analogue of Gaussian 1/sigma^2. Sensitive to the prior (HalfNormal(50)) at small n -> read as indicative."),
        ("autocorr PPC p", "Posterior predictive check of the lag-1 autocorrelation (BARG): p = P(replicated autocorr >= observed). ~0.5 = the conditional-independence model REPRODUCES the observed serial dependence (no need for AR); near 0 = observed autocorr stronger than the model (possible under-modeling -> slightly optimistic pd); near 1 = the reverse. In [0.1;0.9] = adequate."),
        ("Rhat / ESS / diverg.", "CONVERGENCE (prerequisite to any reading): Rhat < 1.01; ESS_bulk > 400/parameter; 0 divergence. Otherwise results NOT reliable."),
        ("Caveat small n", ">= ~8 points/phase and >= ~5 cases advised (Natesan Batley 2020); otherwise wide HDIs, low-power model comparison (LOO) ('indistinguishable' != 'no effect')."),
    ]
    return pd.DataFrame(rows, columns=["Value", "How to read it"])


def _familles_guide_df():
    """Reference sheet describing each outcome family (gaussian/student/beta/binomial) and its link/effect scale. Returns a DataFrame."""
    rows = [
        ("gaussian", "Continuous, ~mid-scale, constant variance", "AR1. Default."),
        ("student", "Continuous WITH outliers", "ROBUST Gaussian (heavy tails), keeps AR1."),
        ("beta", "BOUNDED continuous score, floor/ceiling", "Logit link. Effect in POINTS. Handles the ceiling."),
        ("binomial", "COUNT k/N", "Requires N_TRIALS. Logit link, effect in POINTS."),
        ("Ceiling -> Beta", "Bounded/ceiling outcome: model as Beta (logit link); for a design-comparable",
         "effect size, use R scdhlm (g_AB REML), not a home-made Gaussian SMD."),
    ]
    return pd.DataFrame(rows, columns=["Family", "Outcome type", "Note"])


def _param_glossary():
    """Explains EACH parameter name of the 'Parameters (full)' sheet."""
    rows = [
        (" - RAW COEFFICIENTS (link scale: logit in beta/binomial, otherwise original unit) - ", ""),
        ("mu_b0", "Population mean of the INTERCEPT (baseline level at the center of the series)."),
        ("mu_b1", "Population mean of the baseline TREND (slope per session)."),
        ("mu_b2", "Population mean of the level JUMP at the intervention (raw immediate effect)."),
        ("mu_b3", "Population mean of the effect SLOPE in phase B (per session)."),
        ("sd_b0", "INTER-case standard deviation of the intercept (heterogeneity of baseline levels)."),
        ("sd_b1", "INTER-case standard deviation of the baseline trend (= tau_trend)."),
        ("sd_b2", "INTER-case standard deviation of the jump (= tau_level); 0 if common effect (random_intercept)."),
        ("sd_b3", "INTER-case standard deviation of the effect slope (= tau_slope); 0 if common slope."),
        ("b0[patient] ... b3[patient]", "PER-CASE (shrunk) value of the coefficient; the index in [] is the PATIENT."),
        ("phi", "Precision of the Beta distribution (residual dispersion; large phi = low variance)."),
        ("rho", "Lag-1 autocorrelation of the residuals (gaussian/student with AR1)."),
        ("nu", "Degrees of freedom of the Student-t (robustness to outliers)."),
        ("re_corr_b2", "Intercept<->jump correlation across cases (pooling 'correlated'); captures the ceiling."),
        (" - DERIVED QUANTITIES (oriented by 'improvement', in POINTS = original scale) - ", ""),
        ("pop_level", "Population level jump IN POINTS = invlogit(mu_b0+mu_b2)-invlogit(mu_b0) (beta), oriented."),
        ("logit_b2", "The same population jump but on the LOGIT scale (= sign x mu_b2); beta/binomial."),
        ("tau_level", "Inter-case SD of the jump, as a derived quantity (= sd_b2)."),
        ("pop_trend", "Population baseline trend, oriented coefficient (logit/session in beta) (= sign x mu_b1)."),
        ("pop_trend_pts_ps", "Baseline trend IN POINTS per session (marginal at the middle of the series)."),
        ("tau_trend", "Inter-case SD of the baseline trend (= sd_b1)."),
        ("pop_slope", "Population effect slope, oriented coefficient (logit/session in beta) (= sign x mu_b3)."),
        ("pop_slope_pts_ps", "Effect slope IN POINTS per session (marginal at the start of phase B)."),
        ("pop_slope_pts_cum", "Progressive CUMULATIVE effect in points over the whole phase B (longest horizon)."),
        ("tau_slope", "Inter-case SD of the effect slope (= sd_b3)."),
        (" - DIAGNOSTICS (table columns) - ", ""),
        ("mean / sd / hdi_3% / hdi_97%", "Posterior mean, standard deviation and 94 % HDI of the parameter."),
        ("ess_bulk / ess_tail / r_hat", "Effective sample size and convergence (r_hat < 1.01 required)."),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Meaning"])


def _bayes_glossary():
    """Glossary of the per-case Bayesian report columns/terms. Returns a DataFrame (Term, Definition)."""
    g = [
        ("es", "Standardized effect size (beta_B-beta_A)/sigma, ORIENTED (es>0 = improvement)."),
        ("es HDI", "Highest-density interval 95% (direct interpretation, != frequentist CI)."),
        ("pd", "Probability of direction. >0.95 ~ strong evidence. NO 'significant' in Bayesian terms."),
        ("P(es>ROPE)", "Posterior probability of a practically useful effect (beyond the ROPE, Kruschke)."),
        ("TWO tests (TREND)", "TREND=True: es = immediate jump; slope b3 = FASTER improvement in B."),
        ("Baseline trend", "Tarlow test reported as DIAGNOSTIC; inclusion of the PRE-SPECIFIED baseline slope (BASELINE_TREND True/False), not automatic."),
        ("es_end (cumulative)", "TREND: effect at the END of B = (jump + slope.duration)/sigma (cumulative/late effect)."),
        ("rho", "Lag-1 autocorrelation (BITS/BUCP). 0 forces 'simple'."),
        ("CP->B (mode)", "BUCP: most probable switch session = IMMEDIACY."),
        ("Rhat / ESS / diverg.", "MCMC diagnostics: Rhat<1.01 required; '!' = doubtful."),
        ("simple/BITS/BUCP", "simple = i.i.d.; BITS = +AR1; BUCP = + unknown change-point."),
        ("Prerequisite", ">= ~8 points/phase and expected d >= 3 (Natesan Batley 2020)."),
    ]
    return pd.DataFrame(g, columns=["Term", "Definition"])


def report_sced_bayesian_mbd(
        df, *, pooling="partial", onset="known", ar=True, mode=None, model=None,
        tier_col="tier", session_col="session", phase_col="phase",
        outcome_name_col=None, value_col=None, outcomes=("accuracy",),
        case_insensitive=True, date_col=None, date_dayfirst=False, group_col=None,
        phases=("A", "B"), baseline_phase=None, treatment_phase=None,
        improvement="increase", rope=0.5,
        hypothesis="immediate_flat", terms=None, family="gaussian", bounds=(0, 100),
        n_trials=None, target_accept=0.95,
        trend=False, baseline_trend=False, min_per_phase=3,
        draws=2000, tune=2000, chains=4, seed=42, save_path=None, output_dir=None,
        diagnostics=False, mcid=None, cache_dir=None, force_refit=False,
        forest_estimand="auto", forest_path=None, analysis=None, pooled_title=None,
        style=None, verbose=True):
    """Report-level Bayesian SCED pipeline for phases. Three orthogonal axes:
    ``pooling`` (sharing across cases: partial/random_intercept/correlated/none/meta),
    ``onset`` (``"known"`` = switch = phase label, hierarchical model;
    ``"unknown"`` = estimated BUCP change-point, NOT poolable -> forces ``pooling="none"``),
    ``ar`` (intra-case AR1 noise, = the former BITS; ignored in beta/binomial). The engine is
    DERIVED from these axes. Loops over cohorts x outcomes -> styled .xlsx report (+ .nc) if ``save_path``.

    Backward compat: ``mode`` (``"hierarchical"``/``"per_case"``) + ``model`` (simple/bits/bucp)
    from the old API remain accepted (deprecated). ``output_dir`` = single root;
    ``diagnostics=True`` -> MCMC diagnostics in ``<output>/diagnostics/``.

    References: Van den Noortgate & Onghena 2003 (hierarchical SCED); Moeyaert et al. 2014;
    Makowski et al. 2019 (HDI / pd / ROPE).
    R equivalent: brms (multilevel brm, per-case random effects, ar(1)) + bayestestR.
    """
    import warnings as _warnings
    # --- AXES -> engine resolution (with mode/model backward compat) ---
    if mode is not None or model is not None:               # old API: honor + deprecate
        _warnings.warn("report_sced_bayesian_mbd: `mode`/`model` are deprecated; "
                       "use pooling/onset/ar.", DeprecationWarning, stacklevel=2)
        _per_case = (mode == "per_case") or (model == "bucp")
        if model == "bucp":
            onset = "unknown"
        if model in ("simple",):
            ar = False
    else:
        _per_case = (onset == "unknown")
    if onset == "unknown":                                  # BUCP not poolable -> single-case
        if pooling not in (None, "none") and verbose:
            print(f"[onset='unknown'] pooling='{pooling}' ignored -> 'none' (unknown switch not poolable)")
        pooling = "none"
        _per_case = True
        model = "bucp"
    elif _per_case and model is None:                       # legacy mode='per_case' without model
        model = "bits"
    _family = "meta" if str(pooling) == "meta" else ("per_case" if _per_case else "hier")
    if analysis is None:                                    # analysis sub-folder (may be overridden by model_compare)
        analysis = f"bayes/{_family}"
    # output_dir: per-OUTCOME layout resolved inside the _bayes_run_* functions (outcome-aware).
    #   <output_dir>/<oc>/Analyse/bayes/<hier|meta>/<bN...>/ (xlsx) + models/ + diagnostics/ ;
    #   <output_dir>/<oc>/Plot/forest/<bN...>/ ; <output_dir>/<oc>/Plot/pooled/ (Bayesian pooled).
    # save_path/forest_path: advanced overrides (single folder, consolidated multi-outcome report).
    tc, sc, pc = tier_col, session_col, phase_col
    on, vc = outcome_name_col, value_col
    baseline = baseline_phase if baseline_phase is not None else phases[0]
    treatment = treatment_phase if treatment_phase is not None else phases[-1]

    # --- prep: case, dates->sessions, tidy ---
    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        r = resolve_columns(df, {"tier": tc, "session": sc, "phase": pc, "date": date_col,
                                 "group": group_col, "outcome_name": on, "value": vc})
        tc, sc, pc, date_col, group_col, on, vc = (r["tier"], r["session"], r["phase"], r["date"],
                                                   r["group"], r["outcome_name"], r["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=phases)
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=tc, new_col=sc, dayfirst=date_dayfirst)
        if verbose:
            print(f"STEP 0 - Dates '{date_col}' -> session 1..n per case in '{sc}'")
    outcomes = list(outcomes) if outcomes else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        keep = [group_col] if group_col else []
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=[tc, sc, pc],
                                     keep_cols=keep)
        outcomes = outcomes or names
        if verbose:
            print(f"STEP 0 - tidy: outcomes = {outcomes}")
    if not outcomes:
        if verbose:
            print("No outcome.")
        return None

    def _cohorts():
        co = ([(g, gd) for g, gd in df.groupby(group_col)] if group_col else [(None, df)])
        if group_col and len(co) > 1:
            co = co + [("(all)", df)]
        return co

    # --- reference sheets (parameter-dependent) ---
    def _meta_modele_df():
        fam = (", ".join(f"{k}={v}" for k, v in family.items()) if isinstance(family, dict) else family)
        spec_row = (f"terms={sorted(terms)}" if terms else f"{hypothesis}")
        return pd.DataFrame({"Element": [
            "Model", "Stage 1", "Stage 2", "Meta estimands", "End-B effect", "Heterogeneity",
            "Families", "Terms", "MCMC", "Inference", "Responders", "References"], "Value": [
            "TWO-STAGE Bayesian META-ANALYSIS (NOT a single hierarchical model).",
            "INDEPENDENT per-case fits (pooling='none', PyMC NUTS) -> raw effect_i + SE_i (uncontaminated).",
            "random-effects NON-CENTERED per parameter: mu ~ N; tau ~ HalfNormal; theta_i = mu + tau*z_i; "
            "eff_i ~ N(theta_i, SE_i). 'Population' = mu; 'Per case shrunk' = theta_i.",
            "b1 (trend), b2 (level), b3 (slope), end-B effect -- each meta-analyzed separately.",
            "end-B effect = b2 + b3*T_B (points) -- derived contrast, the best identified.",
            "tau (inter-case standard deviation) + I2 (Higgins-Thompson) + prediction interval.",
            f"{fam}  (beta/binomial = logit, effect in POINTS)",
            spec_row, f"{chains} chains x {draws} (warmup {tune}), seed {seed}",
            "HDI 95 % + pd; per case = RAW (stage 1, individual) AND SHRUNK (stage 2). Detection -> randomization.",
            "Binarize responder/non on the RAW (vs MCID + per-case randomization). The SHRUNK (toward mu) = a "
            "group statement / stabilized ranking, NOT for individual decisions (false pos. near mu, false neg. idiosyncratic).",
            "Van den Noortgate & Onghena 2003/2008; Moeyaert 2017; Burke-Ensor-Riley 2017; Higgins-Thompson 2002; Gelman 2013."]})

    def _modele_priors_df():
        fam = (", ".join(f"{k}={v}" for k, v in family.items()) if isinstance(family, dict) else family)
        use_b1 = (("trend" in terms) if terms else hypothesis in ("immediate_trend", "cumulative_trend"))
        use_b2 = (("level" in terms) if terms else True)
        use_b3 = (("slope" in terms) if terms else hypothesis in ("cumulative_flat", "cumulative_trend"))
        terms_s = "b0 (intercept)" + (" + b1 (baseline trend)" if use_b1 else "") \
            + (" + b2 (jump/level)" if use_b2 else "") + (" + b3 (slope in B)" if use_b3 else "")
        actifs = [t for t, u in (("b1=trend", use_b1), ("b2=level", use_b2), ("b3=slope", use_b3)) if u]
        spec_row = (f"terms={sorted(terms)} (explicit override; HYPOTHESIS ignored)" if terms
                    else f"{hypothesis}")
        rows = [
            ("Model", "Bayesian hierarchical: a single model over all cases (PyMC NUTS)."),
            ("Equation", f"y_it = {terms_s}; effects (b)_i ~ Normal(mu, tau) per case."),
            ("Active terms", ", ".join(actifs)),
            ("Specification", spec_row),
            ("Families", f"{fam}  (gaussian/student = AR1; beta/binomial = logit, effect in POINTS)"),
            ("Pooling", f"{pooling}  (partial=RI+RS | random_intercept=RI | correlated=RI+RS correlated (LKJ) | none)"),
            ("Bounds (beta/binom)", (", ".join(f"{k}={v}" for k, v in bounds.items())
                                     if isinstance(bounds, dict) else str(bounds))),
            ("N items (binomial)", (", ".join(f"{k}={v}" for k, v in n_trials.items())
                                    if isinstance(n_trials, dict) else str(n_trials))),
            ("AR1 (gaussian/student)", (", ".join(f"{k}={v}" for k, v in ar.items())
                                        if isinstance(ar, dict) else str(ar)) + " -- ignored in beta/binomial"),
            ("Orientation", f"improvement={improvement} (effect > 0 = improvement)"),
            ("ROPE", f"{rope} -- on the EFFECT in original units (MCID); P(effect>ROPE) is reported"),
            ("BC-SMD (design-comparable)", "not computed internally -> use R scdhlm (bc_smd_scdhlm): canonical g_AB REML"),
            ("Prior effect mu_b2", "Normal(0, sd_y) [gaussian] / Normal(0, 1.5) [beta] -- skeptical"),
            ("Prior residual sigma", "HalfNormal(sd_y) [gaussian/student]; phi ~ HalfNormal(50) [beta]"),
            ("Prior inter-case SD (tau)", "HalfNormal; correlated: LKJ(eta=2) covariance"),
            ("MCMC", f"{chains} chains x {draws} (warmup {tune}), seed {seed}, target_accept={target_accept}"),
            ("Inference", "HDI 95 % + pd. No 'significant': pd >= 0.95 for a direction."),
            ("References", "Rindskopf 2014; Van den Noortgate & Onghena 2003; Moeyaert; Kruschke 2013."),
        ]
        return pd.DataFrame(rows, columns=["Element", "Value"])

    if _per_case:
        return _bayes_run_per_case(
            df, tc, sc, pc, group_col, outcomes, baseline, treatment, _cohorts,
            improvement=improvement, rope=rope, model=model, trend=trend,
            baseline_trend=baseline_trend, min_per_phase=min_per_phase, family=family,
            draws=draws, tune=tune, chains=chains, seed=seed, save_path=save_path,
            diagnostics=diagnostics, cache_dir=cache_dir, force_refit=force_refit,
            output_dir=output_dir, analysis=analysis, style=style, verbose=verbose)
    return _bayes_run_hierarchical(
        df, tc, sc, pc, group_col, outcomes, baseline, treatment, _cohorts,
        _meta_modele_df, _modele_priors_df, improvement=improvement, rope=rope,
        hypothesis=hypothesis, terms=terms, family=family, bounds=bounds, n_trials=n_trials,
        pooling=pooling, ar=ar, target_accept=target_accept, draws=draws, tune=tune,
        chains=chains, seed=seed, save_path=save_path, diagnostics=diagnostics, mcid=mcid,
        cache_dir=cache_dir, force_refit=force_refit, forest_estimand=forest_estimand,
        forest_path=forest_path, output_dir=output_dir, analysis=analysis,
        pooled_title=pooled_title, style=style, verbose=verbose)


def _bayes_run_hierarchical(df, tc, sc, pc, group_col, outcomes, baseline, treatment, _cohorts,
                            _meta_modele_df, _modele_priors_df, *, improvement, rope, hypothesis,
                            terms, family, bounds, n_trials, pooling, ar, target_accept,
                            draws, tune, chains, seed, save_path, diagnostics=False, mcid=None,
                            cache_dir=None, force_refit=False, forest_estimand="auto",
                            forest_path=None, output_dir=None, analysis=None, pooled_title=None,
                            style=None, verbose=True):
    """Run the hierarchical (or two-stage meta) Bayesian SCED engine over cohorts x outcomes and
    write the styled report. Inputs: tidy ``df`` + resolved column names + fit options; output:
    the population (or per-case) DataFrame. Side effects: writes the .xlsx report, .nc models, and
    forest / pooled / diagnostic figures when ``save_path`` or ``output_dir`` is set.

    References: Van den Noortgate & Onghena 2003 / 2008 (hierarchical + two-stage meta);
    Moeyaert et al. 2014; Higgins & Thompson 2002 (I2).
    R equivalent: brms; bayesmeta / metafor::rma for the two-stage meta.
    """
    import os
    from functions.sced.mbd.bayesian import bayes_hier_sced
    from functions.sced.glossary import std_layout
    _is_meta = str(pooling) == "meta"
    analysis = analysis or f"bayes/{'meta' if _is_meta else 'hier'}"

    def _oc_cache(oc):                                      # cache fit-or-load PAR OUTCOME (= dossier models/)
        if cache_dir:
            return cache_dir
        return std_layout(output_dir, oc, analysis)["models"] if output_dir is not None else None

    def _fam(oc): return family.get(oc, "gaussian") if isinstance(family, dict) else family
    def _bounds(oc): return bounds.get(oc) if isinstance(bounds, dict) else bounds
    def _ntrials(oc): return n_trials.get(oc) if isinstance(n_trials, dict) else n_trials
    def _ar(oc): return ar.get(oc, True) if isinstance(ar, dict) else ar

    if verbose:
        print(f"\nMODE=hierarchical : {hypothesis} | families={family} | pooling={pooling} | MCMC {chains}x{draws}\n")
    pooled_proper = pooling != "none"
    pop_rows, case_rows, models = [], [], {}
    brut_rows = []
    forest_data = {}                                       # (g,oc) -> dict for the end-B effect forest (#9)
    pooled_data = {}                                       # (g,oc) -> dict for the Bayesian pooled plot (systematic)

    def _starts_of(gd):                                    # intervention onset per case (1st B point)
        st = {}
        for t, sub in gd.groupby(tc):
            sub2 = sub[sub[pc].astype(str) == str(treatment)].sort_values(sc)
            if len(sub2):
                st[str(t)] = sub2[sc].iloc[0]
        return st

    def _one(gd, gtag, oc):
        fam = _fam(oc)
        try:
            r = bayes_hier_sced(gd, tier_col=tc, session_col=sc, phase_col=pc, outcome_col=oc,
                                baseline=baseline, treatment=treatment, improvement=improvement,
                                hypothesis=hypothesis, terms=terms, pooling=pooling, family=fam,
                                bounds=_bounds(oc), n_trials=_ntrials(oc), ar=_ar(oc), rope=rope,
                                draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept,
                                cache_dir=_oc_cache(oc), force_refit=force_refit)
        except Exception as e:
            if verbose:
                print(f"  [skipped] {gtag}/{oc}: {type(e).__name__}: {e}")
            return
        g = "(all)" if gtag is None else str(gtag)
        lv = r["population"].get("level")
        flag = "!" if r["rhat_max"] > 1.05 or r["diverging"] > 0 else ""
        if verbose:
            lvtxt = (f"effect(points)={lv['median']:+.2f} [{lv['hdi_low']:+.2f};{lv['hdi_high']:+.2f}] "
                     f"pd={lv['pd']:.3f}" + (f" P>{rope}={lv.get('p_gt_rope', float('nan')):.2f}"
                                             if "p_gt_rope" in (lv or {}) else "") if lv else "(no level term)")
            print(f"  [{g}/{oc}] ({fam}) {lvtxt} | Rhat={r['rhat_max']:.3f}{flag}")
        def _hdi(s, p=2): return f"[{s['hdi_low']:.{p}f};{s['hdi_high']:.{p}f}]"
        prow = {"Group": g, "Outcome": oc, "Family": fam, "Pooling": pooling, "n_cases": r["n_cases"]}
        tr = r["population"].get("trend_points_per_session") or r["population"].get("trend")
        if tr:
            prow.update({"Baseline trend (b1, pts/session)": round(tr["median"], 2),
                         "b1 HDI": _hdi(tr), "b1 pd": round(tr["pd"], 3)})
        prow.update({"Level effect (b2, pts)": (round(lv["median"], 3) if lv else np.nan),
                     "b2 HDI": (_hdi(lv) if lv else ""),
                     "b2 pd": (round(lv["pd"], 3) if lv else np.nan),
                     **({f"P(effect>{rope})": round(lv["p_gt_rope"], 3)} if (lv and "p_gt_rope" in lv) else {})})
        if "slope" in r["population"]:
            sl = r["population"]["slope"]
            base_sl = r["population"].get("slope_points_per_session") or sl
            prow.update({"Effect slope (b3, pts/session)": round(base_sl["median"], 2),
                         "b3 HDI": _hdi(base_sl), "b3 pd": round(base_sl["pd"], 3)})
            slp_cu = r["population"].get("slope_points_cumulative")
            if slp_cu:
                Tb = r["population"].get("slope_horizon_B")
                prow.update({"cumulative effect b3 (pts)": round(slp_cu["median"], 2),
                             "cumulative HDI": _hdi(slp_cu), "cumulative pd": round(slp_cu["pd"], 3),
                             "horizon B (sessions)": int(Tb) if Tb is not None else np.nan})
        ee = r["population"].get("effect_end")
        if ee:
            prow.update({"Total effect end-B (b2+b3*T_B, pts)": round(ee["median"], 2),
                         "end-B HDI": _hdi(ee), "end-B pd": round(ee["pd"], 3)})
            if "rope_decision" in ee:                  # ROPE (Kruschke 2018 / bayestestR) on the clinical effect
                prow["% in ROPE"] = round(ee["pct_in_rope"] * 100, 1)
                prow["ROPE decision"] = ee["rope_decision"]
        prow.update({"tau heterog. (b2)": (round(r["heterogeneity"]["sd_level"]["median"], 2)
                                         if "sd_level" in r["heterogeneity"] else np.nan),
                     "rho": (round(r["rho"]["median"], 3) if "rho" in r else np.nan),
                     "Rhat": round(r["rhat_max"], 3), "ESS min": int(r["ess_min"]) if r.get("ess_min") == r.get("ess_min") else np.nan,
                     "diverg.": r["diverging"]})
        prow.update(_loo_row(r.get("_idata")))             # elpd_loo + p_loo + k>0.7 (ajustement #6)
        if "phi" in r:                                     # Beta precision (dispersion)
            prow["phi (Beta)"] = round(r["phi"]["median"], 1)
        if "autocorr_ppc_p" in r:                          # autocorrelation PPC (BARG)
            prow["autocorr PPC p"] = r["autocorr_ppc_p"]
        if pooled_proper:
            pop_rows.append(prow)
        def _h(s): return f"[{s['hdi_low']:.2f};{s['hdi_high']:.2f}]"
        for pcs in r["per_case"]:
            cr = {"Group": g, "Outcome": oc, "Case": pcs["case"]}
            if "trend" in pcs:
                T = pcs["trend"]
                cr.update({"b1 trend (pts/session, shrunk)": round(T["median"], 3),
                           "b1 HDI": _h(T), "b1 pd": round(T["pd"], 3)})
            if "level" in pcs:
                L = pcs["level"]
                cr.update({"b2 level (pts, shrunk)": round(L["median"], 3),
                           "b2 HDI": _h(L), "b2 pd": round(L["pd"], 3)})
            if "slope" in pcs:
                S3 = pcs["slope"]
                cr.update({"b3 slope (pts/session, shrunk)": round(S3["median"], 3),
                           "b3 HDI": _h(S3), "b3 pd": round(S3["pd"], 3)})
            if "effect_end" in pcs:
                E = pcs["effect_end"]
                cr.update({"Total effect end-B (pts)": round(E["median"], 2),
                           "end-B HDI": _h(E), "end-B pd": round(E["pd"], 3)})
                _mc = mcid.get(oc) if isinstance(mcid, dict) else mcid
                cr["Responder"] = _responder(E["median"], E["pd"], _mc)   # credible + (>=MCID) (#1)
            case_rows.append(cr)
        # collect forest(s) - one per estimand CALIBRATED on the terms (auto) or the explicit estimand (#9)
        _mc = mcid.get(oc) if isinstance(mcid, dict) else mcid
        _bt = _b_terms(r.get("terms")).replace("+", "")     # model tag (b1b3...) to distinguish several models
        _flist = []
        for _fe in _resolve_forest_estimands(forest_estimand, r.get("terms")):
            _fs = _forest_spec(_fe); _ck = _fs["case"]
            fpop = next((r["population"].get(k) for k in _fs["pop"] if r["population"].get(k)), None)
            fl = [(p["case"], p[_ck]) for p in r["per_case"] if _ck in p]
            if fpop and fl:
                hl = {i for i, (_, E) in enumerate(fl) if str(_responder(E["median"], E["pd"], _mc)) == "yes"}
                _flist.append({
                    "labels": [str(c) for c, _ in fl], "meds": [E["median"] for _, E in fl],
                    "los": [E["hdi_low"] for _, E in fl], "his": [E["hdi_high"] for _, E in fl],
                    "pop": (fpop["median"], fpop["hdi_low"], fpop["hdi_high"]), "highlight": hl,
                    "estimand": _fe, "bterms": _bt})
        if _flist:
            forest_data[(g, oc)] = _flist
        models[(g, oc)] = r.get("_idata")
        if pooled_proper and r.get("_idata") is not None:   # Bayesian pooled plot (curve + HDI band), systematic
            try:
                from functions.sced.plots.panels import mbd_bayes_band_draws
                _fitb, _band = mbd_bayes_band_draws(r["_idata"], improvement=improvement)
                if _band:
                    pooled_data[(g, oc)] = {"df": gd, "starts": _starts_of(gd), "fit": _fitb,
                                            "band": _band, "kind": "hier"}
            except Exception:
                pass

    def _meta_one(gd, gtag, oc):
        from functions.sced.mbd.meta import meta_from_idata
        fam = _fam(oc)
        try:
            r = bayes_hier_sced(gd, tier_col=tc, session_col=sc, phase_col=pc, outcome_col=oc,
                                baseline=baseline, treatment=treatment, improvement=improvement,
                                hypothesis=hypothesis, terms=terms, pooling="none", family=fam,
                                bounds=_bounds(oc), n_trials=_ntrials(oc), ar=_ar(oc), rope=rope,
                                draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept,
                                cache_dir=_oc_cache(oc), force_refit=force_refit)
        except Exception as e:
            if verbose:
                print(f"  [skipped] {gtag}/{oc}: {type(e).__name__}: {e}")
            return
        idata = r["_idata"]; g = "(all)" if gtag is None else str(gtag)
        metas = {}
        for est in ("trend", "level", "slope", "effect_end"):
            try:
                metas[est] = meta_from_idata(idata, estimand=est, improvement=improvement,
                                             draws=draws, tune=tune, chains=chains, seed=seed,
                                             target_accept=0.99, progressbar=False)
            except KeyError:
                pass
        if not metas:
            if verbose:
                print(f"  [skipped] {gtag}/{oc}: no meta-analyzable parameter")
            return
        def _hdi(s, p=2): return f"[{s['hdi_low']:.{p}f};{s['hdi_high']:.{p}f}]"
        rmax = max(m["rhat_max"] for m in metas.values())
        dtot = sum(m["diverging"] for m in metas.values())
        flag = "!" if rmax > 1.05 or dtot > 0 else ""
        if verbose:
            lvtxt = (f"end-B(points)={metas['effect_end']['population']['median']:+.2f} "
                     f"pd={metas['effect_end']['population']['pd']:.3f}" if "effect_end" in metas else "(no end-B)")
            print(f"  [{g}/{oc}] (meta/{fam}) {lvtxt} | Rhat={rmax:.3f}{flag}")
        prow = {"Group": g, "Outcome": oc, "Family": fam, "Pooling": "meta", "n_cases": r["n_cases"]}
        if "trend" in metas:
            m = metas["trend"]["population"]
            prow.update({"Baseline trend (b1, pts/session)": round(m["median"], 2),
                         "b1 HDI": _hdi(m), "b1 pd": round(m["pd"], 3)})
        if "level" in metas:
            m = metas["level"]["population"]
            prow.update({"Level effect (b2, pts)": round(m["median"], 3),
                         "b2 HDI": _hdi(m), "b2 pd": round(m["pd"], 3)})
        if "slope" in metas:
            m = metas["slope"]["population"]
            prow.update({"Effect slope (b3, pts/session)": round(m["median"], 2),
                         "b3 HDI": _hdi(m), "b3 pd": round(m["pd"], 3)})
        if "effect_end" in metas:
            ee = metas["effect_end"]; m = ee["population"]; het = ee["heterogeneity"]
            prow.update({"Total effect end-B (b2+b3*T_B, pts)": round(m["median"], 2),
                         "end-B HDI": _hdi(m), "end-B pd": round(m["pd"], 3),
                         "I2 (end-B)": round(het["I2"], 3) if het["I2"] == het["I2"] else np.nan,
                         "tau heterog. (end-B)": round(het["tau"]["median"], 2)})
            pi = het.get("prediction_interval")            # 95 % PI for a NEW case (#8, mu+/-tau)
            if pi:
                prow["end-B PI95 (prediction)"] = f"[{pi[0]:+.2f};{pi[1]:+.2f}]"
            # ROPE (Kruschke 2018) on the meta mu - reuses the ROPE resolved at stage-1
            _rv = (r["population"].get("effect_end") or r["population"].get("level") or {}).get("rope")
            if _rv is not None:
                try:
                    from functions.sced.mbd.bayesian import _rope_stats
                    rs = _rope_stats(ee["_idata"].posterior["mu"].values.ravel(), _rv)
                    prow["% in ROPE"] = round(rs["pct_in_rope"] * 100, 1)
                    prow["ROPE decision"] = rs["rope_decision"]
                except Exception:
                    pass
            _mc = mcid.get(oc) if isinstance(mcid, dict) else mcid
            if _mc is not None:                            # P(cumulative effect > MCID) (#10, clinical utility)
                try:
                    mud = ee["_idata"].posterior["mu"].values.ravel()
                    prow[f"P(end-B>MCID={_mc})"] = round(float((mud > float(_mc)).mean()), 3)
                except Exception:
                    pass
        prow.update({"Rhat": round(rmax, 3),
                     "ESS min": int(r["ess_min"]) if r.get("ess_min") == r.get("ess_min") else np.nan,
                     "diverg.": dtot})
        prow.update(_loo_row(idata))                       # ajustement stage-1 (#6)
        if "phi" in r:                                     # Beta precision (stage-1)
            prow["phi (Beta)"] = round(r["phi"]["median"], 1)
        if "autocorr_ppc_p" in r:                          # autocorrelation PPC (BARG)
            prow["autocorr PPC p"] = r["autocorr_ppc_p"]
        pop_rows.append(prow)
        brut = {pc0["case"]: pc0 for pc0 in r["per_case"]}
        TERMCOL = [("trend", "b1 (pts/session)"), ("level", "b2 (pts)"),
                   ("slope", "b3 (pts/session)"), ("effect_end", "Effect end-B (pts)")]
        HDICOL = {"trend": "b1", "level": "b2", "slope": "b3", "effect_end": "end-B"}
        ref = next(iter(metas.values()))["per_case"]
        for i, pc0 in enumerate(ref):
            case = pc0["case"]; bc = brut.get(case, {})
            cr_b = {"Group": g, "Outcome": oc, "Case": case}
            cr_s = {"Group": g, "Outcome": oc, "Case": case}
            for est, col in TERMCOL:
                if est not in metas:
                    continue
                pre = HDICOL[est]
                b = bc.get(est)
                if b:
                    cr_b.update({col: round(b["median"], 3), f"{pre} HDI": _hdi(b),
                                 f"{pre} pd": round(b["pd"], 3)})
                s = metas[est]["per_case"][i]["shrunk"]
                cr_s.update({col: round(s["median"], 3), f"{pre} HDI": _hdi(s),
                             f"{pre} pd": round(s["pd"], 3)})
            be = bc.get("effect_end")                      # responder on the RAW (uncontaminated, #1)
            if be:
                _mc = mcid.get(oc) if isinstance(mcid, dict) else mcid
                cr_b["Responder"] = _responder(be["median"], be["pd"], _mc)
            brut_rows.append(cr_b)
            case_rows.append(cr_s)
        # collect forest(s) - one per estimand CALIBRATED (auto) or explicit; RAW per case + meta pop (#9)
        _mc = mcid.get(oc) if isinstance(mcid, dict) else mcid
        _bt = _b_terms(r.get("terms")).replace("+", "")
        _flist = []
        for _fe in _resolve_forest_estimands(forest_estimand, r.get("terms")):
            if _fe not in metas:
                continue
            fl = [(pc0["case"], brut.get(pc0["case"], {}).get(_fe)) for pc0 in ref]
            fl = [(c, e) for c, e in fl if e]
            mpop = metas[_fe]["population"]
            if fl:
                hl = {i for i, (_, e) in enumerate(fl) if str(_responder(e["median"], e["pd"], _mc)) == "yes"}
                _flist.append({
                    "labels": [str(c) for c, _ in fl], "meds": [e["median"] for _, e in fl],
                    "los": [e["hdi_low"] for _, e in fl], "his": [e["hdi_high"] for _, e in fl],
                    "pop": (mpop["median"], mpop["hdi_low"], mpop["hdi_high"]), "highlight": hl,
                    "estimand": _fe, "bterms": _bt})
        if _flist:
            forest_data[(g, oc)] = _flist
        models[(g, oc)] = idata
        try:                                                # Bayesian meta pooled plot (mu two-stage), systematic
            sign = 1.0 if improvement == "increase" else -1.0
            _fitb = {"baseline_trend_b1": np.nan, "level_change_b2": np.nan, "slope_change_b3": np.nan}
            _band = {}
            for est, bk, fk in (("trend", "b1", "baseline_trend_b1"), ("level", "b2", "level_change_b2"),
                                ("slope", "b3", "slope_change_b3")):
                if est in metas:
                    mud = sign * metas[est]["_idata"].posterior["mu"].values.ravel()
                    _band[bk] = mud; _fitb[fk] = float(np.mean(mud))
            if _band:
                pooled_data[(g, oc)] = {"df": gd, "starts": _starts_of(gd), "fit": _fitb,
                                        "band": _band, "kind": "meta"}
        except Exception:
            pass

    _run = _meta_one if str(pooling) == "meta" else _one
    for gname, gd in _cohorts():
        gtag = None if (gname is None or gname == "(all)") else gname
        if gname is not None and verbose:
            print(f"######## {'ALL' if gname == '(all)' else gname} (n_cas={gd[tc].nunique()}) ########")
        for oc in outcomes:
            _run(gd, gtag, oc)

    if not case_rows:
        if verbose:
            print("\nAucune analyse.")
        return None
    pop = pd.DataFrame(pop_rows); cas = pd.DataFrame(case_rows)
    pop = pop.dropna(axis=1, how="all")
    cas = cas.dropna(axis=1, how="all")
    if not pop.empty:                                   # order: decision first, diagnostics last
        _front = ["Group", "Outcome", "Family", "Pooling", "n_cases",
                  "Level effect (b2, pts)", "b2 HDI", "b2 pd",
                  "Effect slope (b3, pts/session)", "b3 HDI", "b3 pd",
                  "cumulative effect b3 (pts)", "cumulative HDI", "cumulative pd", "horizon B (sessions)",
                  "Total effect end-B (b2+b3*T_B, pts)", "end-B HDI", "end-B pd",
                  "% in ROPE", "ROPE decision", "end-B PI95 (prediction)",
                  "Baseline trend (b1, pts/session)", "b1 HDI", "b1 pd",
                  "I2 (end-B)", "tau heterog. (b2)", "tau heterog. (end-B)", "rho"]
        _diag = ["phi (Beta)", "autocorr PPC p", "Rhat", "ESS min", "diverg.", "elpd_loo", "p_loo", "k>0.7"]
        _ord = ([c for c in _front if c in pop.columns]
                + [c for c in pop.columns if c not in _front and c not in _diag]   # dynamic (P>MCID, P>ROPE)
                + [c for c in _diag if c in pop.columns])
        pop = pop[_ord]
    cas_sheet = "By case (shrunk)" if pooled_proper else "By case (independent)"

    from functions.sced.glossary import (interpretation_glossary, ascii_sanitize_df, save_idata,
                                         data_recap_df, dedup_pooled_rows, write_stacked, full_param_table,
                                         style_cells, hdi_str_fill, param_table_styles, provenance_df)
    cas = dedup_pooled_rows(cas, "Case")

    def _resp_fill(v):                          # responder: yes -> green; probable/sub-MCID -> amber; no -> grey
        s = str(v).strip().lower()
        if s == "yes":
            return (True, "C6EFCE")
        if s == "no" or not s:
            return (False, "F2F2F2")
        return (True, "FFE699")                 # probable / yes (sub-MCID) / probable (sub-MCID)

    def _rope_dec_fill(v):                      # ROPE decision: effect -> green; undecided -> amber; equivalence -> grey
        s = str(v).strip().lower()
        if s.startswith("effect"):
            return (True, "C6EFCE")
        if s.startswith("undecided"):
            return (True, "FFE699")
        if s.startswith("equivalence"):
            return (False, "F2F2F2")
        return (False, None)

    def _k_fill(v):                             # Pareto-k>0.7: >0 amber (LOO unreliable)
        try:
            return (False, "FFE699") if float(v) > 0 else (False, None)
        except (TypeError, ValueError):
            return (False, None)
    pop_rules = {c: _pd_fill for c in ("b1 pd", "b2 pd", "b3 pd", "cumulative pd", "end-B pd")}
    pop_rules.update({"Rhat": _rhat_fill, "diverg.": _div_fill, "I2 (end-B)": _i2_fill, "k>0.7": _k_fill,
                      "autocorr PPC p": _ppc_fill, "ROPE decision": _rope_dec_fill})
    pop_rules.update({c: hdi_str_fill for c in ("b1 HDI", "b2 HDI", "b3 HDI", "cumulative HDI", "end-B HDI")})
    cas_rules = {c: _pd_fill for c in ("b1 pd", "b2 pd", "b3 pd", "end-B pd", "pd")}
    cas_rules.update({c: hdi_str_fill for c in ("b1 HDI", "b2 HDI", "b3 HDI", "end-B HDI")})
    cas_rules["Responder"] = _resp_fill
    if verbose:
        if pooled_proper:
            print("\n================ POPULATION ================"); print(pop.to_string(index=False))
        else:
            print("\n[pooling=none] no Population row (independent fits) -> meta-analysis (stage 2)")
            print("\n================ BY CASE (independent) ================"); print(cas.to_string(index=False))
    if not (save_path or output_dir):
        return pop if pooled_proper else cas

    fam_tag = family if isinstance(family, str) else "mixed"
    _tmap = {"trend": "b1", "level": "b2", "slope": "b3"}
    _tt = "".join(_tmap[t] for t in ("trend", "level", "slope") if terms and t in terms) or "b2"
    S = ascii_sanitize_df
    _design = ("SCED two-stage Bayesian meta-analysis (group + per case)" if _is_meta
               else "SCED Bayesian hierarchical (population + per-case)")

    def _write_book(out_path, ocs):
        popf = pop[pop["Outcome"].isin(ocs)] if (not pop.empty and "Outcome" in pop.columns) else pop
        casf = cas[cas["Outcome"].isin(ocs)] if "Outcome" in cas.columns else cas
        if casf.empty and popf.empty:
            return
        brutf = [b for b in brut_rows if b.get("Outcome") in ocs]
        modelsf = {k: v for k, v in models.items() if k[1] in ocs}
        recap = data_recap_df(df, tier_col=tc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                              group_col=group_col, baseline=baseline, treatment=treatment, design=_design)
        modele_df = _meta_modele_df() if _is_meta else _modele_priors_df()
        allpar = full_param_table(modelsf)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                write_stacked(w, "Data & model", [("DATA & DESIGN", recap),
                                                  ("MODEL & PRIORS", modele_df),
                                                  ("PROVENANCE", provenance_df(seed=seed))])
                if pooled_proper and not popf.empty:
                    S(popf).to_excel(w, sheet_name="Population", index=False)
                    style_cells(w.sheets["Population"], popf, pop_rules)
                else:
                    note = pd.DataFrame({"Population (not applicable)": [
                        "pooling='none': independent per-case fits -- NO pooled estimation.",
                        "mu (population effect), tau (heterogeneity) and I2 are obtained via a",
                        "META-ANALYSIS (stage 2): functions/SCED_mbd_meta.bayes_meta_analysis(...)",
                        "applied to the effects of the 'By case (independent)' sheet."]})
                    S(note).to_excel(w, sheet_name="Population", index=False)
                if brutf:
                    brutdf = pd.DataFrame(brutf)
                    S(brutdf).to_excel(w, sheet_name="By case (raw)", index=False)
                    style_cells(w.sheets["By case (raw)"], brutdf, cas_rules)
                S(casf).to_excel(w, sheet_name=cas_sheet, index=False)
                style_cells(w.sheets[cas_sheet], casf, cas_rules)
                if not allpar.empty:
                    _pr, _prr = param_table_styles()
                    write_stacked(w, "Parameters (full)",
                                  [("PARAMETERS LEGEND", _param_glossary()),
                                   ("FULL TABLE (mean / sd / HDI / ess / r_hat)", allpar)],
                                  styles=[None, {"rules": _pr, "row_rules": _prr}])
                write_stacked(w, "Families & interpretation",
                              [("FAMILIES (guide)", _familles_guide_df()),
                               ("VALUE APPRAISAL", _bayes_appreciation()),
                               ("INTERPRETATION GUIDE", interpretation_glossary("bayes"))])
            if verbose:
                print(f"\nReport written: {out_path}")
        except Exception as e:
            if verbose:
                print(f"\n[not written] {type(e).__name__}: {e}")

    def _emit_models(models_dir, ocs):
        if cache_dir:                                       # already persisted by the engine (shared cache)
            if verbose:
                print(f"Models in shared cache: {cache_dir}/")
            return
        nsaved = sum(save_idata(i, models_dir, f"{_gtok(g)}__{oc}__{_tt}") is not None
                     for (g, oc), i in models.items() if oc in ocs)
        if nsaved and verbose:
            print(f"Models saved: {nsaved} -> {models_dir}/")

    def _emit_forests(forest_dir, ocs):                     # one forest/model in Plot/forest/<bN...>/ (#9)
        if not forest_data:
            return
        from functions.sced.plots.panels import plot_case_forest
        import matplotlib.pyplot as _plt
        _kind = "meta" if _is_meta else "hierarchical"
        for (g, oc), flist in forest_data.items():
            if oc not in ocs:
                continue
            for fdat in flist:
                _fs = _forest_spec(fdat.get("estimand", "effect_end"))
                _bt = fdat.get("bterms") or ""
                _sub = os.path.join(forest_dir, _bt) if _bt else forest_dir
                try:
                    plot_case_forest(fdat["labels"], fdat["meds"], fdat["los"], fdat["his"],
                                     pop=fdat["pop"], highlight=fdat["highlight"], save_path=_sub,
                                     fname=f"{oc}__{_gtok(g)}__forest_{_fs['tag']}", unit=_fs["unit"],
                                     title=f"Forest {_fs['label']} - {oc} [{g}] ({_bt or _kind})", style=style)
                    _plt.close("all")
                except Exception as e:
                    if verbose:
                        print(f"  [forest skipped] {oc}/{g}/{fdat.get('estimand')}: {type(e).__name__}: {e}")

    def _emit_pooled(pooled_dir, ocs):                      # systematic Bayesian pooled plot (hier & meta)
        if not pooled_data:
            return
        from functions.sced.plots.panels import plot_mbd_pooled_fit
        import matplotlib.pyplot as _plt
        for (g, oc), pdat in pooled_data.items():
            if oc not in ocs:
                continue
            try:
                plot_mbd_pooled_fit(pdat["df"], tier_col=tc, session_col=sc, outcome_col=oc,
                                    starts=pdat["starts"], fit=pdat["fit"], band_draws=pdat["band"],
                                    per_case=True, save_path=pooled_dir, style=style,
                                    fname=f"{oc}__{_gtok(g)}_mbd_pooled_bayesien_{pdat['kind']}_{_tt}",
                                    title=pooled_title or f"MBD pooled - {oc} (Bayesian {pdat['kind']} {_tt}, HDI 95%)")
                _plt.close("all")
            except Exception as e:
                if verbose:
                    print(f"  [Bayesian pooled skipped] {oc}/{g}: {type(e).__name__}: {e}")

    if output_dir is not None:                              # per-OUTCOME layout: <oc>/Analyse/bayes/<fam>/<bN...>/
        for oc in outcomes:
            Lf = std_layout(output_dir, oc, analysis)        # models/diagnostics at family level (shared)
            ad = os.path.join(Lf["analyse"], _tt)
            _write_book(os.path.join(ad, f"bayes_{fam_tag}_{_tt}_summary.xlsx"), [oc])
            # .nc already persisted by the engine (fit-or-load) in Lf["models"] via _oc_cache
            if diagnostics:
                _emit_diagnostics({k: v for k, v in models.items() if k[1] == oc}, Lf["analyse"], style=style, verbose=verbose)
            _emit_forests(Lf["forest"], [oc])
            _emit_pooled(Lf["pooled"], [oc])
    else:                                                   # advanced mode: single consolidated folder
        os.makedirs(save_path, exist_ok=True)
        _otag = _gtok(outcomes[0]) if len(outcomes) == 1 else "multi"
        _write_book(os.path.join(save_path, f"bayes_{_otag}_{pooling}_{fam_tag}_{_tt}_summary.xlsx"), list(outcomes))
        _emit_models(os.path.join(save_path, "models"), list(outcomes))
        if diagnostics:
            _emit_diagnostics(models, save_path, style=style, verbose=verbose)
        _emit_forests(forest_path or os.path.join(save_path, "plots"), list(outcomes))
        _emit_pooled(os.path.join(save_path, "plots"), list(outcomes))
    return pop if pooled_proper else cas


def _bayes_run_per_case(df, tc, sc, pc, group_col, outcomes, baseline, treatment, _cohorts, *,
                        improvement, rope, model, trend, baseline_trend, min_per_phase, family,
                        draws, tune, chains, seed, save_path, diagnostics=False,
                        cache_dir=None, force_refit=False, output_dir=None, analysis=None,
                        style=None, verbose=True):
    """Run the per-case Bayesian SCED engine (one independent fit per case) over cohorts x outcomes
    and write the styled summary. Inputs: tidy ``df`` + resolved column names + model/fit options;
    output: the per-case summary DataFrame. Side effects: writes the .xlsx summary, .nc models, and
    diagnostic figures when ``save_path`` or ``output_dir`` is set.

    References: Natesan & Hedges 2017 (BITS / BUCP change-point); Kruschke 2018 (ROPE).
    R equivalent: brms (simple / BITS); BUCP has no turnkey R equivalent (Stan / JAGS) -
    potential brms change-point, to test.
    """
    import os
    from functions.sced.mbd.bayesian import bayes_phase_model
    from functions.sced.glossary import std_layout
    if verbose:
        print(f"\nMODE=per_case : {model.upper()} | MCMC {chains}x{draws} | ROPE es>{rope}\n")
    rows = []; models = {}

    def _one_case(sub_case, gtag, outcome, tier):
        d = sub_case[sub_case[pc].astype(str).isin([str(baseline), str(treatment)])].sort_values(sc)
        y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(); ph = d[pc].astype(str).to_numpy()
        m = np.isfinite(y); y, ph = y[m], ph[m]
        n_a = int(np.sum(ph == str(baseline))); n_b = int(np.sum(ph == str(treatment)))
        if n_a < min_per_phase or n_b < min_per_phase:
            if verbose:
                print(f"  [skipped] {tier}/{outcome}: phases too short (A={n_a}, B={n_b})")
            return
        start_idx = int(np.argmax(ph == str(treatment)))
        try:
            r = bayes_phase_model(y, start_index=start_idx, model=model, improvement=improvement,
                                  rope=rope, trend=trend, baseline_trend=baseline_trend,
                                  draws=draws, tune=tune, chains=chains, seed=seed,
                                  cache_dir=cache_dir, force_refit=force_refit)
        except Exception as e:
            if verbose:
                print(f"  [erreur] {tier}/{outcome}: {type(e).__name__}: {e}")
            return
        es = r["es"]; flag = "!" if r["rhat_max"] > 1.05 or r["diverging"] > 0 else ""
        if verbose:
            print(f"  {str(tier):>10s}/{outcome:<12s} es={es['median']:+.2f} "
                  f"[{es['hdi_low']:+.2f};{es['hdi_high']:+.2f}] pd={es['pd']:.3f} "
                  f"P(es>{rope})={r['p_es_gt_rope']:.2f} Rhat={r['rhat_max']:.3f}{flag}")
        row = {"Group": "(all)" if gtag is None else str(gtag), "Case": tier, "Outcome": outcome,
               "n_A": n_a, "n_B": n_b, "es": round(es["median"], 3),
               "es HDI": f"[{es['hdi_low']:.2f};{es['hdi_high']:.2f}]", "pd": round(es["pd"], 3),
               f"P(es>{rope})": round(r["p_es_gt_rope"], 3),
               "Delta level": round(r["level_change"]["median"], 2), "rho": round(r["rho"]["median"], 3),
               "Rhat": round(r["rhat_max"], 3), "ESS min": round(r["ess_min"], 0), "diverg.": r["diverging"],
               "Baseline trend p": r.get("baseline_trend_p"), "Trend corrected": r.get("baseline_corrected")}
        if trend:
            ee, sl = r["es_end"], r["slope_change"]
            row.update({"es_end (cumulative)": round(ee["median"], 3),
                        "es_end HDI": f"[{ee['hdi_low']:.2f};{ee['hdi_high']:.2f}]",
                        "es_end pd": round(ee["pd"], 3), f"P(es_end>{rope})": round(r["p_es_end_gt_rope"], 3),
                        "slope b3": round(sl["median"], 3), "b3 pd": round(sl["pd"], 3)})
        if model == "bucp":
            row.update({"CP->B (mode)": r["cp"]["start_B_mode"],
                        "CP concentration": round(r["cp"]["concentration"], 2)})
        rows.append(row); models[(row["Group"], outcome, str(tier))] = r.get("_idata")

    for gname, gd in _cohorts():
        gtag = None if (gname is None or gname == "(all)") else gname
        if gname is not None and verbose:
            print(f"\n######## {'ALL' if gname == '(all)' else str(gname)} (n_cas={gd[tc].nunique()}) ########")
        for oc in outcomes:
            if (len(outcomes) > 1 or gname is not None) and verbose:
                print(f"-- outcome : {oc} --")
            for tier in sorted(gd[tc].dropna().unique().tolist(), key=str):
                _one_case(gd[gd[tc] == tier], gtag, oc, tier)

    if not rows:
        if verbose:
            print("\nNo analyzable case.")
        return None
    summary = pd.DataFrame(rows)
    if verbose:
        print("\n================  SUMMARY (per-case)  ================")
        print(summary.to_string(index=False))
    from functions.sced.glossary import (interpretation_glossary, ascii_sanitize_df, dedup_pooled_rows,
                                         save_idata, data_recap_df, write_stacked, full_param_table,
                                         style_cells, param_table_styles, provenance_df)
    summary = dedup_pooled_rows(summary, "Case")
    if not (save_path or output_dir):
        return summary
    fam_tag = family if isinstance(family, str) else "mixed"
    analysis = analysis or "bayes/per_case"
    S = ascii_sanitize_df
    pd_cols = [c for c in summary.columns if c == "pd" or c.endswith(" pd")]
    sum_rules = {c: _pd_fill for c in pd_cols}
    sum_rules.update({"Rhat": _rhat_fill, "diverg.": _div_fill})
    _mtag = f"{model}{'+trend' if trend else ''}"

    def _write_book(out_path, ocs):
        sumf = summary[summary["Outcome"].isin(ocs)] if "Outcome" in summary.columns else summary
        if sumf.empty:
            return
        modelsf = {k: v for k, v in models.items() if k[1] in ocs}
        recap = data_recap_df(df, tier_col=tc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                              group_col=group_col, baseline=baseline, treatment=treatment,
                              design=f"SCED per-case Bayesian ({model.upper()}{'+trend' if trend else ''})")
        allpar = full_param_table(modelsf)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                write_stacked(w, "Data & design", [("DATA & DESIGN", recap),
                                                   ("PROVENANCE", provenance_df(seed=seed))])
                S(sumf).to_excel(w, sheet_name="Summary", index=False)
                style_cells(w.sheets["Summary"], sumf, sum_rules)
                if not allpar.empty:
                    _pr, _prr = param_table_styles()
                    write_stacked(w, "Parameters (full)",
                                  [("PARAMETERS LEGEND", _param_glossary()),
                                   ("FULL TABLE (mean / sd / HDI / ess / r_hat)", allpar)],
                                  styles=[None, {"rules": _pr, "row_rules": _prr}])
                write_stacked(w, "Glossary & interpretation",
                              [("GLOSSARY", _bayes_glossary()),
                               ("VALUE APPRAISAL", _bayes_appreciation()),
                               ("INTERPRETATION GUIDE", interpretation_glossary("bayes"))])
            if verbose:
                print(f"\nSummary written: {out_path}")
        except Exception as e:
            if verbose:
                print(f"\n[not written] {type(e).__name__}: {e}")

    def _emit_models(models_dir, ocs):
        if cache_dir:
            if verbose:
                print(f"Models in shared cache: {cache_dir}/")
            return
        nsaved = sum(save_idata(i, models_dir, f"{_gtok(g)}__{oc}__{_gtok(t)}") is not None
                     for (g, oc, t), i in models.items() if oc in ocs)
        if nsaved and verbose:
            print(f"Models saved: {nsaved} -> {models_dir}/")

    if output_dir is not None:
        for oc in outcomes:
            Lf = std_layout(output_dir, oc, analysis)
            ad = os.path.join(Lf["analyse"], _mtag)
            _write_book(os.path.join(ad, f"bayes_per-case_{_mtag}_{fam_tag}_summary.xlsx"), [oc])
            _emit_models(Lf["models"], [oc])
            if diagnostics:
                _emit_diagnostics({k: v for k, v in models.items() if k[1] == oc}, Lf["analyse"], style=style, verbose=verbose)
    else:
        os.makedirs(save_path, exist_ok=True)
        _otag = _gtok(outcomes[0]) if len(outcomes) == 1 else "multi"
        _write_book(os.path.join(save_path, f"bayes_{_otag}_per-case_{_mtag}_{fam_tag}_summary.xlsx"), list(outcomes))
        _emit_models(os.path.join(save_path, "models"), list(outcomes))
        if diagnostics:
            _emit_diagnostics(models, save_path, style=style, verbose=verbose)
    return summary


def _percase_glossary():
    """Glossary of the 'per-patient selection' sheet (ASCII)."""
    g = [
        ("Patient", "Analyzed case (INDEPENDENT beta regression per patient; pooling='none')."),
        ("Best model", "Mean structure best ranked PER PATIENT at PSIS-LOO (highest elpd_loo): M0 trend (trend only = no effect); Mi trend+level (immediate jump); Mg trend+slope (gradual/delayed effect); Mf trend+level+slope (both)."),
        ("Terms (b)", "b1 = baseline trend; b2 = level jump; b3 = slope change."),
        ("elpd_loo", "Cross-validated predictive accuracy (PSIS-LOO; Vehtari et al., 2017) RESTRICTED to the patient's observations. Higher = better. Valid per patient because pooling='none' makes its parameters independent."),
        ("elpd_diff", "elpd gap to the patient's best model (0 = best)."),
        ("Decisive", "TRUE if the elpd gap to the 2nd exceeds se_mult x dse (otherwise indistinguishable models -> keep the simplest)."),
        ("k_worst (patient)", "Worst Pareto-k index (PSIS) among the patient's observations under its best model; > 0.7 = LOO unreliable for that point."),
        ("rhat_max / ess_min / divergences (model)", "MCMC diagnostics of the selected model (at the fit level, all patients): rhat ~ 1.00, high ESS, 0 divergence = healthy convergence."),
        ("b1 / b2 / b3 (pts)", "Patient estimands under ITS best model, in POINTS of the original scale (beta, logit link): b1 = baseline trend (pts/session); b2 = level jump (pts); b3 = slope change (pts/session). Each: median, HDI95, pd."),
        ("end effect (pts)", "Total TREATMENT effect at the end of phase B = b2 + b3 x duration(B). Unified clinical estimand (median, HDI95, pd)."),
        ("pd", "Probability of direction (share of the posterior with the dominant sign): >= 0.95 credible."),
        ("HDI", "Highest-density interval at 95% (points scale)."),
        ("ROPE (pts) / P(end effect>ROPE) / ROPE decision", "ROPE = region of practical equivalence (MCID), by default 0.1 x SD(outcome) in points. Applied ONLY to the end effect (the treatment effect): P that the end effect exceeds the ROPE + HDI vs ROPE verdict (Kruschke 2018: effect / equivalence / undecided). n/a for a model without effect (M0)."),
    ]
    return pd.DataFrame(g, columns=["Term", "Definition"])


def _percase_references():
    """Bibliography (sheet 'References') of the per-patient Bayesian report - verified DOIs."""
    refs = [
        ("Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing, 27(5), 1413-1432.", "10.1007/s11222-016-9696-4"),
        ("Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. Bayesian Analysis, 13(3), 917-1007.", "10.1214/17-BA1091"),
        ("Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion. JMLR, 11, 3571-3594.", " - "),
        ("Smithson, M., & Verkuilen, J. (2006). A better lemon squeezer? Maximum-likelihood regression with beta-distributed dependent variables. Psychological Methods, 11(1), 54-71.", "10.1037/1082-989X.11.1.54"),
        ("Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. Advances in Methods and Practices in Psychological Science, 1(2), 270-280.", "10.1177/2515245918771304"),
    ]
    return pd.DataFrame(refs, columns=["Reference", "DOI"])


def report_sced_bayesian_percase(
        df, *, tier_col="tier", session_col="session", phase_col="phase",
        outcome_name_col=None, value_col=None, outcomes=("accuracy",),
        case_insensitive=True, date_col=None, date_dayfirst=False,
        phases=("A", "B"), baseline_phase=None, treatment_phase=None,
        improvement="increase", family="beta", bounds=(0, 100), n_trials=None,
        models=None, rope="auto", se_mult=2.0,
        draws=2000, tune=2000, chains=4, seed=42, target_accept=0.99,
        save_path=None, output_dir=None, diagnostics=False, cache_dir=None,
        force_refit=False, style=None, verbose=True):
    """Per-PATIENT Bayesian report with per-patient MODEL SELECTION (PSIS-LOO).

    Reuses the existing pipeline: each candidate model is fitted once via
    ``select_percase_models`` (-> ``bayes_hier_sced`` with ``pooling='none'``, independent
    per-case beta regressions), which ranks the models PER PATIENT at LOO. The report only
    ASSEMBLES: for each patient, the best model, its diagnostics, and the estimands
    (already computed by ``bayes_hier_sced`` in ``per_case``). Sheets: Data & design,
    Best model, Estimands, Model comparison, Glossary, References.
    Writes ``bayesian_percase_summary.xlsx``. Returns the 'best model' DataFrame (or None).

    References: Vehtari, Gelman & Gabry 2017 (PSIS-LOO per patient); Yao et al. 2018
    (stacking); Kruschke 2018 (ROPE).
    R equivalent: loo::loo_compare / loo_model_weights; brms per-case.
    """
    from functions.sced.mbd.bayesian import select_percase_models, _rope_stats, _DEFAULT_MODELS
    from functions.sced.glossary import (ascii_sanitize_df, data_recap_df, interpretation_glossary,
                                         style_cells, write_stacked, provenance_df, std_layout)
    _analysis = "bayes/percase"
    tc, sc, pc = tier_col, session_col, phase_col
    on, vc = outcome_name_col, value_col
    baseline = baseline_phase if baseline_phase is not None else phases[0]
    treatment = treatment_phase if treatment_phase is not None else phases[-1]

    if case_insensitive:
        from functions.sced.prep import resolve_columns, harmonize_conditions
        r = resolve_columns(df, {"tier": tc, "session": sc, "phase": pc, "date": date_col,
                                 "outcome_name": on, "value": vc})
        tc, sc, pc, date_col, on, vc = (r["tier"], r["session"], r["phase"], r["date"],
                                        r["outcome_name"], r["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=phases)
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=tc, new_col=sc, dayfirst=date_dayfirst)
    outcomes = list(outcomes) if outcomes else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=[tc, sc, pc])
        outcomes = outcomes or names
    if not outcomes:
        if verbose:
            print("No outcome.")
        return None

    models = models or _DEFAULT_MODELS
    name2b = _b_map(models)
    _fam = lambda oc: (family.get(oc, "beta") if isinstance(family, dict) else family)
    _bnd = lambda oc: (bounds.get(oc) if isinstance(bounds, dict) else bounds)
    _ntr = lambda oc: (n_trials.get(oc) if isinstance(n_trials, dict) else n_trials)

    best_rows, est_rows, cmp_tables, idatas = [], [], [], {}
    for oc in outcomes:
        if df[tc].nunique() < 2:
            if verbose:
                print(f"[skipped] {oc}: <2 cases (pooling='none' requires >=2).")
            continue
        _cd = cache_dir or (std_layout(output_dir, oc, _analysis)["models"]
                            if output_dir is not None else None)
        # Default ROPE = 0.1 x SD(outcome) in POINTS (Kruschke/bayestestR convention);
        # resolved numerically here so the engine AND the per-patient computation use the SAME value.
        _y = pd.to_numeric(df[oc], errors="coerce").dropna().to_numpy()
        rope_oc = (round(0.1 * float(np.std(_y)), 4)
                   if (isinstance(rope, str) and rope == "auto") else float(rope))
        try:
            res = select_percase_models(
                df, tier_col=tc, session_col=sc, phase_col=pc, outcome_col=oc, models=models,
                baseline=baseline, treatment=treatment, improvement=improvement, family=_fam(oc),
                bounds=_bnd(oc), n_trials=_ntr(oc), rope=rope_oc, draws=draws, tune=tune,
                chains=chains, seed=seed, target_accept=target_accept, se_mult=se_mult,
                cache_dir=_cd, force_refit=force_refit)
        except Exception as e:
            if verbose:
                print(f"[skipped] {oc}: {type(e).__name__}: {e}")
            continue
        for name, rr in res["fitted"].items():
            idatas[(oc, name)] = rr.get("_idata")
        for ci, case in enumerate(res["cases"]):
            pp = res["per_patient"][case]
            best = pp["best"]; br = res["fitted"][best]
            sub = df[df[tc].astype(str) == str(case)]
            nA = int((sub[pc].astype(str) == str(baseline)).sum())
            nB = int((sub[pc].astype(str) == str(treatment)).sum())
            kworst = float(pp["table"].set_index("model").loc[best, "k_worst"])
            best_rows.append({
                "Outcome": oc, "Patient": case, "Best model": best,
                "Terms (b)": name2b.get(best), "Decisive": pp["decisive"],
                "elpd delta (vs 2nd)": (pp["table"].loc[1, "elpd_diff"] if len(pp["table"]) > 1 else None),
                "rhat_max (model)": round(br["rhat_max"], 3), "ess_min (model)": int(br["ess_min"]),
                "divergences (model)": int(br["diverging"]), "k_worst (patient)": round(kworst, 2),
                "n_A": nA, "n_B": nB, "Decision": pp["decision"]})
            # estimands: PICK from the best model's per_case (already computed)
            pce = next((e for e in br["per_case"] if str(e["case"]) == str(case)), {})
            _m = lambda k, f="median": (round(pce[k][f], 3) if k in pce else None)
            _hd = lambda k: (f"[{pce[k]['hdi_low']:.2f};{pce[k]['hdi_high']:.2f}]" if k in pce else None)
            post = br["_idata"].posterior
            # ROPE ONLY on the end effect (b2 + b3*T = treatment effect); n/a if model
            # without effect (M0, no b2/b3 -> no effect_end).
            p_gt = rdec = None
            if "effect_end" in pce and "effect_end_pts" in post:
                rs = _rope_stats(np.asarray(post["effect_end_pts"].values)[..., ci].ravel(), rope_oc)
                p_gt, rdec = rs["p_gt_rope"], rs["rope_decision"]
            est_rows.append({
                "Outcome": oc, "Patient": case, "Best model": best,
                "b1 (trend, pts/session)": _m("trend"), "b1 HDI": _hd("trend"), "b1 pd": _m("trend", "pd"),
                "b2 (jump, pts)": _m("level"), "b2 HDI": _hd("level"), "b2 pd": _m("level", "pd"),
                "b3 (slope, pts/session)": _m("slope"), "b3 HDI": _hd("slope"), "b3 pd": _m("slope", "pd"),
                "end effect (pts)": _m("effect_end"), "end effect HDI": _hd("effect_end"),
                "end effect pd": _m("effect_end", "pd"),
                "ROPE (pts)": rope_oc, "P(end effect>ROPE)": p_gt, "ROPE decision (end effect)": rdec})
            t = pp["table"].copy()
            t.insert(0, "Patient", case); t.insert(0, "Outcome", oc)
            t["Terms (b)"] = t["model"].map(name2b)
            cmp_tables.append(t)

    if not best_rows:
        if verbose:
            print("\nNo patient analyzed.")
        return None
    best_df = pd.DataFrame(best_rows)
    est_df = pd.DataFrame(est_rows)
    cmp_df = pd.concat(cmp_tables, ignore_index=True) if cmp_tables else pd.DataFrame()
    if verbose:
        print("\n======== BEST MODEL PER PATIENT ========")
        print(best_df[["Outcome", "Patient", "Best model", "Decisive", "elpd delta (vs 2nd)",
                       "rhat_max (model)", "divergences (model)"]].to_string(index=False))

    if not (save_path or output_dir):
        return best_df
    S = ascii_sanitize_df
    best_rules = {"Decisive": _bool_fill}
    est_rules = {"b1 pd": _pd_fill, "b2 pd": _pd_fill, "b3 pd": _pd_fill,
                 "end effect pd": _pd_fill}

    def _write_book(out_path, ocs):
        bf = best_df[best_df["Outcome"].isin(ocs)]
        ef = est_df[est_df["Outcome"].isin(ocs)]
        cf = cmp_df[cmp_df["Outcome"].isin(ocs)] if len(cmp_df) else cmp_df
        if bf.empty:
            return
        recap = data_recap_df(df, tier_col=tc, session_col=sc, phase_col=pc, outcomes=list(ocs),
                              baseline=baseline, treatment=treatment,
                              design="Per-PATIENT Bayesian + model selection (PSIS-LOO)",
                              extra=[("Family", family), ("Bounds", bounds),
                                     ("Models", ", ".join(models.keys())),
                                     ("MCMC", f"{chains}x{draws} (warmup {tune}), target_accept={target_accept}"),
                                     ("Selection", f"pooling='none'; LOO per patient; decisive if elpd delta > {se_mult} x dse")])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as w:
                write_stacked(w, "Data & design", [("DATA & DESIGN", recap),
                                                   ("PROVENANCE", provenance_df(seed=seed))])
                S(bf).to_excel(w, sheet_name="Best model", index=False)
                style_cells(w.sheets["Best model"], bf, best_rules)
                S(ef).to_excel(w, sheet_name="Estimands", index=False)
                style_cells(w.sheets["Estimands"], ef, est_rules)
                S(cf).to_excel(w, sheet_name="Model comparison", index=False)
                S(_percase_glossary()).to_excel(w, sheet_name="Glossary", index=False)
                S(interpretation_glossary("bayes")).to_excel(w, sheet_name="Interpretation guide", index=False)
                S(_percase_references()).to_excel(w, sheet_name="References", index=False)
            if verbose:
                print(f"\nReport written: {out_path}")
        except Exception as e:
            if verbose:
                print(f"\n[not written] {type(e).__name__}: {e}")

    if output_dir is not None:
        for oc in outcomes:
            ad = std_layout(output_dir, oc, _analysis)["analyse"]
            _write_book(os.path.join(ad, "bayesian_percase_summary.xlsx"), [oc])
    else:
        os.makedirs(save_path, exist_ok=True)
        _write_book(os.path.join(save_path, "bayesian_percase_summary.xlsx"), list(outcomes))

    if diagnostics:                                   # MCMC diagnostics per model, PER OUTCOME
        for oc in outcomes:
            oc_id = {k: v for k, v in idatas.items() if k[0] == oc}
            if not oc_id:
                continue
            root = save_path if save_path else std_layout(output_dir, oc, _analysis)["analyse"]
            _emit_diagnostics(oc_id, root, style=style, verbose=verbose)
    return best_df
