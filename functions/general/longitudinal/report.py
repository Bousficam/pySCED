"""Longitudinal multilevel COUPLING report (Phase 1) - report-level orchestration.

``report_longitudinal_coupling`` chains rmcorr + slope-vs-slope + frequentist mixed model
(+ Satterthwaite/KR df via R) + Bayesian mixed model (gaussian + beta), a collinearity check
(VIF of the temporal confounder), writes two styled .xlsx workbooks (standard per-outcome
tree: ``<outcome>/Analyse/{Inferentielle,Bayes}/``) and three figures (``<outcome>/Plot/``).
Target: EFFECT ESTIMATION (estimates + CI/HDI + ROPE/pd), not group NHST.
"""
import os
import numpy as np
import pandas as pd

from functions.general.longitudinal.growth import rmcorr, slope_vs_slope, fit_growth_lmm, fit_growth_bayes, vif_table


def report_longitudinal_coupling(
        df, *, outcome, x, time, group, group_l3=None, cat=None,
        rmcorr_subject=None, rope="auto", run_bayes=True, bayes_families=("gaussian", "beta"),
        bayes_bounds=None, draws=2000, tune=2000, chains=4, seed=42, cache_dir=None,
        output_dir=None, save_path=None, style=None, verbose=True):
    """Within-cluster coupling of ``x`` (focal time-varying covariate, e.g. fatigue)
    on ``outcome`` (e.g. accuracy), at equal time (``time``/block as covariate ``cat``).

    ``group`` = random intercept (session); ``group_l3`` = level-3 (patient, FIXED if <6;
    random on the Bayesian side). ``rmcorr_subject`` = rmcorr cluster (default = ``group``).
    Returns a dict of results. Writes the reports if ``output_dir``/``save_path``.

    References: Bakdash & Marusich 2017 (rmcorr); Laird & Ware 1982 (LMM) with
    Satterthwaite 1946 / Kenward & Roger 1997 df; Gelman & Hill 2007 (Bayesian arm).
    R equivalent: no single equivalent - rmcorr::rmcorr + lme4::lmer/lmerTest + brms::brm."""
    from functions.sced.glossary import (std_layout, ascii_sanitize_df, write_stacked, style_cells,
                                          data_recap_df, provenance_df, interpretation_glossary)
    cat = list(cat or [])
    rmcorr_subject = rmcorr_subject or group
    S = ascii_sanitize_df
    res = {}

    # --- 1. VIF of the temporal confounder (x ~ time) ---
    vif = vif_table(df, [x, time])
    vif = vif[vif["Variable"] != "const"]                   # const = statsmodels artefact
    res["vif"] = vif

    # --- 2. rmcorr (cluster = session) ---
    res["rmcorr"] = rmcorr(df, x=x, y=outcome, subject=rmcorr_subject)

    # --- 3. slope-vs-slope ---
    res["slope_vs_slope"] = slope_vs_slope(df, x=x, y=outcome, time=time, cluster=group)

    # --- 4. frequentist mixed model ---
    fixed = [x] + [f"C({c})" for c in cat]
    _, res["lmm"] = fit_growth_lmm(df, outcome=outcome, fixed=fixed, group=group,
                                   group_l3=group_l3, vif_terms=[x, time])

    # --- 5. Bayesian mixed model (gaussian + beta) ---
    res["bayes"] = {}
    if run_bayes:
        for fam in bayes_families:
            try:
                res["bayes"][fam] = fit_growth_bayes(
                    df, outcome=outcome, x=x, cat=cat, group=group, group_l3=group_l3,
                    family=fam, bounds=bayes_bounds, rope=rope, draws=draws, tune=tune,
                    chains=chains, seed=seed, cache_dir=cache_dir, progressbar=False)
            except Exception as e:
                if verbose:
                    print(f"  [bayes {fam} skipped] {type(e).__name__}: {e}")

    if not (output_dir or save_path):
        return res

    # ---------- tables for writing ----------
    rc = res["rmcorr"]
    rc_df = pd.DataFrame([{"r_rm": rc.get("r_rm"), "CI95": str(rc.get("ci95")), "p": rc.get("p"),
                           "slope (y/x)": rc.get("slope"), "df": rc.get("df"),
                           "n_obs": rc.get("n_obs"), "k_clusters": rc.get("k_subjects"),
                           "Note": "ignores the patient nesting (primary estimate)"}]) \
        if rc.get("available") else pd.DataFrame([{"Note": rc.get("message")}])

    sv = res["slope_vs_slope"]
    sv_df = pd.DataFrame([{"r_slopes": sv.get("r_slopes"), "CI95": str(sv.get("ci95")),
                           "p": sv.get("p"), "var slope x": sv.get("var_slope_x"),
                           "mean slope x": sv.get("mean_slope_x"), "mean slope y": sv.get("mean_slope_y"),
                           "t(slope x=0) p": (sv.get("t_slope_x") or {}).get("p"),
                           "t(slope y=0) p": (sv.get("t_slope_y") or {}).get("p"),
                           "n_clusters": sv.get("n_clusters"), "Warning": sv.get("warn", "")}])

    lmm = res["lmm"]
    lmm_df = pd.DataFrame(lmm.get("fixed", []))
    kr = lmm.get("KR") or {}
    if kr.get("available"):
        kmap = {k: v for k, v in (kr.get("coef") or {}).items()}
        lmm_df["df (" + str(kr.get("method", "KR")) + ")"] = lmm_df["term"].map(
            lambda t: round(kmap.get(t, {}).get("df", np.nan), 1) if t in kmap else np.nan)
        lmm_df["p (" + str(kr.get("method", "KR")) + ")"] = lmm_df["term"].map(
            lambda t: round(kmap.get(t, {}).get("p", np.nan), 5) if t in kmap else np.nan)
    lmm_meta = pd.DataFrame([("Formula", lmm.get("Formula")), ("Random", lmm.get("Random")),
                             ("ICC", lmm.get("ICC")), ("R2 marginal", lmm.get("R2_marginal")),
                             ("R2 conditional", lmm.get("R2_conditional")),
                             ("df model-based", kr.get("method") if kr.get("available")
                              else f"not available ({kr.get('message', 'R missing')})")],
                            columns=["Element", "Value"])

    recap = data_recap_df(df, tier_col=group_l3 or group, session_col=group, outcomes=[outcome],
                          design=f"Longitudinal multilevel coupling: {outcome} ~ {x} + C({time}) "
                                 f"+ (1|{group})" + (f', C({group_l3}) fixed' if group_l3 else ""))

    # ---------- Inferential writing ----------
    def _paths(analysis):
        if output_dir is not None:
            L = std_layout(output_dir, outcome, analysis)
            return L["analyse"], L["plot"]
        return save_path, save_path

    inf_dir, plot_dir = _paths("Inferentielle")
    if inf_dir:
        os.makedirs(inf_dir, exist_ok=True)
        out = os.path.join(inf_dir, "coupling_summary.xlsx")
        try:
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                write_stacked(w, "Data & design", [("DATA & DESIGN", recap),
                                                   ("COLLINEARITY (VIF of the temporal confounder)", vif),
                                                   ("PROVENANCE", provenance_df(seed=seed))])
                S(rc_df).to_excel(w, sheet_name="rmcorr", index=False)
                S(sv_df).to_excel(w, sheet_name="Slope-vs-slope", index=False)
                write_stacked(w, "Frequentist mixed", [("FIXED EFFECTS", lmm_df),
                                                       ("MODEL & VARIANCE", lmm_meta)])
                S(interpretation_glossary("all")).to_excel(w, sheet_name="Interpretation guide", index=False)
            if verbose:
                print(f"Inferential report written: {out}")
        except Exception as e:
            if verbose:
                print(f"[not written] {type(e).__name__}: {e}")

    # ---------- Bayes writing ----------
    if run_bayes and res["bayes"]:
        bdir, _ = _paths("Bayes")
        if bdir:
            os.makedirs(bdir, exist_ok=True)
            brows = []
            for fam, r in res["bayes"].items():
                b = r["b_x"]
                brows.append({"family": fam, "scale": r["scale"],
                              "b_x median": round(b["median"], 5),
                              "HDI95": (round(b["hdi_low"], 5), round(b["hdi_high"], 5)),
                              "pd": round(b["pd"], 3), "ROPE": b.get("rope"),
                              "% in ROPE": b.get("pct_in_rope"), "decision": b.get("rope_decision", ""),
                              "tau_session": round(r["variance"]["tau_session"]["median"], 4),
                              "tau_patient": round(r["variance"].get("tau_patient", {}).get("median", np.nan), 4)
                              if "tau_patient" in r["variance"] else np.nan,
                              "Rhat max": round(r["rhat_max"], 3) if r.get("rhat_max") else np.nan})
            bdf = pd.DataFrame(brows)
            bout = os.path.join(bdir, "coupling_bayes_summary.xlsx")
            try:
                with pd.ExcelWriter(bout, engine="openpyxl") as w:
                    write_stacked(w, "Data & model",
                                  [("DATA & DESIGN", recap),
                                   ("MODEL", pd.DataFrame([("Specification",
                                    f"{outcome} ~ {x} + C({time}) + (1|{group_l3 or 'patient'}/{group})"),
                                    ("Prior SD", "student_t(3,0, 0.1*SD(outcome))"),
                                    ("Inference", "HDI 95% + pd + ROPE (no p)")],
                                    columns=["Element", "Value"])),
                                   ("PROVENANCE", provenance_df(seed=seed))])
                    S(bdf).to_excel(w, sheet_name="Bayesian coupling", index=False)
                    S(interpretation_glossary("bayes")).to_excel(w, sheet_name="Interpretation guide", index=False)
                if verbose:
                    print(f"Bayesian report written: {bout}")
            except Exception as e:
                if verbose:
                    print(f"[bayes not written] {type(e).__name__}: {e}")

    # ---------- figures ----------
    if plot_dir:
        from functions.general.longitudinal.plots import plot_rmcorr, plot_slope_scatter
        from functions.sced.plots.panels import plot_case_forest
        import matplotlib.pyplot as plt
        try:
            if rc.get("available"):
                plot_rmcorr(df, x=x, y=outcome, subject=rmcorr_subject, slope=rc.get("slope"),
                            save_path=plot_dir, fname=f"{outcome}_rmcorr", style=style)
            if sv.get("available") and isinstance(sv.get("slope_table"), pd.DataFrame):
                plot_slope_scatter(sv["slope_table"], save_path=plot_dir,
                                   fname=f"{outcome}_slope_vs_slope",
                                   xlab=f"slope {x} ~ {time}", ylab=f"slope {outcome} ~ {time}", style=style)
            # comparative forest of the coupling SLOPES (same unit: outcome / x)
            labels, meds, los, his = [], [], [], []
            if rc.get("available") and rc.get("slope_ci"):
                labels.append("rmcorr"); meds.append(rc["slope"])
                los.append(rc["slope_ci"][0]); his.append(rc["slope_ci"][1])
            _fe = next((f for f in lmm.get("fixed", []) if f["term"] == x), None)
            if _fe:
                labels.append("LMM (Wald)"); meds.append(_fe["coef"])
                los.append(_fe["CI95"][0]); his.append(_fe["CI95"][1])
            bg = res["bayes"].get("gaussian")
            if bg:
                b = bg["b_x"]; labels.append("Bayes (HDI)"); meds.append(b["median"])
                los.append(b["hdi_low"]); his.append(b["hdi_high"])
            if labels:
                plot_case_forest(labels, meds, los, his, pop=None, save_path=plot_dir,
                                 fname=f"{outcome}_coupling_forest", unit=f"{outcome} / {x}",
                                 title=f"Coupling {outcome}~{x} - compared estimates", style=style)
            plt.close("all")
            if verbose:
                print(f"Figures -> {plot_dir}/")
        except Exception as e:
            if verbose:
                print(f"[figures skipped] {type(e).__name__}: {e}")
    return res


def report_longitudinal_growth(
        df, *, outcome, time, group, shape="auto", knots=4, by=None, predictors=None,
        group_l3=None, random="int_slope", compare_shapes=("linear", "poly2", "spline"),
        within_between=None, kr=False, bayes=False, bayes_family="gaussian", n_trials=None,
        rope="auto", draws=2000, tune=2000, chains=4, seed=42, cache_dir=None, diagnostics=False,
        output_dir=None, save_path=None, style=None, verbose=True):
    """GROWTH-CURVE report: shape selection (AIC/BIC/LRT) -> fit -> fixed effects +
    variance/ICC/R2 + marginal trajectory (+ per ``by``) + caterpillar of the BLUPs.

    ``shape="auto"`` picks the best-AIC shape among ``compare_shapes``; otherwise forces
    ``shape``. ``by`` = time-invariant predictor (differential trajectories). ``within_between`` =
    name of a time-varying covariate to decompose (within/between) BEFORE the fit.
    ``bayes=True`` (Phase 3): adds the Bayesian arm (PyMC, same selected shape) -> ``<outcome>/
    Analyse/Bayes/growth_bayes_summary.xlsx`` + POSTERIOR trajectory (HDI) + MCMC diagnostics
    (if ``diagnostics``). Writes the frequentist xlsx (``<outcome>/Analyse/Inferentielle/``) + figures
    (``<outcome>/Plot/``). Returns a dict.

    References: Laird & Ware 1982 (growth-curve LMM); Akaike 1974 / Schwarz 1978 (shape
    selection); Gelman & Hill 2007 (Bayesian arm).
    R equivalent: nlme::lme / lme4::lmer (+ brms::brm for the Bayesian arm)."""
    from functions.sced.glossary import (std_layout, ascii_sanitize_df, write_stacked, data_recap_df,
                                          provenance_df, interpretation_glossary)
    from functions.general.longitudinal.growth import (fit_growth_curve, compare_growth_models,
                                               marginal_trajectory, blup_table, add_within_between)
    S = ascii_sanitize_df
    d = df.copy()
    predictors = list(predictors or [])
    res_out = {}

    # within/between decomposition of a time-varying covariate
    if within_between:
        d = add_within_between(d, within_between, group)
        predictors += [f"{within_between}_within", f"{within_between}_between"]

    # shape selection
    cmp = compare_growth_models(d, outcome=outcome, time=time, group=group, group_l3=group_l3,
                                shapes=compare_shapes, random=random, by=by, predictors=predictors,
                                knots=knots)
    res_out["compare"] = cmp
    if shape == "auto":
        best = cmp[cmp.get("best", False)] if "best" in cmp else cmp.head(1)
        lbl = best["shape"].iloc[0] if len(best) else "linear"
        shape = {"linear": "linear", "quadratic (poly2)": "poly2"}.get(
            lbl, "spline" if "spline" in lbl else "linear")
    res_out["shape"] = shape

    res, summ = fit_growth_curve(d, outcome=outcome, time=time, shape=shape, knots=knots, by=by,
                                 predictors=predictors, group=group, group_l3=group_l3,
                                 random=random, reml=True, kr=kr)
    res_out["fit"] = summ
    mt = bl = None
    if res is not None:
        try:
            mt = marginal_trajectory(res, d, time=time, shape=shape, by=by, knots=knots)
        except Exception as e:
            if verbose:
                print(f"  [marginal trajectory skipped] {type(e).__name__}: {e}")
        try:
            bl = blup_table(res, group=group)
        except Exception as e:
            if verbose:
                print(f"  [BLUP skipped] {type(e).__name__}: {e}")
    res_out["marginal"] = mt; res_out["blup"] = bl

    if not (output_dir or save_path):
        return res_out

    if output_dir is not None:
        L = std_layout(output_dir, outcome, "Inferentielle")
        inf_dir, plot_dir = L["analyse"], L["plot"]
    else:
        inf_dir = plot_dir = save_path

    fixed_df = pd.DataFrame(summ.get("fixed", []))
    meta = pd.DataFrame([("Formula", summ.get("Formula")), ("Time shape", summ.get("shape_label")),
                         ("Random", summ.get("Random")), ("ICC", summ.get("ICC")),
                         ("R2 marginal", summ.get("R2_marginal")), ("R2 conditional", summ.get("R2_conditional")),
                         ("AIC", summ.get("AIC")), ("BIC", summ.get("BIC")),
                         ("random var", summ.get("var_random")), ("residual var", summ.get("var_resid"))],
                        columns=["Element", "Value"])
    recap = data_recap_df(d, tier_col=group, session_col=time, outcomes=[outcome],
                          design=f"Multilevel growth-curve: {outcome} ~ f({time}) "
                                 f"[shape {shape}]" + (f" x {by}" if by else "") +
                                 (f" + within/between({within_between})" if within_between else ""))
    if inf_dir:
        os.makedirs(inf_dir, exist_ok=True)
        out = os.path.join(inf_dir, "growth_summary.xlsx")
        try:
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                write_stacked(w, "Data & design", [("DATA & DESIGN", recap),
                                                   ("PROVENANCE", provenance_df())])
                S(cmp).to_excel(w, sheet_name="Model selection", index=False)
                write_stacked(w, "Selected model", [("FIXED EFFECTS", fixed_df), ("MODEL & VARIANCE", meta)])
                if mt is not None:
                    S(mt.round(4)).to_excel(w, sheet_name="Marginal trajectory", index=False)
                S(interpretation_glossary("all")).to_excel(w, sheet_name="Interpretation guide", index=False)
            if verbose:
                print(f"Growth report written: {out}")
        except Exception as e:
            if verbose:
                print(f"[not written] {type(e).__name__}: {e}")

    if plot_dir:
        from functions.general.longitudinal.plots import plot_trajectory, plot_blup_caterpillar
        import matplotlib.pyplot as plt
        try:
            plot_trajectory(d, time=time, outcome=outcome, group=group, marginal=mt, by=by,
                            save_path=plot_dir, fname=f"{outcome}_trajectory", style=style)
            if bl is not None and len(bl):
                plot_blup_caterpillar(bl, save_path=plot_dir, fname=f"{outcome}_blup_caterpillar",
                                      unit=outcome, style=style)
            plt.close("all")
            if verbose:
                print(f"Figures -> {plot_dir}/")
        except Exception as e:
            if verbose:
                print(f"[figures skipped] {type(e).__name__}: {e}")

    # ---------- BAYESIAN arm (Phase 3): same selected shape ----------
    if bayes:
        from functions.general.longitudinal.growth import fit_growth_bayes_curve, posterior_trajectory
        try:
            br = fit_growth_bayes_curve(d, outcome=outcome, time=time, shape=shape, knots=knots,
                                        by=by, predictors=predictors, group=group, group_l3=group_l3,
                                        random=random, family=bayes_family, n_trials=n_trials, rope=rope,
                                        draws=draws, tune=tune, chains=chains, seed=seed,
                                        cache_dir=cache_dir, progressbar=False)
            res_out["bayes"] = br
            pmt = posterior_trajectory(br)
            res_out["posterior_marginal"] = pmt
            bdir = std_layout(output_dir, outcome, "Bayes")["analyse"] if output_dir is not None else save_path
            if bdir:
                os.makedirs(bdir, exist_ok=True)
                frows = []
                for term, s in br["fixed"].items():
                    frows.append({"term": term, "median": round(s["median"], 5),
                                  "HDI95": (round(s["hdi_low"], 5), round(s["hdi_high"], 5)),
                                  "pd": round(s["pd"], 3),
                                  "% in ROPE": s.get("pct_in_rope"), "ROPE decision": s.get("rope_decision", "")})
                vrows = [{"component": k, "median": round(v["median"], 4),
                          "HDI95": (round(v["hdi_low"], 4), round(v["hdi_high"], 4))}
                         for k, v in br["variance"].items()]
                bmeta = pd.DataFrame([("Shape", br.get("shape_label")), ("LOO (elpd)", round(br.get("loo", float("nan")), 1)),
                                      ("WAIC (elpd)", round(br.get("waic", float("nan")), 1)),
                                      ("Rhat max", round(br["rhat_max"], 3) if br.get("rhat_max") else "n/a"),
                                      ("Inference", "HDI 95% + pd + ROPE (no p)")],
                                     columns=["Element", "Value"])
                bout = os.path.join(bdir, "growth_bayes_summary.xlsx")
                with pd.ExcelWriter(bout, engine="openpyxl") as w:
                    write_stacked(w, "Data & model", [("DATA & DESIGN", recap),
                                                      ("BAYESIAN MODEL", bmeta),
                                                      ("PROVENANCE", provenance_df(seed=seed))])
                    S(pd.DataFrame(frows)).to_excel(w, sheet_name="Fixed effects (posterior)", index=False)
                    S(pd.DataFrame(vrows)).to_excel(w, sheet_name="Variances", index=False)
                    S(pmt.round(4)).to_excel(w, sheet_name="Posterior trajectory", index=False)
                    S(interpretation_glossary("bayes")).to_excel(w, sheet_name="Interpretation guide", index=False)
                if verbose:
                    print(f"Bayesian growth report written: {bout}")
            if plot_dir:
                from functions.general.longitudinal.plots import plot_trajectory
                import matplotlib.pyplot as plt
                plot_trajectory(d, time=time, outcome=outcome, group=group, marginal=pmt, by=by,
                                save_path=plot_dir, fname=f"{outcome}_trajectory_bayes",
                                title=f"Posterior trajectory - {outcome} (HDI 95%)", style=style)
                plt.close("all")
            if diagnostics and output_dir is not None:
                from functions.sced.bayes.report import _emit_diagnostics
                _emit_diagnostics({(outcome,): br["_idata"]}, std_layout(output_dir, outcome, "Bayes")["analyse"],
                                  style=style, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[bayes growth skipped] {type(e).__name__}: {e}")
    return res_out
