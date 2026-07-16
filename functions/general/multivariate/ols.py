"""
Multivariate OLS pipeline (continuous outcome)
==============================================

Counterpart of ``pipeline_multiv_logit`` for a CONTINUOUS outcome, with the same
call/return contract so the two are interchangeable::

    model_info, olsModel = pipeline_multiv_ols(
        db, var_model, outcome,
        pfilter=0.2, missing_data=10,
        method_multiv='stepwise', direction='alternate',
        save_path=save_path, other_var=other_var, excluded_vars=only_describe)

This module is self-contained on purpose: it reuses the shared, outcome-agnostic
helpers (descriptive table, VIF, correlogram, stepwise selection) but implements
its own linear univariate screening and its own beta / standardised-beta export,
so it does not depend on the logistic-only machinery (OR table, AUC, LLR
p-value, ...).

Two selection strategies, mirroring the logistic pipeline:
  - ``method_multiv='stepwise'`` : sequential AIC/BIC selection.
  - ``method_multiv='penalized'`` with ``direction='lasso'`` or
    ``direction='elasticnet'`` : sklearn ``LassoCV`` / ``ElasticNetCV`` on
    standardised predictors, then an unpenalised OLS refit of the non-zero
    variables (so the reported betas / CIs are unbiased).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV

from functions.general.multivariate.selection import (
    regression_model,
    stepwise_selection,
    select_significant_variables,
    get_criterion,
)
from functions.general.collinearity import (
    compute_vif_table,
    correlogram,
    find_collinear_variables_with_pvalues,
)
from functions.general.univariate import pipeline_analysis_descr


# --------------------------------------------------------------------------- #
# Univariate linear screening
# --------------------------------------------------------------------------- #
def univariate_ols_screen(db, outcome_var, candidate_vars):
    """
    Univariate screen for a continuous outcome.

    For each candidate variable, fit a simple linear regression
    ``outcome ~ const + variable`` (complete cases) and report the slope's
    p-value plus the missing-data count/percentage.

    Returns a DataFrame with columns ``Variable``, ``Beta``, ``Pval`` and
    ``Missing data`` (``"N (P%)"``) - directly consumable by
    ``select_significant_variables``.

    R equivalent: stats::lm (per-variable simple linear regression).
    """
    n_total = len(db)
    rows = []
    for var in candidate_vars:
        y = pd.to_numeric(db[outcome_var], errors='coerce')
        x = pd.to_numeric(db[var], errors='coerce')
        n_missing = int(x.isna().sum())
        pct_missing = (n_missing / n_total * 100) if n_total else 0.0
        sub = pd.concat([y, x], axis=1, keys=['y', 'x']).dropna()
        beta = np.nan
        pval = np.nan
        # Require at least 3 points and a non-zero variance.
        if len(sub) >= 3 and sub['x'].nunique() > 1:
            try:
                model = sm.OLS(sub['y'].to_numpy(float),
                               sm.add_constant(sub['x'].to_numpy(float))).fit()
                beta = float(model.params[1])
                pval = float(model.pvalues[1])
            except Exception:
                pass
        rows.append({
            "Variable": var,
            "Beta": round(beta, 4) if pd.notna(beta) else np.nan,
            "Pval": pval,
            "Missing data": f"{n_missing} ({pct_missing:.1f}%)",
        })
    return pd.DataFrame(rows, columns=["Variable", "Beta", "Pval", "Missing data"])


# --------------------------------------------------------------------------- #
# Model info + export (beta / standardised beta)
# --------------------------------------------------------------------------- #
def _ols_model_info(olsModel, df):
    """Performance metrics of an OLS model (without the Logit-specific metrics).

    R equivalent: broom::glance(model).
    """
    final_included = [v for v in olsModel.params.index if v != 'const']
    pct_dataset = olsModel.nobs / df.shape[0] * 100 if df.shape[0] else 0.0
    info = {
        "Total number of observations (% of dataset)": f"{int(olsModel.nobs)} ({pct_dataset:.2f}%)",
        "Number of predictors": len(final_included),
        "AIC (->0<-)": olsModel.aic,
        "BIC (->0<-)": olsModel.bic,
        "Log-Likelihood": olsModel.llf,
        "F-statistic p-value (<5%)": olsModel.f_pvalue,
        "Explained Variance (R²)(>0.25)": get_criterion(olsModel, "explained_variance"),
        "Adjusted Explained Variance (adjusted_r2)(>0.25)": get_criterion(olsModel, "adjusted_r2"),
        "MSE (->0<-)": get_criterion(olsModel, "MSE"),
        "MAE (->0<-)": get_criterion(olsModel, "MAE"),
    }
    return info


def export_multivariate_OLS_summary(model_info, olsModel):
    """
    Two DataFrames (Excel sheets):
      - "Initial Setup Info": configuration + performance.
      - "Adjusted Variables": beta (95% CI), standardised beta
        (= beta * sd(x)/sd(y), to compare the relative importance of the
        predictors) and p, per variable.
    OLS mirror of export_multivariateModel_summary (which exports ORs).

    R equivalent: broom::tidy(model, conf.int = TRUE) / effectsize::standardize_parameters
      (standardised beta) - potential equivalent, to test.
    """
    info_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in model_info.items()],
                           columns=["Parameter", "Value"])

    rows = []
    if olsModel is not None:
        params = olsModel.params
        conf = olsModel.conf_int()
        pvalues = olsModel.pvalues
        exog = np.asarray(olsModel.model.exog, dtype=float)
        exog_names = list(olsModel.model.exog_names)
        endog = np.asarray(olsModel.model.endog, dtype=float)
        sd_y = endog.std(ddof=1)
        for var in params.index:
            if var == 'const':
                continue
            coef = params[var]
            ci_low, ci_high = conf.loc[var]
            std_beta = np.nan
            if var in exog_names and sd_y > 0:
                sd_x = exog[:, exog_names.index(var)].std(ddof=1)
                std_beta = coef * sd_x / sd_y
            rows.append({
                "Independent Variable": var,
                "Beta (95% CI)": f"{coef:.3f} ({ci_low:.3f} to {ci_high:.3f})",
                "Std. Beta": f"{std_beta:.3f}" if pd.notna(std_beta) else "NA",
                "P": f"{pvalues[var]:.4f}",
            })
    adjusted_df = pd.DataFrame(rows, columns=["Independent Variable", "Beta (95% CI)", "Std. Beta", "P"])
    return info_df, adjusted_df


# --------------------------------------------------------------------------- #
# Penalized linear selection (Lasso / ElasticNet)
# --------------------------------------------------------------------------- #
def penalized_linear_selection(data, var_indep_X, var_dep_Y, *,
                               penalty="lasso", l1_ratio_grid=None,
                               cv=5, n_alphas=100, random_state=0, max_iter=10000):
    """
    Variable selection by penalized linear regression (continuous outcome).

    Standardises the predictors, fits a ``LassoCV`` (penalty='lasso') or an
    ``ElasticNetCV`` (penalty='elasticnet') with the alpha (and the l1_ratio for
    ElasticNet) chosen by cross-validation, then keeps the variables whose
    penalized coefficient is non-zero. The final unpenalized model is refitted
    downstream by the pipeline (unbiased betas / CIs).

    Linear counterpart of ``penalized_selection`` (logistic).

    Returns
    -------
    selected_vars : list[str]
        Variables with a non-zero coefficient.
    selection_info : dict
        Metadata (alpha, l1_ratio where applicable, included/excluded variables).

    References: Tibshirani 1996 (lasso); Zou & Hastie 2005 (elastic net).
    R equivalent: glmnet::cv.glmnet.
    """
    cols = list(var_indep_X) + [var_dep_Y]
    df = data[cols].apply(pd.to_numeric, errors='coerce').dropna()
    selection_info = {
        "Penalized selection": penalty,
        "Total number of initial independent variables": len(var_indep_X),
        "Initial independent variables": list(var_indep_X),
    }
    # CV not feasible (too few observations or no variance on y).
    if df.shape[0] < cv or df[var_dep_Y].nunique() < 2 or len(var_indep_X) == 0:
        selection_info.update({
            "Warning": "Penalized CV not feasible (insufficient sample size or constant outcome)",
            "Total number of selected variables": 0,
            "Final set of included variables": [],
            "Total number of excluded variables": len(var_indep_X),
            "Variables excluded during penalty regression": list(var_indep_X),
        })
        return [], selection_info

    X = StandardScaler().fit_transform(df[var_indep_X].to_numpy(float))
    y = df[var_dep_Y].to_numpy(float)

    if penalty == "lasso":
        est = LassoCV(cv=cv, n_alphas=n_alphas, random_state=random_state, max_iter=max_iter)
        est.fit(X, y)
        selection_info["Alpha"] = float(est.alpha_)
    elif penalty == "elasticnet":
        if l1_ratio_grid is None:
            # Drop the pure Ridge end (l1_ratio=0, no sparsity).
            l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        est = ElasticNetCV(l1_ratio=l1_ratio_grid, cv=cv, n_alphas=n_alphas,
                           random_state=random_state, max_iter=max_iter)
        est.fit(X, y)
        selection_info["Alpha"] = float(est.alpha_)
        selection_info["L1_ratio"] = float(est.l1_ratio_)
    else:
        raise ValueError("penalty must be 'lasso' or 'elasticnet'")

    selected_vars = [v for v, c in zip(var_indep_X, est.coef_) if c != 0]
    excluded_vars = [v for v in var_indep_X if v not in selected_vars]
    selection_info.update({
        "Total number of selected variables": len(selected_vars),
        "Final set of included variables": selected_vars,
        "Total number of excluded variables": len(excluded_vars),
        "Variables excluded during penalty regression": excluded_vars,
    })
    return selected_vars, selection_info


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def pipeline_multiv_ols(db, var_preprocessed, outcome_var, *,
                        other_var=None, excluded_vars=None,
                        method_univ="reg",
                        pfilter=0.20, missing_data=5,
                        method_multiv="stepwise", direction="alternate",
                        entry_criterion="AIC", exit_criterion="AIC",
                        entry_threshold=0.05, exit_threshold=0.05,
                        cv_folds=5, l1_ratio_grid=None,
                        save_path=None, verbose=True):
    """
    Multivariate OLS pipeline (continuous outcome).

    Same parameters and same return as ``pipeline_multiv_logit``:
    ``(model_info: dict, olsModel: statsmodels OLS results | None)``.

    Multivariate selection:
      - ``method_multiv='stepwise'``: ``direction`` in
        {forward, backward, alternate, sequential}, AIC/BIC criteria.
      - ``method_multiv='penalized'``: ``direction`` in {lasso, l1, elasticnet};
        LassoCV/ElasticNetCV on standardised predictors (``cv_folds`` folds,
        ``l1_ratio_grid`` for ElasticNet), then unpenalized OLS refit.
    """
    method = method_multiv.lower()
    if method not in {"stepwise", "penalized"}:
        raise ValueError("method_multiv must be 'stepwise' or 'penalized'")
    if method == "stepwise":
        if direction.lower() not in {"forward", "backward", "alternate", "sequential"}:
            raise ValueError("direction must be forward/backward/alternate/sequential for stepwise")
    else:  # penalized
        if direction.lower() not in {"lasso", "l1", "elasticnet"}:
            raise ValueError("direction must be lasso/l1 or elasticnet for penalized")

    cont_vars = var_preprocessed['Continuous']
    multiclass_vars = var_preprocessed['Multiclass']
    dicho_vars = var_preprocessed['Dicho']
    baseline_vars = var_preprocessed['Baseline']
    outcome_vars = var_preprocessed['Outcomes']
    only_describe = var_preprocessed.get('Describe', [])
    ordered_vars = baseline_vars + outcome_vars
    discret_vars = dicho_vars + multiclass_vars + only_describe

    ### Descriptive (cohort)
    descr_df = pipeline_analysis_descr(db, cont_vars=cont_vars, discret_var=discret_vars,
                                       ordered_vars=ordered_vars, save_path=None)

    ### Linear univariate screening on the baseline variables
    screen_vars = [v for v in baseline_vars if v in db.columns and v != outcome_var]
    univ_df = univariate_ols_screen(db, outcome_var, screen_vars)

    ### Candidate selection
    potential_vars, missing_vars = select_significant_variables(univ_df, alpha=pfilter,
                                                                missing_threshold=missing_data)
    precandidate_vars = [v for v in potential_vars if v in baseline_vars]
    if other_var is not None:
        for var in other_var:
            if var != outcome_var:
                precandidate_vars.append(var)
    if excluded_vars is not None:
        for var in excluded_vars:
            if var in precandidate_vars:
                precandidate_vars.remove(var)

    ### Dummy encoding of the multiclass variables
    precandidate_multiclass = [v for v in precandidate_vars if v in multiclass_vars]
    other_precandidates = [v for v in precandidate_vars if v not in multiclass_vars]
    if precandidate_multiclass:
        db_enc = pd.get_dummies(db, columns=precandidate_multiclass, drop_first=True)
        dummy_cols = [c for c in db_enc.columns
                      if any(c.startswith(mc + "_") for mc in precandidate_multiclass)]
    else:
        db_enc = db.copy()
        dummy_cols = []
    candidate_vars = other_precandidates + dummy_cols

    ### Collinearity (robust: only if >= 2 candidates)
    vif_df = compute_vif_table(db_enc, candidate_vars)
    if len(candidate_vars) >= 2:
        corr, pvals = correlogram(db_enc, candidate_vars, plot=False)
        colinear_pairs_df = find_collinear_variables_with_pvalues(
            corr, pvals, corr_threshold=0, pvalue_threshold=0.05)
    else:
        colinear_pairs_df = pd.DataFrame()

    model_info = {
        'Model type': 'OLS (linear)',
        'Pvalue initial filter': f"{int(pfilter*100)}%",
        'Missing data threshold': f"{missing_data}%",
        'Missing data variables not included': missing_vars,
        'Number of candidate variables for multivariate': len(candidate_vars),
    }
    if len(candidate_vars) < 2:
        model_info['Warning'] = (
            f"Only {len(candidate_vars)} candidate variable(s) after the univariate "
            f"filter (p<{pfilter}, missing<={missing_data}%). Collinearity not evaluated; "
            f"consider increasing pfilter, relaxing missing_data, or forcing "
            f"variables via other_var."
        )

    ### Multivariate selection (stepwise OR penalized)
    olsModel = None
    if candidate_vars:
        if method == "stepwise":
            olsModel, selection_info = stepwise_selection(
                db_enc, candidate_vars, outcome_var, model_type='ols',
                entry_criterion=entry_criterion, exit_criterion=exit_criterion,
                entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                direction=direction, max_removal_count=5, verbose=False, multiclass_vars=None)
            if isinstance(selection_info, dict):
                model_info.update(selection_info)
        else:  # penalized
            penalty = "elasticnet" if direction.lower() == "elasticnet" else "lasso"
            selected_vars, selection_info = penalized_linear_selection(
                db_enc, candidate_vars, outcome_var, penalty=penalty,
                l1_ratio_grid=l1_ratio_grid, cv=cv_folds)
            model_info.update(selection_info)
            # Refit unpenalized OLS on the retained variables (unbiased betas / CIs).
            if selected_vars:
                olsModel = regression_model(db_enc, selected_vars, outcome_var, model_type='ols')
    if olsModel is not None:
        model_info.update(_ols_model_info(olsModel, db_enc))

    ### Excel export
    if save_path is not None:
        if method == "stepwise":
            sub, tag = "Stepwise_OLS", method + "_" + direction
        else:
            sub, tag = "Penalized_OLS", direction.lower()
        save_path_multi = os.path.join(save_path, 'Multivariate models', sub)
        filename = outcome_var + "_OLS_" + tag + "_summary.xlsx"
        filename_multi_path = os.path.join(save_path_multi, filename)
        if not os.path.exists(save_path_multi):
            os.makedirs(save_path_multi)
        info_df, adjusted_df = export_multivariate_OLS_summary(model_info, olsModel)
        with pd.ExcelWriter(filename_multi_path, engine="xlsxwriter") as writer:
            info_df.to_excel(writer, sheet_name="Initial Setup Info", index=False)
            adjusted_df.to_excel(writer, sheet_name="Adjusted Variables", index=False)
            univ_df.to_excel(writer, sheet_name="Univariate OLS", index=False)
            descr_df.to_excel(writer, sheet_name="Cohort", index=False)
            vif_df.to_excel(writer, sheet_name="VIF Table", index=False)
            colinear_pairs_df.to_excel(writer, sheet_name="Correlogram", index=False)
        if verbose:
            print(f"Summary exported to file {filename}")
    elif verbose:
        print("Summary not exported")

    return model_info, olsModel
