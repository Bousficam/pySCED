import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from collections.abc import Sequence
from textwrap import dedent

from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.exceptions import ConvergenceWarning

# Explicit imports from the internal modules (replaces the former `import *`,
# which implicitly exposed os/re/pd/sm/Path and created a fragile dependency).
from functions.general.univariate import (
    export_univariate_summary,
    get_critical,
    pipeline_analysis_descr,
    pipeline_univariate_log,
    pipeline_univariate_tests,
)
from functions.general.collinearity import (
    correlogram,
    compute_vif_table,
    find_collinear_variables_with_pvalues,
)

def select_significant_variables(df: pd.DataFrame, alpha: float = 0.05, missing_threshold: float = 0.05
) -> tuple[list[str], list[str]]:
    """
    Select the variables whose p-value is < alpha and whose percentage of missing
    data is <= missing_threshold, and also return those that are significant but
    excluded because missing > threshold.

    The function:
      1. locates the p-value column (any name containing 'pval', case-insensitive),
      2. locates the missing-percentage column (any name containing 'missing', case-insensitive),
      3. filters rows whose 'Variable' is not empty,
      4. converts the p-value to float and flags p < alpha,
      5. extracts the missing percentage from the "(XX.X%)" string and compares it to missing_threshold,
      6. returns two lists in order of appearance:
         - vars_ok: p < alpha **and** missing <= threshold
         - vars_high_missing: p < alpha **but** missing > threshold

    Parameters
    ----------
    df : pd.DataFrame
        Univariate DataFrame, with at least:
        - 'Variable',
        - a p-value column (e.g. 'Pval'),
        - a missing-data column (e.g. 'Missing data' in the "5 (2.2%)" format).
    alpha : float, default=0.05
        Significance threshold.
    missing_threshold : float, default=0.05
        Maximum tolerated missing (fraction, 0.05 = 5%). If >1, interpreted as a direct percentage.

    Returns
    -------
    vars_ok : list[str]
        Variables with p < alpha and missing <= threshold.
    vars_high_missing : list[str]
        Variables with p < alpha but missing > threshold.
    """
    # 1) identify p-value column
    pval_cols = [c for c in df.columns if ('pval' in c.lower()) | ('p-value' in c.lower())]
    if not pval_cols:
        raise ValueError(
            "select_significant_variables: no p-value column found "
            "(expected a name containing 'pval' or 'p-value')."
        )
    pcol = pval_cols[0]

    # 2) identify missing column
    missing_cols = [c for c in df.columns if 'missing' in c.lower()]
    mcol = missing_cols[0] if missing_cols else None

    # normalise threshold to %
    missing_thresh_pct = missing_threshold * 100 if missing_threshold <= 1 else missing_threshold

    # 3) Mask of non-empty variables
    mask_var = df['Variable'].astype(str).str.strip() != ""

    # 4) p-value mask
    pvals = pd.to_numeric(df[pcol], errors='coerce')
    mask_p = pvals < alpha

    # 5) missing mask
    if mcol:
        def _extract_pct(s: str) -> float:
            m = re.search(r'\(([\d\.]+)%\)', str(s))
            return float(m.group(1)) if m else 0.0
        missing_pct = df[mcol].apply(_extract_pct)
        mask_m_ok = missing_pct <= missing_thresh_pct
    else:
        mask_m_ok = pd.Series(True, index=df.index)
        missing_pct = pd.Series(0.0, index=df.index)

    # 6) build the two lists
    mask_signif = mask_var & mask_p
    mask_ok       = mask_signif & mask_m_ok
    mask_high_m   = mask_signif & ~mask_m_ok

    vars_ok = list(dict.fromkeys(df.loc[mask_ok, 'Variable']))
    vars_high_missing = list(dict.fromkeys(df.loc[mask_high_m, 'Variable']))

    return vars_ok, vars_high_missing
def detect_confounding_effect(df, outcome, var_interest, candidate_vars, threshold=0.10, verbose=True):
    """
    Identify the variables that, when added to the model, significantly change
    the effect (coefficient) of the variable of interest, suggesting confounding.

    Parameters:
    -----------
    df : pd.DataFrame
        Data including all the variables.
    outcome : str
        Name of the outcome variable (binary, coded 0/1).
    var_interest : str
        Variable whose coefficient stability is being tested.
    candidate_vars : list of str
        Other variables likely to act as confounders.
    threshold : float
        Coefficient-change threshold beyond which confounding is assumed (e.g. 0.10 = 10%).
    verbose : bool
        If True, prints the results.

    Returns:
    --------
    pd.DataFrame : Table of confounding effects with the percentage change.

    References: Maldonado & Greenland 1993 (change-in-estimate confounder selection).
    R equivalent: no direct R equivalent (manual change-in-estimate loop).
    """
    results = []

    # Baseline univariate model
    data = df[[outcome, var_interest]].dropna().apply(pd.to_numeric, errors='coerce').astype(float)
    y = data[outcome]
    X_base = sm.add_constant(data[[var_interest]])
    model_base = sm.Logit(y, X_base).fit(disp=0)
    beta_base = model_base.params[var_interest]

    if verbose:
        print(f"\nUnivariate model: Coef({var_interest}) = {beta_base:.4f}, p = {model_base.pvalues[var_interest]:.4f}\n")

    for var in candidate_vars:
        if var == var_interest:
            continue

        cols = [outcome, var_interest, var]
        data_tmp = df[cols].dropna()
        data_tmp = data_tmp.apply(pd.to_numeric, errors='coerce').astype(float)

        if data_tmp[var].nunique() <= 1:
            continue

        y = data_tmp[outcome]
        X = sm.add_constant(data_tmp[[var_interest, var]])

        try:
            model = sm.Logit(y, X).fit(disp=0)
            beta_new = model.params[var_interest]
            pval_new = model.pvalues[var_interest]
            delta = abs(beta_new - beta_base) / abs(beta_base)

            results.append({
                "Confounder": var,
                "Coef_adj": round(beta_new, 4),
                "Pct_change": round(delta * 100, 2),
                "p_value_adj": round(pval_new, 4),
                "Confounding?": "Yes" if (delta >= threshold)  else "No"
            })

            if verbose:
                print(f"{var:<25} -> Change: {delta * 100:<5.2f}%, p = {pval_new:<10.4f} -> {results[-1]['Confounding?']}")

        except Exception as e:
            if verbose:
                print(f"{var:<25} -> ERROR: {e}")
            continue

    if not results:
        if verbose:
            print("No model could be fitted.")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by="Pct_change", ascending=False).reset_index(drop=True)
def regression_model(data, var_indep_X, var_dep_Y, model_type="Logit", maxiter=1000, verbose=False, interactions=None, center_data=False):
    """
    Builds a regression model (Logistic or Linear) with optional interaction terms between independent variables.

    Parameters:
    - data: pd.DataFrame, the dataset
    - var_indep_X: list or str, independent variable(s)
    - var_dep_Y: str, dependent variable
    - model_type: str, "Logit" for logistic regression, "Linear" or "ols" for linear regression
    - maxiter: int, maximum iterations for fitting the model (for Logit)
    - verbose: bool, whether to display detailed output
    - interactions: list of tuples, optional interaction terms (e.g., [('var1', 'var2')])

    Returns:
    - result: Fitted regression model

    References: Cox 1958 (logistic regression, Logit branch).
    R equivalent: stats::glm(family = binomial) (Logit) / stats::lm (OLS).
    """
    # Ensure that var_indep_X and var_dep_Y are lists
    if isinstance(var_indep_X, str):
        var_indep_X = [var_indep_X]
    if isinstance(var_dep_Y, str):
        var_dep_Y = [var_dep_Y]

    list_var = var_indep_X + var_dep_Y
    if isinstance(interactions, list):
        for couple in interactions:
            if isinstance(couple, tuple):
                for var in couple:
                    if var not in list_var:
                        list_var.append(var)
            else:
                raise ValueError("interactions parameter should be a list of couple")

    # Subset data to include only the required variables
    subdata = data[list_var].copy().dropna()

    # Define the independent variables (X)
    X = subdata[var_indep_X].astype(float)
    # Center the independent variables if center_data is True
    if center_data:
        X = X.apply(lambda x: x - x.mean())
    # Create interaction terms if specified
    if interactions:
        for var1, var2 in interactions:
            interaction_term = X[var1] * X[var2]
            interaction_name = f'{var1}:{var2}'
            X[interaction_name] = interaction_term

    # Add a constant to the independent variables
    X = sm.add_constant(X)

    # Define the dependent variable (y)
    y = subdata[var_dep_Y].astype(float)



    # Select the model type
    if model_type == "Logit":
        model = sm.Logit(y, X)
    elif model_type == "Linear" or model_type == "ols":
        model = sm.OLS(y, X)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'Logit' or 'Linear'/'ols'.")

    # Fit the model and return the result
    result = model.fit(disp=verbose, maxiter=maxiter)

    return result
def get_criterion(model, criterion='AIC', pseudo_r2_type='McFadden', var=None):
    """
    Calculate various criteria for model selection.

    Parameters:
    model: statsmodels Logit or OLS fitted model
        The fitted model to evaluate.
    criterion: str
        The criterion to compute. Options are:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'pvalue': Maximum p-value of model coefficients (variable-specific)
        - 'MSE': Mean Squared Error (for linear models)
        - 'explained_variance': Explained variance, R² for linear models, or Pseudo-R² for logistic regression.
            Options for pseudo-R² are:
            - 'McFadden': McFadden pseudo-R² (default)
            - 'Nagelkerke': Normalized pseudo-R² for comparability to linear R²
            - 'Tjur': Tjur's pseudo-R², difference in mean predicted probabilities
        - 'Wald': Wald statistic for the significance of a specific variable's coefficient
        - 'AUC': Area Under the ROC Curve, measures classification performance (for logistic regression)
        - 'log_likelihood' or 'll': Log-Likelihood of the fitted model
        - 'adjusted_r2': Adjusted R² (for linear models)
        - 'MAE': Mean Absolute Error, suitable for linear and logistic regression

    pseudo_r2_type: str, default='McFadden'
        Type of pseudo-R² for logistic regression. Options:
        - 'McFadden' or 'M': Default, measuring model fit quality compared to the null model.
        - 'Nagelkerke' or 'N': Normalizes McFadden's pseudo-R² to obtain a value between 0 and 1, comparable to linear R².
        - 'Tjur' or 'T': Represents the difference in mean predicted probabilities for the outcome classes.

    var: str, optional
        The variable name for which the statistic is computed (used for 'pvalue' or 'Wald').

    Returns:
    float:
        The computed value for the selected criterion.

    References: Akaike 1974 (AIC); Schwarz 1978 (BIC); McFadden 1974, Nagelkerke 1991,
      Tjur 2009 (pseudo-R2); Hanley & McNeil 1982 (AUC).
    R equivalent: stats::AIC / stats::BIC; pscl::pR2 (pseudo-R2); pROC::auc (AUC).
    """

    if criterion == 'AIC':
        return model.aic

    elif criterion == 'BIC':
        return model.bic

    elif criterion == 'pvalue':
        # Return the p-value of the variable
        if var is not None:
            if var in model.pvalues.index:
                return model.pvalues[var]
            else:
                raise ValueError(f"Variable {var} not found in the model.")
        else:
            raise ValueError("Variable name must be provided when criterion is 'pvalue'.")

    elif criterion == 'MSE':
        # Mean squared error, valid for linear models (OLS)
        if hasattr(model, 'resid'):
            residuals = model.resid
            mse = np.mean(residuals ** 2)
            return mse
        else:
            raise ValueError("MSE is not available for this model type.")

    elif criterion == 'explained_variance':
        # Explained variance: R² for linear models, or Pseudo-R² for logistic regression
        if hasattr(model, 'rsquared'):
            return model.rsquared  # Explained variance for OLS models
        elif hasattr(model, 'llf') and hasattr(model, 'llnull'):
            llf = model.llf  # Log-likelihood of the fitted model
            llnull = model.llnull  # Log-likelihood of the null model
            n = model.nobs
            # Default is McFadden unless specified as Nagelkerke
            if pseudo_r2_type == 'McFadden' or pseudo_r2_type == 'M':
                return 1 - (llf / llnull)  # McFadden pseudo-R² as explained variance
            elif pseudo_r2_type == 'Nagelkerke' or pseudo_r2_type == 'N':
                return (1 - np.exp((2 / n) * (llnull - llf))) / (1 - np.exp(2 / n * llnull))  # Nagelkerke pseudo-R²
            elif pseudo_r2_type == 'Tjur' or pseudo_r2_type == 'T':
                # Tjur's R²
                predictions = model.predict()
                actuals = model.model.endog
                mean_pred_success = np.mean(predictions[actuals == 1])
                mean_pred_failure = np.mean(predictions[actuals == 0])
                return mean_pred_success - mean_pred_failure  # Tjur's R²
            return None
        else:
            raise ValueError("Explained variance is not available for this model type.")


    elif criterion == 'Wald':
        # Calculate the Wald statistic for a specific variable in the model
        if var is not None:
            if var in model.params.index:
                coef = model.params[var]  # Coefficient of the variable
                std_err = model.bse[var]  # Standard error of the coefficient
                wald_stat = (coef / std_err) ** 2  # Wald statistic
                return wald_stat
            else:
                raise ValueError(f"Variable {var} not found in the model.")
        else:
            raise ValueError("Variable name must be provided when criterion is 'Wald'.")
    elif criterion == 'AUC':
        if hasattr(model, 'predict'):
            predictions = model.predict()
            actuals = model.model.endog
            return roc_auc_score(actuals, predictions)
        else:
            raise ValueError("AUC is not available for this model type.")
    elif criterion in ('log_likelihood', 'll'):
        return model.llf

    elif criterion == 'adjusted_r2':
        if hasattr(model, 'rsquared_adj'):
            return model.rsquared_adj
        else:
            raise ValueError("Adjusted R² is not available for this model type.")

    elif criterion == 'MAE':
        if hasattr(model, 'resid'):
            residuals = model.resid
            mae = np.mean(np.abs(residuals))
            return mae
        else:
            raise ValueError("MAE is not available for this model type.")

    else:
        raise ValueError(
            "Unsupported criterion. Choose from 'AIC', 'BIC', 'pvalue', 'MSE', 'explained_variance', 'Wald', "
            "'AUC', 'log_likelihood' or 'll', 'adjusted_r2', or 'MAE'.")

#### STEPWISE SELECTION METHOD ####
# Classification of the selection criteria (single source, shared by the forward
# and backward steps to avoid any silent divergence).
SIGNIFICANCE_CRITERIA = frozenset({'pvalue', 'Wald'})
MAXIMIZE_CRITERIA = frozenset({'explained_variance', 'AUC', 'log_likelihood', 'll', 'adjusted_r2'})
MINIMIZE_CRITERIA = frozenset({'AIC', 'BIC', 'MSE', 'MAE'})

def forward_selection_step(data, included, var_indep_X, var_dep_Y, model_type="Logit", entry_criterion='AIC',
                           entry_threshold=0.05, verbose=True):
    """
    Perform one forward step of stepwise selection.

    Fits the current model on ``included`` and, for each not-yet-included
    candidate, refits with that candidate added and evaluates ``entry_criterion``.
    The best candidate is added if it improves the criterion (lower for
    minimize/significance criteria, higher for maximize criteria).

    Inputs: the working ``data``, the currently ``included`` variables, the pool
    ``var_indep_X``, the outcome ``var_dep_Y``, the model type and the entry
    criterion/threshold. Returns the (possibly extended) ``included`` list, the
    fitted ``best_model``, and the variable that was added (or ``[]`` if none).

    References: Hocking 1976 (variable selection in regression); Akaike 1974 / Schwarz 1978
      (AIC/BIC entry criterion).
    R equivalent: stats::add1 / MASS::stepAIC (forward step).
    """
    # Initiate model
    # Order-preserving difference: a set() of strings has an iteration order that
    # depends on PYTHONHASHSEED, which would make tie-breaking (min/max returns
    # the first extremum) non-reproducible from one run to the next.
    candidates_var = [v for v in var_indep_X if v not in included]
    n = len(included)
    minim_model = regression_model(data, included, var_dep_Y, model_type=model_type)

    if entry_criterion in SIGNIFICANCE_CRITERIA:
        best_criterion_value = entry_threshold
    else:
        best_criterion_value = get_criterion(minim_model, criterion=entry_criterion)
    new_criterion = {}

    # Test new variable
    for new_var in candidates_var:
        try:
            # Apply regression logistic for subset and evaluate the criterion
            model_add = regression_model(data, included + [new_var], var_dep_Y, model_type=model_type)
            if entry_criterion in SIGNIFICANCE_CRITERIA:
                new_criterion[new_var] = get_criterion(model_add, criterion=entry_criterion, var=new_var)
            else:
                new_criterion[new_var] = get_criterion(model_add, criterion=entry_criterion)
        except np.linalg.LinAlgError as e:
            print(f"Error while adding variable {new_var}: {e}")
            continue  # Skip if singular matrix error

    # Check improvement
    if new_criterion:
        if entry_criterion in MINIMIZE_CRITERIA or entry_criterion in SIGNIFICANCE_CRITERIA:
            # find the lowest criterion when variable is added or most significant pvalue
            best_new_var = min(new_criterion, key=new_criterion.get)
            best_new_criterion = new_criterion[best_new_var]
            if best_new_criterion < best_criterion_value:
                included.append(best_new_var)
                best_model = regression_model(data, included, var_dep_Y, model_type=model_type)

        elif entry_criterion in MAXIMIZE_CRITERIA:
            # find the highest when removed
            best_new_var = max(new_criterion, key=new_criterion.get)
            best_new_criterion = new_criterion[best_new_var]

            if best_new_criterion > best_criterion_value:
                included.append(best_new_var)
                best_model = regression_model(data, included, var_dep_Y, model_type=model_type)

    # return initial model if no var added
    if not len(included) > n:
        best_model = minim_model
        best_new_var = []
    else:
        if new_criterion and verbose:
            if entry_criterion in ['pvalue', 'Wald']:
                print(f"{entry_criterion} of {best_new_var} :  {best_new_criterion}")
            else:
               print(f"{entry_criterion} improved:  {best_new_criterion}")

    return included, best_model, best_new_var

def backward_selection_step(data, included, var_dep_Y, model_type="Logit", exit_criterion='AIC', exit_threshold=0.05, verbose=True):
    """
    Perform one backward step of stepwise selection.

    Fits the full model on ``included`` and, for each variable, evaluates
    ``exit_criterion`` either on that variable's coefficient (significance
    criteria) or on the model refitted without it. The worst variable is removed
    if that improves the criterion.

    Inputs: the working ``data``, the currently ``included`` variables, the
    outcome ``var_dep_Y``, the model type and the exit criterion/threshold.
    Returns the (possibly reduced) ``included`` list, the fitted ``best_model``,
    and the variable that was removed (or ``[]`` if none).

    References: Hocking 1976 (variable selection in regression); Akaike 1974 / Schwarz 1978
      (AIC/BIC exit criterion).
    R equivalent: stats::drop1 / MASS::stepAIC (backward step).
    """
    # Initiate model
    new_criterion = {}
    n = len(included)
    full_model = regression_model(data, included, var_dep_Y, model_type=model_type)

    if exit_criterion in SIGNIFICANCE_CRITERIA:
        criterion_value = exit_threshold
    else:
        criterion_value = get_criterion(full_model, criterion=exit_criterion)


    # Test var to removed
    for var in included:
        try:
            # Apply regression logistic for each reduced subset and evaluate the criterion
            if exit_criterion in SIGNIFICANCE_CRITERIA:
                model = regression_model(data, included, var_dep_Y, model_type=model_type)
                new_criterion[var] = get_criterion(model, criterion=exit_criterion, var=var)
            else:
                model = regression_model(data, [v for v in included if v != var], var_dep_Y, model_type=model_type)
                new_criterion[var] = get_criterion(model, criterion=exit_criterion)
        except np.linalg.LinAlgError as e:
            print(f"Error while removing variable {var}: {e}")
            continue  # Skip if singular matrix error

    # Check improvement
    if new_criterion:
        if exit_criterion in MAXIMIZE_CRITERIA or exit_criterion in SIGNIFICANCE_CRITERIA:
            # find the highest when removed or less significant pvalue
            worst_var = max(new_criterion, key=new_criterion.get)
            worst_new_criterion = new_criterion[worst_var]
            # less significant value is removed
            # or higher criterion is better without var
            if worst_new_criterion > criterion_value:
                included.remove(worst_var)
                best_model = regression_model(data, included, var_dep_Y, model_type=model_type)

        elif exit_criterion in MINIMIZE_CRITERIA:
            # find the lowest criterion when variable is removed
            worst_var = min(new_criterion, key=new_criterion.get)
            worst_new_criterion = new_criterion[worst_var]

            # lower is better without var
            if worst_new_criterion < criterion_value:
                # if model is better without rather than with all, removed it
                included.remove(worst_var)
                best_model = regression_model(data, included, var_dep_Y, model_type=model_type)

    # return full model if no var removed
    if not n > len(included):
            best_model = full_model
            worst_var = []
    else:
        if new_criterion and verbose:
            print(f"{exit_criterion} improved :  {worst_new_criterion}")


    return included, best_model, worst_var

def stepwise_selection(df, var_indep_X, var_dep_Y, model_type="Logit", entry_criterion='AIC',
                       exit_criterion='AIC', entry_threshold=0.05, exit_threshold=0.05,
                       direction='alternate', max_removal_count=10, verbose=False, multiclass_vars=None):
    """
    Perform stepwise selection to find the best logistic regression model with separate entry and exit criteria.

    Parameters:
    data: pd.DataFrame
        The dataset containing the variables.
    var_indep_X: list
        List of independent variable names.
    var_dep_Y: str
        The dependent variable name.
    entry_criterion: str, default='pvalue'
        The criterion used to add variables to the model (e.g., 'pvalue', 'AIC', 'BIC').
    exit_criterion: str, default='AIC'
        The criterion used to remove variables from the model (e.g., 'AIC', 'BIC', 'pvalue').
    entry_threshold: float, default=0.05
        Threshold for adding variables (if entry_criterion is 'pvalue').
    exit_threshold: float or None, default=None
        Threshold for removing variables (if exit_criterion is 'pvalue'). If None, no specific threshold is applied.
    direction: str, 'forward', 'backward', 'alternate'(default), 'sequential'.
    The criterion to compute. Options are:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'pvalue': Maximum p-value of model coefficients
        - 'MSE': Mean Squared Error (for linear models)
        - 'explained_variance': Variance explained (McFadden or Nagelkerke or Tjur for logistic regression, R² for linear models)
    Returns:
    best_combination: list
        The list of independent variables that resulted in the best model.
    best_model: statsmodels.Logit
        The best model based on the criteria.

    References: Hocking 1976 (stepwise variable selection); Akaike 1974 / Schwarz 1978 (AIC/BIC).
    R equivalent: MASS::stepAIC / stats::step.
    """
    if direction not in ['forward', 'backward', 'alternate', 'sequential']:
        raise ValueError(f'Invalid direction:{direction}')

    data = df.copy()
    # One-hot encode multiclass variables if provided
    if multiclass_vars:
        data = pd.get_dummies(data, columns=multiclass_vars, drop_first=True)
        # Update var_indep_X to include the newly created dummy variables
        dummy_vars = [col for col in data.columns if any(multivar in col for multivar in multiclass_vars)]
        local_var_indep_X = [var for var in var_indep_X if var not in multiclass_vars] + dummy_vars
        var_indep_X = local_var_indep_X
    else:
        local_var_indep_X = var_indep_X.copy()


    # Initiate direction
    included = local_var_indep_X.copy() if  direction == 'backward' else []
    if direction in ['alternate', 'sequential', 'forward']:
        start_forward = True
    else:
        start_forward = False
        start_backward = True


    # Initiate count
    removal_count = {var: 0 for var in local_var_indep_X}  # Track how many times a variable has been removed
    num_steps_forward = 0
    num_steps_backward = 0
    excluded_vars = []
    best_model = None
    changed = True

    while changed:
        changed = False

        # Forward step
        if start_forward:
            included, model_forward, new_var = forward_selection_step(
                data, included, local_var_indep_X, var_dep_Y, model_type, entry_criterion, entry_threshold, verbose=verbose)

            if new_var:
                best_model = model_forward
                num_steps_forward += 1
                changed = True
            if verbose:
                print(f"[Forward Step {num_steps_forward}] Variable '{new_var}' added based on {entry_criterion}.")
                print(f"Included variables after this step: {included}")

            if direction == 'forward':
                continue
            elif direction == 'sequential' and changed:
                start_backward = False  # Stay in forward phase
            else:
                start_backward = True


        # Backward step
        if len(included) > 0 and start_backward:
            included, model_backward, worst_var = backward_selection_step(
                data, included, var_dep_Y, model_type, exit_criterion, exit_threshold, verbose=verbose)

            if worst_var: # If a variable was excluded
                removal_count[worst_var] += 1
                if removal_count[worst_var] > max_removal_count:
                    # Stop including this variable after max_removal_count removals
                    local_var_indep_X.remove(worst_var)
                best_model = model_backward
                num_steps_backward += 1
                excluded_vars.append(worst_var)
                changed = True
                if verbose:
                    print(
                        f"[Backward Step {num_steps_backward}] Variable '{worst_var}' removed based on {exit_criterion}.")
                    print(f"Remaining variables after this step: {included}")

    # Final summary after all steps (order preserved for reproducibility)
    not_included_vars = [v for v in var_indep_X if v not in included]
    variable_selection = {
        "Total number of initial independent variables": len(var_indep_X),
        "Initial independent variables": var_indep_X,
        "Selection_mode" : "Stepwise",
        "Entry criterion": entry_criterion,
        "Entry threshold": entry_threshold if entry_criterion.lower() in ['pvalue', 'wald'] else None,
        "Exit criterion": exit_criterion,
        "Exit threshold": exit_threshold if exit_criterion.lower() in ['pvalue', 'wald'] else None,
        "Number of forward steps taken": num_steps_forward,
        "Number of backward steps taken": num_steps_backward,
        "Final set of included variables": included,
        "Variables excluded during backward steps": excluded_vars,
        "Variables initially available but not included": not_included_vars,
    }
    if best_model is not None:
        print(best_model.summary())
    else:
        print("No model was selected.")

    return best_model, variable_selection

#### PENALTY SELECTION METHOD ####
def optimize_elasticnet_params(data, var_indep_X, var_dep_Y,
                                 C_grid=None, l1_ratio_grid=None,
                                 stratified=True,cv=5, n_repeats=5, solver='saga', scoring='neg_log_loss',
                                 random_state: int = 0, max_iter=5000, verbose=0):
    """
    Jointly optimise the penalisation strength (C) and the L1/L2 mix (l1_ratio)
    for an ElasticNet logistic regression, through a single call to LogisticRegressionCV.

    This version offers two splitting modes:
    - If `stratified=True`, a RepeatedStratifiedKFold (cv folds x n_repeats repetitions)
      is used to keep the same proportion of the positive class in each fold and reduce
      the variance of the estimate. Useful when EPV is low.
    - Otherwise, LogisticRegressionCV splits the folds in a non-stratified way.

    Parameters
    ----------
    data : pandas.DataFrame
        Full DataFrame containing the columns listed in `var_indep_X` and `var_dep_Y`.
        Rows with NaN in these columns are dropped.
    var_indep_X : list[str]
        Names of the predictor columns.
    var_dep_Y : str
        Name of the binary target column (0/1).
    C_grid : array-like of float, optional
        Grid of values for the inverse regularisation strength (C = 1/lambda).
        ``None`` => default np.logspace(-3, 3, 20).
    l1_ratio_grid : array-like of float, optional
        Grid for the mixing parameter between L1 (1.0) and L2 (0.0).
        ``None`` => default [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] (pure Ridge end dropped).
    stratified : bool, default=True
        If True, uses RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_state).
        Otherwise, simply passes `cv=cv` to LogisticRegressionCV (non-stratified).
    cv : int, default=5
        Number of folds for the cross-validation.
    solver : str, default='saga'
        Solver used by LogisticRegression (must support `penalty='elasticnet'`).
    scoring : str, default='neg_log_loss'
        Performance criterion to optimise (e.g. 'neg_log_loss', 'roc_auc').
    random_state : int, default=0
        Seed for the reproducibility of the stratified splitting.
    max_iter : int, default=5000
        Maximum number of iterations for solver convergence.
    verbose : int, default=0
        Verbosity passed to LogisticRegressionCV.

    Returns
    -------
    best_C : float
        Value of C selected by cross-validation.
    best_l1_ratio : float
        Value of l1_ratio selected by cross-validation.

    References: Zou & Hastie 2005 (elastic net).
    R equivalent: glmnet::cv.glmnet (l1_ratio grid via the alpha argument).
    """
    # 0) Default grids (resolved here to avoid mutable default arguments
    #    and to centralise the calibrated values).
    if C_grid is None:
        C_grid = np.logspace(-3, 3, 20)
    if l1_ratio_grid is None:
        # Targeted grid: drop the Ridge end (l1_ratio~0, which yields no sparsity
        # and which saga handles poorly) to keep only the zone useful for
        # variable selection.
        l1_ratio_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # 1) Prepare data
    cols = var_indep_X + [var_dep_Y]
    df = data[cols].dropna().copy()
    X = df[var_indep_X].values
    y = df[var_dep_Y].values
    # 2) Ensure at least two classes are present
    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        if verbose:
            print(f"[CV] not feasible: y has only one class ({unique})")
        return None, None

    # 3) Adapt n_splits to the smallest count
    if stratified:
        min_count = counts.min()
        n_splits = min(cv, min_count)
        if n_splits < 2:
            if verbose:
                print(f"[CV] skip CV : min_count={min_count} < 2")
            return None, None
        cv_strategy = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        cv_strategy = cv
    # 4) Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegressionCV(
            Cs=C_grid,
            penalty='elasticnet',
            solver=solver,
            l1_ratios=l1_ratio_grid,
            cv=cv_strategy,
            scoring=scoring,
            max_iter=max_iter,
            n_jobs=-1,
            verbose=verbose,
        ))
    ])

    # 5) Attempt the fit and catch the "only one label in y_true" case
    try:
        pipeline.fit(X, y)
    except ValueError as e:
        if "contains only one label" in str(e):
            if verbose:
                print(f"[CV] invalid fold (single class) -> skipping ElasticNet")
            return None, None
        else:
            raise
    best_C = pipeline.named_steps['clf'].C_[0]
    best_l1_ratio = pipeline.named_steps['clf'].l1_ratio_[0]

    return best_C, best_l1_ratio

def optimize_lasso_ic(data, var_indep_X, var_dep_Y,
                      C_grid=np.logspace(-4, 4, 80),
                      ic = "AIC",
                      max_iter=1000, verbose=False):
    """
       Return `best_C` that minimises the in-sample information criterion
       using penalized_regression + get_criterion.

       References: Tibshirani 1996 (lasso); Zou, Hastie & Tibshirani 2007 (lasso degrees
         of freedom underpinning AIC/BIC selection).
       R equivalent: HDeconometrics::ic.glmnet - potential equivalent, to test.
       """
    best_C, best_ic = None, np.inf

    for C in C_grid:
        # 1.  Fit penalized Lasso to obtain the list of non-zero variables
        sel_vars, _ = penalized_selection(data, var_indep_X, var_dep_Y,
                                           C=C, l1_ratio=1.0,
                                           max_iter=max_iter, verbose=0)

        if not sel_vars:  # no variable => criterion of the null model
            X_sel = []  # refit without covariates
        else:
            X_sel = sel_vars

        # 2.  Refit unpenalized logit (natural units) on this subset
        try:
            log_model = regression_model(data, X_sel, var_dep_Y)


            # 3.  Compute the information criterion (in-sample)

            if ic.lower() == 'aic':
                ic_val = get_criterion(log_model, criterion="AIC")
            elif ic.lower() == "bic":
                ic_val = get_criterion(log_model, criterion="BIC")
            else:
                raise ValueError ("ic parameters should be 'AIC' or 'BIC'")
        except np.linalg.LinAlgError:
            if verbose:
                print(f"Singular matrix at C={C:.4g}, skipping...")
            continue

        if ic_val < best_ic:
            best_ic, best_C = ic_val, C
        if verbose:
            print(f"{ic} = {ic_val: .2f}   C={C: .4g}  vars={len(X_sel):2d}")

    return best_C
def penalized_selection(data, var_indep_X, var_dep_Y, C, l1_ratio,
                               max_iter=1000, verbose=0):
    """
    Apply penalized regression via statsmodels using the optimised hyperparameters.
    Converts C to alpha = 1/C and uses l1_ratio as L1_wt.
    Then selects the variables whose coefficient is non-zero and refits an
    unpenalized model to obtain standard errors and confidence intervals (CI).

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    var_indep_X : list
        List of the predictor variables.
    var_dep_Y : str
        Dependent variable (binary).
    best_params : dict
        Dictionary of optimised hyperparameters {'C': best_C, 'l1_ratio': best_l1_ratio}.
    max_iter : int
        Maximum number of iterations for convergence.
    verbose : int
        Verbosity level for the fit.

    Returns
    -------
    selected_vars : list
        List of the retained variables (non-zero coefficient).
    selection_info : dict
        Metadata about the selection process.

    References: Tibshirani 1996 (lasso); Zou & Hastie 2005 (elastic net).
    R equivalent: glmnet::glmnet.
    """
    # Data preparation
    cols = var_indep_X + [var_dep_Y]
    df = data[cols].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[var_indep_X])
    for i in range(len(var_indep_X)):
        if (abs(X_scaled.mean(axis=0)[i]) > 0.1) | (abs(X_scaled.std(axis=0)[i])-1 > 0.1):
            print(f"Standardisation failed for {var_indep_X[i]}")
            print(f"Standardised mean: {X_scaled.mean(axis=0)[i]}")
            print(f"Standardised SD: {X_scaled.std(axis=0)[i]}")
    X_scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=var_indep_X)
    X_scaled_sm = sm.add_constant(X_scaled_df)

    # Convert C to alpha for statsmodels: alpha = 1/C
    alpha = 1.0 / C

    # Fit the penalized model with statsmodels
    mod = sm.Logit(df[var_dep_Y], X_scaled_sm)
    try:
        res_reg = mod.fit_regularized(alpha=alpha,
                                      L1_wt=l1_ratio,
                                      maxiter=max_iter,
                                      disp=verbose)

    except np.linalg.LinAlgError:
        # Singular-matrix case: try trim_mode or give up on the refit
        if verbose:
            print("Warning: singular hessian in fit_regularized; skipping refit.")
        # Option 1: retry with trim_mode
        try:
            res_reg = mod.fit_regularized(alpha=alpha,
                                          L1_wt=l1_ratio,
                                          maxiter=max_iter,
                                          trim_mode='size',
                                          disp=verbose)
        except np.linalg.LinAlgError:
            # Option 2: no variable selected
            res_reg = None
            if verbose:
                print("Second attempt also failed; returning no selection.")


    # Select the variables whose coefficient is non-zero
    if res_reg is None:
        selected_vars = []
    else:
        coef_reg = res_reg.params
        selected_indices = [i for i, name in enumerate(coef_reg.index) if (name == 'const' or coef_reg[name] != 0)]
        if len(selected_indices) > 1:
            selected_vars = [var_indep_X[i - 1] for i in selected_indices if i != 0]
        else:
            selected_vars = []

    excluded_vars = list(set(var_indep_X) - set(selected_vars))
    selection_info = {
        "Total number of initial independent variables": len(var_indep_X),
        "Initial independent variables": var_indep_X,
        "C": C,
        "L1_ratio": l1_ratio,
        "Total number of selected variables": len(selected_vars),
        "Final set of included variables": selected_vars,
        "Total number of excluded variables": len(excluded_vars),
        "Variables excluded during penalty regression": excluded_vars,
    }
    return selected_vars, selection_info
def stability_selection(data, var_indep_X, var_dep_Y,
                        scoring="neg_log_loss",
                        n_resamples=100, sample_fraction  = 0.8, freq_threshold=0.60,
                        C_grid=None, l1_ratio_grid=None, n_repeats=5,
                        verbose=0, random_state=0, return_freq=True):
    """
        Perform stability selection for variable selection using penalized logistic regression.

        For each of `n_resamples` stratified subsamples (or bootstraps) of size
        `sample_fraction * n`, this function:
          1. Tunes the ElasticNet hyper-parameters (`C`, `l1_ratio`) via internal
             cross-validation with the specified `scoring`.
          2. Fits an ElasticNet-penalised logistic regression and records which
             variables have non-zero coefficients.
        After all resamples, it computes the selection frequency for each variable
        and retains those with frequency >= `freq_threshold`.

        Parameters
        ----------
        data : pandas.DataFrame
            Full dataset including both predictors and the binary outcome.
        var_indep_X : list[str]
            Names of the predictor variables.
        var_dep_Y : str
            Name of the binary outcome variable (0/1).
        scoring : str, default 'neg_log_loss'
            Scoring metric passed to the internal hyper-parameter tuner (e.g. 'neg_log_loss', 'roc_auc').
        n_resamples : int, default 100
            Number of random subsampling iterations.
        sample_fraction : float, default 0.8
            Fraction of the dataset to include in each subsample (between 0.5 and 0.9).
        freq_threshold : float, default 0.60
            Minimum selection frequency (0-1) to consider a variable "stable".
        C_grid, l1_ratio_grid : array-like, optional
            Elastic-Net tuning grids passed to ``optimize_elasticnet_params``
            at each resample. ``None`` => default calibrated grids.
        n_repeats : int, default 5
            Repetitions of the RepeatedStratifiedKFold of the internal tuning.
        verbose : int, default 0
            If >0, prints progress every 10 resamples and the final frequency table.
        random_state : int, default 0
            Seed for reproducible subsampling.
        return_freq : bool, default True
            If True, also returns the full selection frequency DataFrame.

        Returns
        -------
        stable_vars : list[str]
            Variables selected in at least `freq_threshold * n_resamples` subsamples.
        selection_info : dict
            Summary information including:
              - median values of `C` and `l1_ratio` across resamples,
              - total number of resamples and sample fraction,
              - counts of selected and excluded variables, etc.
        freq_df : pandas.DataFrame, optional
            DataFrame of selection frequencies (column "Selection_Frequency") for
            all variables; only returned if `return_freq=True`.

        References: Meinshausen & Buhlmann 2010 (stability selection).
        R equivalent: stabs::stabsel.
        """

    rng = np.random.RandomState(random_state)
    counts  = Counter()
    C_hist, l1_hist = [], []

    mask = data[var_dep_Y].notna()
    df_all = data.loc[mask].copy()
    y_full = df_all[var_dep_Y].values
    n_total = len(df_all)  #

    for b in range(n_resamples):
        # 1) Stratified subsampling
        sss = StratifiedShuffleSplit(n_splits=1,
                                     train_size=sample_fraction,
                                     random_state=rng.randint(0, 10_000))
        train_idx, _ = next(sss.split(np.zeros(n_total), y_full))
        df_b = data.iloc[train_idx]

        # 2) hyperparameter optimisation
        best_C, best_l1 = optimize_elasticnet_params(df_b, var_indep_X, var_dep_Y,
                                                    scoring=scoring, verbose=0,
                                                    C_grid=C_grid, l1_ratio_grid=l1_ratio_grid,
                                                    n_repeats=n_repeats)

        C_hist.append(best_C)
        l1_hist.append(best_l1)

        # 3) Penalized fit + retained variables
        selected_vars, _ = penalized_selection(
            df_b, var_indep_X, var_dep_Y,
            C=best_C,
            l1_ratio=best_l1,
            verbose=0)

        counts.update(selected_vars)


        if verbose and b % 10 == 0:
            print(f"[{b+1:3d}/{n_resamples}] retained variables: {selected_vars}")

    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # 3) Selection frequency
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    freq = {v: counts[v] / n_resamples for v in var_indep_X}
    freq_df = (pd.Series(freq, name="Selection_Frequency")
               .sort_values(ascending=False)
               .to_frame())
    stable_vars = freq_df[freq_df["Selection_Frequency"] >= freq_threshold].index.tolist()
    unstable_vars = freq_df[freq_df["Selection_Frequency"] < freq_threshold].index.tolist()
    if not stable_vars and unstable_vars:
        # No variable above the threshold: keep the most frequent one
        top_var = freq_df.index[0]
        stable_vars = [top_var]
        if verbose:
            print(f"No variable >= {freq_threshold*100:.0f}% -> keeping '{top_var}' (maximum frequency)")
    elif not stable_vars and not unstable_vars:
        # Extreme case: no candidate predictor at all
        if verbose:
            print("No predictor available for the stability selection.")
    if verbose:
        print(freq_df)
    excluded_vars = list(set(var_indep_X) - set(stable_vars))
    selection_info = {
        "Penalized selection": "ElasticNet & Stability selection",
        "Scoring for optimization": scoring,
        "C median": float(np.median(C_hist)),
        "L1_ratio median": float(np.median(l1_hist)),
        "Total number of resampling (Sample fraction) ": f"{n_resamples}({int(sample_fraction*100)}%)",
        "Total number of initial candidate variables": len(var_indep_X),
        "Initial candidate variables": var_indep_X,
        "Total number of selected variables": len(stable_vars),
        "Final set of included variables": stable_vars,
        f"Unstable variables (< {int(freq_threshold*100)}%)": unstable_vars,
        "Total number of excluded variables": len(excluded_vars),
        "Variables excluded during penalty regression": excluded_vars,
    }

    if verbose:
        print(f"\nStable variables (>= {int(freq_threshold*100)}%): {stable_vars}")
    if return_freq:
        return stable_vars, selection_info, freq_df
    else:
        return stable_vars, selection_info
def cross_validated_auc(df, predictors, outcome_name, n_splits=5, random_state=0):
    """
    Honest (out-of-sample) AUC by stratified cross-validation.

    Unlike the apparent AUC (``roc_auc_score`` computed on the fitting data),
    which is optimistically biased - especially with many predictors or a small
    sample size - , an unpenalized logistic model is refitted on each training
    fold and used to predict on the held-out fold, then the out-of-fold
    probabilities are aggregated.

    Returns ``np.nan`` if the computation is not possible (a single class, not
    enough observations per class, or a fitting failure).

    References: Stone 1974 (cross-validation); Hanley & McNeil 1982 (ROC AUC).
    R equivalent: cvAUC::cvAUC - potential equivalent, to test.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    if not predictors:
        return np.nan
    data = df[list(predictors) + [outcome_name]].apply(pd.to_numeric, errors='coerce').dropna()
    if data.empty:
        return np.nan
    X = data[list(predictors)].to_numpy(dtype=float)
    y = data[outcome_name].to_numpy(dtype=float)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.nan
    k = int(min(n_splits, counts.min()))
    if k < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        try:
            clf = LogisticRegression(penalty=None, max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        except Exception:
            return np.nan

    mask = ~np.isnan(oof)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return np.nan
    return roc_auc_score(y[mask], oof[mask])


def create_model_info_dict(best_model, df,
                           model_type="Logit"):
    """
    Build a summary dictionary from the information of the stepwise process,
    based on the messages and the summary shown in the stepwise_selection function.

    Parameters:
    -----------
    var_indep_initial : list
        List of the initial independent variables.
    entry_criterion : str
        Entry criterion used (e.g. 'AIC').
    exit_criterion : str
        Exit criterion used (e.g. 'AIC').
    entry_threshold : float
        Entry threshold (e.g. for p-value, 0.05).
    exit_threshold : float
        Exit threshold (e.g. for p-value, 0.05).
    num_forward_steps : int
        Number of forward steps taken.
    num_backward_steps : int
        Number of backward steps taken.
    final_included : list
        Final list of the variables included in the model.
    excluded_vars : list
        Variables excluded during the backward steps.
    best_model : statsmodels.LogitResults (or another model type)
        The final fitted model.
    df : pd.DataFrame
        The DataFrame used for the fit.
    model_type : str, optional (default "Logit")
        The model type. For a logistic model, the pseudo-R2 measures (McFadden, Nagelkerke, Tjur) and AUC are included.

    Returns:
    --------
    dict
        Summary dictionary containing:
          - Number and list of the initial variables.
          - The entry and exit criteria (with thresholds if applicable).
          - Number of forward and backward steps taken.
          - Final included, excluded and non-selected variables.
          - The text summary of the final model.
          - The percentage of observations used.
          - The model criteria (AIC, BIC, Log-Likelihood, etc.).
          - For a Logit model, the pseudo-R2 (McFadden, Nagelkerke, Tjur) and the AUC.

    References: Peduzzi et al. 1996 (events per variable, EPV > 10 rule).
    R equivalent: no direct R equivalent (summary assembler; cf. broom::glance).
    """
    # Compute the non-retained variables
    final_included = [var for var in best_model.params.index if var != 'const']
    # Percentage of observations used
    pct_dataset = best_model.nobs / df.shape[0] * 100

    model_info = {
        "Total number of observations (% of dataset)": f"{int(best_model.nobs)} ({pct_dataset:.2f}%)",
        "AIC (->0<-)": best_model.aic,
        "BIC (->0<-)": best_model.bic,
        "Log-Likelihood (->0<-)": best_model.llf,
    }

    if model_type.lower() == "logit":
        # Logit-specific metrics: number of events, EPV, LLR (do not exist for an
        # OLSResults -> only compute them in this branch).
        n_outcome = sum(best_model.model.endog)
        pct_obs = n_outcome / best_model.nobs * 100
        EPV = n_outcome / len(final_included) if len(final_included) > 0 else 0
        model_info["Total number of outcomes (% of observations)"] = f"{n_outcome} ({pct_obs:.2f}%)"
        model_info["Events per variable (>10)"] = EPV
        model_info["LLR p-value (<5%)"] = best_model.llr_pvalue
        # Pseudo-R2 measures and AUC
        model_info["Explained Variance (McFadden) (>0.2)"] = get_criterion(best_model, "explained_variance",
                                                                           pseudo_r2_type="McFadden")
        model_info["Explained Variance (Nagelkerke) (>0.2)"] = get_criterion(best_model, "explained_variance",
                                                                             pseudo_r2_type="Nagelkerke")
        model_info["Explained Variance (Tjur) (>0.2)"] = get_criterion(best_model, "explained_variance",
                                                                       pseudo_r2_type="Tjur")
        # Apparent AUC (in-sample): optimistically biased because computed on the
        # fitting data. A CV-validated (honest) AUC is also reported.
        model_info["AUC apparent (in-sample, optimistic) (>0.7)"] = get_criterion(best_model, "AUC")
        predictors = [v for v in best_model.params.index if v != 'const']
        model_info["AUC 5-fold CV (out-of-sample) (>0.7)"] = cross_validated_auc(
            df, predictors, best_model.model.endog_names)
    else:
        # OLS: global F-test (no llr_pvalue) + linear metrics.
        model_info["F-statistic p-value (<5%)"] = best_model.f_pvalue
        model_info["Explained Variance (R²)(>0.25)"] = get_criterion(best_model, "explained_variance")
        model_info["Adjusted Explained Variance (adjusted_r2)(>0.25)"] = get_criterion(best_model, "adjusted_r2")
        model_info["MSE(->0<-)"] = get_criterion(best_model, "MSE")
        model_info["MAE(->0<-)"] = get_criterion(best_model, "MAE")

    return model_info
def pipeline_multiv_logit(db: pd.DataFrame, var_preprocessed: dict, outcome_var: str, *,
                          other_var: list[str] | None = None, excluded_vars: list[str] | None = None,
                          method_univ: str = "reg",
                          pfilter: float = 0.20, missing_data: int = 5,
                          method_multiv: str = "stepwise", direction: str = "alternate", stability_select: bool = False,
                          entry_criterion: str = "AIC", exit_criterion: str = "AIC",
                          entry_threshold: float = 0.05, exit_threshold: float = 0.05,
                          optimize_hyperparam: bool = True,
                          C: float = 1.0, l1_ratio: float = 0.5, scoring_hyperparam: str = "neg_log_loss",
                          cv_C_grid=None, cv_l1_ratio_grid=None, cv_n_repeats: int = 5,
                          sample_fraction=0.8, freq_threshold=0.60,
                          save_path: str | Path | None = None, verbose=True):
    """
    Unified pipeline for building a **multivariate logistic** model,
    with two possible variable-selection strategies:

    - **stepwise**: sequential search (forward, backward, alternate, sequential).
    - **penalized**: selection by Elastic-Net or Lasso penalisation, optionally
      coupled with a Stability-Selection when ``optimize_hyperparam=True``.

    ----------
    Parameters
    ----------
    db : pandas.DataFrame
        Full dataset (rows=patients, columns=variables).

    var_preprocessed : dict
        Dictionary containing the already-typed variable lists:
        ``{'Continuous': [...], 'Multiclass': [...], 'Dicho': [...],
          'Baseline': [...], 'Outcomes': [...]}``.

    outcome_var : str
        Name of the binary target variable (0/1) to model.

    other_var : list[str], optional
        Additional variables to force into the selection (e.g. clinical markers
        of interest), outside the "Baseline". Ignored if ``None``.

    excluded_vars : list[str], optional
        Variables to exclude from the multivariate search from the outset.
    method_univ : str, default "reg"
        Method for the univariate analysis: either "reg" for a logistic regression
        or "test" for statistical tests (Student, Chi2, etc.)
    multiclass_test : str, default 'KW'
        Ordinal test: "Kruskal-Wallis" or KW, "Cochran-Armitage" or "CA", "Mann-Whitney" or "MW" (only if numeric values)
    pfilter : float, default 0.20
        p-value threshold (univariate analysis) below which a variable is
        retained as a candidate. Set 1.0 to filter nothing.

    missing_data : int, default 5
        Maximum percentage of missing data accepted for a variable to be
        considered (e.g. ``5`` => <=5% missing values).

    method_multiv : {'stepwise', 'penalized'}, default 'stepwise'
        Multivariate selection strategy.

    direction : str, default 'alternate'
        *If ``method=='stepwise'``* -> choice of direction:
        ``'forward' | 'backward' | 'alternate' | 'sequential'``.
        *If ``method=='penalized'``* -> type of penalisation:
        ``'elasticnet' | 'lasso'/'l1'`` (case-insensitive).

    entry_criterion / exit_criterion : {'AIC', 'BIC', 'p'}, default 'AIC'
        Criterion(s) used to decide the entry or exit of a variable during the
        "stepwise" step. Ignored if ``method=='penalized'``.

    entry_threshold / exit_threshold : float, default 0.05
        Numeric threshold(s) applied to the criterion (e.g. p-value).

    optimize_hyperparam : bool, default True
        - **True**: the (hyper)parameters C and l1_ratio are optimised by
          internal cross-validation (or AIC/BIC for the Lasso).
        - **False**: the values passed via ``C`` and ``l1_ratio`` are used
          directly (useful for sensitivity analyses).

    C : float, default 1.0
        Inverse of the penalisation strength (``alpha = 1/C``). Used only
        if ``optimize_hyperparam=False``.

    l1_ratio : float, default 0.5
        Weight of the L1 component in the Elastic-Net (0 = Ridge, 1 = Lasso).
        Ignored if ``direction='lasso'`` (fixed at 1) or if ``optimize_hyperparam``
        is ``True``.

    scoring_hyperparam : str, default 'neg_log_loss'
        Internal cross-validation metric for *Elastic-Net*:
        ``'neg_log_loss'`` (log-loss), ``'roc_auc'``.
        For the *Lasso*, use ``'AIC'`` or ``'BIC'`` for tuning on an information
        criterion, or ``'neg_log_loss'`` / ``'roc_auc'`` for predictive
        tuning.

    cv_C_grid / cv_l1_ratio_grid : array-like, optional
        Elastic-Net tuning grids (C and l1_ratio) passed to the internal
        cross-validation (``optimize_elasticnet_params`` / ``stability_selection``).
        ``None`` => default calibrated grids (C: 20 log-spaced points over
        1e-3..1e3; l1_ratio: [0.1..1.0], Ridge end dropped). Allows a
        sensitivity analysis without modifying the code.
    cv_n_repeats : int, default 5
        Repetitions of the RepeatedStratifiedKFold of the internal Elastic-Net tuning.
        Higher = more stable but slower CV estimate.

    save_path : str | pathlib.Path, optional
        Folder in which to export the summary Excel files
        (univariate + multivariate). No export if ``None``.

    ----------
    Results
    ----------
    Returns
    -------
    model_info : dict
        Complete metadata of the run: univariate filter, candidate variables,
        retained hyperparameters, global model statistics, etc.

    logModel : statsmodels.LogitResults | None
        Logistic model **refitted without penalisation** on the final set of
        selected variables. ``None`` if no variable remains or if the stepwise
        selection did not converge.

    ----------
    Notes
    -----
    *   When ``method='penalized'`` **and** ``optimize_hyperparam=True``,
        selection is done by **stability-selection**:
        100 subsamples (75% of the subjects), internal tuning, then counting
        of the variables kept >=80% of the time.
    *   The final model (``logModel``) is always estimated without penalisation
        in order to obtain unbiased ORs and 95% CIs.
    *   The datasets are systematically standardised before any penalized
        procedure.
    *   ``method='stepwise'`` follows the traditional entry/exit logic,
        controlled by ``entry_criterion`` / ``exit_criterion`` and
        ``entry_threshold`` / ``exit_threshold``.
    """

    if method_multiv.lower() not in {"stepwise", "penalized"}:
        raise ValueError("method must be 'stepwise' or 'penalized'")

    if method_multiv.lower() == "stepwise":
        if direction.lower() not in {"forward", "backward", "alternate", "sequential"}:
            raise ValueError("direction must be forward/backward/alternate/sequential for stepwise")
    else:  # penalized
        if direction.lower() not in {"elasticnet", "lasso", "l1"}:
            raise ValueError("direction must be elasticnet or lasso/l1 for penalized")
        if direction.lower() in {"lasso", "l1"} and scoring_hyperparam.lower() not in {"aic", "bic"}:
            raise ValueError("For Lasso choose scoring 'aic','bic'")
        if direction.lower() == "elasticnet" and scoring_hyperparam.lower() in {"aic", "bic"}:
            raise ValueError("ElasticNet tuning requires a predictive scorer (neg_log_loss or roc_auc)")

    # Extract variables
    cont_vars = var_preprocessed['Continuous']
    multiclass_vars = var_preprocessed['Multiclass']
    dicho_vars = var_preprocessed['Dicho']
    baseline_vars = var_preprocessed['Baseline']
    outcome_vars = var_preprocessed['Outcomes']
    only_describe = var_preprocessed['Describe']
    ordered_vars = baseline_vars + outcome_vars
    discret_vars = dicho_vars + multiclass_vars + only_describe

    ### Decriptive analysis
    descr_df = pipeline_analysis_descr(db, cont_vars=cont_vars, discret_var=discret_vars,
                                         ordered_vars=ordered_vars, save_path=None)

    ### Univariate analysis
    univ_result = pipeline_univariate_log(db, outcome_var, cont_vars=cont_vars,
                                          dicho_vars=dicho_vars, multiclass_vars=multiclass_vars,
                                          save_path=None, ordered_var=ordered_vars)
    univ_df = export_univariate_summary(univ_result, ordered_vars)

    test_results_df = pipeline_univariate_tests(db, outcome_var,
                                             cont_vars=cont_vars, dicho_vars=dicho_vars,
                                             multiclass_vars=multiclass_vars, ordered_vars=ordered_vars, multiclass_test="KW")

    ### Select variable for multivariate analysis
    potential_vars, missing_vars = select_significant_variables(univ_df, alpha=pfilter, missing_threshold=missing_data)

    # Update variable to analysis
    precandidate_vars = [var for var in potential_vars if var in baseline_vars]

    if other_var is not None:
        for var in other_var:
            if var != outcome_var:
                precandidate_vars.append(var)
    if excluded_vars is not None:
        for var in excluded_vars:
            if var in precandidate_vars:
                precandidate_vars.remove(var)
    ### Dummy encoding if necessary
    precandidate_multiclass = [v for v in precandidate_vars if v in multiclass_vars]
    other_precandidates = [v for v in precandidate_vars if v not in multiclass_vars]

    if precandidate_multiclass:
        # drop_first=True to avoid perfect collinearity
        db_enc = pd.get_dummies(db, columns=precandidate_multiclass, drop_first=True)
        dummy_cols = [
            col for col in db_enc.columns
            if any(col.startswith(mc + "_") for mc in precandidate_multiclass)
        ]
    else:
        db_enc = db.copy()
        dummy_cols = []

    # the final list for collinearity and multivariable selection
    candidate_vars = other_precandidates + dummy_cols

    # Compute VIF (robust: returns a trivial table if < 2 candidates)
    # To check: if penalized selection, compute VIF on selected variables?
    vif_df = compute_vif_table(db_enc, candidate_vars)
    # Collinearity (correlogram/pairs) is only meaningful with >= 2 variables.
    if len(candidate_vars) >= 2:
        corr, pvals = correlogram(db_enc, candidate_vars, plot=False)
        colinear_pairs_df = find_collinear_variables_with_pvalues(corr, pvals, corr_threshold=0, pvalue_threshold=0.05)
    else:
        colinear_pairs_df = pd.DataFrame()
    model_info= {
        'Pvalue initial filter' : f"{int(pfilter*100)}%",
        'Missing data threshold' : f"{missing_data}%",
        'Missing data variables not included' : missing_vars,
        'Number of candidate variables for multivariate' : len(candidate_vars),
    }
    if len(candidate_vars) < 2:
        model_info['Warning'] = (
            f"Only {len(candidate_vars)} candidate variable(s) after the univariate "
            f"filter (p<{pfilter}, missing<={missing_data}%). Collinearity not evaluated; "
            f"consider increasing pfilter, relaxing missing_data, or forcing "
            f"variables via other_var."
        )
    if method_multiv.lower() == "stepwise":
        logModel, selection_info = stepwise_selection(db_enc, candidate_vars, outcome_var, model_type='Logit',
                                                    entry_criterion=entry_criterion, exit_criterion=exit_criterion,
                                                    entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                                                    direction=direction, max_removal_count=5, verbose=False,
                                                    multiclass_vars=None)
        model_info.update(selection_info)
        if logModel is not None:
            stepwise_info = create_model_info_dict(logModel, db_enc)
            model_info.update(stepwise_info)


    elif method_multiv.lower() == "penalized":
        # Penalized selection
        if optimize_hyperparam:
            if direction.lower() == "elasticnet":
                if stability_select:
                    selected_variables, selected_dict_info, freq_df = stability_selection(db_enc,
                                        candidate_vars, outcome_var, scoring=scoring_hyperparam,
                                        n_resamples=100, sample_fraction=sample_fraction, freq_threshold=freq_threshold,
                                        C_grid=cv_C_grid, l1_ratio_grid=cv_l1_ratio_grid, n_repeats=cv_n_repeats,
                                        return_freq=True, verbose=verbose, random_state=0)
                else:
                    best_C, best_l1_ratio = optimize_elasticnet_params(db_enc, candidate_vars, outcome_var, scoring=scoring_hyperparam,
                                                                       C_grid=cv_C_grid, l1_ratio_grid=cv_l1_ratio_grid, n_repeats=cv_n_repeats)
                    if best_C is not None:
                        selected_variables, selected_dict_preinfo = penalized_selection(db_enc, candidate_vars, outcome_var,
                                                                                        best_C,
                                                                                        best_l1_ratio)
                        selected_dict_info = {"Penalized selection": direction,
                                              "Scoring for optimization": scoring_hyperparam,
                                              }
                        selected_dict_info.update(selected_dict_preinfo)
                    else:
                        selected_variables = []
                        selected_dict_info = {
                            "Penalized selection": direction,
                            "Warning": "Not enough classes for penalized CV"
                        }
            elif direction.lower() in ['lasso', 'l1']:
                l1_ratio = 1.0
                best_C = optimize_lasso_ic(db_enc, candidate_vars, outcome_var, ic=scoring_hyperparam)
                selected_variables, selected_dict_preinfo = penalized_selection(db_enc, candidate_vars, outcome_var, best_C,
                                                                             l1_ratio)
                selected_dict_info= { "Penalized selection": direction,
                                      "Scoring for optimization": scoring_hyperparam,
                                      }
                selected_dict_info.update(selected_dict_preinfo)
        else:
            best_C, best_l1_ratio = C, l1_ratio
            selected_variables, selected_dict_preinfo = penalized_selection(db_enc, candidate_vars, outcome_var, best_C, best_l1_ratio)
            selected_dict_info = {"Penalized selection": "Fixed",
                                  }
            selected_dict_info.update(selected_dict_preinfo)

        logModel = regression_model(db_enc, selected_variables, outcome_var)
        model_info.update(selected_dict_info)
        if logModel is not None:
            penalized_info = create_model_info_dict(logModel, db_enc)
            model_info.update(penalized_info)


    if save_path is not None:
        if method_multiv.lower() == "stepwise":
            save_path_multi = os.path.join(save_path, 'Multivariate models', 'Stepwise')
            filename = outcome_var + "_" + method_multiv.lower() + "_" + direction + "_summary.xlsx"
            filename_multi_path = os.path.join(save_path_multi, filename)
        elif method_multiv.lower() == "penalized":
            save_path_multi = os.path.join(save_path, 'Multivariate models', 'Penalized')
            if direction.lower() == "elasticnet":
                save_path_multi = os.path.join(save_path_multi, "ElasticNet")
            elif direction.lower() in ['lasso', 'l1']:
                save_path_multi = os.path.join(save_path_multi, "Lasso")
            if optimize_hyperparam:
                filename = outcome_var + "_" + direction + "_" + scoring_hyperparam +  "_summary.xlsx"
            else:
                filename = outcome_var + "_fixed_" + f"C{best_C}_l1r{best_l1_ratio}" +  "_summary.xlsx"
            filename_multi_path = os.path.join(save_path_multi, filename)
        else:
            filename_multi_path = None

        if filename_multi_path is not None:
            if not os.path.exists(save_path_multi):
                os.makedirs(save_path_multi)
            if logModel is not None:
                info_df, adjusted_df = export_multivariateModel_summary(model_info, logModel)
            else:
                # No final model: export the accumulated metadata.
                # Use model_info (always defined, already merged with the stepwise
                # OR penalized selection info) rather than selection_info, which
                # only exists in the stepwise branch.
                info_items = []
                for key, value in model_info.items():
                    info_items.append({"Parameter": key, "Value": value})
                info_df = pd.DataFrame(info_items, columns=["Parameter", "Value"])
                adjusted_df = pd.DataFrame()

            # Use ExcelWriter to create an Excel file with 2 sheets
            with pd.ExcelWriter(filename_multi_path, engine="xlsxwriter") as writer:
                info_df.to_excel(writer, sheet_name="Initial Setup Info", index=False)
                if stability_select and direction.lower() == "elasticnet":
                    freq_df.to_excel(writer, sheet_name="Stability selection", index=False)
                adjusted_df.to_excel(writer, sheet_name="Adjusted Variables", index=False)
                univ_df.to_excel(writer, sheet_name="Univariate Logits", index=False)
                test_results_df.to_excel(writer, sheet_name="Tests", index=False)
                descr_df.to_excel(writer, sheet_name="Cohort", index=False)
                vif_df.to_excel(writer, sheet_name="VIF Table", index=False)
                colinear_pairs_df.to_excel(writer, sheet_name="Correlogram", index=False)
            print(f"Summary exported to file {filename}")
    else:
        print(f"Summary not exported")

    return model_info, logModel

def export_multivariateModel_summary(model_info, logModel):
    """
    Export a complete summary of the stepwise process to an Excel file
    spread over two sheets:
      - "Initial Setup Info": contains the initial information and the performance of the final model.
      - "Adjusted Variables": table of the final variables (excluding the constant) of the final model,
                                with the columns: Independent Variable, OR (95% CI) and P.

    Parameters:
    -----------
    model_info : dict
        Summary dictionary (generated by create_model_info_dict) containing the information of the stepwise process.
        It must include, among others, the "LLR p-value" key.
    logModel : statsmodels.LogitResults (or equivalent)
        The final fitted model.
    filename : str, optional
        Name (and path) of the Excel file in which to save the summary.

    Returns:
    --------
    tuple (info_df, adjusted_df):
       - info_df: DataFrame of the "Initial Setup Info" sheet.
       - adjusted_df: DataFrame of the "Adjusted Variables" sheet.

    R equivalent: broom::tidy(model, exponentiate = TRUE, conf.int = TRUE).
    """
    # Build the table of adjusted variables (excluding the constant)
    adjusted_rows = []
    for var in logModel.params.index:
        if var == "const":
            continue  # Skip the constant
        coef = logModel.params[var]
        se = logModel.bse[var]
        OR = np.exp(coef)
        CI_lower = np.exp(coef - 1.96 * se)
        CI_upper = np.exp(coef + 1.96 * se)
        p_val = logModel.pvalues[var]
        critical = get_critical(p_val)
        adjusted_rows.append({
            "Independent Variable": var,
            "OR (95% CI)": f"{OR:.2f} ({CI_lower:.2f}-{CI_upper:.2f})",
            "p-value": f"{p_val:.4f}",
            "critical": critical
        })
    adjusted_df = pd.DataFrame(adjusted_rows, columns=["Independent Variable", "OR (95% CI)", "p-value", "critical"])

    # Convert the model_info dictionary to a DataFrame: one row per parameter
    info_items = []
    for key, value in model_info.items():
        info_items.append({"Parameter": key, "Value": value})
    info_df = pd.DataFrame(info_items, columns=["Parameter", "Value"])

    return info_df, adjusted_df

#### Generate method text
def _join_human(items: list[str]) -> str:
    """Return 'A', 'B' and 'C' (Oxford comma) for a list of items."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def generate_methods_section(*,
                              outcome_name: str | Sequence[str] | None = None,
                              method: str,
                              direction: str,
                              pfilter: float = 0.20,
                              missing_data: int = 5,
                              entry_criterion: str = "AIC",
                              exit_criterion: str = "AIC",
                              scoring_hyperparam: str = "neg_log_loss",
                              stratified: bool = True,
                              cv: int = 5,
                              n_repeats: int = 10,
                              optimize_hyperparam: bool = True,
                              C: float | None = None,
                              l1_ratio: float | None = None,
                              n_resamples: int | None = 100,
                              sample_fraction: float | None = 0.75,
                              freq_threshold: float | None = 0.80) -> str:
    """Return a ready-to-paste *Methods* paragraph describing the multivariate
    pipeline chosen in `pipeline_multiv_logit`.

    outcome_name : str | list[str] | None
        One or several outcome labels. A list will be rendered as "A, B, and C."
    """

    # detect runtime version
    python_version = sys.version.split()[0]

    # -  -  - Handle outcome(s) -  -  - 
    if isinstance(outcome_name, Sequence) and not isinstance(outcome_name, str):
        outcome_txt = " associated with " + _join_human(list(outcome_name))
    elif isinstance(outcome_name, str):
        outcome_txt = f" associated with {outcome_name}"
    else:
        outcome_txt = ""

    method_l, direction_l = method.lower(), direction.lower()

    # -  -  - Opening -  -  - 
    txt = dedent(f"""
        Continuous variables were summarised using means +/- standard deviations, whereas categorical variables were expressed as counts and percentages. Univariate logistic regression analyses were first performed to explore potential risk factors{outcome_txt}. Variables with a p-value < {pfilter:.2f} and < {missing_data}% missing data were retained as candidates. Multicollinearity was assessed through the variance inflation factor (VIF).""")

    # -  -  - Multivariable model -  -  - 
    if method_l == "stepwise":
        dir_map = {"forward": "forward", "backward": "backward", "alternate": "bidirectional (alternate)", "sequential": "sequential"}
        txt += "\n\n" + dedent(f"""
            Multivariate logistic regression was subsequently conducted with a **{dir_map.get(direction_l, direction_l)} stepwise selection** based on {entry_criterion} for entry and {exit_criterion} for removal. Adjusted odds ratios (aOR) and 95 % confidence intervals (CI) were reported.""")
    else:  # penalized
        if direction_l == "elasticnet":
            if optimize_hyperparam:
                score_map = {
                    "neg_log_loss": "log-loss minimisation",
                    "roc_auc": "area under the ROC curve (AUC) maximisation"
                }
                # description du CV
                if stratified:
                    cv_descr = f"repeated stratified {cv}-fold cross-validation (x{n_repeats})"
                else:
                    cv_descr = f"{cv}-fold cross-validation"
                tuning_line = (
                    f"hyper-parameters (C and l1_ratio) were selected by {cv_descr} with "
                    f"{score_map.get(scoring_hyperparam.lower(), scoring_hyperparam)}"
                )
                stab_line = (
                    f"Stability selection ({n_resamples} resamples of {sample_fraction:.0%}) "
                    f"retained variables selected in >={freq_threshold:.0%} of the resamples."
                )
            else:
                tuning_line = (
                    f"the penalisation strength was fixed a priori (C = {C}, l1_ratio = {l1_ratio})."
                )
                stab_line = ""
            txt += "\n\n" + (
                f"An Elastic-Net penalised logistic regression was fitted; {tuning_line} {stab_line}"
                " Adjusted odds ratios and 95 % CI were obtained by refitting an unpenalised "
                "model on the final subset of variables to avoid shrinkage bias."
            )
        else:  # lasso
            if optimize_hyperparam:
                ic = scoring_hyperparam.upper()
                tuning_line = f"the shrinkage parameter (C) was chosen by minimising the {ic} computed in-sample"
            else:
                tuning_line = f"the shrinkage parameter was fixed a priori (C = {C})"
            txt += (f"A Lasso-penalised logistic regression was fitted; {tuning_line}." +
                    " Adjusted odds ratios and 95 % CI were obtained by refitting an unpenalised model on the "
                    "final subset of variables to avoid shrinkage bias.")

    # -  -  - Software footer -  -  - 
    txt += f"Statistical significance was defined as p < 0.05. All analyses were conducted with Python {python_version}. "

    # -  -  - References -  -  - 
    refs = []
    if method_l == "stepwise":
        # references for stepwise
        if entry_criterion.upper() == 'AIC' or exit_criterion.upper() == 'AIC':
            refs.append("Akaike H. A new look at the statistical model identification. IEEE Transactions on Automatic Control. 1974;19(6):716-723.")
        if entry_criterion.upper() == 'BIC' or exit_criterion.upper() == 'BIC':
            refs.append("Schwarz G. Estimating the dimension of a model. The Annals of Statistics. 1978;6(2):461-464.")
    else:  # penalized
        if direction_l == "elasticnet":
            refs.append("Zou H, Hastie T. Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B (Methodological). 2005;67(2):301-320.")
            if optimize_hyperparam:
                refs.append("Meinshausen N, Bühlmann P. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology). 2010;72(4):417-473.")
        else:  # lasso
            refs.append("Tibshirani R. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological). 1996;58(1):267-288.")
            if optimize_hyperparam:
                refs.append("Zou H, Hastie T, Tibshirani R. On the 'Degrees of Freedom' of the Lasso. The Annals of Statistics. 2007;35(5):2173-2192.")

    if refs:
        txt += "\n\n**References:**" + "\n" + "\n".join([f"- {r}" for r in refs])
    print(txt)
    return txt.strip()


def plot_deviance_residuals_vs_predictions(result):
    """
       Plot deviance residuals versus predicted probabilities for a logistic regression model.

       Parameters:
       result: statsmodels fitted model
           The fitted logistic regression model.

       Returns:
       None

       References: McCullagh & Nelder 1989 (deviance residuals for GLM diagnostics).
       R equivalent: stats::residuals(model, type = "deviance") + base plot.
       """
    y_pred = result.predict()
    residuals = result.resid_dev  # Deviance residuals for logistic regression

    # Plot residuals vs. predicted probabilities
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Deviance Residuals')
    plt.title('Deviance Residuals vs. Predicted Probabilities')
    plt.show()


def plot_auc_roc(result, y_true):
    """
    Plot the ROC curve and calculate the AUC for a logistic regression model.

    Parameters:
    result: statsmodels fitted model
        The fitted logistic regression model.
    y_true: array-like
        The true binary labels (0 or 1) of the dataset.

    Returns:
    None

    References: Hanley & McNeil 1982 (ROC curve and AUC).
    R equivalent: pROC::roc / pROC::plot.roc.
    """
    # Step 1: Get the predicted probabilities
    y_pred_prob = result.predict()

    # Step 2: Compute the False Positive Rate and True Positive Rate for different thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Step 3: Calculate the AUC score
    auc_value = roc_auc_score(y_true, y_pred_prob)

    # Step 4: Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)  # Dashed diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()