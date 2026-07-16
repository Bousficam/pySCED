import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def generate_random_effect_formula(random_effect=None, global_intercept=True):
    """
    Generate the random effect formula for a hierarchical model based on the list of random effect variables
    and the presence of a global intercept.

    Parameters:
    - random_effect: list of str or str, optional
        A list of variables (or a single variable as a string) to include as random effects.
        If None or empty, no random effect will be included.
    - global_intercept: bool, optional (default=True)
        Whether to include a global intercept in the formula.
        If False, the formula will suppress the global intercept.

    Returns:
    - str: The generated random effect formula. If no random effect and global_intercept is False, returns "0".
    """

    # Handle the case where no random effect is provided
    if random_effect is None or (isinstance(random_effect, list) and len(random_effect) == 0):
        return "1" if global_intercept else "0"

    # If a single string is passed instead of a list, convert it to a list
    if isinstance(random_effect, str):
        random_effect = [random_effect]

    # Construct the formula based on whether a global intercept is included
    if global_intercept:
        re_formula = " + ".join([f"C({re})" for re in random_effect])
    else:
        re_formula = "0 + " + " + ".join([f"C({re})" for re in random_effect])

    return re_formula

def regression_model(data, var_indep_X, var_dep_Y, model_type="Logit", family_distribution=None, random_effect=None, formula=None,
                     verbose=False):
    """
    Generic regression model function to support Logit, OLS, GLM, and HLM.

    Parameters:
    - data: pd.DataFrame, Dataset containing the variables.
    - var_indep_X: list or str, Independent variables.
    - var_dep_Y: str, Dependent variable for simple regression, fixed effect for hierarchical models.
    - model_type: str, Type of model to fit. Options are "Logit", "Linear" or "OLS", "GLM", "HLM".
    - family: str or None, distribution family for GLM
        Common choices include:
            - 'Logit': Logistic Regression (binary outcomes)
            - 'Linear' or 'OLS': Linear Regression (linear outcomes)
            - 'Poisson': for count data
            - 'Gamma': for positive continuous data
            - 'NegativeBinomial': for overdispersed count data
        Default is None, for non-GLM models.
    - random_effect: str,
     For HLM (hierarchical linear models), specify the random effect grouping variable (e.g., the variable representing
        clusters or groups such as "schools", "hospitals", or "patients"). This defines the random intercept or slope model.
        Example: `random_effect='group_id'`, where `group_id` is a column representing clusters or groups in the data.The random effect grouping variable for HLM.
    - formula: str or None, Optional. If provided, use the formula approach.
    - verbose: bool, Print additional information.

    Returns:
    - result: statsmodels fitted model result.

    References: Cox 1958 (logistic regression); Nelder & Wedderburn 1972 (generalized linear
      models); Laird & Ware 1982 (random-effects / mixed models, HLM branch).
    R equivalent: stats::glm / stats::lm (Logit/OLS/GLM) ; nlme::lme / lme4::lmer (HLM).
    """
    if model_type == "GLM":
        random_effect = None
        # GLM family selection. Note: compare with '==' (not 'is', which tests
        # object identity and is only reliable at the mercy of string interning
        # -> otherwise the wrong family / link would be silently selected).
        fam = str(family_distribution).strip().lower()
        if fam == 'logit':
            family = sm.families.Binomial()
        elif fam in ('linear', 'ols'):
            family = sm.families.Gaussian()
        elif fam == 'poisson':
            family = sm.families.Poisson()
        elif fam == 'gamma':
            family = sm.families.Gamma()
        elif fam == 'negativebinomial':
            family = sm.families.NegativeBinomial()
        else:
            raise ValueError(
                "For GLM, you must provide a valid family "
                "('Logit', 'Linear'/'OLS', 'Poisson', 'Gamma', 'NegativeBinomial').")
    elif model_type == "HLM":
        family_distribution = None


    # Use formula-based approach if formula is provided
    if formula is not None:
        if model_type == "Logit":
            model = smf.logit(formula, data=data)
        elif model_type == "Linear" or model_type == "OLS":
            model = smf.ols(formula, data=data)
        elif model_type == "GLM":
            model = smf.glm(formula, data=data, family=family)
        elif model_type == "HLM":
            model = smf.mixedlm(formula, data=data, groups=data[random_effect])
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    else:
        if isinstance(var_indep_X, str):
            var_indep_X = [var_indep_X]

        subdata = data[var_indep_X + [var_dep_Y]].copy().dropna()

        X = subdata[var_indep_X].astype(float)
        X = sm.add_constant(X)

        y = subdata[var_dep_Y].astype(float)

        if model_type == "Logit":
            model = sm.Logit(y, X)
        elif model_type == "Linear" or model_type == "OLS":
            model = sm.OLS(y, X)
        elif model_type == "GLM":
            model = sm.GLM(y, X, family=family)
        elif model_type == "HLM":
            if random_effect is None:
                raise ValueError("For HLM, you must provide a random_effect (grouping variable).")
            model = sm.MixedLM(y, X, groups=subdata[random_effect])
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    result = model.fit(disp=verbose)
    return result
