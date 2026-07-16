import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, spearmanr, kendalltau
import itertools

def compute_vif_table(data, columns):
    """
    Compute the Variance Inflation Factor (VIF) for each variable in the DataFrame.

    Inputs: a DataFrame and the list of columns to assess.
    Output: a DataFrame with columns 'Variable' and 'VIF', sorted by descending VIF.

    The VIF of a variable is derived from the R-squared of its regression on all
    the other variables, so it requires at least two variables and one complete
    observation.

    References: Belsley, Kuh & Welsch 1980 (variance inflation factor / collinearity diagnostics).
    R equivalent: car::vif.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    cols = list(columns)
    # The VIF of a variable is the regression of that variable on ALL the others.
    # It is therefore only defined with >= 2 variables and at least one observation;
    # otherwise statsmodels fails (empty matrix). The function degrades gracefully.
    if len(cols) == 0:
        return pd.DataFrame({"Variable": [], "VIF": []})

    X = data[cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    X = X.dropna()

    if X.shape[0] == 0:
        # No complete observation: VIF undefined.
        return pd.DataFrame({"Variable": cols, "VIF": [np.nan] * len(cols)})
    if X.shape[1] < 2:
        # Single variable: no collinearity possible, VIF = 1 by definition.
        return pd.DataFrame({"Variable": cols, "VIF": [1.0] * len(cols)})

    X = add_constant(X)  # Add a constant term (required for VIF)

    vif_df = pd.DataFrame()
    vif_df["Variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def correlogram(df, variables, method="pearson", plot=False):
    """
    Generate a correlogram (correlation matrix) for a list of variables in a DataFrame
    and report the significance (p-value) of the correlations.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the variables.
    variables (list): List of column names for which to generate the correlogram.
    method (str): Correlation method - 'pearson', 'spearman' or 'kendall'.
    plot (bool): If True, display the correlation and p-value heatmaps.

    Returns:
    corr (pd.DataFrame): Correlation matrix of the selected variables.
    p_values (pd.DataFrame): Matrix of p-values associated with each correlation.

    Side effect: displays two matplotlib heatmaps when plot=True.

    References: Pearson 1895 (product-moment r); Spearman 1904 (rank r); Kendall 1938 (tau).
    R equivalent: Hmisc::rcorr; psych::corr.test (r + p-value matrices).
    """
    # Check that the specified variables are present in the DataFrame
    for var in variables:
        if var not in df.columns:
            raise ValueError(f"The variable '{var}' does not exist in the DataFrame.")

    # Correlation test CONSISTENT with the requested method (both the r and the
    # p-values must match `method`, not default systematically to Pearson).
    method_l = str(method).lower()
    corr_test = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}.get(method_l)
    if corr_test is None:
        raise ValueError(f"Unsupported correlation method: '{method}'.")

    # LOCAL numeric copy - the caller's DataFrame must not be mutated.
    data = df[variables].apply(pd.to_numeric, errors='coerce').astype(float)
    corr = data.corr(method=method)

    # Initialize an empty matrix for the p-values
    p_values = pd.DataFrame(np.ones(corr.shape), columns=variables, index=variables)

    # Compute r and p for each pair of variables (consistent method)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):  # Loop over the upper triangle only
            x = data[variables[i]]
            y = data[variables[j]]

            # Remove NaN values from both variables
            valid_idx = x.notna() & y.notna()

            if valid_idx.sum() > 1:  # at least 2 valid pairs
                r, p = corr_test(x[valid_idx], y[valid_idx])
            else:
                r, p = np.nan, np.nan  # Not enough valid values: assign NaN

            corr.iloc[i, j] = r
            corr.iloc[j, i] = r
            p_values.iloc[i, j] = p
            p_values.iloc[j, i] = p
    if plot:
        # Create a figure for the correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0, xticklabels=variables, yticklabels=variables,
                    square=True, linewidths=.5, cbar_kws={"shrink": .75})
        plt.title("Correlogram of selected variables", size=15)
        plt.show()

        # Create a figure for the p-value heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(p_values, annot=False, cmap='YlGnBu', vmin=0, vmax=0.05, center=0, xticklabels=variables, yticklabels=variables,
                    square=True, linewidths=.5, cbar_kws={"shrink": .75})
        plt.title("P-values of correlations", size=15)
        plt.show()

    return corr, p_values

def detect_colinear_pairs(corr_matrix, threshold=0.8):
    """
    Return a list of variable pairs whose absolute correlation exceeds a threshold.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    threshold : float
        Threshold above which a correlation is considered problematic.

    Returns:
    --------
    pd.DataFrame : Correlated variable pairs with their coefficient.
    """
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = upper_tri.stack().reset_index()
    pairs.columns = ['Variable1', 'Variable2', 'Correlation']
    pairs = pairs[abs(pairs['Correlation']) >= threshold]
    # Sort by correlation strength (absolute value): -0.95 must rank above +0.85.
    pairs = pairs.reindex(pairs['Correlation'].abs().sort_values(ascending=False).index)
    return pairs.reset_index(drop=True)


def find_collinear_variables_with_pvalues(corr_matrix, p_values_matrix, corr_threshold=0.8, pvalue_threshold=0.05):
    """
    Identify collinear variable pairs (high correlation + significant p-value).

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix.
        p_values_matrix (pd.DataFrame): Matrix of associated p-values.
        corr_threshold (float): Correlation threshold.
        pvalue_threshold (float): Significance threshold.

    Returns:
        pd.DataFrame: Table of collinear pairs.
    """
    collinear_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            p_value = p_values_matrix.iloc[i, j]

            if abs(corr_value) > corr_threshold and p_value < pvalue_threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                collinear_pairs.append({
                    "Variable_1": var1,
                    "Variable_2": var2,
                    "Correlation": round(corr_value, 3),
                    "P_value": p_value
                })

    if collinear_pairs:
        out = pd.DataFrame(collinear_pairs)
        # Sort by correlation strength (absolute value).
        out = out.reindex(out['Correlation'].abs().sort_values(ascending=False).index)
        return out.reset_index(drop=True)
    else:
        return pd.DataFrame(columns=["Variable_1", "Variable_2", "Correlation", "P_value"])

def create_collinearity_summary(collinear_pairs):
    """
    Build a table summarizing collinear variable pairs with their correlation coefficients and p-values.

    Parameters:
    collinear_pairs (list): List of tuples containing the collinear variable pairs,
                            their correlation coefficients and p-values.

    Returns:
    summary_df (pd.DataFrame): A DataFrame summarizing the collinear pairs.
    """
    # Build a list of dictionaries, one per collinear pair
    summary_data = [{'Variable 1': var1, 'Variable 2': var2, 'Correlation': corr_value, 'P-value': p_value}
                    for var1, var2, corr_value, p_value in collinear_pairs]

    # Build a DataFrame from the list
    summary_df = pd.DataFrame(summary_data)

    return summary_df


def group_collinear_variables(collinear_pairs):
    """
    Group collinear variables into clusters.

    Parameters:
    - collinear_pairs (list of tuples): List of collinear pairs.

    Returns:
    - groups (list of sets): A list of collinear variable clusters.
    """
    groups = []

    for var1, var2, _, _ in collinear_pairs:
        found_group = False

        # Check whether var1 or var2 already belongs to an existing group
        for group in groups:
            if var1 in group or var2 in group:
                group.update([var1, var2])
                found_group = True
                break

        # If neither var1 nor var2 is in a group, create a new group
        if not found_group:
            groups.append(set([var1, var2]))

    # Merge clusters that overlap
    merged_groups = []
    while groups:
        first, *rest = groups
        first = set(first)

        merged = True
        while merged:
            merged = False
            rest2 = []
            for group in rest:
                if first & group:  # If two groups share an element
                    first |= group  # Merge them
                    merged = True
                else:
                    rest2.append(group)
            rest = rest2
        merged_groups.append(first)
        groups = rest

    return merged_groups

def select_collin_var(results, collinear_grp_auto, chosen_vars):
    """
    Select the most significant variable from each collinear group and remove the others from the chosen list.

    Parameters:
    - results (dict): Output of `pipeline_univariate_analysis`, containing test results for Continuous, Categorial, and Ordinal variables.
    - collinear_grp_auto (list of lists): Grouped collinear variables from `group_collinear_variables` function.
    - chosen_vars (list of str, optional): A list of variables chosen for the model. Only these variables will be checked for collinearity.

    Returns:
    - list: A list of non-collinear, significant variable names.
    """


    # Initialize a list to store the selected variables
    selected_vars = []

    # Create a set to track the variables to exclude
    excluded_vars = set()

    # Iterate over each collinear variable group
    for group in collinear_grp_auto:
        # Keep only the group variables that are present in the chosen list
        group = [var for var in group if var in chosen_vars]

        if not group:
            continue

        min_pval = float('inf')
        best_var = None

        # Find the variable with the smallest p-value in the group
        for var in group:
            # Locate the variable in the results and retrieve its p-value
            if var in results['Continuous']:
                pval = results['Continuous'][var]['Pval']
            elif var in results['Categorial']:
                pval = results['Categorial'][var]['Pval']
            elif var in results['Ordinal']:
                pval = results['Ordinal'][var]['Pval']
            else:
                continue

            # Check whether this p-value is the smallest found so far
            if pval < min_pval:
                min_pval = pval
                best_var = var

        # Add the best variable of the group
        if best_var is not None:
            selected_vars.append(best_var)

        # Add the remaining group variables to the exclusion set
        for var in group:
            if var != best_var:
                excluded_vars.add(var)

    # Add variables that are not collinear or not part of any collinear group
    for var in chosen_vars:
        if var not in excluded_vars and var not in selected_vars:
            selected_vars.append(var)

    return selected_vars