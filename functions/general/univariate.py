import os.path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, f, kruskal, chi2_contingency, fisher_exact, norm, wilcoxon
from statsmodels.stats.contingency_tables import Table
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path
import re


def get_missing_data(data_propre):
    """
    Count missing values per variable and format them as "count (percentage)".

    Inputs: a DataFrame. Output: a DataFrame with columns 'Variable' and
    'Missing data', where the percentage is relative to the number of rows.
    """
    # Count the missing values for each variable
    missing = data_propre.isnull().sum().reset_index()
    missing.columns = ['Variable', 'Missing Count']
    total_count = data_propre.shape[0]
    if total_count > 0:
        missing["Missing data"] = (
            missing["Missing Count"]
            .apply(lambda x: f"{x} ({x/total_count*100:.1f}%)")
        )
    else:
        missing["Missing data"] = "NA"

    return missing[["Variable", "Missing data"]]
def extract_missing_percent(missing_str):
    """
    Parse the percentage out of a "count (percentage%)" string.

    Inputs: a string such as "3 (12.5%)". Output: the percentage as a float, or
    0.0 when the string does not match the expected pattern.
    """
    if isinstance(missing_str, str) and "(" in missing_str and "%" in missing_str:
        try:
            percent_str = missing_str.split("(")[1].replace(")", "").replace("%", "")
            return float(percent_str.strip())
        except:
            return 0.0
    return 0.0

### Description of the cohort
def analyse_descriptive_continuous(df, var_continuous):
    """
    For each continuous variable, return a 3-column DataFrame:
      - Variable
      - Level  (non-missing count)
      - Cohort: distribution-appropriate summary (biomedical convention) -
        mean +/- standard deviation when normality is not rejected (Shapiro,
        p > 0.05), otherwise median [Q1 ; Q3].

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    var_continuous : list[str]
        List of continuous variable names.

    Returns
    -------
    pd.DataFrame with columns ['Variable','Level','Cohort']

    References: Shapiro & Wilk 1965 (normality test driving mean+/-SD vs median [IQR]).
    R equivalent: stats::shapiro.test (normality branch); gtsummary::tbl_summary (summary table).
    """
    rows = []
    missing_data = get_missing_data(df)
    missing_map = missing_data.set_index('Variable')['Missing data']
    for var in var_continuous:
        # Exclude NaN from the computation
        serie = df[var].dropna().astype(float)
        n = len(serie)
        if n == 0:
            label = ""
        else:
            # Shapiro requires n >= 3; below that, report median [IQR] by default.
            normal = n >= 3 and shapiro(serie)[1] > 0.05
            if normal:
                label = f"{serie.mean():.2f} ± {serie.std(ddof=1):.2f}"
            else:
                q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
                label = f"{serie.median():.2f} [{q1:.2f} ; {q3:.2f}]"
        rows.append({
            "Variable": var,
            "Level": n,
            "Cohort": label
        })
    descr_continuous = pd.DataFrame(rows, columns=["Variable","Level","Cohort"])
    descr_continuous['Missing data'] = descr_continuous['Variable'].map(missing_map)

    return descr_continuous

def analyse_descriptive_continuous_extend(df, var_continuous):
    """
    Full descriptive statistics for continuous variables (pandas describe output).

    Inputs: a DataFrame and the list of continuous variable names. Output: a
    DataFrame with one row per variable (count, mean, std, min, quartiles, max)
    plus a 'Missing data' column.
    """
    # 1) Descriptive statistics + move the index back into a column
    descr_continuous = (
        df[var_continuous]
        .astype(float)
        .describe()
        .T
        .reset_index()
        .rename(columns={'index': 'Variable'})
    )

    # 3) Retrieve the missing-value percentages
    missing_data = get_missing_data(df)
    missing_map = missing_data.set_index('Variable')['Missing data']
    descr_continuous['Missing data'] = descr_continuous['Variable'].map(missing_map)
    return descr_continuous
def analyse_descriptive_dicho(df, var_dicho):
    """
    For each dichotomous variable, return a 3-column DataFrame:
      - Variable
      - Level         ('Total' then each category)
      - Cohort (count (%))

    Detail rows after the first have an empty 'Variable' column.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    var_dicho : list[str]
        List of dichotomous variable names.

    Returns
    -------
    pd.DataFrame with columns ['Variable','Level','Cohort (count (%))']
    """
    rows = []
    missing_data = get_missing_data(df)
    missing_map = missing_data.set_index('Variable')['Missing data']
    for col in var_dicho:
        # Count each category (ignores NaN)
        counts = df[col].value_counts(dropna=True)
        total = counts.sum()
        # "Total" row
        rows.append({
            "Variable": col,
            "Level":   "Total",
            "Cohort": f"{total} (100.0%)"
        })
        # One row per category
        for lvl, cnt in counts.items():
            pct = cnt / total * 100
            rows.append({
                "Variable": "",
                "Level":   lvl,
                "Cohort": f"{cnt} ({pct:.1f}%)"
            })
    descr_dicho = pd.DataFrame(rows, columns=["Variable","Level","Cohort"])
    descr_dicho['Missing data'] = descr_dicho['Variable'].map(missing_map)
    return descr_dicho
def analyse_descriptive_dicho_extend(df, var_dicho):
    """
    Detailed frequency table for dichotomous/categorical variables.

    Inputs: a DataFrame and the list of variable names. Output: a DataFrame with
    a 'Total' row and one row per category, giving Count and relative frequency
    (Freq) plus a 'Missing data' column. The variable name is shown only once.
    """
    results = []
    missing_data = get_missing_data(df)
    missing_dict = missing_data.set_index("Variable")["Missing data"].to_dict()
    # Key function to correctly sort numbers and strings
    def sort_key(x):
        try:
            return (0, float(x))
        except:
            return (1, str(x))

    for col in var_dicho:
        counts = df[col].value_counts(dropna=True)
        freq = df[col].value_counts(normalize=True, dropna=True)

        # Overall statistic
        results.append({
            'Variable': col,
            'Class': 'Total',
            'Count': counts.sum(),
            'Freq': 1.0,
            'Missing data': missing_dict.get(col)

        })

        kclass = [classe for classe in counts.index]

        for classe in kclass:
            results.append({
                'Variable': col,
                'Class': classe,
                'Count': counts[classe],
                'Freq': round(freq[classe], 3)
            })
    descr_dicho = pd.DataFrame(results)
    # For each variable, show the name only once (on the "Total" row)
    for var in descr_dicho['Variable'].unique():
        idx = descr_dicho[descr_dicho['Variable'] == var].index.tolist()
        if len(idx) > 1:
            descr_dicho.loc[idx[1:], 'Variable'] = ""

    return descr_dicho
def pipeline_analysis_descr(data_propre, cont_vars : list, discret_var : list, ordered_vars : list,save_path=None):
    """
    Build a single cohort description table (continuous + dichotomous variables).

    Inputs: the cleaned DataFrame, the lists of continuous, discrete and ordered
    variables, and an optional save path. Output: the combined description
    DataFrame. Side effect: writes 'cohort_descript.xlsx' when save_path is set.
    """
    # Descriptive analysis for the continuous data
    descr_continuous = analyse_descriptive_continuous(data_propre, cont_vars)
    descr_dicho = analyse_descriptive_dicho(data_propre, discret_var)
    descr_all = pd.concat(
        [descr_continuous, descr_dicho],
        axis=0,  # vertical concatenation
        ignore_index=True  # renumber the index from 0 to n-1
    )
    if ordered_vars is not None:
        descr_all = reorder_summary_by_variable(descr_all, ordered_vars)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        xlsx_path = os.path.join(save_path, 'cohort_descript.xlsx')
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            descr_all.to_excel(writer, sheet_name="Cohort", index=False)
    return descr_all
def pipeline_analysis_descr_extend(data_propre, cont_vars : list, discret_var : list, ordered_vars : list,save_path=None):
    """
    Build separate cohort description sheets for continuous and dichotomous variables.

    Inputs: the cleaned DataFrame, the lists of continuous, discrete and ordered
    variables, and an optional save path. Output: a dict of DataFrames keyed by
    sheet name. Side effect: writes a multi-sheet 'cohort_descript.xlsx' when
    save_path is set.
    """
    descr_sheet = {}

    # Descriptive analysis for the continuous data
    descr_continuous = analyse_descriptive_continuous(data_propre, cont_vars)
    if ordered_vars is not None:
        descr_continuous = reorder_summary_by_variable(descr_continuous, ordered_vars)
    descr_sheet['Descr Continuous'] = descr_continuous

    # Descriptive analysis for the dichotomous/categorical data
    descr_dicho = analyse_descriptive_dicho(data_propre, discret_var)
    if ordered_vars is not None:
        descr_dicho = reorder_summary_by_variable(descr_dicho, ordered_vars)
    descr_sheet['Descr Dichotomic'] = descr_dicho

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        xlsx_path = os.path.join(save_path, 'cohort_descript.xlsx')
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            for sheet_name, df in descr_sheet.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


    return descr_sheet


### Tests
def perform_distrib_charact(cohort, var_indep, var_dep, var_type = 'continuous'):
    """
    Describe the distribution of a dependent variable across the two groups of a
    binary independent variable (coded 0/1).

    Inputs: the cohort DataFrame, the independent and dependent variable names,
    and var_type ('continuous' or 'categorial'). Output for continuous variables:
    (mean group 0, mean group 1, delta, n0, n1, std0, std1). Output for
    categorical variables: (n0, n1, number of categories, row labels, column
    labels, first-row counts, second-row counts).
    """
    data = cohort.copy()
    subdata = data[[var_indep, var_dep]].copy().dropna()
    group_1 = np.array(subdata[subdata[var_indep] == 0][var_dep].copy().dropna())
    group_2 = np.array(subdata[subdata[var_indep] == 1][var_dep].copy().dropna())
    n1 = len(group_1)
    n2 = len(group_2)

    if var_type == 'continuous':
        # Compute the means while avoiding empty arrays
        mean1 = np.mean(group_1) if n1 > 0 else np.nan
        mean2 = np.mean(group_2) if n2 > 0 else np.nan
        Delta = mean1 - mean2 if (n1 > 0 and n2 > 0) else np.nan

        # Compute the standard deviations when the group has at least 2 values
        std1 = np.std(group_1, ddof=1) if n1 > 1 else np.nan
        std2 = np.std(group_2, ddof=1) if n2 > 1 else np.nan

        return mean1, mean2, Delta, n1, n2, std1, std2

    else:
        contingency_table = pd.crosstab(subdata[var_indep], subdata[var_dep])
        if contingency_table.empty:
            # Nothing to return when the table is empty
            return n1, n2, 0, [], [], [], []

        kclass = contingency_table.shape[1]
        n_lignes = contingency_table.shape[0]
        label_dep = contingency_table.columns.tolist()
        label_indep = contingency_table.index.tolist()

        # Check that at least one row exists before accessing iloc[0]
        nk1 = contingency_table.iloc[0].tolist() if n_lignes > 0 else []
        nk2 = contingency_table.iloc[1].tolist() if n_lignes > 1 else None

        return n1, n2, kclass, label_indep, label_dep, nk1, nk2
def perform_categorial_tests(cohort, var_indep, var_dep):
    """Select and apply the association test appropriate for a contingency table.

    The decision is based on the EXPECTED counts (Cochran's rule), not on the
    observed counts:
      - 2x2: Fisher's exact test if at least one expected count < 5
             (zero observed cells are then handled correctly by Fisher, which is
             designed precisely for this case); otherwise chi-square with Yates'
             continuity correction.
      - r x c (>2x2): Pearson chi-square. Flagged as approximate if an expected
             count < 5 (scipy does not provide an exact r x c Fisher test).

    Returns (stat, p_value, comment). (None, None, comment) only when no test is
    computable: empty table or constant variable (a single category).

    References: Pearson 1900 (chi-square); Yates 1934 (continuity correction);
    Cochran 1954 (expected-count rule); Fisher exact test.
    R equivalent: stats::fisher.test; stats::chisq.test (correct=TRUE for Yates).
    """
    stats, p_val = None, None
    subdata = cohort[[var_indep, var_dep]].dropna()

    contingency_table = pd.crosstab(subdata[var_indep], subdata[var_dep])
    if contingency_table.empty or contingency_table.size == 0:
        return None, None, "empty contingency table"

    # Constant variable (single row or column): no testable association
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None, None, f"{contingency_table.shape}, constant variable (no test)"

    observed = contingency_table.values

    if contingency_table.shape == (2, 2):
        # Expected counts to arbitrate Fisher vs chi-square
        _, _, _, expected = chi2_contingency(observed, correction=False)
        if (expected < 5).any():
            # Fisher exact: valid even with zero cells
            stats, p_val = fisher_exact(observed)
            comment = "Fisher exact test (expected < 5)"
        else:
            stats, p_val, _, _ = chi2_contingency(observed, correction=True)
            comment = "Chi2 (Yates' correction)"
    else:
        # r x c: no exact Fisher test in scipy -> Pearson chi-square
        stats, p_val, _, expected = chi2_contingency(observed, correction=False)
        if (expected < 5).any():
            comment = f"Chi2 {contingency_table.shape} (expected < 5, approximate)"
        else:
            comment = f"Chi2 {contingency_table.shape}"

    return stats, p_val, comment
def perform_multiclass_test(cohort, var_indep, var_dep, test = "Kruskal-Wallis"):
    """
    Compare a multi-category dependent variable between the two groups of a binary
    independent variable, using the requested test.

    Inputs: the cohort DataFrame, the independent and dependent variable names,
    and the test name ('Kruskal-Wallis', 'Cochran-Armitage' or 'Mann-Whitney').
    Output: (statistic, p_value, comment). Cochran-Armitage assumes ordered
    categories (linear-by-linear trend); Kruskal-Wallis makes no normality
    assumption.

    References: Kruskal & Wallis 1952; Mann & Whitney 1947; Armitage 1955 (linear trend test).
    R equivalent: stats::kruskal.test; stats::wilcox.test; DescTools::CochranArmitageTest.
    """
    stats, p_val, comment = None, None, None
    data = cohort.copy()
    subdata = data[[var_indep, var_dep]].copy().dropna()

    # group_1 = np.array(subdata[subdata[var_indep] == 1][var_dep].copy().dropna()).codes
    # group_2 = np.array(subdata[subdata[var_indep] == 0][var_dep].copy().dropna()).codes
    # n1, n2 = len(group_1), len(group_2)
    group_1 = pd.Categorical(subdata[subdata[var_indep] == 1][var_dep]).codes
    group_2 = pd.Categorical(subdata[subdata[var_indep] == 0][var_dep]).codes
    contingency_table = pd.crosstab(subdata[var_indep], subdata[var_dep]).dropna()
    # rows_with_zero_sum, columns_with_zero_sum = check_contingency(contingency_table)

    if test.lower() in ["kruskal-wallis", "kw"]:
        stats, p_val = kruskal(group_1, group_2)
        comment = "Kruskal-Wallis test"
        # Kruskal-Wallis is robust to violations of the normality assumption,
        # but it can be affected by highly skewed distributions or very unequal sample sizes.
    elif test.lower() in ["cochran-armitage", "ca"]:
        table = Table(contingency_table, shift_zeros=True)
        # Cochran-Armitage trend test = linear-by-linear association test
        # (test_ordinal_association; test_multiclass_association does not exist).
        trend_test = table.test_ordinal_association()
        stats, p_val = trend_test.statistic, trend_test.pvalue
        comment = "Cochran Armitage trend test"
        # identifies linear trends across these levels.
    elif test.lower() in ["mann-whitney", "mw"]:
        if np.issubdtype(group_1.dtype, np.number):
            stats, p_val = mannwhitneyu(group_1, group_2, method='auto')
            comment = "Mannwhitney's test"
        else:
            comment = "value should not be string"
    else:
        stats, p_val, comment = None, None, None
    return stats, p_val, comment
def perform_OR(cohort, var_indep, var_dep):
    """
    Compute the odds ratio and its 95% confidence interval from a 2x2 table.

    Inputs: the cohort DataFrame and the two binary variable names. Output:
    (OR, std of log(OR), lower 95% CI, upper 95% CI, corrected, comment). A 0.5
    Haldane-Anscombe correction is applied when any cell is zero; the OR is only
    defined for a 2x2 table, otherwise None is returned with the table shape.

    References: Woolf 1955 (logit CI for the odds ratio); Haldane 1956 & Anscombe 1956 (0.5 correction).
    R equivalent: epitools::oddsratio(method="wald"); DescTools::OddsRatio.
    """
    data = cohort.copy()
    subdata = data[[var_indep, var_dep]].copy().dropna()

    # Compute the contingency table
    contingency_table = pd.crosstab(subdata[var_indep], subdata[var_dep])

    # Check that the contingency table is indeed 2x2
    if contingency_table.shape != (2, 2):
        OR, std_log_OR, IC95l, IC95u, corrected = None, None, None, None, None
        comment = f'{contingency_table.shape}'

        return OR, std_log_OR, IC95l, IC95u, corrected, comment

    else:
        comment = None

    if (contingency_table.values == 0).any():
        contingency_table += 0.5
        corrected = 'yes'
    else:
        corrected = 'no'

    # Extract the values from the contingency table
    a = contingency_table.iloc[0, 0]  # Exposed with the event
    b = contingency_table.iloc[0, 1]  # Exposed without the event
    c = contingency_table.iloc[1, 0]  # Unexposed with the event
    d = contingency_table.iloc[1, 1]  # Unexposed without the event

    # Compute the Odds Ratio (OR)
    OR = (a * d) / (b * c)

    # Compute the standard deviation of log(OR)
    std_log_OR = np.sqrt(1/a + 1/b + 1/c + 1/d)

    # Compute the 95% confidence interval
    z = norm.ppf(0.975)  # Z-value for 95% confidence
    log_OR = np.log(OR)
    IC95l = np.exp(log_OR - z * std_log_OR)
    IC95u = np.exp(log_OR + z * std_log_OR)

    return OR, std_log_OR, IC95l, IC95u, corrected, comment
def check_equal_var(group_1, group_2):
    """
    Test the equality of variances of two groups with an F-test.

    Inputs: two arrays of values. Output: True when the equal-variance hypothesis
    is not rejected (two-sided F-test, p > 0.05), False otherwise. The larger
    variance is placed at the numerator so the F-ratio is >= 1.

    References: two-sample F-test for equality of variances (Snedecor F distribution).
    R equivalent: stats::var.test.
    """
    var1 = np.var(group_1, ddof=1)
    var2 = np.var(group_2, ddof=1)

    # Compute the F statistic
    if var1 > var2:
        f_stat = var1 / var2
        dfn = len(group_1) - 1
        dfd = len(group_2) - 1
    else:
        f_stat = var2 / var1
        dfn = len(group_2) - 1
        dfd = len(group_1) - 1

    # Compute the p-value
    p_val_f = 2 * min(f.cdf(f_stat, dfn, dfd), 1 - f.cdf(f_stat, dfn, dfd))  # Two-sided test
    equal_var = p_val_f > 0.05

    return equal_var
def z_test(group_1, group_2, D=0, std_known=False):
    """
    Two-sample z-test on the difference of means.

    Inputs: two arrays, the hypothesized difference D (default 0), and std_known
    as either False (variances estimated from the samples via a pooled standard
    deviation) or a tuple (std1, std2) of known standard deviations. Output:
    (z statistic, two-sided p-value).

    References: two-sample z-test on the difference of means (large-sample normal approximation).
    R equivalent: BSDA::z.test.
    """
    # std_known should be a tuple (std1, std2)
    # Null Hypothesis = mu_1-mu_2 = 0
    # Hypothesized difference (under the null hypothesis)
    mean1 = np.mean(group_1)
    mean2 = np.mean(group_2)
    n1 = len(group_1)
    n2 = len(group_2)

    if std_known is False:
        # Use sample standard deviations and pooled variance
        std1 = np.std(group_1, ddof=1)
        std2 = np.std(group_2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        z_stat = (mean1 - mean2 - D) / (pooled_std * np.sqrt(1/n1 + 1/n2))
    else:
        # Known variances: the standard error is directly sqrt(s1^2/n1 + s2^2/n2),
        # with no additional factor.
        std1, std2 = std_known
        se = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
        z_stat = (mean1 - mean2 - D) / se

    # Compute the p-value associated with this Z statistic
    p_val = 2 * (1 - norm.cdf(np.abs(z_stat)))

    return  z_stat, p_val
def perform_continuous_tests(cohort, var_indep, var_dep, normal_known=False, std_known=False):
    """
    Compare a continuous dependent variable between the two groups of a binary
    independent variable, selecting the test from the distribution.

    Inputs: the cohort DataFrame, the independent (0/1) and dependent variable
    names, and optional flags to declare normality/variances as known. Output:
    (statistic, p_value, comment). Decision rule: Student t-test (Welch when
    variances differ) if both groups pass Shapiro; otherwise a z-test when both
    groups have n >= 30 (central limit theorem), else Mann-Whitney. Returns None
    when either group has fewer than 5 observations.

    References: Student 1908 (t-test); Welch 1947 (unequal-variance t); Mann & Whitney 1947.
    R equivalent: stats::t.test (var.equal toggles Student/Welch); stats::wilcox.test.
    """
    stats, pval = None, None
    data = cohort.copy()
    subdata = data[[var_indep, var_dep]].copy()
    group_1 = np.array(subdata[subdata[var_indep] == 1][var_dep].copy().dropna()).astype(float)
    group_2 = np.array(subdata[subdata[var_indep] == 0][var_dep].copy().dropna()).astype(float)
    n1, n2 = len(group_1), len(group_2)

    if n1 < 5 or n2 < 5:
        return None, None, f"Not enough subdata (n1={n1}, n2={n2})"
    else:
        if normal_known is False:
            normal_1 = shapiro(group_1)[1] > 0.05
            normal_2 = shapiro(group_2)[1] > 0.05
        else:
            normal_1, normal_2 = True, True

        if normal_1 and normal_2:
            equal_var = check_equal_var(group_1, group_2)
            stats, pval = ttest_ind(group_1, group_2, equal_var=equal_var)
            comment = 'T-test'
        else:
            if n1 >= 30 and n2 >= 30:
                stats, pval = z_test(group_1, group_2, std_known=std_known)
                comment = 'Z-Test (TCL)'
            else:
                stats, pval = mannwhitneyu(group_1, group_2, method='auto')
                comment = 'Mann_Whitney'
    return stats, pval, comment
def get_critical(pval):
    """
    Map a p-value to a significance star string.

    Inputs: a p-value. Output: '***' (p < 0.001), '**' (p < 0.01), '*' (p < 0.05),
    or None otherwise or when the input is not a float.
    """
    if isinstance(pval, float):
        if pval < 0.001:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        return None
    else:
        return None
def export_test_continuous_df(cont_results):
    """
    Convert the `cont_results` dict from `pipeline_univariate_tests` into a
    pandas.DataFrame ready for an Excel sheet, with:

    - columns:
      ['Variable', 'n_0', 'Group0 (mean \u00B1 std)',
       'n_1', 'Group1 (mean \u00B1 std)',
       'Delta', 'Stats', 'Pval', 'critical', 'Comment']


    Parameters
    ----------
    cont_results : dict
        Per-variable results for continuous variables, structured as:
        { var: {mean_0, std_0, n_0, mean_1, std_1, n_1, Delta, Stats, Pval, ...}, ... }

    Returns
    -------
    result_df : pandas.DataFrame
        One row per variable, ready to be exported to Excel with
        `df.to_excel(..., sheet_name=...)`.
    """

    # Build the continuous DataFrame in the new format

    cols = [
        "Variable",
        "Level",
        "Group0",
        "Group1",
        "Delta/OR(IC95%)",
        "Pval",
        "critical",
        "Test",
        "Stats",
        "Missing data"
    ]
    rows = []
    for var, res in cont_results.items():
        m0 = res.get("mean_0", "")
        s0 = res.get("std_0", "")
        m1 = res.get("mean_1", "")
        s1 = res.get("std_1", "")
        n_0 = res.get("n_0", "")
        n_1 = res.get("n_1", "")
        rows.append({
            "Variable": var,
            "Group0": f"{m0} ± {s0} (n={n_0})" if m0 != "" and s0 != "" else "",
            "Group1": f"{m1} ± {s1} (n={n_1})" if m1 != "" and s1 != "" else "",
            "Delta/OR(IC95%)": res.get("Delta", ""),
            "Stats": res.get("Stats", ""),
            "Pval": res.get("Pval", ""),
            "critical": res.get("critical", ""),
            "Test": res.get("Comment", ""),
            "Missing data": res.get("Missing data", "")
        })
    result_df = pd.DataFrame(rows, columns=cols).fillna("")

    return result_df
def export_test_dicho_df(dicho_results):
    """
    Convert the `dicho_results` dict from `pipeline_univariate_tests` into a
    DataFrame ready to be written to Excel, with columns:
      - Variable
      - Level
      - Group0
      - Group1
      - OR (95% CI)
      - Pval
      - critical
      - corrected Pval
      - Test
      - Stats
      - Missing data
    Each category ("level") appears on its own row, and the overall statistics
    (OR, p-value, etc.) are shown only on the first row of the variable.
    """
    header = [
        "Variable", "Level", "Group0", "Group1",
        "Delta/OR(IC95%)", "Pval", "critical",
        "corrected Pval", "Test", "Stats", "Missing data"
    ]
    rows = []

    for var, res in dicho_results.items():
        # extraire infos
        levels = res.get("Label_dep", [])
        counts = res.get("Count_subclasse", [])
        totals = res.get("Count", [None, None])
        n0, n1 = totals if len(totals) >= 2 else (None, None)

        # formater l'OR et son IC
        or_v   = res.get("OR")
        ic_l   = res.get("IC95l")
        ic_u   = res.get("IC95u")
        or_ci  = f"{or_v} ({ic_l}-{ic_u})" if None not in (or_v, ic_l, ic_u) else ""

        first = True
        for i, lvl in enumerate(levels):
            cnt0 = counts[i][0] if i < len(counts) and len(counts[i]) > 0 else None
            cnt1 = counts[i][1] if i < len(counts) and len(counts[i]) > 1 else None

            # build the row
            row = {
                "Variable": var if first else "",
                "Level":    lvl,
                "Group0":   f"{cnt0} ({cnt0/n0*100:.1f}%)" if n0 and cnt0 is not None else "",
                "Group1":   f"{cnt1} ({cnt1/n1*100:.1f}%)" if n1 and cnt1 is not None else "",
            }

            if first:
                row.update({
                    "Delta/OR(IC95%)":    or_ci,
                    "Pval":          res.get("Pval", ""),
                    "critical":      res.get("critical", ""),
                    "corrected Pval":res.get("corrected", ""),
                    "Test":          res.get("comment_stat", ""),
                    "Stats":         res.get("Stats", ""),
                    "Missing data":  res.get("Missing data", "")
                })
                first = False
            else:
                # leave empty to avoid repetition
                for col in header[4:]:
                    row[col] = ""

            rows.append(row)

    return pd.DataFrame(rows, columns=header)
def export_test_multiclass_df(multiclass_results):
    """
    Convert the `multiclass_results` dict from `pipeline_univariate_tests` into a
    DataFrame ready to be written to Excel, with columns:
      - Variable
      - Level
      - Group0
      - Group1
      - Stats
      - Pval
      - critical
      - Test
      - Missing data

    Each category ("level") appears on its own row, and the overall statistics
    (Stats, p-value, etc.) are shown only on the first row of the variable.
    """
    header = [
        "Variable", "Level", "Group0", "Group1", "Delta/OR(IC95%)", "Pval", "critical", "corrected Pval",
        "Test", "Stats", "Missing data"
    ]
    rows = []

    for var, res in multiclass_results.items():
        levels = res.get("Label_dep", [])
        counts = res.get("Count_subclasse", [])
        totals = res.get("Count", [None, None])
        n0, n1 = (totals[0], totals[1]) if len(totals) >= 2 else (None, None)

        stats = res.get("Stats", "")
        pval  = res.get("Pval", "")
        crit  = res.get("critical", "")
        test  = res.get("comment", "")
        miss  = res.get("Missing data", "")

        first = True
        for i, lvl in enumerate(levels):
            cnt0 = counts[i][0] if i < len(counts) and len(counts[i]) > 0 else None
            cnt1 = counts[i][1] if i < len(counts) and len(counts[i]) > 1 else None

            row = {
                "Variable": var if first else "",
                "Level":    lvl,
                "Group0":   f"{cnt0} ({cnt0/n0*100:.1f}%)" if n0 and cnt0 is not None else "",
                "Group1":   f"{cnt1} ({cnt1/n1*100:.1f}%)" if n1 and cnt1 is not None else "",
            }

            if first:
                row.update({
                    "Stats":        stats,
                    "Pval":         pval,
                    "critical":     crit,
                    "Test":         test,
                    "Missing data": miss,
                })
                first = False
            else:
                # clear the stats columns to avoid repetition
                for col in header[4:]:
                    row[col] = ""

            rows.append(row)

    return pd.DataFrame(rows, columns=header)
def pipeline_univariate_tests(db : pd.DataFrame,  var_indep : str, *,
                              cont_vars: list | str = None, dicho_vars: list | str = None, multiclass_vars: list | str = None,
                              multiclass_test: str ="Kruskal-Wallis",
                              ordered_vars: list = None,
                              save_path: str | Path =None, verbose=False):
    """
    Run univariate association tests of a binary independent variable against
    continuous, dichotomous and multi-category dependent variables.

    Inputs: the DataFrame, the independent variable name, and keyword lists of
    continuous / dichotomous / multiclass variables; multiclass_test selects the
    multi-category test; ordered_vars optionally reorders the output; save_path
    optionally writes the result. Output: a single combined DataFrame of test
    results. Side effect: writes an Excel file when save_path is set.
    """
    data = db.copy()

    missing_data = get_missing_data(db)
    missing_dict = missing_data.set_index("Variable")["Missing data"].to_dict()

    if cont_vars is not None:
        if not isinstance(cont_vars, list):
            cont_vars = [cont_vars]
        results_dict = {}
        for var_dep in cont_vars:
            if var_dep in data.columns and var_dep is not var_indep:
                if verbose:
                    print(f"Processing Continuous:  Dependent: {var_dep}")
                results_dict[var_dep] = {}
                try:
                    mean1, mean2, Delta, n1, n2, std1, std2 = perform_distrib_charact(data, var_indep=var_indep,
                                                                                      var_dep=var_dep)
                    stats, pval, comment = perform_continuous_tests(data, var_indep=var_indep, var_dep=var_dep)
                except ValueError as e:
                    raise ValueError(f"Error while processing continuous variable '{var_dep}': {e}")

                results_dict[var_dep] = {
                    'n_0': n1,
                    'mean_0': round(mean1, 1) if mean1 is not None else mean1,
                    'std_0': round(std1, 3) if std1 is not None else std1,
                    'n_1': n2,
                    'mean_1': round(mean2, 1) if mean2 is not None else mean2,
                    'std_1': round(std2, 3) if std2 is not None else std2,
                    'Delta': round(Delta, 1) if Delta is not None else Delta,
                    'Stats': round(stats, 3) if stats is not None else stats,
                    'Pval': pval,
                    'critical': get_critical(pval),
                    'Comment': comment,
                    'Missing data' : missing_dict.get(var_dep)

                }
                continous_df = export_test_continuous_df(results_dict)

    if dicho_vars is not None:
        if not isinstance(dicho_vars, list):
            dicho_vars = [dicho_vars]
        results_dict = {}
        for var_dep in dicho_vars:
            if var_dep in data.columns and var_dep != var_indep:
                if verbose:
                    print(f"Processing Categorical: Dependent: {var_dep}")
                results_dict[var_dep] = {}
                try:
                    n1, n2, kclass, label_indep, label_dep, nk1, nk2 = perform_distrib_charact(data, var_indep=var_indep,
                                                                                               var_dep=var_dep,
                                                                                               var_type="categorial")
                    OR, std_log_OR, IC95l, IC95u, corrected, comment_OR = perform_OR(data, var_indep=var_indep,
                                                                                     var_dep=var_dep)
                    stats, pval, comment_stats = perform_categorial_tests(data, var_indep=var_indep, var_dep=var_dep)
                except ValueError as e:
                    raise ValueError(f"Error while processing categorical variable '{var_dep}': {e}")

                if nk2 is None:
                    count_subclasse = [[x] for x in nk1]
                else:
                    count_subclasse = list(map(list, zip(nk1, nk2)))
                results_dict[var_dep] = {
                    'Exposed': label_indep,
                    'Count': [n1,n2],
                    'OR': round(OR, 3) if OR is not None else OR,
                    'std_log_OR': round(std_log_OR, 3) if std_log_OR is not None else std_log_OR,
                    'IC95l': round(IC95l, 3) if IC95l is not None else IC95l,
                    'IC95u': round(IC95u, 3) if IC95u is not None else IC95u,
                    'corrected': corrected,
                    'comment_OR': comment_OR,
                    'Stats': round(stats, 3) if stats is not None else stats,
                    'Pval': pval,
                    'critical': get_critical(pval),
                    'comment_stat': comment_stats,
                    'kclass': kclass,
                    'Label_dep': label_dep,
                    'Count_subclasse': count_subclasse,
                    'Missing data': missing_dict.get(var_dep)

                }
                dicho_df = export_test_dicho_df(results_dict)

    if multiclass_vars is not None:
        if not isinstance(multiclass_vars, list):
            multiclass_vars = [multiclass_vars]
        results_dict = {}
        for var_dep in multiclass_vars:
            if var_dep in data.columns and var_dep is not var_indep:
                if verbose:
                    print(f"Processing Multiclass: Dependent: {var_dep}")
                results_dict[var_dep] = {}
                try:
                    n1, n2, kclass, label_indep, label_dep, nk1, nk2 = perform_distrib_charact(data, var_indep=var_indep,
                                                                                               var_dep=var_dep,
                                                                                               var_type="categorial")
                    stats, pval, comment = perform_multiclass_test(data, var_indep=var_indep, var_dep=var_dep,
                                                                test=multiclass_test)
                except ValueError as e:
                    raise ValueError(f"Error while processing multiclass variable '{var_dep}': {e}")
                if nk2 is None:
                    count_subclasse = [[x] for x in nk1]
                else:
                    count_subclasse = list(map(list, zip(nk1, nk2)))
                results_dict[var_dep] = {
                    'Exposed': label_indep,
                    'Count': [n1,n2],
                    'Stats': round(stats, 3) if stats is not None else stats,
                    'Pval': pval,
                    'critical': get_critical(pval),
                    'comment': comment,
                    'kclass': kclass,
                    'Label_dep': label_dep,
                    'Count_subclasse': count_subclasse,
                    'Missing data' : missing_dict.get(var_dep)
                }
                multiclass_df = export_test_multiclass_df(results_dict)
    test_all = pd.concat(
        [continous_df, dicho_df, multiclass_df],
        axis=0,  # vertical concatenation
        ignore_index=True  # renumber the index from 0 to n-1
    )
    if ordered_vars is not None:
        test_all = reorder_summary_by_variable(test_all, ordered_vars)
    if save_path:
        if save_path.endswith('.xlsx'):
            file_path = save_path
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if multiclass_test.lower() in ["cochran-armitage", "ca"]:
                tag_test = "-CA"
            elif multiclass_test.lower() in ["kruskal-wallis", "kw"]:
                tag_test = "-KW"
            else:
                tag_test = "-MW"
            file_path = os.path.join(save_path, f'univ_{var_indep}_{tag_test}.xlsx')
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            test_all.to_excel(writer, sheet_name="Tests", index=False)

    return test_all


def check_contingency(contingency_table):
    """
    Detect empty rows/columns in a contingency table and apply a continuity
    correction when a zero cell is present.

    Inputs: a contingency table (DataFrame). Output: (result dict, possibly
    corrected table). A 0.5 shift is added to every cell when at least one cell
    is zero, and 'correction' is set to 'continuity' in the result.
    """
    zero_rows = contingency_table.sum(axis=1) == 0
    zero_columns = contingency_table.sum(axis=0) == 0

    rows_with_zero_sum = contingency_table.index[zero_rows].tolist()
    columns_with_zero_sum = contingency_table.columns[zero_columns].tolist()

    result = {}

    if (contingency_table == 0).any().any():
        contingency_table += 0.5
        result['correction'] = 'continuity'

    return result, contingency_table

### ANOVA
def perform_anova(data, var_indep, var_dep):
    """
    Fit a (one-way or factorial) ANOVA model and return the effect statistics plus
    Tukey post-hoc results.

    Inputs: a DataFrame, the independent variable(s) (a name or a list of factors
    for a factorial model with interaction) and the continuous dependent variable.
    Output: a dict with per-factor F, p-value and mean square, the residual mean
    square, and (single-factor case) the Tukey HSD summary. Uses type II sums of
    squares.

    References: Fisher 1925 (analysis of variance); Tukey 1949 (HSD post-hoc).
    R equivalent: stats::aov + car::Anova(type=2) + stats::TukeyHSD.
    """
    subdata = data[[var_indep, var_dep]].copy()
    if isinstance(var_indep, str):
        formula = f'{var_dep} ~ C({var_indep})',
    elif isinstance(var_indep, list):
        formula = f'{var_dep} ~ ' + ' * '.join([f'C({factor})' for factor in var_indep])

    model = ols(formula, data=subdata).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    result = {}
    for factor in anova_table.index[:-1]:  # Skip the last entry as it is for the residuals
        result[f'{factor}_F'] = anova_table.loc[factor, 'F']
        result[f'{factor}_pval'] = anova_table.loc[factor, 'PR(>F)']
        result[f'{factor}_mean_sq'] = anova_table.loc[factor, 'mean_sq']

    # Residual mean_sq
    result['Residual_mean_sq'] = anova_table.loc['Residual', 'mean_sq']

    # Tukey post-hoc test

    if isinstance(var_indep, str):
        tukey = pairwise_tukeyhsd(endog=subdata[var_dep], groups=subdata[var_indep], alpha=0.05)
        result['Tukey'] = tukey.summary()
        # tukey_df = pd.DataFrame(data=tukey_summary.data[1:], columns=tukey_summary.data[0])
        # important_info = tukey_df[['group1', 'group2', 'meandiff', 'lower', 'upper', 'p-adj']]

    return result

### LOGIT
def logistic_regression_univariable(data, outcome, predictor):
    """
    Fit a univariate logistic model and return OR, 95% CI, coefficient, SE,
    p-value and the number of observations.

    Warning: under (quasi-)separation, the maximum-likelihood estimate diverges -
    the coefficient and standard error blow up, the OR becomes huge with a CI
    approaching (0, inf) and the Wald p-value collapses towards 1 (an artefact,
    not an absence of effect). A warning is emitted in this case; a penalized
    (Firth) regression would be required for a reliable estimator.

    References: logistic regression (Wald OR + CI); Firth 1993 (penalized-likelihood fix for separation).
    R equivalent: stats::glm(family = binomial); logistf::logistf (Firth).
    """
    df = data[[outcome, predictor]].dropna()
    n = len(df)
    if n == 0:
        raise ValueError(f"No data available for {predictor}")

    try:
        df[predictor] = df[predictor].astype(float)
    except Exception as e:
        return None, None, None, None, None, None, None

    # For non-numeric predictors with 2 categories, encode them as 0/1
    if not pd.api.types.is_numeric_dtype(df[predictor]):
        if df[predictor].nunique() == 2:
            df[predictor] = pd.Categorical(df[predictor]).codes
        else:
            raise ValueError(f"The variable {predictor} is not numeric and does not have exactly 2 categories.")

    y = df[outcome]
    X = df[[predictor]]
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=False)
    except Exception as e:
        # On error (e.g. singular matrix), return None
        return (None, None, None, None, None, None, None)
    try:
        coef = model.params[predictor]
        OR = np.exp(coef)
        se = model.bse[predictor]
        lower = np.exp(coef - 1.96 * se)
        upper = np.exp(coef + 1.96 * se)
        pval = model.pvalues[predictor]
    except Exception as e:
        return (None, None, None, None, None, None, None)

    # (Quasi-)separation detection: aberrant coefficient/standard error.
    # The OR and CI are still returned, but flagged as unreliable.
    if not np.isfinite(se) or abs(coef) > 15 or se > 15:
        warnings.warn(
            f"logistic_regression_univariable: probable (quasi-)separation for "
            f"'{predictor}' (coef={coef:.2f}, se={se:.2f}). Wald OR/CI/p unreliable "
            f"- consider a Firth regression.",
            RuntimeWarning,
        )
    return OR, lower, upper, coef, se, pval, n
def export_dict_log_result(data, predictor, outcome, continuous=True, ordinal=False, verbose=False):
    """
    Return a results dictionary for a predictor in a univariate logistic regression.

    For a categorical variable with more than 2 categories, the OR of each level is
    estimated VERSUS THE REFERENCE via a single global model (outcome ~ all dummies,
    reference dropped); the 'ref' row carries the global p-value (likelihood-ratio
    test). If ``ordinal=True``, a 'trend' entry is added: a trend OR obtained by
    entering the variable as a single linear term (1 OR per increment).

    References: Wilks 1938 (likelihood-ratio test for the global p-value across levels).
    R equivalent: stats::glm + stats::anova(., test="LRT") or car::Anova(., type="II").
    """
    result = {}
    missing_data = get_missing_data(data)
    missing_dict = missing_data.set_index("Variable")["Missing data"].to_dict()
    if continuous is True:
        if predictor in data.columns and predictor != outcome:
            if verbose:
                print(f"Processing Continuous predictor: {predictor}")
            try:
                mean0, mean1, Delta, n0, n1, std0, std1 = perform_distrib_charact(
                    data, var_indep=outcome, var_dep=predictor, var_type="continuous")
            except Exception as e:
                mean0 = mean1 = std0 = std1 = n0 = n1 = None
            try:
                OR, lower, upper, coef, se, pval, n_used = logistic_regression_univariable(data, outcome, predictor)
            except Exception as e:
                raise ValueError(f"Error while processing continuous variable '{predictor}': {e}")
            result = {
                'n': n_used,
                'mean_0': round(mean0, 1) if mean0 is not None else None,
                'std_0': round(std0, 3) if std0 is not None else None,
                'n_0': n0,
                'mean_1': round(mean1, 1) if mean1 is not None else None,
                'std_1': round(std1, 3) if std1 is not None else None,
                'n_1': n1,
                'Missing': missing_dict.get(predictor),
                'OR': round(OR, 3) if OR is not None else None,
                'CI_lower': round(lower, 3) if lower is not None else None,
                'CI_upper': round(upper, 3) if upper is not None else None,
                'coef': round(coef, 3) if coef is not None else None,
                'se': round(se, 3) if se is not None else None,
                'Pval': pval,
                'critical': get_critical(pval)
            }
    elif continuous is False:
        if predictor in data.columns and predictor != outcome:
            if verbose:
                print(f"Processing Categorical predictor: {predictor}")
            try:
                n0, n1, kclass, label_outcome, label_pred, nk0, nk1 = perform_distrib_charact(
                    data, var_indep=outcome, var_dep=predictor, var_type="categorial")
            except Exception as e:
                raise ValueError(f"Error in distribution for categorical variable '{predictor}': {e}")
            # If kclass > 2, recode into dummies and compute the global p-value
            if kclass > 2:
                # Dummies WITH the reference dropped (drop_first=True) for numeric AND
                # text variables: each dummy represents one level, compared to the
                # reference category (the first one, dropped).
                dummies = pd.get_dummies(data[predictor], drop_first=True).astype(float)
                # Global multivariable model: outcome ~ const + dummies. The
                # coefficients are the log-OR of each level VERSUS THE REFERENCE
                # (not "one-versus-rest").
                X_global = sm.add_constant(dummies)
                y_global = pd.to_numeric(data[outcome].loc[dummies.index], errors='coerce')
                mask = y_global.notna()
                y_global, X_global = y_global.loc[mask], X_global.loc[mask]
                try:
                    model_global = sm.Logit(y_global, X_global).fit(disp=False)
                    global_pval = model_global.llr_pvalue
                except Exception:
                    model_global, global_pval = None, "Fail"

                dummy_results = {}
                ct = pd.crosstab(data[outcome], data[predictor])
                total0 = ct.loc[0].sum() if 0 in ct.index else 0
                total1 = ct.loc[1].sum() if 1 in ct.index else 0
                n_log = int(model_global.nobs) if model_global is not None else None

                # Reference: OR = 1 by convention, carries the GLOBAL p-value (LLR test).
                ref = pd.Categorical(data[predictor]).categories[0]
                ref_count0 = ct.loc[0, ref] if (0 in ct.index and ref in ct.columns) else 0
                ref_count1 = ct.loc[1, ref] if (1 in ct.index and ref in ct.columns) else 0
                dummy_results["ref"] = {
                    'n_log': n_log,
                    'Missing': missing_dict.get(predictor),
                    'OR': 1.0,
                    'CI_lower': "",
                    'CI_upper': "",
                    'coef': 0.0,
                    'se': 0.0,
                    'Pval': global_pval,  # global p-value for all levels combined
                    'critical': get_critical(global_pval),
                    'Count_0': ref_count0,
                    'Perc_0': f"{(ref_count0 / total0 * 100) if total0 > 0 else 0:.1f}%",
                    'Count_1': ref_count1,
                    'Perc_1': f"{(ref_count1 / total1 * 100) if total1 > 0 else 0:.1f}%"
                }
                # Each level: OR / 95% CI / p VERSUS THE REFERENCE, read from the global model.
                for col in dummies.columns:
                    if model_global is not None and col in model_global.params.index:
                        coef = model_global.params[col]
                        se = model_global.bse[col]
                        OR = np.exp(coef)
                        lower, upper = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
                        pval_dummy = model_global.pvalues[col]
                    else:
                        OR = lower = upper = coef = se = pval_dummy = None

                    ct_dummy = pd.crosstab(data[outcome], dummies[col])
                    count0 = ct_dummy.loc[0, 1.0] if (0 in ct_dummy.index and 1.0 in ct_dummy.columns) else 0
                    count1 = ct_dummy.loc[1, 1.0] if (1 in ct_dummy.index and 1.0 in ct_dummy.columns) else 0
                    dummy_results[col] = {
                        'n_log': n_log,
                        'Missing': missing_dict.get(predictor),
                        'OR': round(OR, 3) if OR is not None else None,
                        'CI_lower': round(lower, 3) if lower is not None else None,
                        'CI_upper': round(upper, 3) if upper is not None else None,
                        'coef': round(coef, 3) if coef is not None else None,
                        'se': round(se, 3) if se is not None else None,
                        'Pval': pval_dummy,
                        'critical': get_critical(pval_dummy),
                        'Count_0': count0,
                        'Perc_0': f"{(count0 / total0 * 100) if total0 > 0 else 0:.1f}%",
                        'Count_1': count1,
                        'Perc_1': f"{(count1 / total1 * 100) if total1 > 0 else 0:.1f}%"
                    }

                # ORDINAL variable: additionally add a TREND OR - the variable is
                # entered as a single linear term (1 OR per level increment, under
                # the log-linearity assumption).
                if ordinal:
                    trend_df = data[[outcome, predictor]].copy()
                    if not pd.api.types.is_numeric_dtype(trend_df[predictor]):
                        # ordered codes (preserve the order of the sorted categories)
                        trend_df[predictor] = pd.Categorical(trend_df[predictor]).codes
                    try:
                        tOR, tlo, tup, tcoef, tse, tpval, tn = logistic_regression_univariable(
                            trend_df, outcome, predictor)
                    except Exception:
                        tOR = tlo = tup = tcoef = tse = tpval = tn = None
                    dummy_results["trend"] = {
                        'n_log': tn,
                        'Missing': missing_dict.get(predictor),
                        'OR': round(tOR, 3) if tOR is not None else None,
                        'CI_lower': round(tlo, 3) if tlo is not None else None,
                        'CI_upper': round(tup, 3) if tup is not None else None,
                        'coef': round(tcoef, 3) if tcoef is not None else None,
                        'se': round(tse, 3) if tse is not None else None,
                        'Pval': tpval,
                        'critical': get_critical(tpval),
                        'Count_0': "", 'Perc_0': "", 'Count_1': "", 'Perc_1': ""
                    }

                result = dummy_results
            else:
                try:
                    OR, lower, upper, coef, se, pval, n_used = logistic_regression_univariable(data, outcome, predictor)
                except Exception as e:
                    raise ValueError(f"Error while processing categorical variable '{predictor}': {e}")
                count_subclasse = list(map(list, zip(nk0, nk1))) if nk1 is not None else [[x] for x in nk0]
                result = {
                    'n_log': n_used,
                    'n_outcome_0': n0,
                    'n_outcome_1': n1,
                    'Missing': missing_dict.get(predictor),
                    'OR': round(OR, 3) if OR is not None else None,
                    'CI_lower': round(lower, 3) if lower is not None else None,
                    'CI_upper': round(upper, 3) if upper is not None else None,
                    'coef': round(coef, 3) if coef is not None else None,
                    'se': round(se, 3) if se is not None else None,
                    'Pval': pval,
                    'critical': get_critical(pval),
                    'kclass': kclass,
                    'Label_outcome': label_outcome,
                    'Label_pred': label_pred,
                    'Count': [n0, n1],
                    'Count_subclasse': count_subclasse
                }
    return result
def export_univariate_summary(test_results, ordered_var=None):
    """
    Build a summary table combining the results for continuous, dichotomous and
    multiclass variables, with the format:
      - Variable
      - Level (for non-continuous variables, the level/category; empty for continuous)
      - Group0: e.g. for continuous variables "mean ± SD (n=n0)",
                    for non-continuous variables "Count (Perc)" for outcome==0,
      - Group1: same for outcome==1,
      - OR (95% CI)
      - p-value (shown only for the reference or the first row)
      - critical
      - Missing data

    Parameters:
    -----------
    test_results: dict
         Dictionary of univariate results split into "Continuous", "Dichotomic"
         and "Multiclass" sub-keys.
    ordered_var: list, optional
         Ordered list of variable names used to reorder the final table.

    Returns:
    --------
    pd.DataFrame: The exported summary table.
    """
    # Collect the rows for each type
    rows = []
    for test_type, results in test_results.items():
        if test_type == "Continuous":
            for var, res in results.items():
                row = export_univariate_continuous_summary(var, res)
                rows.append(row)
        else:
            for var, res in results.items():
                if isinstance(res, dict) and "n_outcome_0" in res:
                    # Non-dummy case (kclass <= 2)
                    row = export_univariate_dichotomous_summary(var, res)
                    rows.extend(row)
                else:
                    # Dummy case: res is a dictionary keyed by dummy column name.
                    # The reference (key "ref") carries the global p-value; for the
                    # other levels, the p-value is left empty.
                    for dummy_name, dummy_res in res.items():
                        row = export_univariate_multiclass_summary(var, dummy_name, dummy_res)
                        rows.append(row)


    # Specify the column order
    cols = ["Variable", "Level", "Group0", "Group1", "OR (95% CI)", "p-value", "critical", "Missing data"]
    summary_df = pd.DataFrame(rows, columns=cols)

    # Reorder if an ordered list is provided
    if ordered_var is not None:
        summary_df = reorder_summary_by_variable(summary_df, ordered_var)

    return summary_df
def export_univariate_continuous_summary(var, res):
    """
    Build a summary row for a continuous variable.

    Inputs:
    var: str
         The variable name.
    res: dict
         The per-variable results (e.g. mean_0, std_0, n_0, mean_1, std_1, n_1,
         OR, CI_lower, CI_upper, coef, se, Pval, n).

    Returns:
         A dict representing one summary row with the target columns.
    """

    row = {}
    row["Variable"] = var
    row["Level"] = ""
    # Format the measures for outcome = 0 and outcome = 1
    group0 = f"{res.get('mean_0', 'NA')} ± {res.get('std_0', 'NA')} (n={res.get('n_0', 'NA')})"
    group1 = f"{res.get('mean_1', 'NA')} ± {res.get('std_1', 'NA')} (n={res.get('n_1', 'NA')})"
    row["Group0"] = group0
    row["Group1"] = group1
    # Format de l'effet
    if res.get("OR") is not None:
        row["OR (95% CI)"] = f"{res.get('OR')} ({res.get('CI_lower')}-{res.get('CI_upper')})"
    else:
        row["OR (95% CI)"] = ""
    row["p-value"] = res.get("Pval", "")
    row["critical"] = res.get("critical", "")
    n_used = res.get("n", res.get("n_log", 0))
    row["Missing data"] = res.get("Missing", "")

    return row
def export_univariate_dichotomous_summary(var, res):
    """
    Build the summary rows for a dichotomous variable (one row per level).

    Inputs: the variable name and its results dict (n_outcome_0, n_outcome_1,
    Count_subclasse, Label_pred, OR, CI, Pval, ...). Output: a list of row dicts;
    the variable name and overall statistics appear only on the first row.
    """
    rows = []
    if isinstance(res, dict) and "n_outcome_0" in res:
        n0 = res.get("n_outcome_0", None)
        n1 = res.get("n_outcome_1", None)
        count_pairs = res.get("Count_subclasse", [])
        level_labels = res.get("Label_pred", [])
        or_ci = f"{res.get('OR')} ({res.get('CI_lower')}-{res.get('CI_upper')})" if res.get("OR") is not None else ""
        pval = res.get("Pval", "")
        missing_str = res.get("Missing", "")
        first_line = True
        for i, level in enumerate(level_labels):
            row = {}
            row["Variable"] = var if first_line else ""
            row["Level"] = level
            try:
                count0_val = count_pairs[i][0]
            except Exception:
                count0_val = None
            try:
                count1_val = count_pairs[i][1]
            except Exception:
                count1_val = None
            if n0 and count0_val is not None:
                perc0 = count0_val / n0 * 100
                row["Group0"] = f"{count0_val} ({perc0:.1f}%)"
            else:
                row["Group0"] = "NA"
            if n1 and count1_val is not None:
                perc1 = count1_val / n1 * 100
                row["Group1"] = f"{count1_val} ({perc1:.1f}%)"
            else:
                row["Group1"] = "NA"
            if first_line:
                row["OR (95% CI)"] = or_ci
                row["p-value"] = pval
                row["critical"] = res.get("critical", "")
                row["Missing data"] = missing_str
                first_line = False
            else:
                row["OR (95% CI)"] = ""
                row["p-value"] = ""
                row["critical"] = ""
                row["Missing data"] = ""
            rows.append(row)
    return rows
def export_univariate_multiclass_summary(var, dummy_name, dummy_res):
    """
    Build a summary row for a categorical variable with more than 2 categories that
    was recoded into dummy variables.
    For these variables, the per-variable result is a dictionary keyed by category
    (e.g. 'ref' for the reference and others for the dummies).

    Inputs: the variable name, the dummy/level name, and its results dict.

    Returns:
         A dict representing one summary row for the multiclass variable.
    """

    # Dummy case: for a multiclass variable, "n_outcome_0" is absent from res.
    row = {}
    # The variable name appears only on the reference row.
    row["Variable"] = var if dummy_name == "ref" else ""
    row["Level"] = "trend (per +1)" if dummy_name == "trend" else dummy_name

    if dummy_name == "trend":
        # No per-category count for the trend.
        row["Group0"] = ""
        row["Group1"] = ""
    else:
        row["Group0"] = f"{dummy_res.get('Count_0', 'NA')} ({dummy_res.get('Perc_0', 'NA')})"
        row["Group1"] = f"{dummy_res.get('Count_1', 'NA')} ({dummy_res.get('Perc_1', 'NA')})"

    if dummy_name == "ref":
        # Reference: OR=1 (no CI) and global p-value.
        row["OR (95% CI)"] = f"{dummy_res.get('OR')}" if dummy_res.get("OR") is not None else ""
        row["p-value"] = dummy_res.get("Pval", "")
        row["Missing data"] = dummy_res.get("Missing", "")
    else:
        # Levels (vs reference) and trend: own OR (95% CI) + p-value.
        row["OR (95% CI)"] = (
            f"{dummy_res.get('OR')} ({dummy_res.get('CI_lower')}-{dummy_res.get('CI_upper')})"
            if dummy_res.get("OR") is not None else ""
        )
        row["p-value"] = dummy_res.get("Pval", "")
        row["Missing data"] = ""
    row["critical"] = dummy_res.get("critical", "")
    return row
def pipeline_univariate_log(db, outcome, *,
                            cont_vars: list | str, dicho_vars: list | str = None, multiclass_vars: list | str = None,
                            ordinal_vars: list | str = None,
                            save_path=None,
                            ordered_var=None, verbose=False):
    """
    Pipeline running univariate logistic regressions.

    For each predictor (continuous, categorical or multiclass) other than the
    outcome, a univariate logistic model is fitted and the results are stored in a
    dictionary.

    For continuous variables, the following are also added:
      - mean_0, std_0, n_0 (for outcome=0)
      - mean_1, std_1, n_1 (for outcome=1)
    For categorical/multiclass variables, the distribution is retrieved via
    perform_distrib_charact and the 'Count' key is built as [n_outcome0, n_outcome1].

    The number of missing values (Missing) is also computed for each predictor.

    Returns:
      test_results : dict, with keys 'Continuous', 'Dichotomic' and 'Multiclass'.

    Side effect: writes an Excel summary when save_path is set.
    """
    test_results = {
        'Continuous': {},
        'Dichotomic': {},
        'Multiclass': {}
    }
    data = db.copy()
    total_sample = len(data)
    results = {}

    # Continuous variables
    if cont_vars is not None:
        if not isinstance(cont_vars, list):
            cont_vars = [cont_vars]
        for predictor in cont_vars:
            test_results['Continuous'][predictor] = export_dict_log_result(data, predictor, outcome)

    # Dichotomous variables
    if dicho_vars is not None:
        if not isinstance(dicho_vars, list):
            dicho_vars = [dicho_vars]
        for predictor in dicho_vars:
            test_results['Dichotomic'][predictor] = export_dict_log_result(data, predictor, outcome, continuous=False)

    # Multiclass variables
    if multiclass_vars is not None:
        if not isinstance(multiclass_vars, list):
            multiclass_vars = [multiclass_vars]
        if ordinal_vars is None:
            ordinal_set = set()
        elif isinstance(ordinal_vars, list):
            ordinal_set = set(ordinal_vars)
        else:
            ordinal_set = {ordinal_vars}
        for predictor in multiclass_vars:
            test_results['Multiclass'][predictor] = export_dict_log_result(
                data, predictor, outcome, continuous=False, ordinal=(predictor in ordinal_set))

    # Optional saving to an Excel file
    if save_path:
        if save_path.endswith('.xlsx'):
            file_path = save_path
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, f'univ_log_{outcome}.xlsx')
        # Export the table to Excel
        summary_df = export_univariate_summary(test_results,  ordered_var)
        summary_df.to_excel(file_path, index=False)
    return test_results

def reorder_summary_by_variable(summary_df, ordered_vars):
    """
    Reorder a DataFrame block by block according to the variable order, keeping the
    mini-table rows (Group0, Group1, separators) attached to their parent variable.

    summary_df : pd.DataFrame
        Must contain a "Variable" column in which only the real variables appear in
        ordered_vars. The other rows (labels, counts...) are attached to the last
        valid variable encountered.

    ordered_vars : list[str]
        Desired order for the main variables.
    """
    df = summary_df.copy()

    # Keep only the real variables for ffill; all others become NaN
    df['Variable_group'] = (
        df['Variable']
          .where(df['Variable'].isin(ordered_vars))  # otherwise NaN
          .ffill()
    )

    # Order mapping
    order_mapping = {var: i for i, var in enumerate(ordered_vars)}

    # Assign each row the "rank" of its Variable_group
    df['sort_order'] = df['Variable_group'].map(
        lambda x: order_mapping.get(x, len(ordered_vars))
    )

    # Sort by sort_order (stable sort to keep the intra-group order)
    df = df.sort_values(by='sort_order', kind='mergesort').reset_index(drop=True)

    # Drop the auxiliary columns
    df = df.drop(columns=['Variable_group', 'sort_order'])

    return df
