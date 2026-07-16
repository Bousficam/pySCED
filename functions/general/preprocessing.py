import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import datetime
import re
import string
import difflib
from dataclasses import dataclass, field


def _require_columns(df, columns, func_name="this function"):
    """
    Check that all requested columns exist in ``df`` and raise a clear error
    (with close-name suggestions) otherwise.

    Inputs: a DataFrame, a column name or list of names, and the caller name used
    in the message. Output: None; raises KeyError when a column is missing.

    Avoids cryptic ``KeyError`` messages: it names the missing column(s) and
    proposes the closest existing names (useful for typos or columns not yet
    created).
    """
    if isinstance(columns, str):
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        details = []
        for col in missing:
            close = difflib.get_close_matches(str(col), [str(c) for c in df.columns], n=3, cutoff=0.6)
            if close:
                details.append(f"'{col}' (did you mean: {', '.join(close)} ?)")
            else:
                details.append(f"'{col}'")
        raise KeyError(
            f"{func_name}: missing column(s): {'; '.join(details)}. "
            f"The DataFrame contains {df.shape[1]} columns."
        )


@dataclass
class VariableTypes:
    """
    NAMED result of variable classification (always the same shape, with
    autocompletion), to avoid variable-shaped tuples.

    Attributes
    ----------
    dichotomous / multiclass / continuous : list[str]
        Column names by type (2 categories / 2 < categories < threshold / >= threshold).
    counts : dict[str, int]
        Number of unique values (excluding NaN) per variable, all types combined.
    """
    dichotomous: list = field(default_factory=list)
    multiclass: list = field(default_factory=list)
    continuous: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def to_frame(self):
        """Tabular view: one row per variable (variable, type, n_unique)."""
        rows = []
        for kind, names in (("dichotomous", self.dichotomous),
                            ("multiclass", self.multiclass),
                            ("continuous", self.continuous)):
            for v in names:
                rows.append({"variable": v, "type": kind, "n_unique": self.counts.get(v)})
        return pd.DataFrame(rows, columns=["variable", "type", "n_unique"])



# creat
### Cockcroft-Gault: creatinine clearance = [(140 - age) x weight x k / serum creatinine]
# With k = 1.04 for women and k = 1.23 for men.

## MDRD: 175 * (Creat / 88.4) -1.154 * Age -0.203 * (0.742 if FEMALE) * (1.212 if black subject)
# Levey, A. S., Coresh, J., Greene, T., Stevens, L. A., Zhang, Y., Hendriksen, S., ... & Chronic Kidney Disease Epidemiology Collaboration*. (2006). Using standardized serum creatinine values in the modification of diet in renal disease study equation for estimating glomerular filtration rate. Annals of internal medicine, 145(4), 247-254.
## CKD-EPI ## ClCr = 141 * min(Creat / k) alpha * max(Creat / k) -1.209* 0.993 Age * (1.018 if female sex) * (1.159 if black subject); with k=0.7 for women and 0.9 for men, and alpha=-0.329 for women and -0.411 for men. Serum creatinine is expressed in mg/dl.
# A New Equation to Estimate Glomerular Filtration Rate. Andrew S. Levey, MD; Lesley A. Stevens, MD, MS; Christopher H. Schmid, PhD; Yaping (Lucy) Zhang, MS; Alejandro F. Castro III, MPH; Harold I. Feldman, MD, MSCE; John W. Kusek, PhD; Paul Eggers, PhD; Frederick Van Lente, PhD; Tom Greene, PhD; and Josef Coresh, MD, PhD, MHS, for the CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration). Annals of Internal Medicine 2009;150(9):604-61.


### Dichotomic
# Sex: M/F: 0/1
# nri_hemorr_j1_ecass: 0/HI1/HI2/PH1/PH2: 0/1/2/3/4
# tici*: TICI 0/1/2A/2B/2C/3: 0/1/2/3/4/5, poor TICI < 2B/3  // moderate = 3 // success > 3 ?
# artefact_occul: 0/1/2: 0/interpretable/uninterpretable == GLOS ~ NaN

# MRI -> 1.5T: Sola XQ, 1.5TA: Aera 1.5T/ 3T: Vida XT, 3TS: Skyra 3T
# type_gado: gadoteric acid / Gadobutrol
### arterio_occl: ACI / ACI-E / ACI-I / ACI-L / ACI-T / M1 / M2 / M3 / TB / ACP / VB // ACI + M3 // TB + ACP or TB + VB
## ACI-T: ACI + prox A1+M1
## ACI-E: ACI extraC
## ACI-I: ACI intraC
## ACI-L: ACI + M1
# anest_a: 0: local / 1: conscious / 2: general / 3: conversion
# grade_collateralite = 0/1/2/3/4 -> ASITN/SIR collateral grading scale
#### Subclass:
### Sub-GLOS/Sub-HARM
# HARM_lob -> 1: UL, 2 BL
# HARM_diff -> 1: hemiS UL, hemiS BL
# HARM_flou -> 1: UL, 2: BL ~ dirty CSR
# HARM_CL -> if HARM_diff & HARM_flou != 2 -> TODO(check)
# CA/CP/NO -> 1: UL, 2 BL

# Identification
def extract_excel_columns(xlsx_path, sheet_name=0, col_range=None):
    """
    Return the column names of an Excel sheet, optionally restricted to a range.

    Inputs: the workbook path, the sheet name/index, and an optional column range
    given as spreadsheet letters ("A:O") or numeric indices (1, 15).
    Output: a list of column names. Only the header row is read from disk.
    """
    # Read only the first row to obtain the column names
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, nrows=1)

    if col_range:
        if isinstance(col_range, str):
            col_list = list(string.ascii_uppercase)
            col_list += [f"{x}{y}" for x in string.ascii_uppercase for y in string.ascii_uppercase]
            col_list += [f"{x}{y}{z}" for x in string.ascii_uppercase for y in string.ascii_uppercase for z in
                         string.ascii_uppercase]

            start_col, end_col = col_range.split(":")
            start_idx = col_list.index(start_col)
            end_idx = col_list.index(end_col) + 1
        else:
            # If the range is given as numeric indices (e.g. 1:15)
            start_idx, end_idx = col_range

        return df.columns[start_idx:end_idx].tolist()

    return df.columns.tolist()
def extraire_variables_excel(xlsx_path, variables):
    """
    Read an Excel workbook and return a copy restricted to the requested columns.

    Inputs: the workbook path and the list of variables (columns) to keep.
    Output: a DataFrame with only those columns.
    """
    # Read the Excel file
    df = pd.read_excel(xlsx_path)
    df_interest = df[variables].copy()

    return df_interest
def check_variable_names(data):
    """
    Function to check if the variable names (column names) in a DataFrame or a list are valid Python identifiers.
    A valid Python identifier starts with a letter or underscore and contains only alphanumeric characters and underscores.

    Parameters:
    - data : pd.DataFrame or list
        The DataFrame whose columns or list of names you want to check.

    Returns:
    - valid_names : list
        A list of valid variable names.
    - invalid_names : list
        A list of invalid variable names, i.e., those containing spaces or special characters.
    """
    valid_names = []
    invalid_names = []

    # Determine if input is a DataFrame or a list
    if isinstance(data, pd.DataFrame):
        names_to_check = data.columns
    elif isinstance(data, list):
        names_to_check = data
    else:
        raise TypeError("Input must be either a DataFrame or a list.")

    for name in names_to_check:
        # Check if the name is a valid Python identifier
        if name.isidentifier() and not re.search(r'\W', name):
            valid_names.append(name)
        else:
            invalid_names.append(name)

    return valid_names, invalid_names
def extract_var_int(variables, var_no_int=None):
    """
    Return the variables that are not listed in ``var_no_int``.

    Inputs: the full list of variables and the list of variables to exclude
    (e.g. float and dichotomous columns). Output: the remaining variable names.
    """
    # Remove the var_float and var_dicho columns from variables to build var_int
    var_int = [col for col in variables if col not in var_no_int]
    return var_int
def find_string_variables(df, return_values=False):
    """
    Searches for string columns in a DataFrame and, optionally, returns the unique values for each string column.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to analyze.
    - return_values (bool): If True, also returns a dictionary containing the unique values of each string column.

    Returns:
    - string_columns (list): A list of column names that are of string type.
    - unique_values_dict (dict, optional): A dictionary with the column names as keys and their unique values as the associated values (only if return_values=True).
    """

    # Find columns that are of string (object) data type
    string_columns = [col for col in df.columns if df[col].dtype == 'object']

    if return_values:
        # If return_values is True, create a dictionary with unique values for each string column
        unique_values_dict = {col: df[col].dropna().unique().tolist() for col in string_columns}
        return string_columns, unique_values_dict
    else:
        # If return_values is False, return only the list of string columns
        return string_columns
def find_non_numeric_variable(df, return_values=False):
    """
        Searches for non-numeric columns in a DataFrame and, optionally, returns the unique values for each non-numeric column.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to analyze.
        - return_values (bool): If True, also returns a dictionary containing the unique values of each non-numeric column.

        Returns:
        - non_numeric_columns (list): A list of column names that are not of numeric type.
        - unique_values_dict (dict, optional): A dictionary with the column names as keys and their unique values as the associated values (only if return_values=True).
        """

    # Find columns that are not of numeric data type
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    if return_values:
        # If return_values is True, create a dictionary with unique values for each non-numeric column
        unique_values_dict = {col: df[col].dropna().unique().tolist() for col in non_numeric_columns}
        return non_numeric_columns, unique_values_dict
    else:
        # If return_values is False, return only the list of non-numeric columns
        return non_numeric_columns
def find_categorical_variables(df, unique_threshold=10, return_count=True, return_unique_values=False,
                               return_continuous=False, as_result=False):
    """
    Function to identify categorical and continuous variables based on the number of unique values.
    It classifies variables into three categories: dichotomous (2 unique values), multi-category
    (more than 2 but less than the unique threshold), and continuous (with unique values greater than
    or equal to the threshold).

    Parameters:
    - df : pd.DataFrame
        The DataFrame containing the variables.
    - unique_threshold : int, optional, default=10
        The maximum number of unique values for a variable to be considered as multi-category. Variables
        with unique values greater than or equal to this threshold will be classified as continuous.
    - return_count : bool, optional, default=True
        If True, returns the count of unique values for each categorical variable.
    - return_unique_values : bool, optional, default=False
        If True, returns a dictionary with unique values and their counts for each categorical variable.
    - return_continuous : bool, optional, default=False
        If True, returns continuous variables (those with unique values greater than or equal to the `unique_threshold`).

 Returns:
    - dichotomous_vars : list
        List of column names identified as dichotomous variables (2 unique values).
    - multi_category_vars : list
        List of column names identified as multi-category variables (more than 2 but less than `unique_threshold` unique values).
    - continuous_vars : list, optional
        List of column names identified as continuous variables (unique values greater than or equal to `unique_threshold`).
    - dichotomous_counts : dict
        Dictionary containing the count of unique values (if `return_count` is True) or a dictionary of unique value frequencies
        (if `return_unique_values` is True) for dichotomous variables.
    - multi_category_counts : dict
        Dictionary containing the count of unique values (if `return_count` is True) or a dictionary of unique value frequencies
        (if `return_unique_values` is True) for multi-category variables.
    - continuous_counts : dict, optional
        Dictionary containing the count of unique values for continuous variables, returned only if `return_continuous` is True.

    Return logic:
    - If `return_count` or `return_unique_values` is True:
        The function returns a tuple of the form:
        (
            (dichotomous_vars, dichotomous_counts),
            (multi_category_vars, multi_category_counts),
            (continuous_vars, continuous_counts)  # This is returned only if `return_continuous` is True
        )
        Where:
        - `dichotomous_vars` and `multi_category_vars` are lists of variable names.
        - `dichotomous_counts` and `multi_category_counts` are dictionaries:
          - If `return_count` is True, the dictionaries contain the number of unique values for each variable.
          - If `return_unique_values` is True, the dictionaries contain unique values and their counts for each variable.
        - `continuous_vars` and `continuous_counts` are included only if `return_continuous` is True.
          These contain variable names and counts for continuous variables.

    - If `return_continuous` is False, continuous variables and their counts are not returned.

    - If `return_count` and `return_unique_values` are both False:
        The function returns only the lists of dichotomous and multi-category variables without any counts or unique values:
        (
            dichotomous_vars,
            multi_category_vars
        )
    """

    dichotomous_vars = []
    multi_category_vars = []
    continuous_vars = []

    dichotomous_counts = {}
    multi_category_counts = {}
    continuous_counts = {}

    all_counts = {}  # number of unique values per column (for the named return)

    for column in df.columns:
        # Drop NA values for counting unique values properly
        unique_value_counts = df[column].dropna().value_counts()

        num_unique = len(unique_value_counts)
        all_counts[column] = num_unique

        # Find dichotomous variables (2 unique values)
        if num_unique == 2:
            dichotomous_vars.append(column)

            # Store the counts for unique values in dichotomous variables
            if return_unique_values:
                dichotomous_counts[column] = unique_value_counts.to_dict()  # Dictionary of unique values and their counts
            elif return_count:
                dichotomous_counts[column] = num_unique

        # Find multi-category variables (less than the threshold, but more than 2 unique values)
        elif 2 < num_unique < unique_threshold:
            multi_category_vars.append(column)

            # Store the counts for unique values in multi-category variables
            if return_unique_values:
                multi_category_counts[column] = unique_value_counts.to_dict()  # Dictionary of unique values and their counts
            elif return_count:
                multi_category_counts[column] = num_unique

        # Identify continuous variables (with unique values greater than or equal to the threshold)
        elif num_unique >= unique_threshold:
            continuous_vars.append(column)
            if return_count or return_unique_values:
                continuous_counts[column] = num_unique

    # NAMED, predictable return (opt-in), recommended for new code.
    if as_result:
        return VariableTypes(
            dichotomous=dichotomous_vars,
            multiclass=multi_category_vars,
            continuous=continuous_vars,
            counts=all_counts,
        )

    ## Return based on what was requested
    if return_count or return_unique_values:
        if return_continuous:
            return (dichotomous_vars,dichotomous_counts), (multi_category_vars, multi_category_counts), (continuous_vars , continuous_counts)
        else:
            return (dichotomous_vars, dichotomous_counts), (multi_category_vars, multi_category_counts)
    else:
        if return_continuous:
            return dichotomous_vars, multi_category_vars, continuous_vars
        else:
            return dichotomous_vars, multi_category_vars


def data_overview(df, unique_threshold=10, max_samples=3):
    """
    Readable overview of a dataset, in a SINGLE call.

    For each column, returns a DataFrame giving: the pandas dtype, the detected
    statistical type, the number of unique values, the count and percentage of
    missing values, sample values, quality flags (constant, quasi-constant,
    mixed types) and a processing suggestion. Intended for a quick understanding
    of the data before analysis.

    Parameters
    ----------
    df : pd.DataFrame
    unique_threshold : int, default=10
        At or above this number of categories, a numeric variable is treated as
        continuous (consistent with find_categorical_variables).
    max_samples : int, default=3
        Number of sample values displayed per column.

    Returns
    -------
    pd.DataFrame sorted by descending percentage of missing values, with columns:
        variable, dtype, detected_type, n_unique, n_missing, pct_missing,
        sample_values, flags, suggestion.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n * 100) if n else 0.0
        n_unique = int(non_null.nunique())

        # Mixed types (e.g. int and str in the same column)
        mixed = False
        if s.dtype == object and not non_null.empty:
            mixed = non_null.map(type).nunique() > 1

        # Statistical type detection
        if n_unique <= 1:
            detected = "constant"
        elif pd.api.types.is_datetime64_any_dtype(s):
            detected = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            if n_unique == 2:
                detected = "dichotomous"
            elif n_unique < unique_threshold:
                detected = "multiclass"
            else:
                detected = "continuous"
        else:
            if n_unique == 2:
                detected = "dichotomous"
            elif n_unique < unique_threshold:
                detected = "categorical"
            else:
                detected = "text"

        # Quality flags
        flags = []
        if detected == "constant":
            flags.append("constant")
        elif n and n_unique / max(len(non_null), 1) < 0.02 and detected != "dichotomous":
            flags.append("quasi-constant")
        if mixed:
            flags.append("mixed types")
        if pct_missing >= 50:
            flags.append("high missingness")

        # Processing suggestion
        suggestion = {
            "constant": "constant -> exclude from the analysis",
            "datetime": "date -> recode_time_difference / encode_datetime",
            "dichotomous": "dichotomous -> 0/1 (encode_variables if text)",
            "multiclass": "few categories -> categorical (ordinal? to be checked)",
            "categorical": "nominal categorical -> dummies (recode_categorial_var)",
            "continuous": "continuous -> keep as is",
            "text": "high-cardinality text -> recode/group before analysis",
        }[detected]
        if mixed:
            suggestion = "clean the types (clean_and_convert_columns); " + suggestion

        sample_values = ", ".join(map(str, non_null.unique()[:max_samples]))

        rows.append({
            "variable": col,
            "dtype": str(s.dtype),
            "detected_type": detected,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 1),
            "sample_values": sample_values,
            "flags": ", ".join(flags),
            "suggestion": suggestion,
        })

    cols = ["variable", "dtype", "detected_type", "n_unique", "n_missing",
            "pct_missing", "sample_values", "flags", "suggestion"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values("pct_missing", ascending=False)
            .reset_index(drop=True))


def find_non_uniform_variables(df, return_values=False):
    """
    Searches for columns in a DataFrame that contain mixed data types (e.g., both int and str),
    and optionally returns the unique values and their types for each column.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to analyze.
    - return_values (bool): If True, also returns a dictionary containing the unique values and their types for each column with mixed types.

    Returns:
    - mixed_type_columns (list): A list of column names that have mixed types of values.
    - unique_values_dict (dict, optional): A dictionary with the column names as keys and their unique values and types as the associated values (only if return_values=True).
    """

    # Find columns with mixed types (more than one type of data)
    mixed_type_columns = []
    mixed_values_dict = {}

    for col in df.columns:
        # Get unique values and their types
        unique_values = df[col].dropna().unique()
        types_in_col = set(type(val) for val in unique_values)

        # Check if the column has more than one type
        if len(types_in_col) > 1:
            mixed_type_columns.append(col)
            if return_values:
                # Store the unique values with their types
                mixed_values_dict[col] = {val: type(val).__name__ for val in unique_values}

    if return_values:
        return mixed_type_columns, mixed_values_dict
    else:
        return mixed_type_columns

def find_time_variables(df):
    """
    Identify date and time variables in a DataFrame by attempting to convert each column's values.

    Parameters:
    - df : pd.DataFrame
        The DataFrame to analyze.

    Returns:
    - date_vars : list
        List of column names identified as date variables (convertible to datetime).
    - time_vars : list
        List of column names identified as time variables (values resembling time formats).
    """

    date_vars = []
    time_vars = []

    # List of common date formats to try
    date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']
    time_formats = ['%H:%M:%S', '%H:%M', '%I:%M %p']  # Added AM/PM format for time detection

    for column in df.columns:
        # Try to parse column as a date
        for date_format in date_formats:
            try:
                parsed_date = pd.to_datetime(df[column], errors='raise', format=date_format)
                if parsed_date.notna().all():
                    date_vars.append(column)
                    break  # Stop checking other formats once a match is found
            except (ValueError, TypeError):
                pass  # If the conversion fails, it's not in this date format

        # Try to parse column as a time if it's not already identified as a date
        if column not in date_vars:
            for time_format in time_formats:
                try:
                    parsed_time = pd.to_datetime(df[column], errors='raise', format=time_format)
                    if parsed_time.notna().all():
                        time_vars.append(column)
                        break  # Stop checking other formats once a match is found
                except (ValueError, TypeError):
                    pass  # If the conversion fails, it's not in this time format

    return date_vars, time_vars
def find_time_variables_splitting(df, return_format=False):
    """
    Identify date and time variables in a DataFrame by splitting the values and checking if the parts are convertible to integers.

    Parameters:
    - df : pd.DataFrame
        The DataFrame to analyze.
    - return_format : bool, optional
        If True, returns the detected formats of date and time variables.

    Returns:
    - date_vars : list
        List of column names identified as date variables (convertible to datetime).
    - time_vars : list
        List of column names identified as time variables (values resembling time formats).
    - date_formats : dict (optional)
        Dictionary of detected formats for date variables if return_format=True.
    - time_formats : dict (optional)
        Dictionary of detected formats for time variables if return_format=True.
    """

    date_vars = []
    time_vars = []
    date_formats = {}
    time_formats = {}

    # Define common delimiters for date and time formats
    date_delimiters = ['-', '/']
    time_delimiters = [':']

    for column in df.columns:
        sample_data = df[column].dropna().astype(str).head(10).tolist()  # Take a sample of non-null values

        # Check for date formats by splitting with common date delimiters
        is_date = False
        for delimiter in date_delimiters:
            split_samples = [x.split(delimiter) for x in sample_data if delimiter in x]
            if split_samples and all(len(parts) == 3 and is_valid_int_list(parts) for parts in split_samples):
                is_date = True
                if delimiter == '-':
                    date_format = '%Y-%m-%d'  # Default to ISO format if using '-' as delimiter
                elif delimiter == '/':
                    date_format = '%d/%m/%Y'  # Assume European-style if using '/' as delimiter
                if return_format:
                    date_formats[column] = date_format
                date_vars.append(column)
                break  # No need to check further delimiters if this one matches

        # If not a date, check for time formats by splitting with time delimiters
        if not is_date:
            for delimiter in time_delimiters:
                split_samples = [x.split(delimiter) for x in sample_data if delimiter in x]
                if split_samples and all(
                        len(parts) == 2 or len(parts) == 3 and is_valid_int_list(parts) for parts in split_samples):
                    if return_format:
                        time_formats[column] = '%H:%M:%S' if len(split_samples[0]) == 3 else '%H:%M'
                    time_vars.append(column)
                    break  # No need to check further delimiters if this one matches

    if return_format:
        return date_vars, time_vars, date_formats, time_formats
    else:
        return date_vars, time_vars



# Replace
def rename_columns(df, name_changes):
    """
    Function to rename columns in a DataFrame based on a list of (old_name, new_name) tuples.

    Parameters:
    - df : pd.DataFrame
        The DataFrame whose column names you want to rename.
    - name_changes : list of tuples
        A list of tuples where each tuple contains (old_name, new_name).

    Returns:
    - df : pd.DataFrame
        The DataFrame with renamed columns.

    """
    unmatched_names = []
    rename_dict = {}

    for old_name, new_name in name_changes:
        if old_name in df.columns:
            rename_dict[old_name] = new_name
        else:
            unmatched_names.append(old_name)

    # Rename columns in the DataFrame
    df = df.rename(columns=rename_dict)

    if unmatched_names:
        print(unmatched_names)

    return df
def replace_target_value(df, variable, target_value, new_value=None, addNa=None, new_column=None):
    """
    Replaces a target value in a specific column of a DataFrame with either a new value or NaN.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the target column.
    - variable (str): The column in which the target value will be replaced.
    - target_value: The value in the column to replace.
    - new_value: The new value to replace the target_value with. Cannot be used together with addNa.
    - addNa (bool, optional): If True, replaces target_value with NaN. Cannot be used with new_value.
    - new_column (str, optional): The name of the new column to store the result. If None, the changes are made in the original column.

    Returns:
    - df (pandas.DataFrame): The DataFrame with the target value replaced.
    """

    # Check for conflicting parameters: cannot have both new_value and addNa set
    if new_value is not None and addNa:
        raise ValueError("You cannot use both 'new_value' and 'addNa' at the same time. Choose one.")

    _require_columns(df, variable, "replace_target_value")
    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Determine which column to modify (overwrite the original or create a new one)
    if new_column is None:
        new_column = variable

    # Replace the target_value with either the new_value or NaN
    if addNa:
        df_copy[new_column] = df_copy[variable].replace(target_value, np.nan)
    elif new_value is not None:
        df_copy[new_column] = df_copy[variable].replace(target_value, new_value)

    return df_copy
def modify_list(total_list, variables_to_operate, operation="remove"):
    """
    Function to modify a list by either adding or removing specific variables, without modifying the original list.

    Parameters:
    - total_list : list
        The original list of variables that you want to modify.
    - variables_to_operate : list
        The variables that you want to add or remove from the total list.
    - operation : str, optional, default="remove"
        The operation to perform. Options are:
        - "remove": Remove the variables from the total list.
        - "add": Add the variables to the total list.

    Returns:
    - modified_list : list
        A new list after applying the specified operation, without modifying the original list.
    """

    # Create a copy of the total list to avoid modifying the original list
    modified_list = total_list.copy()

    # Perform the specified operation
    if operation == "remove":
        # Remove variables from the list
        modified_list = [var for var in modified_list if var not in variables_to_operate]
    elif operation == "add":
        # Add variables to the list, ensuring no duplicates
        modified_list.extend([var for var in variables_to_operate if var not in modified_list])
    else:
        raise ValueError("Invalid operation. Choose 'remove' or 'add'.")

    return modified_list
def classify_columns_by_type(df):
    """
    Classify the DataFrame columns by data type.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    dict: A dictionary whose keys are the data types and whose values are the lists of corresponding columns.
    """
    type_dict = {}

    for col in df.columns:
        var_type = df[col].dtype
        if var_type not in type_dict:
            type_dict[var_type] = []
        type_dict[var_type].append(col)

    return type_dict
def clean_and_convert_columns(df, columns):
    """
    Coerce the given object columns to numeric in place, mapping invalid entries to NaN.

    Inputs: a DataFrame and the columns to convert. Output: the same DataFrame with
    the columns cast to numeric. Side effect: mutates ``df`` and prints a status
    line per converted column (and the offending values on failure).
    """
    for col in columns:
        if df[col].dtype == 'object':
            try:
                # Convert to numeric, forcing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric.")
            except Exception as e:
                print(f"Failed to convert {col}: {e}")
                print(f"Problematic values in {col}:")
                print(df[col].unique())
    return df


def is_valid_int_list(lst):
    """Helper function to check if all elements in the list are integers."""
    try:
        return all(int(x) == float(x) for x in lst)
    except (ValueError, TypeError):
        return False



def encode_datetime(df, date_hour_pairs, fill_missing='', date_format='%Y-%m-%d', drop_original=True):
    """
    Function to create datetime columns by combining date and time columns.

    Parameters:
    - df : pd.DataFrame
        The DataFrame containing the date and time columns.
    - date_hour_pairs : list of tuples
        List of tuples where each tuple contains:
        (new_column_name, date_column, hour_column)
        Example: [('datetime_avc', 'report_date_avc', 'heure_onset')]
    - fill_missing : str, optional, default=''
        Value to fill missing data in time columns. Default is an empty string.
    - date_format : str, optional, default='%Y-%m-%d'
        The format to use for parsing dates when necessary.

    Returns:
    - df : pd.DataFrame
        The DataFrame with new datetime columns created based on the provided pairs.
    """

    # Check that all referenced date/time columns exist.
    needed = [c for _, date_col, hour_col in date_hour_pairs for c in (date_col, hour_col)]
    _require_columns(df, needed, "encode_datetime")
    # Defensive copy: the caller's DataFrame must not be mutated.
    df = df.copy()

    for new_col, date_col, hour_col in date_hour_pairs:
        # Fill missing hours with a default value.
        # Note: direct assignment (not `df[col].fillna(inplace=True)`, which under
        # pandas 3.0 operates on a temporary copy -> changes nothing).
        df[hour_col] = df[hour_col].fillna(fill_missing)

        # Handle cases where date columns may need formatting before combining
        df[date_col] = pd.to_datetime(df[date_col].dt.strftime(date_format), errors='coerce')

        # Combine date and hour into a single datetime column
        df[new_col] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[hour_col].astype(str), errors='coerce'
        )
        if drop_original:
            df = df.drop(columns=date_col)
            df = df.drop(columns=hour_col)

    return df


def recode_time_difference(df, new_variable_name, datetime1, datetime2, unit='minutes', drop_original=False,
                           handle_nodiff=True, date_format='%d/%m/%y'):
    """
    Compute the difference between two datetime columns.

    Inputs: a DataFrame, the two datetime column names, the output unit
    ('seconds', 'minutes', 'hours' or 'days'), and the string date format used to
    parse non-datetime columns. Output: the DataFrame with a new numeric column.
    A warning is emitted when a supplied format silently turns non-null values
    into NaT (so the loss of observations is not overlooked).
    """
    _require_columns(df, [datetime1, datetime2], "recode_time_difference")
    # Force a full copy to avoid the SettingWithCopyWarning
    df = df.copy()
    #df[datetime1] = df[datetime1].astype(str).str.strip()
    #df[datetime2] = df[datetime2].astype(str).str.strip()

    # Convert to datetime WITHOUT silently destroying valid dates:
    #  - a column already in datetime64 is not re-parsed (the format does not apply
    #    and could, depending on the pandas version, coerce it to NaT);
    #  - otherwise parse, but WARN if an unsuitable format turns non-null values
    #    into NaT (otherwise the loss of observations goes unnoticed).
    for col in (datetime1, datetime2):
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        original_notna = df[col].notna()
        parsed = pd.to_datetime(df[col], format=date_format, errors='coerce')
        lost = int((parsed.isna() & original_notna).sum())
        if lost:
            warnings.warn(
                f"recode_time_difference: {lost}/{int(original_notna.sum())} non-null "
                f"values of '{col}' could not be parsed with format={date_format!r} "
                f"and became NaT. Check the date format of the data.",
                stacklevel=2,
            )
        df[col] = parsed  # column replacement (dtype change string -> datetime)

    # Compute the time difference
    time_diff = df[datetime1] - df[datetime2]
    time_diff = pd.to_timedelta(time_diff)

    # Convert according to the requested unit
    if unit in ['seconds', 'sec', 's']:
        df.loc[:, new_variable_name] = time_diff.dt.total_seconds()
    elif unit in ['minutes', 'min', 'm']:
        df.loc[:, new_variable_name] = time_diff.dt.total_seconds() / 60
    elif unit in ['hours', 'hrs', 'h']:
        df.loc[:, new_variable_name] = time_diff.dt.total_seconds() / 3600
    elif unit in ['days', 'jours', 'd']:
        df.loc[:, new_variable_name] = time_diff.dt.total_seconds() / 86400 # Direct conversion to days
    else:
        raise ValueError("Invalid unit. Choose 'seconds', 'minutes', 'hours', or 'days'.")

    # Handle zero differences
    if handle_nodiff:
        df.loc[:, new_variable_name] = df[new_variable_name].apply(lambda x: np.nan if x == 0 else x)

    # Convert to float
    df.loc[:, new_variable_name] = df[new_variable_name].astype('float64')

    # Drop the original columns if requested
    if drop_original:
        df.drop(columns=[datetime1, datetime2], inplace=True)

    return df
def encode_variables(df, label_vars=None, dummy_var=None):
    """
    Encode ordinal and nominal categorical variables using Label Encoding for ordinal variables
    and Dummy Encoding for non-ordinal variables, except for dichotomous variables (variables with exactly two categories).

    Parameters:
    df: pd.DataFrame
        The dataframe containing the variables to be encoded.
    label_vars: list
        List of variable names to be label encoded.
    dummy_var: list
        List of non-ordinal variable names to be dummy encoded.

    Returns:
    df_encoded: pd.DataFrame
        The dataframe with the encoded variables (without the original variables).
    encoding_dict: dict
        A dictionary containing the encoding details for each variable.
    new_labeled_vars: list
        List of the new encoded ordinal variables.
    new_dummied: list
        List of the new dummy-encoded variables.
    """
    if label_vars is None:
        label_vars = []
    if dummy_var is None:
        dummy_var = []

    df_encoded = df.copy()
    encoding_dict = {}
    new_labeled_vars = []
    new_dummied = []

    # To store dichotomous variables for final print
    dichotomous_vars = []

    # Label Encoding for ordinal variables
    for var in label_vars:
        if var in df_encoded.columns:
            # Check if variable is dichotomous (two unique values)
            if df_encoded[var].nunique() == 2:
                if pd.api.types.is_numeric_dtype(df_encoded[var]):
                    # Already numeric (0/1...): compatible with a downstream astype(float).
                    encoding_dict[var] = {'type': 'dichotomous variable (numeric, unchanged)'}
                else:
                    # TEXT dichotomous -> encode as 0/1 (NaN preserved), otherwise the
                    # column stays a string and breaks astype(float) in regression.
                    cats = sorted(df_encoded[var].dropna().unique(), key=str)
                    mapping = {c: i for i, c in enumerate(cats)}
                    df_encoded[var] = df_encoded[var].map(mapping)
                    encoding_dict[var] = {'type': 'dichotomous label_encoding', 'mapping': mapping}
                new_labeled_vars.append(var)  # Still add to the list of ordinal variables
                dichotomous_vars.append(var)  # Store the variable name for printing
            else:
                # Apply Label Encoding
                le = LabelEncoder()
                df_encoded[var] = le.fit_transform(df_encoded[var].astype(str))
                encoding_dict[var] = {'type': 'label_encoding',
                                      'mapping': dict(zip(le.classes_, le.transform(le.classes_)))}
                new_labeled_vars.append(var)  # Append the new ordinal variable to the list

    # Dummy Encoding for non-ordinal variables
    for var in dummy_var:
        if var in df_encoded.columns:
            # Check if variable is dichotomous (two unique values)
            if df_encoded[var].nunique() == 2:
                if pd.api.types.is_numeric_dtype(df_encoded[var]):
                    # Already numeric (0/1...): no dummy expansion needed.
                    encoding_dict[var] = {'type': 'dichotomous variable (numeric, unchanged)'}
                else:
                    # TEXT dichotomous -> 0/1 (single column, NaN preserved) instead
                    # of a dummy; avoids the downstream astype(float) failure.
                    cats = sorted(df_encoded[var].dropna().unique(), key=str)
                    mapping = {c: i for i, c in enumerate(cats)}
                    df_encoded[var] = df_encoded[var].map(mapping)
                    encoding_dict[var] = {'type': 'dichotomous label_encoding', 'mapping': mapping}
                new_dummied.append(var)  # Still add to the list
                dichotomous_vars.append(var)  # Store the variable name for printing
            else:
                # Apply Dummy Encoding
                dummies = pd.get_dummies(df_encoded[var], prefix=var, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(var, axis=1), dummies], axis=1)
                encoding_dict[var] = {'type': 'dummy_encoding', 'categories': df[var].unique().tolist()}
                new_dummied.extend(dummies.columns.tolist())  # Append the dummy-encoded variable names

    # Print dichotomous variables at the end
    if dichotomous_vars:
        print("Dichotomous variables (kept as a single 0/1 column):", dichotomous_vars)
    else:
        print("No dichotomous variables found.")

    # Return matching the docstring, regardless of the arguments supplied.
    # (Previously: the `label_vars is None` / `dummy_var is None` tests were always
    #  false because both are reassigned to [] at the top -> the function returned
    #  None when both lists were passed, and inconsistently shaped tuples otherwise.)
    return df_encoded, encoding_dict, new_labeled_vars, new_dummied

def fill_na_columns(df, variables, fill_value=0, suffix='_filled', return_filled_list=True):
    """
    For each variable in 'variables', create a new column with the given suffix
    in which the missing values (NaN) are replaced by 'fill_value'.

    Parameters:
    - df : pd.DataFrame
        The DataFrame to modify.
    - variables : list of str
        The list of columns to process.
    - fill_value : any, default=0
        The value used to replace NaN.
    - suffix : str, default='_filled'
        The suffix appended to the variable name for the new column.

    Returns:
    - pd.DataFrame
        The modified DataFrame with the new columns.
    """
    _require_columns(df, variables, "fill_na_columns")
    df = df.copy()  # do not mutate the caller's DataFrame
    filled_var_list = []
    for var in variables:
        filled_var_name = var + suffix
        filled_var_list.append(filled_var_name)
        df[filled_var_name] = df[var].fillna(fill_value)
    if return_filled_list:
        return df, filled_var_list
    else:
        return df

def recode_categorial_var(df, variable, recode_func, new_variable_name, drop_original=False, ordered=False, category_order=None, encoding_type=None):
    """
    Generic function to recode a variable in a DataFrame with options for ordered categories and removing the original variable.
    If no explicit order is provided, it assumes the order based on the recoding logic (i.e., the order in the function or dictionary).

   Parameters:
    - df : pd.DataFrame
        The DataFrame containing the variable to be recoded.
    - variable : str
        The name of the column to be recoded.
    - recode_func : function or dict
        Either a recoding function or a mapping dictionary.
    - new_variable_name : str
        The name of the new column to be created.
    - encoding_type : str, optional
        Specify 'dummy' for dummy encoding (nominal), 'label' for label encoding (ordinal), or None for no encoding.
    - drop_original : bool, optional, default=False
        If True, the original variable will be dropped from the DataFrame.
    - ordered : bool, optional, default=False
        If True, the new variable will be treated as an ordered categorical variable.
    - category_order : list, optional
        The explicit order of the categories. If not provided, the order is assumed from the recoding logic.

    Returns:
    - df : pd.DataFrame
        The DataFrame with the new recoded variable added (and optionally, the original variable removed).
    """
    # Warn if dummy encoding is chosen with ordered=True, as they conflict
    if encoding_type == 'dummy' and ordered:
        raise ValueError("Dummy encoding does not support ordered categories. Please set 'ordered' to False.")

    _require_columns(df, variable, "recode_categorial_var")
    df = df.copy()  # do not mutate the caller's DataFrame

    # Set to track the order based on recoding logic (preserving order)
    recode_order = []

    # If a dictionary is passed, map using the dictionary and assume the order of the dictionary's values
    if isinstance(recode_func, dict):
        df[new_variable_name] = df[variable].map(recode_func)
        if ordered and category_order is None:
            recode_order = list(dict.fromkeys(recode_func.values()))  # Order based on the dictionary's values

    else:
        # If a function is passed, apply the function and track the recoding order
        def track_recode(value):
            recoded_value = recode_func(value)
            if recoded_value not in recode_order and pd.notna(recoded_value):
                recode_order.append(recoded_value)  # Track the first occurrence based on recoding logic
            return recoded_value

        df[new_variable_name] = df[variable].apply(track_recode)


    # Use the tracked recode order if ordered=True and no explicit category order is provided
    if ordered and category_order is None:
        category_order = recode_order  # Use the order tracked during recoding

    # Convert to an ordered categorical type if requested
    if ordered or encoding_type == 'label':
        df[new_variable_name] = pd.Categorical(df[new_variable_name], categories=category_order, ordered=ordered)

    # Apply encoding based on the specified type
    if encoding_type == 'dummy':
        # Dummy encoding (one-hot encoding)
        df = pd.get_dummies(df, columns=[new_variable_name], prefix=new_variable_name, drop_first=True)
    elif encoding_type == 'label':
        # Label encoding (convert categorical to numerical labels)
        df[new_variable_name] = df[new_variable_name].cat.codes

    # Optionally drop the original variable
    if drop_original:
        df = df.drop(columns=[variable])

    return df

def recode_formula_var(df, formula, output_type='Int64', drop_original=False, handle_nan=True,
                       var_list_to_fill=None, fill_value=0):
    """
    Generic function to create a new variable in a DataFrame based on a simple formula logic,
    with support for handling NaN values in a boolean context.

    Parameters:
    - df : pd.DataFrame
        The DataFrame containing the variables.
    - formula : str
The formula representing the logic for creating the new variable.
        Example for dichotomous variable: 'A ~ B == 1 | C == 1'
        Example for scaled variable: 'A ~ B + C'
        Example for ratio: 'A ~ B / C'
    - new_variable_name : str
        The name of the new column to be created.
    - output_type : str, optional
        The desired output data type for the new variable (e.g., 'Int64', 'float', etc.). Default is 'Int64'.
    - drop_original : bool, optional, default=False
        If True, the original variables used in the formula will be dropped from the DataFrame.
    - handle_nan : bool, optional, default=True
        If True, the function will propagate NaNs if any variable in the expression is NaN.
    - var_list_to_fill : list of str, optional, default=None
        If provided, for each variable in this list present in the formula, a new column
        is created with the suffix (default '_filled') and its NaN are replaced by fill_value.
    - fill_value : any, optional, default=0
        The value used to replace NaN for the variables listed in var_list_to_fill.

    Returns:
    - df : pd.DataFrame
        The DataFrame with the new variable added (and optionally, the original variables removed).
    """

    # Split the formula into the right-hand side (RHS) to extract the logic
    formula_split = formula.split('~')
    if len(formula_split) != 2:
        raise ValueError(f"The formula must be in the format 'new_variable_name ~ condition_expression'. \n {formula}")

    new_variable_name = formula_split[0].strip()
    condition_expression = formula_split[1].strip()

    df = df.copy()  # do not mutate the caller's DataFrame

    # Track the "filled" columns that were created
    filled_columns_created = []

    # If var_list_to_fill is provided, check for each variable whether the expression
    # contains the original name and not already the "filled" version
    if var_list_to_fill is not None:
        fill_suffix = '_filled'
        fill_vars = []
        for var in var_list_to_fill:
            # If the filled version is already present in the expression, do nothing
            if re.search(r'\b' + re.escape(var + fill_suffix) + r'\b', condition_expression):
                continue
            # Otherwise, if the original name is present in the expression, add it
            if re.search(r'\b' + re.escape(var) + r'\b', condition_expression):
                fill_vars.append(var)
        # If variables to fill were identified, fill them
        if fill_vars:
            df, filled_var_list = fill_na_columns(df, fill_vars, fill_value=fill_value, suffix=fill_suffix)
            filled_columns_created = filled_var_list.copy()
            # Update the expression to use the "filled" columns
            for original, filled in zip(fill_vars, filled_var_list):
                condition_expression = re.sub(r'\b' + re.escape(original) + r'\b', filled, condition_expression)

    # Check the types of the columns involved in the formula
    variables_in_formula = [
        var for var in df.columns
        if re.search(r'\b' + re.escape(var) + r'\b', condition_expression)
    ]

    # Evaluate the condition or mathematical expression using the DataFrame
    try:
        evaluated = df.eval(condition_expression, engine='python')
        if handle_nan:
            # Special logical case: boolean variables combined with OR "|"
            if pd.api.types.is_bool_dtype(evaluated):
                # NaN only if ALL variables are NaN
                all_nan_mask = df[variables_in_formula].isna().all(axis=1)
                evaluated_filled = df[variables_in_formula].fillna(0).eval(condition_expression, engine='python')
                df[new_variable_name] = np.where(all_nan_mask, pd.NA, evaluated_filled)
            else:
                # Standard case
                any_nan_mask = df[variables_in_formula].isna().any(axis=1)
                df[new_variable_name] = evaluated.mask(any_nan_mask, pd.NA)
        else:
            df[new_variable_name] = evaluated
        #
        # try:
        #     if handle_nan:
        #         nan_mask = df[variables_in_formula].isna().any(axis=1)  # Only check NaNs in relevant columns
        #         df[new_variable_name] = df.eval(condition_expression, engine='python').where(~nan_mask, pd.NA)
        #     else:
        #         df[new_variable_name] = df.eval(condition_expression, engine='python')

        # Cast the output to the desired type
        df[new_variable_name] = df[new_variable_name].replace({pd.NA: np.nan}).astype(output_type)

    except Exception as e:
        raise ValueError(f"Error evaluating the formula: \n {formula} \n {e} ")


    # Optionally, drop the original variables used in the formula
    if drop_original:
        vars_to_drop = set(variable for variable in df.columns if variable in condition_expression)
        df = df.drop(columns=list(vars_to_drop))

    # Drop the temporary "filled" columns if they exist in the final DataFrame
    if filled_columns_created:
        df = df.drop(columns=filled_columns_created, errors='ignore')
    return df





