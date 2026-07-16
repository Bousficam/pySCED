from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm

def plot_continuous_relationships(df, var_indep, var_deps, layout='balanced', hue=None, pvalue_types=None, save=None, legend='eng'):
    """
    Plots the relationship between one continuous independent variable and multiple continuous dependent variables
    in a single figure with multiple subplots. Optionally, colors the points by a categorical variable and shows the
    statistical significance of linear regression, Pearson correlation, and/or Spearman correlation.

    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data
    - var_indep: str, name of the continuous independent variable
    - var_deps: list, list of names of the continuous dependent variables
    - layout: str, layout of the subplots ('row', 'column', 'balanced')
    - hue: str, name of the categorical variable to color the points
    - pvalue_types: list, list of p-values to display ('regression', 'pearson', 'spearman')
    - save: str, file path to save the figure (optional)
    - legend: str, language for plot legends ('fr' for French, 'eng' for English)

    Returns:
    - Plots scatter plots with linear regression for each relationship in a single figure and displays the specified p-values.

    References: ordinary least squares regression; Pearson product-moment and Spearman (1904) rank correlations.
    R equivalent: stats::lm + stats::cor.test (pearson / spearman); display via ggplot2::geom_smooth(method="lm").
    """

    # Set default p-value types if not specified
    if pvalue_types is None:
        pvalue_types = ['regression', 'pearson', 'spearman']

    # Number of dependent variables
    n_deps = len(var_deps)

    # Define the layout of subplots
    if layout == 'row':
        n_rows = 1
        n_cols = n_deps
    elif layout == 'column':
        n_rows = n_deps
        n_cols = 1
    else:  # 'balanced' layout
        n_cols = math.ceil(math.sqrt(n_deps))
        n_rows = math.ceil(n_deps / n_cols)

    # Create a figure with multiple subplots (grid layout)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Ensure axes is a list of axes, even for single rows or columns
    if n_deps == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.ravel()

    # Iterate over each dependent variable to create subplots
    for i, var_dep in enumerate(var_deps):
        ax = axes[i]
        sub_df = df[[var_indep, var_dep, hue]].dropna().astype(float) if hue else df[[var_indep, var_dep]].dropna().astype(float)

        # Dictionary to store p-values
        p_values = {}

        # Compute linear regression and p-value
        if 'regression' in pvalue_types:
            X = sm.add_constant(sub_df[var_indep])
            model = sm.OLS(sub_df[var_dep], X).fit()
            p_values['regression'] = model.pvalues[1]  # P-value for the regression slope

        # Compute Pearson correlation and p-value
        if 'pearson' in pvalue_types:
            corr_pearson, p_values['pearson'] = pearsonr(sub_df[var_indep], sub_df[var_dep])

        # Compute Spearman correlation and p-value
        if 'spearman' in pvalue_types:
            corr_spearman, p_values['spearman'] = spearmanr(sub_df[var_indep], sub_df[var_dep])

        # Plot scatterplot with hue if defined, otherwise use a simple regplot
        if hue:
            sns.scatterplot(x=var_indep, y=var_dep, hue=hue, data=sub_df, ax=ax, palette='Set1', legend=True, alpha=0.7)
            sns.regplot(x=var_indep, y=var_dep, data=sub_df, ax=ax, scatter=False, line_kws={'color': 'red'})
        else:
            sns.regplot(x=var_indep, y=var_dep, data=sub_df, ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

        # Display p-values on the plot
        pval_text = '\n'.join([f'{key} p-value = {p_values[key]:.3f}' for key in p_values])
        ax.text(0.05, 0.95, pval_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        # Set titles and labels for each subplot
        if legend == 'fr':
            ax.set_title(f'Relation entre {var_indep} et {var_dep}', fontsize=14)
            ax.set_xlabel(var_indep, fontsize=12)
            ax.set_ylabel(var_dep, fontsize=12)
        else:
            ax.set_title(f'Relationship between {var_indep} and {var_dep}', fontsize=14)
            ax.set_xlabel(var_indep, fontsize=12)
            ax.set_ylabel(var_dep, fontsize=12)

        ax.grid(True)

    # Remove unused subplots in case of an incomplete grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure if 'save' parameter is provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
def plot_pairwise_relationships(df, var_indep, var_deps):
    """
    Plots a scatterplot matrix (pairplot) to visualize the relationships between one independent variable
    and several dependent variables.

    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data
    - var_indep: str, name of the continuous independent variable
    - var_deps: list, list of names of the continuous dependent variables

    Returns:
    - Plots a matrix of pairplots (scatter plots between each pair of variables).

    R equivalent: GGally::ggpairs (scatterplot matrix).
    """
    # Build a new DataFrame with the independent and the dependent variables
    plot_data = df[[var_indep] + var_deps].dropna().astype(float)

    # Build a pairplot with seaborn
    sns.pairplot(plot_data)
    plt.show()
def plot_pairwise_indep_vs_dep(df, var_indep_list, var_dep_list):
    """
    Plots a grid of scatter plots to visualize the relationships between several continuous
    independent variables and several continuous dependent variables, similar to a pairplot.

    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data
    - var_indep_list: list, list of names of the continuous independent variables
    - var_dep_list: list, list of names of the continuous dependent variables

    Returns:
    - Plots a grid of scatter plots for each (var_indep, var_dep) combination.

    R equivalent: no exact 1:1 - closest is GGally::ggpairs or a ggplot2 facet grid of geom_smooth panels.
    """
    # Determine the grid size
    n_indep = len(var_indep_list)
    n_dep = len(var_dep_list)

    # Create a grid of subplots
    fig, axes = plt.subplots(n_dep, n_indep, figsize=(5 * n_indep, 5 * n_dep))

    # With a single dependent or independent variable, axes is not an array; force it to 2D
    if n_indep == 1:
        axes = np.expand_dims(axes, axis=1)
    if n_dep == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop to create the plots over the grid
    for i, var_indep in enumerate(var_indep_list):
        for j, var_dep in enumerate(var_dep_list):
            ax = axes[j, i]

            # Data subset
            sub_df = df[[var_indep, var_dep]].dropna().astype(float)

            # Create the scatter plot with linear regression
            sns.regplot(x=var_indep, y=var_dep, data=sub_df, ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

            # Set titles and labels for each subplot
            ax.set_title(f'{var_indep} vs {var_dep}', fontsize=14)
            ax.set_xlabel(var_indep, fontsize=12)
            ax.set_ylabel(var_dep, fontsize=12)

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    plt.show()
def plot_continuous_vs_dichotomous(db, continuous_var, dichotomous_var, method="boxplot", title=None):
    """
    Plot the relationship between a continuous variable and a dichotomous variable.

    Parameters:
    - data: pd.DataFrame, the dataset containing the variables.
    - continuous_var: str, the name of the continuous variable.
    - dichotomous_var: str, the name of the dichotomous variable.
    - method: str, the method of visualization ('boxplot', 'violin', 'barplot').

    Returns:
    - A plot representing the relationship between the continuous and dichotomous variables.

    R equivalent: ggplot2::geom_boxplot / geom_violin / geom_bar (no 1:1 base-graphics equivalent).
    """

    data = db[[continuous_var, dichotomous_var]].dropna().astype(float)
    fig, ax = plt.subplots()

    if method == "boxplot":
        sns.boxplot(x=dichotomous_var, y=continuous_var, data=data)

    elif method == "violin":
        sns.violinplot(x=dichotomous_var, y=continuous_var, data=data)

    elif method == "barplot":
        sns.barplot(x=dichotomous_var, y=continuous_var, data=data, errorbar="sd")
    else:
        raise ValueError(f"Method '{method}' is not supported. Choose from 'boxplot', 'violin', or 'barplot'.")

    ax.set_xlabel(dichotomous_var)
    ax.set_ylabel(continuous_var)
    ax.set_title(title)
    plt.show()

    return ax
def combine_axes_into_figure(axes_list, layout='balanced', figsize=(10, 10)):
    """
    Combine multiple plot axes into a single figure with subplots.

    Parameters:
    - axes_list: list of Matplotlib Axes objects to be combined.
    - layout: str, layout of the subplots ('row', 'column', 'balanced').
    - figsize: tuple, size of the combined figure (width, height).

    Returns:
    - A Matplotlib figure containing the combined axes.
    """
    n_axes = len(axes_list)

    # Define grid layout based on layout parameter
    if layout == 'row':
        n_rows, n_cols = 1, n_axes
    elif layout == 'column':
        n_rows, n_cols = n_axes, 1
    else:  # 'balanced' layout
        n_cols = int(n_axes ** 0.5)
        n_rows = (n_axes + n_cols - 1) // n_cols  # Round up to fill the grid

    # Create a new figure for combined plots
    fig, new_axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten new_axes if needed
    if n_axes == 1:
        new_axes = [new_axes]
    elif n_rows == 1 or n_cols == 1:
        new_axes = new_axes.flatten()
    else:
        new_axes = new_axes.ravel()

    # Loop through the existing axes and re-plot them in the new figure
    for i, original_ax in enumerate(axes_list):
        # Copy each original axis to the new figure
        for line in original_ax.get_lines():
            new_axes[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())

        # Copy titles, labels, legends, and other details
        new_axes[i].set_title(original_ax.get_title())
        new_axes[i].set_xlabel(original_ax.get_xlabel())
        new_axes[i].set_ylabel(original_ax.get_ylabel())

        if original_ax.get_legend():
            new_axes[i].legend()

    # Adjust layout and show the combined figure
    plt.tight_layout()
    plt.show()

    return fig
def plot_2class_distribution(db, dichomotomic, outcome, kde=True, bins=30, color1='blue', color2='orange'):
    """
    Plots overlaid histograms of a continuous outcome for the two classes of a dichotomous variable.

    Parameters:
    - db: pd.DataFrame, the DataFrame containing the data
    - dichomotomic: str, name of the dichotomous variable (values 0 and 1)
    - outcome: str, name of the continuous outcome (divided by 60 to convert to hours)
    - kde: bool, whether to overlay a kernel density estimate
    - bins: int, number of histogram bins
    - color1: str, color for the class dichomotomic = 1
    - color2: str, color for the class dichomotomic = 0

    Returns:
    - Displays a Matplotlib figure with the two overlaid distributions.

    R equivalent: ggplot2::geom_histogram(position="identity") + geom_density (overlaid class histograms).
    """
    # Split the data into two groups: dichomotomic = 1 and dichomotomic = 0
    data_1 = db[db[dichomotomic] == 1][outcome] / 60
    data_0 = db[db[dichomotomic] == 0][outcome] / 60

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data_1, kde=kde, bins=bins, label=f"{dichomotomic} = 1", color=color1, alpha=0.6)
    sns.histplot(data_0, kde=kde, bins=bins, label=f"{dichomotomic} = 0", color=color2, alpha=0.6)

    # Add titles and legends
    plt.title(f"Distribution of {outcome} for {dichomotomic}")
    plt.xlabel(f'{outcome} (hours)')
    plt.ylabel("Frequency")
    plt.legend(title=dichomotomic)
    plt.show()
def plot_2class_distribution_split(db, dichomotomic, outcome, split_var, save=False, bins=30, color1='blue', color2='orange', inverse_palette=False):
    """
        Plot the distribution of an outcome variable split by a dichotomous variable
        and further grouped by a secondary categorical variable.

        Parameters:
        -----------
        db : pandas.DataFrame
            The input DataFrame containing the data to be plotted.
        dichotomous : str
            The name of the column containing the dichotomous variable (e.g., 0 or 1).
        outcome : str
            The name of the column containing the outcome variable to be plotted on the x-axis.
        split_var : str
            The name of the secondary categorical variable used for grouping within each dichotomous class.
        bins : int, optional, default=30
            Number of bins to use in the histogram.
        color1 : str, optional, default='blue'
            Color for the first category in the palette for dichotomous = 1.
        color2 : str, optional, default='orange'
            Color for the first category in the palette for dichotomous = 0.
        inverse_palette : bool, optional, default=False
            If True, inverts the color mapping for the palettes (e.g., assigns colors differently).

        Returns:
        --------
        None
            Displays a Matplotlib figure with two subplots:
            - Left subplot: Distribution of the outcome for dichotomous = 1.
            - Right subplot: Distribution of the outcome for dichotomous = 0.

        R equivalent: ggplot2::geom_histogram(position="stack") + facet_wrap (split by the dichotomous var).
        """
    # Validate columns
    for col in [dichomotomic, outcome, split_var]:
        if col not in db.columns:
            raise ValueError(f"The column '{col}' is not present in the provided DataFrame.")

    # Clean the data
    db = db.copy()
    db = db.dropna(subset=[outcome, dichomotomic, split_var])
    db.loc[:, split_var] = db[split_var].astype(int)

    # Configure palettes
    if inverse_palette:
        palette1 = {1: color1, 0: 'black'}
        palette2 = {1: color2, 0: 'black'}
        hue_order=[0, 1]
    else:
        palette1 = {0: color1, 1: 'black'}
        palette2 = {0: color2, 1: 'black'}
        hue_order=[1, 0]

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot for dichotomous = 1
    sns.histplot(
        data=db[db[dichomotomic] == 1],
        x=outcome,
        hue=split_var,
        bins=bins,
        element="bars",
        palette=palette1,
        stat='count',
        multiple="stack",
        hue_order=hue_order,
        alpha=0.7,
        zorder=1,
        ax=axes[0]
    )

    axes[0].set_title(f"Distribution of {outcome} for {dichomotomic} = 1")
    axes[0].set_xlabel(f'{outcome}')
    axes[0].set_ylabel("Count")

    # Plot for dichotomous = 0
    sns.histplot(
            data=db[db[dichomotomic] == 0],
            x=outcome,
            hue=split_var,
            bins=bins,
            element="bars",
            palette=palette2,
            stat='count',
            multiple="stack",
            hue_order=hue_order,
            alpha=0.7,
            zorder=1,
            ax=axes[1]
        )

    axes[1].set_title(f"Distribution of {outcome} for {dichomotomic} = 0")
    axes[1].set_xlabel(f'{outcome} ')

    # Finalize the layout
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=1200, bbox_inches='tight')
    plt.show()
