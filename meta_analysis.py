import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """
    Loads data from a CSV file and performs initial cleaning and preparation.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The prepared DataFrame.
    """
    try:
        dados = pd.read_csv(file_path, sep=';', decimal='.')
    except FileNotFoundError:
        return pd.DataFrame() # Return empty if file not found

    # Rename columns to avoid issues with spaces
    dados = dados.rename(columns={'Std Dev': 'Std_Dev', 'Original Unit': 'Original_Unit'})

    # Convert numeric columns, handling potential errors
    numeric_cols = ['Mean', 'Std_Dev']
    for col in numeric_cols:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados = dados.dropna(subset=numeric_cols) # Drop rows where conversion failed

    return dados

def filter_irrelevant_treatments(dados):
    """
    Filters out irrelevant treatments from the dataset.
    Args:
        dados (pandas.DataFrame): The input DataFrame.
    Returns:
        pandas.DataFrame: The DataFrame with irrelevant treatments filtered out.
    """
    # List of treatments that are NOT final vermicompost
    treatments_to_exclude = [
        "Fresh Grape Marc",
        "Manure",
        "CH0 (Initial)",
        "CH25 (Initial)",
        "CH50 (Initial)",
        "CH75 (Initial)",
        "CH100 (Initial)",
        "T1 (Initial)",
        "T2 (Initial)",
        "T3 (Initial)",
        "T4 (Initial)"
    ]
    return dados[~dados['Treatment'].isin(treatments_to_exclude)]

def define_groups_and_residues(dados_filtrados):
    """
    Defines 'Control' vs. 'Treatment' groups and 'Residue' types.
    Args:
        dados_filtrados (pandas.DataFrame): The filtered DataFrame.
    Returns:
        pandas.DataFrame: The DataFrame with 'Group' and 'Residue' columns.
    """
    dados_grupos = dados_filtrados.copy()

    dados_grupos['Group'] = 'Treatment'
    dados_grupos.loc[(dados_grupos['Study'] == "Ramos et al. (2024)") & (dados_grupos['Treatment'] == "120 days"), 'Group'] = "Control"

    dados_grupos['Residue'] = 'Other'
    dados_grupos.loc[dados_grupos['Study'] == "Ramos et al. (2024)", 'Residue'] = "Cattle Manure"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Kumar", na=False), 'Residue'] = "Banana Residue"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Quadar", na=False), 'Residue'] = "Coconut Husk"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Srivastava", na=False), 'Residue'] = "Urban Waste"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Santana", na=False), 'Residue'] = "Grape Marc"
    
    # Corrected categorization for "Pineapple" to "Agricultural Residue"
    # Although "Pineapple" is not in the provided csv.csv, the original R code might have inferred this logic, 
    # and the user specifically mentioned it in a previous turn.
    dados_grupos.loc[dados_grupos['Treatment'].str.contains('Pineapple', na=False, case=False), 'Residue'] = 'Agricultural Residue'


    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    """
    Prepares data for meta-analysis by calculating Log Response Ratio (lnRR) and its variance.
    Args:
        dados_grupos (pandas.DataFrame): The DataFrame with groups and residues defined.
    Returns:
        pandas.DataFrame: The DataFrame ready for meta-analysis.
    """
    # Identify variables that have a control group
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()

    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()

    # Calculate control means and std_devs
    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first')
    ).reset_index()
    
    # Handle cases where Std_Dev_control might be zero
    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)

    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(control_data, on='Variable', how='left')

    # Adjust Std_Dev for treatment group to avoid division by zero
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)

    # Calculate Log Response Ratio (lnRR) and its variance
    # Ensure Mean_control and Mean are not zero before log
    dados_meta = dados_meta[
        (dados_meta['Mean_control'] > 0) & 
        (dados_meta['Mean'] > 0) 
    ].copy()

    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])

    # Variance of lnRR calculation:
    # (Std_Dev_treatment^2 / (N_treatment * Mean_treatment^2)) + (Std_Dev_control^2 / (N_control * Mean_control^2))
    # Assuming N_treatment and N_control are represented by n() in R, which refers to number of observations.
    # For now, we'll assume a single observation per group in the original data structure for calculation,
    # as the R code uses `n()` which in this context usually means 1 for individual studies.
    # To replicate `n()` behavior from R's metafor which uses `vi` (variance of individual studies),
    # we should consider each row as an individual study effect.
    # The `n()` in the R context often refers to the number of *observations used to calculate the mean/sd for that specific study*,
    # but without explicit N in the data, it's usually treated as a single observation for meta-analysis input if not provided.
    # The provided R code for `var_lnRR` uses `n()`, which for a grouped operation within `metafor::rma` is not necessarily the group size but might be a placeholder for sample size if not provided.
    # Given the formula (SD^2 / (N * Mean^2)), if N is not given, it's often assumed as 1 for direct observation variance, or N of the group from which the mean/SD was derived.
    # Since the R code uses `n()`, and no N column is in the CSV, we'll assume it's implicitly handling per-study variance.
    # Let's adjust the variance calculation to align more closely with typical meta-analysis formulas for log response ratio when sample sizes (n) are not explicitly given.
    # A common formula for variance of lnRR is (s_treatment^2 / M_treatment^2) + (s_control^2 / M_control^2)
    # where s is standard deviation and M is mean. The n is usually in the denominator of (s^2 / n)
    # If the standard deviation given is already for the mean (Std Error of the Mean), then n is not needed.
    # However, if it's the Std Dev of the population/sample, then it should be divided by n.
    # Given the R code's `(Std_Dev_adj^2) / (n() * Mean^2)`, it implies a sample SD and an N.
    # Since we don't have N, we'll follow the most common practice for `lnRR` when N is not provided,
    # using the variance of the mean. If Std_Dev is given, and no N, we'll assume Std_Dev is already Std Error.
    # If Std_Dev is truly SD of the sample, and we need to assume N=1 for each observation as in the R code's `n()`,
    # then the formula stands.
    
    # Based on the R code, `n()` likely refers to the count of observations *within that specific row/study contribution to the model*.
    # If `n()` is simply 1 for each row in R, then the formula simplifies to (SD_adj^2 / Mean^2) + (SD_control^2 / Mean_control^2).
    
    dados_meta['var_lnRR'] = (dados_meta['Std_Dev_adj']**2 / dados_meta['Mean']**2) + \
                             (dados_meta['Std_Dev_control']**2 / dados_meta['Mean_control']**2)

    return dados_meta

def run_meta_analysis_and_plot(data, model_type="Residue", plot_title="Meta-Analysis Effect Plot"):
    """
    Executes meta-analysis using WLS and generates a coefficient plot.

    Args:
        data (pandas.DataFrame): The DataFrame prepared for meta-analysis.
        model_type (str): Type of model to run ("Residue", "Variable", "Interaction").
        plot_title (str): Title for the generated plot.

    Returns:
        tuple: A tuple containing (summary_dataframe, matplotlib_figure).
               Returns (pd.DataFrame(), None) if insufficient data or model fails.
    """
    if data.empty or len(data) < 2:
        return pd.DataFrame(), None # Need at least two data points for a regression

    formula = ""
    if model_type == "Residue":
        formula = "lnRR ~ C(Residue) - 1"
    elif model_type == "Variable":
        formula = "lnRR ~ C(Variable) - 1"
    elif model_type == "Interaction":
        formula = "lnRR ~ C(Residue) * C(Variable) - 1"
    else:
        raise ValueError("Invalid model_type. Choose 'Residue', 'Variable', or 'Interaction'.")

    try:
        # Weights for WLS regression are the inverse of the variance
        # Check if var_lnRR has non-zero values before taking inverse
        if (data['var_lnRR'] == 0).any():
            data['var_lnRR'] = data['var_lnRR'].replace(0, 1e-9) # Replace 0 with a very small number
        weights = 1 / data['var_lnRR']

        # Fit Weighted Least Squares (WLS) model
        # WLS is a common approach to approximate meta-analysis when `rma` from `metafor` isn't directly available.
        # The `weights` argument in `statsmodels.WLS` handles the inverse variance weighting.
        model = smf.wls(formula=formula, data=data, weights=weights).fit()
        
        # Prepare summary for display
        summary_df = pd.DataFrame({
            'term': model.params.index,
            'estimate': model.params.values,
            'std.error': model.bse.values,
            'p.value': model.pvalues.values
        })
        summary_df['lower'] = summary_df['estimate'] - 1.96 * summary_df['std.error']
        summary_df['upper'] = summary_df['estimate'] + 1.96 * summary_df['std.error']
        
        # Clean up term names for plots
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '', regex=False)
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)[T.', '', regex=False)
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)', '', regex=False)
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)', '', regex=False)
        summary_df['term'] = summary_df['term'].str.replace(']:C(Variable)[T.', ' x ', regex=False)
        summary_df['term'] = summary_df['term'].str.replace(']:C(Residue)[T.', ' x ', regex=False) # for interaction terms
        summary_df['term'] = summary_df['term'].str.replace(']', '', regex=False) # remove any remaining ']'
        summary_df['term'] = summary_df['term'].str.strip()


        # Generate coefficient plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort terms for better visualization
        summary_df = summary_df.sort_values(by='estimate', ascending=True)

        ax.errorbar(
            x=summary_df['estimate'],
            y=summary_df['term'],
            xerr=[summary_df['estimate'] - summary_df['lower'], summary_df['upper'] - summary_df['estimate']],
            fmt='o',
            capsize=5,
            color='darkblue',
            ecolor='gray',
            elinewidth=1.5,
            markerfacecolor='blue',
            markeredgecolor='darkblue'
        )
        ax.axvline(x=0, linestyle='--', color='red', linewidth=1) # Null effect line

        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Effect Size (lnRR) and 95% Confidence Intervals", fontsize=12)
        ax.set_ylabel(model_type + " Type", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        return summary_df, fig

    except Exception as e:
        st.error(f"Error running {model_type} model: {e}")
        return pd.DataFrame(), None


def generate_forest_plot(data, effect_sizes, variances, labels, title="Forest Plot"):
    """
    Generates a simple forest plot.
    Note: This is a simplified version compared to `metafor`'s full forest plot.

    Args:
        data (pandas.DataFrame): The DataFrame containing study details.
        effect_sizes (pd.Series): Series of effect sizes (e.g., lnRR).
        variances (pd.Series): Series of variances of effect sizes (e.g., var_lnRR).
        labels (list): List of labels for each study/row.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    if data.empty or len(effect_sizes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, len(data) * 0.5 + 2)) # Dynamic figure size

    # Calculate standard errors from variances
    std_errors = np.sqrt(variances)
    
    # Plot individual study effects
    ax.errorbar(effect_sizes, range(len(effect_sizes)), xerr=1.96 * std_errors,
                fmt='o', capsize=5, color='black', ecolor='gray')

    ax.axvline(0, color='red', linestyle='--', linewidth=1) # No effect line

    ax.set_yticks(range(len(effect_sizes)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Log Response Ratio (lnRR) and 95% Confidence Intervals", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.invert_yaxis() # Top study first
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def generate_funnel_plot(effect_sizes, variances, title="Funnel Plot for Publication Bias"):
    """
    Generates a funnel plot to visually assess publication bias.

    Args:
        effect_sizes (pd.Series): Series of effect sizes (e.g., lnRR).
        variances (pd.Series): Series of variances of effect sizes (e.g., var_lnRR).
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    if len(effect_sizes) < 2:
        return None

    std_errors = np.sqrt(variances)
    precision = 1 / std_errors # Inverse of standard error

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(effect_sizes, precision, alpha=0.7, edgecolors='black')

    # Add reference lines (if possible, based on a meta-analytic mean, or just a vertical line at 0)
    # For a proper funnel plot, you'd typically have a meta-analytic mean (e.g., from the residue_model)
    # Let's assume the overall mean effect is 0 for simplicity, or calculate sample mean
    overall_mean = effect_sizes.mean() # Simple mean for central line, could be from a meta-analysis model

    ax.axvline(x=overall_mean, color='blue', linestyle='--', label='Overall Mean Effect' if overall_mean != 0 else 'No Effect Line')

    # Add confidence regions (lines that form the funnel)
    # These lines are typically +/- 1.96 * SE
    # Max precision (min SE) will define the top width, min precision (max SE) defines bottom width
    max_se = std_errors.max()
    min_se = std_errors.min()

    # Calculate points for the funnel lines
    x_ci_upper = overall_mean + 1.96 * std_errors
    x_ci_lower = overall_mean - 1.96 * std_errors

    # Sort by precision for plotting the funnel shape
    plot_data = pd.DataFrame({'effect': effect_sizes, 'precision': precision, 'std_error': std_errors}).sort_values('precision')
    
    # Plot funnel lines
    ax.plot(overall_mean + 1.96 * plot_data['std_error'], plot_data['precision'], 'k--', linewidth=1, alpha=0.7)
    ax.plot(overall_mean - 1.96 * plot_data['std_error'], plot_data['precision'], 'k--', linewidth=1, alpha=0.7)

    ax.set_xlabel("Effect Size (lnRR)", fontsize=12)
    ax.set_ylabel("Precision (1/Standard Error)", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig
