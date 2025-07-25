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
    Matches R code functionality with ';' delimiter and '.' decimal.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Prepared DataFrame.
    """
    try:
        dados = pd.read_csv(file_path, sep=';', decimal='.')
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()

    # Rename columns to avoid issues with spaces
    dados = dados.rename(columns={
        'Std Dev': 'Std_Dev',
        'Original Unit': 'Original_Unit'
    })

    # Convert numeric columns, handling potential errors
    numeric_cols = ['Mean', 'Std_Dev']
    for col in numeric_cols:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados = dados.dropna(subset=numeric_cols)

    return dados

def filter_irrelevant_treatments(dados):
    """
    Filters out irrelevant treatments from the dataset.
    Matches exact treatment exclusions from R code.

    Args:
        dados (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Filtered DataFrame.
    """
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
    Exactly matches R code logic for group and residue assignment.

    Args:
        dados_filtrados (pandas.DataFrame): Filtered DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with groups and residues.
    """
    dados_grupos = dados_filtrados.copy()

    # Group assignment
    dados_grupos['Group'] = 'Treatment'
    dados_grupos.loc[(dados_grupos['Study'] == "Ramos et al. (2024)") & 
                    (dados_grupos['Treatment'] == "120 days"), 'Group'] = "Control"

    # Residue assignment
    dados_grupos['Residue'] = 'Other'
    dados_grupos.loc[dados_grupos['Study'] == "Ramos et al. (2024)", 'Residue'] = "Cattle Manure"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Kumar", na=False), 'Residue'] = "Banana Residue"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Quadar", na=False), 'Residue'] = "Coconut Husk"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Srivastava", na=False), 'Residue'] = "Urban Waste"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Santana", na=False), 'Residue'] = "Grape Marc"

    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    """
    Prepares data for meta-analysis by calculating Log Response Ratio (lnRR) and its variance.
    Matches R code calculations exactly.

    Args:
        dados_grupos (pandas.DataFrame): DataFrame with groups and residues.

    Returns:
        pandas.DataFrame: DataFrame ready for meta-analysis.
    """
    # Identify variables with control group
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()

    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()

    # Calculate control means and std_devs
    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first')
    ).reset_index()
    
    # Handle zero std_dev (matches R code)
    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)

    # Merge with treatment data
    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(control_data, on='Variable', how='left')

    # Adjust Std_Dev for treatment (matches R code)
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)

    # Calculate lnRR and variance (exact R code calculations)
    dados_meta = dados_meta[
        (dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)
    ].copy()

    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    
    # Variance calculation matches R code (using n=1 as in original R implementation)
    dados_meta['var_lnRR'] = (dados_meta['Std_Dev_adj']**2 / (1 * dados_meta['Mean']**2)) + \
                             (dados_meta['Std_Dev_control']**2 / (1 * dados_meta['Mean_control']**2))
    
    # Filter invalid values
    dados_meta = dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])
    
    return dados_meta

def run_meta_analysis_and_plot(dados_meta, model_type="Residue"):
    """
    Runs meta-analysis and generates coefficient plot.
    Uses weighted least squares to approximate R's metafor package.

    Args:
        dados_meta (pandas.DataFrame): Prepared meta-analysis data.
        model_type (str): Model type ("Residue", "Variable", or "Interaction").

    Returns:
        tuple: (summary DataFrame, matplotlib figure)
    """
    if dados_meta.empty or len(dados_meta['Residue'].unique()) < 2:
        return pd.DataFrame(), None

    # Model formulas match R code
    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue) - 1'
        title = "Effect of Different Residues on Vermicompost Quality"
        x_label = "Effect Size (lnRR)"
        y_label = "Residue Type"
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable) - 1'
        title = "Effect of Variables on Vermicompost Quality"
        x_label = "Effect Size (lnRR)"
        y_label = "Variable"
    elif model_type == "Interaction":
        formula = 'lnRR ~ C(Residue):C(Variable) - 1'
        title = "Interaction Effects: Residue × Variable on Vermicompost Quality"
        x_label = "Effect Size (lnRR)"
        y_label = "Residue:Variable Interaction"
    else:
        raise ValueError("Invalid model_type")

    # Weighted least squares (1/var as weights)
    dados_meta['weights'] = 1 / dados_meta['var_lnRR']
    
    try:
        model = smf.wls(formula, data=dados_meta, weights=dados_meta['weights']).fit()
    except Exception as e:
        print(f"Model fitting error: {e}")
        return pd.DataFrame(), None

    # Extract and format results
    summary_df = model.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
    
    # Clean term names to match R output
    if model_type == "Residue":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '').str.replace(']', '')
    elif model_type == "Variable":
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)[T.', '').str.replace(']', '')
    elif model_type == "Interaction":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '') \
                                           .str.replace(']:C(Variable)[T.', ':') \
                                           .str.replace(']', '')

    # Calculate 95% CIs
    summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
    summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']

    # Create publication-quality coefficient plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.pointplot(x='Coef.', y='term', data=summary_df, join=False, 
                  errorbar=None, ax=ax, color='blue')
    ax.hlines(y=summary_df['term'], xmin=summary_df['lower'], 
              xmax=summary_df['upper'], color='grey', alpha=0.7)
    ax.axvline(x=0, linetype="--", color="red", alpha=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    plt.tight_layout()

    return summary_df, fig

def generate_forest_plot(dados_meta):
    """
    Generates a forest plot of individual study effects.
    Matches R's forest plot functionality as closely as possible.

    Args:
        dados_meta (pandas.DataFrame): Prepared meta-analysis data.

    Returns:
        matplotlib.figure.Figure: Forest plot figure.
    """
    if dados_meta.empty:
        return None

    # Sort by effect size
    dados_meta = dados_meta.sort_values(by='lnRR')

    # Dynamic figure size
    fig_height = max(4, len(dados_meta) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.set_style("whitegrid")

    # Plot each study
    for i, (_, row) in enumerate(dados_meta.iterrows()):
        label = f"{row['Residue']} - {row['Variable']} ({row['Study']})"
        ci_lower = row['lnRR'] - 1.96 * np.sqrt(row['var_lnRR'])
        ci_upper = row['lnRR'] + 1.96 * np.sqrt(row['var_lnRR'])
        
        # CI line
        ax.plot([ci_lower, ci_upper], [i, i], color='black', linewidth=1)
        # Point estimate
        ax.plot(row['lnRR'], i, 's', color='blue', markersize=8)
        # Effect size text
        ax.text(0.95, i, f"{row['lnRR']:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]", 
                ha='right', va='center', transform=ax.get_yaxis_transform())
        # Study label
        ax.text(0.01, i, label, ha='left', va='center', transform=ax.get_yaxis_transform())

    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.set_title("Forest Plot of Individual Study Effects", fontsize=14, pad=20)
    ax.set_xlabel("Log Response Ratio (lnRR) with 95% CI", fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-3, 3)  # Reasonable default range
    plt.tight_layout()

    return fig

def generate_funnel_plot(dados_meta):
    """
    Generates a funnel plot for publication bias assessment.
    Matches R's funnel plot functionality.

    Args:
        dados_meta (pandas.DataFrame): Prepared meta-analysis data.

    Returns:
        matplotlib.figure.Figure: Funnel plot figure.
    """
    if dados_meta.empty:
        return None

    # Calculate standard error
    dados_meta['se_lnRR'] = np.sqrt(dados_meta['var_lnRR'])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("whitegrid")

    # Plot studies
    ax.scatter(dados_meta['lnRR'], dados_meta['se_lnRR'], color='blue', alpha=0.7)

    # Add 95% CI lines
    max_se = dados_meta['se_lnRR'].max()
    x_vals = np.linspace(-max_se * 2, max_se * 2, 100)
    ax.plot(x_vals, np.abs(x_vals) / 1.96, color='grey', linestyle='--', label='95% CI')
    ax.plot(x_vals, -np.abs(x_vals) / 1.96, color='grey', linestyle='--')

    ax.axvline(0, color='red', linestyle=':', label='No Effect')
    ax.set_title("Funnel Plot for Publication Bias Assessment", fontsize=14, pad=20)
    ax.set_xlabel("Log Response Ratio (lnRR)", fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    return fig
