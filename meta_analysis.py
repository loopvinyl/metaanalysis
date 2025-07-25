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
    """Loads and prepares data from CSV file."""
    try:
        dados = pd.read_csv(file_path, sep=';', decimal='.')
    except FileNotFoundError:
        return pd.DataFrame()
    
    dados = dados.rename(columns={'Std Dev': 'Std_Dev', 'Original Unit': 'Original_Unit'})
    
    numeric_cols = ['Mean', 'Std_Dev']
    for col in numeric_cols:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados = dados.dropna(subset=numeric_cols)
    
    return dados

def filter_irrelevant_treatments(dados):
    """Filters out irrelevant treatments."""
    treatments_to_exclude = [
        "Fresh Grape Marc", "Manure",
        "CH0 (Initial)", "CH25 (Initial)", "CH50 (Initial)", 
        "CH75 (Initial)", "CH100 (Initial)",
        "T1 (Initial)", "T2 (Initial)", "T3 (Initial)", "T4 (Initial)"
    ]
    return dados[~dados['Treatment'].isin(treatments_to_exclude)]

def define_groups_and_residues(dados_filtrados):
    """Defines control/treatment groups and residue types."""
    dados_grupos = dados_filtrados.copy()
    
    dados_grupos['Group'] = 'Treatment'
    dados_grupos.loc[(dados_grupos['Study'] == "Ramos et al. (2024)") & 
                    (dados_grupos['Treatment'] == "120 days"), 'Group'] = "Control"
    
    dados_grupos['Residue'] = 'Other'
    dados_grupos.loc[dados_grupos['Study'] == "Ramos et al. (2024)", 'Residue'] = "Cattle Manure"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Kumar", na=False), 'Residue'] = "Banana Residue"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Quadar", na=False), 'Residue'] = "Coconut Husk"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Srivastava", na=False), 'Residue'] = "Urban Waste"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Santana", na=False), 'Residue'] = "Grape Marc"
    
    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    """Prepares data for meta-analysis by calculating lnRR and its variance."""
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()
    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()
    
    # Calculate control means and std_devs with group sizes
    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first'),
        n_control=('Study', 'count')  # Number of control observations per variable
    ).reset_index()
    
    # Merge with treatment data
    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(
        control_data, on='Variable', how='left')
    
    # Calculate number of treatment observations per variable
    n_treatment = dados_meta.groupby('Variable')['Study'].transform('count')
    
    # Adjust Std_Dev to avoid division by zero
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)
    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)
    
    # Filter out rows where Mean or Mean_control is <= 0
    dados_meta = dados_meta[(dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)].copy()
    
    # Calculate Log Response Ratio (lnRR)
    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    
    # Calculate variance of lnRR using actual group sizes
    dados_meta['var_lnRR'] = (
        (dados_meta['Std_Dev_adj']**2 / (n_treatment * dados_meta['Mean']**2)) + 
        (dados_meta['Std_Dev_control']**2 / (dados_meta['n_control'] * dados_meta['Mean_control']**2))
    )
    
    # Filter out rows with NaN or infinite values
    dados_meta = dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])
    
    return dados_meta

def run_meta_analysis_and_plot(dados_meta, model_type="Residue"):
    """Runs meta-analysis and generates plots."""
    if dados_meta.empty or len(dados_meta['Residue'].unique()) < 2:
        return pd.DataFrame(), None
    
    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue) - 1'
        title = "Effect of Different Residues on Vermicompost"
        y_label = "Residue Type"
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable) - 1'
        title = "Effect of Variables in Vermicompost"
        y_label = "Variable"
    elif model_type == "Interaction":
        formula = 'lnRR ~ C(Residue):C(Variable) - 1'
        title = "Interaction Effect of Residue × Variable"
        y_label = "Residue:Variable Interaction"
    else:
        raise ValueError("Invalid model_type. Choose 'Residue', 'Variable', or 'Interaction'.")
    
    # Use inverse variance as weights (1/vi)
    dados_meta['weights'] = 1 / dados_meta['var_lnRR']
    
    try:
        model = smf.wls(formula, data=dados_meta, weights=dados_meta['weights']).fit()
    except Exception as e:
        print(f"Error fitting model for {model_type}: {e}")
        return pd.DataFrame(), None
    
    # Extract coefficients and confidence intervals
    summary_df = model.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
    
    # Clean up term names based on model type
    if model_type == "Residue":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '').str.replace(']', '')
    elif model_type == "Variable":
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)[T.', '').str.replace(']', '')
    elif model_type == "Interaction":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '') \
                                           .str.replace(']:C(Variable)[T.', ':') \
                                           .str.replace(']', '')
    
    # Calculate confidence intervals
    summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
    summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']
    
    # Create coefficient plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pointplot(x='Coef.', y='term', data=summary_df, join=False, errorbar=None, ax=ax)
    ax.hlines(y=summary_df['term'], xmin=summary_df['lower'], xmax=summary_df['upper'], color='grey')
    ax.axvline(x=0, linetype="dashed", color="red")
    ax.set_title(title)
    ax.set_xlabel("Effect Size (lnRR)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return summary_df, fig

def generate_forest_plot(dados_meta, title="Forest Plot"):
    """Generates a forest plot for individual studies."""
    if dados_meta.empty:
        return None
    
    # Sort data by lnRR for better visualization
    dados_meta = dados_meta.sort_values(by='lnRR')
    
    fig, ax = plt.subplots(figsize=(10, len(dados_meta) * 0.4 + 2))
    
    # Plot individual study effects
    for i, (idx, row) in enumerate(dados_meta.iterrows()):
        label = f"{row['Residue']} - {row['Variable']} ({row['Study']})"
        ci_lower = row['lnRR'] - 1.96 * np.sqrt(row['var_lnRR'])
        ci_upper = row['lnRR'] + 1.96 * np.sqrt(row['var_lnRR'])
        
        ax.plot([ci_lower, ci_upper], [i, i], color='gray', linestyle='-', linewidth=1)
        ax.plot(row['lnRR'], i, 's', color='blue', markersize=5)
        ax.text(ax.get_xlim()[1] + 0.1, i, f"{row['lnRR']:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]", va='center')
        ax.text(ax.get_xlim()[0] - 0.1, i, label, va='center', ha='right')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
    ax.set_yticks(range(len(dados_meta)))
    ax.set_yticklabels([])
    ax.set_xlabel("Log Response Ratio (lnRR) [95% CI]")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    
    return fig

def generate_funnel_plot(dados_meta):
    """Generates a funnel plot for publication bias."""
    if dados_meta.empty:
        return None
    
    dados_meta['se_lnRR'] = np.sqrt(dados_meta['var_lnRR'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(dados_meta['lnRR'], dados_meta['se_lnRR'], color='blue', alpha=0.7)
    
    max_se = dados_meta['se_lnRR'].max()
    x_vals = np.linspace(-max_se * 2, max_se * 2, 100)
    
    # 95% CI lines
    ax.plot(x_vals, np.abs(x_vals) / 1.96, color='grey', linestyle='--', label='95% CI')
    ax.plot(x_vals, -np.abs(x_vals) / 1.96, color='grey', linestyle='--')
    
    ax.axvline(0, color='red', linestyle=':', label='No Effect (lnRR=0)')
    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot for Publication Bias")
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    
    return fig

def run_analysis_by_variable(dados_meta, variables=None):
    """Runs analysis for specific variables (similar to section 7.1 in R code)."""
    if variables is None:
        variables = ["TOC", "N", "pH", "EC"]
    
    results = {}
    for var in variables:
        temp_data = dados_meta[dados_meta['Variable'] == var]
        if len(temp_data) > 1:
            try:
                model = smf.wls(
                    'lnRR ~ C(Residue) - 1', 
                    data=temp_data, 
                    weights=1/temp_data['var_lnRR']
                ).fit()
                
                summary_df = model.summary2().tables[1]
                summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
                summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '').str.replace(']', '')
                
                # Calculate confidence intervals
                summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
                summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.pointplot(
                    x='Coef.', y='term', data=summary_df, 
                    join=False, errorbar=None, ax=ax
                )
                ax.hlines(
                    y=summary_df['term'], 
                    xmin=summary_df['lower'], 
                    xmax=summary_df['upper'], 
                    color='grey'
                )
                ax.axvline(x=0, linetype="dashed", color="red")
                ax.set_title(f"Effect of Residues on {var}")
                ax.set_xlabel("Effect Size (lnRR)")
                ax.set_ylabel("Residue Type")
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                results[var] = {
                    'summary': summary_df,
                    'plot': fig
                }
                
            except Exception as e:
                print(f"Error analyzing variable {var}: {e}")
                results[var] = None
    
    return results
