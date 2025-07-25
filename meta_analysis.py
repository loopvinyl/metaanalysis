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
    Loads data from CSV file with robust error handling and column standardization.
    """
    try:
        # Read CSV with semicolon delimiter
        dados = pd.read_csv(file_path, sep=';', decimal='.')
        
        # Standardize column names
        column_map = {
            'Std Dev': 'Std_Dev',
            'Standard Deviation': 'Std_Dev',
            'StdDev': 'Std_Dev',
            'Original Unit': 'Original_Unit',
            'Unit': 'Original_Unit'
        }
        dados = dados.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['Study', 'Treatment', 'Variable', 'Mean', 'Std_Dev']
        missing_cols = [col for col in required_cols if col not in dados.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert numeric columns
        numeric_cols = ['Mean', 'Std_Dev']
        for col in numeric_cols:
            dados[col] = pd.to_numeric(dados[col], errors='coerce')
        
        # Drop rows with missing numeric values
        return dados.dropna(subset=numeric_cols)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def filter_irrelevant_treatments(dados):
    """
    Filters out non-vermicompost treatments exactly matching the R code.
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
    Defines Control vs Treatment groups and residue types.
    Exactly matches the R code logic.
    """
    dados_grupos = dados_filtrados.copy()
    
    # Group assignment
    dados_grupos['Group'] = 'Treatment'
    dados_grupos.loc[(dados_grupos['Study'] == "Ramos et al. (2024)") & 
                    (dados_grupos['Treatment'] == "120 days"), 'Group'] = "Control"

    # Residue assignment
    residue_map = {
        "Ramos et al. (2024)": "Cattle Manure",
        "Kumar": "Banana Residue",
        "Quadar": "Coconut Husk",
        "Srivastava": "Urban Waste",
        "Santana": "Grape Marc"
    }
    
    dados_grupos['Residue'] = 'Other'
    for study_pattern, residue in residue_map.items():
        mask = dados_grupos['Study'].str.contains(study_pattern, na=False)
        dados_grupos.loc[mask, 'Residue'] = residue
        
    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    """
    Prepares data for meta-analysis by calculating lnRR and its variance.
    Matches R code calculations exactly.
    """
    # Get variables with control groups
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()
    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()

    # Get control values
    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first'),
        N_control=('Study', 'count')
    ).reset_index()
    
    # Handle zero std dev
    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)

    # Merge with treatment data
    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(
        control_data, on='Variable', how='left'
    )
    
    # Adjust treatment std dev
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)
    
    # Calculate effect sizes (lnRR)
    valid_rows = (dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)
    dados_meta = dados_meta[valid_rows].copy()
    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    
    # Calculate variance (using n=1 as in R code)
    dados_meta['var_lnRR'] = (
        (dados_meta['Std_Dev_adj']**2 / (1 * dados_meta['Mean']**2)) + 
        (dados_meta['Std_Dev_control']**2 / (1 * dados_meta['Mean_control']**2)
    )
    
    return dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])

def run_meta_analysis_and_plot(dados_meta, model_type="Residue"):
    """
    Runs meta-analysis and generates coefficient plot.
    """
    if dados_meta.empty or len(dados_meta['Residue'].unique()) < 2:
        return pd.DataFrame(), None

    # Model specification
    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue) - 1'
        title = "Effect of Different Residues on Vermicompost Quality"
        y_label = "Residue Type"
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable) - 1'
        title = "Effect of Variables on Vermicompost Quality"
        y_label = "Variable"
    elif model_type == "Interaction":
        formula = 'lnRR ~ C(Residue):C(Variable) - 1'
        title = "Residue × Variable Interaction Effects"
        y_label = "Residue:Variable"
    else:
        raise ValueError("Invalid model_type")

    # Weighted least squares regression
    dados_meta['weights'] = 1 / dados_meta['var_lnRR']
    model = smf.wls(formula, data=dados_meta, weights=dados_meta['weights']).fit()

    # Prepare results
    summary_df = model.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
    
    # Clean term names
    if model_type == "Residue":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '').str.replace(']', '')
    elif model_type == "Variable":
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)[T.', '').str.replace(']', '')
    elif model_type == "Interaction":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '') \
                                           .str.replace(']:C(Variable)[T.', ':') \
                                           .str.replace(']', '')

    # Add confidence intervals
    summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
    summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.pointplot(x='Coef.', y='term', data=summary_df, join=False, color='blue')
    plt.hlines(y=summary_df['term'], xmin=summary_df['lower'], xmax=summary_df['upper'], color='grey')
    plt.axvline(0, linestyle='--', color='red')
    plt.title(title, fontsize=14)
    plt.xlabel("Effect Size (lnRR)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    
    return summary_df, plt.gcf()

def generate_forest_plot(dados_meta):
    """Generates a forest plot of individual study effects."""
    if dados_meta.empty:
        return None

    # Sort data
    dados_sorted = dados_meta.sort_values('lnRR').reset_index(drop=True)
    
    # Create plot
    plt.figure(figsize=(12, max(6, len(dados_sorted)*0.3)))
    sns.set_style("whitegrid")
    
    # Plot each study
    for i, row in dados_sorted.iterrows():
        ci_lower = row['lnRR'] - 1.96 * np.sqrt(row['var_lnRR'])
        ci_upper = row['lnRR'] + 1.96 * np.sqrt(row['var_lnRR'])
        label = f"{row['Residue']} - {row['Variable']} ({row['Study']})"
        
        plt.plot([ci_lower, ci_upper], [i, i], color='black')
        plt.plot(row['lnRR'], i, 'o', color='blue')
        plt.text(0.95, i, f"{row['lnRR']:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]", 
                ha='right', va='center', transform=plt.gca().transData)
        plt.text(0.01, i, label, ha='left', va='center', transform=plt.gca().transData)
    
    plt.axvline(0, linestyle='--', color='red')
    plt.title("Forest Plot of Individual Study Effects", fontsize=14)
    plt.xlabel("Log Response Ratio (lnRR) with 95% CI", fontsize=12)
    plt.yticks([])
    plt.tight_layout()
    
    return plt.gcf()

def generate_funnel_plot(dados_meta):
    """Generates a funnel plot for publication bias assessment."""
    if dados_meta.empty:
        return None

    # Calculate standard error
    dados_meta['se_lnRR'] = np.sqrt(dados_meta['var_lnRR'])
    
    # Create plot
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    
    plt.scatter(dados_meta['lnRR'], dados_meta['se_lnRR'], color='blue', alpha=0.7)
    
    # Add pseudo-confidence intervals
    max_se = dados_meta['se_lnRR'].max()
    x_vals = np.linspace(-max_se*3, max_se*3, 100)
    plt.plot(x_vals, np.abs(x_vals)/1.96, '--', color='grey', label='95% CI')
    plt.plot(x_vals, -np.abs(x_vals)/1.96, '--', color='grey')
    
    plt.axvline(0, linestyle=':', color='red', label='No Effect')
    plt.title("Funnel Plot for Publication Bias Assessment", fontsize=14)
    plt.xlabel("Log Response Ratio (lnRR)", fontsize=12)
    plt.ylabel("Standard Error", fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()
    
