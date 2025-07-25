import streamlit as st

st.set_page_config(layout="wide", page_title="Vermicompost Meta-Analysis")
st.write("🚀 App iniciado com sucesso.")  # Adicione isso no topo


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
        dados = pd.read_csv(file_path, sep=';', decimal='.', encoding='latin1')
    except FileNotFoundError:
        return pd.DataFrame()  # Return empty if file not found
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()

    # Rename columns to avoid issues with spaces
    dados = dados.rename(columns={'Std Dev': 'Std_Dev', 'Original Unit': 'Original_Unit'})

    # Convert numeric columns, handling potential errors
    numeric_cols = ['Mean', 'Std_Dev']
    for col in numeric_cols:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados = dados.dropna(subset=numeric_cols)  # Drop rows where conversion failed

    return dados

def filter_irrelevant_treatments(dados):
    treatments_to_exclude = [
        "Fresh Grape Marc", "Manure",
        "CH0 (Initial)", "CH25 (Initial)", "CH50 (Initial)", "CH75 (Initial)", "CH100 (Initial)",
        "T1 (Initial)", "T2 (Initial)", "T3 (Initial)", "T4 (Initial)"
    ]
    return dados[~dados['Treatment'].isin(treatments_to_exclude)]

def define_groups_and_residues(dados_filtrados):
    dados_grupos = dados_filtrados.copy()

    dados_grupos['Group'] = 'Treatment'
    dados_grupos.loc[
        (dados_grupos['Study'] == "Ramos et al. (2024)") & (dados_grupos['Treatment'] == "120 days"),
        'Group'
    ] = "Control"

    dados_grupos['Residue'] = 'Other'
    dados_grupos.loc[dados_grupos['Study'] == "Ramos et al. (2024)", 'Residue'] = "Cattle Manure"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Kumar", na=False), 'Residue'] = "Banana Residue"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Quadar", na=False), 'Residue'] = "Coconut Husk"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Srivastava", na=False), 'Residue'] = "Urban Waste"
    dados_grupos.loc[dados_grupos['Study'].str.contains("Santana", na=False), 'Residue'] = "Grape Marc"
    dados_grupos.loc[dados_grupos['Treatment'].str.contains('Pineapple', na=False, case=False), 'Residue'] = 'Agricultural Residue'

    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()
    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()

    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first')
    ).reset_index()

    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)
    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(control_data, on='Variable', how='left')
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)

    dados_meta = dados_meta[
        (dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)
    ].copy()

    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    dados_meta['var_lnRR'] = (dados_meta['Std_Dev_adj']**2 / (1 * dados_meta['Mean']**2)) + \
                             (dados_meta['Std_Dev_control']**2 / (1 * dados_meta['Mean_control']**2))

    dados_meta = dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])

    return dados_meta

def run_meta_analysis_and_plot(dados_meta, model_type="Residue"):
    if dados_meta.empty or len(dados_meta['Residue'].unique()) < 2:
        return pd.DataFrame(), None

    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue) - 1'
        title = "Effect of Different Residues on Vermicompost"
        x_label = "Effect Size (lnRR)"
        y_label = "Residue Type"
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable) - 1'
        title = "Effect of Variables in Vermicompost"
        x_label = "Effect Size (lnRR)"
        y_label = "Variable"
    elif model_type == "Interaction":
        formula = 'lnRR ~ C(Residue):C(Variable) - 1'
        title = "Interaction Effect of Residue × Variable"
        x_label = "Effect Size (lnRR)"
        y_label = "Residue:Variable Interaction"
    else:
        raise ValueError("Invalid model_type. Choose 'Residue', 'Variable', or 'Interaction'.")

    dados_meta['weights'] = 1 / dados_meta['var_lnRR']
    
    try:
        model = smf.wls(formula, data=dados_meta, weights=dados_meta['weights']).fit()
    except Exception as e:
        print(f"Error fitting model for {model_type}: {e}")
        return pd.DataFrame(), None

    summary_df = model.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={'index': 'term'})

    if model_type == "Residue":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '', regex=False).str.replace(']', '', regex=False)
    elif model_type == "Variable":
        summary_df['term'] = summary_df['term'].str.replace('C(Variable)[T.', '', regex=False).str.replace(']', '', regex=False)
    elif model_type == "Interaction":
        summary_df['term'] = summary_df['term'].str.replace('C(Residue)[T.', '', regex=False) \
                                               .str.replace(']:C(Variable)[T.', ':', regex=False) \
                                               .str.replace(']', '', regex=False)

    summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
    summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pointplot(x='Coef.', y='term', data=summary_df, join=False, errorbar=None, ax=ax)
    ax.hlines(y=summary_df['term'], xmin=summary_df['lower'], xmax=summary_df['upper'], color='grey')
    ax.axvline(x=0, linestyle="dashed", color="red")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return summary_df, fig

def generate_forest_plot(dados_meta, title="Forest Plot"):
    if dados_meta.empty:
        return None

    dados_meta = dados_meta.sort_values(by='lnRR')
    fig, ax = plt.subplots(figsize=(10, len(dados_meta) * 0.4 + 2))

    for i, row in dados_meta.iterrows():
        label = f"{row['Residue']} - {row['Variable']} ({row['Study']})"
        ci_lower = row['lnRR'] - 1.96 * np.sqrt(row['var_lnRR'])
        ci_upper = row['lnRR'] + 1.96 * np.sqrt(row['var_lnRR'])

        ax.plot([ci_lower, ci_upper], [i, i], color='gray', linewidth=1)
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
    if dados_meta.empty:
        return None

    dados_meta['se_lnRR'] = np.sqrt(dados_meta['var_lnRR'])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(dados_meta['lnRR'], dados_meta['se_lnRR'], color='blue', alpha=0.7)

    max_se = dados_meta['se_lnRR'].max()
    x_vals = np.linspace(-max_se * 2, max_se * 2, 100)
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
