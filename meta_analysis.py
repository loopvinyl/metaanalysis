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
    
    # Add remembered information: 'abacaxi' to 'Resíduos Agrícolas' -> 'Pineapple' to 'Agricultural Residues'
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
        (dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)
    ].copy()
    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    
    # Variance of lnRR calculation:
    # (Std_Dev_treatment^2 / (N_treatment * Mean_treatment^2)) + (Std_Dev_control^2 / (N_control * Mean_control^2))
    # Assuming N_treatment and N_control are 1 as per the R script's n() context for individual observations
    dados_meta['var_lnRR'] = (dados_meta['Std_Dev_adj']**2 / (1 * dados_meta['Mean']**2)) + \
                             (dados_meta['Std_Dev_control']**2 / (1 * dados_meta['Mean_control']**2))

    # Filter out rows with NaN or infinite values in lnRR or var_lnRR
    dados_meta = dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])
    
    # Filter out rows where var_lnRR is zero or very close to zero, as this causes issues with weights
    dados_meta = dados_meta[dados_meta['var_lnRR'] > 1e-10]

    return dados_meta


def run_meta_analysis_and_plot(dados_meta):
    """
    Executes meta-analysis models and generates plots.
    Args:
        dados_meta (pandas.DataFrame): The DataFrame prepared for meta-analysis.
    Returns:
        dict: A dictionary containing the results and plots.
    """
    results = {}
    
    if dados_meta.empty:
        print("Não há dados suficientes para a meta-análise após filtragem.")
        return results

    # Ensure weights are not zero or NaN
    weights = 1 / dados_meta['var_lnRR']
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle potential inf from 1/0
    dados_meta = dados_meta[weights > 0].copy() # Filter data where weights are valid
    weights = weights[weights > 0] # Re-filter weights

    if dados_meta.empty:
        print("Não há dados suficientes após a validação dos pesos.")
        return results

    # 6.1. Modelo por tipo de resíduo
    try:
        # Use C() for categorical variables and -1 for no intercept
        model_residue = smf.wls(
            formula='lnRR ~ C(Residue) - 1', 
            data=dados_meta, 
            weights=weights
        ).fit()
        results['model_residue'] = model_residue
        print("\n=== MODELO POR TIPO DE RESÍDUO ===\n")
        print(model_residue.summary())
        
        # Generate Coefficient Plot for Residue Model
        fig_coeff_residue = generate_coefficient_plot(model_residue, "Efeito de Resíduos no Vermicomposto", "Tipo de Resíduo")
        results['coeff_plot_residue'] = fig_coeff_residue

    except Exception as e:
        print(f"Erro ao executar o modelo por tipo de resíduo: {e}")
        results['model_residue_error'] = str(e)

    # 6.2. Modelo por variável
    try:
        model_variable = smf.wls(
            formula='lnRR ~ C(Variable) - 1', 
            data=dados_meta, 
            weights=weights
        ).fit()
        results['model_variable'] = model_variable
        print("\n=== MODELO POR VARIÁVEL ===\n")
        print(model_variable.summary())
        
        # Generate Coefficient Plot for Variable Model
        fig_coeff_variable = generate_coefficient_plot(model_variable, "Efeito por Variável no Vermicomposto", "Variável")
        results['coeff_plot_variable'] = fig_coeff_variable

    except Exception as e:
        print(f"Erro ao executar o modelo por variável: {e}")
        results['model_variable_error'] = str(e)

    # 6.3. Modelo de interação (Resíduo × Variável)
    try:
        model_interaction = smf.wls(
            formula='lnRR ~ C(Residue) * C(Variable) - 1', 
            data=dados_meta, 
            weights=weights
        ).fit()
        results['model_interaction'] = model_interaction
        print("\n=== MODELO DE INTERAÇÃO (RESÍDUO × VARIÁVEL) ===\n")
        print(model_interaction.summary())
        
        # Generate Coefficient Plot for Interaction Model (can be complex, might need filtering)
        # fig_coeff_interaction = generate_coefficient_plot(model_interaction, "Efeito de Interação (Resíduo x Variável)", "Interação")
        # results['coeff_plot_interaction'] = fig_coeff_interaction

    except Exception as e:
        print(f"Erro ao executar o modelo de interação: {e}")
        results['model_interaction_error'] = str(e)

    # 7.1. Análise por variável específica (ex: TOC, N, pH)
    important_vars = ["TOC", "N", "pH", "EC"]
    results['specific_variable_models'] = {}
    for var in important_vars:
        print(f"\n=== ANÁLISE PARA A VARIÁVEL: {var} ===\n")
        temp_data = dados_meta[dados_meta['Variable'] == var].copy()
        temp_weights = weights[dados_meta['Variable'] == var].copy() # Filter weights accordingly
        
        if len(temp_data) > 1 and not temp_weights.empty:
            try:
                temp_model = smf.wls(
                    formula='lnRR ~ C(Residue) - 1',
                    data=temp_data,
                    weights=temp_weights
                ).fit()
                results['specific_variable_models'][var] = temp_model
                print(temp_model.summary())

                # Forest plot for this specific variable
                fig_forest_specific = generate_forest_plot(
                    temp_data, temp_model, 
                    title=f"Efeito do Resíduo em {var}", 
                    slab_col='Residue'
                )
                results[f'forest_plot_{var}'] = fig_forest_specific

            except Exception as e:
                print(f"Erro ao executar o modelo para a variável {var}: {e}")
                results['specific_variable_models'][var + '_error'] = str(e)
        else:
            print(f"Não há dados suficientes para a meta-análise para a variável {var} após filtragem.")

    # 7.2. Diagnósticos (viés de publicação, heterogeneidade)
    if 'model_residue' in results:
        try:
            fig_funnel = generate_funnel_plot(dados_meta, results['model_residue'])
            results['funnel_plot'] = fig_funnel
        except Exception as e:
            print(f"Erro ao gerar o gráfico de funil: {e}")
            results['funnel_plot_error'] = str(e)
        
        # Influence plot is complex to replicate generically without a meta-analysis package.
        # It typically involves diagnostics like DFFITS, Cook's distance etc. for meta-analysis.
        # For now, it's omitted as `statsmodels.wls` doesn't provide direct equivalents of `metafor::influence` plots easily.
        # You could implement custom influence diagnostics if needed.

    return results

def generate_forest_plot(data, model, title="Forest Plot", slab_col='Residue'):
    """
    Generates a forest plot for meta-analysis results.
    Args:
        data (pandas.DataFrame): The DataFrame used for the model.
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model.
        title (str): Title of the plot.
        slab_col (str): Column to use for labels in the forest plot.
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, len(data) * 0.4 + 2)) # Adjust figure size dynamically
    
    # Get coefficients and their standard errors
    coefs = model.params
    conf_int = model.conf_int() # Get confidence intervals

    # Ensure data has a column for 'effect_size' and 'lower'/'upper' bounds
    # For a simple forest plot, we need to plot each individual study's effect size.
    # The 'data' DataFrame already contains 'lnRR' and 'var_lnRR'.
    # We need to calculate individual CIs for 'lnRR' if we want to show all studies,
    # or just show the model's aggregated effects if that's the intent.
    # The R forest plot typically shows individual studies + summary.
    # Here, we'll plot individual studies and then the aggregated model effects if desired.

    # Plot individual studies
    y_pos = np.arange(len(data))
    
    # Calculate individual study CIs for lnRR
    data['lnRR_se'] = np.sqrt(data['var_lnRR'])
    data['lnRR_lower'] = data['lnRR'] - 1.96 * data['lnRR_se']
    data['lnRR_upper'] = data['lnRR'] + 1.96 * data['lnRR_se']

    # Sort data for better visualization in forest plot
    data_sorted = data.sort_values(by='lnRR', ascending=True)
    y_pos_sorted = np.arange(len(data_sorted))

    ax.errorbar(data_sorted['lnRR'], y_pos_sorted, 
                xerr=[data_sorted['lnRR'] - data_sorted['lnRR_lower'], data_sorted['lnRR_upper'] - data_sorted['lnRR']],
                fmt='o', capsize=5, color='gray', alpha=0.7, label='Individual Studies')
    
    # Add labels
    ax.set_yticks(y_pos_sorted)
    ax.set_yticklabels(data_sorted[slab_col] + " - " + data_sorted['Variable']) # Combine Residue and Variable for slab

    # Add vertical line at null effect (lnRR = 0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_title(title)
    ax.invert_yaxis() # Top study first
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot overall effect (if applicable, e.g., model_residue for specific residue)
    # This part depends on how you want to present the summary.
    # For a model with multiple coefficients (e.g., ~C(Residue)-1), you'd plot each coefficient.
    # The 'generate_coefficient_plot' function is more suited for displaying model coefficients.
    # This forest plot focuses on individual studies, so a single summary line might not be appropriate here for multi-coeff models.

    plt.tight_layout()
    return fig

def generate_coefficient_plot(model, title="Coefficient Plot", y_label="Term"):
    """
    Generates a coefficient plot with confidence intervals.
    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model.
        title (str): Title of the plot.
        y_label (str): Label for the y-axis.
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    results_df = pd.DataFrame({
        'estimate': model.params,
        'lower': model.conf_int()[0],
        'upper': model.conf_int()[1]
    })
    results_df['term'] = results_df.index
    
    # Remove 'C(Residue)[T.' or 'C(Variable)[T.' prefixes for cleaner labels
    results_df['term'] = results_df['term'].str.replace(r'C\(Residue\)\[T\.', '', regex=True)
    results_df['term'] = results_df['term'].str.replace(r'C\(Variable\)\[T\.', '', regex=True)
    results_df['term'] = results_df['term'].str.replace(r'\]', '', regex=True)
    results_df['term'] = results_df['term'].str.replace(r'\:C\(Variable\)\[T\.', ' x ', regex=True) # For interaction terms
    results_df['term'] = results_df['term'].str.replace(r'C\(Variable\)', 'Variable', regex=True) # For single variable terms

    fig, ax = plt.subplots(figsize=(10, max(5, len(results_df) * 0.6)))
    ax.errorbar(x=results_df['estimate'], y=results_df['term'], 
                xerr=[results_df['estimate'] - results_df['lower'], results_df['upper'] - results_df['estimate']],
                fmt='o', capsize=5, color='blue')
    ax.axvline(x=0, color='red', linestyle='dashed')
    ax.set_xlabel("Tamanho do Efeito (lnRR)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig

def generate_funnel_plot(data, model):
    """
    Generates a funnel plot to assess publication bias.
    Args:
        data (pandas.DataFrame): The original data with lnRR and var_lnRR.
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model.
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate standard error from variance of lnRR
    data['se_lnRR'] = np.sqrt(data['var_lnRR'])

    # Get the overall effect estimate (e.g., from the intercept or average of coefficients if no intercept)
    # For a model with C(Residue) - 1, there's no overall intercept.
    # We can use the mean of the lnRR or the estimate from a fixed-effects model if one was run for overall effect.
    # For a funnel plot, often you plot against the *overall* effect.
    # Let's use the average of the observed lnRR as the center for the funnel plot lines for now.
    # Or, if `model` is a single overall effect model, use its estimate.
    
    # If using a model like model_residue or model_variable, the 'summary'
    # gives estimates for each category, not a single overall effect.
    # For a typical funnel plot, we need a single overall effect (e.g., from an intercept-only model).
    # Let's calculate a pooled mean lnRR for the center of the funnel.
    
    pooled_lnRR = np.average(data['lnRR'], weights=1/data['var_lnRR']) # Inverse variance weighted average

    # Plot points
    ax.scatter(data['lnRR'], data['se_lnRR'], alpha=0.7, edgecolors='w', s=50)

    # Add funnel lines (approximate for typical standard errors)
    # The funnel boundaries are typically +/- 1.96 * SE, centered around the pooled effect.
    max_se = data['se_lnRR'].max()
    se_range = np.linspace(0, max_se, 100)
    
    # Upper and lower bounds for 95% CI
    upper_bound = pooled_lnRR + 1.96 * se_range
    lower_bound = pooled_lnRR - 1.96 * se_range
    
    ax.plot(upper_bound, se_range, linestyle='--', color='grey')
    ax.plot(lower_bound, se_range, linestyle='--', color='grey')
    ax.axvline(x=pooled_lnRR, color='red', linestyle=':', label=f'Pooled lnRR: {pooled_lnRR:.2f}')

    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Standard Error (SE)")
    ax.set_title("Gráfico de Funil para Viés de Publicação")
    ax.invert_yaxis() # Larger SE at the bottom
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig
