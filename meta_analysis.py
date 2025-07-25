import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import os

# --- Funções de Pré-processamento e Cálculo ---

def load_and_prepare_data(file_path):
    """
    Carrega o arquivo Excel e realiza a preparação inicial dos dados.
    Retorna um DataFrame vazio em caso de erro de leitura ou processamento.
    """
    dados = pd.DataFrame()
    
    try:
        dados = pd.read_excel(file_path)
        print(f"Successfully loaded data from: {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Excel file '{file_path}': {e}")
        return pd.DataFrame()

    try:
        # Renomear colunas para consistência e facilitar o acesso
        # Remover 'Residues': 'Residue' daqui, pois Residue será criado dinamicamente
        dados = dados.rename(columns={
            'Variable': 'Variable',
            'Study': 'Study',
            'Treatment': 'Treatment',
            'Mean': 'Mean',
            'Std Dev': 'Std_Dev',
            'Unit': 'Unit',
            'Original Unit': 'Original_Unit', # Renomeado para evitar espaço
            'Notes': 'Notes'
        })
        # Converter colunas numéricas, forçando erros para NaN
        dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
        dados['Std_Dev'] = pd.to_numeric(dados['Std_Dev'], errors='coerce')

        # Substituir 0 em Std_Dev por um valor pequeno para evitar divisão por zero
        dados['Std_Dev'] = dados['Std_Dev'].replace(0, 0.001)

        # Remover linhas com valores NaN nas colunas críticas após a conversão
        # Removida 'Residue' do subset de dropna, pois ela será criada depois.
        dados.dropna(subset=['Mean', 'Std_Dev', 'Treatment', 'Variable', 'Study'], inplace=True)
        
        if dados.empty:
            print("DataFrame became empty after numeric conversion and dropping rows with missing critical data.")
            return pd.DataFrame()
            
        print("Data preparation successful.")
        return dados
    except Exception as e:
        print(f"Error during data preparation after initial load for '{file_path}': {e}")
        return pd.DataFrame()


def filter_irrelevant_treatments(df):
    """
    Filtra tratamentos irrelevantes para a meta-análise de vermicompostagem.
    """
    tratamentos_excluir = [
        'Fresh Grape Marc', 'Manure', 'Initial Vermicompost',
        'Initial Grape Marc', 'Initial Soil', 'Initial Sewage Sludge',
        'Initial Waste Mixture', 'Initial Municipal Solid Waste',
        'Initial Fruit Waste', 'Initial Vegetable Waste',
        'Initial Paddy Straw', 'Initial Sugarcane Bagasse',
        'Initial Wheat Straw', 'Initial Coconut Coir',
        'Initial Coffee Pulp', 'Initial Rice Straw',
        'Initial Corn Stover', 'Initial Leaf Litter',
        'Initial Paper Waste', 'Initial Food Waste',
        'Initial Brewery Waste', 'Initial Distillery Waste',
        'Initial Industrial Waste', 'Initial Livestock Waste',
        'Initial Agro-waste', 'Initial Slurry',
        'CH0 (Initial)', 'CH25 (Initial)', 'CH50 (Initial)', 'CH75 (Initial)', 'CH100 (Initial)',
        'T1 (Initial)', 'T2 (Initial)', 'T3 (Initial)', 'T4 (Initial)'
    ]
    df_filtered = df[~df['Treatment'].isin(tratamentos_excluir)].copy()
    return df_filtered


def define_groups_and_residues(df):
    """
    Define grupos de controle e tratamento e atribui categorias de resíduos.
    """
    df['Treatment'] = df['Treatment'].astype(str)
    df['Study'] = df['Study'].astype(str) # Garantir que 'Study' é string

    # Classificar como Controle ou Tratamento
    # Replicando a lógica do R: "Ramos et al. (2024)" & "120 days" é o controle
    df['Group'] = np.where(
        (df['Study'] == "Ramos et al. (2024)") & (df['Treatment'] == "120 days"),
        "Control",
        "Treatment"
    )

    # --- CORREÇÃO AQUI: Criar a coluna 'Residue' baseada na coluna 'Study' ---
    def assign_residue_from_study(row):
        study = row['Study']
        if "Ramos et al. (2024)" == study:
            return "Cattle Manure"
        elif "Kumar" in study:
            return "Banana Residue"
        elif "Quadar" in study:
            return "Coconut Husk"
        elif "Srivastava" in study:
            return "Urban Waste"
        elif "Santana" in study:
            return "Grape Marc"
        else:
            return "Other"

    df['Residue'] = df.apply(assign_residue_from_study, axis=1)
    # --- FIM DA CORREÇÃO ---

    # Atribuir tipo de resíduo (agora usando a coluna 'Residue' recém-criada)
    def assign_residue_type(row):
        residue = str(row['Residue']).lower()
        if 'sewage' in residue or 'sludge' in residue or 'municipal' in residue or 'waste' in residue or 'industrial' in residue or 'brewery' in residue or 'distillery' in residue or 'urban' in residue: # Adicionado 'urban'
            return 'Sewage Sludge/Industrial Waste'
        elif 'grape' in residue or 'fruit' in residue or 'vegetable' in residue or 'pineapple' in residue or 'coffee' in residue or 'sugarcane' in residue or 'paddy' in residue or 'rice' in residue or 'corn' in residue or 'leaf' in residue or 'agro-waste' in residue or 'livestock' in residue or 'manure' in residue or 'straw' in residue or 'abacaxi' in residue or 'banana' in residue or 'coconut' in residue: # Adicionado 'banana', 'coconut'
            return 'Agricultural Residue'
        elif 'paper' in residue or 'wood' in residue or 'lignin' in residue:
            return 'Paper/Wood Waste'
        else:
            return 'Other'

    df['Residue_Type'] = df.apply(assign_residue_type, axis=1)

    return df

def prepare_for_meta_analysis(df_groups):
    """
    Prepara os dados para meta-análise, calculando lnRR e var_lnRR.
    Esta função foi adaptada para replicar a lógica de efeitos aleatórios simples,
    onde cada par controle-tratamento contribui para o cálculo.
    """
    dados_meta = []

    # Identificar variáveis que têm grupo controle
    variables_with_control = df_groups.loc[df_groups['Group'] == "Control", 'Variable'].unique()
    
    # Filtrar df_groups para incluir apenas variáveis com controles presentes
    df_groups_filtered_by_control_var = df_groups[df_groups['Variable'].isin(variables_with_control)]

    # Iterar sobre cada estudo e variável
    for (study, variable, residue_type), group_df in df_groups_filtered_by_control_var.groupby(['Study', 'Variable', 'Residue_Type']):
        control_data = group_df[group_df['Group'] == 'Control']
        treatment_data = group_df[group_df['Group'] == 'Treatment']

        if not control_data.empty and not treatment_data.empty:
            control_mean = control_data['Mean'].iloc[0]
            control_sd = control_data['Std_Dev'].iloc[0]
            control_n = control_data['N'].iloc[0] if 'N' in control_data.columns and not control_data['N'].empty else 10 # Default N=10 se não houver N

            for index, row in treatment_data.iterrows():
                treatment_mean = row['Mean']
                treatment_sd = row['Std_Dev']
                treatment_n = row['N'] if 'N' in treatment_data.columns and not row['N'].empty else 10 # Default N=10 se não houver N

                if control_mean == 0:
                    control_mean = 0.001
                if treatment_mean == 0:
                    treatment_mean = 0.001

                lnRR = np.log(treatment_mean / control_mean)
                var_lnRR = (treatment_sd**2 / (treatment_n * treatment_mean**2)) + \
                           (control_sd**2 / (control_n * control_mean**2))

                dados_meta.append({
                    'Study': study,
                    'Variable': variable,
                    'Residue_Type': residue_type,
                    'lnRR': lnRR,
                    'var_lnRR': var_lnRR,
                    'Treatment_Mean': treatment_mean,
                    'Control_Mean': control_mean,
                    'Treatment_N': treatment_n,
                    'Control_N': control_n
                })

    dados_meta = pd.DataFrame(dados_meta)
    dados_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    dados_meta.dropna(subset=['lnRR', 'var_lnRR'], inplace=True)

    return dados_meta


# --- Funções de Análise e Plotagem ---

def run_meta_analysis_and_plot(data, model_type="Residue"):
    """
    Executa o modelo de meta-análise e gera um gráfico de coeficientes.
    Utiliza WLS como uma aproximação para modelos de meta-análise de efeitos fixos/mistos
    pelo método de regressão. Para heterogeneidade, seriam necessários cálculos adicionais.
    """
    if data.empty:
        return pd.DataFrame(), None

    data = sm.add_constant(data, has_constant='add')

    model = None
    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue_Type)'
        if len(data['Residue_Type'].unique()) > 1:
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else:
            print("Warning: Only one residue type found, cannot run regression by Residue Type.")
            return pd.DataFrame(), None
            
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable)'
        if len(data['Variable'].unique()) > 1:
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else:
            print("Warning: Only one variable type found, cannot run regression by Variable.")
            return pd.DataFrame(), None

    elif model_type == "Interaction":
        formula = 'lnRR ~ C(Residue_Type) * C(Variable)'
        if len(data['Residue_Type'].unique()) > 1 and len(data['Variable'].unique()) > 1:
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else:
            print("Warning: Not enough unique Residue Types or Variables for interaction analysis.")
            return pd.DataFrame(), None
    else:
        print(f"Invalid model type: {model_type}")
        return pd.DataFrame(), None

    try:
        results = model.fit()
    except Exception as e:
        print(f"Error fitting WLS model for {model_type}: {e}")
        return pd.DataFrame(), None

    summary_df = results.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
    summary_df = summary_df[['term', 'Coef.', 'Std.Err.', 't', 'P>|t|', '[0.025', '0.975]']]
    summary_df.columns = ['term', 'estimate', 'std_error', 't_value', 'p_value', 'conf_low', 'conf_high']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=0, linestyle="dashed", color="red")
    plot_data = summary_df[~summary_df['term'].str.contains('Intercept|C\(', na=False)]
    
    if plot_data.empty and not summary_df.empty:
        plot_data = summary_df[summary_df['term'] == 'const']

    plot_data = plot_data.sort_values(by='estimate')

    y_pos = np.arange(len(plot_data))
    ax.barh(y_pos, plot_data['estimate'], xerr=plot_data['std_error']*1.96, align='center', color='skyblue', ecolor='gray', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['term'])
    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_title(f"Meta-Analysis Coefficients by {model_type}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return summary_df, fig


def generate_forest_plot(data):
    """
    Gera um Forest Plot básico para efeitos individuais.
    """
    if data.empty or 'lnRR' not in data.columns or 'var_lnRR' not in data.columns:
        print("Data is empty or missing required columns for Forest Plot.")
        return None

    data['se_lnRR'] = np.sqrt(data['var_lnRR'])
    data['lower_ci'] = data['lnRR'] - 1.96 * data['se_lnRR']
    data['upper_ci'] = data['lnRR'] + 1.96 * data['se_lnRR']

    fig, ax = plt.subplots(figsize=(10, len(data) * 0.4 + 2))

    data_sorted = data.sort_values(by='lnRR', ascending=False)

    y_pos = np.arange(len(data_sorted))
    ax.errorbar(data_sorted['lnRR'], y_pos, xerr=1.96 * data_sorted['se_lnRR'],
                fmt='o', markersize=5, capsize=5, color='blue', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data_sorted['Study'] + ' - ' + data_sorted['Variable'])
    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_title("Forest Plot of Individual Studies")
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    min_x = data_sorted['lower_ci'].min() * 1.1
    max_x = data_sorted['upper_ci'].max() * 1.1
    ax.set_xlim(min(min_x, -1), max(max_x, 1))

    plt.tight_layout()
    return fig

def generate_funnel_plot(data):
    """
    Gera um Funnel Plot básico para avaliar viés de publicação.
    """
    if data.empty or 'lnRR' not in data.columns or 'var_lnRR' not in data.columns:
        print("Data is empty or missing required columns for Funnel Plot.")
        return None

    data['precision'] = 1 / np.sqrt(data['var_lnRR'])

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data['lnRR'], data['precision'], alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Precision (1/Standard Error)")
    ax.set_title("Funnel Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig
