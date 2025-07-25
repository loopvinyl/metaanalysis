import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import os
import io # Adicionado para auxiliar na leitura de arquivos em memória, caso necessário no futuro

# --- Funções de Pré-processamento e Cálculo ---

def load_and_prepare_data(file_path):
    """
    Carrega o arquivo de dados (agora especificamente CSV) e realiza a preparação inicial.
    Retorna um DataFrame vazio em caso de erro de leitura ou processamento.
    """
    dados = pd.DataFrame()
    
    # Ajustando para ler o CSV, conforme o script R original e seus dados
    # Forçaremos a leitura de 'data/csv.csv' para consistência com o R
    csv_file_path = "data/csv.csv" 
    
    try:
        # Usar read_csv com delimitador e decimal_mark como no script R
        dados = pd.read_csv(csv_file_path, delimiter=';', decimal='.')
        print(f"Successfully loaded data from: {csv_file_path}")
        print(f"Initial data loaded: {len(dados)} rows.") # Log
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please ensure it is in the 'data/' directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV file '{csv_file_path}': {e}")
        return pd.DataFrame()

    try:
        # Renomear colunas para consistência e facilitar o acesso
        # Note: No R, `Std Dev` virou `Std_Dev` e `Original Unit` virou `Original_Unit`
        dados = dados.rename(columns={
            'Variable': 'Variable',
            'Study': 'Study',
            'Treatment': 'Treatment',
            'Mean': 'Mean',
            'Std Dev': 'Std_Dev', # Ajustado para o nome original do CSV
            'Unit': 'Unit',
            'Original Unit': 'Original_Unit', # Ajustado para o nome original do CSV
            'Notes': 'Notes'
        })
        
        # Converter colunas numéricas, forçando erros para NaN
        dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
        dados['Std_Dev'] = pd.to_numeric(dados['Std_Dev'], errors='coerce')

        # Substituir 0 em Std_Dev por um valor pequeno para evitar divisão por zero
        dados['Std_Dev'] = dados['Std_Dev'].replace(0, 0.001)

        # Remover linhas com valores NaN nas colunas críticas após a conversão
        dados.dropna(subset=['Mean', 'Std_Dev', 'Treatment', 'Variable', 'Study'], inplace=True)
        print(f"After initial numeric conversion and dropping NaNs: {len(dados)} rows.") # Log
        
        if dados.empty:
            print("DataFrame became empty after numeric conversion and dropping rows with missing critical data.")
            return pd.DataFrame()
            
        print("Data preparation successful.")
        return dados
    except Exception as e:
        print(f"Error during data preparation after initial load for '{csv_file_path}': {e}")
        return pd.DataFrame()


def filter_irrelevant_treatments(df):
    """
    Filtra tratamentos irrelevantes para a meta-análise de vermicompostagem,
    sendo IDÊNTICA à lógica do script R.
    """
    # Lista de tratamentos a serem excluídos, replicando exatamente o script R.
    tratamentos_excluir = [
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
    # Certifique-se de que a coluna 'Treatment' é do tipo string para a comparação
    df['Treatment'] = df['Treatment'].astype(str)
    df_filtered = df[~df['Treatment'].isin(tratamentos_excluir)].copy()
    print(f"After filtering irrelevant treatments (identical to R script): {len(df_filtered)} rows.") # Log
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
    print(f"After assigning Group (Control/Treatment): {len(df)} rows.") # Log
    print(f"Group counts: {df['Group'].value_counts().to_dict()}") # Log
    
    # Criar a coluna 'Residue' baseada na coluna 'Study', replicando a lógica do R
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
    print(f"After creating Residue column: {len(df)} rows.") # Log
    print(f"Residue type counts: {df['Residue'].value_counts().to_dict()}") # Log

    # Atribuir tipo de resíduo (agora usando a coluna 'Residue' recém-criada)
    # Lógica ligeiramente expandida para incluir 'banana' e 'coconut' explicitamente,
    # caso não sejam pegos por 'fruit' ou 'vegetable' e para 'urban' em industrial waste.
    def assign_residue_type(row):
        residue = str(row['Residue']).lower()
        if 'sewage' in residue or 'sludge' in residue or 'municipal' in residue or 'waste' in residue or 'industrial' in residue or 'brewery' in residue or 'distillery' in residue or 'urban' in residue: # Adicionado 'urban'
            return 'Sewage Sludge/Industrial Waste'
        elif 'grape' in residue or 'fruit' in residue or 'vegetable' in residue or 'pineapple' in residue or 'coffee' in residue or 'sugarcane' in residue or 'paddy' in residue or 'rice' in residue or 'corn' in residue or 'leaf' in residue or 'agro-waste' in residue or 'livestock' in residue or 'manure' in residue or 'straw' in residue or 'abacaxi' in residue or 'banana' in residue or 'coconut' in residue: # Adicionado 'banana', 'coconut', e lembrando de 'abacaxi'
            return 'Agricultural Residue'
        elif 'paper' in residue or 'wood' in residue or 'lignin' in residue:
            return 'Paper/Wood Waste'
        else:
            return 'Other'

    df['Residue_Type'] = df.apply(assign_residue_type, axis=1)
    print(f"After assigning Residue_Type: {len(df)} rows.") # Log
    print(f"Residue_Type counts: {df['Residue_Type'].value_counts().to_dict()}") # Log

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
    print(f"Variables with control group found: {variables_with_control.tolist()}") # Log
    
    # Filtrar df_groups para incluir apenas variáveis com controles presentes
    df_groups_filtered_by_control_var = df_groups[df_groups['Variable'].isin(variables_with_control)]
    print(f"After filtering for variables with controls: {len(df_groups_filtered_by_control_var)} rows.") # Log

    # Iterar sobre cada estudo e variável para encontrar pares Controle-Tratamento
    # Agrupamos por 'Study', 'Variable' e 'Residue_Type'
    grouped_data = df_groups_filtered_by_control_var.groupby(['Study', 'Variable', 'Residue_Type'])
    print(f"Number of unique Study-Variable-Residue_Type groups: {len(grouped_data)}") # Log

    for (study, variable, residue_type), group_df in grouped_data:
        control_data = group_df[group_df['Group'] == 'Control']
        treatment_data = group_df[group_df['Group'] == 'Treatment']

        if not control_data.empty and not treatment_data.empty:
            control_mean = control_data['Mean'].iloc[0]
            control_sd = control_data['Std_Dev'].iloc[0]
            # Assumindo N=10 se não houver coluna N ou ela estiver vazia, ou se houver 'N', usar o valor.
            # No R, n() é o número de observações no grupo. Se 'N' não existe ou é NaN, 
            # estamos usando um valor padrão de 10.
            control_n = control_data['N'].iloc[0] if 'N' in control_data.columns and pd.notna(control_data['N'].iloc[0]) else 10

            for index, row in treatment_data.iterrows():
                treatment_mean = row['Mean']
                treatment_sd = row['Std_Dev']
                treatment_n = row['N'] if 'N' in treatment_data.columns and pd.notna(row['N']) else 10

                # Lidar com Mean == 0 para evitar log(0) ou divisão por zero
                if control_mean == 0:
                    control_mean = 0.001
                if treatment_mean == 0:
                    treatment_mean = 0.001

                lnRR = np.log(treatment_mean / control_mean)
                # A fórmula para var_lnRR no R usa n() (que é o número de observações).
                # Aqui, n() seria o número de réplicas para aquela média específica.
                # Se 'N' representa isso, usaremos 'N'. Se não, o 10 padrão.
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
        # else: # Se quiser depurar grupos que não formam pares, descomente estas linhas
        #     if control_data.empty:
        #         print(f"Warning: No control data for Study: {study}, Variable: {variable}, Residue_Type: {residue_type}")
        #     if treatment_data.empty:
        #         print(f"Warning: No treatment data for Study: {study}, Variable: {variable}, Residue_Type: {residue_type}")

    dados_meta = pd.DataFrame(dados_meta)
    dados_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    dados_meta.dropna(subset=['lnRR', 'var_lnRR'], inplace=True)
    print(f"Final data for meta-analysis after lnRR/var_lnRR calculation and NaN removal: {len(dados_meta)} rows.") # Log

    return dados_meta


# --- Funções de Análise e Plotagem (Sem alterações significativas aqui para esta depuração) ---

def run_meta_analysis_and_plot(data, model_type="Residue"):
    """
    Executa o modelo de meta-análise e gera um gráfico de coeficientes.
    Utiliza WLS como uma aproximação para modelos de meta-análise de efeitos fixos/mistos
    pelo método de regressão. Para heterogeneidade, seriam necessários cálculos adicionais.
    """
    if data.empty:
        print(f"Cannot run meta-analysis for {model_type}: input data is empty.")
        return pd.DataFrame(), None

    data = sm.add_constant(data, has_constant='add')

    model = None
    if model_type == "Residue":
        formula = 'lnRR ~ C(Residue_Type)'
        if len(data['Residue_Type'].unique()) > 1:
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else:
            print(f"Warning: Only one residue type ({data['Residue_Type'].unique()[0]}) found, cannot run regression by Residue Type.")
            return pd.DataFrame(), None
            
    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable)'
        if len(data['Variable'].unique()) > 1:
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else:
            print(f"Warning: Only one variable ({data['Variable'].unique()[0]}) found, cannot run regression by Variable.")
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
    # Para o gráfico de coeficientes, precisamos dos termos reais, não do intercepto 'const'
    # ou os termos 'C(Residue_Type)[T.<categoria>]'
    plot_data = summary_df[~summary_df['term'].str.contains('Intercept|C\(', na=False)]
    
    # Se, por algum motivo, não houver termos além do intercepto (e C()), ainda podemos plotar o 'const'
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
