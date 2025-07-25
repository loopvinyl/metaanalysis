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
    Carrega o arquivo CSV e realiza a preparação inicial dos dados.
    Assume que o CSV usa ';' como separador e '.' como decimal.
    """
    try:
        dados = pd.read_csv(file_path, sep=';', decimal='.')
        # Renomear colunas para consistência e facilitar o acesso
        dados = dados.rename(columns={
            'Mean': 'Mean',
            'Std Dev': 'Std_Dev',
            'Treatments': 'Treatment',
            'Residues': 'Residue',
            'Variables': 'Variable'
        })
        # Converter colunas numéricas, forçando erros para NaN
        dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
        dados['Std_Dev'] = pd.to_numeric(dados['Std_Dev'], errors='coerce')

        # Substituir 0 em Std_Dev por um valor pequeno para evitar divisão por zero
        dados['Std_Dev'] = dados['Std_Dev'].replace(0, 0.001)

        # Remover linhas com valores NaN nas colunas críticas após a conversão
        dados.dropna(subset=['Mean', 'Std_Dev', 'Treatment', 'Variable', 'Residue'], inplace=True)
        return dados
    except Exception as e:
        st.error(f"Error loading or initially preparing data: {e}")
        return pd.DataFrame()

def filter_irrelevant_treatments(df):
    """
    Filtra tratamentos irrelevantes para a meta-análise de vermicompostagem.
    """
    # Lista de tratamentos a serem excluídos (não são vermicompostos finais)
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
        'Initial Agro-waste', 'Initial Slurry'
    ]
    df_filtered = df[~df['Treatment'].isin(tratamentos_excluir)].copy()
    return df_filtered

def define_groups_and_residues(df):
    """
    Define grupos de controle e tratamento e atribui categorias de resíduos.
    """
    # Garantir que a coluna 'Treatment' é string para operações de texto
    df['Treatment'] = df['Treatment'].astype(str)

    # Classificar como Controle ou Tratamento
    # Assumimos que 'Control' é explicitamente mencionado ou inferido.
    # Se 'Control' não for o nome exato, ajuste esta lógica.
    df['Group'] = np.where(df['Treatment'].str.contains('Control', case=False, na=False), 'Control', 'Treatment')

    # Atribuir tipo de resíduo
    def assign_residue_type(row):
        residue = str(row['Residue']).lower()
        if 'sewage' in residue or 'sludge' in residue or 'municipal' in residue or 'waste' in residue or 'industrial' in residue or 'brewery' in residue or 'distillery' in residue:
            return 'Sewage Sludge/Industrial Waste'
        elif 'grape' in residue or 'fruit' in residue or 'vegetable' in residue or 'pineapple' in residue or 'coffee' in residue or 'sugarcane' in residue or 'paddy' in residue or 'rice' in residue or 'corn' in residue or 'leaf' in residue or 'agro-waste' in residue or 'livestock' in residue or 'manure' in residue or 'straw' in residue: # Added 'pineapple' here
            return 'Agricultural Residue'
        elif 'paper' in residue or 'wood' in residue or 'lignin' in residue:
            return 'Paper/Wood Waste'
        else:
            return 'Other' # Default category if not matched

    df['Residue_Type'] = df.apply(assign_residue_type, axis=1)

    return df

def prepare_for_meta_analysis(df_groups):
    """
    Prepara os dados para meta-análise, calculando lnRR e var_lnRR.
    Esta função foi adaptada para replicar a lógica de efeitos aleatórios simples,
    onde cada par controle-tratamento contribui para o cálculo.
    """
    dados_meta = []

    # Iterar sobre cada estudo e variável
    # Agrupamos por 'Study', 'Variable' e 'Residue_Type' para garantir pares únicos
    for (study, variable, residue_type), group_df in df_groups.groupby(['Study', 'Variable', 'Residue_Type']):
        control_data = group_df[group_df['Group'] == 'Control']
        treatment_data = group_df[group_df['Group'] == 'Treatment']

        if not control_data.empty and not treatment_data.empty:
            # Para cada variável e estudo, vamos parear o controle com todos os tratamentos
            # Simplificação: Usar o primeiro controle encontrado para cada variável/estudo.
            # Em meta-análise complexa, pode-se usar média de controles ou abordar múltiplas comparações.
            control_mean = control_data['Mean'].iloc[0]
            control_sd = control_data['Std_Dev'].iloc[0]
            control_n = control_data['N'].iloc[0] if 'N' in control_data.columns else 10 # Default N if not available

            for index, row in treatment_data.iterrows():
                treatment_mean = row['Mean']
                treatment_sd = row['Std_Dev']
                treatment_n = row['N'] if 'N' in treatment_data.columns else 10 # Default N if not available

                # Evitar divisões por zero ou log(0)
                if control_mean == 0:
                    control_mean = 0.001
                if treatment_mean == 0:
                    treatment_mean = 0.001

                # Calcular Log Response Ratio (lnRR)
                lnRR = np.log(treatment_mean / control_mean)

                # Calcular Variância do lnRR
                # Formula para variância do lnRR (Hedges et al., 1999)
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
    
    # Remover linhas onde lnRR ou var_lnRR são NaN ou infinitos após o cálculo
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

    # Adiciona uma constante para o intercepto
    data = sm.add_constant(data, has_constant='add')

    model = None
    if model_type == "Residue":
        # Removendo 'Other' para evitar multicolinearidade se for a única categoria base
        # Ou garantindo que a referência seja tratada implicitamente pelo statsmodels
        formula = 'lnRR ~ C(Residue_Type)'
        # Para garantir que 'Other' não cause problemas se houver apenas 2 tipos
        if 'Other' in data['Residue_Type'].unique() and len(data['Residue_Type'].unique()) > 1:
            # Categórica com tratamento de referência (primeira categoria alfabética por padrão ou outra)
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
        else: # Se 'Other' não existe ou é a única, apenas C(Residue_Type) funciona
            model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])

    elif model_type == "Variable":
        formula = 'lnRR ~ C(Variable)'
        model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])

    elif model_type == "Interaction":
        # Interação entre tipo de resíduo e variável
        # Pode ser complexo e exigir dados suficientes para cada combinação
        formula = 'lnRR ~ C(Residue_Type) * C(Variable)'
        model = sm.WLS.from_formula(formula, data=data, weights=1/data['var_lnRR'])
    else:
        st.error("Invalid model type specified.")
        return pd.DataFrame(), None

    try:
        results = model.fit()
    except Exception as e:
        st.error(f"Error fitting the model: {e}. Not enough data for this combination of variables/residues.")
        return pd.DataFrame(), None

    # Criar DataFrame de resumo para exibição
    summary_df = results.summary2().tables[1] # Tabela de coeficientes
    summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
    summary_df = summary_df[['term', 'Coef.', 'Std.Err.', 't', 'P>|t|', '[0.025', '0.975]']]
    summary_df.columns = ['term', 'estimate', 'std_error', 't_value', 'p_value', 'conf_low', 'conf_high']

    # Plotagem dos coeficientes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Linha vertical no zero (ponto de nenhum efeito)
    ax.axvline(x=0, linestyle="dashed", color="red") # CORREÇÃO APLICADA AQUI

    # Plotar coeficientes (excluindo o intercepto se for um modelo categórico)
    # Filtra o intercepto se for um modelo que usa C() para categóricas
    plot_data = summary_df[~summary_df['term'].str.contains('Intercept|C\(', na=False)] # Filtra o intercepto e os termos C()

    # Reordenar para melhor visualização
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
        st.warning("Data is insufficient for Forest Plot. Make sure 'lnRR' and 'var_lnRR' columns exist.")
        return None

    # Calculate standard error from variance
    data['se_lnRR'] = np.sqrt(data['var_lnRR'])
    data['lower_ci'] = data['lnRR'] - 1.96 * data['se_lnRR']
    data['upper_ci'] = data['lnRR'] + 1.96 * data['se_lnRR']

    fig, ax = plt.subplots(figsize=(10, len(data) * 0.4 + 2)) # Dynamic height

    # Sort data for better visualization
    data_sorted = data.sort_values(by='lnRR', ascending=False)

    # Plot individual studies
    y_pos = np.arange(len(data_sorted))
    ax.errorbar(data_sorted['lnRR'], y_pos, xerr=1.96 * data_sorted['se_lnRR'],
                fmt='o', markersize=5, capsize=5, color='blue', alpha=0.7)

    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data_sorted['Study'] + ' - ' + data_sorted['Variable']) # Combine study and variable for labels
    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_title("Forest Plot of Individual Studies")

    # Add a vertical line at 0 (no effect)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)

    # Set limits
    min_x = data_sorted['lower_ci'].min() * 1.1
    max_x = data_sorted['upper_ci'].max() * 1.1
    ax.set_xlim(min(min_x, -1), max(max_x, 1)) # Ensure 0 is visible

    plt.tight_layout()
    return fig

def generate_funnel_plot(data):
    """
    Gera um Funnel Plot básico para avaliar viés de publicação.
    """
    if data.empty or 'lnRR' not in data.columns or 'var_lnRR' not in data.columns:
        st.warning("Data is insufficient for Funnel Plot. Make sure 'lnRR' and 'var_lnRR' columns exist.")
        return None

    # Precisão (precision) é o inverso do erro padrão
    data['precision'] = 1 / np.sqrt(data['var_lnRR'])

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data['lnRR'], data['precision'], alpha=0.7)

    # Adicionar linha de efeito médio (assumindo 0 para plot básico, ou o lnRR médio se calculado)
    # Aqui, vamos traçar uma linha vertical no zero.
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Precision (1/Standard Error)")
    ax.set_title("Funnel Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig
