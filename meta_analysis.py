import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import os
import io # Importado para uso potencial, mas não diretamente usado com o file_path atual

# --- Funções de Pré-processamento e Cálculo ---

def load_and_prepare_data():
    """
    Carrega o arquivo de dados CSV (esperado no mesmo diretório do app.py)
    e realiza a preparação inicial.
    Retorna um DataFrame vazio em caso de erro de leitura ou processamento.
    """
    dados = pd.DataFrame()
    
    # O arquivo CSV é esperado no MESMO diretório que o script principal (app.py)
    csv_file_path = "csv.csv" 
    
    try:
        # Usar read_csv com delimitador e decimal como no script R
        dados = pd.read_csv(csv_file_path, delimiter=';', decimal='.')
        print(f"Successfully loaded data from: {csv_file_path}")
        print(f"Initial data loaded: {len(dados)} rows.") # Log
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please ensure it is in the same directory as app.py.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV file '{csv_file_path}': {e}")
        return pd.DataFrame()

    try:
        # Renomear colunas para consistência e facilitar o acesso
        # Os nomes das colunas são os do seu CSV, com ajuste para Std_Dev e Original_Unit
        dados = dados.rename(columns={
            'Variable': 'Variable',
            'Study': 'Study',
            'Treatment': 'Treatment',
            'Mean': 'Mean',
            'Std Dev': 'Std_Dev', # Nome da coluna no CSV com espaço
            'Unit': 'Unit',
            'Original Unit': 'Original_Unit', # Nome da coluna no CSV com espaço
            'Notes': 'Notes'
        })
        
        # Converter colunas numéricas, forçando erros para NaN
        dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
        dados['Std_Dev'] = pd.to_numeric(dados['Std_Dev'], errors='coerce')

        # Substituir 0 em Std_Dev por um valor pequeno para evitar divisão por zero
        # Um desvio padrão de 0 causa problemas em cálculos de metanálise.
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

def calculate_lnrr(df):
    """
    Calcula o Log Response Ratio (lnRR) para cada variável.
    """
    lnrr_data = []
    
    # Identificar tratamentos e controles dentro de cada estudo e variável
    for (study, variable), group in df.groupby(['Study', 'Variable']):
        # Tentar identificar o controle pelo nome mais comum
        control_row = group[group['Treatment'].str.contains('control|cont|t0|m0|ch0|r0', case=False, na=False)]
        
        # Se não encontrar um controle claro, pode ser necessário ajustar a lógica
        if control_row.empty:
            # Fallback: tentar usar o tratamento com a menor média como um proxy de "linha de base"
            # Esta é uma suposição e pode precisar ser refinada com mais conhecimento do dataset.
            control_row = group.loc[group['Mean'].idxmin()]
            print(f"Warning: No explicit 'control' found for {variable} in {study}. Using lowest mean as control proxy: {control_row['Treatment']}")
            
        if not control_row.empty:
            control_mean = control_row['Mean'].iloc[0]
            control_std = control_row['Std_Dev'].iloc[0]
            
            # Ajuste para evitar divisão por zero ou log de zero para control_mean
            if control_mean <= 0:
                print(f"Skipping {variable} in {study} due to non-positive control mean.")
                continue

            for _, row in group.iterrows():
                if row['Treatment'] not in control_row['Treatment'].values: # Evitar comparar o controle com ele mesmo
                    treatment_mean = row['Mean']
                    treatment_std = row['Std_Dev']

                    # Adicionar um pequeno epsilon para evitar log(0) ou divisão por 0 se treatment_mean for 0
                    if treatment_mean <= 0:
                        treatment_mean = 0.001 # Ou outro valor razoável baseado na escala dos seus dados

                    lnrr = np.log(treatment_mean / control_mean)
                    
                    # Calcular a variância do lnRR
                    # Formula: V_lnRR = (Std_Dev_treatment^2 / (N_treatment * Mean_treatment^2)) + (Std_Dev_control^2 / (N_control * Mean_control^2))
                    # Assumindo N=1 para fins de cálculo de SEM. Para meta-análise robusta, N é crucial.
                    # Se N não estiver disponível, o cálculo do erro padrão pode ser um desafio.
                    # Para simplificar aqui, vamos usar a propagação de erro básico com as SEMs
                    
                    # Convertendo Std_Dev para SEM para usar na fórmula da variância (se Std_Dev já for de fato SEM)
                    # Se Std_Dev for o desvio padrão da amostra, a variância do lnRR é (SD_t^2/Mt^2) + (SD_c^2/Mc^2)
                    # Se for o erro padrão da média, a fórmula é mais complexa ou exige N.
                    # Para fins de demonstração, assumimos que Std_Dev é o desvio padrão e N=1 (um estudo por "linha")
                    # Esta parte é uma simplificação e é o ponto mais crítico para a precisão da metanálise.
                    
                    # Vamos assumir que 'Std_Dev' é o desvio padrão da média (SEM) ou foi calculado com N já embutido.
                    # A fórmula mais comum para a variância do LRR (com base em desvios padrão da amostra):
                    # var_lrr = (std_t**2 / (n_t * mean_t**2)) + (std_c**2 / (n_c * mean_c**2))
                    # Como não temos N, se Std_Dev é o desvio padrão da amostra, precisamos de N ou usar SEM diretamente.
                    # Dada a natureza dos seus dados, vamos assumir que Std_Dev é um indicador de erro já "escalado" para a média.
                    
                    # Simplificação para o desvio padrão do lnRR, sem N explícito
                    # Isso é uma estimativa muito rudimentar sem o N dos estudos.
                    # Para uma meta-análise real, você precisa do N (número de repetições/amostras).
                    # A ausência de N é um problema comum em dados extraídos de artigos.
                    
                    # Método simplificado (aproximação da variância se Std_Dev for desvio padrão):
                    # var_lnrr = (treatment_std**2 / (treatment_mean**2)) + (control_std**2 / (control_mean**2))
                    
                    # Se Std_Dev é o SEM, então var(mean) = SEM^2.
                    # A fórmula do lnRR exige a variância das médias.
                    # Simplificação: Considerando Std_Dev como o erro associado.
                    
                    # Tentativa de cálculo de variância do lnRR (simplificada, mas comum na ausência de N)
                    # Esta fórmula é para quando Mean e Std_Dev são da amostra e N é desconhecido ou assumido como 1.
                    # Idealmente, você teria N para calcular (Std_Dev_sample / sqrt(N))^2
                    
                    # Para fins de funcionamento e com base na prática comum para dados agregados:
                    # Se 'Std_Dev' é de fato o desvio padrão da amostra, e N não está disponível:
                    # var_lnrr = (treatment_std**2 / treatment_mean**2) + (control_std**2 / control_mean**2)
                    
                    # No entanto, se o 'Std_Dev' é o SEM (Standard Error of the Mean), que é desvio padrão / sqrt(N),
                    # então a variância da média é simplesmente SEM^2.
                    # Sem mais informações, o uso direto do Std_Dev como erro é a abordagem mais direta.

                    # Para robustez e evitar erros de divisão por zero se mean for muito pequeno
                    adjusted_treatment_mean = treatment_mean if treatment_mean != 0 else 0.001
                    adjusted_control_mean = control_mean if control_mean != 0 else 0.001

                    var_lnrr = (treatment_std**2 / adjusted_treatment_mean**2) + (control_std**2 / adjusted_control_mean**2)
                    
                    # Certificar-se de que a variância não é zero ou negativa
                    if var_lnrr <= 0:
                        var_lnrr = 1e-6 # Um valor muito pequeno para evitar problemas, se for o caso
                        
                    lnrr_data.append({
                        'Study': study,
                        'Variable': variable,
                        'Treatment': row['Treatment'],
                        'lnRR': lnrr,
                        'Variance_lnRR': var_lnrr
                    })
    
    return pd.DataFrame(lnrr_data)


def meta_analysis_model(df, model_type="overall"):
    """
    Executa um modelo de meta-análise (efeitos aleatórios) e retorna os resultados.
    model_type pode ser "overall", "variable", "residue", ou "interaction".
    """
    if df.empty:
        return None, None

    # Adicionar o intercepto para o modelo
    df['Intercept'] = 1

    # Pesos são o inverso da variância
    df['Weights'] = 1 / df['Variance_lnRR']

    formula = 'lnRR ~ Intercept'
    if model_type == "residue":
        df['Residue'] = df['Treatment'].apply(assign_residue_group) # Re-aplica a classificação de resíduos
        df.dropna(subset=['Residue'], inplace=True) # Remove linhas onde não foi possível classificar
        if df['Residue'].nunique() > 1:
            formula = 'lnRR ~ C(Residue)' # C() para tratar como categórica
        else:
            print("Warning: Only one residue type found for meta-analysis, running overall model.")
            model_type = "overall" # Reverte para o modelo geral se apenas um tipo de resíduo
    elif model_type == "variable":
        if df['Variable'].nunique() > 1:
            formula = 'lnRR ~ C(Variable)'
        else:
            print("Warning: Only one variable type found for meta-analysis, running overall model.")
            model_type = "overall"
    elif model_type == "interaction":
        df['Residue'] = df['Treatment'].apply(assign_residue_group)
        df.dropna(subset=['Residue'], inplace=True)
        if df['Variable'].nunique() > 1 and df['Residue'].nunique() > 1:
            # Modelar a interação entre Resíduo e Variável
            formula = 'lnRR ~ C(Variable) * C(Residue)'
        else:
            print("Warning: Not enough unique variables or residues for interaction model, running overall model.")
            model_type = "overall" # Reverte para o modelo geral se não houver variação suficiente

    # Modelagem com WLS (Weighted Least Squares) como proxy para efeitos aleatórios
    # Em meta-análise real, usaríamos pacotes como `meta` ou `pymeta` para efeitos aleatórios,
    # mas para demonstração com statsmodels, WLS é uma aproximação.
    try:
        if 'Weights' not in df.columns or df['Weights'].isnull().any() or (df['Weights'] == 0).any():
            print("Warning: Invalid weights found, using uniform weights.")
            model = sm.WLS(df['lnRR'], sm.add_constant(df[['Intercept']]), weights=np.ones(len(df))).fit()
        else:
            X = sm.add_constant(df[[col.split('~')[1].strip() for col in formula.split('+')] if '~' in formula else ['Intercept']])
            model = sm.WLS(df['lnRR'], X, weights=df['Weights']).fit()
        
        return model, df # Retorna o modelo e o DataFrame com a coluna 'Residue' se adicionada
    except Exception as e:
        print(f"Error running meta-analysis model ({model_type}): {e}")
        return None, None

def assign_residue_group(treatment_name):
    """
    Classifica um nome de tratamento em um grupo de resíduos.
    """
    treatment_name = str(treatment_name).lower() # Garante que seja string e minúsculas
    
    # Classificação baseada nas entradas do seu CSV e conhecimento comum
    if 'manure' in treatment_name or 'cow dung' in treatment_name or 'cowmanure' in treatment_name:
        return 'Manure'
    elif 'sewage sludge' in treatment_name or 'sludge' in treatment_name:
        return 'Sewage Sludge'
    elif 'food waste' in treatment_name or 'food' in treatment_name:
        return 'Food Waste'
    elif 'biomass' in treatment_name: # Genérico, pode ser refinado
        return 'Biomass'
    elif 'bagasse' in treatment_name or 'sugarcane' in treatment_name:
        return 'Sugarcane Bagasse'
    elif 'straw' in treatment_name or 'rice straw' in treatment_name:
        return 'Straw'
    elif 'agro-waste' in treatment_name or 'agricultural waste' in treatment_name:
        return 'Agricultural Waste'
    elif 'industrial waste' in treatment_name:
        return 'Industrial Waste'
    # Exemplos dos seus tratamentos: C1B0, C1B2, CH0, R0, M0, T1
    # 'C' pode ser crop, 'B' bagasse, 'R' resíduo, 'M' material, 'T' tratamento
    # Para tratamentos genéricos, precisamos de uma regra mais inteligente ou mapeamento manual.
    # Ex: 'C1B2' - se C é Crop e B é Bagasse, talvez seja 'Crop/Bagasse Mix'.
    # Como não temos um mapeamento claro de todos os códigos de tratamento, 
    # faremos um chute educado baseado nos estudos ou marcamos como "Mixed/Other".
    
    # Adicionando classificações para os padrões do seu CSV
    if any(x in treatment_name for x in ['c1', 'c2', 'c3']): # Assumindo 'C' de 'Crop' ou similar
        return 'Agricultural Residue'
    elif any(x in treatment_name for x in ['ch0', 'ch25', 'ch50', 'ch75', 'ch100']): # Quadar et al. (2022) - parece mistura com esterco
        return 'Manure/Compost Mix' # Ou outro nome mais específico se soubermos a base 'CH'
    elif any(x in treatment_name for x in ['r0', 'r1', 'r2', 'r3', 'r4']): # Srivastava et al. (2021) - pode ser geral 'Residues'
        return 'Mixed Residues' # Nome genérico se os tipos de R não forem especificados
    elif any(x in treatment_name for x in ['m0', 'm1', 'm2', 'm3', 'm4']): # Santana et al. (2018) - pode ser geral 'Manure'
        return 'Manure'
    elif any(x in treatment_name for x in ['t1', 't2', 't3', 't4']): # Suthar et al. (2018) ou Arora et al. (2020)
        return 'Mixed Residues' # Geral
    
    # Para 'abacaxi' que você mencionou:
    elif 'abacaxi' in treatment_name or 'pineapple' in treatment_name:
        return 'Agricultural Residue' # Ou 'Fruticultural Waste' se quiser mais específico
        
    return 'Other/Unspecified'


def plot_meta_analysis_results(model, df_for_plot, title_prefix="Meta-Analysis Results"):
    """
    Gera um gráfico de barras com os resultados da meta-análise (lnRR e IC).
    """
    if model is None:
        return None

    try:
        results_df = pd.DataFrame({
            'Coefficient': model.params.index,
            'lnRR_Estimate': model.params.values,
            'Std_Err': model.bse.values
        })
        
        # Calcular IC 95%
        results_df['Lower_CI'] = results_df['lnRR_Estimate'] - 1.96 * results_df['Std_Err']
        results_df['Upper_CI'] = results_df['lnRR_Estimate'] + 1.96 * results_df['Std_Err']

        # Excluir o intercepto para gráficos de fatores se houver outros termos
        if 'Intercept' in results_df['Coefficient'].values and results_df['Coefficient'].nunique() > 1:
            results_df = results_df[results_df['Coefficient'] != 'Intercept']
            
        # Limpar nomes dos coeficientes (remover "C(...)")
        results_df['Coefficient'] = results_df['Coefficient'].str.replace('C\(', '', regex=True).str.replace('\[T\.([^\]]+)\]', r'\1', regex=True).str.replace('\:C\(', ' x ', regex=True).str.replace('\]', '', regex=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar as estimativas
        ax.barh(results_df['Coefficient'], results_df['lnRR_Estimate'], xerr=results_df['Std_Err']*1.96, capsize=5)
        
        # Adicionar linha de referência em lnRR = 0 (sem efeito)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        
        ax.set_xlabel("Log Response Ratio (lnRR) ± 95% CI")
        ax.set_ylabel("Group")
        ax.set_title(f"{title_prefix} (Random Effects Model Approximation)")
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None


def run_meta_analysis_and_plot(dados_preparados, model_type="overall"):
    """
    Orquestra o cálculo do lnRR, a execução do modelo e a geração do gráfico.
    """
    if dados_preparados.empty:
        print("No data available for meta-analysis.")
        return pd.DataFrame(), None

    lnrr_df = calculate_lnrr(dados_preparados.copy()) # Passa uma cópia para evitar modificação do original

    if lnrr_df.empty:
        print("No lnRR data generated. Check data preparation and control identification.")
        return pd.DataFrame(), None

    # Exibe as primeiras linhas do lnRR_df para depuração
    print("Sample of lnRR_df:")
    print(lnrr_df.head())
    print(f"Total lnRR records: {len(lnrr_df)}")

    model, df_with_residues = meta_analysis_model(lnrr_df.copy(), model_type=model_type) # Passa uma cópia

    if model is None:
        print(f"Meta-analysis model for {model_type} could not be run.")
        return pd.DataFrame(), None

    summary_df = pd.DataFrame(model.summary().tables[1].data[1:], columns=model.summary().tables[1].data[0])
    summary_df = summary_df.set_index(summary_df.columns[0]) # Definir a primeira coluna como índice

    # Gerar gráfico
    title_map = {
        "overall": "Overall Effect of Vermicompost",
        "residue": "Effect of Different Residues on Vermicompost Quality",
        "variable": "Effect on Different Quality Variables",
        "interaction": "Interaction Effect of Variable and Residue"
    }
    fig = plot_meta_analysis_results(model, df_with_residues, title_prefix=title_map.get(model_type, "Meta-Analysis Results"))

    return summary_df, fig


def generate_forest_plot(dados_preparados):
    """
    Gera um Forest Plot básico para as estimativas de lnRR por estudo/variável.
    Cada ponto representa um efeito, e a linha é o intervalo de confiança.
    """
    if dados_preparados.empty:
        return None

    lnrr_df = calculate_lnrr(dados_preparados.copy())

    if lnrr_df.empty:
        return None

    # Ordenar para melhor visualização
    lnrr_df['Study_Variable'] = lnrr_df['Study'] + " - " + lnrr_df['Variable'] + " (" + lnrr_df['Treatment'] + ")"
    lnrr_df = lnrr_df.sort_values(by='lnRR', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(lnrr_df) * 0.4))) # Ajusta o tamanho da figura
    
    # Plotar cada estudo/variável como um ponto com barras de erro
    ax.errorbar(lnrr_df['lnRR'], lnrr_df['Study_Variable'], 
                xerr=np.sqrt(lnrr_df['Variance_lnRR']) * 1.96, # Erro padrão * 1.96 para 95% CI
                fmt='o', color='blue', capsize=3, markersize=5, linestyle='None')

    # Linha de nenhum efeito (lnRR = 0)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Log Response Ratio (lnRR) ± 95% CI")
    ax.set_ylabel("Study - Variable (Treatment)")
    ax.set_title("Forest Plot of Individual Treatment Effects")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def generate_funnel_plot(dados_preparados):
    """
    Gera um Funnel Plot para avaliar o viés de publicação.
    """
    if dados_preparados.empty:
        return None

    lnrr_df = calculate_lnrr(dados_preparados.copy())

    if lnrr_df.empty:
        return None

    # Precisamos da precisão (inverso do erro padrão do lnRR)
    lnrr_df['Precision'] = 1 / np.sqrt(lnrr_df['Variance_lnRR'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plotar os pontos
    ax.scatter(lnrr_df['lnRR'], lnrr_df['Precision'], alpha=0.7)

    # Adicionar a linha de 'nenhum efeito' (lnRR = 0)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    # Adicionar linhas de confiança (aproximadas para ±1.96 SEM)
    # Estas linhas formam o "funil" e representam o IC de 95%
    # A largura do funil diminui com o aumento da precisão
    max_precision = lnrr_df['Precision'].max()
    lnrr_range = np.linspace(min(lnrr_df['lnRR']) - 0.1, max(lnrr_df['lnRR']) + 0.1, 100)
    
    # Calcular os limites do funil
    # Limites são dados por: 0 ± 1.96 / Precision
    upper_bound = 1.96 / lnrr_df['Precision']
    lower_bound = -1.96 / lnrr_df['Precision']

    # Para plotar o funil, precisamos de pontos ao longo da precisão
    # Para simplificar, podemos plotar os limites usando a média ponderada do lnRR como centro
    # e os limites de confiança para cada ponto de precisão
    
    # Média ponderada do lnRR
    weighted_mean_lnrr = (lnrr_df['lnRR'] * lnrr_df['Precision']).sum() / lnrr_df['Precision'].sum()
    
    # Criar eixos para o funil
    precision_values = np.linspace(0.01, max_precision * 1.1, 100)
    lower_funnel = weighted_mean_lnrr - (1.96 / precision_values)
    upper_funnel = weighted_mean_lnrr + (1.96 / precision_values)

    ax.plot(lower_funnel, precision_values, color='grey', linestyle=':', alpha=0.7)
    ax.plot(upper_funnel, precision_values, color='grey', linestyle=':', alpha=0.7)
    ax.axvline(weighted_mean_lnrr, color='blue', linestyle='--', linewidth=0.8, label='Weighted Mean lnRR')


    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Precision (1/Standard Error)")
    ax.set_title("Funnel Plot")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig
