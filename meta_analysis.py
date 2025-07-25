import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM # Import para REML
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suprime avisos para uma saída mais limpa
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """
    Carrega dados de um arquivo CSV e realiza a limpeza e preparação inicial.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        pandas.DataFrame: O DataFrame preparado.
    """
    try:
        dados = pd.read_csv(file_path, sep=';', decimal='.')
    except FileNotFoundError:
        return pd.DataFrame() # Retorna vazio se o arquivo não for encontrado
    except Exception as e:
        print(f"Erro ao carregar ou ler o arquivo CSV: {e}")
        return pd.DataFrame()

    # Renomear colunas para evitar problemas com espaços
    dados = dados.rename(columns={'Std Dev': 'Std_Dev', 'Original Unit': 'Original_Unit'})

    # Converter colunas numéricas, tratando possíveis erros
    numeric_cols = ['Mean', 'Std_Dev']
    for col in numeric_cols:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados = dados.dropna(subset=numeric_cols) # Remove linhas onde a conversão falhou

    return dados

def filter_irrelevant_treatments(dados):
    """
    Filtra tratamentos irrelevantes do conjunto de dados.
    Args:
        dados (pandas.DataFrame): O DataFrame de entrada.
    Returns:
        pandas.DataFrame: O DataFrame com tratamentos irrelevantes filtrados.
    """
    # Lista de tratamentos que NÃO são vermicomposto final
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
    Define os grupos 'Control' vs. 'Treatment' e os tipos de 'Residue'.
    Args:
        dados_filtrados (pandas.DataFrame): O DataFrame filtrado.
    Returns:
        pandas.DataFrame: O DataFrame com as colunas 'Group' e 'Residue'.
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
    
    # Adicionar a regra do abacaxi para "Resíduos Agrícolas"
    dados_grupos.loc[dados_grupos['Treatment'].str.contains('abacaxi', na=False, case=False), 'Residue'] = 'Resíduos Agrícolas'

    return dados_grupos

def prepare_for_meta_analysis(dados_grupos):
    """
    Prepara os dados para meta-análise, calculando o Log Response Ratio (lnRR) e sua variância.
    Args:
        dados_grupos (pandas.DataFrame): O DataFrame com grupos e resíduos definidos.
    Returns:
        pandas.DataFrame: O DataFrame pronto para meta-análise.
    """
    # Identifica variáveis que possuem grupo controle
    variables_with_control = dados_grupos[dados_grupos['Group'] == "Control"]['Variable'].unique()

    dados_meta = dados_grupos[dados_grupos['Variable'].isin(variables_with_control)].copy()

    # Calcula as médias e desvios padrão do controle
    control_data = dados_meta[dados_meta['Group'] == "Control"].groupby('Variable').agg(
        Mean_control=('Mean', 'first'),
        Std_Dev_control=('Std_Dev', 'first')
    ).reset_index()
    
    # Lida com casos onde Std_Dev_control pode ser zero
    control_data['Std_Dev_control'] = control_data['Std_Dev_control'].replace(0, 0.001)

    dados_meta = dados_meta[dados_meta['Group'] == "Treatment"].merge(control_data, on='Variable', how='left')

    # Ajusta Std_Dev para o grupo de tratamento para evitar divisão por zero
    dados_meta['Std_Dev_adj'] = dados_meta['Std_Dev'].replace(0, 0.001)

    # Calcula o Log Response Ratio (lnRR) e sua variância
    # Garante que Mean_control e Mean não sejam zero antes do log
    dados_meta = dados_meta[
        (dados_meta['Mean_control'] > 0) & (dados_meta['Mean'] > 0)
    ].copy()

    dados_meta['lnRR'] = np.log(dados_meta['Mean'] / dados_meta['Mean_control'])
    
    # Cálculo da variância do lnRR:
    # (Std_Dev_treatment^2 / (N_treatment * Mean_treatment^2)) + (Std_Dev_control^2 / (N_control * Mean_control^2))
    # Assumindo N_treatment e N_control como 1 para estudos individuais,
    # se você tiver tamanhos de amostra reais (n) nos seus dados, use-os em vez de 1.
    dados_meta['var_lnRR'] = (dados_meta['Std_Dev_adj']**2 / (1 * dados_meta['Mean']**2)) + \
                             (dados_meta['Std_Dev_control']**2 / (1 * dados_meta['Mean_control']**2))
    
    # Filtra linhas com valores NaN ou infinitos em lnRR ou var_lnRR
    dados_meta = dados_meta.replace([np.inf, -np.inf], np.nan).dropna(subset=['lnRR', 'var_lnRR'])
    
    return dados_meta

def run_meta_analysis(dados_meta, model_type="Residue"):
    """
    Executa a meta-análise usando MixedLM para aproximar a estimação REML, semelhante ao metafor.

    Args:
        dados_meta (pandas.DataFrame): Dados preparados para meta-análise.
        model_type (str): Tipo de modelo a ser executado ("Residue", "Variable", ou "Interaction").

    Returns:
        dict: Um dicionário contendo os resultados do modelo (beta, se, pval, ci, tau2, I2, etc.).
    """
    if dados_meta.empty or len(dados_meta) < 2:
        return {} # Retorna um dicionário vazio se não houver dados suficientes para a análise

    formula = 'lnRR ~ '
    groups_col = 'Study' # Variável de agrupamento para efeitos aleatórios (estudos)

    if model_type == "Residue":
        if len(dados_meta['Residue'].unique()) < 2: return {}
        formula += 'C(Residue) - 1'
    elif model_type == "Variable":
        if len(dados_meta['Variable'].unique()) < 2: return {}
        formula += 'C(Variable) - 1'
    elif model_type == "Interaction":
        # Precisa garantir que há combinações únicas suficientes para a interação
        if dados_meta[['Residue', 'Variable']].drop_duplicates().shape[0] < 2: return {}
        formula += 'C(Residue):C(Variable) - 1'
    else:
        raise ValueError("model_type inválido. Escolha 'Residue', 'Variable' ou 'Interaction'.")

    # Garante que a coluna 'Study' existe para agrupamento, ou cria um ID temporário
    if groups_col not in dados_meta.columns:
        dados_meta['Study_ID'] = dados_meta.index.astype(str)
        groups_col = 'Study_ID'
        warnings.warn(f"Coluna '{groups_col}' não encontrada, usando índice da linha como variável de agrupamento para efeitos aleatórios MixedLM.")
    
    # Calcula os pesos (inverso da variância do efeito)
    dados_meta['weights_mlm'] = 1 / dados_meta['var_lnRR']
    dados_meta['weights_mlm'] = dados_meta['weights_mlm'].replace([np.inf, -np.inf], np.nan).fillna(0) # Zera NaNs/Inf

    try:
        model = MixedLM.from_formula(
            formula,
            data=dados_meta,
            groups=dados_meta[groups_col],
            re_formula="1", # Intercepto aleatório para cada estudo
            # vc_formula é opcional e pode ser usado para especificar a estrutura da variância do efeito aleatório
            # No metafor, a variância entre estudos (tau^2) é estimada. MixedLM faz isso com re_formula="1".
            # vc_formula={"study_var": "0 + C(" + groups_col + ")"} # Uma forma de explicitar a variação por grupo
        )
        result = model.fit(reml=True)
    except Exception as e:
        print(f"Erro ao ajustar o modelo MixedLM para {model_type}: {e}")
        warnings.warn(f"MixedLM falhou para {model_type}, tentando WLS como fallback. Erro: {e}")
        try:
            # Fallback para Weighted Least Squares (WLS) se MixedLM falhar
            dados_meta['weights_wls'] = 1 / dados_meta['var_lnRR']
            dados_meta['weights_wls'] = dados_meta['weights_wls'].replace([np.inf, -np.inf], np.nan).fillna(0)
            model_wls = smf.wls(formula, data=dados_meta, weights=dados_meta['weights_wls']).fit()
            
            summary_df = model_wls.summary2().tables[1]
            summary_df = summary_df.reset_index().rename(columns={'index': 'term'})
            summary_df['lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err.']
            summary_df['upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err.']

            return {
                'beta': model_wls.params,
                'se': model_wls.bse,
                'zval': model_wls.tvalues,
                'pval': model_wls.pvalues,
                'ci.lb': summary_df['lower'], 
                'ci.ub': summary_df['upper'],
                'tau2': np.nan, # Não estimado por WLS
                'I2': np.nan,   # Não estimado por WLS
                'QE': np.nan,   # Não direto de WLS
                'QE_pval': np.nan,
                'model': model_wls,
                'model_type': 'WLS_FALLBACK'
            }
        except Exception as e_wls:
            print(f"O fallback para WLS também falhou: {e_wls}")
            return {}

    # Extrai as principais estatísticas do resultado do MixedLM
    tau2 = result.cov_re.iloc[0, 0] if not result.cov_re.empty else 0.0
    
    # Calcula I^2: 100 * tau^2 / (tau^2 + variância de amostragem média)
    avg_sampling_var = dados_meta['var_lnRR'].mean()
    I2 = 100 * (tau2 / (tau2 + avg_sampling_var)) if (tau2 + avg_sampling_var) > 0 else 0.0

    # Para QE (teste de heterogeneidade residual), MixedLM não fornece diretamente como o metafor.
    # O valor de QE e seu p-valor abaixo são placeholders ou aproximações grosseiras
    # para manter a estrutura de saída consistente, mas não são equivalentes exatos ao metafor.
    QE_val = np.nan # Não diretamente disponível
    QE_pval = np.nan # Não diretamente disponível

    # Prepara a saída correspondendo à estrutura de resumo do metafor em R
    output = {
        'beta': result.fe_params,
        'se': result.bse,
        'zval': result.tvalues,
        'pval': result.pvalues,
        'ci.lb': result.conf_int()[0], # CI Inferior
        'ci.ub': result.conf_int()[1], # CI Superior
        'tau2': tau2,
        'I2': I2,
        'QE': QE_val,
        'QE_pval': QE_pval,
        'model': result,
        'model_type': 'MixedLM_REML'
    }
    
    return output

def generate_forest_plot(dados_meta, model_results, title="Forest Plot"):
    """
    Gera um Forest Plot similar ao metafor, mostrando os efeitos de estudos individuais
    e a estimativa do efeito geral do modelo de meta-análise.

    Args:
        dados_meta (pandas.DataFrame): Dados preparados para meta-análise (estudos individuais).
        model_results (dict): Resultados de run_meta_analysis, contendo o efeito geral.
        title (str): Título do forest plot.

    Returns:
        matplotlib.figure.Figure: A figura matplotlib para o forest plot.
    """
    if dados_meta.empty or not model_results: # Verifica se model_results não está vazio
        return None

    # Ordena os dados para melhor visualização
    dados_meta = dados_meta.sort_values(by='lnRR', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(dados_meta) * 0.4 + 2))) # Altura dinâmica

    # Determina os limites do eixo x com base nos dados e no efeito geral
    all_effects = dados_meta['lnRR'].tolist()
    # Adiciona os coeficientes do modelo se existirem para influenciar os limites
    if 'beta' in model_results and not model_results['beta'].empty:
        all_effects.extend(model_results['beta'].tolist())

    min_x = min(all_effects) - 0.5
    max_x = max(all_effects) + 0.5
    
    # Plotando efeitos de estudos individuais
    for i, row in dados_meta.iterrows():
        label = f"{row['Study']} - {row['Residue']} - {row['Variable']}"
        effect = row['lnRR']
        # Usa variância para o CI do estudo individual
        se_individual = np.sqrt(row['var_lnRR'])
        ci_lower_individual = effect - 1.96 * se_individual
        ci_upper_individual = effect + 1.96 * se_individual
        
        ax.plot([ci_lower_individual, ci_upper_individual], [i, i], color='gray', linestyle='-', linewidth=1.5, solid_capstyle='butt')
        ax.plot(effect, i, 's', color='blue', markersize=6, zorder=3) # Marcador quadrado
        
        # Adiciona texto para tamanho do efeito e CI no lado direito
        ax.text(max_x + 0.1, i, f"{effect:.2f} [{ci_lower_individual:.2f}, {ci_upper_individual:.2f}]", va='center', ha='left', fontsize=8)
        # Adiciona rótulo do estudo no lado esquerdo
        ax.text(min_x - 0.1, i, label, va='center', ha='right', fontsize=8)

    # Adiciona o efeito geral (dos resultados do modelo)
    if 'beta' in model_results and not model_results['beta'].empty:
        pooled_effect = model_results['beta'].mean()
        ax.axvline(pooled_effect, color='purple', linestyle='-', linewidth=1, label='Estimativa Média do Modelo (Média Coef)')

    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, label='Sem Efeito (lnRR=0)') # Linha de nenhum efeito

    ax.set_yticks([]) # Oculta os rótulos do eixo y já que estamos usando texto para os rótulos
    ax.set_yticklabels([])
    ax.set_xlabel("Log Response Ratio (lnRR) [95% CI]")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    ax.set_xlim(min_x - 0.2, max_x + 0.2) # Ajusta os limites do eixo x dinamicamente
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def generate_funnel_plot(dados_meta):
    """
    Gera um Funnel Plot para viés de publicação, usando lnRR e Erro Padrão.
    Args:
        dados_meta (pandas.DataFrame): Dados preparados para meta-análise.
    Returns:
        matplotlib.figure.Figure: A figura matplotlib para o funnel plot.
    """
    if dados_meta.empty:
        return None

    # Calcula o erro padrão se ainda não estiver presente
    dados_meta['se_lnRR'] = np.sqrt(dados_meta['var_lnRR'])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plota estudos individuais
    ax.scatter(dados_meta['lnRR'], dados_meta['se_lnRR'], color='blue', alpha=0.7)

    # Adiciona limites de pseudo-confiança (assumindo nenhum efeito, lnRR=0)
    max_se = dados_meta['se_lnRR'].max()
    # Define um intervalo para o eixo x que cobre tamanhos de efeito típicos mais alguma margem
    x_range = np.linspace(-3 * max_se, 3 * max_se, 100)

    # Linhas de CI de 95% (assumindo um efeito fixo de 0)
    ax.plot(x_range, np.abs(x_range) / 1.96, color='grey', linestyle='--', label='Pseudo 95% CI')
    ax.plot(x_range, -np.abs(x_range) / 1.96, color='grey', linestyle='--')

    ax.axvline(0, color='red', linestyle=':', label='No Effect (lnRR=0)') # Linha de nenhum efeito

    ax.set_xlabel("Log Response Ratio (lnRR)")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot for Publication Bias")
    ax.invert_yaxis() # Padrão para funnel plots (SE maior no topo)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig
