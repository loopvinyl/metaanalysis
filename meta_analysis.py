import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.meta_analysis import effectsize_smd, combine_effects
import matplotlib.pyplot as plt

# Função para carregar e preparar os dados
def load_and_prepare_data(file_path):
    try:
        dados = pd.read_csv(file_path, delimiter=';', decimal='.')
        print(f"✅ Dados carregados com sucesso de: {file_path}")
    except Exception as e:
        print(f"❌ Erro ao carregar o CSV: {e}")
        return pd.DataFrame()

    # Verificação das colunas obrigatórias
    colunas_necessarias = ['Variable', 'Study', 'Treatment', 'Mean', 'Std Dev']
    for col in colunas_necessarias:
        if col not in dados.columns:
            print(f"❌ Coluna ausente: {col}")
            return pd.DataFrame()

    # Conversão de colunas numéricas
    dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
    dados['Std Dev'] = pd.to_numeric(dados['Std Dev'], errors='coerce')

    # Remoção de linhas com dados ausentes
    dados.dropna(subset=['Mean', 'Std Dev'], inplace=True)

    return dados


# Função para executar a meta-análise
def rodar_meta_analise(dados, variavel, tratamento, modelo='random'):
    df = dados[dados['Variable'] == variavel].copy()

    # Seleciona grupos controle e tratamento
    tratamento_df = df[df['Treatment'] == tratamento]
    controle_df = df[df['Treatment'] != tratamento]

    # Verifica se há pares suficientes
    estudos = sorted(set(tratamento_df['Study']) & set(controle_df['Study']))
    if len(estudos) < 2:
        return None, "Número insuficiente de estudos comparáveis."

    efeitos = []
    erros = []

    for estudo in estudos:
        t = tratamento_df[tratamento_df['Study'] == estudo]
        c = controle_df[controle_df['Study'] == estudo]

        if t.empty or c.empty:
            continue

        m1, sd1 = t['Mean'].values[0], t['Std Dev'].values[0]
        m2, sd2 = c['Mean'].values[0], c['Std Dev'].values[0]

        # Usa N=10 como aproximação para ambos os grupos
        es, var_es = effectsize_smd(mean1=m1, sd1=sd1, nobs1=10,
                                    mean2=m2, sd2=sd2, nobs2=10,
                                    usevar='pooled', ddof=1)

        efeitos.append(es)
        erros.append(var_es)

    if modelo == 'random':
        resultado = combine_effects(efeitos, erros, method_re="dl")
    else:
        resultado = combine_effects(efeitos, erros, method_re=None)

    return resultado, None


# Função para gerar o gráfico forest plot
def gerar_forest_plot(resultado, efeitos, erros, titulo):
    fig, ax = plt.subplots(figsize=(8, 6))

    ic_baixo = [eff - 1.96 * (err**0.5) for eff, err in zip(efeitos, erros)]
    ic_cima = [eff + 1.96 * (err**0.5) for eff, err in zip(efeitos, erros)]

    ax.errorbar(efeitos, range(len(efeitos)), xerr=[[
        eff - low for eff, low in zip(efeitos, ic_baixo)
    ], [
        high - eff for eff, high in zip(efeitos, ic_cima)
    ]], fmt='o', color='black', ecolor='gray', capsize=5)

    ax.axvline(x=0, linestyle='--', color='red')
    ax.set_xlabel("Efeito Padronizado (SMD)")
    ax.set_yticks(range(len(efeitos)))
    ax.set_yticklabels([f"Estudo {i+1}" for i in range(len(efeitos))])
    ax.set_title(titulo)
    plt.tight_layout()

    return fig
