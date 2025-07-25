import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.meta_analysis import effectsize_smd, combine_effects
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    try:
        # Detecta delimitador automaticamente, remove BOM
        dados = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python')
        print(f"✅ Dados carregados de: {file_path}")
    except Exception as e:
        print(f"❌ Erro ao carregar o CSV: {e}")
        return pd.DataFrame()

    # Limpa e padroniza os nomes das colunas
    dados.columns = dados.columns.str.strip().str.replace('ï»¿', '').str.replace('_', ' ')

    # Renomeia colunas conhecidas
    rename_dict = {
        'Mean Value': 'Mean',
        'Std Dev': 'Std Dev',
        'Std_Dev': 'Std Dev'
    }
    dados.rename(columns=rename_dict, inplace=True)

    # Verifica colunas obrigatórias
    colunas_necessarias = ['Variable', 'Study', 'Treatment', 'Mean', 'Std Dev']
    for col in colunas_necessarias:
        if col not in dados.columns:
            print(f"❌ Coluna ausente: {col}")
            return pd.DataFrame()

    # Conversão de valores
    dados['Mean'] = pd.to_numeric(dados['Mean'], errors='coerce')
    dados['Std Dev'] = pd.to_numeric(dados['Std Dev'], errors='coerce')
    dados.dropna(subset=['Mean', 'Std Dev'], inplace=True)

    return dados


def rodar_meta_analise(dados, variavel, tratamento, modelo='random'):
    df = dados[dados['Variable'] == variavel].copy()
    tratamento_df = df[df['Treatment'] == tratamento]
    controle_df = df[df['Treatment'] != tratamento]

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

        es, var_es = effectsize_smd(
            mean1=m1, sd1=sd1, nobs1=10,
            mean2=m2, sd2=sd2, nobs2=10,
            usevar='pooled', ddof=1
        )

        efeitos.append(es)
        erros.append(var_es)

    if modelo == 'random':
        resultado = combine_effects(efeitos, erros, method_re="dl")
    else:
        resultado = combine_effects(efeitos, erros, method_re=None)

    return resultado, None


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
    ax.set_xlabel("Standardized Mean Difference (SMD)")
    ax.set_yticks(range(len(efeitos)))
    ax.set_yticklabels([f"Study {i+1}" for i in range(len(efeitos))])
    ax.set_title(titulo)
    plt.tight_layout()

    return fig
