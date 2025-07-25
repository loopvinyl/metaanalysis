import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from meta_analysis import (
    load_and_prepare_data,
    filter_irrelevant_treatments,
    define_groups_and_residues,
    prepare_for_meta_analysis,
    run_meta_analysis, # Importa a função atualizada para MixedLM/REML
    generate_forest_plot,
    generate_funnel_plot
)

# --- Configuração do Aplicativo ---
st.set_page_config(layout="wide", page_title="Vermicompost Meta-Analysis")

# --- Título ---
st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")

st.markdown("""
Esta aplicação realiza uma meta-análise para avaliar o efeito de diferentes resíduos na qualidade do vermicomposto.
Ela carrega automaticamente os dados do arquivo 'csv.csv' para a análise.
""")

# --- Carregamento e Processamento de Dados ---
st.header("1. Carregamento e Preparação de Dados")

# Define o caminho para o seu arquivo CSV.
# Assume que 'csv.csv' está no mesmo diretório que 'app.py'.
file_path = "csv.csv" 
# Se o 'csv.csv' estiver em uma subpasta 'data', use:
# file_path = os.path.join("data", "csv.csv")

dados_meta_analysis = pd.DataFrame() # Inicializa um DataFrame vazio

if os.path.exists(file_path):
    st.info(f"Carregando dados de '{file_path}'...")
    # --- Pipeline de Processamento de Dados ---
    dados = load_and_prepare_data(file_path)
    if not dados.empty:
        dados_filtrados = filter_irrelevant_treatments(dados)
        dados_grupos = define_groups_and_residues(dados_filtrados)
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)
        
        if dados_meta_analysis.empty:
            st.warning("Dados insuficientes para realizar a meta-análise após filtragem e preparação. Por favor, verifique os dados em 'csv.csv'.")
        else:
            st.success(f"Dados preparados para meta-análise. {len(dados_meta_analysis)} registros disponíveis.")
            st.subheader("Amostra dos Dados Preparados:")
            st.dataframe(dados_meta_analysis.head())
    else:
        st.error(f"Não foi possível carregar ou processar dados de '{file_path}'. Por favor, verifique o formato do arquivo e certifique-se de que usa ';' como delimitador e '.' como separador decimal.")
else:
    st.error(f"O arquivo '{file_path}' não foi encontrado. Por favor, certifique-se de que 'csv.csv' está no diretório correto.")

st.markdown("---")

# --- Seção de Meta-Análise ---
st.header("2. Executar Modelos de Meta-Análise e Gerar Gráficos")

# Mostra os botões de análise apenas se os dados estiverem prontos e houver pelo menos 2 registros
if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2: 
    st.markdown("Selecione um modelo para executar e visualizar seus resultados. Todas as saídas e gráficos estão em inglês.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze by Residue Type"):
            st.subheader("Analysis by Residue Type")
            with st.spinner("Calculating..."):
                results = run_meta_analysis(dados_meta_analysis, model_type="Residue")
                if results:
                    st.markdown("#### Mixed-Effects Model (k = {k}; tau² estimator: REML)".format(k=len(dados_meta_analysis)))
                    st.write(f"**tau² (estimated residual heterogeneity):** {results['tau2']:.4f}")
                    st.write(f"**I² (residual heterogeneity / unaccounted variability):** {results['I2']:.2f}%")
                    # No QE e p-value direto do MixedLM para heterogeneidade residual, conforme discutido
                    # st.write(f"**Test for Residual Heterogeneity (QE):** {results['QE']:.4f}, p-val = {results['QE_pval']:.4f}") # Comentei essa linha
                    st.write("---")
                    st.markdown("#### Model Results:")

                    coef_df = pd.DataFrame({
                        'estimate': results['beta'],
                        'se': results['se'],
                        'zval': results['zval'],
                        'pval': results['pval'],
                        'ci.lb': results['ci.lb'],
                        'ci.ub': results['ci.ub']
                    })
                    # Ajustar nomes dos termos para exibição
                    coef_df.index = coef_df.index.str.replace('C(Residue)[T.', '', regex=False).str.replace(']', '', regex=False)
                    st.dataframe(coef_df)
                    
                    # Gerar e mostrar gráficos
                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Residue Type Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest) # Fecha a figura para liberar memória
                    else:
                        st.warning("Não foi possível gerar o Forest Plot para o modelo de Tipo de Resíduo. Verifique a suficiência dos dados.")

                else:
                    st.warning("Não foi possível executar a análise para Tipo de Resíduo. Dados insuficientes ou problema no ajuste do modelo.")

    with col2:
        if st.button("📊 Analyze by Variable"):
            st.subheader("Analysis by Variable")
            with st.spinner("Calculating..."):
                results = run_meta_analysis(dados_meta_analysis, model_type="Variable")
                if results:
                    st.markdown("#### Mixed-Effects Model (k = {k}; tau² estimator: REML)".format(k=len(dados_meta_analysis)))
                    st.write(f"**tau² (estimated residual heterogeneity):** {results['tau2']:.4f}")
                    st.write(f"**I² (residual heterogeneity / unaccounted variability):** {results['I2']:.2f}%")
                    st.write("---")
                    st.markdown("#### Model Results:")

                    coef_df = pd.DataFrame({
                        'estimate': results['beta'],
                        'se': results['se'],
                        'zval': results['zval'],
                        'pval': results['pval'],
                        'ci.lb': results['ci.lb'],
                        'ci.ub': results['ci.ub']
                    })
                    # Ajustar nomes dos termos para exibição
                    coef_df.index = coef_df.index.str.replace('C(Variable)[T.', '', regex=False).str.replace(']', '', regex=False)
                    st.dataframe(coef_df)
                    
                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Variable Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest)
                    else:
                        st.warning("Não foi possível gerar o Forest Plot para o modelo de Variável. Verifique a suficiência dos dados.")
                else:
                    st.warning("Não foi possível executar a análise para Variável. Dados insuficientes ou problema no ajuste do modelo.")

    with col3:
        if st.button("🔗 Analyze Interaction (Residue × Variable)"):
            st.subheader("Analysis by Interaction (Residue × Variable)")
            with st.spinner("Calculating..."):
                results = run_meta_analysis(dados_meta_analysis, model_type="Interaction")
                if results:
                    st.markdown("#### Mixed-Effects Model (k = {k}; tau² estimator: REML)".format(k=len(dados_meta_analysis)))
                    st.write(f"**tau² (estimated residual heterogeneity):** {results['tau2']:.4f}")
                    st.write(f"**I² (residual heterogeneity / unaccounted variability):** {results['I2']:.2f}%")
                    st.write("---")
                    st.markdown("#### Model Results:")

                    coef_df = pd.DataFrame({
                        'estimate': results['beta'],
                        'se': results['se'],
                        'zval': results['zval'],
                        'pval': results['pval'],
                        'ci.lb': results['ci.lb'],
                        'ci.ub': results['ci.ub']
                    })
                    # Ajustar nomes dos termos para interação
                    coef_df.index = coef_df.index.str.replace('C(Residue)[T.', '', regex=False) \
                                               .str.replace(']:C(Variable)[T.', ':', regex=False) \
                                               .str.replace(']', '', regex=False)
                    st.dataframe(coef_df)

                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Interaction Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest)
                    else:
                        st.warning("Não foi possível gerar o Forest Plot para o modelo de Interação. Verifique a suficiência dos dados.")
                else:
                    st.warning("Não foi possível executar a análise para Interação. Dados insuficientes ou problema no ajuste do modelo.")

    st.markdown("---")
 
    st.header("3. Gráficos Adicionais")

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Gerar Forest Plot Geral"): # Texto do botão alterado
            st.subheader("Forest Plot de Estudos Individuais")
            with st.spinner("Gerando Forest Plot..."):
                # Para o Forest Plot geral, passamos um dicionário vazio para model_results
                # porque este plot mostra os estudos individuais, não um modelo específico.
                fig_forest = generate_forest_plot(dados_meta_analysis, {}, title="Forest Plot Geral (Estudos Individuais)")
                if fig_forest:
                    st.pyplot(fig_forest)
                    plt.close(fig_forest)
                else:
                    st.warning("Não foi possível gerar o Forest Plot Geral. Verifique a suficiência dos dados.")
    
    with col_funnel:
        if st.button("🧪 Gerar Funnel Plot"):
            st.subheader("Funnel Plot para Viés de Publicação")
            with st.spinner("Gerando Funnel Plot..."):
                fig_funnel = generate_funnel_plot(dados_meta_analysis)
                if fig_funnel:
                    st.pyplot(fig_funnel)
                    plt.close(fig_funnel)
                else:
                    st.warning("Não foi possível gerar o Funnel Plot. Verifique a suficiência dos dados.")

else:
    st.info("Por favor, certifique-se de que 'csv.csv' está no diretório correto e foi processado com sucesso (com pelo menos 2 registros) para prosseguir com a análise.")

st.markdown("---")
st.markdown("🔬 Desenvolvido usando Streamlit e Python para meta-análise da qualidade do vermicomposto.")
