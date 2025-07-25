import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from meta_analysis import (
    load_and_prepare_data,
    filter_irrelevant_treatments,
    define_groups_and_residues,
    prepare_for_meta_analysis,
    run_meta_analysis_and_plot,
    generate_forest_plot,
    generate_funnel_plot
)

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Vermicompost Meta-Analysis")

# --- Startup Check ---
st.write("🚀 App iniciado com sucesso.")

# --- Title ---
st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")

st.markdown("""
This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality.
Reading data from a CSV file located in the `data/` folder.
""")

# --- Data Load ---
st.header("1. Load Data from Directory")

file_path = "data/csv.csv"
dados_meta_analysis = pd.DataFrame()

st.write(f"🔍 Verificando existência do arquivo: `{file_path}`")
st.write("📁 Arquivos no diretório 'data/':")
st.write(os.listdir("data") if os.path.exists("data") else "❌ Diretório não encontrado.")

if os.path.exists(file_path):
    st.success("📂 Lendo arquivo CSV local...")

    # --- Processamento dos Dados ---
    dados = load_and_prepare_data(file_path)
    st.write("🧪 Prévia dos dados brutos:")
    st.dataframe(dados.head())

    if not dados.empty:
        dados_filtrados = filter_irrelevant_treatments(dados)
        dados_grupos = define_groups_and_residues(dados_filtrados)
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)
        
        if dados_meta_analysis.empty:
            st.warning("⚠️ Dados insuficientes após o pré-processamento.")
        else:
            st.success(f"✅ Dados prontos para meta-análise ({len(dados_meta_analysis)} registros).")
            st.subheader("🔎 Amostra dos dados preparados:")
            st.dataframe(dados_meta_analysis.head())
    else:
        st.error("❌ Erro ao carregar ou processar os dados. Verifique o conteúdo do arquivo CSV.")
else:
    st.error(f"❌ Arquivo não encontrado: {file_path}")
    st.stop()

st.markdown("---")

# --- Meta-Analysis Models ---
st.header("2. Run Meta-Analysis Models & Generate Plots")

if not dados_meta_analysis.empty:
    st.markdown("Select a model to run and visualize its results. All plots and outputs are in English.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze by Residue Type"):
            st.subheader("Analysis by Residue Type")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Residue")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Residue Type)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("⚠️ Plot not generated. Possibly insufficient data.")

    with col2:
        if st.button("📊 Analyze by Variable"):
            st.subheader("Analysis by Variable")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Variable")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Variable)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("⚠️ Plot not generated. Possibly insufficient data.")

    with col3:
        if st.button("🔗 Analyze Interaction (Residue × Variable)"):
            st.subheader("Analysis by Interaction (Residue × Variable)")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Interaction")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Interaction)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("⚠️ Plot not generated. Possibly insufficient data.")

    st.markdown("---")
    st.header("3. Additional Plots")

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Generate Forest Plot"):
            st.subheader("Forest Plot of Individual Studies")
            with st.spinner("Generating forest plot..."):
                fig_forest = generate_forest_plot(dados_meta_analysis)
                if fig_forest:
                    st.pyplot(fig_forest)
                else:
                    st.warning("⚠️ Could not generate Forest Plot.")

    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            st.subheader("Funnel Plot for Publication Bias")
            with st.spinner("Generating funnel plot..."):
                fig_funnel = generate_funnel_plot(dados_meta_analysis)
                if fig_funnel:
                    st.pyplot(fig_funnel)
                else:
                    st.warning("⚠️ Could not generate Funnel Plot.")
else:
    st.info("Please ensure data is processed before running analyses.")

st.markdown("---")
st.markdown("🔬 Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
