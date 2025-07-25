import streamlit as st
import pandas as pd
import meta_analysis as ma # Importa seu módulo com as funções de análise
import os

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Meta-análise de Vermicompostagem",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")
st.markdown("This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality, using pre-loaded example data from the repository.")

# --- Seção 1: Carregamento e Preparação dos Dados ---
st.header("1. Loading Data")

# Link para o diagrama PRISMA (se aplicável, ajuste o URL se necessário)
st.markdown("""
<a href="https://example.com/prisma_diagram.png" target="_blank">
    PRISMA 2020 Flow Diagram
</a>
""", unsafe_allow_html=True)

st.info("View Study Selection Process") # Pode ser expandido para mostrar mais detalhes ou um modal

# Define o caminho do arquivo de dados (agora apontando para o CSV)
file_path = "data/csv.csv"

# Carregar e preparar os dados usando a função do meta_analysis.py
st.write(f"Loading data from: '{file_path}'")

# Adiciona um placeholder para a mensagem de status de carregamento
status_message = st.empty()

try:
    dados_preparados = ma.load_and_prepare_data(file_path)

    if not dados_preparados.empty:
        status_message.success("Data prepared for meta-analysis. "
                               f"{len(dados_preparados)} records available.")
        st.subheader("Prepared Data Sample:")
        st.dataframe(dados_preparados.head())

        st.info(
            f"Note on Data Filtering: The initial dataset contained **{len(dados_preparados)}** records "
            "after initial load and numeric conversion. During preparation, records were filtered to include "
            "only relevant vermicompost treatments, exclude initial/raw material samples, ensure the presence "
            "of control groups for variables, and remove any entries with missing or invalid data for "
            "meta-analysis calculations. This process resulted in **{len(dados_preparados)}** records for the meta-analysis. "
            "The exact number after all filtering steps (e.g., control group presence) is reflected in the logs."
        )

    else:
        status_message.error(
            f"Could not load or process data from '{file_path}'. Please check the file format or content."
        )
except Exception as e:
    status_message.error(f"An unexpected error occurred during data loading: {e}")
    dados_preparados = pd.DataFrame() # Garante que dados_preparados seja um DataFrame vazio em caso de erro

# --- Seção 2: Rodar Modelos de Meta-Análise e Gerar Gráficos ---
st.header("2. Run Meta-Analysis Models & Generate Plots")

if not dados_preparados.empty:
    model_choice = st.selectbox(
        "Select a model to run and visualize its results. All plots and outputs are in English.",
        ("Residue", "Variable", "Interaction")
    )

    summary_df, fig = ma.run_meta_analysis_and_plot(dados_preparados, model_type=model_choice)

    if fig:
        st.pyplot(fig)
        if not summary_df.empty:
            st.subheader(f"Summary Table for {model_choice} Model:")
            st.dataframe(summary_df)
    else:
        st.warning(f"Could not generate plot for {model_choice} model. Check data or model choice.")
else:
    st.warning("Data could not be loaded or processed. Please check the 'data/csv.csv' file.")


# --- Seção 3: Gráficos Adicionais ---
st.header("3. Additional Plots")

if not dados_preparados.empty:
    plot_choice = st.selectbox(
        "Select an additional plot to generate:",
        ("Forest Plot", "Funnel Plot")
    )

    if plot_choice == "Forest Plot":
        forest_fig = ma.generate_forest_plot(dados_preparados)
        if forest_fig:
            st.pyplot(forest_fig)
        else:
            st.warning("Could not generate Forest Plot. Data might be insufficient or invalid.")
    elif plot_choice == "Funnel Plot":
        funnel_fig = ma.generate_funnel_plot(dados_preparados)
        if funnel_fig:
            st.pyplot(funnel_fig)
        else:
            st.warning("Could not generate Funnel Plot. Data might be insufficient or invalid.")
else:
    st.warning("Cannot generate additional plots. Data was not loaded or processed correctly.")


# --- Rodapé ---
st.markdown("---")
st.markdown("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
