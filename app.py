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

# --- Title ---
st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")

st.markdown("""
This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality,
using pre-loaded example data from the repository.
""")

# --- Data Loading Section (including PRISMA) ---
st.header("1. Loading Data")

# --- PRISMA Flow Diagram Section ---
st.subheader("PRISMA 2020 Flow Diagram")

with st.expander("View Study Selection Process"):
    st.markdown("""
    Here's the adapted PRISMA 2020 flow diagram illustrating the study selection process:

    **Search Strategy:**
    The following search terms were used in **SCOPUS, Web of Science, and PubMed** databases:
    `"vermicomposting" AND ( "characterization of vermicompost" OR "Chemical and physical variables" OR "Physico-chemical characterization" OR "Nutrient dynamics" )`

    **Identification of studies via databases and registers**
    ----------------------------------------------------

    Records identified from*:
    - Databases (n = 125)
        - Scopus (n = 48)
        - Web of Science (n = 8)
        - PubMed (n = 69)
    - Registers (n = 0)

    Records removed before screening:
    - Duplicate records removed (n = 1)
    - Records marked as ineligible by automation tools (n = 0)
    - Records removed for other reasons (n = 0)

    Identification
    --------------

    Records screened (n = 124)

    Records excluded**:
    - Missing key variables (pH, CE, TOC/MO, P, K, N, C/N) (n = 117)
    - Non-primary literature (reviews/book chapters) (n = 2)

    Reports sought for retrieval (n = 5)

    Reports not retrieved (n = 0)

    Screening
    ---------

    Reports assessed for eligibility (n = 5)

    Reports excluded:
    - Insufficient data reporting (n = 0)
    - Other reasons (n = 0)

    Studies included in review (n = 5)

    Reports of included studies:
    - Ramos et al. (2024)
    - Kumar et al. (2023)
    - Quadar et al. (2022)
    - Srivastava et al. (2020)
    - Santana et al. (2020)
    """)

st.markdown("---") # Separador após a seção PRISMA, ainda dentro do contexto de carregamento de dados

# --- Data Loading Logic (now comes after PRISMA within section 1) ---
dados_meta_analysis = pd.DataFrame() # Initialize empty DataFrame
# ALTERAÇÃO AQUI: Mude de csv.csv para excel.xlsx
file_path_to_process = os.path.join("data", "excel.xlsx") # Caminho para o novo arquivo Excel

if os.path.exists(file_path_to_process):
    st.info(f"Loading data from: '{file_path_to_process}'")

    # --- Data Processing Pipeline ---
    dados_raw = load_and_prepare_data(file_path_to_process) # Renamed to dados_raw for clarity

    if not dados_raw.empty:
        total_initial_records = len(dados_raw)

        dados_filtrados = filter_irrelevant_treatments(dados_raw)
        dados_grupos = define_groups_and_residues(dados_filtrados)
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)

        if dados_meta_analysis.empty:
            st.warning("Not enough data to perform meta-analysis after filtering and preparation. Please check the data file.")
        else:
            st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
            st.subheader("Prepared Data Sample:")
            st.dataframe(dados_meta_analysis.head())

            st.markdown(f"""
            **Note on Data Filtering:**
            The initial dataset contained **{total_initial_records}** records.
            During preparation, records were filtered to include only relevant vermicompost treatments,
            exclude initial/raw material samples, ensure the presence of control groups for variables,
            and remove any entries with missing or invalid data for meta-analysis calculations.
            This process resulted in **{len(dados_meta_analysis)}** records for the meta-analysis.
            """)
    else:
        st.error(f"Could not load or process data from '{file_path_to_process}'. Please check the file format or content.")
else:
    st.error(f"Error: Default data file '{file_path_to_process}' not found in the repository. Please ensure it is present.")
    st.info("The application requires this file to run.")


st.markdown("---") # Separador final da seção 1

# --- Meta-Analysis Section ---
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
                    st.warning("Could not generate plot for Residue Type model. Check data sufficiency.")

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
                    st.warning("Could not generate plot for Variable model. Check data sufficiency.")

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
                    st.warning("Could not generate plot for Interaction model. Check data sufficiency.")

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
                    st.warning("Could not generate Forest Plot. Check data sufficiency.")

    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            st.subheader("Funnel Plot for Publication Bias")
            with st.spinner("Generating funnel plot..."):
                fig_funnel = generate_funnel_plot(dados_meta_analysis)
                if fig_funnel:
                    st.pyplot(fig_funnel)
                else:
                    st.warning("Could not generate Funnel Plot. Check data sufficiency.")

else:
    st.info("Data could not be loaded or processed. Please check the 'data/excel.xlsx' file.") # Atualizado para Excel

st.markdown("---")
st.markdown("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
