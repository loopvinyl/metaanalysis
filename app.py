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
This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality.
You can either upload your own data or use the pre-loaded example data.
""")

# --- Data Upload / Loading ---
st.header("1. Load Data")

uploaded_file = st.file_uploader("Choose a CSV file to upload (optional)", type="csv")

dados_meta_analysis = pd.DataFrame() # Initialize empty DataFrame
file_path_to_process = None

if uploaded_file is not None:
    # If a file is uploaded, save it and set its path
    if not os.path.exists("data"):
        os.makedirs("data")
    temp_file_path = os.path.join("data", "uploaded_data.csv")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path_to_process = temp_file_path
    st.success("File uploaded successfully! Processing data...")
else:
    # If no file is uploaded, try to use the example file from the repo
    # Make sure 'data/example.csv' exists in your GitHub repo!
    # If your file is named csv.csv, ensure it's in data/csv.csv and adjust this path
    default_file_path = os.path.join("data", "csv.csv") # <--- ALTERADO AQUI: 'csv.csv' ao invés de 'example.csv'
    if os.path.exists(default_file_path):
        file_path_to_process = default_file_path
        st.info("No file uploaded. Using the example data from the repository.")
    else:
        st.warning(f"No default data found at '{default_file_path}'. Please upload a CSV file.")


if file_path_to_process:
    # --- Data Processing Pipeline ---
    dados = load_and_prepare_data(file_path_to_process)
    if not dados.empty:
        dados_filtrados = filter_irrelevant_treatments(dados)
        dados_grupos = define_groups_and_residues(dados_filtrados)
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)

        if dados_meta_analysis.empty:
            st.warning("Not enough data to perform meta-analysis after filtering and preparation.")
        else:
            st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
            st.subheader("Prepared Data Sample:")
            st.dataframe(dados_meta_analysis.head())
    else:
        st.error("Could not load or process data. Please check the file format or default data.")
else:
    st.info("Please upload a CSV file or ensure default data is available in the 'data/' directory to proceed with the analysis.")

st.markdown("---")

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
    st.info("Please upload data and ensure it's successfully processed before running analyses.")

st.markdown("---")
st.markdown("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
