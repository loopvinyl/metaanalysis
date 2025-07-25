import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle # Import pickle for saving results, but we will comment out or remove its usage

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
This application performs a meta-analysis to evaluate the effect of different organic residues on vermicompost quality.
It automatically loads data from `csv.csv` and performs analysis using weighted regression models (WLS) to estimate log-response ratios (lnRR).
""")

# --- Data Loading and Preparation ---
st.header("1. Data Loading and Preparation")

# Define the path to the CSV file - ADJUSTED TO BE IN THE SAME DIRECTORY AS app.py
file_path = "csv.csv" 

dados_meta_analysis = pd.DataFrame() # Initialize empty DataFrame

if os.path.exists(file_path):
    st.success("📂 File 'csv.csv' loaded successfully! Processing data...")

    # --- Data Processing Pipeline ---
    dados = load_and_prepare_data(file_path)
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
        st.error("Could not load or process data from 'csv.csv'. Please check the file format and ensure it uses ';' as a delimiter and '.' as a decimal separator.")
else:
    st.error("❌ File 'csv.csv' not found in the same directory as 'app.py'. Please place it there and reload the app.")

# --- Meta-Analysis Modeling and Visualization ---
st.header("2. Meta-Analysis Modeling and Visualization")

if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2: # At least 2 records needed for models
    
    st.markdown("Performing meta-analysis for different models:")

    # 2.1. Model by residue type
    st.subheader("2.1. Model by Residue Type")
    residue_model_summary_df, residue_coeff_plot_fig = run_meta_analysis_and_plot(
        dados_meta_analysis, 
        model_type="Residue",
        plot_title="Effect of Residues on Vermicompost Quality"
    )
    if residue_coeff_plot_fig:
        st.pyplot(residue_coeff_plot_fig)
        st.write("Model Summary:")
        st.dataframe(residue_model_summary_df)

    # 2.2. Model by variable
    st.subheader("2.2. Model by Variable")
    variable_model_summary_df, _ = run_meta_analysis_and_plot(
        dados_meta_analysis, 
        model_type="Variable",
        plot_title="Effect of Variables on Vermicompost Quality"
    )
    st.write("Model Summary:")
    st.dataframe(variable_model_summary_df)

    # 2.3. Interaction model (Residue × Variable)
    st.subheader("2.3. Interaction Model (Residue × Variable)")
    interaction_model_summary_df, _ = run_meta_analysis_and_plot(
        dados_meta_analysis, 
        model_type="Interaction",
        plot_title="Interaction Effect of Residue and Variable on Vermicompost Quality"
    )
    st.write("Model Summary:")
    st.dataframe(interaction_model_summary_df)

    # --- Save Results ---
    # Removido o cabeçalho e a lógica de salvamento.
    # st.header("4. Save Results")
    # try:
    #     # Saving results (models and data)
    #     with open("meta_analysis_results.pkl", "wb") as f:
    #         pickle.dump({
    #             "Residue_Model_Summary": residue_model_summary_df,
    #             "Variable_Model_Summary": variable_model_summary_df,
    #             "Interaction_Model_Summary": interaction_model_summary_df,
    #             "Data": dados_meta_analysis
    #         }, f)
    #     st.success("Meta-analysis results saved successfully to `meta_analysis_results.pkl`.")
    # except Exception as e:
    #     st.error(f"Error saving results: {e}")

else:
    st.info("Please ensure 'csv.csv' is in the same directory as 'app.py' and processed successfully (with at least 2 records) to proceed with the analysis.")

# --- Diagnostic and Additional Analyses ---
st.header("3. Diagnostic and Additional Analyses")

if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2:
    st.markdown("Further analyses and diagnostic plots:")

    # 3.1. Analysis by specific variable (e.g., TOC, N, pH, EC)
    st.subheader("3.1. Detailed Analysis for Key Variables")
    important_vars = ["TOC", "N", "pH", "EC"]
    
    for var in important_vars:
        st.markdown(f"#### Analysis for Variable: **{var}**")
        temp_data = dados_meta_analysis[dados_meta_analysis["Variable"] == var]
        
        if len(temp_data) > 1: # At least 2 data points for a meaningful model
            temp_model_summary_df, temp_coeff_plot_fig = run_meta_analysis_and_plot(
                temp_data, 
                model_type="Residue",
                plot_title=f"Effect of Residues on {var} in Vermicompost"
            )
            if temp_coeff_plot_fig:
                st.pyplot(temp_coeff_plot_fig)
                st.write("Model Summary:")
                st.dataframe(temp_model_summary_df)
        else:
            st.info(f"Insufficient data for detailed analysis of variable: {var}")

    # 3.2. Diagnostics (Funnel Plot for Publication Bias)
    st.subheader("3.2. Funnel Plot for Publication Bias")
    # Using the residue_model data for funnel plot as it's typically done for overall effects
    if 'lnRR' in dados_meta_analysis.columns and 'var_lnRR' in dados_meta_analysis.columns and not dados_meta_analysis.empty:
        funnel_fig = generate_funnel_plot(dados_meta_analysis['lnRR'], dados_meta_analysis['var_lnRR'])
        if funnel_fig:
            st.pyplot(funnel_fig)
    else:
        st.info("Data for funnel plot not available or insufficient after meta-analysis preparation.")
else:
    st.info("No meta-analysis data available for additional analyses and diagnostics.")

st.markdown("""
---
🔬 Developed using Streamlit and Python for meta-analysis of vermicompost quality.
""")
