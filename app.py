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
It automatically loads data from 'csv.csv' for analysis.
""")

# --- Data Loading ---
st.header("1. Data Loading and Preparation")
st.write("Loading data from 'csv.csv'...")

dados_meta_analysis = pd.DataFrame()  # Initialize empty DataFrame

try:
    # --- Data Processing Pipeline ---
    dados = load_and_prepare_data("csv.csv")
    if not dados.empty:
        dados_filtrados = filter_irrelevant_treatments(dados)
        dados_grupos = define_groups_and_residues(dados_filtrados)
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)
        
        if dados_meta_analysis.empty:
            st.warning("Insufficient data for meta-analysis after filtering and preparation.")
        else:
            st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
            st.subheader("Prepared Data Sample:")
            st.dataframe(dados_meta_analysis.head())
    else:
        st.error("Could not load or process data from 'csv.csv'. Please check the file format and ensure it uses ';' as delimiter and '.' as decimal separator.")
except FileNotFoundError:
    st.error("File 'csv.csv' not found. Please ensure it is in the root directory.")

st.markdown("---")

# --- Meta-Analysis Section ---
st.header("2. Run Meta-Analysis Models & Generate Plots")

if not dados_meta_analysis.empty:
    st.markdown("Select a model to run and visualize its results. All outputs are in academic English suitable for publication.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze by Residue Type"):
            st.subheader("Meta-Analysis by Residue Type")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Residue")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Residue Type)")
                    st.dataframe(summary_df.set_index('term'))
                    st.caption("Note: Effects represent Log Response Ratio (lnRR) compared to control. Positive values indicate higher values in treatment groups.")
                else:
                    st.warning("Insufficient data for residue type analysis.")

    with col2:
        if st.button("📊 Analyze by Variable"):
            st.subheader("Meta-Analysis by Variable")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Variable")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Variable)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("Insufficient data for variable analysis.")

    with col3:
        if st.button("🔗 Analyze Interaction (Residue × Variable)"):
            st.subheader("Meta-Analysis of Residue × Variable Interaction")
            with st.spinner("Calculating..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Interaction")
                if fig:
                    st.pyplot(fig)
                    st.subheader("Model Summary (Interaction)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("Insufficient data for interaction analysis.")

    st.markdown("---")
    st.header("3. Additional Diagnostic Plots")

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Generate Forest Plot"):
            st.subheader("Forest Plot of Individual Studies")
            with st.spinner("Generating..."):
                fig_forest = generate_forest_plot(dados_meta_analysis)
                if fig_forest:
                    st.pyplot(fig_forest)
                    st.caption("Forest plot showing effect sizes (lnRR) with 95% confidence intervals for individual studies.")
                else:
                    st.warning("Could not generate forest plot. Check data sufficiency.")
    
    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            st.subheader("Funnel Plot for Publication Bias Assessment")
            with st.spinner("Generating..."):
                fig_funnel = generate_funnel_plot(dados_meta_analysis)
                if fig_funnel:
                    st.pyplot(fig_funnel)
                    st.caption("Funnel plot for assessing potential publication bias. Asymmetry may indicate bias.")
                else:
                    st.warning("Could not generate funnel plot. Check data sufficiency.")

else:
    st.info("Please ensure 'csv.csv' is in the correct directory and was processed successfully (with at least 2 records) to proceed with analysis.")

st.markdown("---")
st.markdown("🔬 Developed using Streamlit and Python for meta-analysis of vermicompost quality. Academic English output suitable for publication.")
