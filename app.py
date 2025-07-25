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

dados_meta_analysis = pd.DataFrame()

try:
    # Check file exists
    if not os.path.exists("csv.csv"):
        st.error("Error: 'csv.csv' not found in the root directory.")
        st.info("Please ensure the file is in the same directory as the application.")
    else:
        # Show file preview
        with open("csv.csv", 'r') as f:
            preview_lines = [next(f) for _ in range(3)]
        st.code("File preview (first 3 lines):\n" + "".join(preview_lines))
        
        # Process data
        dados = load_and_prepare_data("csv.csv")
        
        if not dados.empty:
            st.success(f"Data loaded successfully! Initial records: {len(dados)}")
            
            # Processing pipeline
            dados_filtrados = filter_irrelevant_treatments(dados)
            st.write(f"After treatment filtering: {len(dados_filtrados)} records")
            
            dados_grupos = define_groups_and_residues(dados_filtrados)
            dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)
            
            if not dados_meta_analysis.empty:
                st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
                st.subheader("Sample of Prepared Data:")
                st.dataframe(dados_meta_analysis.head())
            else:
                st.warning("Insufficient data for meta-analysis after processing.")
        else:
            st.error("Could not process the data. Please check:")
            st.error("- File uses semicolons (;) as delimiters")
            st.error("- File uses periods (.) for decimals")
            st.error("- Required columns exist: Study, Treatment, Variable, Mean, Std Dev")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# --- Meta-Analysis Section ---
if not dados_meta_analysis.empty:
    st.header("2. Meta-Analysis Models")
    st.markdown("""
    Select a model to analyze the effects of residues and variables on vermicompost quality.
    All outputs are in academic English suitable for publication.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze by Residue Type"):
            with st.spinner("Running residue type analysis..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, "Residue")
                if fig:
                    st.subheader("Residue Type Effects")
                    st.pyplot(fig)
                    st.dataframe(summary_df.set_index('term'))
                    st.caption("""
                    Effect sizes (lnRR) for different residue types. 
                    Positive values indicate higher values compared to control.
                    """)

    with col2:
        if st.button("📊 Analyze by Variable"):
            with st.spinner("Running variable analysis..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, "Variable")
                if fig:
                    st.subheader("Variable Effects")
                    st.pyplot(fig)
                    st.dataframe(summary_df.set_index('term'))

    with col3:
        if st.button("🔗 Analyze Interaction"):
            with st.spinner("Running interaction analysis..."):
                summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, "Interaction")
                if fig:
                    st.subheader("Residue × Variable Interactions")
                    st.pyplot(fig)
                    st.dataframe(summary_df.set_index('term'))

    # Diagnostic Plots
    st.header("3. Diagnostic Plots")
    st.markdown("Generate additional plots to assess results and potential biases.")

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Generate Forest Plot"):
            with st.spinner("Creating forest plot..."):
                fig = generate_forest_plot(dados_meta_analysis)
                if fig:
                    st.subheader("Forest Plot")
                    st.pyplot(fig)
                    st.caption("""
                    Forest plot showing effect sizes (lnRR) with 95% confidence intervals 
                    for individual studies.
                    """)

    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            with st.spinner("Creating funnel plot..."):
                fig = generate_funnel_plot(dados_meta_analysis)
                if fig:
                    st.subheader("Funnel Plot")
                    st.pyplot(fig)
                    st.caption("""
                    Funnel plot for assessing publication bias. Asymmetry may indicate bias.
                    """)

else:
    st.info("Please ensure the data is loaded and processed successfully to run analyses.")

st.markdown("---")
st.markdown("""
🔬 Developed for academic research using Streamlit and Python.  
All outputs are in English suitable for publication.
""")
