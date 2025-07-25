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
    generate_funnel_plot,
    run_analysis_by_variable
)

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Vermicompost Meta-Analysis")

# --- Title ---
st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")

st.markdown("""
This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality.
Upload your data (a CSV file) to begin the analysis.
""")

# --- Data Upload ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

dados_meta_analysis = pd.DataFrame() # Initialize empty DataFrame

if uploaded_file is not None:
    # Save the uploaded file to the 'data' directory
    if not os.path.exists("data"):
        os.makedirs("data")
    
    file_path = os.path.join("data", "uploaded_data.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully! Processing data...")

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
        st.error("Could not load or process data from the uploaded file. Please check the file format.")
else:
    st.info("Please upload a CSV file to proceed with the analysis.")

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

    st.markdown("---")
    st.header("4. Analysis by Important Variables")
    
    if st.button("📊 Analyze TOC, N, pH, EC"):
        st.subheader("Analysis by Important Variables")
        with st.spinner("Calculating..."):
            results = run_analysis_by_variable(dados_meta_analysis)
            for var, result in results.items():
                if result:
                    st.subheader(f"Analysis for {var}")
                    st.pyplot(result['plot'])
                    st.dataframe(result['summary'].set_index('term'))

else:
    st.info("Please upload data and ensure it's successfully processed before running analyses.")

st.markdown("---")
st.markdown("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
