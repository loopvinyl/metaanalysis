import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from meta_analysis import (
    load_and_prepare_data,
    filter_irrelevant_treatments,
    define_groups_and_residues,
    prepare_for_meta_analysis,
    run_meta_analysis, # Changed to run_meta_analysis
    generate_forest_plot,
    generate_funnel_plot
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

# Initialize dados_meta_analysis outside the if block to ensure it always exists
dados_meta_analysis = pd.DataFrame() 

if uploaded_file is not None:
    # Create 'data' directory if it doesn't exist
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
            st.warning("Not enough data to perform meta-analysis after filtering and preparation. Please ensure your CSV contains relevant data for meta-analysis.")
        else:
            st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
            st.subheader("Prepared Data Sample:")
            st.dataframe(dados_meta_analysis.head())
    else:
        st.error("Could not load or process data from the uploaded file. Please check the file format and ensure it uses ';' as delimiter and '.' as decimal separator.")
else:
    st.info("Please upload a CSV file to proceed with the analysis.")

st.markdown("---")

# --- Meta-Analysis Section ---
st.header("2. Run Meta-Analysis Models & Generate Plots")

# Only show analysis buttons if data is ready
if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2: # At least 2 data points for meta-analysis
    st.markdown("Select a model to run and visualize its results. All plots and outputs are in English.")

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
                    # Adjust term names
                    coef_df.index = coef_df.index.str.replace('C(Residue)[T.', '', regex=False).str.replace(']', '', regex=False)
                    st.dataframe(coef_df)
                    
                    # Generate and show plots (using the specific model results for forest plot if desired)
                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Residue Type Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest) # Close figure to free memory
                    else:
                        st.warning("Could not generate Forest Plot for Residue Type model. Check data sufficiency.")

                else:
                    st.warning("Could not run analysis for Residue Type. Not enough data or issue with model fitting.")

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
                    # Adjust term names
                    coef_df.index = coef_df.index.str.replace('C(Variable)[T.', '', regex=False).str.replace(']', '', regex=False)
                    st.dataframe(coef_df)
                    
                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Variable Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest)
                    else:
                        st.warning("Could not generate Forest Plot for Variable model. Check data sufficiency.")
                else:
                    st.warning("Could not run analysis for Variable. Not enough data or issue with model fitting.")

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
                    # Adjust term names for interaction
                    coef_df.index = coef_df.index.str.replace('C(Residue)[T.', '', regex=False) \
                                               .str.replace(']:C(Variable)[T.', ':', regex=False) \
                                               .str.replace(']', '', regex=False)
                    st.dataframe(coef_df)

                    fig_forest = generate_forest_plot(dados_meta_analysis, results, title="Forest Plot - Interaction Model")
                    if fig_forest:
                        st.pyplot(fig_forest)
                        plt.close(fig_forest)
                    else:
                        st.warning("Could not generate Forest Plot for Interaction model. Check data sufficiency.")
                else:
                    st.warning("Could not run analysis for Interaction. Not enough data or issue with model fitting.")

    st.markdown("---")
 
    st.header("3. Additional Plots")

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Generate Overall Forest Plot"): # Changed button text to differentiate
            st.subheader("Forest Plot of Individual Studies")
            # For the overall forest plot, we don't need model_results, just individual study data
            # Passing an empty dict for model_results as it expects it.
            # The generate_forest_plot should handle if model_results['beta'] is empty.
            fig_forest = generate_forest_plot(dados_meta_analysis, {}, title="Overall Forest Plot (Individual Studies)")
            if fig_forest:
                st.pyplot(fig_forest)
                plt.close(fig_forest)
            else:
                st.warning("Could not generate Overall Forest Plot. Check data sufficiency.")
    
    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            st.subheader("Funnel Plot for Publication Bias")
            with st.spinner("Generating funnel plot..."):
                fig_funnel = generate_funnel_plot(dados_meta_analysis)
                if fig_funnel:
                    st.pyplot(fig_funnel)
                    plt.close(fig_funnel)
                else:
                    st.warning("Could not generate Funnel Plot. Check data sufficiency.")

else:
    st.info("Please upload data and ensure it's successfully processed with at least 2 records before running analyses.")

st.markdown("---")
st.markdown("🔬 Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
