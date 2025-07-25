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
**Select your data file directly from the server's data directory to begin the analysis.**
""")

# --- Data Selection from Directory ---
st.header("1. Select Data File")

# Define the directory where your CSV files are located
# IMPORTANT: Adjust this path to your actual data directory on the server
DATA_DIR = "data" # Make sure this 'data' folder exists in the same directory as app.py

dados_meta_analysis = pd.DataFrame() # Initialize empty DataFrame

# Check if the data directory exists
if not os.path.exists(DATA_DIR):
    st.error(f"The data directory '{DATA_DIR}' was not found. Please create it and add your CSV files.")
    st.stop() # Stop execution if the directory doesn't exist

# List all CSV files in the data directory
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

if not csv_files:
    st.warning(f"No CSV files found in the directory '{DATA_DIR}'. Please add your data files.")
    # No data to process, so keep dados_meta_analysis empty
else:
    # Allow the user to select a file from the list
    selected_file = st.selectbox("Choose a CSV file from the directory:", ["-- Select a file --"] + csv_files)

    if selected_file != "-- Select a file --":
        file_path = os.path.join(DATA_DIR, selected_file)
        
        st.success(f"File '{selected_file}' selected. Processing data...")

        # --- Data Processing Pipeline ---
        [cite_start]dados = load_and_prepare_data(file_path) # [cite: 3]
        if not dados.empty:
            [cite_start]dados_filtrados = filter_irrelevant_treatments(dados) # [cite: 5]
            [cite_start]dados_grupos = define_groups_and_residues(dados_filtrados) # [cite: 10]
            [cite_start]dados_meta_analysis = prepare_for_meta_analysis(dados_grupos) # [cite: 14]
            
            if dados_meta_analysis.empty:
                st.warning("Not enough data to perform meta-analysis after filtering and preparation.")
            else:
                [cite_start]st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.") # [cite: 41]
                st.subheader("Prepared Data Sample:")
                st.dataframe(dados_meta_analysis.head())
        else:
            st.error("Could not load or process data from the selected file. Please check the file format or its content.")
    else:
        st.info("Please select a CSV file from the directory to proceed with the analysis.")

st.markdown("---")

# --- Meta-Analysis Section ---
st.header("2. Run Meta-Analysis Models & Generate Plots")

if not dados_meta_analysis.empty:
    [cite_start]st.markdown("Select a model to run and visualize its results. All plots and outputs are in English.") # [cite: 42]

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze by Residue Type"):
            st.subheader("Analysis by Residue Type")
            with st.spinner("Calculating..."):
                [cite_start]summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Residue") # [cite: 22]
                [cite_start]if fig: # [cite: 43]
                    [cite_start]st.pyplot(fig) # [cite: 45]
                    st.subheader("Model Summary (Residue Type)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    [cite_start]st.warning("Could not generate plot for Residue Type model. Check data sufficiency.") # [cite: 44]

    with col2:
        if st.button("📊 Analyze by Variable"):
            st.subheader("Analysis by Variable")
            with st.spinner("Calculating..."):
                [cite_start]summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Variable") # [cite: 22]
                [cite_start]if fig: # [cite: 45]
                    [cite_start]st.pyplot(fig) # [cite: 45]
                    st.subheader("Model Summary (Variable)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    [cite_start]st.warning("Could not generate plot for Variable model. Check data sufficiency.") # [cite: 46]

    with col3:
        if st.button("🔗 Analyze Interaction (Residue × Variable)"):
            st.subheader("Analysis by Interaction (Residue × Variable)")
            with st.spinner("Calculating..."):
                [cite_start]summary_df, fig = run_meta_analysis_and_plot(dados_meta_analysis, model_type="Interaction") # [cite: 22]
                [cite_start]if fig: # [cite: 47]
                    [cite_start]st.pyplot(fig) # [cite: 47]
                    st.subheader("Model Summary (Interaction)")
                    st.dataframe(summary_df.set_index('term'))
                else:
                    st.warning("Could not generate plot for Interaction model. Check data sufficiency.")

    st.markdown("---")
 
    [cite_start]st.header("3. Additional Plots") # [cite: 48]

    col_forest, col_funnel = st.columns(2)
    with col_forest:
        if st.button("🌳 Generate Forest Plot"):
            st.subheader("Forest Plot of Individual Studies")
            with st.spinner("Generating forest plot..."):
                [cite_start]fig_forest = generate_forest_plot(dados_meta_analysis) # [cite: 28]
                [cite_start]if fig_forest: # [cite: 49]
                    [cite_start]st.pyplot(fig_forest) # [cite: 49]
                else:
                    [cite_start]st.warning("Could not generate Forest Plot. Check data sufficiency.") # [cite: 50]
    
    with col_funnel:
        if st.button("🧪 Generate Funnel Plot"):
            st.subheader("Funnel Plot for Publication Bias")
            with st.spinner("Generating funnel plot..."):
                [cite_start]fig_funnel = generate_funnel_plot(dados_meta_analysis) # [cite: 34]
                [cite_start]if fig_funnel: # [cite: 51]
                    [cite_start]st.pyplot(fig_funnel) # [cite: 51]
                else:
                    st.warning("Could not generate Funnel Plot. Check data sufficiency.")

else:
    st.info("Please select data from the directory and ensure it's successfully processed before running analyses.")

st.markdown("---")
st.markdown("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
