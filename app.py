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
This application performs a meta-analysis to evaluate the effect of different
organic residues on vermicompost quality.
It automatically loads data from `csv.csv` and performs analysis using weighted
regression models (WLS) to estimate log-response ratios (lnRR).
""")

# --- Added Results Overview Section ---
st.markdown("""
**Results Overview:** Our meta-analysis delved into **217 distinct treatments**,
with **vermicompost derived from cattle manure** serving as the foundational control.
Key findings highlight that **banana residue** significantly influenced the **electrical conductivity (EC)** of the vermicompost,
while **grape marc** showed a marked impact on its **nitrogen (N) content**.
""")

# --- New Section: 1. PRISMA Flow Diagram ---
st.header("1. PRISMA Flow Diagram: Study Identification and Selection")

st.markdown("""
A systematic literature search was conducted to identify relevant studies on the characterization and nutrient dynamics of vermicompost. The search strategy was formulated using a combination of keywords related to vermicomposting and its physico-chemical properties.

### Search Strategy and Database Selection

The primary search string employed was:
`"vermicomposting" AND ("characterization of vermicompost" OR "Chemical and physical variables" OR "Physico-chemical characterization" OR "Nutrient dynamics")`

This search query was systematically applied across three major scientific databases: **Scopus**, **Web of Science**, and **PubMed**. The initial search yielded a total of **125 records**. The breakdown of identified records by database was as follows:

* **Scopus**: 48 records
* **Web of Science**: 8 records
* **PubMed**: 69 records

### Study Identification and Screening Process

The identification and selection of studies followed a rigorous, multi-stage process, largely aligning with the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines.

#### **Identification Phase**
From the initial 125 records retrieved from the databases, one duplicate record was identified and removed, resulting in **124 unique records** for screening. No records were marked as ineligible by automation tools or removed for other reasons at this stage.

#### **Screening Phase**
The 124 unique records underwent a comprehensive screening process based on their titles and abstracts. During this phase, **119 records were excluded** due to predefined exclusion criteria. The reasons for exclusion were:
* **Missing key variables (pH, EC, TOC/MO, P, K, N, C/N)**: 117 records lacked essential physico-chemical and nutrient parameters required for the meta-analysis.
* **Non-primary literature (reviews/book chapters)**: 2 records were identified as review articles or book chapters, which were outside the scope of this primary research synthesis.

Following this screening, **5 reports were sought for retrieval** for full-text assessment. All 5 of these reports were successfully retrieved.

#### **Eligibility Phase**
The full texts of the 5 retrieved reports were then assessed for eligibility against the inclusion and exclusion criteria. At this stage, no additional reports were excluded due to insufficient data reporting or other reasons.

### Studies Included in Review

Consequently, **5 studies were deemed eligible and included** in the final systematic review and subsequent meta-analysis. These included studies were:

* Ramos et al. (2024)
* Kumar et al. (2023)
* Quadar et al. (2022)
* Srivastava et al. (2020)
* Santana et al. (2020)
""")

# Place the markdown horizontal line within st.markdown()
st.markdown("---")

# --- Renumbered Section: 2. Data Loading and Preparation ---
st.header("2. Data Loading and Preparation")

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
        # Lembre-se que o resíduo 'abacaxi' está categorizado no grupo 'Resíduos Agrícolas'.
        dados_meta_analysis = prepare_for_meta_analysis(dados_grupos)

        if dados_meta_analysis.empty:
            st.warning("Not enough data to perform meta-analysis after filtering and preparation.")
        else:
            st.success(f"Data prepared for meta-analysis. {len(dados_meta_analysis)} records available.")
            st.subheader("Prepared Data Sample:")
            st.dataframe(dados_meta_analysis.head()) # Display the head of the DataFrame
    else:
        st.error("Could not load or process data from 'csv.csv'. Please check the file format and ensure it uses ';' as a delimiter and '.' as a decimal separator.")
else:
    st.error("❌ File 'csv.csv' not found in the same directory as 'app.py'. Please place it there and reload the app.")

# Place the markdown horizontal line within st.markdown()
st.markdown("---")

# --- Renumbered Section: 3. Meta-Analysis Modeling and Visualization ---
st.header("3. Meta-Analysis Modeling and Visualization")

if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2: # At least 2 records needed for models

    st.markdown("Performing meta-analysis for different models:")

    # 3.1. Model by residue type
    st.subheader("3.1. Model by Residue Type")
    residue_model_summary_df, residue_coeff_plot_fig = run_meta_analysis_and_plot(
        dados_meta_analysis,
        model_type="Residue",
        plot_title="Effect of Residues on Vermicompost Quality"
    )
    if residue_coeff_plot_fig:
        st.pyplot(residue_coeff_plot_fig)
        st.write("Model Summary:")
        st.dataframe(residue_model_summary_df)

    # 3.2. Model by variable
    st.subheader("3.2. Model by Variable")
    variable_model_summary_df, _ = run_meta_analysis_and_plot(
        dados_meta_analysis,
        model_type="Variable",
        plot_title="Effect of Variables on Vermicompost Quality"
    )
    st.write("Model Summary:")
    st.dataframe(variable_model_summary_df)

    # 3.3. Interaction model (Residue × Variable)
    st.subheader("3.3. Interaction Model (Residue × Variable)")
    interaction_model_summary_df, _ = run_meta_analysis_and_plot(
        dados_meta_analysis,
        model_type="Interaction",
        plot_title="Interaction Effect of Residue and Variable on Vermicompost Quality"
    )
    st.write("Model Summary:")
    st.dataframe(interaction_model_summary_df)

    # --- Save Results ---
    # Removed the header and saving logic.
    # st.header("4. Save Results")
    # try:
    #    # Saving results (models and data)
    #    with open("meta_analysis_results.pkl", "wb") as f:
    #        pickle.dump({
    #             "Residue_Model_Summary": residue_model_summary_df,
    #            "Variable_Model_Summary": variable_model_summary_df,
    #            "Interaction_Model_Summary": interaction_model_summary_df,
    #             "Data": dados_meta_analysis
    #         }, f)
    #    st.success("Meta-analysis results saved successfully to `meta_analysis_results.pkl`.")
    # except Exception as e:
    #    st.error(f"Error saving results: {e}")

else:
    st.info("Please ensure 'csv.csv' is in the same directory as 'app.py' and processed successfully (with at least 2 records) to proceed with the analysis.")

# Place the markdown horizontal line within st.markdown()
st.markdown("---")

# --- Renumbered Section: 4. Diagnostic and Additional Analyses ---
st.header("4. Diagnostic and Additional Analyses")

if not dados_meta_analysis.empty and len(dados_meta_analysis) >= 2:
    st.markdown("Further analyses and diagnostic plots:")

    # 4.1. Analysis by specific variable (e.g., TOC, N, pH, EC)
    st.subheader("4.1. Detailed Analysis for Key Variables")
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

    # 4.2. Diagnostics (Funnel Plot for Publication Bias)
    st.subheader("4.2. Funnel Plot for Publication Bias")
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
