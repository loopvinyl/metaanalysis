import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
# import pickle # No longer needed if save results section is removed

from meta_analysis import (
    load_and_prepare_data,
    filter_irrelevant_treatments,
    define_groups_and_residues,
    prepare_for_meta_analysis,
    run_meta_analysis_and_plot,
    # generate_forest_plot, # Not directly used in the current app.py structure for a separate button
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

st.markdown("""
Para compilar os dados para esta meta-análise, realizamos uma busca sistemática nas bases de dados **Scopus**, **Web of Science** e **PubMed**. A estratégia de busca utilizada foi:
`"vermicomposting" AND ("characterization of vermicompost" OR "Chemical and physical variables" OR "Physico-chemical characterization" OR "Nutrient dynamics")`

A identificação e seleção dos estudos seguiram o seguinte fluxo:
