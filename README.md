# Vermicompost Meta-Analysis Streamlit App

This repository hosts a Streamlit application designed for performing a meta-analysis on the effects of different residues on vermicompost quality. The application automatically loads data from 'csv.csv' and performs comprehensive meta-analyses with visualization capabilities.

## Key Features
- Automated data loading from 'csv.csv'
- Three analytical models:
  - Residue type effects
  - Variable effects
  - Residue × Variable interactions
- Publication-quality visualizations:
  - Forest plots
  - Funnel plots
  - Coefficient plots
- Academic English output suitable for publication

## Project Structure
- `app.py`: Main Streamlit application
- `meta_analysis.py`: Core analysis functions
- `csv.csv`: Input data file (must be present in root directory)
