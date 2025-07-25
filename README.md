# Vermicompost Meta-Analysis Streamlit App

This repository hosts a Streamlit application designed for performing a meta-analysis on the effects of different residues on vermicompost quality. 

## Features

- Upload and process CSV data files
- Filter irrelevant treatments
- Define control and treatment groups
- Run three types of meta-analysis models:
  - By residue type
  - By variable
  - Interaction between residue and variable
- Generate visualizations:
  - Coefficient plots
  - Forest plots
  - Funnel plots
- Analyze specific important variables (TOC, N, pH, EC)

## Requirements

- Python 3.7+
- Packages listed in requirements.txt

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

## Data Format

The CSV file should contain the following columns:
- Variable: The measured variable (e.g., pH, EC, TOC)
- Study: The study/source of the data
- Treatment: The treatment applied
- Mean: The mean value
- Std Dev: The standard deviation
- Unit: The unit of measurement
- Original Unit: The original unit if conversion was needed
- Notes: Any additional notes

## Project Structure
