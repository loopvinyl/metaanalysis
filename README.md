# Vermicompost Meta-Analysis Streamlit App

This Streamlit application performs a meta-analysis to assess the effect of different organic residues on vermicompost quality. It's designed to automatically load and process a dataset, perform meta-analytic modeling, and visualize the results in an academic publication-ready format.

## Features

* **Automatic Data Loading:** Reads `csv.csv` directly from the `data/` directory.
* **Data Preparation:** Cleans and transforms raw data, including renaming columns, converting types, filtering irrelevant treatments, and defining control/treatment groups and residue types.
* **Meta-Analysis Models:**
    * Model by Residue Type
    * Model by Variable
    * Interaction Model (Residue × Variable)
* **Publication-Ready Visualizations:** Generates coefficient plots and funnel plots with academic aesthetics.
* **Detailed Variable Analysis:** Provides specific meta-analyses for key variables like Total Organic Carbon (TOC), Nitrogen (N), pH, and Electrical Conductivity (EC).
* **Result Saving:** Saves meta-analysis model summaries and prepared data to a `.pkl` file for further use.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

Ensure you have Python installed (version 3.8 or higher is recommended).

### Installation

1.  **Clone the Repository (or download files):**
    If you have a Git repository, clone it:
    ```bash
    git clone <your-repository-url>
    cd vermicompost-meta
    ```
    Otherwise, create a directory (e.g., `vermicompost-meta`) and place all the provided files (`app.py`, `meta_analysis.py`, `requirements.txt`, `README.md`, and the `data/` directory with `csv.csv` inside) into it.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Navigate to the project directory (where `requirements.txt` is located) and install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

**Crucial Step:** Place your dataset named `csv.csv` inside a subdirectory named `data/` within your project root.
Your project structure should look like this:
