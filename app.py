import streamlit as st
import meta_analysis as ma

st.set_page_config(page_title="🌱 Vermicompost Meta-Analysis", layout="wide")

st.title("🌱 Vermicompost Meta-Analysis: Effect of Different Residues")
st.markdown("""
This application performs a meta-analysis to evaluate the effect of different residues on vermicompost quality, using pre-loaded example data from the repository.
""")

# 1. Loading Data
st.header("1. Loading Data")
st.markdown("**PRISMA 2020 Flow Diagram**  \nView Study Selection Process")

# Caminho do CSV no GitHub
file_path = "https://raw.githubusercontent.com/loopvinyl/metaanalysis/main/data/csv.csv"
st.write(f"Loading data from: `{file_path}`")

# Carrega os dados
dados_preparados = ma.load_and_prepare_data(file_path)

if dados_preparados.empty:
    st.error("Could not load or process data from the GitHub CSV. Please check the file format or content.")
    st.stop()
else:
    st.success("Data loaded and processed successfully!")
    st.dataframe(dados_preparados)

# 2. Meta-analysis section
st.header("2. Run Meta-Analysis Models & Generate Plots")

variaveis = dados_preparados['Variable'].unique()
tratamentos = dados_preparados['Treatment'].unique()

variavel_escolhida = st.selectbox("Select variable", variaveis)
tratamento_escolhido = st.selectbox("Select treatment group", tratamentos)
modelo = st.radio("Choose meta-analysis model", ['random', 'fixed'])

resultado, erro = ma.rodar_meta_analise(dados_preparados, variavel_escolhida, tratamento_escolhido, modelo)

if erro:
    st.error(erro)
else:
    st.success("Meta-analysis completed.")
    efeito, variancia = resultado[0], resultado[1]
    st.markdown(f"**Combined Effect (SMD):** `{efeito:.3f}`")
    st.markdown(f"**Variance:** `{variancia:.4f}`")

    fig = ma.gerar_forest_plot(resultado, resultado.effects, resultado.variances, f"{variavel_escolhida} ({modelo.title()} Effects)")
    st.pyplot(fig)

# 3. Additional Plots (placeholder)
st.header("3. Additional Plots")
if dados_preparados.empty:
    st.warning("Cannot generate additional plots. Data was not loaded or processed correctly.")
else:
    st.info("Additional visualizations coming soon!")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit and Python for meta-analysis of vermicompost quality.")
