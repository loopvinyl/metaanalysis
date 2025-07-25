# Add metafor-like functionality using statsmodels mixed-effects
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

def run_meta_analysis(dados_meta, model_type="Residue"):
    """Replicates R's metafor rma() function using REML"""
    # Prepare formula based on model type
    if model_type == "Residue":
        groups = dados_meta['Residue']
        formula = 'lnRR ~ C(Residue) - 1'
    elif model_type == "Variable":
        groups = dados_meta['Variable']
        formula = 'lnRR ~ C(Variable) - 1'
    elif model_type == "Interaction":
        groups = dados_meta[['Residue', 'Variable']].astype(str).agg(':'.join, axis=1)
        formula = 'lnRR ~ C(Residue):C(Variable) - 1'
    
    # Use MixedLM to approximate REML estimation
    model = MixedLM.from_formula(
        formula,
        groups=groups,
        re_formula="1",
        vc_formula={"variance": "0 + C(Residue)"} if model_type == "Residue" else None,
        data=dados_meta
    )
    result = model.fit(reml=True)
    
    # Extract key statistics
    tau2 = result.cov_re.iloc[0,0]  # Between-study variance
    I2 = 100 * tau2 / (tau2 + np.mean(dados_meta['var_lnRR']))  # I-squared
    
    # Prepare output matching R's metafor
    output = {
        'beta': result.fe_params,
        'se': result.bse,
        'zval': result.tvalues,
        'pval': result.pvalues,
        'ci.lb': result.conf_int()[0],
        'ci.ub': result.conf_int()[1],
        'tau2': tau2,
        'I2': I2,
        'QE': result.llf,  # Approximation for QE
        'model': result
    }
    
    return output

def generate_forest_plot(dados_meta, model_results):
    """Replicates metafor's forest plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Study-level effects
    for i, (_, row) in enumerate(dados_meta.iterrows()):
        label = f"{row['Residue']} - {row['Variable']}"
        effect = row['lnRR']
        se = np.sqrt(row['var_lnRR'])
        
        ax.errorbar(effect, i, xerr=1.96*se, 
                   fmt='o', color='blue', capsize=5)
        ax.text(0.05, i, label, ha='left', va='center')
    
    # Overall effect
    overall_effect = model_results['beta'].mean()
    ax.axvline(overall_effect, color='red', linestyle='--')
    
    ax.set_yticks(range(len(dados_meta)))
    ax.set_yticklabels([])
    ax.set_xlabel('Log Response Ratio (lnRR)')
    ax.set_title('Forest Plot')
    ax.grid(True, axis='x')
    
    return fig
