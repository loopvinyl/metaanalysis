# Update the analysis buttons section to show full statistics:

if st.button("📈 Analyze by Residue Type"):
    st.subheader("Analysis by Residue Type")
    with st.spinner("Calculating..."):
        results = run_meta_analysis(dados_meta_analysis, "Residue")
        
        # Display full model summary like R
        st.markdown("""
        ### Mixed-Effects Model (k = {k}; tau² estimator: REML)
        
        tau² (estimated residual heterogeneity): {tau2:.4f}  
        I² (residual heterogeneity): {I2:.2f}%
        
        Test for Residual Heterogeneity:  
        QE = {QE:.4f}, p-val = {pval:.4f}
        
        Model Results:
        """.format(
            k=len(dados_meta_analysis),
            tau2=results['tau2'],
            I2=results['I2'],
            QE=results['QE'],
            pval=results['pval'].mean()
        ))
        
        # Create a dataframe for coefficients
        coef_df = pd.DataFrame({
            'estimate': results['beta'],
            'se': results['se'],
            'zval': results['zval'],
            'pval': results['pval'],
            'ci.lb': results['ci.lb'],
            'ci.ub': results['ci.ub']
        })
        st.dataframe(coef_df)
        
        # Generate and show plots
        fig = generate_forest_plot(dados_meta_analysis, results)
        st.pyplot(fig)
