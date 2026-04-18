"""
regression_models.py
--------------------
Robust regression checks: heteroscedasticity, VIF, jackknife stability.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def breusch_pagan_test(model) -> dict:
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp = het_breuschpagan(model.resid, model.model.exog)
    return {
        'LM_statistic':   round(bp[0], 4),
        'p_value':        round(bp[1], 4),
        'heteroscedastic': bp[1] < 0.05,
    }


def vif_analysis(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(df[feature_cols].fillna(0))
    vif_data = pd.DataFrame({
        'feature': X.columns,
        'VIF':     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    vif_data['interpretation'] = vif_data['VIF'].apply(
        lambda v: 'OK' if v < 5 else ('Moderate' if v < 10 else 'High Multicollinearity')
    )
    return vif_data[vif_data['feature'] != 'const']


def jackknife_stability(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    model_full = smf.ols(formula, data=data).fit()
    full_params = model_full.params
    results = []
    for county in data['county']:
        subset = data[data['county'] != county]
        try:
            m = smf.ols(formula, data=subset).fit()
            results.append({'excluded': county, **m.params.to_dict()})
        except Exception:
            pass

    loo_df    = pd.DataFrame(results).set_index('excluded')
    stability = {}
    for col in full_params.index:
        if col in loo_df.columns:
            pct_dev = ((loo_df[col] - full_params[col]).abs() / abs(full_params[col]) * 100)
            stability[col] = {
                'full_coeff':  round(full_params[col], 4),
                'loo_mean':    round(loo_df[col].mean(), 4),
                'max_pct_dev': round(pct_dev.max(), 2),
                'stable':      pct_dev.max() < 20,
            }
    return pd.DataFrame(stability).T


def robust_regression(formula: str, data: pd.DataFrame):
    """OLS with HC3 heteroscedasticity-robust standard errors."""
    return smf.ols(formula, data=data).fit(cov_type='HC3')
