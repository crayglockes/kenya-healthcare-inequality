# 🏥 Kenya Healthcare Access Inequality Study

> A quantitative spatial analysis of healthcare access inequality across
> Kenya's 47 counties, applying econometric methods from applied statistics.

**[🗺️ Interactive Map →](https://crayglockes.github.io/kenya-healthcare-inequality)**

## Research Question

Is there systematic spatial and socioeconomic inequality in healthcare access
across Kenyan counties, and what socioeconomic factors predict it?

## Methodology

**Data Sources:** KNBS Census 2019, KHIS 2022, KIHBS 2021/22 Poverty Estimates,
UNDP Human Development Index 2022, KeNHA Road Density 2022

**Analytical Framework:**
1. **Inequality Measurement** — Gini coefficient, Theil T index (between/within decomposition), Lorenz curves
2. **Spatial Analysis** — Global Moran's I, LISA cluster mapping (Queen contiguity, 999 permutations)
3. **Regression Modelling** — OLS log-log specification, interaction model for urban-rural moderation; HC3 robust standard errors
4. **Robustness** — Breusch-Pagan test, VIF multicollinearity analysis, jackknife LOO stability

## Key Findings

| Hypothesis | Result | p-value |
|---|---|---|
| H₀₁: No spatial autocorrelation | **Rejected** | < 0.01 |
| H₀₂: Poverty has no effect on access | **Rejected** | < 0.001 |
| H₀₃: No urban-rural moderation | See report | — |

**Gini Coefficient (HFD):** ~0.42 — comparable to sub-Saharan income inequality  
**LISA Analysis:** Clear deprivation cluster in North Eastern region  
**Theil Decomposition:** ~60% of inequality attributable to between-region variation

## Skills Demonstrated

`Spatial Statistics` `Inequality Measurement` `Econometrics` `GeoPandas`  
`Statsmodels` `Moran's I` `LISA` `OLS` `Robustness Checks` `Folium`
