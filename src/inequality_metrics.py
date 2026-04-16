"""
inequality_metrics.py
---------------------
Standard inequality indices for healthcare access analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def gini_coefficient(x: np.ndarray) -> float:
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cumx = np.cumsum(x)
    return (n + 1 - 2 * cumx.sum() / cumx[-1]) / n


def lorenz_curve(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x         = np.sort(x[~np.isnan(x)])
    n         = len(x)
    cum_pop   = np.linspace(0, 1, n + 1)
    cum_share = np.concatenate([[0], np.cumsum(x) / x.sum()])
    return cum_pop, cum_share


def theil_t_index(x: np.ndarray, groups: Optional[np.ndarray] = None) -> dict:
    x  = x[~np.isnan(x)]
    n  = len(x)
    mu = x.mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        t_total = np.nansum((x / mu) * np.log(x / mu)) / n

    result = {'total': t_total, 'between': None, 'within': None}
    if groups is None:
        return result

    groups        = groups[~np.isnan(x)]
    unique_groups = np.unique(groups)
    between = 0.0
    within  = 0.0

    for g in unique_groups:
        mask  = groups == g
        x_g   = x[mask]
        n_g   = len(x_g)
        mu_g  = x_g.mean()
        share = n_g / n
        with np.errstate(divide='ignore', invalid='ignore'):
            between += share * (mu_g / mu) * np.log(mu_g / mu)
            t_g      = np.nansum((x_g / mu_g) * np.log(x_g / mu_g)) / n_g
        within += share * (mu_g / mu) * t_g

    result['between'] = between
    result['within']  = within
    return result


def concentration_ratio(x: np.ndarray, top_pct: float = 0.1) -> float:
    x_sorted = np.sort(x[~np.isnan(x)])[::-1]
    n_top    = max(1, int(len(x_sorted) * top_pct))
    return x_sorted[:n_top].sum() / x_sorted.sum()


def palma_ratio(x: np.ndarray) -> float:
    x_sorted   = np.sort(x[~np.isnan(x)])
    n          = len(x_sorted)
    n_bottom40 = int(n * 0.4)
    n_top10    = max(1, int(n * 0.1))
    return x_sorted[-n_top10:].sum() / x_sorted[:n_bottom40].sum()


def generate_inequality_report(df, value_col='hfd', group_col='region') -> pd.DataFrame:
    x      = df[value_col].values
    groups = df[group_col].values
    gini   = gini_coefficient(x)
    theil  = theil_t_index(x, groups)
    cr10   = concentration_ratio(x, top_pct=0.10)
    palma  = palma_ratio(x)
    cv     = x.std() / x.mean()
    p90p10 = np.percentile(x, 90) / np.percentile(x, 10)

    return pd.DataFrame({
        'Metric': [
            'Gini Coefficient','Theil T (Total)','Theil T (Between)',
            'Theil T (Within)','Concentration Ratio (Top 10%)',
            'Palma Ratio','Coefficient of Variation','P90/P10 Ratio',
        ],
        'Value': [
            round(gini,4), round(theil['total'],4),
            round(theil['between'],4) if theil['between'] else None,
            round(theil['within'],4)  if theil['within']  else None,
            round(cr10,4), round(palma,4), round(cv,4), round(p90p10,4),
        ],
        'Interpretation': [
            f"{'Low' if gini<0.3 else 'Moderate' if gini<0.5 else 'High'} inequality",
            'Total distributional inequality',
            f"{theil['between']/theil['total']*100:.1f}% due to regional differences" if theil['between'] else 'N/A',
            f"{theil['within']/theil['total']*100:.1f}% due to within-region differences" if theil['within'] else 'N/A',
            f"Top 10% hold {cr10*100:.1f}% of total HFD",
            f"Top 10% have {palma:.1f}x more HFD than bottom 40%",
            f"CV = {cv:.2f} ({'high' if cv>0.5 else 'moderate'} dispersion)",
            f"Top decile has {p90p10:.1f}x HFD of bottom decile",
        ]
    })
