"""Missing-data/coverage diagnostics, ADF/I(1) diagnostics -- documentation only, never a filter.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 3, 18.5 "Asset-Level I(1)
Diagnostic").
"""
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .data import _coverage


def coverage_table(ret: pd.DataFrame) -> pd.DataFrame:
    """Per-coin coverage table (first/last valid date, observation count, missing %)."""
    return ret.apply(_coverage).T


def delisted_coins_report(px: pd.DataFrame) -> pd.Series:
    """Coins whose last valid (non-zero) observation falls before the sample's final bar --
    i.e. coins that stopped trading before the data cutoff. Documentation only."""
    last_valid = px.apply(lambda c: c.replace(0, float('nan')).last_valid_index())
    return last_valid[last_valid < px.index[-1]]


def i1_diagnostic_table(price_data: pd.DataFrame, min_obs: int = 500) -> pd.DataFrame:
    """ADF test on log-price levels and on first differences (log returns) for every asset
    with sufficient history. I(1)-compatible = level ADF fails to reject a unit root (p > 0.05,
    i.e. non-stationary in levels) AND the first difference rejects a unit root (p <= 0.05,
    i.e. stationary in differences). One-time, methodological documentation check -- does not
    filter or select any asset."""
    import numpy as np
    rows = []
    for coin in price_data.columns:
        s = price_data[coin].dropna()
        if len(s) < min_obs:
            continue
        log_s = np.log(s)
        log_diff = log_s.diff().dropna()
        try:
            level_stat, level_p, *_ = adfuller(log_s.values, autolag='AIC')
            diff_stat, diff_p, *_ = adfuller(log_diff.values, autolag='AIC')
        except Exception:
            continue
        i1_compatible = (level_p > 0.05) and (diff_p <= 0.05)
        rows.append({'asset': coin, 'level_adf_pvalue': level_p, 'diff_adf_pvalue': diff_p,
                     'i1_compatible': 'Yes' if i1_compatible else 'No', 'n_obs': len(s)})
    return pd.DataFrame(rows).sort_values('asset').reset_index(drop=True)
