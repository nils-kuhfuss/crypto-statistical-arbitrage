"""Reusable statistical primitives: OLS, spread, Engle-Granger, BH/FDR, half-life, crossings,
rolling beta, benchmark regression.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 5, 6, 17, 25). Two functions
here (`run_engle_granger_tests`, `apply_bh_fdr`) consolidate logic that the original notebook
implemented twice, independently, in nearly identical form: once inline in Section 6 (S0's
static screen) and once inside `screen_pairs_at_date` (the walk-forward point-in-time screen).
Both call sites now call these same two functions -- this does not change any statistical
result (both original implementations already performed the identical steps), it only removes
the duplicate implementation, per REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md Section 8's explicit
requirement not to maintain two independent implementations of Engle-Granger/FDR. This is
verified byte-for-byte against the Golden Master in `results/refactor_validation.md`.
"""
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from config.research_config import CONFIG, WF_CONFIG


def fit_static_ols(y: pd.Series, x: pd.Series):
    """Fit Y = alpha + beta*X + eps via OLS. y is the dependent variable, x the independent.

    Point-in-time: uses only the observations present in `y`/`x` as passed by the caller --
    callers are responsible for slicing to a formation-only window before calling this.
    """
    d = pd.concat([x, y], axis=1).dropna()
    d.columns = ['x', 'y']
    design = sm.add_constant(d['x'].values)
    model = sm.OLS(d['y'].values, design).fit()
    alpha, beta = model.params[0], model.params[1]
    return alpha, beta, model


def compute_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    """spread = Y - alpha - beta*X. Single definition used throughout the project."""
    return y - alpha - beta * x


def sanity_check_residual_mean(y: pd.Series, x: pd.Series, alpha: float, beta: float,
                                tol: Optional[float] = None, label: str = '') -> bool:
    """Assert the in-sample OLS residual has ~zero mean, catching a swapped-variable
    regression (dependent/independent reversed), which biases the spread mean away from
    zero without necessarily failing the cointegration test itself. Used live as a
    correctness guard in the S0/S1 orchestration, not only as a test."""
    spread = compute_spread(y, x, alpha, beta)
    mean, std = spread.mean(), spread.std()
    if tol is None:
        tol = 0.1 * std  # mean should be small relative to the spread's own dispersion
    ok = abs(mean) < tol
    print(f'[{"OK" if ok else "FAILED"}] {label} residual mean={mean:.5f}  std={std:.5f}  (tol={tol:.5f})')
    assert ok, f'Residual mean sanity check failed for {label}: mean={mean:.5f}, std={std:.5f}'
    return ok


def compute_half_life(spread: pd.Series) -> float:
    """OU half-life via AR(1): spread_t - spread_{t-1} = k*(spread_{t-1} - mean) + eps,
    half_life = -ln(2)/ln(1+k). A finite half-life additionally requires 0 < 1+k < 1: k >= 0
    implies no reversion, and 1+k <= 0 implies an oscillating or explosive AR(1) root, which
    is not interpretable as a decaying mean-reversion timescale."""
    s = spread.dropna()
    lagged = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lagged = lagged.loc[delta.index]
    design = np.column_stack([np.ones(len(lagged)), lagged.values])
    coef, *_ = np.linalg.lstsq(design, delta.values, rcond=None)
    k = coef[1]
    phi = 1 + k
    if k >= 0 or phi <= 0:
        return np.inf
    return -np.log(2) / np.log(phi)


def count_crossings(spread: pd.Series, level: float = 0.0) -> int:
    """Number of times the spread crosses `level` (mean-reversion cycles)."""
    s = spread.dropna()
    sign = np.sign(s - level)
    return int((sign.diff().abs() == 2).sum())


def run_engle_granger_tests(log_px: pd.DataFrame, pairs, min_obs: int) -> pd.DataFrame:
    """Directional Engle-Granger cointegration test for every (c1, c2) pair in `pairs`, both
    regression directions. Point-in-time: uses only the rows already present in `log_px` --
    callers are responsible for passing a formation/trailing-window-only panel.

    Consolidates the identical EG-test loop previously duplicated between S0's Section 6
    orchestration and the walk-forward `screen_pairs_at_date` (both called `coint()` both
    directions on a `dropna()`-aligned pair slice with the same `min_obs` gate).

    Returns a DataFrame with one row per (pair, direction) tested: columns
    ['pair', 'y', 'x', 'eg_stat', 'eg_pvalue', 'n_obs'].
    """
    rows = []
    for c1, c2 in pairs:
        d = log_px[[c1, c2]].dropna()
        if len(d) < min_obs:
            continue
        # coint() regresses its first argument on its second, so calling it both ways
        # checks both candidate (Y, X) assignments for this unordered pair.
        for y_name, x_name in [(c1, c2), (c2, c1)]:
            try:
                eg_stat, eg_pvalue, _ = coint(d[y_name].values, d[x_name].values)
            except Exception:
                continue
            rows.append({
                'pair': tuple(sorted([c1, c2])),
                'y': y_name, 'x': x_name,
                'eg_stat': eg_stat, 'eg_pvalue': eg_pvalue, 'n_obs': len(d),
            })
    return pd.DataFrame(rows)


def apply_bh_fdr(eg_df: pd.DataFrame, alpha: float, fdr_method: str) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction over the full directional-test family in `eg_df`
    (not per-coin, not re-applied after any subsetting). Returns `eg_df` with two columns
    added: 'pvalue_adj' and 'significant_fdr'. Empty input returns empty output unchanged.

    Consolidates the identical `multipletests(...)` call previously duplicated between S0's
    Section 6 orchestration and `screen_pairs_at_date`.
    """
    if len(eg_df) == 0:
        return eg_df
    reject, pvals_adj, _, _ = multipletests(eg_df['eg_pvalue'].values, alpha=alpha, method=fdr_method)
    eg_df = eg_df.copy()
    eg_df['pvalue_adj'] = pvals_adj
    eg_df['significant_fdr'] = reject
    return eg_df


def compute_rolling_beta(x_price: pd.Series, y_price: pd.Series,
                          window_bars: int = WF_CONFIG['WF2_ROLLING_BETA_WINDOW_BARS']) -> pd.Series:
    """Rolling OLS beta of log(y) on log(x) over a trailing window, LAGGED by one bar so the
    beta value at index t is estimated using only log-price data up to and including t-1 --
    it contains no information from the bar at which it will be used for an entry decision.
    """
    log_x, log_y = np.log(x_price), np.log(y_price)
    cov = log_x.rolling(window_bars).cov(log_y)
    var = log_x.rolling(window_bars).var()
    beta_raw = cov / var
    beta_lagged = beta_raw.shift(1)
    return beta_lagged


def benchmark_regression_stats(strategy_daily: pd.Series, btc_daily: pd.Series) -> dict:
    """OLS: strategy_daily = alpha + beta*btc_daily + eps, on the exact-aligned intersection
    of both daily series. Returns correlation, beta, daily/annualized alpha, conventional OLS
    alpha t-stat (primary), Newey-West/HAC alpha t-stat (robustness, clearly separate), and R^2.
    """
    d = pd.concat([strategy_daily.rename('strat'), btc_daily.rename('btc')], axis=1).dropna()
    if len(d) < 10 or d['btc'].std() == 0:
        return {k: np.nan for k in ['n_obs', 'correlation', 'beta', 'daily_alpha',
                                     'ann_alpha', 'alpha_tstat_ols', 'alpha_tstat_hac', 'r_squared']}
    X = sm.add_constant(d['btc'].values)
    y = d['strat'].values
    ols_model = sm.OLS(y, X).fit()
    hac_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': CONFIG['HAC_MAXLAGS']})
    return {
        'n_obs': len(d),
        'correlation': d['strat'].corr(d['btc']),
        'beta': ols_model.params[1],
        'daily_alpha': ols_model.params[0],
        'ann_alpha': ols_model.params[0] * 252,
        'alpha_tstat_ols': ols_model.tvalues[0],
        'alpha_tstat_hac': hac_model.tvalues[0],
        'r_squared': ols_model.rsquared,
    }
