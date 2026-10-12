"""Standalone resumable precompute for the Section 19 pair-stability diagnostic cache.

Reconstructed from `Stat_Arb_cont_res_1.ipynb` (Sections 18.5-19) because the cache file the
notebook expects (`pair_stability_{fingerprint}.pkl`) and its documented generator
(`build/run_stability_screen_v2.py`) were both missing from the project directory. Every
function and constant below is copied verbatim from the notebook cells that define them --
this script does not reimplement or alter any statistical logic, only orchestrates it outside
the notebook (screen is ~1,600+ directional Engle-Granger tests per rebalance date x 11 dates,
~15 min total, hence a resumable script rather than a notebook cell per project convention).

Run from the project root:  python scripts/run_stability_screen.py
"""
import hashlib
import itertools as it
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

# --- CONFIG subset needed by WF_CONFIG (verbatim values from notebook cell 3 / "c004") -------
CONFIG = {
    'FORMATION_START': '2020-01-01',
    'OOS_END': '2026-06-30',
    'BARS_PER_DAY': 6,
    'EG_ALPHA': 0.05,
    'FDR_METHOD': 'fdr_bh',
}

# --- WF_CONFIG (verbatim from notebook cell "1537e188", Section 18.5) ------------------------
WF_CONFIG = {
    'MANUAL_EXCLUDE': ['USDCUSDT', 'FDUSDUSDT'],
    'FORMATION_MONTHS': 12,
    'BARS_PER_DAY': CONFIG['BARS_PER_DAY'],
    'MIN_COVERAGE_FRACTION': 0.90,
    'EG_ALPHA': CONFIG['EG_ALPHA'],
    'FDR_METHOD': CONFIG['FDR_METHOD'],
    'MAX_HALF_LIFE_BARS': 300,
    'MIN_CROSSINGS': 15,
    'REQUIRE_POSITIVE_BETA': True,
    'REBALANCE_MONTHS_PRIMARY': 6,
    'REBALANCE_MONTHS_ROBUSTNESS': 3,
    'TIME_STOP_MULTIPLIER_PRIMARY': 2.5,
}
_expected_bars_per_year = 365.25 * WF_CONFIG['BARS_PER_DAY']
WF_CONFIG['FORMATION_WINDOW_BARS_NOMINAL'] = int(round(_expected_bars_per_year * (WF_CONFIG['FORMATION_MONTHS'] / 12)))
WF_CONFIG['MIN_OBS_FOR_TEST'] = int(round(WF_CONFIG['FORMATION_WINDOW_BARS_NOMINAL'] * WF_CONFIG['MIN_COVERAGE_FRACTION']))


# --- Shared statistical helpers (verbatim from notebook cell "c015" / Section 5) -------------
def fit_static_ols(y, x):
    """Fit Y = alpha + beta*X + eps via OLS. y is the dependent variable, x the independent."""
    d = pd.concat([x, y], axis=1).dropna()
    d.columns = ['x', 'y']
    design = sm.add_constant(d['x'].values)
    model = sm.OLS(d['y'].values, design).fit()
    alpha, beta = model.params[0], model.params[1]
    return alpha, beta, model


def compute_spread(y, x, alpha, beta):
    """spread = Y - alpha - beta*X. Single definition used throughout the notebook."""
    return y - alpha - beta * x


# --- Half-life / crossings (verbatim from notebook cell before "c028", Section 6) ------------
def compute_half_life(spread):
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


def count_crossings(spread, level=0.0):
    """Number of times the spread crosses `level` (mean-reversion cycles)."""
    s = spread.dropna()
    sign = np.sign(s - level)
    return int((sign.diff().abs() == 2).sum())


# --- Point-in-time eligibility + screen (verbatim from notebook cell "23ff3faa", Section 19) -
def check_active_at_rebalance(price_data, coins, rebalance_date, lookback_bars=3):
    """An asset is 'active at rebalance' if it has a valid (non-NaN) price within the last
    `lookback_bars` bars strictly before rebalance_date -- a small tolerance for isolated data
    gaps, not a loophole for genuinely delisted assets (Binance delistings leave a long
    trailing all-NaN tail, e.g. XMRUSDT after 2024-02-20, which this correctly fails).
    """
    check_window = price_data[coins].loc[:rebalance_date - pd.Timedelta(hours=4)].tail(lookback_bars)
    return check_window.notna().any(axis=0)


def screen_pairs_at_date(rebalance_date, price_data, coins, eg_alpha=WF_CONFIG['EG_ALPHA'],
                          min_obs=WF_CONFIG['MIN_OBS_FOR_TEST'], formation_months=WF_CONFIG['FORMATION_MONTHS'],
                          max_half_life_bars=WF_CONFIG['MAX_HALF_LIFE_BARS'],
                          min_crossings=WF_CONFIG['MIN_CROSSINGS'],
                          require_positive_beta=WF_CONFIG['REQUIRE_POSITIVE_BETA']):
    """Point-in-time pair screen, corrected per expert feedback to match S0's own screen
    exactly (Section 6) plus true 12-month coverage and active-at-rebalance eligibility.

    All filters use ONLY the trailing `formation_months` of data strictly before
    `rebalance_date`. Returns a dataframe with one row per unordered pair that survives every
    filter at this date (empty if none survive).
    """
    window_start = rebalance_date - pd.DateOffset(months=formation_months)
    window_px = price_data.loc[window_start:rebalance_date - pd.Timedelta(hours=4)]
    assert window_px.index.max() < rebalance_date, 'Leakage: formation window reaches rebalance date.'

    coverage_ok = window_px[coins].notna().sum() >= min_obs
    active_ok = check_active_at_rebalance(price_data, coins, rebalance_date)
    valid_coins = [c for c in coins if coverage_ok.get(c, False) and active_ok.get(c, False)]
    if len(valid_coins) < 2:
        return pd.DataFrame()
    log_px = np.log(window_px[valid_coins])
    pairs = list(it.combinations(valid_coins, 2))

    rows = []
    for c1, c2 in pairs:
        d = log_px[[c1, c2]].dropna()
        if len(d) < min_obs:
            continue
        for y_name, x_name in [(c1, c2), (c2, c1)]:
            try:
                eg_stat, eg_pvalue, _ = coint(d[y_name].values, d[x_name].values)
            except Exception:
                continue
            rows.append({'pair': tuple(sorted([c1, c2])), 'y': y_name, 'x': x_name,
                        'eg_stat': eg_stat, 'eg_pvalue': eg_pvalue, 'n_obs': len(d)})
    if not rows:
        return pd.DataFrame()
    eg_df_t = pd.DataFrame(rows)

    reject, pvals_adj, _, _ = multipletests(eg_df_t['eg_pvalue'].values, alpha=eg_alpha, method=WF_CONFIG['FDR_METHOD'])
    eg_df_t['pvalue_adj'] = pvals_adj
    eg_df_t['significant_fdr'] = reject
    sig_t = eg_df_t[eg_df_t['significant_fdr']].sort_values('pvalue_adj')
    if len(sig_t) == 0:
        return pd.DataFrame()
    sig_t = sig_t.drop_duplicates(subset='pair', keep='first').reset_index(drop=True)

    out_rows = []
    for _, row in sig_t.iterrows():
        y_name, x_name = row['y'], row['x']
        a, b, _ = fit_static_ols(log_px[y_name], log_px[x_name])
        if require_positive_beta and b <= 0:
            continue
        spread = compute_spread(log_px[y_name], log_px[x_name], a, b)
        hl = compute_half_life(spread)
        if not np.isfinite(hl) or hl <= 0 or hl >= max_half_life_bars:
            continue
        crossings = count_crossings(spread)
        if crossings < min_crossings:
            continue
        corr = window_px[y_name].pct_change(fill_method=None).corr(window_px[x_name].pct_change(fill_method=None))
        out_rows.append({
            'rebalance_date': rebalance_date, 'pair': row['pair'], 'y': y_name, 'x': x_name,
            'eg_pvalue': row['eg_pvalue'], 'pvalue_adj': row['pvalue_adj'],
            'return_correlation': corr, 'alpha': a, 'beta': b,
            'spread_std': spread.std(), 'half_life_bars': hl, 'crossings': crossings,
            'n_obs': row['n_obs'],
        })
    return pd.DataFrame(out_rows)


# --- Cache fingerprint (verbatim from notebook cell "1ab02d36", Section 19) -------------------
def wf_cache_fingerprint(*, universe_cols, rebalance_dates, extra=''):
    """Full methodology-aware cache fingerprint (feedback #3): hashes the universe, the
    rebalance-date schedule, AND every WF_CONFIG parameter that affects a walk-forward
    screen's result (coverage rule, EG alpha, FDR method, half-life/crossing/beta filters).
    Changing any one of these automatically produces a different fingerprint, so a stale
    cache can never be silently reused under a new methodology -- it simply won't be found,
    and the cell below recomputes instead of asserting on its absence.
    """
    methodology_src = '|'.join([
        f"min_obs={WF_CONFIG['MIN_OBS_FOR_TEST']}",
        f"coverage_frac={WF_CONFIG['MIN_COVERAGE_FRACTION']}",
        f"formation_months={WF_CONFIG['FORMATION_MONTHS']}",
        f"eg_alpha={WF_CONFIG['EG_ALPHA']}",
        f"fdr_method={WF_CONFIG['FDR_METHOD']}",
        f"max_hl={WF_CONFIG['MAX_HALF_LIFE_BARS']}",
        f"min_crossings={WF_CONFIG['MIN_CROSSINGS']}",
        f"positive_beta={WF_CONFIG['REQUIRE_POSITIVE_BETA']}",
        f"manual_exclude={sorted(WF_CONFIG['MANUAL_EXCLUDE'])}",
        f"extra={extra}",
    ])
    fingerprint_src = (','.join(sorted(universe_cols)) + '|' +
                        '|'.join(str(d.date()) for d in rebalance_dates) + '|' + methodology_src)
    return hashlib.md5(fingerprint_src.encode()).hexdigest()[:10]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    data_raw_dir = os.path.join(project_root, 'data', 'raw')
    data_cache_dir = os.path.join(project_root, 'data', 'cache')
    os.makedirs(data_cache_dir, exist_ok=True)

    with open(os.path.join(data_raw_dir, 'px.pkl'), 'rb') as f:
        _wf_px_raw = pickle.load(f)
    # ret.pkl is not loaded here: screen_pairs_at_date only ever reads price_data (log prices),
    # never ret_data -- see its signature below and stat_arb.pair_selection.screen_pairs_at_date.
    WF_PX = _wf_px_raw.drop(columns=WF_CONFIG['MANUAL_EXCLUDE'])
    print(f'Walk-forward universe: {WF_PX.shape[1]} coins')

    REBALANCE_DATES = pd.date_range('2021-01-01', CONFIG['OOS_END'], freq='6MS', tz=None)
    REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['FORMATION_START']) + pd.DateOffset(months=12)]
    print(f'{len(REBALANCE_DATES)} rebalance dates: {[d.date() for d in REBALANCE_DATES]}')

    fingerprint = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=REBALANCE_DATES)
    cache_file_path = os.path.join(data_cache_dir, f'pair_stability_{fingerprint}.pkl')
    progress_path = os.path.join(data_cache_dir, f'pair_stability_{fingerprint}.progress.pkl')
    print(f'Target cache: {cache_file_path}')

    if os.path.exists(cache_file_path):
        print(f'Cache already exists at {cache_file_path} -- nothing to do.')
        return

    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, 'rb') as f:
            done = pickle.load(f)
        print(f'Resuming from progress file: {len(done)}/{len(REBALANCE_DATES)} dates already computed.')

    for rebalance_date in REBALANCE_DATES:
        if rebalance_date in done:
            continue
        result = screen_pairs_at_date(rebalance_date, WF_PX, WF_PX.columns.tolist())
        done[rebalance_date] = result
        n = 0 if result.empty else len(result)
        print(f'{rebalance_date.date()}: {n} qualifying pairs')
        with open(progress_path, 'wb') as f:
            pickle.dump(done, f)

    non_empty = [df for df in done.values() if not df.empty]
    pair_stability_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    pair_stability_df.to_pickle(cache_file_path)
    print(f'Wrote {cache_file_path}: {pair_stability_df.shape[0]} (pair, date) rows '
          f'across {pair_stability_df["rebalance_date"].nunique() if not pair_stability_df.empty else 0} rebalance dates')
    if os.path.exists(progress_path):
        os.remove(progress_path)


if __name__ == '__main__':
    main()
