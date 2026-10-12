"""Standalone precompute for the Phase B2.5 one-coin (one-pair-per-coin) constraint ablation
(`WF1_NO_OVERLAP_CONSTRAINT`; see `stat_arb.ablation`).

Mirrors `scripts/run_selection_funnel.py`'s convention: reuses the exact same WF1 baseline
pipeline (`run_wf1_selection_pipeline`, `run_wf1_full`) for the control, and only the
selection-stage call for the ablation differs (`run_no_overlap_selection_pipeline`, which
disables `enforce_one_pair_per_coin`). Backtesting, portfolio construction, and reporting reuse
`run_wf1_full`/`evaluate_coin_netted_portfolio`/`standardized_wf1_summary` verbatim -- nothing
here re-simulates a trade or re-derives a metric those functions already compute.

Run from the project root:  python scripts/run_no_overlap_ablation.py
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))
sys.path.insert(0, _PROJECT_ROOT)

import json

import pandas as pd

from config.research_config import CONFIG, WF_CONFIG, MODEL_IDS
from stat_arb.data import build_walkforward_universe
from stat_arb.walkforward import generate_rebalance_dates, run_wf1_selection_pipeline, run_wf1_full
from stat_arb.ablation import run_no_overlap_selection_pipeline, coin_cluster_pair_table, coin_cluster_return_correlation
from stat_arb.cache import wf_cache_fingerprint, cache_path
from stat_arb.pair_selection import select_pairs_at_rebalance
from stat_arb.portfolio import coin_concentration_diagnostics
from stat_arb.reporting import (standardized_wf1_summary, pair_level_contribution,
                                 active_day_benchmark_stats, active_mask_from_bar_frame)
from stat_arb.metrics import aggregate_daily_returns, build_btc_daily_benchmark


def main():
    os.chdir(_PROJECT_ROOT)

    WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
    REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                                WF_CONFIG['REBALANCE_MONTHS_PRIMARY'], WF_CONFIG['FORMATION_MONTHS'])
    WF1_REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]
    coins = WF_PX.columns.tolist()
    max_pairs = WF_CONFIG['MAX_PAIRS_PER_REBALANCE']

    # --- Baseline WF1 (control, unchanged) -----------------------------------------------------
    wf1_extra = f"max_pairs={max_pairs},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}"
    wf1_sel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                       wf_config=WF_CONFIG, extra=wf1_extra)
    WF1_SELECTIONS = run_wf1_selection_pipeline(WF1_REBALANCE_DATES, WF_PX, coins, WF_CONFIG,
                                                 cache_path('wf1_selections', wf1_sel_fp), max_pairs=max_pairs)
    WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES = run_wf1_full(
        WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    wf1_row = standardized_wf1_summary(MODEL_IDS['WF1'], WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES,
                                        WF_PX, CONFIG['OOS_START'], CONFIG['OOS_END'])

    # --- Ablation: WF1 with the one-pair-per-coin constraint disabled --------------------------
    no_overlap_extra = f"{wf1_extra},no_overlap_constraint"
    no_overlap_sel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                              wf_config=WF_CONFIG, extra=no_overlap_extra)
    NO_OVERLAP_SELECTIONS = run_no_overlap_selection_pipeline(
        WF1_REBALANCE_DATES, WF_PX, coins, WF_CONFIG, cache_path('wf1_no_overlap_selections', no_overlap_sel_fp),
        max_pairs=max_pairs)
    NO_OVERLAP_FULL_FRAME, NO_OVERLAP_ALL_RESULTS, NO_OVERLAP_ALL_TRADES = run_wf1_full(
        NO_OVERLAP_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    no_overlap_row = standardized_wf1_summary(MODEL_IDS['WF1_NO_OVERLAP'], NO_OVERLAP_FULL_FRAME,
                                               NO_OVERLAP_ALL_RESULTS, NO_OVERLAP_ALL_TRADES, WF_PX,
                                               CONFIG['OOS_START'], CONFIG['OOS_END'])

    for d in WF1_REBALANCE_DATES:
        print(f"{d.date()}: baseline selected {len(WF1_SELECTIONS[d])} pair(s), "
              f"no-overlap selected {len(NO_OVERLAP_SELECTIONS[d])} pair(s)")

    # --- Active-day BTC beta, both models --------------------------------------------------------
    def _active_day_beta(full_frame):
        daily_ret, _ = aggregate_daily_returns(full_frame['net_ret'])
        active_mask = active_mask_from_bar_frame(full_frame['active_flag']).reindex(daily_ret.index, fill_value=False)
        btc_daily = build_btc_daily_benchmark(daily_ret.index, WF_PX)
        return active_day_benchmark_stats(daily_ret, active_mask, btc_daily)

    wf1_active_bench = _active_day_beta(WF1_FULL_FRAME)
    no_overlap_active_bench = _active_day_beta(NO_OVERLAP_FULL_FRAME)

    # --- Concentration diagnostics, no-overlap ablation only ------------------------------------
    # A dedicated coin-netted portfolio spanning the FULL evaluation window's active blocks
    # (built exactly like `evaluate_coin_netted_portfolio` -- reused verbatim -- but requires a
    # single contiguous set of pairs; NO_OVERLAP_ALL_RESULTS spans several rebalance blocks
    # with different pair sets, so concentration diagnostics are computed per BLOCK below,
    # then combined, rather than assuming one global pair list across the whole OOS window.)
    from stat_arb.portfolio import evaluate_coin_netted_portfolio
    block_bounds = list(zip(WF1_REBALANCE_DATES, WF1_REBALANCE_DATES[1:] + [pd.Timestamp(CONFIG['OOS_END'])]))
    concentration_by_block = {}
    for block_start, block_end in block_bounds:
        block_end_adj = block_end - pd.Timedelta(hours=4) if block_end != pd.Timestamp(CONFIG['OOS_END']) else block_end
        block_pairs = [s['pair'] for s in NO_OVERLAP_SELECTIONS.get(block_start, [])]
        if not block_pairs:
            continue
        pair_results = {s['pair']: NO_OVERLAP_ALL_RESULTS[(block_start, s['pair'])]
                         for s in NO_OVERLAP_SELECTIONS[block_start]}
        portfolio, _ = evaluate_coin_netted_portfolio(
            pair_results, block_pairs, WF_PX.loc[:block_end_adj], CONFIG['COST_PER_LEG_BPS'],
            block_start, block_end_adj, CONFIG['BARS_PER_YEAR'])
        concentration_by_block[block_start] = coin_concentration_diagnostics(portfolio)

    # Report the richest block (2024-01-01) explicitly; other blocks select 0-1 pairs each
    # (see funnel findings) so concentration is not a live phenomenon there.
    richest_block = WF1_REBALANCE_DATES[0]
    conc_2024_01 = concentration_by_block.get(richest_block)

    # --- Coin-cluster diagnostics at the richest rebalance date ---------------------------------
    _, _, pair_trail_2024_01, _ = select_pairs_at_rebalance(
        richest_block, WF_PX, coins, WF_CONFIG, max_pairs=max_pairs,
        collect_diagnostics=True, enforce_one_pair_per_coin=False)
    fully_qualifying = pair_trail_2024_01[pair_trail_2024_01['fully_qualifying']]
    coin_counts = pd.Series([c for p in fully_qualifying['pair'] for c in p]).value_counts()
    cluster_coin = coin_counts.index[0] if len(coin_counts) else None
    print(f'Most-shared coin among fully-qualifying pairs at {richest_block.date()}: '
          f'{cluster_coin} ({coin_counts.iloc[0] if len(coin_counts) else 0} pairs)')

    cluster_table = pd.DataFrame()
    cluster_corr = pd.DataFrame()
    if cluster_coin is not None:
        cluster_table = coin_cluster_pair_table(richest_block, cluster_coin, pair_trail_2024_01,
                                                  NO_OVERLAP_SELECTIONS, NO_OVERLAP_ALL_TRADES)
        cluster_corr = coin_cluster_return_correlation(richest_block, cluster_coin, NO_OVERLAP_SELECTIONS,
                                                          NO_OVERLAP_ALL_RESULTS)

    pnl_contrib = pair_level_contribution(NO_OVERLAP_ALL_TRADES, NO_OVERLAP_ALL_RESULTS, CONFIG['COST_PER_LEG_BPS'])
    cluster_pnl_contrib = pd.DataFrame()
    if cluster_coin is not None and len(pnl_contrib):
        richest_block_str = str(richest_block.date())
        cluster_pnl_contrib = pnl_contrib[
            pnl_contrib['pair_instance'].str.contains(cluster_coin, na=False) &
            pnl_contrib['pair_instance'].str.contains(richest_block_str, na=False)]

    # --- Write outputs ----------------------------------------------------------------------------
    tables_dir = os.path.join(_PROJECT_ROOT, 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    summary = {
        'wf1_baseline': wf1_row, 'wf1_no_overlap': no_overlap_row,
        'wf1_baseline_active_day_beta': wf1_active_bench,
        'wf1_no_overlap_active_day_beta': no_overlap_active_bench,
        'cluster_coin_at_richest_block': cluster_coin,
        'richest_block': str(richest_block.date()),
    }
    with open(os.path.join(tables_dir, 'wf1_no_overlap_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    selected_pairs_out = {
        str(d.date()): [{'x': s['x'], 'y': s['y'], 'pvalue_adj': s['pvalue_adj']} for s in sels]
        for d, sels in NO_OVERLAP_SELECTIONS.items()
    }
    with open(os.path.join(tables_dir, 'wf1_no_overlap_selected_pairs.json'), 'w') as f:
        json.dump(selected_pairs_out, f, indent=2, default=str)

    coin_exposure_out = {}
    if conc_2024_01 is not None:
        coin_exposure_out = {
            'richest_block': str(richest_block.date()),
            'max_gross_exposure_by_coin': conc_2024_01['max_gross_exposure_by_coin'].to_dict(),
            'max_abs_net_exposure_by_coin': conc_2024_01['max_abs_net_exposure_by_coin'].to_dict(),
            'avg_gross_exposure_by_coin_active': conc_2024_01['avg_gross_exposure_by_coin_active'].to_dict(),
            'avg_abs_net_exposure_by_coin_active': conc_2024_01['avg_abs_net_exposure_by_coin_active'].to_dict(),
            'max_share_of_total_gross_single_coin': conc_2024_01['max_share_of_total_gross_single_coin'],
            'top5_coins_by_max_gross': conc_2024_01['top5_coins_by_max_gross'].to_dict(),
            'top5_coins_by_max_abs_net': conc_2024_01['top5_coins_by_max_abs_net'].to_dict(),
        }
        if cluster_coin is not None and cluster_coin in conc_2024_01['gross_exposure_pct_by_coin'].columns:
            gross_series = conc_2024_01['gross_exposure_pct_by_coin'][cluster_coin]
            net_series = conc_2024_01['net_exposure_pct_by_coin'][cluster_coin]
            largest = conc_2024_01['largest_gross_coin_by_bar']
            active_bars = (largest.notna()).sum()
            coin_exposure_out[f'{cluster_coin}_gross_exposure_pct_timeseries'] = {
                str(ts): v for ts, v in gross_series.items() if v != 0
            }
            coin_exposure_out[f'{cluster_coin}_net_exposure_pct_timeseries'] = {
                str(ts): v for ts, v in net_series.items() if v != 0
            }
            coin_exposure_out[f'pct_active_bars_{cluster_coin}_is_largest_gross'] = (
                float((largest == cluster_coin).sum() / active_bars) if active_bars > 0 else float('nan')
            )
    with open(os.path.join(tables_dir, 'wf1_no_overlap_coin_exposure.json'), 'w') as f:
        json.dump(coin_exposure_out, f, indent=2, default=str)

    pair_diagnostics_out = {
        'cluster_coin': cluster_coin, 'richest_block': str(richest_block.date()),
        'cluster_pair_table': cluster_table.to_dict(orient='records'),
        'cluster_return_correlation': cluster_corr.to_dict() if len(cluster_corr) else {},
        'cluster_pnl_contribution': cluster_pnl_contrib.to_dict(orient='records'),
    }
    with open(os.path.join(tables_dir, 'wf1_no_overlap_pair_diagnostics.json'), 'w') as f:
        json.dump(pair_diagnostics_out, f, indent=2, default=str)

    print('Wrote results/tables/wf1_no_overlap_summary.json, wf1_no_overlap_selected_pairs.json, '
          'wf1_no_overlap_coin_exposure.json, wf1_no_overlap_pair_diagnostics.json')
    print('\n--- Baseline vs. no-overlap summary ---')
    print(pd.DataFrame([wf1_row, no_overlap_row]).set_index('model')[
        ['daily_sharpe_252', 'cum_net_return', 'n_trades', 'n_pairs_traded', 'time_in_market_pct',
         'turnover', 'total_costs_pct', 'btc_beta']].round(4))


if __name__ == '__main__':
    main()
