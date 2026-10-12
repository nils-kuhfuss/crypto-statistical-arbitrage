"""Reconcile the new src/stat_arb pipeline against results/golden_master.json, via direct
function calls (no notebook/kernel overhead) -- the fast-path validation companion to actually
Restart & Run All-ing the four notebooks. Prints a pass/fail report per field per model.

Run from the project root: python scripts/validate_against_golden_master.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from config.research_config import CONFIG, WF_CONFIG, MAX_HALF_LIFE_BARS, MIN_CROSSINGS, S1_TIME_STOP_MULTIPLIER
from stat_arb.data import load_raw_prices, load_raw_returns, filter_universe, split_samples, build_walkforward_universe
from stat_arb.pair_selection import run_static_pair_screen, compute_pair_time_stops
from stat_arb.execution import run_model_on_pair, run_s1_universe
from stat_arb.reporting import standardized_model_summary, standardized_wf1_summary
from stat_arb.walkforward import generate_rebalance_dates, run_wf1_selection_pipeline, run_wf1_full, run_wf2_full
from stat_arb.cache import wf_cache_fingerprint, cache_path

TOL = 1e-6

with open('results/golden_master.json') as f:
    GOLDEN = json.load(f)


def compare_row(model_id, computed_row, golden_row):
    print(f'\n=== {model_id} ===')
    all_ok = True
    for key, golden_val in golden_row.items():
        if key == 'model':
            continue
        computed_val = computed_row.get(key)
        if computed_val is None:
            print(f'  [MISSING] {key}')
            all_ok = False
            continue
        if isinstance(golden_val, (int, float)) and golden_val is not None:
            if (isinstance(golden_val, float) and np.isnan(golden_val)) or golden_val != golden_val:
                ok = computed_val != computed_val or (isinstance(computed_val, float) and np.isnan(computed_val))
            else:
                ok = abs(float(computed_val) - float(golden_val)) < max(TOL, abs(golden_val) * 1e-6)
        else:
            ok = computed_val == golden_val
        status = 'OK' if ok else 'MISMATCH'
        if not ok:
            all_ok = False
            print(f'  [{status}] {key}: computed={computed_val!r} golden={golden_val!r}')
    if all_ok:
        print('  ALL FIELDS MATCH')
    return all_ok


def main():
    px_raw = load_raw_prices()
    ret_raw = load_raw_returns()
    px, ret, dropped = filter_universe(px_raw, ret_raw, CONFIG)
    formation_px, validation_px, oos_px = split_samples(px, CONFIG)

    log_formation_all = np.log(formation_px)
    candidate_pairs = run_static_pair_screen(log_formation_all, formation_px.columns.tolist(), CONFIG,
                                              MAX_HALF_LIFE_BARS, MIN_CROSSINGS)
    FROZEN_PAIRS = [(row['x'], row['y']) for _, row in candidate_pairs.iterrows()]
    print(f'FROZEN_PAIRS: {len(FROZEN_PAIRS)} pairs')
    golden_s0_pairs = [tuple(p) for p in GOLDEN['S0_STATIC_BASELINE']['selected_pairs']]
    print('FROZEN_PAIRS match golden selected_pairs:', set(FROZEN_PAIRS) == set(golden_s0_pairs))

    FROZEN_MODEL = {
        'pairs': FROZEN_PAIRS, 'model': 'S',
        'rolling_window_days': CONFIG['ROLLING_WINDOW_DAYS'], 'rolling_window_bars': CONFIG['ROLLING_WINDOW_BARS'],
        'entry_z': CONFIG['ENTRY_Z'], 'exit_z': CONFIG['EXIT_Z'], 'hard_stop_z': CONFIG['HARD_STOP_Z'],
        'time_stop_bars': CONFIG['TIME_STOP_BARS'], 'cost_per_leg_bps': CONFIG['COST_PER_LEG_BPS'],
    }
    px_oos_engine = px.loc[:CONFIG['OOS_END']]
    ret_oos_engine = ret.loc[:CONFIG['OOS_END']]
    oos_results, oos_trades = {}, {}
    for x_name, y_name in FROZEN_MODEL['pairs']:
        result, trades = run_model_on_pair(
            x_name, y_name, FROZEN_MODEL['model'], FROZEN_MODEL['rolling_window_bars'],
            FROZEN_MODEL['entry_z'], FROZEN_MODEL['exit_z'], FROZEN_MODEL['hard_stop_z'],
            FROZEN_MODEL['time_stop_bars'], FROZEN_MODEL['cost_per_leg_bps'],
            px_oos_engine, ret_oos_engine, CONFIG, state_start=CONFIG['OOS_START'])
        result = result.loc[CONFIG['OOS_START']:CONFIG['OOS_END']]
        oos_results[(x_name, y_name)] = result
        oos_trades[(x_name, y_name)] = trades

    S0_ROW, S0_PORTFOLIO = standardized_model_summary(
        'S0_STATIC_BASELINE', oos_results, FROZEN_MODEL['pairs'], px_oos_engine,
        FROZEN_MODEL['cost_per_leg_bps'], CONFIG['OOS_START'], CONFIG['OOS_END'], CONFIG['BARS_PER_YEAR'],
        trades_dict=oos_trades)
    s0_ok = compare_row('S0_STATIC_BASELINE', S0_ROW, GOLDEN['S0_STATIC_BASELINE']['summary_row'])
    s0_equity_ok = abs(S0_PORTFOLIO['frame']['equity'].iloc[-1] - GOLDEN['S0_STATIC_BASELINE']['final_equity']) < 1e-6
    print(f'  final_equity match: {s0_equity_ok} (computed={S0_PORTFOLIO["frame"]["equity"].iloc[-1]:.8f}, golden={GOLDEN["S0_STATIC_BASELINE"]["final_equity"]:.8f})')

    s1_time_stops, s1_ineligible = compute_pair_time_stops(FROZEN_PAIRS, formation_px, S1_TIME_STOP_MULTIPLIER)
    S1_ELIGIBLE_PAIRS = [p for p in FROZEN_PAIRS if p in s1_time_stops]
    S1_OOS_RESULTS, S1_OOS_TRADES = run_s1_universe(S1_ELIGIBLE_PAIRS, s1_time_stops, px_oos_engine, ret_oos_engine, CONFIG, CONFIG['OOS_START'])
    S1_OOS_RESULTS = {k: v.loc[CONFIG['OOS_START']:CONFIG['OOS_END']] for k, v in S1_OOS_RESULTS.items()}
    S1_ROW, S1_PORTFOLIO = standardized_model_summary(
        'S1_STATIC_PAIR_HALFLIFE_STOP', S1_OOS_RESULTS, S1_ELIGIBLE_PAIRS, px_oos_engine,
        CONFIG['COST_PER_LEG_BPS'], CONFIG['OOS_START'], CONFIG['OOS_END'], CONFIG['BARS_PER_YEAR'],
        trades_dict=S1_OOS_TRADES)
    s1_ok = compare_row('S1_STATIC_PAIR_HALFLIFE_STOP', S1_ROW, GOLDEN['S1_STATIC_PAIR_HALFLIFE_STOP']['summary_row'])
    s1_equity_ok = abs(S1_PORTFOLIO['frame']['equity'].iloc[-1] - GOLDEN['S1_STATIC_PAIR_HALFLIFE_STOP']['final_equity']) < 1e-6
    print(f'  final_equity match: {s1_equity_ok}')

    WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
    REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                                WF_CONFIG['REBALANCE_MONTHS_PRIMARY'], WF_CONFIG['FORMATION_MONTHS'])
    WF1_REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]
    wf1_sel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                       wf_config=WF_CONFIG,
                                       extra=f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}")
    wf1_sel_cache_path = cache_path('wf1_selections', wf1_sel_fp)
    WF1_SELECTIONS = run_wf1_selection_pipeline(WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
                                                 wf1_sel_cache_path, max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])

    WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES = run_wf1_full(WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    WF1_ROW = standardized_wf1_summary('WF1_SEMIANNUAL_REESTIMATION', WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES, WF_PX,
                                        CONFIG['OOS_START'], CONFIG['OOS_END'])
    wf1_ok = compare_row('WF1_SEMIANNUAL_REESTIMATION', WF1_ROW, GOLDEN['WF1_SEMIANNUAL_REESTIMATION']['summary_row'])
    wf1_equity = (1 + WF1_FULL_FRAME['net_ret'].fillna(0)).cumprod().iloc[-1]
    wf1_equity_ok = abs(wf1_equity - GOLDEN['WF1_SEMIANNUAL_REESTIMATION']['final_equity']) < 1e-6
    print(f'  final_equity match: {wf1_equity_ok}')

    WF2_FULL_FRAME, WF2_ALL_RESULTS, WF2_ALL_TRADES = run_wf2_full(WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    wf2_row = standardized_wf1_summary('WF2_ROLLING_ENTRY_HEDGE_RATIO', WF2_FULL_FRAME, WF2_ALL_RESULTS, WF2_ALL_TRADES, WF_PX,
                                        CONFIG['OOS_START'], CONFIG['OOS_END'])
    wf2_ok = compare_row('WF2_ROLLING_ENTRY_HEDGE_RATIO', wf2_row, GOLDEN['WF2_ROLLING_ENTRY_HEDGE_RATIO']['summary_row'])
    wf2_equity = (1 + WF2_FULL_FRAME['net_ret'].fillna(0)).cumprod().iloc[-1]
    wf2_equity_ok = abs(wf2_equity - GOLDEN['WF2_ROLLING_ENTRY_HEDGE_RATIO']['final_equity']) < 1e-6
    print(f'  final_equity match: {wf2_equity_ok}')

    print('\n' + '=' * 60)
    all_pass = s0_ok and s0_equity_ok and s1_ok and s1_equity_ok and wf1_ok and wf1_equity_ok and wf2_ok and wf2_equity_ok
    print('OVERALL:', 'ALL MODELS RECONCILE EXACTLY' if all_pass else 'DISCREPANCIES FOUND -- see above')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
