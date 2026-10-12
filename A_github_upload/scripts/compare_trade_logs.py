"""Direct trade-log comparison against results/golden_master_trades/*.json -- entry/exit
timestamps, directions, and returns, per REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md Section 12
checklist item "Compare trade logs" (distinct from the summary-row field comparison in
scripts/validate_against_golden_master.py)."""
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
from stat_arb.walkforward import generate_rebalance_dates, run_wf1_selection_pipeline, run_wf1_full, run_wf2_full
from stat_arb.cache import wf_cache_fingerprint, cache_path

KEY_FIELDS = ['direction', 'entry_time', 'exit_time', 'exit_reason', 'duration_bars']


def normalize_trades(trades_list):
    out = []
    for t in trades_list:
        row = {k: t.get(k) for k in KEY_FIELDS}
        if isinstance(row['entry_time'], str):
            row['entry_time'] = pd.Timestamp(row['entry_time'])
        if isinstance(row['exit_time'], str):
            row['exit_time'] = pd.Timestamp(row['exit_time'])
        out.append(row)
    return sorted(out, key=lambda r: (str(r['entry_time']), str(r['direction'])))


def compare(model_id, computed_trades_dict, golden_path):
    with open(golden_path) as f:
        golden_dict = json.load(f)
    computed_flat = normalize_trades([t for trades in computed_trades_dict.values() for t in trades])
    golden_flat = normalize_trades([t for trades in golden_dict.values() for t in trades])
    print(f'{model_id}: computed {len(computed_flat)} trades, golden {len(golden_flat)} trades')
    if len(computed_flat) != len(golden_flat):
        print('  MISMATCH: trade count differs')
        return False
    all_ok = True
    for i, (c, g) in enumerate(zip(computed_flat, golden_flat)):
        for k in KEY_FIELDS:
            if str(c[k]) != str(g[k]):
                print(f'  MISMATCH trade {i} field {k}: computed={c[k]!r} golden={g[k]!r}')
                all_ok = False
    print('  ALL TRADES MATCH' if all_ok else '  DISCREPANCIES FOUND')
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

    px_oos_engine = px.loc[:CONFIG['OOS_END']]
    ret_oos_engine = ret.loc[:CONFIG['OOS_END']]
    oos_trades = {}
    for x_name, y_name in FROZEN_PAIRS:
        _, trades = run_model_on_pair(
            x_name, y_name, 'S', CONFIG['ROLLING_WINDOW_BARS'], CONFIG['ENTRY_Z'], CONFIG['EXIT_Z'],
            CONFIG['HARD_STOP_Z'], CONFIG['TIME_STOP_BARS'], CONFIG['COST_PER_LEG_BPS'],
            px_oos_engine, ret_oos_engine, CONFIG, state_start=CONFIG['OOS_START'])
        oos_trades[(x_name, y_name)] = trades
    ok_s0 = compare('S0', oos_trades, 'results/golden_master_trades/S0_trades.json')

    s1_time_stops, s1_ineligible = compute_pair_time_stops(FROZEN_PAIRS, formation_px, S1_TIME_STOP_MULTIPLIER)
    S1_ELIGIBLE_PAIRS = [p for p in FROZEN_PAIRS if p in s1_time_stops]
    _, S1_OOS_TRADES = run_s1_universe(S1_ELIGIBLE_PAIRS, s1_time_stops, px_oos_engine, ret_oos_engine, CONFIG, CONFIG['OOS_START'])
    ok_s1 = compare('S1', S1_OOS_TRADES, 'results/golden_master_trades/S1_trades.json')

    WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
    REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                                WF_CONFIG['REBALANCE_MONTHS_PRIMARY'], WF_CONFIG['FORMATION_MONTHS'])
    WF1_REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]
    wf1_sel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                       wf_config=WF_CONFIG,
                                       extra=f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}")
    WF1_SELECTIONS = run_wf1_selection_pipeline(WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
                                                 cache_path('wf1_selections', wf1_sel_fp),
                                                 max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])
    _, _, WF1_ALL_TRADES = run_wf1_full(WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    ok_wf1 = compare('WF1', WF1_ALL_TRADES, 'results/golden_master_trades/WF1_trades.json')

    _, _, WF2_ALL_TRADES = run_wf2_full(WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    ok_wf2 = compare('WF2', WF2_ALL_TRADES, 'results/golden_master_trades/WF2_trades.json')

    all_ok = ok_s0 and ok_s1 and ok_wf1 and ok_wf2
    print('\nOVERALL:', 'ALL TRADE LOGS MATCH' if all_ok else 'DISCREPANCIES FOUND')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
