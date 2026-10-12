"""Standalone precompute for the Phase B2 selection-funnel diagnostic
(see `stat_arb.funnel.build_selection_funnel`).

Expensive: re-runs the point-in-time EG/FDR/Beta/Half-Life/crossing screen with diagnostics
enabled for every WF1 rebalance date, since the funnel's intermediate stage counts are not
preserved by the plain `wf1_selections_*.pkl` cache (which only stores the final selected
pairs). Mirrors `scripts/run_stability_screen.py`'s convention: a standalone script, not a
notebook cell -- notebook 03 only loads its output (`load_or_assert`).

Caches the expensive intermediate result to `data/cache/selection_funnel_<fingerprint>.pkl`
(keyed by the same `wf_cache_fingerprint`, with `extra` distinguishing it from the plain
WF1-selections cache -- this does not change the existing wf1_selections/pair_stability
fingerprints or their cached files) and writes the machine-readable deliverable to
`results/tables/selection_funnel*.json`.

Run from the project root:  python scripts/run_selection_funnel.py
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from config.research_config import CONFIG, WF_CONFIG
from stat_arb.data import build_walkforward_universe
from stat_arb.walkforward import generate_rebalance_dates, run_wf1_selection_pipeline, run_wf1_full
from stat_arb.cache import wf_cache_fingerprint, cache_path, load_or_compute
from stat_arb.funnel import build_selection_funnel


def main():
    os.chdir(_PROJECT_ROOT)

    WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
    REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                                WF_CONFIG['REBALANCE_MONTHS_PRIMARY'], WF_CONFIG['FORMATION_MONTHS'])
    WF1_REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]

    # Reuse the exact same WF1 selections + backtest pipeline already validated elsewhere --
    # the funnel's entry-signal/trade-level counts come from these REAL results, never
    # re-simulated. This also confirms the plain wf1_selections cache fingerprint/content are
    # untouched by anything in this script.
    wf1_extra = f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}"
    wf1_sel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                       wf_config=WF_CONFIG, extra=wf1_extra)
    WF1_SELECTIONS = run_wf1_selection_pipeline(WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
                                                 cache_path('wf1_selections', wf1_sel_fp),
                                                 max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])
    _, _, WF1_ALL_TRADES = run_wf1_full(WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)

    funnel_fp = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                      wf_config=WF_CONFIG, extra=f'{wf1_extra},funnel_diagnostics')
    funnel_cache_path = cache_path('selection_funnel', funnel_fp)

    def _compute():
        print(f'Computing selection funnel for {len(WF1_REBALANCE_DATES)} rebalance dates '
              f'(re-runs the point-in-time screen with diagnostics enabled) ...')
        funnel_df, pair_level_df, conflicts_df = build_selection_funnel(
            WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
            WF_CONFIG['MAX_PAIRS_PER_REBALANCE'], WF1_ALL_TRADES)
        return {'funnel_df': funnel_df, 'pair_level_df': pair_level_df, 'conflicts_df': conflicts_df}

    result = load_or_compute(funnel_cache_path, _compute)
    print(f'Selection funnel cache: {funnel_cache_path}')
    print(result['funnel_df'].to_string(index=False))

    tables_dir = os.path.join(_PROJECT_ROOT, 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    result['funnel_df'].to_json(os.path.join(tables_dir, 'selection_funnel.json'),
                                 orient='records', indent=2, date_format='iso')
    result['pair_level_df'].to_json(os.path.join(tables_dir, 'selection_funnel_pair_level.json'),
                                     orient='records', indent=2, date_format='iso')
    result['conflicts_df'].to_json(os.path.join(tables_dir, 'selection_funnel_overlap_conflicts.json'),
                                    orient='records', indent=2, date_format='iso')
    print('Wrote results/tables/selection_funnel.json, '
          'selection_funnel_pair_level.json, selection_funnel_overlap_conflicts.json')


if __name__ == '__main__':
    main()
