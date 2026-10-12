"""Phase B2.5: one-coin (one-pair-per-coin) constraint ablation.

Exploratory post-sample ablation, NOT a new primary specification. Removes exactly one
assumption from `WF1_SEMIANNUAL_REESTIMATION`'s pair selection -- the one-pair-per-coin
portfolio constraint (`select_pairs_at_rebalance(..., enforce_one_pair_per_coin=False)`,
`pair_selection.py`) -- and reuses every other piece of the pipeline completely unmodified:
`stat_arb.walkforward.run_wf1_full` (execution + equal-sleeve portfolio stitching, generic over
however many pairs a rebalance block selects), `stat_arb.portfolio.evaluate_coin_netted_portfolio`
/`coin_concentration_diagnostics`, and `stat_arb.reporting.standardized_wf1_summary`/
`pair_level_contribution`/`active_day_benchmark_stats`. Nothing here re-simulates or re-derives
what those functions already compute.

This module contributes only the two pieces that are genuinely new for the ablation:
1. `run_no_overlap_selection_pipeline` -- the selection stage with the constraint disabled
   (mirrors `walkforward.run_wf1_selection_pipeline`'s cache convention exactly).
2. `coin_cluster_pair_table` / `coin_cluster_return_correlation` -- diagnostics answering
   whether a cluster of pairs sharing one coin (e.g. every fully-qualifying LDOUSDT pair at
   2024-01-01) behaves like diversified relative-value opportunities or one concentrated
   coin-level factor trade expressed through many pairs.
"""
import pandas as pd

from .pair_selection import select_pairs_at_rebalance
from .cache import load_or_compute


def run_no_overlap_selection_pipeline(rebalance_dates, price_data: pd.DataFrame, coins, wf_config: dict,
                                       cache_path: str, max_pairs: int) -> dict:
    """Same cache-then-compute convention as `walkforward.run_wf1_selection_pipeline`, but with
    `enforce_one_pair_per_coin=False` -- the ONLY methodological difference from WF1's own
    selection pipeline. `max_pairs` (the existing cap) is preserved unchanged and still applies."""
    def _compute():
        selections = {}
        for rebal_date in rebalance_dates:
            selections[rebal_date] = select_pairs_at_rebalance(
                rebal_date, price_data, coins, wf_config, max_pairs,
                enforce_one_pair_per_coin=False)
        return selections
    return load_or_compute(cache_path, _compute)


def coin_cluster_pair_table(rebalance_date, coin_name: str, pair_trail: pd.DataFrame,
                             selections_by_date: dict, all_trades: dict) -> pd.DataFrame:
    """One row per testable pair at `rebalance_date` involving `coin_name`, joining
    `pair_trail` (from `select_pairs_at_rebalance(..., collect_diagnostics=True)`, sorted-pair
    convention) against the real selections/trades (winning-regression-direction convention) --
    normalized via the selected-pairs list itself, exactly the join pattern already used (and
    the exact bug already caught once) in `stat_arb.funnel.build_selection_funnel`; never
    assume the two conventions coincide.

    Columns: pair, pvalue_adj, fully_qualifying, selected, elimination_stage,
    generated_entry_signal, n_executed_trades.
    """
    if len(pair_trail) == 0:
        return pd.DataFrame(columns=['pair', 'pvalue_adj', 'fully_qualifying', 'selected',
                                      'elimination_stage', 'generated_entry_signal', 'n_executed_trades'])
    trail = pair_trail[pair_trail['pair'].apply(lambda p: coin_name in p)].copy()
    sel_by_sorted = {tuple(sorted(s['pair'])): s for s in selections_by_date.get(rebalance_date, [])}

    rows = []
    for _, r in trail.iterrows():
        sorted_pair = r['pair']
        sel = sel_by_sorted.get(sorted_pair)
        winning_pair = sel['pair'] if sel is not None else None
        trades = all_trades.get((rebalance_date, winning_pair), []) if winning_pair is not None else []
        rows.append({
            'pair': f'{sorted_pair[0]}/{sorted_pair[1]}', 'pvalue_adj': r['pvalue_adj'],
            'fully_qualifying': bool(r['fully_qualifying']), 'selected': bool(r['selected']),
            'elimination_stage': r['elimination_stage'],
            'generated_entry_signal': (len(trades) > 0) if sel is not None else None,
            'n_executed_trades': len(trades),
        })
    return pd.DataFrame(rows).sort_values('pvalue_adj').reset_index(drop=True)


def coin_cluster_return_correlation(rebalance_date, coin_name: str, selections_by_date: dict,
                                     all_results: dict) -> pd.DataFrame:
    """Pairwise correlation matrix of 4h `net_ret` series among every pair involving
    `coin_name` SELECTED at `rebalance_date` (not only the ones that traded), computed over the
    full rebalance block -- flat bars contribute a return of 0, matching `all_results`'s own
    convention, not conditioned only on bars where both pairs are simultaneously active. A pair
    that never traded has zero variance and therefore an undefined (NaN) correlation with every
    other pair -- reported as NaN, not silently dropped, since "never traded" is itself part of
    the finding.

    Answers: do this coin's pairs behave like independent relative-value opportunities (low/
    mixed correlation) or one concentrated coin-level factor trade expressed through many pairs
    (high positive correlation)?
    """
    sels = [s for s in selections_by_date.get(rebalance_date, []) if coin_name in s['pair']]
    series = {}
    for s in sels:
        key = (rebalance_date, s['pair'])
        if key in all_results:
            series[f"{s['pair'][0]}/{s['pair'][1]}"] = all_results[key]['net_ret']
    if len(series) < 2:
        return pd.DataFrame()
    return pd.DataFrame(series).corr()
