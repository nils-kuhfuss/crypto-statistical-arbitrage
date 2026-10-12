"""Selection-funnel diagnostics (Phase B2): explains where the walk-forward opportunity set
shrinks between the eligible universe and executed trades.

Diagnostic only -- introduces no new threshold, ranking rule, or selection outcome. Every
stage count here is captured by `stat_arb.pair_selection`'s `collect_diagnostics=True`
instrumentation of the real `screen_pairs_at_date`/`select_pairs_at_rebalance` filtering loop
(the same code path used to build `WF1_SELECTIONS`), never a parallel reimplementation. The
entry-signal/executed-trade counts come from the REAL trades already produced by
`stat_arb.walkforward.run_wf1_full`/`run_wf2_full` (passed in as `all_trades`), never
re-simulated here.

Units, to avoid conflating them (see module-level column names below):
- assets: individual coins.
- candidate (unordered) pairs: `len(itertools.combinations(eligible_assets, 2))`.
- testable pairs: candidate pairs with >= MIN_OBS_FOR_TEST overlapping observations (i.e.
  pairs `run_engle_granger_tests` actually tested), counted from its own output.
- directed EG tests: one row per (pair, direction) actually tested -- twice the number of
  testable pairs (barring a `coint()` exception on one direction), NEVER compared directly
  against pair-level counts without this doubling being explicit.
- FDR-significant pairs (after direction resolution): testable pairs with at least one
  direction surviving Benjamini-Hochberg, deduplicated to the lower-adjusted-p-value direction.
- fully-qualifying pairs: FDR-significant pairs that also pass positive-Beta, Half-Life, and
  crossing count, in that exact order (matching `screen_pairs_at_date`'s own loop).
- selected pairs: fully-qualifying pairs surviving the one-pair-per-coin greedy accept and the
  `MAX_PAIRS_PER_REBALANCE` cap, in p-value-ascending rank order.
- entry events / executed trades: from the real per-pair `run_trade_state_machine`/
  `execute_fixed_quantity_pair` results -- in this fixed-quantity, next-bar-execution engine
  every signal-level trade record is always executed (never dropped at execution time), so
  these two counts are structurally identical here; they are reported as one figure with this
  noted explicitly, not invented as two different numbers.
"""
import pandas as pd

from .pair_selection import select_pairs_at_rebalance

FUNNEL_STAGE_COLUMNS = [
    'rebalance_date', 'n_eligible_assets', 'n_candidate_pairs', 'n_testable_pairs',
    'n_directed_eg_tests', 'n_fdr_significant_pairs', 'n_pass_beta', 'n_pass_half_life',
    'n_pass_crossings', 'n_fully_qualifying', 'n_selected', 'n_removed_by_overlap',
    'n_removed_by_max_pairs_cap', 'n_selected_with_entry_signal',
    'n_selected_without_entry_signal', 'n_entry_events', 'n_executed_trades',
]


def build_selection_funnel(rebalance_dates, price_data: pd.DataFrame, coins, wf_config: dict,
                            max_pairs: int, all_trades: dict):
    """For every rebalance date, run the real point-in-time screen + selection
    (`select_pairs_at_rebalance(..., collect_diagnostics=True)`) and merge in the REAL
    executed-trade outcome for each selected pair instance from `all_trades` (the actual
    `{(rebalance_date, pair): trades}` dict already produced by `run_wf1_full`/`run_wf2_full`).

    Returns (funnel_df, pair_level_df, overlap_conflicts_df):
      funnel_df: one row per rebalance date, every stage count (columns: `FUNNEL_STAGE_COLUMNS`).
      pair_level_df: one row per testable pair at every rebalance date, with pass/fail flags
        through every stage and the first stage at which it was eliminated (None only for
        pairs that were selected AND generated at least one executed trade).
      overlap_conflicts_df: one row per pair rejected specifically by the one-pair-per-coin
        constraint, naming the conflicting coin, the winning pair, and the ranking metric
        (pvalue_adj) that decided it -- the ranking rule itself is never altered here.
    """
    funnel_rows = []
    pair_level_frames = []
    conflict_rows = []

    for rebalance_date in rebalance_dates:
        selected, selection_funnel, pair_trail, overlap_conflicts = select_pairs_at_rebalance(
            rebalance_date, price_data, coins, wf_config, max_pairs, collect_diagnostics=True)

        entry_flags = {}
        trade_counts = {}
        n_with_signal = n_without_signal = n_entry_events = n_executed = 0
        for sel in selected:
            # `sel['pair']` (x, y winning-direction order) is the correct key into `all_trades`
            # (it matches WF1_ALL_TRADES's real instance keys, built the same way in
            # walkforward.py) -- but `pair_trail['pair']` uses the sorted-unordered-pair
            # convention (see pair_selection.select_pairs_at_rebalance's coin_owner note), so
            # entry_flags/trade_counts are keyed by the sorted tuple for the `.map()` below.
            instance_key = (rebalance_date, sel['pair'])
            trades = all_trades.get(instance_key, [])
            has_signal = len(trades) > 0
            sorted_pair = tuple(sorted(sel['pair']))
            entry_flags[sorted_pair] = has_signal
            trade_counts[sorted_pair] = len(trades)
            if has_signal:
                n_with_signal += 1
            else:
                n_without_signal += 1
            # Entry events and executed trades are the same count in this engine (see module
            # docstring) -- both accumulated explicitly from the same real trades list rather
            # than assumed equal.
            n_entry_events += len(trades)
            n_executed += len(trades)

        if len(pair_trail):
            pair_trail = pair_trail.copy()
            pair_trail['rebalance_date'] = rebalance_date
            pair_trail['generated_entry_signal'] = pair_trail['pair'].map(entry_flags)
            pair_trail['n_executed_trades'] = pair_trail['pair'].map(trade_counts)
            no_signal_mask = pair_trail['generated_entry_signal'] == False  # noqa: E712 (explicit vs NaN)
            pair_trail.loc[no_signal_mask, 'elimination_stage'] = 'no_entry_signal'
            pair_level_frames.append(pair_trail)

        for c in overlap_conflicts:
            c = dict(c)
            c['rebalance_date'] = rebalance_date
            conflict_rows.append(c)

        row = dict(selection_funnel)
        row['n_selected_with_entry_signal'] = n_with_signal
        row['n_selected_without_entry_signal'] = n_without_signal
        row['n_entry_events'] = n_entry_events
        row['n_executed_trades'] = n_executed
        funnel_rows.append(row)

    funnel_df = pd.DataFrame(funnel_rows, columns=FUNNEL_STAGE_COLUMNS)
    pair_level_df = pd.concat(pair_level_frames, ignore_index=True) if pair_level_frames else pd.DataFrame()
    conflicts_df = pd.DataFrame(conflict_rows)
    return funnel_df, pair_level_df, conflicts_df
