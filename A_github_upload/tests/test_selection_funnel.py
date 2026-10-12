"""Selection-funnel diagnostic tests (Phase B2): the `collect_diagnostics=True` path must not
change default behavior, and its reported counts must satisfy the structural identities a
sequential filter chain guarantees (survivor counts cannot increase through the chain,
executed-trade counts must reconcile with the trades actually returned)."""
import numpy as np
import pandas as pd
import pytest

from stat_arb.pair_selection import screen_pairs_at_date, select_pairs_at_rebalance
from stat_arb.funnel import build_selection_funnel, FUNNEL_STAGE_COLUMNS

BASE_WF_CONFIG = {
    'EG_ALPHA': 0.5, 'MIN_OBS_FOR_TEST': 100, 'FORMATION_MONTHS': 12,
    'MAX_HALF_LIFE_BARS': 100000, 'MIN_CROSSINGS': 0, 'REQUIRE_POSITIVE_BETA': True,
    'FDR_METHOD': 'fdr_bh', 'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,
    'TIME_STOP_MULTIPLIER_PRIMARY': 2.5,
}


def _synthetic_universe(n=3000, n_coins=8, seed=0):
    """Several coins driven by a common factor with different loadings -- produces a mix of
    cointegrated and non-cointegrated pairs, some sharing the same coin, so both the FDR/Beta/
    Half-Life/crossing funnel and the one-pair-per-coin overlap logic have real work to do."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='4h')
    common = np.cumsum(rng.normal(0, 1, n))
    data = {}
    for i in range(n_coins):
        beta = 0.4 + 0.25 * i
        data[f'COIN{i}USDT'] = np.exp((common * beta + rng.normal(0, 0.15, n)) * 0.01 + 3.0)
    return pd.DataFrame(data, index=idx)


def test_screen_pairs_at_date_default_unchanged_shape():
    px = _synthetic_universe()
    rebalance_date = px.index[2500]
    result_default = screen_pairs_at_date(rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG)
    assert isinstance(result_default, pd.DataFrame)

    result_diag = screen_pairs_at_date(rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG,
                                        collect_diagnostics=True)
    assert isinstance(result_diag, tuple) and len(result_diag) == 3
    result_with_diag, funnel, pair_trail = result_diag
    pd.testing.assert_frame_equal(result_default, result_with_diag)


def test_screen_pairs_at_date_funnel_stage_counts_are_monotonic():
    px = _synthetic_universe()
    rebalance_date = px.index[2500]
    result, funnel, pair_trail = screen_pairs_at_date(rebalance_date, px, px.columns.tolist(),
                                                        BASE_WF_CONFIG, collect_diagnostics=True)
    assert funnel['n_testable_pairs'] <= funnel['n_candidate_pairs']
    assert funnel['n_directed_eg_tests'] <= 2 * funnel['n_testable_pairs']
    assert funnel['n_fdr_significant_pairs'] <= funnel['n_testable_pairs']
    assert funnel['n_pass_beta'] <= funnel['n_fdr_significant_pairs']
    assert funnel['n_pass_half_life'] <= funnel['n_pass_beta']
    assert funnel['n_pass_crossings'] <= funnel['n_pass_half_life']
    assert funnel['n_fully_qualifying'] == funnel['n_pass_crossings']
    assert funnel['n_fully_qualifying'] == len(result)


def test_pair_trail_elimination_stage_consistency():
    px = _synthetic_universe(seed=1)
    rebalance_date = px.index[2500]
    result, funnel, pair_trail = screen_pairs_at_date(rebalance_date, px, px.columns.tolist(),
                                                        BASE_WF_CONFIG, collect_diagnostics=True)
    if len(pair_trail) == 0:
        pytest.skip('no testable pairs for this synthetic seed')
    # Every fully-qualifying pair has no elimination reason; every non-qualifying pair has one.
    assert (pair_trail.loc[pair_trail['fully_qualifying'], 'elimination_stage'].isna()).all()
    assert (~pair_trail.loc[~pair_trail['fully_qualifying'], 'elimination_stage'].isna()).all()
    # passed_fdr must match the FDR-significance flag used to build the trail.
    failed_fdr = pair_trail[pair_trail['elimination_stage'] == 'fdr']
    assert not failed_fdr['passed_fdr'].any()


def test_select_pairs_at_rebalance_overlap_diagnostics_identify_winner():
    px = _synthetic_universe(n_coins=4, seed=2)
    wf_config = dict(BASE_WF_CONFIG, REQUIRE_POSITIVE_BETA=False)
    rebalance_date = px.index[2500]
    selected, selection_funnel, pair_trail, overlap_conflicts = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), wf_config, max_pairs=10, collect_diagnostics=True)

    assert selection_funnel['n_selected'] == len(selected)
    assert selection_funnel['n_fully_qualifying'] >= selection_funnel['n_selected']
    assert (selection_funnel['n_selected'] + selection_funnel['n_removed_by_overlap'] +
            selection_funnel['n_removed_by_max_pairs_cap']) == selection_funnel['n_fully_qualifying']

    selected_pairs = {s['pair'] for s in selected}
    for conflict in overlap_conflicts:
        # The winning pair must actually be one of the selected pairs, and must rank at least
        # as well (lower or equal pvalue_adj) as the pair it beat -- the ranking rule itself.
        assert conflict['winning_pair'] in selected_pairs
        assert conflict['winning_pvalue_adj'] <= conflict['rejected_pvalue_adj'] + 1e-12
        assert conflict['rejected_pair'] not in selected_pairs


def test_select_pairs_at_rebalance_max_pairs_cap_diagnostics():
    px = _synthetic_universe(n_coins=8, seed=3)
    wf_config = dict(BASE_WF_CONFIG, REQUIRE_POSITIVE_BETA=False)
    rebalance_date = px.index[2500]
    # First, discover how many pairs would fully qualify uncapped.
    _, funnel_uncapped, _ = screen_pairs_at_date(rebalance_date, px, px.columns.tolist(), wf_config,
                                                  collect_diagnostics=True)
    if funnel_uncapped['n_fully_qualifying'] < 2:
        pytest.skip('not enough qualifying pairs in this synthetic seed to exercise the cap')

    cap = 1
    selected, selection_funnel, pair_trail, overlap_conflicts = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), wf_config, max_pairs=cap, collect_diagnostics=True)
    assert len(selected) <= cap
    assert selection_funnel['n_selected'] == len(selected)
    if selection_funnel['n_fully_qualifying'] > cap and selection_funnel['n_removed_by_overlap'] == 0:
        assert selection_funnel['n_removed_by_max_pairs_cap'] > 0
        capped = pair_trail[pair_trail['elimination_stage'] == 'max_pairs_cap']
        assert len(capped) == selection_funnel['n_removed_by_max_pairs_cap']
        assert not capped['selected'].any()


def test_selected_pair_trail_row_matches_regardless_of_tuple_order():
    """Regression test: `selected[i]['pair']` is (x, y) in the winning regression direction,
    while `pair_trail`'s 'pair' column is the alphabetically-sorted unordered pair -- these
    are different tuples whenever the winning direction isn't already alphabetical (e.g.
    ('OMUSDT', 'LDOUSDT') vs ('LDOUSDT', 'OMUSDT')). A selected pair must always show up as
    `selected=True` in the pair-level trail regardless of which tuple order won the
    regression -- this must never silently fail to match."""
    px = _synthetic_universe(n_coins=4, seed=2)
    wf_config = dict(BASE_WF_CONFIG, REQUIRE_POSITIVE_BETA=False)
    rebalance_date = px.index[2500]
    selected, selection_funnel, pair_trail, overlap_conflicts = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), wf_config, max_pairs=10, collect_diagnostics=True)
    if not selected:
        pytest.skip('no selected pairs for this synthetic seed')

    for s in selected:
        sorted_pair = tuple(sorted(s['pair']))
        matching_rows = pair_trail[pair_trail['pair'] == sorted_pair]
        assert len(matching_rows) == 1, f'pair {sorted_pair} not found exactly once in pair_trail'
        assert matching_rows.iloc[0]['selected'], (
            f"selected pair {s['pair']} (sorted: {sorted_pair}) not marked selected=True in pair_trail "
            f"-- likely a tuple-order mismatch between selected[]['pair'] and pair_trail['pair']")
        assert pd.isna(matching_rows.iloc[0]['elimination_stage']) or matching_rows.iloc[0]['elimination_stage'] is None


def test_build_selection_funnel_reconciles_with_real_trades():
    px = _synthetic_universe(n_coins=4, seed=4)
    wf_config = dict(BASE_WF_CONFIG, REQUIRE_POSITIVE_BETA=False)
    rebalance_dates = [px.index[2500]]

    selected, _, _, _ = select_pairs_at_rebalance(rebalance_dates[0], px, px.columns.tolist(),
                                                    wf_config, max_pairs=10, collect_diagnostics=True)
    if not selected:
        pytest.skip('no selected pairs for this synthetic seed')

    # Fabricate a real-shaped trades dict: first selected pair gets 2 trades, rest get none.
    all_trades = {(rebalance_dates[0], selected[0]['pair']): [{'trade_id': 1}, {'trade_id': 2}]}
    for s in selected[1:]:
        all_trades[(rebalance_dates[0], s['pair'])] = []

    funnel_df, pair_level_df, conflicts_df = build_selection_funnel(
        rebalance_dates, px, px.columns.tolist(), wf_config, 10, all_trades)

    assert list(funnel_df.columns) == FUNNEL_STAGE_COLUMNS
    row = funnel_df.iloc[0]
    expected_total_trades = sum(len(v) for v in all_trades.values())
    assert row['n_entry_events'] == expected_total_trades
    assert row['n_executed_trades'] == expected_total_trades
    assert row['n_selected_with_entry_signal'] + row['n_selected_without_entry_signal'] == row['n_selected']
    assert row['n_selected_with_entry_signal'] == 1  # only the first selected pair has trades

    selected_pair_rows = pair_level_df[pair_level_df['selected']]
    traded = selected_pair_rows[selected_pair_rows['pair'] == selected[0]['pair']]
    assert (traded['n_executed_trades'] == 2).all()
    assert (traded['elimination_stage'].isna()).all()  # selected AND traded -> no elimination reason
