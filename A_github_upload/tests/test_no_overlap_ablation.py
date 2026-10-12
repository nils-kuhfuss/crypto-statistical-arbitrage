"""Phase B2.5 no-overlap-constraint ablation tests: `enforce_one_pair_per_coin=False` must
change ONLY the coin-overlap rejection (pair_selection.py) -- ranking, the max-pairs cap, and
every downstream execution/portfolio/metrics function stay shared and untouched. Also covers
the meaningful structural reconciliation the user's brief asked for in place of relying on the
pre-existing tautological sleeve-allocation assertion (`n * (1/n) <= 1` is always true and
proves nothing about whether coin quantities were actually scaled correctly)."""
import numpy as np
import pandas as pd
import pytest

from stat_arb.pair_selection import select_pairs_at_rebalance
from stat_arb.execution import backtest_single_pair
from stat_arb.portfolio import build_coin_netted_portfolio, coin_concentration_diagnostics
from stat_arb.reporting import standardized_wf1_summary

BASE_WF_CONFIG = {
    'EG_ALPHA': 0.5, 'MIN_OBS_FOR_TEST': 100, 'FORMATION_MONTHS': 12,
    'MAX_HALF_LIFE_BARS': 100000, 'MIN_CROSSINGS': 0, 'REQUIRE_POSITIVE_BETA': True,
    'FDR_METHOD': 'fdr_bh', 'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,
    'TIME_STOP_MULTIPLIER_PRIMARY': 2.5,
}


def _hub_universe(n=3000, n_coins=6, seed=0):
    """A single common random walk drives every coin (with a different loading each) -- nearly
    every unordered pair cointegrates, so each of the `n_coins` appears in several
    fully-qualifying pairs simultaneously. Mirrors the real 2024-01-01 finding (B2) where every
    one of 15 fully-qualifying pairs happened to involve LDOUSDT."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='4h')
    common = np.cumsum(rng.normal(0, 1, n))
    data = {}
    for i in range(n_coins):
        beta = 0.5 + 0.2 * i
        data[f'COIN{i}USDT'] = np.exp((common * beta + rng.normal(0, 0.1, n)) * 0.01 + 3.0)
    return pd.DataFrame(data, index=idx)


def _rebalance_date(px):
    return px.index[2500]


def test_default_enforces_one_pair_per_coin_unchanged():
    px = _hub_universe()
    rebalance_date = _rebalance_date(px)
    default = select_pairs_at_rebalance(rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG, max_pairs=10)
    explicit = select_pairs_at_rebalance(rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG, max_pairs=10,
                                          enforce_one_pair_per_coin=True)
    assert [s['pair'] for s in default] == [s['pair'] for s in explicit]
    coins_used = [c for s in default for c in (s['x'], s['y'])]
    assert len(coins_used) == len(set(coins_used)), 'default behavior must still enforce one-pair-per-coin'


def test_no_overlap_mode_retains_pairs_sharing_a_coin():
    px = _hub_universe(n_coins=6)
    rebalance_date = _rebalance_date(px)
    selected, funnel, _, overlap_conflicts = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG, max_pairs=15,
        collect_diagnostics=True, enforce_one_pair_per_coin=False)
    if funnel['n_fully_qualifying'] < 4:
        pytest.skip('not enough qualifying pairs in this synthetic seed')
    assert funnel['n_removed_by_overlap'] == 0
    assert len(overlap_conflicts) == 0
    coin_counts = pd.Series([c for s in selected for c in (s['x'], s['y'])]).value_counts()
    assert coin_counts.max() > 1, 'no coin appears in more than one selected pair -- test fixture not exercising overlap'


def test_ranking_order_unchanged_in_no_overlap_mode():
    px = _hub_universe(n_coins=6)
    rebalance_date = _rebalance_date(px)
    selected, *_ = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG, max_pairs=15,
        collect_diagnostics=True, enforce_one_pair_per_coin=False)
    pvalues = [s['pvalue_adj'] for s in selected]
    assert pvalues == sorted(pvalues), 'no-overlap selection must still be in ascending pvalue_adj rank order'


def test_max_pairs_cap_still_applies_without_overlap_constraint():
    px = _hub_universe(n_coins=6)
    rebalance_date = _rebalance_date(px)
    selected, funnel, pair_trail, _ = select_pairs_at_rebalance(
        rebalance_date, px, px.columns.tolist(), BASE_WF_CONFIG, max_pairs=3,
        collect_diagnostics=True, enforce_one_pair_per_coin=False)
    if funnel['n_fully_qualifying'] <= 3:
        pytest.skip('not enough qualifying pairs in this synthetic seed to exercise the cap')
    assert len(selected) == 3
    assert funnel['n_removed_by_overlap'] == 0
    assert funnel['n_removed_by_max_pairs_cap'] == funnel['n_fully_qualifying'] - 3
    capped = pair_trail[pair_trail['elimination_stage'] == 'max_pairs_cap']
    assert len(capped) == funnel['n_removed_by_max_pairs_cap']
    assert not capped['selected'].any()


def _synthetic_pair_result(idx, seed, beta=2.0):
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx)
    y = pd.Series(50 + np.cumsum(rng.normal(0, 0.3, len(idx))), index=idx)
    z = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
    z = z - z.mean()
    result, trades = backtest_single_pair(
        x, y, alpha=0.0, beta=beta, ret_x=x.pct_change(fill_method=None), ret_y=y.pct_change(fill_method=None),
        entry_z=1.5, exit_z=0.5, hard_stop_z=4.0, time_stop_bars=156, cost_per_leg_bps=20,
        z_score=z, beta_series=pd.Series(beta, index=idx))
    return x, y, result, trades


def test_shared_coin_netting_reconciles_structurally():
    """The primary WF1 selection can never select two pairs sharing a coin (one-pair-per-coin
    is enforced by construction), so the existing portfolio-reconciliation tests never exercise
    multi-pair coin sharing. The no-overlap ablation makes this a real, expected case (e.g. 3
    different pairs all trading LDOUSDT as one leg) -- verify `build_coin_netted_portfolio`
    nets a shared coin's quantity as the sleeve-weighted SUM of every pair's own quantity for
    that coin, not merely that the tautological `n * (1/n) <= 1` holds."""
    idx = pd.date_range('2022-01-01', periods=40, freq='4h')
    shared_coin = 'HUBUSDT'
    n_pairs = 4
    pair_results = {}
    price_cols = {shared_coin: None}
    pairs = []
    for i in range(n_pairs):
        x, y, result, _ = _synthetic_pair_result(idx, seed=i, beta=1.5 + 0.3 * i)
        pair = (f'LEG{i}USDT', shared_coin)
        pairs.append(pair)
        # Re-run with the shared coin as the "y" leg so its quantity is `qty_y` in every result,
        # and store the leg coin's own price for `x` -- x/y is a naming convention here, not a
        # regression-direction claim.
        pair_results[pair] = result
        price_cols[pair[0]] = x
        price_cols[shared_coin] = y if i == 0 else price_cols[shared_coin]  # one shared price series
    price_data = pd.DataFrame(price_cols)

    portfolio = build_coin_netted_portfolio(pair_results, pairs, price_data, cost_per_leg_bps=20,
                                             start=idx[0], end=idx[-1])
    sleeve_weight = 1.0 / n_pairs

    # Manual reconciliation: the shared coin's netted quantity at every bar must equal the
    # sleeve-weighted sum of each pair's OWN qty_y for that coin -- not merely close to it.
    expected_shared_qty = sum(pair_results[p]['qty_y'].reindex(portfolio['coin_qty'].index).fillna(0.0) * sleeve_weight
                               for p in pairs)
    pd.testing.assert_series_equal(portfolio['coin_qty'][shared_coin], expected_shared_qty,
                                    check_names=False, atol=1e-10)

    # Every leg coin's quantity must equal exactly its own pair's qty_x * sleeve_weight (no
    # cross-contamination from the other pairs sharing HUBUSDT).
    for i, pair in enumerate(pairs):
        expected_leg_qty = pair_results[pair]['qty_x'].reindex(portfolio['coin_qty'].index).fillna(0.0) * sleeve_weight
        pd.testing.assert_series_equal(portfolio['coin_qty'][pair[0]], expected_leg_qty,
                                        check_names=False, atol=1e-10)

    # coin_notional must reconcile to coin_qty * mark price exactly (portfolio.py's own
    # definition, re-derived here independently rather than trusted).
    expected_notional = portfolio['coin_qty'][shared_coin] * portfolio['coin_marks'][shared_coin]
    pd.testing.assert_series_equal(portfolio['coin_notional'][shared_coin], expected_notional,
                                    check_names=False, atol=1e-8)


def test_coin_concentration_diagnostics_reconcile_with_portfolio_state():
    idx = pd.date_range('2022-01-01', periods=40, freq='4h')
    shared_coin = 'HUBUSDT'
    n_pairs = 3
    pair_results = {}
    price_cols = {}
    pairs = []
    for i in range(n_pairs):
        x, y, result, _ = _synthetic_pair_result(idx, seed=10 + i, beta=1.0 + 0.1 * i)
        pair = (f'LEG{i}USDT', shared_coin)
        pairs.append(pair)
        pair_results[pair] = result
        price_cols[pair[0]] = x
        if shared_coin not in price_cols:
            price_cols[shared_coin] = y
    price_data = pd.DataFrame(price_cols)

    from stat_arb.portfolio import evaluate_coin_netted_portfolio
    portfolio, _ = evaluate_coin_netted_portfolio(pair_results, pairs, price_data, cost_per_leg_bps=20,
                                                    start=idx[0], end=idx[-1], bars_per_year=365 * 6)
    diag = coin_concentration_diagnostics(portfolio)

    equity = portfolio['frame']['equity']
    expected_gross_pct = (portfolio['coin_notional'][shared_coin].abs() / equity).max()
    assert diag['max_gross_exposure_by_coin'][shared_coin] == pytest.approx(expected_gross_pct)

    expected_net_pct = (portfolio['coin_notional'][shared_coin] / equity).abs().max()
    assert diag['max_abs_net_exposure_by_coin'][shared_coin] == pytest.approx(expected_net_pct)

    # Every bar's largest-gross-coin label must actually be argmax of that bar's own gross %.
    gross_pct = diag['gross_exposure_pct_by_coin']
    for ts in gross_pct.index[gross_pct.sum(axis=1) > 0][:5]:
        assert diag['largest_gross_coin_by_bar'][ts] == gross_pct.loc[ts].idxmax()


def test_no_overlap_trade_counts_reconcile_with_trade_logs():
    """Fabricated (rebalance_date, pair)-keyed trades/results dict -- confirms
    `standardized_wf1_summary`'s n_trades identity holds regardless of how many pairs share a
    coin (it sums len(trades) over all_trades.values(), independent of selection method)."""
    rebalance_date = pd.Timestamp('2024-01-01')
    idx = pd.date_range('2024-01-01', periods=20, freq='4h')
    shared_coin = 'HUBUSDT'
    all_results, all_trades = {}, {}
    for i in range(3):
        pair = (f'LEG{i}USDT', shared_coin)
        key = (rebalance_date, pair)
        all_results[key] = pd.DataFrame({'net_ret': 0.0, 'gross_exposure': 0.0}, index=idx)
        all_trades[key] = [{'duration_bars': 3, 'exit_reason': 'exit'}] * (i + 1)
    full_frame = pd.DataFrame({'net_ret': 0.0, 'gross_ret': 0.0, 'turnover': 0.0, 'costs': 0.0,
                                'active_flag': False}, index=idx)
    price_data = pd.DataFrame({shared_coin: 1.0, 'BTCUSDT': 100.0}, index=idx)
    row = standardized_wf1_summary('WF1_NO_OVERLAP_TEST', full_frame, all_results, all_trades,
                                    price_data, idx[0], idx[-1])
    expected_total_trades = sum(len(t) for t in all_trades.values())
    assert row['n_trades'] == expected_total_trades == 1 + 2 + 3
    assert row['n_pairs_traded'] == 3
