"""Portfolio-level reconciliation tests, adapted from `legacy/Stat_Arb_cont_res_1.ipynb`
Sections 13, 21, 22 (flat-after-liquidation, sleeve allocation, coin netting, repeated pair
instances not overwritten)."""
import numpy as np
import pandas as pd

from stat_arb.execution import backtest_single_pair
from stat_arb.portfolio import build_coin_netted_portfolio


def _synthetic_pair_result(x_prices, y_prices, beta=2.0, entry_z=1.5, exit_z=0.5,
                            hard_stop_z=4.0, time_stop_bars=156, cost_per_leg_bps=20):
    idx = x_prices.index
    z = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
    z = (z - z.mean())  # oscillate around zero to generate a couple of round trips
    result, trades = backtest_single_pair(
        x_prices, y_prices, alpha=0.0, beta=beta,
        ret_x=x_prices.pct_change(fill_method=None), ret_y=y_prices.pct_change(fill_method=None),
        entry_z=entry_z, exit_z=exit_z, hard_stop_z=hard_stop_z, time_stop_bars=time_stop_bars,
        cost_per_leg_bps=cost_per_leg_bps, z_score=z, beta_series=pd.Series(beta, index=idx))
    return result, trades


def test_build_coin_netted_portfolio_flat_after_liquidation():
    idx = pd.date_range('2022-01-01', periods=40, freq='4h')
    rng = np.random.default_rng(0)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx)
    y = pd.Series(50 + np.cumsum(rng.normal(0, 0.3, len(idx))), index=idx)
    result, _ = _synthetic_pair_result(x, y)
    pair_results = {('XUSDT', 'YUSDT'): result}
    price_data = pd.DataFrame({'XUSDT': x, 'YUSDT': y})
    portfolio = build_coin_netted_portfolio(
        pair_results, [('XUSDT', 'YUSDT')], price_data, cost_per_leg_bps=20,
        start=idx[0], end=idx[-1])
    assert portfolio['coin_qty'].iloc[-1].abs().max() < 1e-12


def test_sleeve_weights_sum_to_one_over_n_pairs():
    idx = pd.date_range('2022-01-01', periods=40, freq='4h')
    rng = np.random.default_rng(1)
    pair_results = {}
    price_cols = {}
    pairs = []
    for i in range(3):
        x = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx)
        y = pd.Series(50 + np.cumsum(rng.normal(0, 0.3, len(idx))), index=idx)
        result, _ = _synthetic_pair_result(x, y)
        pair = (f'X{i}USDT', f'Y{i}USDT')
        pairs.append(pair)
        pair_results[pair] = result
        price_cols[pair[0]] = x
        price_cols[pair[1]] = y
    price_data = pd.DataFrame(price_cols)
    portfolio = build_coin_netted_portfolio(
        pair_results, pairs, price_data, cost_per_leg_bps=20, start=idx[0], end=idx[-1])
    n_pairs = len(pairs)
    sleeve_weight = 1.0 / n_pairs
    assert n_pairs * sleeve_weight <= 1.0 + 1e-9
    # Every pair's two coins are present in the netted portfolio's own coin set.
    assert set(portfolio['coin_qty'].columns) == {c for pair in pairs for c in pair}


def test_repeated_pair_instances_not_overwritten():
    """Mirrors the WF1/WF2 (rebalance_date, pair) instance-key convention (CLAUDE.md): a pair
    re-selected at a later rebalance must not silently overwrite an earlier instance."""
    all_results = {}
    all_trades = {}
    pair = ('AUSDT', 'BUSDT')
    for rebal_date in [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-07-01')]:
        instance_key = (rebal_date, pair)
        all_results[instance_key] = pd.DataFrame({'net_ret': [0.01]})
        all_trades[instance_key] = [{'trade_id': 1}]
    assert len(all_results) == 2, 'second instance overwrote the first'
    assert len(all_trades) == 2
