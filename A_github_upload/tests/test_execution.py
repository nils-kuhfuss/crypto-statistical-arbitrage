"""Execution engine regression tests, ported from `legacy/Stat_Arb_cont_res_1.ipynb`
Sections 9 and 11's inline synthetic-data checks.

Point-in-time / correctness properties tested: signal precedes execution by exactly one bar,
fixed quantities do not implicitly rebalance intra-trade, costs are charged exactly once per
execution (entry and exit), and trade-level P&L reconciles exactly with the bar-level ledger.
"""
import numpy as np
import pandas as pd
import pytest

from stat_arb.execution import backtest_single_pair, compute_trade_pnl
from stat_arb.signals import TradeState, run_trade_state_machine

COST_PER_LEG_BPS = 20


class PairPosition:
    """Minimal isolation container, ported from the notebook's Section 11 regression cell --
    not part of the real execution path (execute_fixed_quantity_pair manages its own local
    state); kept here only to reproduce the original pair-isolation sanity check."""
    def __init__(self, x_name, y_name, index):
        self.x_name, self.y_name, self.index = x_name, y_name, index
        self.q_x = pd.Series(np.nan, index=index, name=x_name)
        self.q_y = pd.Series(np.nan, index=index, name=y_name)

    def as_frame(self):
        return pd.DataFrame({self.x_name: self.q_x, self.y_name: self.q_y})


def test_pair_quantity_isolation():
    idx = pd.date_range('2021-01-01', periods=5, freq='4h')
    p1 = PairPosition('AAAUSDT', 'GRTUSDT', idx)
    p2 = PairPosition('GRTUSDT', 'BBBUSDT', idx)
    p1.q_y.iloc[2] = -3.0
    p2.q_x.iloc[2] = 2.0
    assert p1.q_y.iloc[2] == -3.0 and p2.q_x.iloc[2] == 2.0


def test_state_machine_normal_round_trip():
    z1 = pd.Series([0.2, 1.8, 1.9, 1.6, 0.4, 0.1], index=pd.date_range('2022-01-01', periods=6, freq='4h'))
    states1, trades1 = run_trade_state_machine(z1, 1.5, 0.5, 4.0, 156)
    assert len(trades1) == 1 and trades1[0]['direction'] == 'short_spread'
    assert trades1[0]['exit_reason'] == 'exit' and trades1[0]['duration_bars'] == 3


def test_state_machine_no_pyramiding():
    z2 = pd.Series([0.1, 1.6, 1.7, 1.8, 1.9, 0.3], index=pd.date_range('2022-01-01', periods=6, freq='4h'))
    _, trades2 = run_trade_state_machine(z2, 1.5, 0.5, 4.0, 156)
    assert len(trades2) == 1, 'sustained threshold breach must not pyramid'


def test_state_machine_hard_stop():
    z3 = pd.Series([0.1, 1.6, 4.5, 0.2], index=pd.date_range('2022-01-01', periods=4, freq='4h'))
    _, trades3 = run_trade_state_machine(z3, 1.5, 0.5, 4.0, 156)
    assert trades3[0]['exit_reason'] == 'hard_stop'


def test_state_machine_time_stop():
    z4 = pd.Series([0.1] + [1.6] * 10, index=pd.date_range('2022-01-01', periods=11, freq='4h'))
    _, trades4 = run_trade_state_machine(z4, 1.5, 0.5, 4.0, 5)
    assert trades4[0]['exit_reason'] == 'time_stop' and trades4[0]['duration_bars'] == 5


def test_state_machine_entry_rejected_beyond_hard_stop():
    z5 = pd.Series([0.1] + [7.0] * 20, index=pd.date_range('2022-01-01', periods=21, freq='4h'))
    states5, trades5 = run_trade_state_machine(z5, 1.5, 0.5, 4.0, 156)
    assert len(trades5) == 0 and (states5 == TradeState.FLAT).all()


def test_state_machine_ordinary_exit():
    z5c = pd.Series([0.1, 2.0, 2.1, 0.3, 0.2], index=pd.date_range('2022-01-01', periods=5, freq='4h'))
    _, trades5c = run_trade_state_machine(z5c, 1.5, 0.5, 4.0, 156)
    assert len(trades5c) == 1 and trades5c[0]['exit_reason'] == 'exit'


def test_state_machine_locked_rearm_then_reentry():
    z5b = pd.Series([0.1, 2.0, 4.5, 0.2, 0.3, 1.8, 1.9, 0.2, 0.1, 0.1],
                     index=pd.date_range('2022-01-01', periods=10, freq='4h'))
    _, trades5b = run_trade_state_machine(z5b, 1.5, 0.5, 4.0, 156)
    assert len(trades5b) == 2 and trades5b[0]['exit_reason'] == 'hard_stop'


def test_state_machine_end_of_sample_forced_exit():
    z6 = pd.Series([0.1, 1.8, 1.9, 1.7], index=pd.date_range('2022-01-01', periods=4, freq='4h'))
    states6, trades6 = run_trade_state_machine(z6, 1.5, 0.5, 4.0, 156)
    assert trades6[-1]['exit_reason'] == 'end_of_sample'
    assert trades6[-1]['exit_bar'] == len(z6) - 2
    assert (states6.iloc[-2:] == TradeState.FLAT).all()


def test_state_machine_missing_data_forced_close():
    z7 = pd.Series([0.1, 1.8, 1.9, np.nan, np.nan, np.nan],
                    index=pd.date_range('2022-01-01', periods=6, freq='4h'))
    states7, trades7 = run_trade_state_machine(z7, 1.5, 0.5, 4.0, 156)
    assert len(trades7) == 1 and trades7[0]['exit_reason'] == 'force_close_missing_data'
    assert trades7[0]['exit_bar'] == 3 and (states7.iloc[3:] == TradeState.LOCKED).all()


def test_state_machine_signal_too_close_to_sample_end_rejected():
    z8 = pd.Series([0.1, 0.2, 1.8], index=pd.date_range('2022-01-01', periods=3, freq='4h'))
    _, trades8 = run_trade_state_machine(z8, 1.5, 0.5, 4.0, 156)
    assert len(trades8) == 0, 'a signal without room for next-bar entry and liquidation is invalid'


@pytest.fixture
def synth_backtest():
    idx = pd.date_range('2022-01-01', periods=8, freq='4h')
    z = pd.Series([0.1, 1.8, 1.9, 1.8, 1.6, 0.3, 0.2, 0.1], index=idx)
    x = pd.Series([100., 101., 102., 100., 103., 104., 105., 106.], index=idx)
    y = pd.Series([50., 50., 49., 48., 49., 50., 51., 52.], index=idx)
    result, trades = backtest_single_pair(
        x, y, alpha=0.0, beta=2.0,
        ret_x=x.pct_change(fill_method=None), ret_y=y.pct_change(fill_method=None),
        entry_z=1.5, exit_z=0.5, hard_stop_z=4.0, time_stop_bars=156,
        cost_per_leg_bps=COST_PER_LEG_BPS, z_score=z,
        beta_series=pd.Series(2.0, index=idx))
    return idx, x, y, result, trades


def test_next_bar_execution_mapping(synth_backtest):
    idx, x, y, result, trades = synth_backtest
    assert len(trades) == 1
    trade = trades[0]
    assert trade['signal_entry_bar'] == 1 and trade['entry_bar'] == 2
    assert trade['signal_exit_bar'] == 5 and trade['exit_bar'] == 6


def test_fixed_quantity_no_intratrade_rebalancing(synth_backtest):
    idx, x, y, result, trades = synth_backtest
    expected_qx = (2 / 3) / x.iloc[2]
    expected_qy = (-1 / 3) / y.iloc[2]
    expected_qty_x = pd.Series([0., 0., expected_qx, expected_qx, expected_qx, expected_qx, 0., 0.], index=idx)
    expected_qty_y = pd.Series([0., 0., expected_qy, expected_qy, expected_qy, expected_qy, 0., 0.], index=idx)
    pd.testing.assert_series_equal(result['qty_x'], expected_qty_x, check_names=False, atol=1e-12)
    pd.testing.assert_series_equal(result['qty_y'], expected_qty_y, check_names=False, atol=1e-12)
    assert (result['turnover_cash'].iloc[3:6] == 0).all()


def test_cash_pnl_and_cost_reconciliation(synth_backtest):
    idx, x, y, result, trades = synth_backtest
    expected_qx = (2 / 3) / x.iloc[2]
    expected_qy = (-1 / 3) / y.iloc[2]
    rate = COST_PER_LEG_BPS * 1e-4

    expected_gross_pnl = pd.Series(0.0, index=idx)
    for i in range(3, 7):
        expected_gross_pnl.iloc[i] = (expected_qx * (x.iloc[i] - x.iloc[i-1]) +
                                       expected_qy * (y.iloc[i] - y.iloc[i-1]))
    expected_cost_cash = pd.Series(0.0, index=idx)
    expected_cost_cash.iloc[2] = rate * (abs(expected_qx) * x.iloc[2] + abs(expected_qy) * y.iloc[2])
    expected_cost_cash.iloc[6] = rate * (abs(expected_qx) * x.iloc[6] + abs(expected_qy) * y.iloc[6])

    expected_equity = pd.Series(index=idx, dtype=float)
    eq = 1.0
    for i in range(len(idx)):
        eq += expected_gross_pnl.iloc[i] - expected_cost_cash.iloc[i]
        expected_equity.iloc[i] = eq

    pd.testing.assert_series_equal(result['gross_pnl_cash'], expected_gross_pnl, check_names=False, atol=1e-12)
    pd.testing.assert_series_equal(result['cost_cash'], expected_cost_cash, check_names=False, atol=1e-12)
    pd.testing.assert_series_equal(result['equity'], expected_equity, check_names=False, atol=1e-12)
    assert result['cost_cash'].iloc[6] > 0 and result['qty_x'].iloc[6] == 0


def test_trade_level_pnl_reconciles_with_ledger(synth_backtest):
    idx, x, y, result, trades = synth_backtest
    expected_qx = (2 / 3) / x.iloc[2]
    expected_qy = (-1 / 3) / y.iloc[2]
    rate = COST_PER_LEG_BPS * 1e-4
    expected_gross_pnl = pd.Series(0.0, index=idx)
    for i in range(3, 7):
        expected_gross_pnl.iloc[i] = (expected_qx * (x.iloc[i] - x.iloc[i-1]) +
                                       expected_qy * (y.iloc[i] - y.iloc[i-1]))
    expected_cost_cash = pd.Series(0.0, index=idx)
    expected_cost_cash.iloc[2] = rate * (abs(expected_qx) * x.iloc[2] + abs(expected_qy) * y.iloc[2])
    expected_cost_cash.iloc[6] = rate * (abs(expected_qx) * x.iloc[6] + abs(expected_qy) * y.iloc[6])
    expected_equity = pd.Series(index=idx, dtype=float)
    eq = 1.0
    for i in range(len(idx)):
        eq += expected_gross_pnl.iloc[i] - expected_cost_cash.iloc[i]
        expected_equity.iloc[i] = eq

    g, n, c = compute_trade_pnl(trades[0], result)
    expected_g = expected_gross_pnl.iloc[2:7].sum()
    expected_n = expected_equity.iloc[6] - 1.0
    expected_c = expected_cost_cash.iloc[2:7].sum()
    assert abs(g - expected_g) < 1e-12
    assert abs(n - expected_n) < 1e-12
    assert abs(c - expected_c) < 1e-12
