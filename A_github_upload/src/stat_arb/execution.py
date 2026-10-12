"""Trade execution: next-bar execution mapping, fixed-quantity cash ledger, trade P&L.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Section 10-11).

Note: the notebook's `PairPosition` class (Section 11) was test-only scaffolding used by a
single regression cell to demonstrate that two pairs sharing a coin keep fully independent
quantity series -- it is not part of the real execution path (`execute_fixed_quantity_pair`
manages its own local state) and is therefore ported to `tests/` instead of here.
"""
import numpy as np
import pandas as pd

from .signals import TradeState


def _entry_weights(direction: str, beta: float):
    """Signed entry notionals under spread = Y - alpha - beta*X, gross normalized to one."""
    denom = 1.0 + abs(beta)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f'Invalid beta at entry: {beta}')
    if direction == 'short_spread':
        return beta / denom, -1.0 / denom
    if direction == 'long_spread':
        return -beta / denom, 1.0 / denom
    raise ValueError(f'Unknown direction: {direction}')


def _execution_trades(signal_trades, index):
    """Map signal bars to next-bar executions and preserve both timestamps."""
    executed = []
    for tr in signal_trades:
        entry_exec = tr['entry_bar'] + 1
        exit_exec = tr['exit_bar'] + 1
        if entry_exec >= len(index) or exit_exec >= len(index):
            raise AssertionError(f'Trade cannot execute inside sample: {tr}')
        ex = dict(tr)
        ex.update({
            'signal_entry_bar': tr['entry_bar'], 'signal_entry_time': tr['entry_time'],
            'signal_exit_bar': tr['exit_bar'], 'signal_exit_time': tr['exit_time'],
            'signal_duration_bars': tr['duration_bars'],
            'entry_bar': entry_exec, 'entry_time': index[entry_exec],
            'exit_bar': exit_exec, 'exit_time': index[exit_exec],
            'duration_bars': exit_exec - entry_exec,
        })
        executed.append(ex)
    return executed


def execute_fixed_quantity_pair(x_price: pd.Series, y_price: pd.Series, beta_series: pd.Series,
                                 signal_state: pd.Series, signal_trades, cost_per_leg_bps: float,
                                 initial_equity: float = 1.0):
    """Cash ledger with next-bar execution and fixed quantities between executions.

    At entry, the current pair equity is allocated across the two legs using the signal-bar
    beta. Coin quantities are then held unchanged. Each bar marks the existing quantities
    to market before any order at that bar is executed. Trading cost is charged immediately
    as `abs(delta_quantity) * execution_price * bps`, including the last-bar liquidation.
    """
    idx = x_price.index
    x_price = x_price.reindex(idx).astype(float)
    y_price = y_price.reindex(idx).astype(float)
    beta_series = beta_series.reindex(idx)
    signal_state = signal_state.reindex(idx)
    trades = _execution_trades(signal_trades, idx)
    entries = {tr['entry_bar']: tr for tr in trades}
    exits = {tr['exit_bar']: tr for tr in trades}
    rate = cost_per_leg_bps * 1e-4

    # Valuation fallback is used only for a forced missing-data liquidation. It is explicit
    # and flagged in the trade record; a normal entry always requires a current tradable price.
    x_mark = x_price.ffill()
    y_mark = y_price.ffill()

    numeric_columns = ['qty_x', 'qty_y', 'notional_x', 'notional_y', 'w_x', 'w_y',
                       'turnover_cash', 'turnover', 'cost_cash', 'costs',
                       'gross_pnl_cash', 'net_pnl_cash', 'gross_ret', 'net_ret',
                       'equity_before', 'equity', 'gross_equity', 'gross_exposure']
    # Populate NumPy arrays inside the loop, then construct one DataFrame -- repeated
    # DataFrame.loc writes make a full multi-pair Run All needlessly slow.
    values = {name: np.zeros(len(idx), dtype=float) for name in numeric_columns}
    state_values = np.empty(len(idx), dtype=object)

    q_x = q_y = 0.0
    direction = None
    equity = gross_equity = float(initial_equity)

    for i, ts in enumerate(idx):
        prev_equity = equity
        gross_pnl = 0.0
        if i > 0 and (q_x != 0.0 or q_y != 0.0):
            gross_pnl = (q_x * (x_mark.iloc[i] - x_mark.iloc[i-1]) +
                         q_y * (y_mark.iloc[i] - y_mark.iloc[i-1]))
            if not np.isfinite(gross_pnl):
                raise ValueError(f'Cannot mark open position at {ts}.')

        equity_before_trade = prev_equity + gross_pnl
        target_q_x, target_q_y = q_x, q_y
        next_direction = direction

        if i in exits:
            target_q_x = target_q_y = 0.0
            next_direction = None

        if i in entries:
            tr = entries[i]
            if pd.isna(x_price.iloc[i]) or pd.isna(y_price.iloc[i]):
                raise ValueError(f'Entry execution has no tradable price at {ts}: {tr}')
            beta_entry = float(beta_series.iloc[tr['signal_entry_bar']])
            target_w_x, target_w_y = _entry_weights(tr['direction'], beta_entry)
            capital = max(equity_before_trade, 0.0)
            target_q_x = target_w_x * capital / x_price.iloc[i]
            target_q_y = target_w_y * capital / y_price.iloc[i]
            next_direction = tr['direction']
            tr.update({'entry_beta': beta_entry, 'entry_equity_before': equity_before_trade,
                       'entry_price_x': x_price.iloc[i], 'entry_price_y': y_price.iloc[i],
                       'entry_qty_x': target_q_x, 'entry_qty_y': target_q_y})

        exec_px_x, exec_px_y = x_mark.iloc[i], y_mark.iloc[i]
        delta_q_x, delta_q_y = target_q_x - q_x, target_q_y - q_y
        # 0 * NaN is NaN in floating-point arithmetic. Before a coin's listing both the
        # order and position are zero, so the economically correct order notional is zero.
        turnover_x = 0.0 if delta_q_x == 0.0 else abs(delta_q_x) * exec_px_x
        turnover_y = 0.0 if delta_q_y == 0.0 else abs(delta_q_y) * exec_px_y
        turnover_cash = turnover_x + turnover_y
        cost_cash = turnover_cash * rate
        equity = equity_before_trade - cost_cash
        gross_equity += gross_pnl
        net_pnl = gross_pnl - cost_cash

        if i in entries:
            entries[i]['entry_cost_cash'] = cost_cash
        if i in exits:
            exits[i].update({'exit_equity_after': equity, 'exit_price_x': exec_px_x,
                             'exit_price_y': exec_px_y, 'exit_cost_cash': cost_cash,
                             'stale_exit_price': bool(pd.isna(x_price.iloc[i]) or pd.isna(y_price.iloc[i]))})

        q_x, q_y, direction = target_q_x, target_q_y, next_direction
        notional_x = 0.0 if q_x == 0.0 else q_x * x_mark.iloc[i]
        notional_y = 0.0 if q_y == 0.0 else q_y * y_mark.iloc[i]
        denom = equity if equity != 0 else np.nan
        gross_exposure = (abs(notional_x) + abs(notional_y)) / denom

        row = [q_x, q_y, notional_x, notional_y, notional_x / denom, notional_y / denom,
               turnover_cash, turnover_cash / prev_equity, cost_cash, cost_cash / prev_equity,
               gross_pnl, net_pnl, gross_pnl / prev_equity, net_pnl / prev_equity,
               prev_equity, equity, gross_equity, gross_exposure]
        for name, value in zip(numeric_columns, row):
            values[name][i] = value
        state_values[i] = (TradeState.SHORT_SPREAD if direction == 'short_spread'
                           else TradeState.LONG_SPREAD if direction == 'long_spread'
                           else TradeState.FLAT)

    assert abs(q_x) < 1e-14 and abs(q_y) < 1e-14, 'Pair is not flat after final execution.'
    out = pd.DataFrame(values, index=idx)
    out['state'] = pd.Series(state_values, index=idx, dtype=object)
    out['signal_state'] = signal_state
    return out, trades


def backtest_single_pair(x_price: pd.Series, y_price: pd.Series, alpha: float, beta: float,
                          ret_x: pd.Series, ret_y: pd.Series,
                          entry_z: float, exit_z: float, hard_stop_z: float, time_stop_bars: int,
                          cost_per_leg_bps: float,
                          z_score: pd.Series = None, alpha_series: pd.Series = None,
                          beta_series: pd.Series = None, state_start=None):
    """Single-pair path: signal generation -> next-bar execution -> fixed-quantity cash P&L.

    state_start, if given, restricts the trade state machine to bars at or after that date --
    the signal itself still uses full history for its rolling/formation statistics, but no
    position can be open before state_start and the state machine is seeded FLAT there. This
    is used for both Validation and Final OOS evaluation so each run begins with zero exposure
    while still allowing prior history to inform the signal.
    """
    # Imported here (not at module top) to avoid a circular import: signals.py does not
    # depend on execution.py, but backtest_single_pair needs both layers.
    from .signals import run_trade_state_machine
    from .statistics import compute_spread

    log_x, log_y = np.log(x_price), np.log(y_price)
    if z_score is None:
        spread = compute_spread(log_y, log_x, alpha, beta)
        z_score = (spread - spread.mean()) / spread.std()
    else:
        spread = compute_spread(log_y, log_x, alpha, beta)

    beta_for_entry = (beta_series.reindex(z_score.index) if beta_series is not None
                      else pd.Series(beta, index=z_score.index))
    z_for_state = z_score
    if state_start is not None:
        z_for_state = z_score.where(z_score.index >= pd.Timestamp(state_start))

    signal_state, signal_trades = run_trade_state_machine(
        z_for_state, entry_z, exit_z, hard_stop_z, time_stop_bars)
    result, trades = execute_fixed_quantity_pair(
        x_price, y_price, beta_for_entry, signal_state, signal_trades,
        cost_per_leg_bps=cost_per_leg_bps, initial_equity=1.0)
    result['z_score'] = z_score
    result['spread'] = spread
    return result, trades


def compute_trade_pnl(trade: dict, result_df: pd.DataFrame):
    """Cash P&L for one executed trade, including both execution-bar costs."""
    start, end = trade['entry_bar'], trade['exit_bar']
    capital = float(result_df['equity_before'].iloc[start])
    gross_pnl = result_df['gross_pnl_cash'].iloc[start:end+1].sum() / capital
    net_pnl = result_df['equity'].iloc[end] / capital - 1.0
    total_costs = result_df['cost_cash'].iloc[start:end+1].sum() / capital
    return gross_pnl, net_pnl, total_costs


def run_model_on_pair(x_name: str, y_name: str, model: str, window_bars: int, entry_z: float,
                       exit_z: float, hard_stop_z: float, time_stop_bars: int, cost_per_leg_bps: float,
                       price_data: pd.DataFrame, ret_data: pd.DataFrame, config: dict, state_start=None):
    """model in {'S', 'RZ', 'RB'}. state_start is forwarded to backtest_single_pair -- see its
    docstring. Model S's alpha/beta are fit on formation data only, then applied unchanged to
    the full price series; Model RZ reuses this exact frozen (alpha, beta), so RZ vs. S
    isolates only the rolling-Z-normalization effect. Model RB estimates its own rolling beta.
    """
    from .data import get_tradable_price
    from .signals import gen_rolling_signal
    from .statistics import compute_spread, fit_static_ols

    x_price = get_tradable_price(price_data[x_name])
    y_price = get_tradable_price(price_data[y_name])
    ret_x = x_price.pct_change(fill_method=None)
    ret_y = y_price.pct_change(fill_method=None)

    # Formation-only fit, shared by Model S and Model RZ.
    log_x_form = np.log(x_price.loc[config['FORMATION_START']:config['FORMATION_END']])
    log_y_form = np.log(y_price.loc[config['FORMATION_START']:config['FORMATION_END']])
    fit_data_form = pd.concat([log_x_form, log_y_form], axis=1).dropna()
    fit_data_form.columns = ['x', 'y']
    a0, b0, _ = fit_static_ols(fit_data_form['y'], fit_data_form['x'])

    log_x, log_y = np.log(x_price), np.log(y_price)

    if model == 'S':
        spread = compute_spread(log_y, log_x, a0, b0)
        spread_mean = spread.loc[config['FORMATION_START']:config['FORMATION_END']].mean()
        spread_std = spread.loc[config['FORMATION_START']:config['FORMATION_END']].std()
        z_score = (spread - spread_mean) / spread_std
        beta_series = pd.Series(b0, index=x_price.index)
        alpha_used, beta_used = a0, b0
    elif model == 'RZ':
        alpha_series, beta_series, spread, z_score = gen_rolling_signal(
            x_price, y_price, window_bars, rolling_beta=False,
            frozen_alpha=a0, frozen_beta=b0)
        alpha_used, beta_used = a0, b0
    elif model == 'RB':
        alpha_series, beta_series, spread, z_score = gen_rolling_signal(
            x_price, y_price, window_bars, rolling_beta=True)
        alpha_used, beta_used = 0.0, 1.0  # unused: backtest_single_pair only needs alpha/beta
                                            # scalars for a diagnostic reference spread when
                                            # z_score/beta_series are supplied directly
    else:
        raise ValueError(f'Unknown model: {model}')

    # Force-close if either leg's tradable price is unavailable.
    untradable = x_price.isna() | y_price.isna()
    z_score = z_score.where(~untradable, np.nan)

    result, trades = backtest_single_pair(
        x_price, y_price, alpha_used, beta_used,
        ret_x, ret_y, entry_z=entry_z, exit_z=exit_z, hard_stop_z=hard_stop_z,
        time_stop_bars=time_stop_bars, cost_per_leg_bps=cost_per_leg_bps,
        z_score=z_score, beta_series=beta_series, state_start=state_start)

    return result, trades


def run_s1_on_pair(x_name: str, y_name: str, time_stop_bars: int, price_data: pd.DataFrame,
                    ret_data: pd.DataFrame, config: dict, state_start=None):
    """S1 = S0's Model-S signal/backtest path (formation-fit alpha/beta, formation-frozen
    calibration) with only the time stop replaced. Everything else -- entry/exit/hard-stop
    thresholds, execution delay, cost assumptions, position sizing -- is taken from `config`,
    identical to S0."""
    from .data import get_tradable_price
    from .statistics import compute_spread, fit_static_ols

    x_price = get_tradable_price(price_data[x_name])
    y_price = get_tradable_price(price_data[y_name])
    ret_x = x_price.pct_change(fill_method=None)
    ret_y = y_price.pct_change(fill_method=None)

    log_x_form = np.log(x_price.loc[config['FORMATION_START']:config['FORMATION_END']])
    log_y_form = np.log(y_price.loc[config['FORMATION_START']:config['FORMATION_END']])
    fit_data_form = pd.concat([log_x_form, log_y_form], axis=1).dropna()
    fit_data_form.columns = ['x', 'y']
    a0, b0, _ = fit_static_ols(fit_data_form['y'], fit_data_form['x'])

    log_x, log_y = np.log(x_price), np.log(y_price)
    spread = compute_spread(log_y, log_x, a0, b0)
    spread_mean = spread.loc[config['FORMATION_START']:config['FORMATION_END']].mean()
    spread_std = spread.loc[config['FORMATION_START']:config['FORMATION_END']].std()
    z_score = (spread - spread_mean) / spread_std
    beta_series = pd.Series(b0, index=x_price.index)

    untradable = x_price.isna() | y_price.isna()
    z_score = z_score.where(~untradable, np.nan)

    result, trades = backtest_single_pair(
        x_price, y_price, a0, b0, ret_x, ret_y,
        entry_z=config['ENTRY_Z'], exit_z=config['EXIT_Z'], hard_stop_z=config['HARD_STOP_Z'],
        time_stop_bars=time_stop_bars, cost_per_leg_bps=config['COST_PER_LEG_BPS'],
        z_score=z_score, beta_series=beta_series, state_start=state_start)
    return result, trades


def run_s1_universe(pairs, time_stops: dict, price_data: pd.DataFrame, ret_data: pd.DataFrame,
                     config: dict, state_start):
    """Run S1 on every eligible pair (pairs missing from `time_stops` are skipped -- they
    were marked ineligible upstream) and return per-pair results/trades keyed by (x, y)."""
    results, trades_d = {}, {}
    for x_name, y_name in pairs:
        if (x_name, y_name) not in time_stops:
            continue
        result, trades = run_s1_on_pair(
            x_name, y_name, time_stops[(x_name, y_name)], price_data, ret_data, config, state_start)
        results[(x_name, y_name)] = result
        trades_d[(x_name, y_name)] = trades
    return results, trades_d
