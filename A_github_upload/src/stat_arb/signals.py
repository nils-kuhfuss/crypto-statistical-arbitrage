"""Trading signal generation: rolling/static spread and Z-score, entry/exit/stop state machine.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 7, 9).
"""
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .statistics import compute_spread


class TradeState(Enum):
    FLAT = 0
    LONG_SPREAD = 1   # long Y, short beta*X
    SHORT_SPREAD = 2  # short Y, long beta*X
    LOCKED = 3        # flat cooldown after stop/missing-data exit


def gen_rolling_signal(x_price: pd.Series, y_price: pd.Series, window_bars: int, rolling_beta: bool = True,
                        frozen_alpha: Optional[float] = None, frozen_beta: Optional[float] = None):
    """Compute alpha/beta (static or rolling) and a rolling-normalized Z-score for one pair.

    rolling_beta=False: alpha/beta are held fixed at `frozen_alpha`/`frozen_beta` -- Model RZ
        must reuse the exact formation-period values from Model S so the comparison isolates
        only the rolling-normalization effect, with an identical hedge ratio at every bar.
    rolling_beta=True: alpha/beta are recomputed at every bar from the trailing window
        (Model RB).

    In both cases the Z-score's mean/std are recomputed on the trailing rolling window.

    min_periods=window_bars (a full window is required, not a partial one), so beta, alpha,
    and the spread mean/std are all computed consistently once warmed up, at the cost of more
    leading NaNs at the start of each rolling series.

    Point-in-time: `.rolling(window_bars, min_periods=window_bars)` only ever looks backward
    from the current bar, so the value at bar t depends only on bars up to and including t.
    """
    log_x, log_y = np.log(x_price), np.log(y_price)

    if rolling_beta:
        roll_cov = log_x.rolling(window=window_bars, min_periods=window_bars).cov(log_y)
        roll_var = log_x.rolling(window=window_bars, min_periods=window_bars).var()
        beta_series = roll_cov / roll_var
        alpha_series = (log_y.rolling(window=window_bars, min_periods=window_bars).mean()
                         - beta_series * log_x.rolling(window=window_bars, min_periods=window_bars).mean())
    else:
        if frozen_alpha is None or frozen_beta is None:
            raise ValueError(
                'gen_rolling_signal(rolling_beta=False) requires frozen_alpha/frozen_beta: '
                'Model RZ must reuse the frozen formation-period alpha/beta, not re-fit its own.')
        alpha_series = pd.Series(frozen_alpha, index=log_x.index)
        beta_series = pd.Series(frozen_beta, index=log_x.index)

    spread = compute_spread(log_y, log_x, alpha_series, beta_series)
    spread_mean = spread.rolling(window=window_bars, min_periods=window_bars).mean()
    spread_std = spread.rolling(window=window_bars, min_periods=window_bars).std()
    z_score = (spread - spread_mean) / spread_std

    return alpha_series, beta_series, spread, z_score


def run_trade_state_machine(z_score: pd.Series, entry_z: float, exit_z: float, hard_stop_z: float,
                             time_stop_bars: int):
    """Generate signal states/trades using close-t information only.

    Point-in-time: at bar i, only `z_score.iloc[i]` and `z_score.iloc[i-1]` (via `z_prev`) are
    read -- no future bar is inspected.

    Trade bars returned here are SIGNAL bars. `backtest_single_pair()` converts them to
    execution bars by adding one. Entry requires a fresh threshold crossing that does not
    land beyond the hard stop. Hard/time/missing-data exits enter LOCKED until the spread
    re-arms at |z| <= exit_z. New entries are disabled during the last two signal bars so
    every accepted trade has time to execute and, if necessary, liquidate inside the sample.
    """
    idx = z_score.index
    state = TradeState.FLAT
    state_series = pd.Series(TradeState.FLAT, index=idx, dtype=object)
    trades = []
    entry_bar = entry_time = direction = None
    trade_id = 0
    z_prev = np.nan

    for i, ts in enumerate(idx):
        z = z_score.iloc[i]

        if state == TradeState.FLAT:
            can_complete_inside_sample = i < len(idx) - 2
            if can_complete_inside_sample and pd.notna(z) and pd.notna(z_prev):
                crossed_up = (z_prev <= entry_z) and (entry_z < z <= hard_stop_z)
                crossed_down = (z_prev >= -entry_z) and (-hard_stop_z <= z < -entry_z)
                if crossed_up:
                    state, direction = TradeState.SHORT_SPREAD, 'short_spread'
                    entry_bar, entry_time = i, ts
                    trade_id += 1
                elif crossed_down:
                    state, direction = TradeState.LONG_SPREAD, 'long_spread'
                    entry_bar, entry_time = i, ts
                    trade_id += 1

        elif state == TradeState.LOCKED:
            if pd.notna(z) and abs(z) <= exit_z:
                state = TradeState.FLAT

        else:
            duration = i - entry_bar
            exit_reason = None
            if pd.isna(z):
                exit_reason = 'force_close_missing_data'
            elif abs(z) > hard_stop_z:
                exit_reason = 'hard_stop'
            elif abs(z) <= exit_z:
                exit_reason = 'exit'
            elif duration >= time_stop_bars:
                exit_reason = 'time_stop'
            elif i >= len(idx) - 2:
                exit_reason = 'end_of_sample'

            if exit_reason is not None:
                trades.append({
                    'trade_id': trade_id,
                    'direction': direction,
                    'entry_bar': entry_bar, 'entry_time': entry_time,
                    'exit_bar': i, 'exit_time': ts,
                    'duration_bars': i - entry_bar,
                    'exit_reason': exit_reason,
                })
                state = (TradeState.LOCKED if exit_reason in
                         ('hard_stop', 'time_stop', 'force_close_missing_data')
                         else TradeState.FLAT)
                direction = None
                entry_bar = entry_time = None

        state_series.iloc[i] = state
        if pd.notna(z):
            z_prev = z

    assert state not in (TradeState.LONG_SPREAD, TradeState.SHORT_SPREAD), (
        'Open signal state survived the penultimate-bar liquidation guard.')
    return state_series, trades


def state_to_position(state_series: pd.Series, beta_series: pd.Series, x_name: str, y_name: str) -> pd.DataFrame:
    """Diagnostic signal targets only; actual holdings use fixed quantities (execution.py)."""
    w_x = pd.Series(0.0, index=state_series.index)
    w_y = pd.Series(0.0, index=state_series.index)
    is_short = state_series == TradeState.SHORT_SPREAD
    is_long = state_series == TradeState.LONG_SPREAD
    w_x[is_short], w_y[is_short] = beta_series[is_short], -1.0
    w_x[is_long], w_y[is_long] = -beta_series[is_long], 1.0
    return pd.DataFrame({x_name: w_x, y_name: w_y})
