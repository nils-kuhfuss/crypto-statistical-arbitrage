"""4h-to-daily reconciliation, Sharpe/vol/drawdown, BTC benchmark regression.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 13, 16, 17).

Two functions (`bars_per_year_sharpe`, `build_btc_daily_benchmark`) originally defaulted a
parameter from the notebook's global `CONFIG`/`px` (e.g. `price_data=px`). Every real call
site in the notebook already passed its own value explicitly (S0 passed `px`, WF1/WF2 passed
`WF_PX`), so the default was unused in practice but was still a hidden-global-state risk (a
future call site omitting the argument would silently fall back to the wrong universe). Per
REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md Section 10 ("avoid hidden global state, prefer explicit
inputs/outputs"), the parameter is now required with no default -- this does not change any
existing call's behavior, since all of them already passed the value explicitly.
"""
from typing import Optional

import numpy as np
import pandas as pd


def aggregate_daily_returns(bar_net_ret: pd.Series, calendar_index: Optional[pd.Index] = None):
    """Geometrically compound 4h portfolio returns into UTC daily returns.

    Days with no active position remain in the series with return 0.0 (not dropped), so the
    daily series has one row per UTC calendar day spanned by `bar_net_ret`'s index -- this is
    required for the daily-252 Sharpe convention and for exact alignment with the BTC daily
    benchmark series. If `calendar_index` is given, the output is reindexed onto it (filling
    any day with literally no bars, e.g. a data gap, as 0.0 -- flagged via the `n_bars` count
    so a genuine gap is distinguishable from a flat trading day).
    """
    r = bar_net_ret.fillna(0.0)
    daily_compound = (1.0 + r).groupby(r.index.floor('D')).prod() - 1.0
    n_bars = r.groupby(r.index.floor('D')).size()
    daily_compound.name = 'daily_ret'
    if calendar_index is not None:
        daily_compound = daily_compound.reindex(calendar_index, fill_value=0.0)
        n_bars = n_bars.reindex(calendar_index, fill_value=0)
    return daily_compound, n_bars


def daily_sharpe_252(daily_ret: pd.Series) -> float:
    """Headline Sharpe convention used consistently across every model comparison:
    daily mean return / daily std x sqrt(252). Flat (return=0) days are included in the
    mean/std computation, not dropped -- excluding them would overstate the Sharpe of any
    strategy that is flat a meaningful fraction of the time."""
    r = daily_ret.dropna()
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(252)


def bars_per_year_sharpe(bar_net_ret: pd.Series, bars_per_year: int) -> float:
    """Optional secondary diagnostic: the original 24/7 bar-level annualized Sharpe
    convention. Kept only as a cross-check, never the headline number in any model
    comparison."""
    r = bar_net_ret.dropna()
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() * bars_per_year) / (r.std() * np.sqrt(bars_per_year))


def summarize_model_performance(result: pd.DataFrame, ann_factor: float) -> dict:
    """Bar-level annualized Sharpe/return/vol for one result frame's 'net_ret' column.
    Superseded as the headline convention by `daily_sharpe_252` (see CLAUDE.md "Evaluation
    conventions"); kept as the `bar_sharpe_secondary` diagnostic in the standardized summary."""
    r = result['net_ret'].dropna()
    if len(r) == 0 or r.std() == 0:
        return {'sharpe': np.nan, 'ann_return': np.nan, 'ann_vol': np.nan, 'n_bars': len(r)}
    sharpe = (r.mean() * ann_factor) / (r.std() * np.sqrt(ann_factor))
    return {
        'sharpe': sharpe,
        'ann_return': r.mean() * ann_factor,
        'ann_vol': r.std() * np.sqrt(ann_factor),
        'n_bars': len(r),
    }


def drawdown(returns: pd.Series) -> pd.Series:
    """Cumulative drawdown series from a bar-level net return series."""
    cumulative = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def build_btc_daily_benchmark(calendar_index: pd.Index, price_data: pd.DataFrame) -> pd.Series:
    """Daily BTC return series on the same UTC calendar as the strategy's daily returns,
    built via the same get_tradable_price/geometric-compounding path as the strategy series
    for an apples-to-apples comparison."""
    from .data import get_tradable_price
    btc_price = get_tradable_price(price_data['BTCUSDT'])
    btc_ret = btc_price.pct_change(fill_method=None)
    btc_daily, _ = aggregate_daily_returns(btc_ret, calendar_index=calendar_index)
    return btc_daily


def exit_reason_fractions(trades_dict: dict) -> dict:
    """Fraction of exits by reason, pooled across all pairs. Hard-stop/time-stop/
    mean-reversion buckets map to the state machine's exit_reason values (signals.py);
    force_close_missing_data and end_of_sample are reported separately since they are data
    boundary effects, not economically meaningful exit types."""
    all_reasons = [t['exit_reason'] for trades in trades_dict.values() for t in trades]
    if not all_reasons:
        return {}
    s = pd.Series(all_reasons).value_counts(normalize=True)
    return s.to_dict()


def trade_pnl_stats(trades_dict: dict) -> dict:
    """Win rate, average/median trade return (net_pnl as a fraction of that trade's entry
    equity, i.e. per-pair-sleeve return -- consistent with each pair's own initial_equity=1.0
    convention in execute_fixed_quantity_pair)."""
    all_trades = [t for trades in trades_dict.values() for t in trades]
    if not all_trades:
        return {'win_rate': np.nan, 'avg_trade_return': np.nan, 'median_trade_return': np.nan}
    rets = []
    for t in all_trades:
        entry_eq = t.get('entry_equity_before')
        exit_eq = t.get('exit_equity_after')
        if entry_eq is not None and exit_eq is not None and entry_eq != 0:
            rets.append((exit_eq - entry_eq) / entry_eq)
    if not rets:
        return {'win_rate': np.nan, 'avg_trade_return': np.nan, 'median_trade_return': np.nan}
    rets = np.array(rets)
    return {'win_rate': (rets > 0).mean(), 'avg_trade_return': rets.mean(), 'median_trade_return': np.median(rets)}
