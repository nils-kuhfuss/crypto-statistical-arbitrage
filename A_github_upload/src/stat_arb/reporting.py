"""Standardized model summaries, pair-level contribution -- the single shared output schema
every model version (S0, S1, WF1, WF2) reports through, so numbers are directly comparable.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 16, 23).
"""
import numpy as np
import pandas as pd

from .metrics import (aggregate_daily_returns, build_btc_daily_benchmark, daily_sharpe_252,
                       drawdown, exit_reason_fractions, trade_pnl_stats)
from .statistics import benchmark_regression_stats
from .portfolio import evaluate_coin_netted_portfolio


def standardized_model_summary(model_id: str, pair_results: dict, pairs, price_data: pd.DataFrame,
                                cost_per_leg_bps: float, start, end, bars_per_year: int,
                                trades_dict: dict = None):
    """One row per model, built from build_coin_netted_portfolio + the daily-252/benchmark
    layer -- the single shared summary function every model version (S0, S1, WF1, WF2)
    reports through (S0/S1 call this directly; WF1/WF2 use `standardized_wf1_summary` below,
    built from their own pre-stitched portfolio-level frame instead)."""
    portfolio, metrics = evaluate_coin_netted_portfolio(
        pair_results, pairs, price_data, cost_per_leg_bps, start, end, bars_per_year)
    f = portfolio['frame']
    daily_ret, n_bars_per_day = aggregate_daily_returns(f['net_ret'])
    btc_daily = build_btc_daily_benchmark(daily_ret.index, price_data)
    bench = benchmark_regression_stats(daily_ret, btc_daily)

    if trades_dict is not None:
        all_trades = [t for trades in trades_dict.values() for t in trades]
        durations = [t['duration_bars'] for t in all_trades]
        exit_fracs = exit_reason_fractions(trades_dict)
        pnl_stats = trade_pnl_stats(trades_dict)
    else:
        all_trades, durations, exit_fracs, pnl_stats = [], [], {}, {'win_rate': np.nan, 'avg_trade_return': np.nan, 'median_trade_return': np.nan}

    time_in_market = (f['gross_exposure'] > 0).mean()

    row = {
        'model': model_id,
        'daily_sharpe_252': daily_sharpe_252(daily_ret),
        'bar_sharpe_secondary': metrics['sharpe'],
        'ann_volatility': metrics['ann_vol'],
        'cum_net_return': metrics['cum_return'],
        'cum_gross_return': metrics['gross_cum_return'],
        'max_drawdown': metrics['max_dd'],
        'n_trades': len(all_trades),
        'n_pairs_traded': len({k for k, trades in (trades_dict or {}).items() if trades}),
        'win_rate': pnl_stats['win_rate'],
        'avg_trade_return': pnl_stats['avg_trade_return'],
        'median_trade_return': pnl_stats['median_trade_return'],
        'avg_holding_bars': np.mean(durations) if durations else np.nan,
        'median_holding_bars': np.median(durations) if durations else np.nan,
        'time_in_market_pct': time_in_market,
        'turnover': metrics['total_turnover'],
        'pct_exit_mean_reversion': exit_fracs.get('exit', 0.0),
        'pct_exit_time_stop': exit_fracs.get('time_stop', 0.0),
        'pct_exit_hard_stop': exit_fracs.get('hard_stop', 0.0),
        'total_costs_pct': metrics['total_costs'],
        'btc_correlation': bench['correlation'],
        'btc_beta': bench['beta'],
        'ann_alpha': bench['ann_alpha'],
        'alpha_tstat_ols': bench['alpha_tstat_ols'],
        'alpha_tstat_hac': bench['alpha_tstat_hac'],
        'r_squared': bench['r_squared'],
    }
    return row, portfolio


def standardized_wf1_summary(model_id: str, full_frame: pd.DataFrame, all_results: dict, all_trades: dict,
                              price_data: pd.DataFrame, start, end) -> dict:
    """WF1/WF2-equivalent of standardized_model_summary, built from the pre-stitched
    continuous PORTFOLIO-LEVEL frame (turnover, exposure, time-in-market all come directly
    from full_frame's own columns -- built once during simulation in walkforward.run_wf1_full/
    run_wf2_full -- never reconstructed by summing pair-level outputs after the fact, which
    would double count when multiple pairs are active simultaneously).

    all_trades/all_results are keyed by (rebalance_date, pair) -- a unique pair-INSTANCE key --
    so n_pairs_traded below counts DISTINCT pairs (the pair component of the key), which can
    be lower than the number of instances if the same pair recurs.

    Annualized volatility uses the same daily-252 convention as headline Sharpe, not the
    bar-level sqrt(BARS_PER_YEAR) convention.
    """
    f = full_frame.loc[start:end]
    daily_ret, _ = aggregate_daily_returns(f['net_ret'])
    btc_daily = build_btc_daily_benchmark(daily_ret.index, price_data)
    bench = benchmark_regression_stats(daily_ret, btc_daily)

    all_trades_flat = [t for trades in all_trades.values() for t in trades]
    durations = [t['duration_bars'] for t in all_trades_flat]
    exit_fracs = exit_reason_fractions(all_trades)
    pnl_stats = trade_pnl_stats(all_trades)

    cum_net = (1 + f['net_ret'].fillna(0.0)).prod() - 1
    cum_gross = (1 + f['gross_ret'].fillna(0.0)).prod() - 1 if 'gross_ret' in f else np.nan
    dd = drawdown(f['net_ret'])
    ann_vol_daily_252 = daily_ret.std() * np.sqrt(252)

    time_in_market = f['active_flag'].mean() if 'active_flag' in f else np.nan
    turnover_sum = f['turnover'].sum() if 'turnover' in f else np.nan
    total_costs_pct = f['costs'].sum() if 'costs' in f else np.nan
    assert 0.0 <= time_in_market <= 1.0 + 1e-9, f'time_in_market_pct out of [0,1] range: {time_in_market}'

    distinct_pairs_traded = {pair for (rebal_date, pair), trades in all_trades.items() if trades}

    row = {
        'model': model_id,
        'daily_sharpe_252': daily_sharpe_252(daily_ret),
        'ann_volatility': ann_vol_daily_252,
        'cum_net_return': cum_net,
        'cum_gross_return': cum_gross,
        'max_drawdown': dd.min(),
        'n_trades': len(all_trades_flat),
        'n_pairs_traded': len(distinct_pairs_traded),
        'win_rate': pnl_stats['win_rate'],
        'avg_trade_return': pnl_stats['avg_trade_return'],
        'median_trade_return': pnl_stats['median_trade_return'],
        'avg_holding_bars': np.mean(durations) if durations else np.nan,
        'median_holding_bars': np.median(durations) if durations else np.nan,
        'time_in_market_pct': time_in_market,
        'turnover': turnover_sum,
        'pct_exit_mean_reversion': exit_fracs.get('exit', 0.0),
        'pct_exit_time_stop': exit_fracs.get('time_stop', 0.0),
        'pct_exit_hard_stop': exit_fracs.get('hard_stop', 0.0),
        'total_costs_pct': total_costs_pct,
        'btc_correlation': bench['correlation'],
        'btc_beta': bench['beta'],
        'ann_alpha': bench['ann_alpha'],
        'alpha_tstat_ols': bench['alpha_tstat_ols'],
        'alpha_tstat_hac': bench['alpha_tstat_hac'],
        'r_squared': bench['r_squared'],
    }
    return row


def active_day_benchmark_stats(daily_ret: pd.Series, active_mask: pd.Series, btc_daily: pd.Series) -> dict:
    """`benchmark_regression_stats` restricted to days with portfolio gross_exposure > 0 --
    reported alongside the unconditional benchmark stats for any model with low time-in-market
    (feedback #11): an unconditional beta near zero can look market-neutral purely because the
    strategy is flat most days, while its active-day beta tells a different story."""
    active_daily = daily_ret.where(active_mask)
    return benchmark_regression_stats(active_daily, btc_daily)


def active_mask_from_bar_frame(bar_active_flag: pd.Series) -> pd.Series:
    """Aggregate a 4h active-flag series to a daily 'any bar active' mask."""
    return bar_active_flag.groupby(bar_active_flag.index.floor('D')).any()


def pair_level_contribution(trades_dict: dict, pair_results: dict, cost_per_leg_bps: float) -> pd.DataFrame:
    """Per-pair-instance trade count, net P&L contribution (sum of trade-level net returns,
    each on that pair's own initial_equity=1.0 sleeve basis), and win rate. Handles both
    S0/S1's key shape (pair tuple) and WF1/WF2's key shape (rebalance-date instance key) --
    reports one row per instance rather than collapsing them, so a pair traded in two
    different rebalance blocks shows as two rows.

    Concentration is reported as each instance's share of total ABSOLUTE P&L magnitude (sum
    of |contribution|), not share of the net total -- the net total can be near zero when
    winners and losers roughly offset, which makes a share-of-net calculation blow up to
    meaningless three-digit percentages.
    """
    rows = []
    for key, trades in trades_dict.items():
        if not trades:
            continue
        # Normalize key to a (label, pair_tuple) pair regardless of which shape it is.
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], tuple):
            rebal_date, pair = key
            label = f'{pair[0]}/{pair[1]} @ {rebal_date.date()}'
        else:
            pair = key
            label = f'{pair[0]}/{pair[1]}'
        rets = [(t['exit_equity_after'] - t['entry_equity_before']) / t['entry_equity_before']
                for t in trades if t.get('entry_equity_before') and t.get('exit_equity_after') is not None]
        rows.append({
            'pair_instance': label, 'n_trades': len(trades),
            'sum_trade_return': sum(rets) if rets else np.nan,
            'win_rate': (np.array(rets) > 0).mean() if rets else np.nan,
            'avg_holding_bars': np.mean([t['duration_bars'] for t in trades]),
        })
    df = pd.DataFrame(rows).sort_values('sum_trade_return', ascending=False) if rows else pd.DataFrame(
        columns=['pair_instance', 'n_trades', 'sum_trade_return', 'win_rate', 'avg_holding_bars'])
    if len(df):
        total_abs = df['sum_trade_return'].abs().sum()
        df['pct_of_abs_pnl'] = df['sum_trade_return'].abs() / total_abs if total_abs > 0 else np.nan
    return df.reset_index(drop=True)
