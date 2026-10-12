"""Multi-pair portfolio aggregation: equal-sleeve coin netting, cash ledger, exposure/turnover.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Section 13). Used identically by
S0, S1, and (via walkforward.py's per-block calls) WF1/WF2 -- one shared aggregation engine
across every model version.
"""
import numpy as np
import pandas as pd

from .data import get_tradable_price
from .metrics import drawdown, summarize_model_performance


def build_coin_netted_portfolio(pair_results: dict, pairs, price_data: pd.DataFrame,
                                 cost_per_leg_bps: float, start, end, initial_equity: float = 1.0) -> dict:
    """Aggregate equal-initial-capital pair sleeves into coin quantities and a cash ledger."""
    pairs = list(pairs)
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError('Cannot build a portfolio without pairs.')
    idx = pair_results[pairs[0]].loc[start:end].index
    sleeve_weight = 1.0 / n_pairs

    coin_qty = {}
    pair_level_turnover_notional = pd.Series(0.0, index=idx)
    for x_name, y_name in pairs:
        r = pair_results[(x_name, y_name)].reindex(idx)
        coin_qty.setdefault(x_name, pd.Series(0.0, index=idx))
        coin_qty.setdefault(y_name, pd.Series(0.0, index=idx))
        coin_qty[x_name] = coin_qty[x_name].add(r['qty_x'].fillna(0.0) * sleeve_weight)
        coin_qty[y_name] = coin_qty[y_name].add(r['qty_y'].fillna(0.0) * sleeve_weight)
        pair_level_turnover_notional = pair_level_turnover_notional.add(
            r['turnover_cash'].fillna(0.0) * sleeve_weight)
    coin_qty_df = pd.DataFrame(coin_qty, index=idx)

    coin_prices = pd.DataFrame({
        coin: get_tradable_price(price_data[coin]).reindex(idx) for coin in coin_qty_df.columns
    }, index=idx)
    coin_marks = coin_prices.ffill()
    qty_prev = coin_qty_df.shift().fillna(0.0)
    gross_pnl_cash = (qty_prev * coin_marks.diff()).sum(axis=1).fillna(0.0)
    order_qty = (coin_qty_df - qty_prev).fillna(0.0)
    turnover_notional = (order_qty.abs() * coin_marks).sum(axis=1).fillna(0.0)
    cost_cash = turnover_notional * cost_per_leg_bps * 1e-4

    equity_before = pd.Series(index=idx, dtype=float)
    equity = pd.Series(index=idx, dtype=float)
    gross_equity = pd.Series(index=idx, dtype=float)
    net_ret = pd.Series(index=idx, dtype=float)
    gross_ret = pd.Series(index=idx, dtype=float)
    current_equity = current_gross_equity = float(initial_equity)
    for i, ts in enumerate(idx):
        equity_before.iloc[i] = current_equity
        gross_ret.iloc[i] = gross_pnl_cash.iloc[i] / current_equity
        net_pnl = gross_pnl_cash.iloc[i] - cost_cash.iloc[i]
        net_ret.iloc[i] = net_pnl / current_equity
        current_equity += net_pnl
        current_gross_equity += gross_pnl_cash.iloc[i]
        equity.iloc[i] = current_equity
        gross_equity.iloc[i] = current_gross_equity

    coin_notional = coin_qty_df * coin_marks
    coin_weights = coin_notional.div(equity.replace(0, np.nan), axis=0)
    gross_notional = coin_notional.abs().sum(axis=1)
    gross_exposure = gross_notional / equity.replace(0, np.nan)
    coin_concentration = coin_notional.abs().max(axis=1).div(gross_notional.replace(0, np.nan))

    frame = pd.DataFrame({
        'gross_pnl_cash': gross_pnl_cash,
        'cost_cash': cost_cash,
        'net_pnl_cash': gross_pnl_cash - cost_cash,
        'gross_ret': gross_ret,
        'net_ret': net_ret,
        'equity_before': equity_before,
        'equity': equity,
        'gross_equity': gross_equity,
        'turnover_notional': turnover_notional,
        'turnover': turnover_notional / equity_before,
        'costs': cost_cash / equity_before,
        'gross_exposure': gross_exposure.fillna(0.0),
        'coin_concentration': coin_concentration,
        'pair_level_turnover_notional': pair_level_turnover_notional,
    }, index=idx)

    assert coin_qty_df.iloc[-1].abs().max() < 1e-12, (
        'Portfolio is not flat after final next-bar liquidation.')
    return {'frame': frame, 'coin_qty': coin_qty_df, 'coin_notional': coin_notional,
            'coin_weights': coin_weights, 'order_qty': order_qty,
            'coin_prices': coin_prices, 'coin_marks': coin_marks}


def evaluate_coin_netted_portfolio(pair_results: dict, pairs, price_data: pd.DataFrame,
                                    cost_per_leg_bps: float, start, end, bars_per_year: int):
    """Build the coin-netted portfolio and compute its standard performance metrics."""
    portfolio = build_coin_netted_portfolio(
        pair_results, pairs, price_data, cost_per_leg_bps, start, end)
    f = portfolio['frame']
    net_stats = summarize_model_performance(pd.DataFrame({'net_ret': f['net_ret']}), bars_per_year)
    gross_stats = summarize_model_performance(pd.DataFrame({'net_ret': f['gross_ret']}), bars_per_year)
    dd = drawdown(f['net_ret'])
    active_conc = f.loc[f['gross_exposure'] > 0, 'coin_concentration']
    pair_turnover = f['pair_level_turnover_notional'].sum()
    net_turnover = f['turnover_notional'].sum()
    metrics = {
        'sharpe': net_stats['sharpe'], 'gross_sharpe': gross_stats['sharpe'],
        'ann_return': net_stats['ann_return'], 'gross_ann_return': gross_stats['ann_return'],
        'ann_vol': net_stats['ann_vol'],
        'cum_return': f['equity'].iloc[-1] - 1.0,
        'gross_cum_return': f['gross_equity'].iloc[-1] - 1.0,
        'max_dd': dd.min(), 'total_turnover': f['turnover'].sum(),
        'total_costs': f['cost_cash'].sum(),
        'pair_level_turnover_notional': pair_turnover,
        'netted_turnover_notional': net_turnover,
        'netting_savings_turnover': pair_turnover - net_turnover,
        'avg_gross_exposure': f['gross_exposure'].mean(),
        'max_gross_exposure': f['gross_exposure'].max(),
        'avg_coin_concentration': active_conc.mean() if len(active_conc) else np.nan,
        'max_coin_concentration': active_conc.max() if len(active_conc) else np.nan,
        'max_single_coin_weight': portfolio['coin_weights'].abs().max(axis=1).max(),
    }
    portfolio['drawdown'] = dd
    portfolio['metrics'] = metrics
    return portfolio, metrics


def coin_concentration_diagnostics(portfolio: dict) -> dict:
    """Per-coin exposure concentration diagnostics, derived read-only from an already-built
    `build_coin_netted_portfolio` result -- never a second coin-netting implementation.

    Introduced for the Phase B2.5 no-overlap ablation (`stat_arb.ablation`), where multiple
    selected pairs may legitimately share the same coin (the primary WF1 selection never
    exercises this case, since its one-pair-per-coin constraint makes it structurally
    impossible), but generic to any coin-netted portfolio -- concentration is a property of
    `coin_notional`/`equity`, not of how the pairs were selected.

    Returns bar-level frames (`gross_exposure_pct_by_coin`, `net_exposure_pct_by_coin`,
    `share_of_total_gross_by_coin`, `largest_gross_coin_by_bar` -- the latter `NaN` on bars
    with zero gross exposure) plus scalar/per-coin summary Series (`max_gross_exposure_by_coin`,
    `max_abs_net_exposure_by_coin`, `avg_gross_exposure_by_coin_active`,
    `avg_abs_net_exposure_by_coin_active`, `max_share_of_total_gross_single_coin`,
    `top5_coins_by_max_gross`, `top5_coins_by_max_abs_net`). "Active" bars are those with
    portfolio `gross_exposure > 0` (same convention as `time_in_market_pct` elsewhere).
    """
    frame = portfolio['frame']
    equity = frame['equity']
    active = frame['gross_exposure'] > 0
    coin_notional = portfolio['coin_notional']

    gross_pct = coin_notional.abs().div(equity.replace(0, np.nan), axis=0)
    net_pct = coin_notional.div(equity.replace(0, np.nan), axis=0)
    total_gross_pct = gross_pct.sum(axis=1)
    share_of_gross = gross_pct.div(total_gross_pct.replace(0, np.nan), axis=0)
    largest_gross_coin = gross_pct.idxmax(axis=1).where(total_gross_pct > 0)

    return {
        'gross_exposure_pct_by_coin': gross_pct,
        'net_exposure_pct_by_coin': net_pct,
        'share_of_total_gross_by_coin': share_of_gross,
        'largest_gross_coin_by_bar': largest_gross_coin,
        'max_gross_exposure_by_coin': gross_pct.max(),
        'max_abs_net_exposure_by_coin': net_pct.abs().max(),
        'avg_gross_exposure_by_coin_active': gross_pct[active].mean(),
        'avg_abs_net_exposure_by_coin_active': net_pct[active].abs().mean(),
        'max_share_of_total_gross_single_coin': share_of_gross.max(axis=1).max() if len(share_of_gross) else np.nan,
        'top5_coins_by_max_gross': gross_pct.max().sort_values(ascending=False).head(5),
        'top5_coins_by_max_abs_net': net_pct.abs().max().sort_values(ascending=False).head(5),
    }


def build_validation_portfolio(model: str, pairs, model_results: dict, price_data: pd.DataFrame,
                                config: dict) -> dict:
    """2023-validation portfolio for one of S0's candidate models ('S', 'RZ', 'RB')."""
    pair_results = {(x, y): model_results[(x, y, model)] for x, y in pairs}
    portfolio, metrics = evaluate_coin_netted_portfolio(
        pair_results, pairs, price_data, config['COST_PER_LEG_BPS'],
        config['VALIDATION_START'], config['VALIDATION_END'], config['BARS_PER_YEAR'])
    return portfolio, {'model': model, **metrics}
