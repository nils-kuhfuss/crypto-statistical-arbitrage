"""Walk-forward orchestration: rebalance schedule, per-block backtests, WF1/WF2 stitching.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 19-21, 25).

`run_wf1_block`/`run_wf2_block` and `run_wf1_full`/`run_wf2_full` were, in the original
notebook, four separate functions where the WF2 pair was a near-identical copy of the WF1
pair, differing only in using a rolling (vs. frozen) beta for entry sizing. Both pairs are now
one shared implementation each (`_run_block`, `_run_full`) parameterized by `use_rolling_beta`,
per REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md Section 8's explicit requirement not to maintain two
independent implementations of the same walk-forward mechanics. `run_wf1_block`/`run_wf1_full`
and `run_wf2_block`/`run_wf2_full` remain as thin, distinctly-named wrappers so existing call
sites and model-registry naming are unaffected.

Note on the sleeve-allocation assertion in `_run_full`: it is `n_pairs * (1/n_pairs) <= 1`,
which is tautologically true for any n_pairs > 0 and can never actually catch an allocation
violation. This is a pre-existing property of the original notebook (both `run_wf1_full` and
`run_wf2_full` had the identical tautology, just spelled slightly differently) -- it is ported
as-is per this task's "pure software refactor, no logic changes" scope, and flagged in
`results/refactor_validation.md` rather than silently corrected.
"""
import numpy as np
import pandas as pd

from .data import get_tradable_price
from .statistics import compute_spread, compute_rolling_beta
from .execution import backtest_single_pair
from .portfolio import evaluate_coin_netted_portfolio
from .pair_selection import select_pairs_at_rebalance
from .cache import load_or_compute
from config.research_config import WF_CONFIG


def generate_rebalance_dates(formation_start: str, oos_end: str, spacing_months: int = 6,
                              min_formation_months: int = 12) -> list:
    """Historical rebalance dates: `spacing_months`-spaced from 2021-01-01 through `oos_end`,
    starting only once `min_formation_months` of trailing data would be available."""
    dates = pd.date_range('2021-01-01', oos_end, freq=f'{spacing_months}MS', tz=None)
    return [d for d in dates if d >= pd.Timestamp(formation_start) + pd.DateOffset(months=min_formation_months)]


def _run_block(selections, block_start, block_end, price_data: pd.DataFrame, config: dict,
                use_rolling_beta: bool,
                rolling_window: int = WF_CONFIG['WF2_ROLLING_BETA_WINDOW_BARS']) -> dict:
    """Backtest one rebalance block: for each selected pair, run the same signal +
    fixed-quantity execution path as S0/S1 (`backtest_single_pair`, reused verbatim) using the
    pair's FROZEN formation alpha/spread_mean/spread_std/time_stop. Beta is either the frozen
    formation beta (WF1, `use_rolling_beta=False`) or a lagged rolling beta re-estimated at
    every bar (WF2, `use_rolling_beta=True`) -- the Z-score signal itself always uses the
    frozen formation alpha/beta in both cases; only entry-time position sizing differs.

    Price series are clipped to price_data.loc[:block_end] (not full history) before being
    passed to backtest_single_pair, so the state machine's 'no new entry in the last two
    bars' guard applies at each block boundary, guaranteeing every trade liquidates before
    the next block starts.

    Returns {(rebalance_date, pair): (result_df, trades, pair)} for pairs active in this
    block, or {} if the block is empty.
    """
    if not selections:
        return {}
    block_results = {}
    price_data_clipped = price_data.loc[:block_end]
    for sel in selections:
        x_name, y_name = sel['x'], sel['y']
        x_price = get_tradable_price(price_data_clipped[x_name])
        y_price = get_tradable_price(price_data_clipped[y_name])
        ret_x = x_price.pct_change(fill_method=None)
        ret_y = y_price.pct_change(fill_method=None)

        log_x, log_y = np.log(x_price), np.log(y_price)
        spread = compute_spread(log_y, log_x, sel['alpha'], sel['beta'])
        z_score = (spread - sel['spread_mean']) / sel['spread_std']
        untradable = x_price.isna() | y_price.isna()
        z_score = z_score.where(~untradable, np.nan)

        if use_rolling_beta:
            beta_series = compute_rolling_beta(x_price, y_price, rolling_window)
        else:
            beta_series = pd.Series(sel['beta'], index=x_price.index)

        result, trades = backtest_single_pair(
            x_price, y_price, sel['alpha'], sel['beta'], ret_x, ret_y,
            entry_z=config['ENTRY_Z'], exit_z=config['EXIT_Z'], hard_stop_z=config['HARD_STOP_Z'],
            time_stop_bars=sel['time_stop_bars'], cost_per_leg_bps=config['COST_PER_LEG_BPS'],
            z_score=z_score, beta_series=beta_series, state_start=block_start)
        # Unique key per pair-instance -- a later rebalance's re-selection of the same (x, y)
        # pair must not silently overwrite an earlier instance's result.
        instance_key = (sel['rebalance_date'], sel['pair'])
        block_results[instance_key] = (result.loc[block_start:block_end], trades, sel['pair'])
    return block_results


def _run_full(selections_by_date: dict, rebalance_dates, oos_end, price_data: pd.DataFrame,
              config: dict, use_rolling_beta: bool,
              rolling_window: int = WF_CONFIG['WF2_ROLLING_BETA_WINDOW_BARS']):
    """Run every rebalance block back-to-back and build ONE portfolio-level source-of-truth
    frame spanning all blocks: net return, gross return, turnover, costs, gross exposure, net
    exposure, number of active pairs, and an active/not-active flag -- all derived directly
    from `build_coin_netted_portfolio`'s own per-block frame, never reconstructed by summing
    pair-level outputs after the fact (which would double-count when multiple pairs are
    active simultaneously).

    all_trades/all_results are keyed by (rebalance_date, pair) -- a unique pair-INSTANCE key,
    so the same (x, y) pair selected at two different rebalances produces two separate
    entries rather than one overwriting the other.
    """
    block_bounds = list(zip(rebalance_dates, rebalance_dates[1:] + [pd.Timestamp(oos_end)]))
    all_block_frames = []
    all_trades, all_results = {}, {}

    for block_start, block_end in block_bounds:
        block_end_adj = block_end - pd.Timedelta(hours=4) if block_end != pd.Timestamp(oos_end) else block_end
        selections = selections_by_date.get(block_start, [])
        block_results_raw = _run_block(selections, block_start, block_end_adj, price_data, config,
                                        use_rolling_beta, rolling_window)

        if not block_results_raw:
            idx = price_data.loc[block_start:block_end_adj].index
            block_frame = pd.DataFrame({
                'net_ret': 0.0, 'gross_ret': 0.0, 'turnover': 0.0, 'costs': 0.0,
                'gross_exposure': 0.0, 'net_exposure': 0.0, 'n_active_pairs': 0,
            }, index=idx)
            all_block_frames.append(block_frame)
            continue

        pair_results = {}
        n_active_by_bar = pd.Series(0, index=price_data.loc[block_start:block_end_adj].index)
        for instance_key, (res, trades, pair) in block_results_raw.items():
            all_trades[instance_key] = trades
            all_results[instance_key] = res
            pair_results[pair] = res  # one instance per pair per block, so this key is safe
            n_active_by_bar = n_active_by_bar.add((res['gross_exposure'] > 0).astype(int).reindex(n_active_by_bar.index, fill_value=0), fill_value=0)
        block_pairs = list(pair_results.keys())
        n_pairs_this_block = len(block_pairs)
        sleeve_weight_this_block = 1.0 / n_pairs_this_block
        assert n_pairs_this_block * sleeve_weight_this_block <= 1.0 + 1e-9, (
            f'Sleeve allocation exceeds 100% of portfolio at block starting {block_start}.')
        portfolio, _ = evaluate_coin_netted_portfolio(
            pair_results, block_pairs, price_data.loc[:block_end_adj], config['COST_PER_LEG_BPS'],
            block_start, block_end_adj, config['BARS_PER_YEAR'])
        f = portfolio['frame']

        # Net exposure: sum of signed coin notional / equity (long-short net, not gross
        # absolute) -- distinguishes true market-neutrality from simply being in cash.
        net_notional = portfolio['coin_notional'].sum(axis=1)
        net_exposure = (net_notional / f['equity'].replace(0, np.nan)).fillna(0.0)

        block_frame = pd.DataFrame({
            'net_ret': f['net_ret'], 'gross_ret': f['gross_ret'],
            'turnover': f['turnover'], 'costs': f['costs'],
            'gross_exposure': f['gross_exposure'], 'net_exposure': net_exposure,
            'n_active_pairs': n_active_by_bar.reindex(f.index, fill_value=0).astype(int),
        }, index=f.index)
        all_block_frames.append(block_frame)

    full_frame = pd.concat(all_block_frames).sort_index()
    full_frame = full_frame[~full_frame.index.duplicated(keep='first')]
    full_frame['active_flag'] = full_frame['gross_exposure'] > 0
    # Realized gross_exposure can legitimately drift above 1.0 intra-trade: fixed quantities
    # are held between executions while the pair's own equity mark-to-markets, so a losing
    # trade shrinks the denominator (equity) while notional stays roughly fixed. This is a
    # known property of fixed-quantity sizing (same in S0), not an allocation violation --
    # reported as a diagnostic, not asserted on. The actual sleeve-allocation invariant is
    # checked above, per block, at selection time.
    _max_exposure_bar = full_frame['gross_exposure'].idxmax() if len(full_frame) else None
    if _max_exposure_bar is not None and full_frame['gross_exposure'].max() > 1.0 + 1e-6:
        print(f"[NOTE] Peak realized gross_exposure = {full_frame['gross_exposure'].max():.3f} "
              f'at {_max_exposure_bar} (intra-trade mark-to-market drift on fixed quantities, not a '
              f'sleeve-allocation violation -- sleeve weights themselves sum to <=1.0 by construction).')
    return full_frame, all_results, all_trades


def run_wf1_block(selections, block_start, block_end, price_data: pd.DataFrame, config: dict) -> dict:
    """WF1: frozen formation beta for entry sizing. See `_run_block`."""
    return _run_block(selections, block_start, block_end, price_data, config, use_rolling_beta=False)


def run_wf1_full(selections_by_date: dict, rebalance_dates, oos_end, price_data: pd.DataFrame, config: dict):
    """WF1: frozen formation beta for entry sizing. See `_run_full`."""
    return _run_full(selections_by_date, rebalance_dates, oos_end, price_data, config, use_rolling_beta=False)


def run_wf2_block(selections, block_start, block_end, price_data: pd.DataFrame, config: dict,
                   rolling_window: int = WF_CONFIG['WF2_ROLLING_BETA_WINDOW_BARS']) -> dict:
    """WF2: lagged rolling beta for entry sizing (Z-score signal still uses the frozen
    formation alpha/beta). See `_run_block`. Do not describe this as a rolling-spread model --
    only entry-time position sizing is rolling."""
    return _run_block(selections, block_start, block_end, price_data, config, use_rolling_beta=True,
                       rolling_window=rolling_window)


def run_wf2_full(selections_by_date: dict, rebalance_dates, oos_end, price_data: pd.DataFrame, config: dict,
                  rolling_window: int = WF_CONFIG['WF2_ROLLING_BETA_WINDOW_BARS']):
    """WF2: lagged rolling beta for entry sizing. See `_run_full`."""
    return _run_full(selections_by_date, rebalance_dates, oos_end, price_data, config, use_rolling_beta=True,
                      rolling_window=rolling_window)


def run_wf1_selection_pipeline(rebalance_dates, price_data: pd.DataFrame, coins, wf_config: dict,
                                cache_path: str,
                                max_pairs: int = WF_CONFIG['MAX_PAIRS_PER_REBALANCE']) -> dict:
    """Load WF1 pair selections from cache if present, otherwise compute and cache them.
    Mirrors the notebook's compute-and-write-on-miss cache convention (`cache.load_or_compute`).
    """
    def _compute():
        selections = {}
        for rebal_date in rebalance_dates:
            selections[rebal_date] = select_pairs_at_rebalance(rebal_date, price_data, coins, wf_config, max_pairs)
        return selections
    return load_or_compute(cache_path, _compute)
