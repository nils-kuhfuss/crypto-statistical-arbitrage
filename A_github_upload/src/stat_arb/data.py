"""Loading prices/returns, universe construction, sample splits, point-in-time availability.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Sections 3, 4, 18.5, 19).

`get_tradable_price`/`tradable_mask` originally defaulted `ffill_limit` from the notebook's
global `CONFIG['FFILL_LIMIT_BARS']`. Every call site already passed nothing but relied on that
default with the same value everywhere, so it is now `config.CONFIG['FFILL_LIMIT_BARS']` passed
explicitly by the one caller convention established in the notebooks -- see
`config/research_config.py`. The default is kept here (reading the same config value) so
existing call patterns port unchanged, but is now sourced from the central config module
instead of a bare notebook global.
"""
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

from config.research_config import CONFIG, WF_CONFIG, PX_PATH, RET_PATH


def load_raw_prices() -> pd.DataFrame:
    """Load the full, unfiltered 4h price panel from px.pkl."""
    with open(PX_PATH, 'rb') as f:
        return pickle.load(f)


def load_raw_returns() -> pd.DataFrame:
    """Load the full, unfiltered 4h return panel from ret.pkl."""
    with open(RET_PATH, 'rb') as f:
        return pickle.load(f)


def _coverage(col: pd.Series) -> pd.Series:
    """Per-coin coverage summary: first/last valid date, observation count, missing %."""
    first = col.first_valid_index()
    last = col.last_valid_index()
    active = col.loc[first:last]
    td = len(active)
    valid = active.count()
    missing_pct = ((td - valid) / td) * 100 if td > 0 else np.nan
    return pd.Series({'First Date': first, 'Last Date': last,
                       'Observations': valid, 'Missing (%)': round(missing_pct, 2)})


def filter_universe(px: pd.DataFrame, ret: pd.DataFrame, config: dict = CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """S0's one-time static universe filter (Section 3): drop `config['MANUAL_EXCLUDE']`
    (stablecoins etc.) and any coin whose first observation falls after
    `config['MIN_HISTORY_CUTOFF']` -- too little formation-period history to support a
    reliable regression fit. Point-in-time note: this is a ONE-TIME filter applied before any
    sample split, used only by the static S0/S1 models -- the walk-forward universe
    (`WF_PX`/`WF_RET`, `build_walkforward_universe` below) does NOT use this filter, since its
    point-in-time coverage/active-at-rebalance rules are designed to handle late listings and
    delistings correctly without a manual cutoff (see CLAUDE.md "Walk-forward universe").

    Returns (px_filtered, ret_filtered, dropped_coins).
    """
    overview = ret.apply(_coverage).T
    cut = pd.to_datetime(config['MIN_HISTORY_CUTOFF'])
    drop_late = overview[overview['First Date'] > cut].index.tolist()
    drop = sorted(set(config['MANUAL_EXCLUDE'] + drop_late))
    return px.drop(columns=drop), ret.drop(columns=drop), drop


def get_tradable_price(px_series: pd.Series, ffill_limit: int = CONFIG['FFILL_LIMIT_BARS']) -> pd.Series:
    """Return a price series that is NaN outside the coin's active trading window,
    with only small gaps (<= ffill_limit bars) bridged. No unbounded forward-fill.

    Forward-fill is applied before reindexing onto the full sample index, strictly within
    [first_valid_index, last_valid_index]. Reindexing first and forward-filling afterward
    would let the fill bridge past last_valid_index into the newly introduced trailing
    NaNs -- i.e. treat a coin as tradable for a few bars after its last real print. Filling
    only within the slice ending at the coin's actual last observation avoids this.

    Point-in-time: at any bar t, the returned value depends only on px_series up to and
    including t (a forward-fill only carries a past value forward).
    """
    s = px_series.replace(0, np.nan)
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return s
    s = s.loc[first:last].ffill(limit=ffill_limit)
    s = s.reindex(px_series.index)
    return s


def tradable_mask(px_series: pd.Series, ffill_limit: int = CONFIG['FFILL_LIMIT_BARS']) -> pd.Series:
    """True where the coin has a usable (real or small-gap-filled) price."""
    return get_tradable_price(px_series, ffill_limit).notna()


def split_samples(px: pd.DataFrame, config: dict = CONFIG):
    """Slice the static (S0/S1) universe into formation/validation/OOS windows and assert
    the splits do not overlap. Point-in-time: formation strictly precedes validation, which
    strictly precedes OOS -- enforced by assertion, not just by construction.

    Returns (formation_px, validation_px, oos_px).
    """
    formation_px = px.loc[config['FORMATION_START']:config['FORMATION_END']]
    validation_px = px.loc[config['VALIDATION_START']:config['VALIDATION_END']]
    oos_px = px.loc[config['OOS_START']:config['OOS_END']]
    assert formation_px.index.max() < validation_px.index.min(), 'Formation/Validation overlap!'
    assert validation_px.index.max() < oos_px.index.min(), 'Validation/OOS overlap!'
    return formation_px, validation_px, oos_px


def build_walkforward_universe(wf_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the walk-forward universe fresh from the raw pickles, dropping only
    `wf_config['MANUAL_EXCLUDE']` (true stablecoins) -- unlike `filter_universe` above, no
    `MIN_HISTORY_CUTOFF` is applied, since late listings and delistings are handled
    point-in-time by `check_active_at_rebalance`/formation-coverage rules instead. See
    CLAUDE.md "Walk-forward universe: WF_PX / WF_RET (not S0's px/ret)".

    Returns (WF_PX, WF_RET).
    """
    px_raw = load_raw_prices()
    ret_raw = load_raw_returns()
    wf_px = px_raw.drop(columns=wf_config['MANUAL_EXCLUDE'])
    wf_ret = ret_raw.drop(columns=wf_config['MANUAL_EXCLUDE'])
    return wf_px, wf_ret


def check_active_at_rebalance(price_data: pd.DataFrame, coins, rebalance_date: pd.Timestamp,
                               lookback_bars: int = WF_CONFIG['ACTIVE_AT_REBALANCE_LOOKBACK_BARS']) -> pd.Series:
    """An asset is 'active at rebalance' if it has a valid (non-NaN) price within the last
    `lookback_bars` bars strictly before rebalance_date -- a small tolerance for isolated data
    gaps, not a loophole for genuinely delisted assets (Binance delistings leave a long
    trailing all-NaN tail, e.g. XMRUSDT after 2024-02-20, which this correctly fails).

    Point-in-time: only bars strictly before `rebalance_date` are inspected.
    """
    check_window = price_data[coins].loc[:rebalance_date - pd.Timedelta(hours=4)].tail(lookback_bars)
    return check_window.notna().any(axis=0)
