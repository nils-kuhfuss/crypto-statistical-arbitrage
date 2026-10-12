"""Pair generation, cointegration screening, eligibility, ranking, one-pair-per-coin selection.

Ported verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` (Section 6 -- S0's one-time static
screen; Sections 19-20 -- the walk-forward point-in-time screen `screen_pairs_at_date` and
`select_pairs_at_rebalance`).

`run_static_pair_screen` and `screen_pairs_at_date` both now call the shared
`statistics.run_engle_granger_tests`/`statistics.apply_bh_fdr` and `dedup_pair_directions`
below, instead of each containing its own independent copy of the EG-test + BH-FDR + dedup
loop (see `statistics.py`'s module docstring for why this consolidation is safe).
"""
import itertools as it
from typing import List, Tuple

import numpy as np
import pandas as pd

from .statistics import (apply_bh_fdr, compute_half_life, compute_spread, count_crossings,
                          fit_static_ols, run_engle_granger_tests)
from .data import check_active_at_rebalance
from config.research_config import WF_CONFIG


def generate_candidate_pairs(coins) -> List[Tuple[str, str]]:
    """All unordered pairs from a coin universe."""
    return list(it.combinations(coins, 2))


def dedup_pair_directions(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Because both regression directions were tested for every unordered pair, a pair can
    appear twice in the FDR-significant set. These are the same economic relationship traded
    in two mirrored parameterizations; retaining both would double-count the pair in later
    selection. For each unordered pair, only the direction with the lower adjusted p-value is
    kept. `sig_df` must already be sorted ascending by 'pvalue_adj'."""
    deduped = sig_df.sort_values('pvalue_adj').drop_duplicates(subset='pair', keep='first').reset_index(drop=True)
    assert deduped['pair'].is_unique, 'Deduplication failed: an unordered pair still appears twice.'
    return deduped


def run_static_pair_screen(log_formation_px: pd.DataFrame, coins, config: dict,
                            max_half_life_bars: int, min_crossings: int) -> pd.DataFrame:
    """S0's one-time static cointegration + half-life + crossing screen (Section 6).

    Point-in-time: `log_formation_px` must already be restricted to the formation window --
    this function does not slice dates itself.

    `max_half_life_bars`/`min_crossings` are passed explicitly (matching
    `config/research_config.py`'s `MAX_HALF_LIFE_BARS`/`MIN_CROSSINGS`, which are bare
    module-level constants in the original notebook, not part of `CONFIG`) rather than nested
    inside `config`, consistent with `screen_pairs_at_date`'s equivalent parameters.

    Steps: generate all unordered pairs -> Engle-Granger both directions -> Benjamini-Hochberg
    FDR -> keep the lower-pvalue_adj direction per pair -> per surviving pair, fit OLS on the
    full formation window, compute half-life and mean-crossings -> filter to
    `max_half_life_bars`/`min_crossings`. No positive-Beta filter here -- that requirement is
    specific to the walk-forward screen (`screen_pairs_at_date`), added later as a correction;
    S0's own screen never had it.

    Returns `candidate_pairs`: one row per pair surviving every filter, columns
    ['pair', 'y', 'x', 'pvalue_adj', 'alpha', 'beta', 'half_life_bars', 'crossings',
    'crossings_per_year'].
    """
    pairs = generate_candidate_pairs(coins)
    eg_df = run_engle_granger_tests(log_formation_px, pairs, config['MIN_OBS_FOR_TEST'])
    assert len(eg_df) > 0, 'No directional tests could be run -- check universe/formation window.'
    eg_df = apply_bh_fdr(eg_df, config['EG_ALPHA'], config['FDR_METHOD'])
    sig_df = eg_df[eg_df['significant_fdr']].sort_values('pvalue_adj').reset_index(drop=True)
    assert len(sig_df) > 0, (
        'No pairs survived the Benjamini-Hochberg FDR correction. A negative/empty result is '
        'a legitimate outcome, but there is nothing to half-life-screen. Do not loosen EG_ALPHA '
        'or the FDR method to force a non-empty list.'
    )
    sig_df = dedup_pair_directions(sig_df)

    screen_rows = []
    for _, row in sig_df.iterrows():
        y_name, x_name = row['y'], row['x']
        log_y = log_formation_px[y_name]
        log_x = log_formation_px[x_name]
        a, b, _ = fit_static_ols(log_y, log_x)
        spread = compute_spread(log_y, log_x, a, b)
        hl = compute_half_life(spread)
        crossings = count_crossings(spread)
        years = (spread.dropna().index[-1] - spread.dropna().index[0]).days / 365.25
        screen_rows.append({
            'pair': row['pair'], 'y': y_name, 'x': x_name,
            'pvalue_adj': row['pvalue_adj'], 'alpha': a, 'beta': b,
            'half_life_bars': hl, 'crossings': crossings,
            'crossings_per_year': crossings / years if years > 0 else np.nan,
        })
    screen_df = pd.DataFrame(screen_rows).sort_values('half_life_bars')

    candidate_pairs = screen_df[
        (screen_df['half_life_bars'] < max_half_life_bars) &
        (screen_df['half_life_bars'] > 0) &
        (screen_df['crossings'] >= min_crossings)
    ].copy()
    return candidate_pairs


def screen_pairs_at_date(rebalance_date: pd.Timestamp, price_data: pd.DataFrame, coins,
                          wf_config: dict, collect_diagnostics: bool = False):
    """Point-in-time pair screen, corrected per expert feedback to match S0's own screen
    exactly (Section 6) plus true 12-month coverage and active-at-rebalance eligibility:

    1. Eligibility: >= min_obs valid observations in the trailing formation window (true
       ~90%-coverage 12-month requirement) AND the asset must still be actively trading at
       rebalance_date (`check_active_at_rebalance`).
    2. Engle-Granger cointegration test, both directions, run on EVERY eligible pair -- return
       correlation is NOT used as a pre-filter (feedback #5); it is computed and reported
       purely as a descriptive relationship characteristic.
    3. Benjamini-Hochberg FDR correction over the full directional-test family at this date.
    4/5. Valid positive half-life below `max_half_life_bars` (same rule as S0's Section 6).
    6. Minimum mean-crossing count (same rule as S0's Section 6).
    7. Positive Hedge Ratio (Beta > 0) if `require_positive_beta` -- feedback #6.

    Point-in-time: all filters use ONLY the trailing `formation_months` of data strictly
    before `rebalance_date`. Returns a dataframe with one row per unordered pair that
    survives every filter at this date (empty if none survive).

    `collect_diagnostics=True` (Phase B2 selection-funnel diagnostic) additionally returns a
    `funnel` dict (stage counts) and a `pair_trail` DataFrame (one row per TESTABLE pair,
    tracking pass/fail through Beta -> Half-Life -> Crossings in the exact order evaluated
    below, and the first stage at which it was eliminated). This is purely additive
    instrumentation of the same loop that already produces the returned DataFrame -- it does
    not change what that DataFrame contains, and `collect_diagnostics=False` (the default)
    is byte-for-byte identical to this function's behavior before this parameter existed
    (every existing call site uses the default and is unaffected).
    """
    eg_alpha = wf_config['EG_ALPHA']
    min_obs = wf_config['MIN_OBS_FOR_TEST']
    formation_months = wf_config['FORMATION_MONTHS']
    max_half_life_bars = wf_config['MAX_HALF_LIFE_BARS']
    min_crossings = wf_config['MIN_CROSSINGS']
    require_positive_beta = wf_config['REQUIRE_POSITIVE_BETA']

    window_start = rebalance_date - pd.DateOffset(months=formation_months)
    window_px = price_data.loc[window_start:rebalance_date - pd.Timedelta(hours=4)]
    assert window_px.index.max() < rebalance_date, 'Leakage: formation window reaches rebalance date.'

    coverage_ok = window_px[coins].notna().sum() >= min_obs
    active_ok = check_active_at_rebalance(price_data, coins, rebalance_date,
                                           wf_config['ACTIVE_AT_REBALANCE_LOOKBACK_BARS'])
    valid_coins = [c for c in coins if coverage_ok.get(c, False) and active_ok.get(c, False)]

    def _funnel(n_candidate_pairs=0, n_testable_pairs=0, n_directed_eg_tests=0,
                n_fdr_significant_pairs=0, n_pass_beta=0, n_pass_half_life=0,
                n_pass_crossings=0, n_fully_qualifying=0):
        return {
            'rebalance_date': rebalance_date, 'n_eligible_assets': len(valid_coins),
            'n_candidate_pairs': n_candidate_pairs, 'n_testable_pairs': n_testable_pairs,
            'n_directed_eg_tests': n_directed_eg_tests,
            'n_fdr_significant_pairs': n_fdr_significant_pairs, 'n_pass_beta': n_pass_beta,
            'n_pass_half_life': n_pass_half_life, 'n_pass_crossings': n_pass_crossings,
            'n_fully_qualifying': n_fully_qualifying,
        }

    if len(valid_coins) < 2:
        if collect_diagnostics:
            return pd.DataFrame(), _funnel(), pd.DataFrame()
        return pd.DataFrame()
    log_px = np.log(window_px[valid_coins])
    pairs = generate_candidate_pairs(valid_coins)

    eg_df_t = run_engle_granger_tests(log_px, pairs, min_obs)
    if len(eg_df_t) == 0:
        if collect_diagnostics:
            return pd.DataFrame(), _funnel(n_candidate_pairs=len(pairs)), pd.DataFrame()
        return pd.DataFrame()
    eg_df_t = apply_bh_fdr(eg_df_t, eg_alpha, wf_config['FDR_METHOD'])
    sig_t = eg_df_t[eg_df_t['significant_fdr']].sort_values('pvalue_adj')

    # Diagnostics-only: one row per TESTABLE pair (not just FDR-significant ones), so the
    # pair-level table can show pairs that failed FDR too. Computed via the same shared
    # dedup_pair_directions primitive used for the real sig_t below -- not a second
    # implementation of the dedup rule -- and never feeds back into sig_t/out_rows.
    pair_trail_index = None
    if collect_diagnostics:
        full_dedup = dedup_pair_directions(eg_df_t.sort_values('pvalue_adj'))
        pair_trail_index = {
            row['pair']: {
                'pair': row['pair'], 'y': row['y'], 'x': row['x'],
                'eg_pvalue': row['eg_pvalue'], 'pvalue_adj': row['pvalue_adj'],
                'passed_fdr': bool(row['significant_fdr']),
                'passed_beta': np.nan, 'passed_half_life': np.nan, 'passed_crossings': np.nan,
                'fully_qualifying': False,
                'elimination_stage': (None if row['significant_fdr'] else 'fdr'),
            }
            for _, row in full_dedup.iterrows()
        }

    if len(sig_t) == 0:
        if collect_diagnostics:
            funnel = _funnel(n_candidate_pairs=len(pairs), n_testable_pairs=int(eg_df_t['pair'].nunique()),
                              n_directed_eg_tests=int(len(eg_df_t)))
            return pd.DataFrame(), funnel, pd.DataFrame(list(pair_trail_index.values()))
        return pd.DataFrame()
    sig_t = dedup_pair_directions(sig_t)

    out_rows = []
    n_pass_beta = n_pass_hl = n_pass_crossings = 0
    for _, row in sig_t.iterrows():
        y_name, x_name = row['y'], row['x']
        pair_key = row['pair']
        a, b, _ = fit_static_ols(log_px[y_name], log_px[x_name])
        if require_positive_beta and b <= 0:
            if collect_diagnostics:
                pair_trail_index[pair_key]['elimination_stage'] = 'positive_beta'
            continue  # feedback #6: negative-Beta pairs are not conventional market-neutral relative value
        n_pass_beta += 1
        if collect_diagnostics:
            pair_trail_index[pair_key]['passed_beta'] = True
        spread = compute_spread(log_px[y_name], log_px[x_name], a, b)
        hl = compute_half_life(spread)
        if not np.isfinite(hl) or hl <= 0 or hl >= max_half_life_bars:
            if collect_diagnostics:
                pair_trail_index[pair_key]['elimination_stage'] = 'half_life'
            continue  # feedback #4: same half-life-max rule as S0's Section 6 screen
        n_pass_hl += 1
        if collect_diagnostics:
            pair_trail_index[pair_key]['passed_half_life'] = True
        crossings = count_crossings(spread)
        if crossings < min_crossings:
            if collect_diagnostics:
                pair_trail_index[pair_key]['elimination_stage'] = 'crossings'
            continue  # feedback #4: same min-crossings rule as S0's Section 6 screen
        n_pass_crossings += 1
        if collect_diagnostics:
            pair_trail_index[pair_key]['passed_crossings'] = True
            pair_trail_index[pair_key]['fully_qualifying'] = True
            pair_trail_index[pair_key]['elimination_stage'] = None
        corr = window_px[y_name].pct_change(fill_method=None).corr(window_px[x_name].pct_change(fill_method=None))
        out_rows.append({
            'rebalance_date': rebalance_date, 'pair': row['pair'], 'y': y_name, 'x': x_name,
            'eg_pvalue': row['eg_pvalue'], 'pvalue_adj': row['pvalue_adj'],
            'return_correlation': corr, 'alpha': a, 'beta': b,
            'spread_std': spread.std(), 'half_life_bars': hl, 'crossings': crossings,
            'n_obs': row['n_obs'],
        })
    result = pd.DataFrame(out_rows)
    if not collect_diagnostics:
        return result

    funnel = _funnel(n_candidate_pairs=len(pairs), n_testable_pairs=int(eg_df_t['pair'].nunique()),
                      n_directed_eg_tests=int(len(eg_df_t)), n_fdr_significant_pairs=int(len(sig_t)),
                      n_pass_beta=n_pass_beta, n_pass_half_life=n_pass_hl,
                      n_pass_crossings=n_pass_crossings, n_fully_qualifying=len(result))
    pair_trail = pd.DataFrame(list(pair_trail_index.values()))
    return result, funnel, pair_trail


def select_pairs_at_rebalance(rebalance_date: pd.Timestamp, price_data: pd.DataFrame, coins,
                               wf_config: dict, max_pairs: int = WF_CONFIG['MAX_PAIRS_PER_REBALANCE'],
                               collect_diagnostics: bool = False, enforce_one_pair_per_coin: bool = True):
    """WF1 pair selection at one rebalance date: rank the point-in-time screen's survivors by
    corrected p-value (ascending -- deterministic, no trading-performance input), then greedily
    accept in rank order, skipping any pair that would reuse a coin already claimed by a
    higher-ranked pair at this date (max one pair per coin).

    Returns a list of dicts, one per selected pair, each carrying its own frozen
    alpha/beta/spread_mean/spread_std/half_life/time_stop -- capped at `max_pairs`.

    `collect_diagnostics=True` (Phase B2 selection-funnel diagnostic) additionally returns a
    `selection_funnel` dict (screening funnel stages plus n_selected/n_removed_by_overlap/
    n_removed_by_max_pairs_cap), the `pair_trail` DataFrame from `screen_pairs_at_date`
    extended with 'selected' and an updated 'elimination_stage' for pairs removed by the
    one-pair-per-coin constraint or the max-pairs cap, and `overlap_conflicts` (one row per
    pair rejected specifically due to coin overlap, naming the conflicting coin and the
    already-selected pair/ranking metric that won it). This does not change the ranking rule,
    the greedy accept/reject loop, or which pairs get selected -- it only observes that exact
    loop. `collect_diagnostics=False` (the default) is unaffected.

    `enforce_one_pair_per_coin=False` (Phase B2.5 ablation, `stat_arb.ablation`) disables ONLY
    the coin-overlap rejection below -- the ranking order, the `max_pairs` cap, and every
    upstream statistical filter (`screen_pairs_at_date`) are untouched. A fully-qualifying pair
    is then never rejected merely because one of its coins already appears in another selected
    pair at this date; `n_removed_by_overlap` is always 0 in this mode by construction. This is
    the ONLY methodological difference between the primary `WF1_SEMIANNUAL_REESTIMATION`
    selection and the exploratory `WF1_NO_OVERLAP_CONSTRAINT` ablation -- every other function
    (execution, portfolio, metrics) is shared, unmodified. Default (`True`) is byte-for-byte
    unchanged from this parameter's absence.
    """
    if collect_diagnostics:
        screened, screen_funnel, pair_trail = screen_pairs_at_date(
            rebalance_date, price_data, coins, wf_config, collect_diagnostics=True)
    else:
        screened = screen_pairs_at_date(rebalance_date, price_data, coins, wf_config)

    if len(screened) == 0:
        if collect_diagnostics:
            selection_funnel = dict(screen_funnel)
            selection_funnel.update({'n_selected': 0, 'n_removed_by_overlap': 0, 'n_removed_by_max_pairs_cap': 0})
            return [], selection_funnel, pair_trail, []
        return []

    screened = screened.sort_values('pvalue_adj').reset_index(drop=True)
    time_stop_multiplier = wf_config['TIME_STOP_MULTIPLIER_PRIMARY']
    selected, used_coins = [], set()
    coin_owner = {}
    overlap_conflicts = []
    last_idx = -1
    hit_cap = False
    for idx, (_, row) in enumerate(screened.iterrows()):
        x_name, y_name = row['x'], row['y']
        last_idx = idx
        if enforce_one_pair_per_coin and (x_name in used_coins or y_name in used_coins):
            if collect_diagnostics:
                pair_key = row['pair']
                for coin in (x_name, y_name):
                    if coin in coin_owner:
                        winning_pair, winning_pvalue = coin_owner[coin]
                        overlap_conflicts.append({
                            'rejected_pair': pair_key, 'rejected_pvalue_adj': float(row['pvalue_adj']),
                            'conflicting_coin': coin, 'winning_pair': winning_pair,
                            'winning_pvalue_adj': float(winning_pvalue),
                        })
            continue
        ts = int(np.ceil(time_stop_multiplier * row['half_life_bars']))
        selected.append({
            'rebalance_date': rebalance_date, 'x': x_name, 'y': y_name,
            'pair': (x_name, y_name), 'alpha': row['alpha'], 'beta': row['beta'],
            'spread_mean': None, 'spread_std': row['spread_std'],
            'half_life_bars': row['half_life_bars'], 'time_stop_bars': ts,
            'pvalue_adj': row['pvalue_adj'], 'eg_pvalue': row['eg_pvalue'],
        })
        used_coins.add(x_name); used_coins.add(y_name)
        if collect_diagnostics:
            # NOTE: `selected[i]['pair']` is (x_name, y_name) in the WINNING REGRESSION
            # DIRECTION (matching the real pipeline's WF1_ALL_TRADES/WF1_ALL_RESULTS instance
            # keys). `pair_trail`'s 'pair' column -- and `row['pair']` here, inherited from
            # `run_engle_granger_tests`'s `tuple(sorted([c1, c2]))` -- is the UNORDERED,
            # alphabetically-sorted pair. These are two different tuples for the same
            # economic pair whenever the winning direction isn't already alphabetical (e.g.
            # ('OMUSDT','LDOUSDT') vs ('LDOUSDT','OMUSDT')). `coin_owner` is diagnostic-only
            # (reported in `overlap_conflicts`), so it is normalized to the sorted convention
            # here to match `rejected_pair`'s convention below -- never mix the two silently.
            winning_pair_sorted = tuple(sorted((x_name, y_name)))
            coin_owner[x_name] = (winning_pair_sorted, row['pvalue_adj'])
            coin_owner[y_name] = (winning_pair_sorted, row['pvalue_adj'])
        if len(selected) >= max_pairs:
            hit_cap = True
            break

    # spread_mean was not carried by screen_pairs_at_date -- recompute it here from the
    # exact same formation window (strictly prior to rebalance_date), for consistency with
    # how S0/S1 freeze spread_mean/spread_std at formation time.
    formation_months = wf_config['FORMATION_MONTHS']
    window_start = rebalance_date - pd.DateOffset(months=formation_months)
    window_px = price_data.loc[window_start:rebalance_date - pd.Timedelta(hours=4)]
    log_px = np.log(window_px)
    for sel in selected:
        spread = compute_spread(log_px[sel['y']], log_px[sel['x']], sel['alpha'], sel['beta'])
        sel['spread_mean'] = spread.mean()
        sel['spread_std'] = spread.std()

    if enforce_one_pair_per_coin:
        assert len({s['x'] for s in selected} | {s['y'] for s in selected}) == 2 * len(selected), (
            'One-pair-per-coin violated at rebalance %s' % rebalance_date)

    if not collect_diagnostics:
        return selected

    capped_rows = screened.iloc[last_idx + 1:] if hit_cap else screened.iloc[0:0]
    overlap_rejected_pairs = {c['rejected_pair'] for c in overlap_conflicts}

    # NOTE: `pair` values are (x, y) tuples. pandas' `.loc[tuple_key, col]` is ambiguous for a
    # regular (non-MultiIndex) Index of tuples -- it can raise inside pandas' indexing engine,
    # which tries to interpret the tuple as multi-level positional indexing. `.at[]` does not
    # have this ambiguity for scalar access, so it is used throughout below instead of `.loc`.
    pair_trail = pair_trail.set_index('pair', drop=False) if len(pair_trail) else pair_trail
    if len(pair_trail):
        pair_trail['selected'] = False
        for s in selected:
            # `s['pair']` is (x, y) in winning-direction order (see the coin_owner note
            # above); `pair_trail`'s index is the sorted-unordered-pair convention -- convert
            # before every lookup so a pair is never silently missed here.
            sorted_pair = tuple(sorted(s['pair']))
            if sorted_pair in pair_trail.index:
                pair_trail.at[sorted_pair, 'selected'] = True
        for p in overlap_rejected_pairs:
            if p in pair_trail.index and not pair_trail.at[p, 'selected']:
                pair_trail.at[p, 'elimination_stage'] = 'coin_overlap'
        for _, row in capped_rows.iterrows():
            p = row['pair']
            if p in pair_trail.index and not pair_trail.at[p, 'selected']:
                pair_trail.at[p, 'elimination_stage'] = 'max_pairs_cap'
        pair_trail = pair_trail.reset_index(drop=True)

    selection_funnel = dict(screen_funnel)
    selection_funnel.update({
        'n_selected': len(selected),
        'n_removed_by_overlap': len(overlap_rejected_pairs),
        'n_removed_by_max_pairs_cap': len(capped_rows),
    })
    return selected, selection_funnel, pair_trail, overlap_conflicts


def compute_pair_time_stops(pairs, formation_px: pd.DataFrame, multiplier: float):
    """S1's pair-specific time stop = ceil(multiplier * formation-period half-life), using
    ONLY formation-window prices. Reuses fit_static_ols/compute_spread/compute_half_life
    verbatim -- the same functions used for the Section 6 screen -- so a pair's S1 half-life
    is identical to its already-computed, already-frozen screening-stage half-life.

    Returns (time_stops: dict[pair -> int bars], ineligible: list[pairs]). A pair whose
    formation half-life is non-positive/infinite is marked ineligible rather than assigned a
    fallback stop, matching compute_half_life's own non-interpretability convention.
    """
    time_stops, ineligible = {}, []
    for x_name, y_name in pairs:
        log_x = np.log(formation_px[x_name])
        log_y = np.log(formation_px[y_name])
        a, b, _ = fit_static_ols(log_y, log_x)
        spread = compute_spread(log_y, log_x, a, b)
        hl = compute_half_life(spread)
        if not np.isfinite(hl) or hl <= 0:
            ineligible.append((x_name, y_name))
            continue
        time_stops[(x_name, y_name)] = int(np.ceil(multiplier * hl))
    return time_stops, ineligible
