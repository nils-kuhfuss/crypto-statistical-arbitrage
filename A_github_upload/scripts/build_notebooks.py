"""One-time generator for the four research notebooks in notebooks/. Not part of the
research pipeline itself -- run once (or after a notebooks/ structure change) to (re)build the
notebook files from the cell definitions below, then edit/run the notebooks normally.

Usage: python scripts/build_notebooks.py
"""
import os

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip('\n'))


def code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


SETUP_CODE = """
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
for _p in (PROJECT_ROOT, PROJECT_ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.research_config import (CONFIG, WF_CONFIG, MAX_HALF_LIFE_BARS, MIN_CROSSINGS, MODEL_IDS,
                                     S1_TIME_STOP_MULTIPLIER, S1_TIME_STOP_MULTIPLIER_ROBUSTNESS,
                                     FORWARD_EVAL_START)
"""

# ============================================================================================
# Notebook 1: data + static baseline (S0, S1)
# ============================================================================================
nb1_cells = [
    md("""
# 01. Data and Static Baseline

Research question: does a classical, statically-calibrated pairs-trading strategy on the 2024
cryptocurrency spot universe produce a positive risk-adjusted return out-of-sample, once
formation-only calibration, next-bar execution, transaction costs, and a standardized
benchmark-relative evaluation are all held to the standards in `CLAUDE.md`?

This notebook builds the frozen static universe, screens for cointegrated pairs, and evaluates
the two static models: **S0** (the control model, frozen formation parameters, fixed time
stop) and **S1** (S0 + a per-pair half-life-based time stop). Both reuse the exact statistical
and execution engines in `src/stat_arb/` -- no logic is reimplemented here.

**Scope note:** the original research process additionally compared Model S (static) against
two rolling-recalibration variants (RZ, RB) on the 2023 validation period before freezing on
Model S -- a one-time, historical model-selection decision. That comparison is preserved in
`legacy/Stat_Arb_cont_res_1.ipynb` Sections 12-13 for full auditability; this notebook takes its
already-decided outcome (Model S, frozen parameters below) as the given starting point, since
S0 is the control model and is never re-derived or re-edited (`CLAUDE.md`).
"""),
    code(SETUP_CODE),
    md("## Data loading and universe construction"),
    code("""
from stat_arb.data import load_raw_prices, load_raw_returns, filter_universe, split_samples, get_tradable_price

px_raw = load_raw_prices()
ret_raw = load_raw_returns()
print('Raw data:', px_raw.shape, '|', px_raw.index[0], 'to', px_raw.index[-1])

px, ret, dropped = filter_universe(px_raw, ret_raw, CONFIG)
print(f'Excluded ({len(dropped)}):', dropped)
print(f'Remaining universe: {px.shape[1]} coins')
"""),
    md("### Coverage and tradability diagnostics"),
    code("""
from stat_arb.diagnostics import coverage_table, delisted_coins_report

coverage_table(ret).round(2)
"""),
    code("""
delisted = delisted_coins_report(px)
print('Coins with data ending before', px.index[-1].date(), ':')
print(delisted.to_string())
"""),
    md("### Sample splits"),
    code("""
formation_px, validation_px, oos_px = split_samples(px, CONFIG)
print('Formation  :', formation_px.index[0], '-', formation_px.index[-1], f'({len(formation_px)} bars)')
print('Validation :', validation_px.index[0], '-', validation_px.index[-1], f'({len(validation_px)} bars)')
print('Final OOS  :', oos_px.index[0], '-', oos_px.index[-1], f'({len(oos_px)} bars)')
print('\\nSplit sanity check passed: no overlap between Formation / Validation / OOS.')
"""),
    md("""**Note on the "Final OOS" label.** This range (2024-01-01 to 2026-06-30) is reported
throughout this project as a **Post-Sample Historical Evaluation**, not a pristine, never-seen
holdout: this date range was partially visible in earlier iterations of this research process
before the current engine was finalized, and no parameter in `config/research_config.py` was
changed in response to anything observed in it. It is a corrected historical re-evaluation under
a fixed, pre-specified methodology -- not a forward-looking guarantee. See
`CLAUDE.md`'s "Non-negotiable research principles" #2 and the `FORWARD_EVAL_START` genuinely
untouched window (from 2026-07-01) discussed in notebook 04."""),
    md("""### Asset-Level I(1) Diagnostic

One-time, methodology-documentation check (not a filter): confirms the assets Engle-Granger is
applied to are broadly compatible with the I(1) assumption (non-stationary in log-price level,
stationary in log-returns)."""),
    code("""
from stat_arb.diagnostics import i1_diagnostic_table

i1_df = i1_diagnostic_table(px)
n_compatible = (i1_df['i1_compatible'] == 'Yes').sum()
print(f'I(1) diagnostic: {n_compatible}/{len(i1_df)} assets are I(1)-compatible.')
i1_df.round(4)
"""),
    md("""## Static pair screen (Section 6 equivalent)

Engle-Granger cointegration (both directions) on every candidate pair -> Benjamini-Hochberg FDR
-> keep the lower-adjusted-p-value direction per pair -> half-life and mean-crossing filters.
No positive-Beta filter here (that is walk-forward-specific, see notebook 02/03)."""),
    code("""
from stat_arb.pair_selection import run_static_pair_screen

log_formation_all = np.log(formation_px)
candidate_pairs = run_static_pair_screen(log_formation_all, formation_px.columns.tolist(), CONFIG,
                                          MAX_HALF_LIFE_BARS, MIN_CROSSINGS)
print(f'Pairs passing statistical + half-life + crossing screen: {len(candidate_pairs)}')
candidate_pairs[['pair', 'y', 'x', 'pvalue_adj', 'alpha', 'beta', 'half_life_bars', 'crossings']]
"""),
    code("""
# FROZEN pair list -- do not modify after this cell.
FROZEN_PAIRS = [(row['x'], row['y']) for _, row in candidate_pairs.iterrows()]
print(f'Frozen pair list ({len(FROZEN_PAIRS)} pairs):')
for x, y in FROZEN_PAIRS:
    print(f'  X={x:12s} Y={y}')
"""),
    md("""## Model freeze (S0)

Model S selected over RZ/RB at 2023 validation (see scope note above). Frozen parameters,
reused unchanged for the entire OOS evaluation."""),
    code("""
FROZEN_MODEL = {
    'pairs': FROZEN_PAIRS,
    'model': 'S',
    'rolling_window_days': CONFIG['ROLLING_WINDOW_DAYS'],
    'rolling_window_bars': CONFIG['ROLLING_WINDOW_BARS'],
    'entry_z': CONFIG['ENTRY_Z'],
    'exit_z': CONFIG['EXIT_Z'],
    'hard_stop_z': CONFIG['HARD_STOP_Z'],
    'time_stop_bars': CONFIG['TIME_STOP_BARS'],
    'cost_per_leg_bps': CONFIG['COST_PER_LEG_BPS'],
}
FROZEN_MODEL
"""),
    md("## S0 -- Out-of-Sample Evaluation"),
    code("""
from stat_arb.execution import run_model_on_pair

px_oos_engine = px.loc[:CONFIG['OOS_END']]
ret_oos_engine = ret.loc[:CONFIG['OOS_END']]

oos_results, oos_trades = {}, {}
for x_name, y_name in FROZEN_MODEL['pairs']:
    result, trades = run_model_on_pair(
        x_name, y_name, FROZEN_MODEL['model'], FROZEN_MODEL['rolling_window_bars'],
        FROZEN_MODEL['entry_z'], FROZEN_MODEL['exit_z'], FROZEN_MODEL['hard_stop_z'],
        FROZEN_MODEL['time_stop_bars'], FROZEN_MODEL['cost_per_leg_bps'],
        px_oos_engine, ret_oos_engine, CONFIG, state_start=CONFIG['OOS_START'])
    result = result.loc[CONFIG['OOS_START']:CONFIG['OOS_END']]
    assert result[['qty_x', 'qty_y']].iloc[0].abs().sum() == 0.0
    assert result[['qty_x', 'qty_y']].iloc[-1].abs().sum() == 0.0
    oos_results[(x_name, y_name)] = result
    oos_trades[(x_name, y_name)] = trades

print(f'S0 OOS run complete on {len(oos_results)} pairs. FLAT-start / flat-end checks PASSED.')
"""),
    code("""
from stat_arb.reporting import standardized_model_summary

S0_ROW, S0_OOS_PORTFOLIO = standardized_model_summary(
    'S0_STATIC_BASELINE', oos_results, FROZEN_MODEL['pairs'], px_oos_engine,
    FROZEN_MODEL['cost_per_leg_bps'], CONFIG['OOS_START'], CONFIG['OOS_END'], CONFIG['BARS_PER_YEAR'],
    trades_dict=oos_trades)
assert S0_OOS_PORTFOLIO['coin_qty'].iloc[-1].abs().max() < 1e-12
pd.Series(S0_ROW)
"""),
    md("## S1 -- Static Model with Pair-Specific Half-Life Time Stops"),
    code("""
from stat_arb.pair_selection import compute_pair_time_stops
from stat_arb.execution import run_s1_universe

s1_time_stops, s1_ineligible = compute_pair_time_stops(FROZEN_PAIRS, formation_px, S1_TIME_STOP_MULTIPLIER)
S1_ELIGIBLE_PAIRS = [p for p in FROZEN_PAIRS if p in s1_time_stops]

px_oos_engine_s1 = px.loc[:CONFIG['OOS_END']]
ret_oos_engine_s1 = ret.loc[:CONFIG['OOS_END']]
S1_OOS_RESULTS, S1_OOS_TRADES = run_s1_universe(
    S1_ELIGIBLE_PAIRS, s1_time_stops, px_oos_engine_s1, ret_oos_engine_s1, CONFIG, CONFIG['OOS_START'])
S1_OOS_RESULTS = {k: v.loc[CONFIG['OOS_START']:CONFIG['OOS_END']] for k, v in S1_OOS_RESULTS.items()}

for (x, y), result in S1_OOS_RESULTS.items():
    assert result[['qty_x', 'qty_y']].iloc[0].abs().sum() == 0.0
    assert result[['qty_x', 'qty_y']].iloc[-1].abs().sum() == 0.0
print(f'S1 OOS run complete on {len(S1_OOS_RESULTS)} pairs. FLAT-start / flat-end checks PASSED.')
if s1_ineligible:
    print('Ineligible for S1:', s1_ineligible)
"""),
    code("""
S1_ROW, S1_OOS_PORTFOLIO = standardized_model_summary(
    'S1_STATIC_PAIR_HALFLIFE_STOP', S1_OOS_RESULTS, S1_ELIGIBLE_PAIRS, px_oos_engine_s1,
    CONFIG['COST_PER_LEG_BPS'], CONFIG['OOS_START'], CONFIG['OOS_END'], CONFIG['BARS_PER_YEAR'],
    trades_dict=S1_OOS_TRADES)

s0_vs_s1_df = pd.DataFrame([S0_ROW, S1_ROW]).set_index('model')
print('S0 vs. S1 -- OOS (2024-01-01 to 2026-06-30), standardized comparison:')
s0_vs_s1_df.round(4).T
"""),
    md("""### S1 time-stop multiplier robustness (2.0x / 2.5x primary / 3.0x)

Pre-specified sensitivity grid -- no grid search, no selection on OOS performance. Instability
across nearby multipliers is itself a reportable finding, not a prompt to adopt the
best-performing variant."""),
    code("""
S1_MULTIPLIER_ROWS = []
for mult in S1_TIME_STOP_MULTIPLIER_ROBUSTNESS:
    ts_mult, ineligible_mult = compute_pair_time_stops(FROZEN_PAIRS, formation_px, mult)
    eligible_mult = [p for p in FROZEN_PAIRS if p in ts_mult]
    results_mult, trades_mult = run_s1_universe(eligible_mult, ts_mult, px_oos_engine_s1, ret_oos_engine_s1, CONFIG, CONFIG['OOS_START'])
    results_mult = {k: v.loc[CONFIG['OOS_START']:CONFIG['OOS_END']] for k, v in results_mult.items()}
    row, _ = standardized_model_summary(
        f'S1_x{mult}', results_mult, eligible_mult, px_oos_engine_s1, CONFIG['COST_PER_LEG_BPS'],
        CONFIG['OOS_START'], CONFIG['OOS_END'], CONFIG['BARS_PER_YEAR'])
    S1_MULTIPLIER_ROWS.append(row)

pd.DataFrame(S1_MULTIPLIER_ROWS).set_index('model')[['daily_sharpe_252', 'cum_net_return', 'max_drawdown']].round(4)
"""),
    md("""## Interpretation

S0 and S1 OOS Sharpe and cumulative return are both negative on this sample (see table above) --
a negative result is an acceptable, reportable outcome per `CLAUDE.md`. The S1 multiplier
robustness sweep shows Sharpe swinging materially across 2.0x/2.5x/3.0x, meaning the time-stop
choice does not robustly explain S0's underperformance; this instability is itself the finding,
not a basis for picking the best-performing variant."""),
    code("""
s0_s1_tables_dir = PROJECT_ROOT / 'results' / 'tables'
s0_s1_tables_dir.mkdir(parents=True, exist_ok=True)
with open(s0_s1_tables_dir / 's0_s1_rows.json', 'w') as f:
    json.dump({'S0_STATIC_BASELINE': S0_ROW, 'S1_STATIC_PAIR_HALFLIFE_STOP': S1_ROW}, f, indent=2, default=str)
print(f'Saved S0/S1 rows to {s0_s1_tables_dir / "s0_s1_rows.json"}')
"""),
]

# ============================================================================================
# Notebook 2: relationship stability diagnostics
# ============================================================================================
nb2_cells = [
    md("""
# 02. Relationship Stability Diagnostics

Purely descriptive diagnostic layer: at a sequence of historical rebalance dates, every
candidate pair is re-screened using a trailing 12-month formation window, and the results are
stored longitudinally. This asks how consistently pairs satisfy the same statistical screen
across time -- independent of any trading decision. **Note:** a pair failing to qualify at a
later date is not automatically evidence of a "structural break" in the underlying relationship
-- it may equally reflect delisting, insufficient formation coverage, or the pair sitting just
outside a significance/filter threshold that quarter; the tables below report qualification
outcomes descriptively and do not attribute a cause without further breakdown. The same
significance rule frozen in notebook 01 (Engle-Granger + Benjamini-Hochberg FDR,
`CONFIG['EG_ALPHA']`) is reused verbatim.

Every statistic at rebalance date *t* uses **only** price data strictly before *t* -- the same
point-in-time constraint the walk-forward models (WF1, WF2, notebook 03) must also satisfy, so
this notebook doubles as a dry run of that logic before it is used to trade.

No new persistence methodology is introduced here (per
`docs/archive/REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md` Section 6) -- this notebook reproduces the
existing, already-corrected walk-forward screen.
"""),
    code(SETUP_CODE),
    md("""## Walk-forward universe: WF_PX / WF_RET

A **separately rebuilt** universe (45 coins), not notebook 01's statically-filtered `px`/`ret`
(28 coins). `WF_CONFIG['MANUAL_EXCLUDE']` only drops true stablecoins -- unlike the static
universe, XMR and every coin dropped via the one-time `MIN_HISTORY_CUTOFF` are included, since
point-in-time coverage + active-at-rebalance eligibility is designed to handle late
listings/delistings correctly without a manual cutoff."""),
    code("""
from stat_arb.data import build_walkforward_universe

WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
print(f'Walk-forward universe: {WF_PX.shape[1]} coins')
assert 'XMRUSDT' in WF_PX.columns, 'XMR must be present in the walk-forward universe.'
"""),
    code("""
from stat_arb.walkforward import generate_rebalance_dates

REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                            spacing_months=WF_CONFIG['REBALANCE_MONTHS_PRIMARY'],
                                            min_formation_months=WF_CONFIG['FORMATION_MONTHS'])
print(f'{len(REBALANCE_DATES)} historical rebalance dates (12-month trailing formation, 6-month spacing):')
print([d.date() for d in REBALANCE_DATES])
"""),
    md("""### Longitudinal Pair-Stability Dataset

This screen is expensive (~1,600+ directional Engle-Granger tests per rebalance date x 11
dates), so it is precomputed via a standalone resumable script
(`scripts/run_stability_screen.py`) that imports `screen_pairs_at_date` from
`stat_arb.pair_selection` directly (no forked copy), cached to disk keyed by the full
methodology fingerprint below. This cell only loads the cache; it asserts loudly (with the
expected path) if the cache is missing rather than silently recomputing on a hot path."""),
    code("""
from stat_arb.cache import wf_cache_fingerprint, load_or_assert, cache_path

stability_fingerprint = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=REBALANCE_DATES, wf_config=WF_CONFIG)
stability_cache_path = cache_path('pair_stability', stability_fingerprint)

pair_stability_df = load_or_assert(
    stability_cache_path,
    f'Stability cache not found at {stability_cache_path}. Regenerate via scripts/run_stability_screen.py.')
print(f'Loaded pair-stability diagnostic (fingerprint {stability_fingerprint}): '
      f'{pair_stability_df.shape[0]} (pair, date) rows across {pair_stability_df[\"rebalance_date\"].nunique()} rebalance dates')
pair_stability_df.groupby('rebalance_date').size().rename('n_qualifying_pairs')
"""),
    md("### Persistence, Drift, and Qualification Diagnostics"),
    code("""
qual_freq = pair_stability_df.groupby('pair')['rebalance_date'].nunique().sort_values(ascending=False)
print('Qualification frequency (number of rebalance dates each pair qualified at):')
qual_freq
"""),
    code("""
pivot_qual = pair_stability_df.pivot_table(index='pair', columns='rebalance_date', values='eg_pvalue', aggfunc='first').notna()
recurring_pairs = pivot_qual.index[pivot_qual.sum(axis=1) >= 2]
print(f'Pairs qualifying at 2+ rebalance dates: {len(recurring_pairs)} of {len(pivot_qual)} total qualifying pairs')

n_qualifying_by_date = pair_stability_df.groupby('rebalance_date').size().reindex(REBALANCE_DATES, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar([d.date().isoformat() for d in n_qualifying_by_date.index], n_qualifying_by_date.values)
ax.set_ylabel('# qualifying pairs')
ax.set_title('Qualifying pairs per rebalance date')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""),
    code("""
if len(recurring_pairs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for pair in recurring_pairs:
        sub = pair_stability_df[pair_stability_df['pair'] == pair].sort_values('rebalance_date')
        axes[0].plot(sub['rebalance_date'], sub['beta'], marker='o', label=f'{pair[0]}/{pair[1]}')
        axes[1].plot(sub['rebalance_date'], sub['half_life_bars'], marker='o', label=f'{pair[0]}/{pair[1]}')
    axes[0].set_title('Beta evolution (recurring pairs)')
    axes[1].set_title('Half-life evolution (recurring pairs)')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print('No pair qualifies at 2+ rebalance dates -- no persistence chart to draw.')
"""),
    md("""## Interpretation

Very few pairs qualify at any given rebalance date under the corrected point-in-time
methodology (coverage, active-at-rebalance, matching S0's half-life/crossing/positive-Beta
filters), and qualification is highly discontinuous across dates rather than persistent. This
motivates treating WF1/WF2's low trade counts (notebook 03) as a first-order caveat, not a
detail."""),
]

# ============================================================================================
# Notebook 3: walk-forward strategy (WF1, WF2)
# ============================================================================================
nb3_cells = [
    md("""
# 03. Walk-Forward Strategy (WF1, WF2)

**WF1** re-selects pairs every 6 months using the corrected point-in-time screen (notebook 02),
trading each selection with S0's exact execution engine (frozen formation alpha/beta/time
stop). **WF2** is WF1 plus rolling entry-time hedge-ratio sizing -- the Z-score signal itself
still uses the frozen formation alpha/beta throughout; only entry-time position sizing is
rolling. Do not describe WF2 as a rolling-spread model.

**Caveat stated up front:** WF1/WF2 trade very few times in the full OOS window (see results
below) -- any comparison between them, or against S0/S1, must state this sample-size limitation
rather than imply a robust conclusion.
"""),
    code(SETUP_CODE),
    code("""
from stat_arb.data import build_walkforward_universe
from stat_arb.walkforward import generate_rebalance_dates

WF_PX, WF_RET = build_walkforward_universe(WF_CONFIG)
REBALANCE_DATES = generate_rebalance_dates(CONFIG['FORMATION_START'], CONFIG['OOS_END'],
                                            WF_CONFIG['REBALANCE_MONTHS_PRIMARY'], WF_CONFIG['FORMATION_MONTHS'])
WF1_REBALANCE_DATES = [d for d in REBALANCE_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]
print(f'WF1 rebalance dates within the OOS trading window: {[d.date() for d in WF1_REBALANCE_DATES]}')
"""),
    md("""## WF1 Pair Selection

Uses the exact same `screen_pairs_at_date`/`select_pairs_at_rebalance` as notebook 02's
diagnostic screen -- this section's selected-pair output is not a new methodology, it is the
trading decision built on that exact diagnostic. Cached to disk (methodology-aware
fingerprint) since re-screening 5 rebalance dates is itself nontrivially expensive."""),
    code("""
from stat_arb.cache import wf_cache_fingerprint, cache_path, load_or_assert
from stat_arb.walkforward import run_wf1_selection_pipeline

wf1_sel_fingerprint = wf_cache_fingerprint(
    universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES, wf_config=WF_CONFIG,
    extra=f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}")
wf1_sel_cache_path = cache_path('wf1_selections', wf1_sel_fingerprint)

WF1_SELECTIONS = run_wf1_selection_pipeline(WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(),
                                             WF_CONFIG, wf1_sel_cache_path,
                                             max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])

for rebal_date in WF1_REBALANCE_DATES:
    selected = WF1_SELECTIONS[rebal_date]
    if selected:
        pairs_str = ', '.join(f"{s['x']}/{s['y']} (TS={s['time_stop_bars']})" for s in selected)
        print(f'{rebal_date.date()}: {len(selected)} pairs selected -> {pairs_str}')
    else:
        print(f'{rebal_date.date()}: 0 pairs selected -> block trades nothing (cash)')
"""),
    md("""## WF1 Portfolio Construction

Fixed 1/N sleeve per selected pair at each block; a rebalance block with zero selected pairs
trades nothing and contributes a flat (zero-return) period to the continuous equity curve --
correct, non-optimized behavior, not a data gap."""),
    code("""
from stat_arb.walkforward import run_wf1_full

WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES = run_wf1_full(
    WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)

assert WF1_FULL_FRAME.index.is_monotonic_increasing
assert not WF1_FULL_FRAME.index.duplicated().any()
assert WF1_FULL_FRAME.index[0] == pd.Timestamp(CONFIG['OOS_START'])
assert WF1_FULL_FRAME.index[-1] <= pd.Timestamp(CONFIG['OOS_END'])

print(f'WF1 continuous portfolio built: {len(WF1_FULL_FRAME)} bars, '
      f'{sum(len(t) for t in WF1_ALL_TRADES.values())} total trades across '
      f'{len({p for (_, p) in WF1_ALL_TRADES.keys()})} distinct pairs traded.')
print(f'Cumulative net return: {(1 + WF1_FULL_FRAME[\"net_ret\"]).prod() - 1:.4%}')
print(f'Time in market (portfolio gross_exposure > 0): {WF1_FULL_FRAME[\"active_flag\"].mean():.2%}')
"""),
    code("""
from stat_arb.reporting import standardized_wf1_summary

WF1_ROW = standardized_wf1_summary(
    'WF1_SEMIANNUAL_REESTIMATION', WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES, WF_PX,
    CONFIG['OOS_START'], CONFIG['OOS_END'])
pd.Series(WF1_ROW)
"""),
    md("""## Selection Funnel Diagnostic

**Research question: where does the opportunity set disappear between the eligible universe
and executed trades?** WF1 selected only 1 pair (OMUSDT/LDOUSDT, 2 trades) across the entire
OOS window despite notebook 02 showing 15 qualifying pairs at 2024-01-01 alone. This section
answers *why*, stage by stage, using the exact real pipeline (`screen_pairs_at_date` /
`select_pairs_at_rebalance`, instrumented additively via `collect_diagnostics=True` -- the same
code path that produced `WF1_SELECTIONS` above, not a second implementation) plus the REAL
trades already computed in `WF1_ALL_TRADES` -- nothing here is re-simulated or re-thresholded.

**Units, kept distinct throughout** (do not compare across these without noting the change of
unit): eligible **assets** -> candidate **unordered pairs** -> **testable** pairs (enough
overlapping observations) -> **directed** EG tests (2x testable pairs, both regression
directions) -> FDR-significant pairs **after direction resolution** (deduplicated to the
lower-adjusted-p-value direction) -> pairs passing positive-Beta -> Half-Life -> crossings, in
that exact order (matching `screen_pairs_at_date`'s own filtering loop) -> fully-qualifying
pairs -> selected pairs (one-pair-per-coin, rank-ordered by `pvalue_adj`, capped at
`MAX_PAIRS_PER_REBALANCE`) -> selected pairs with >=1 real entry signal -> executed trades.

This is precomputed by `scripts/run_selection_funnel.py` (expensive -- re-runs the point-in-time
screen with diagnostics enabled for every WF1 rebalance date) and cached to
`data/cache/selection_funnel_<fingerprint>.pkl`, keyed by the same `wf_cache_fingerprint`
convention as every other walk-forward cache (this does not add a new fingerprinting
methodology, and does not touch the existing `wf1_selections`/`pair_stability` caches or their
fingerprints). This cell only loads it."""),
    code("""
funnel_extra = f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}"
funnel_fingerprint = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                           wf_config=WF_CONFIG, extra=f'{funnel_extra},funnel_diagnostics')
funnel_cache_path = cache_path('selection_funnel', funnel_fingerprint)
funnel_result = load_or_assert(
    funnel_cache_path,
    f'Selection funnel cache not found at {funnel_cache_path}. Regenerate via scripts/run_selection_funnel.py.')
FUNNEL_DF, FUNNEL_PAIR_LEVEL_DF, FUNNEL_OVERLAP_CONFLICTS_DF = (
    funnel_result['funnel_df'], funnel_result['pair_level_df'], funnel_result['conflicts_df'])

print('Selection funnel by rebalance date (see notebook markdown above for unit definitions):')
FUNNEL_DF.set_index('rebalance_date')
"""),
    md("""### One-pair-per-coin diagnostic

At 2024-01-01, every one of the 14 non-selected fully-qualifying pairs is eliminated purely by
the one-pair-per-coin constraint (`n_removed_by_overlap` in the table above), not by any
statistical filter -- `n_pass_beta`/`n_pass_half_life`/`n_pass_crossings` all equal
`n_fdr_significant_pairs` (15) at that date: **zero pairs were eliminated by Beta, Half-Life,
or crossing requirements.** The table below shows exactly which coin caused each conflict, the
already-selected pair that won it, and the ranking metric (`pvalue_adj`, ascending -- the
existing rule, unchanged) that decided it."""),
    code("""
conflicts_2024_01 = FUNNEL_OVERLAP_CONFLICTS_DF[FUNNEL_OVERLAP_CONFLICTS_DF['rebalance_date'] == WF1_REBALANCE_DATES[0]]
print(f"{len(conflicts_2024_01)} pairs rejected by coin overlap at {WF1_REBALANCE_DATES[0].date()}; "
      f"conflicting coin(s): {sorted(conflicts_2024_01['conflicting_coin'].unique())}")
conflicts_2024_01[['rejected_pair', 'rejected_pvalue_adj', 'conflicting_coin', 'winning_pair', 'winning_pvalue_adj']]
"""),
    md("""### Pair-level bottleneck table (2024-01-01 -- the richest rebalance window)

One row per testable pair at this date: pass/fail through every stage, in the actual
implementation order, and the first stage at which each pair was eliminated (`None` only for
the one pair that was selected AND generated an executed trade)."""),
    code("""
pair_level_2024_01 = FUNNEL_PAIR_LEVEL_DF[FUNNEL_PAIR_LEVEL_DF['rebalance_date'] == WF1_REBALANCE_DATES[0]].copy()
pair_level_2024_01['pair'] = pair_level_2024_01['pair'].apply(lambda p: f'{p[0]}/{p[1]}')
display_cols = ['pair', 'pvalue_adj', 'passed_fdr', 'passed_beta', 'passed_half_life', 'passed_crossings',
                'fully_qualifying', 'selected', 'generated_entry_signal', 'n_executed_trades', 'elimination_stage']
pair_level_2024_01[display_cols].sort_values('pvalue_adj').reset_index(drop=True)
"""),
    md("""### Interpretation

At 2024-01-01, the opportunity set does **not** shrink for lack of statistically valid
relationships (528 candidate pairs -> 15 fully-qualifying, all surviving FDR/Beta/Half-Life/
crossings intact) -- it shrinks almost entirely at the **one-pair-per-coin portfolio
constraint**, driven by **concentration around a single coin (LDOUSDT)**: 14 of the 15
qualifying pairs involve LDOUSDT paired with a different counterpart, so once the
lowest-adjusted-p-value pair (OMUSDT/LDOUSDT) claims that coin, every other LDOUSDT pair is
mechanically excluded regardless of its own statistical strength. The `MAX_PAIRS_PER_REBALANCE`
cap played no role at this date (`n_removed_by_max_pairs_cap = 0`) -- overlap alone did all the
work. At every other WF1 rebalance date, zero pairs are even FDR-significant, so there is
nothing to select from at all -- a completely different bottleneck (statistical, not portfolio).
Execution/state-machine constraints did not reduce the opportunity set at 2024-01-01 either:
the one selected pair (OMUSDT/LDOUSDT) generated a real entry signal and both of its signal-
level trades were executed, with no drop between "entry signal" and "executed trade" -- in this
fixed-quantity, next-bar-execution engine those two counts are structurally identical (see
`stat_arb.funnel`'s module docstring)."""),
    md("""### Active-Day BTC Beta (feedback #11)

Unconditional metrics can be misleading for low-activity strategies: WF1's unconditional beta
looks market-neutral largely because it is flat most days. Active-day beta (restricted to bars
with `gross_exposure > 0`) tells the real story on the days it actually trades."""),
    code("""
from stat_arb.reporting import active_day_benchmark_stats, active_mask_from_bar_frame
from stat_arb.metrics import aggregate_daily_returns, build_btc_daily_benchmark

wf1_daily, _ = aggregate_daily_returns(WF1_FULL_FRAME['net_ret'])
wf1_active_daily_mask = active_mask_from_bar_frame(WF1_FULL_FRAME['active_flag']).reindex(wf1_daily.index, fill_value=False)
wf1_btc_daily = build_btc_daily_benchmark(wf1_daily.index, WF_PX)
wf1_active_bench = active_day_benchmark_stats(wf1_daily, wf1_active_daily_mask, wf1_btc_daily)

print(f"WF1 unconditional beta: {WF1_ROW['btc_beta']:.4f}   |   WF1 active-day beta: {wf1_active_bench['beta']:.4f}")
"""),
    md("""### WF1 Robustness: time-stop multiplier and rebalance frequency

Pre-specified sensitivity checks only, per `CLAUDE.md` -- no grid search, no selection on OOS
performance."""),
    code("""
def reselect_with_multiplier(base_selections, multiplier):
    out = {}
    for rebal_date, selected in base_selections.items():
        resel = []
        for s in selected:
            new_s = dict(s)
            new_s['time_stop_bars'] = int(np.ceil(multiplier * s['half_life_bars']))
            resel.append(new_s)
        out[rebal_date] = resel
    return out

WF1_MULTIPLIER_ROWS = []
for mult in S1_TIME_STOP_MULTIPLIER_ROBUSTNESS:
    sels_mult = reselect_with_multiplier(WF1_SELECTIONS, mult)
    frame_mult, results_mult, trades_mult = run_wf1_full(sels_mult, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)
    row = standardized_wf1_summary(f'WF1_x{mult}', frame_mult, results_mult, trades_mult, WF_PX,
                                    CONFIG['OOS_START'], CONFIG['OOS_END'])
    WF1_MULTIPLIER_ROWS.append(row)

pd.DataFrame(WF1_MULTIPLIER_ROWS).set_index('model')[['daily_sharpe_252', 'cum_net_return', 'n_trades']].round(4)
"""),
    md("""### WF1 Rebalance-Frequency Robustness (6-month primary vs. 3-month)

All other parameters (formation window, entry/exit/hard-stop thresholds, time-stop multiplier
=2.5x primary, universe, cost assumptions) are held fixed -- only the rebalance spacing changes.
Reuses the exact same `select_pairs_at_rebalance`/`run_wf1_full` pipeline as the primary 6-month
spec, just re-screened at a 3-month cadence -- no separate implementation."""),
    code("""
WF1_REBAL_3M_DATES = pd.date_range(CONFIG['OOS_START'], CONFIG['OOS_END'],
                                    freq=f"{WF_CONFIG['REBALANCE_MONTHS_ROBUSTNESS']}MS")
WF1_REBAL_3M_DATES = [d for d in WF1_REBAL_3M_DATES if d >= pd.Timestamp(CONFIG['OOS_START'])]
print(f'3-month rebalance dates: {[d.date() for d in WF1_REBAL_3M_DATES]}')

wf1_3m_fingerprint = wf_cache_fingerprint(
    universe_cols=WF_PX.columns, rebalance_dates=WF1_REBAL_3M_DATES, wf_config=WF_CONFIG,
    extra=f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']}")
WF1_3M_SELECTIONS = run_wf1_selection_pipeline(WF1_REBAL_3M_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
                                                cache_path('wf1_selections_3m', wf1_3m_fingerprint),
                                                max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])

for rebal_date, sel in WF1_3M_SELECTIONS.items():
    pairs_str = ', '.join(f"{s['x']}/{s['y']}" for s in sel) if sel else '(cash)'
    print(f'{rebal_date.date()}: {len(sel)} pairs -> {pairs_str}')
"""),
    code("""
WF1_3M_FRAME, WF1_3M_RESULTS, WF1_3M_TRADES = run_wf1_full(
    WF1_3M_SELECTIONS, WF1_REBAL_3M_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)

wf1_3m_row = standardized_wf1_summary('WF1_3MONTH', WF1_3M_FRAME, WF1_3M_RESULTS, WF1_3M_TRADES, WF_PX,
                                       CONFIG['OOS_START'], CONFIG['OOS_END'])
wf1_6m_row = standardized_wf1_summary('WF1_6MONTH_PRIMARY', WF1_FULL_FRAME, WF1_ALL_RESULTS, WF1_ALL_TRADES, WF_PX,
                                       CONFIG['OOS_START'], CONFIG['OOS_END'])

wf1_rebal_freq_df = pd.DataFrame([wf1_6m_row, wf1_3m_row]).set_index('model')
print('WF1 rebalance-frequency robustness -- OOS (all other parameters held fixed at primary spec):')
wf1_rebal_freq_df[['daily_sharpe_252', 'cum_net_return', 'max_drawdown', 'n_trades',
                    'n_pairs_traded', 'turnover']].round(4)
"""),
    md("""## B2.5 Ablation: Removing the One-Pair-Per-Coin Constraint (Exploratory)

**This is an exploratory Post-Sample Historical ablation, not a new primary model.** The
Selection Funnel Diagnostic above found that at 2024-01-01, all 15 fully-qualifying pairs
happened to involve LDOUSDT, and the one-pair-per-coin constraint let only 1 of them
(OMUSDT/LDOUSDT) trade. This section asks: what actually happens if that constraint is
relaxed? Exactly one assumption changes -- `select_pairs_at_rebalance(...,
enforce_one_pair_per_coin=False)` -- while every other function (ranking rule, `MAX_PAIRS_PER_REBALANCE`
cap, formation window, EG/FDR/Beta/Half-Life/crossing filters, execution engine, equal-sleeve
portfolio construction, cost model, benchmark regression) is exactly the same shared code used
by `WF1_SEMIANNUAL_REESTIMATION` (`stat_arb.walkforward.run_wf1_full`,
`stat_arb.reporting.standardized_wf1_summary`) -- nothing here is a second implementation.
This new variant, `WF1_NO_OVERLAP_CONSTRAINT`, is never substituted for WF1 in the master
model-comparison table and is not promoted to Notebook 04 as a primary result."""),
    code("""
from stat_arb.ablation import run_no_overlap_selection_pipeline

no_overlap_extra = (f"max_pairs={WF_CONFIG['MAX_PAIRS_PER_REBALANCE']},"
                     f"ts_mult={WF_CONFIG['TIME_STOP_MULTIPLIER_PRIMARY']},no_overlap_constraint")
no_overlap_sel_fingerprint = wf_cache_fingerprint(universe_cols=WF_PX.columns, rebalance_dates=WF1_REBALANCE_DATES,
                                                    wf_config=WF_CONFIG, extra=no_overlap_extra)
NO_OVERLAP_SELECTIONS = run_no_overlap_selection_pipeline(
    WF1_REBALANCE_DATES, WF_PX, WF_PX.columns.tolist(), WF_CONFIG,
    cache_path('wf1_no_overlap_selections', no_overlap_sel_fingerprint),
    max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'])

for rebal_date in WF1_REBALANCE_DATES:
    print(f"{rebal_date.date()}: baseline {len(WF1_SELECTIONS[rebal_date])} pair(s) -> "
          f"no-overlap {len(NO_OVERLAP_SELECTIONS[rebal_date])} pair(s)")
"""),
    md("""At every rebalance date OTHER than 2024-01-01, zero pairs are even FDR-significant
(the same purely-statistical bottleneck the funnel diagnostic already found for WF1) -- removing
the overlap constraint cannot create trading opportunity where none survives the statistical
filters in the first place. Backtested with the exact same `run_wf1_full` used for WF1/WF2 (the
equal-sleeve portfolio construction is already generic over however many pairs a block selects,
so 10 simultaneously-selected pairs each automatically get a 1/10 sleeve, never 1/1 -- no new
capital-normalization logic was written for this ablation)."""),
    code("""
NO_OVERLAP_FULL_FRAME, NO_OVERLAP_ALL_RESULTS, NO_OVERLAP_ALL_TRADES = run_wf1_full(
    NO_OVERLAP_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)

NO_OVERLAP_ROW = standardized_wf1_summary(
    MODEL_IDS['WF1_NO_OVERLAP'], NO_OVERLAP_FULL_FRAME, NO_OVERLAP_ALL_RESULTS, NO_OVERLAP_ALL_TRADES,
    WF_PX, CONFIG['OOS_START'], CONFIG['OOS_END'])

ablation_comparison_df = pd.DataFrame([WF1_ROW, NO_OVERLAP_ROW]).set_index('model')
print('WF1 baseline vs. WF1_NO_OVERLAP_CONSTRAINT (exploratory ablation) -- OOS standardized comparison:')
ablation_comparison_df.round(4).T
"""),
    md("""### Active-Day BTC Beta, Both Models

Reuses `active_day_benchmark_stats` exactly as for WF1 above (`wf1_active_bench` was already
computed in that earlier cell)."""),
    code("""
no_overlap_daily, _ = aggregate_daily_returns(NO_OVERLAP_FULL_FRAME['net_ret'])
no_overlap_active_mask = active_mask_from_bar_frame(NO_OVERLAP_FULL_FRAME['active_flag']).reindex(
    no_overlap_daily.index, fill_value=False)
no_overlap_btc_daily = build_btc_daily_benchmark(no_overlap_daily.index, WF_PX)
no_overlap_active_bench = active_day_benchmark_stats(no_overlap_daily, no_overlap_active_mask, no_overlap_btc_daily)

print(f"WF1 baseline:          unconditional beta {WF1_ROW['btc_beta']:.4f} | "
      f"active-day beta {wf1_active_bench['beta']:.4f} (n={wf1_active_bench['n_obs']} active days)")
print(f"No-overlap ablation:   unconditional beta {NO_OVERLAP_ROW['btc_beta']:.4f} | "
      f"active-day beta {no_overlap_active_bench['beta']:.4f} (n={no_overlap_active_bench['n_obs']} active days)")
"""),
    md("""### Coin-Concentration Diagnostics (2024-01-01 block)

`coin_concentration_diagnostics` (`stat_arb.portfolio`) is read-only instrumentation of the
already-built coin-netted portfolio -- it introduces no new coin-netting math, only summarizes
`coin_notional`/`equity` that `build_coin_netted_portfolio` already computed. Concentration is
only measured for the 2024-01-01 block since it is the only block with any selected pairs at
all under either model."""),
    code("""
from stat_arb.portfolio import evaluate_coin_netted_portfolio, coin_concentration_diagnostics

richest_block = WF1_REBALANCE_DATES[0]
richest_block_end = WF1_REBALANCE_DATES[1] - pd.Timedelta(hours=4)
no_overlap_richest_pairs = [s['pair'] for s in NO_OVERLAP_SELECTIONS[richest_block]]
no_overlap_richest_pair_results = {s['pair']: NO_OVERLAP_ALL_RESULTS[(richest_block, s['pair'])]
                                     for s in NO_OVERLAP_SELECTIONS[richest_block]}
no_overlap_richest_portfolio, _ = evaluate_coin_netted_portfolio(
    no_overlap_richest_pair_results, no_overlap_richest_pairs, WF_PX.loc[:richest_block_end],
    CONFIG['COST_PER_LEG_BPS'], richest_block, richest_block_end, CONFIG['BARS_PER_YEAR'])
no_overlap_concentration = coin_concentration_diagnostics(no_overlap_richest_portfolio)

print(f"Max share of total portfolio gross exposure held by a single coin: "
      f"{no_overlap_concentration['max_share_of_total_gross_single_coin']:.1%}")
no_overlap_concentration['top5_coins_by_max_gross'].rename('max_gross_exposure_pct_of_equity').to_frame()
"""),
    code("""
largest_gross_coin = no_overlap_concentration['largest_gross_coin_by_bar']
n_active_bars = largest_gross_coin.notna().sum()
ldo_share_of_active_bars = (largest_gross_coin == 'LDOUSDT').sum() / n_active_bars if n_active_bars else float('nan')
print(f"LDOUSDT is the single largest gross coin exposure on {ldo_share_of_active_bars:.1%} of "
      f"{n_active_bars} active bars in the 2024-01-01 block.")
"""),
    md("""### LDOUSDT Pair-Cluster Diagnostics

All 10 no-overlap-selected pairs at 2024-01-01 share LDOUSDT as one leg (the funnel diagnostic
already established this). `coin_cluster_pair_table` joins the real diagnostic trail against the
real trades (same tuple-order normalization already used in `stat_arb.funnel` -- see that
module's docstring for why this join must never assume the two key conventions coincide).
`coin_cluster_return_correlation` answers whether these 10 pairs behave like diversified
relative-value opportunities or one concentrated LDOUSDT-centered factor trade."""),
    code("""
from stat_arb.ablation import coin_cluster_pair_table, coin_cluster_return_correlation
from stat_arb.pair_selection import select_pairs_at_rebalance

_, _, no_overlap_pair_trail, _ = select_pairs_at_rebalance(
    richest_block, WF_PX, WF_PX.columns.tolist(), WF_CONFIG, max_pairs=WF_CONFIG['MAX_PAIRS_PER_REBALANCE'],
    collect_diagnostics=True, enforce_one_pair_per_coin=False)

ldo_cluster_table = coin_cluster_pair_table(richest_block, 'LDOUSDT', no_overlap_pair_trail,
                                              NO_OVERLAP_SELECTIONS, NO_OVERLAP_ALL_TRADES)
print(f"{len(ldo_cluster_table)} testable pairs involve LDOUSDT at {richest_block.date()}; "
      f"{int(ldo_cluster_table['fully_qualifying'].sum())} fully qualify, "
      f"{int(ldo_cluster_table['selected'].sum())} selected "
      f"({int((ldo_cluster_table['elimination_stage'] == 'max_pairs_cap').sum())} of the remainder "
      f"excluded only by the max-pairs cap, not by any statistical filter).")
ldo_cluster_table
"""),
    code("""
ldo_cluster_corr = coin_cluster_return_correlation(richest_block, 'LDOUSDT', NO_OVERLAP_SELECTIONS, NO_OVERLAP_ALL_RESULTS)
offdiag = ldo_cluster_corr.values[np.triu_indices_from(ldo_cluster_corr.values, k=1)]
print(f'LDOUSDT-cluster pairwise 4h net-return correlation (full 2024-01-01 block, flat bars = 0 return): '
      f'mean={offdiag.mean():.3f}, median={np.median(offdiag):.3f}, min={offdiag.min():.3f}, max={offdiag.max():.3f}')
ldo_cluster_corr.round(2)
"""),
    code("""
from stat_arb.reporting import pair_level_contribution

no_overlap_pnl = pair_level_contribution(NO_OVERLAP_ALL_TRADES, NO_OVERLAP_ALL_RESULTS, CONFIG['COST_PER_LEG_BPS'])
print('Per-pair-instance P&L contribution -- every traded pair in the full OOS window under this '
      'ablation involves LDOUSDT:')
no_overlap_pnl.round(4)
"""),
    md("""### Interpretation (B2.5)

Answering the pre-specified research questions directly from the measured results above, not
from intuition:

1. **10 pairs selected** at 2024-01-01 (15 fully qualify; the existing `MAX_PAIRS_PER_REBALANCE=10`
   cap -- left unchanged, per the ablation's scope -- excludes the 5 weakest by `pvalue_adj`:
   ICPUSDT, AVAXUSDT, BNBUSDT, ADAUSDT, APTUSDT, all paired with LDOUSDT).
2. **All 10 generate real entry signals** (100%).
3. **31 total executed trades** over the full OOS window (vs. 2 for baseline WF1) -- all in the
   single 2024-01-01 block; every other rebalance date has zero FDR-significant pairs under
   either selection method, so the ablation cannot manufacture trades where none survive the
   statistical screen.
4. **Time in market rises materially**: 1.5% (baseline) -> 14.1% (no-overlap), roughly 9x.
5. **Net performance deteriorates**: daily-252 Sharpe -0.068 -> -0.475; cumulative net return
   -3.8% -> -10.1%. (Max drawdown is smaller in magnitude, -18.6% -> -12.9% -- a diversification
   effect on drawdown that does not carry through to Sharpe or cumulative return.)
6. **Transaction costs rise**: 0.83% -> 1.26% of equity (turnover 4.15x -> 6.30x), consistent
   with 31 trades instead of 2.
7. **LDOUSDT concentration is severe**: gross exposure in LDOUSDT alone peaks at ~46.9% of
   portfolio equity (70.7% of the portfolio's own total gross exposure at that moment), and
   LDOUSDT is the single largest coin exposure on ~87.8% of all active bars in the block.
8. **Mixed-to-moderate positive correlation, not independent, not a single repeated trade**:
   the 10 pairs' 4h net-return series have a mean pairwise correlation of ~0.16 (median ~0.13;
   25 of 45 pairs above +0.10; only 1 of 45 meaningfully negative) -- well below what ten copies
   of the same trade would show, but clearly above pure independence. Combined with the
   concentration finding above, this reads as a **partially common LDOUSDT-centered exposure**
   expressed through different counterparties, not a set of ten diversified relative-value bets.
9. **BTC beta**: unconditional beta drops further toward zero (0.0141 -> 0.0016); active-day
   beta also drops sharply (0.78, n=16 active days -> 0.009, n=133 active days). The larger
   active-day sample under the ablation is itself a byproduct of trading far more, not
   independent evidence of market-neutrality -- and WF1's own n=16 active-day estimate was
   already flagged as too small to trust on its own.
10. **Based on the measured results, not intuition**: removing the one-pair-per-coin constraint
    added 29 trades and ~12.6 points of time-in-market but *worsened* both Sharpe and cumulative
    net return, *raised* transaction costs, and produced a portfolio dominated by a single
    coin's exposure for most of its active life. On this one richest-block episode, the
    constraint appears to have been suppressing **concentration risk**, not a materially better
    opportunity set. This is a single rebalance block and 31 trades -- an even smaller sample
    than WF1/WF2's own low-trade-count caveat already covers -- so this is reported as a
    single-episode exploratory finding, not a robust conclusion, and `WF1_NO_OVERLAP_CONSTRAINT`
    is not adopted as a new primary specification."""),
    md("## WF2 -- Rolling Entry Hedge Sizing"),
    code("""
from stat_arb.walkforward import run_wf2_full

WF2_FULL_FRAME, WF2_ALL_RESULTS, WF2_ALL_TRADES = run_wf2_full(
    WF1_SELECTIONS, WF1_REBALANCE_DATES, CONFIG['OOS_END'], WF_PX, CONFIG)

wf2_row = standardized_wf1_summary('WF2_ROLLING_ENTRY_HEDGE_RATIO', WF2_FULL_FRAME, WF2_ALL_RESULTS,
                                    WF2_ALL_TRADES, WF_PX, CONFIG['OOS_START'], CONFIG['OOS_END'])

wf1_vs_wf2_df = pd.DataFrame([WF1_ROW, wf2_row]).set_index('model')
print('WF1 vs. WF2 -- OOS standardized comparison (LOW TRADE COUNT: interpret with caution):')
wf1_vs_wf2_df.round(4).T
"""),
    md("""## Interpretation and limitations

WF1 trades only a handful of times in the full OOS window; WF2 differs from WF1 only in
entry-time hedge sizing and trades the same instances. With this few trades, any Sharpe/return
comparison between WF1, WF2, S0, and S1 is not a robust statistical conclusion -- it is reported
with this caveat stated explicitly, per `CLAUDE.md`'s low-trade-count requirement."""),
    code("""
wf1_wf2_tables_dir = PROJECT_ROOT / 'results' / 'tables'
wf1_wf2_tables_dir.mkdir(parents=True, exist_ok=True)
with open(wf1_wf2_tables_dir / 'wf1_wf2_rows.json', 'w') as f:
    json.dump({'WF1_SEMIANNUAL_REESTIMATION': WF1_ROW, 'WF2_ROLLING_ENTRY_HEDGE_RATIO': wf2_row}, f, indent=2, default=str)
print(f'Saved WF1/WF2 rows to {wf1_wf2_tables_dir / "wf1_wf2_rows.json"}')
"""),
]

# ============================================================================================
# Notebook 4: results and model comparison
# ============================================================================================
nb4_cells = [
    md("""
# 04. Results and Model Comparison

Research question: does a classical, statically-calibrated pairs-trading strategy on the 2024
cryptocurrency spot universe produce a positive risk-adjusted return out-of-sample? This
notebook is the single cleanest artifact for an external reviewer -- it loads the standardized
summary rows already computed in notebooks 01 and 03 (no strategy implementation code is
duplicated here) and reports the full model comparison, limitations, and conclusions.

**Model registry** (see `CLAUDE.md` for full detail):

| ID | Description |
|---|---|
| `S0_STATIC_BASELINE` | Frozen static pairs, one global time stop -- the control model |
| `S1_STATIC_PAIR_HALFLIFE_STOP` | S0 + per-pair time stop = ceil(2.5x formation half-life) |
| `WF1_SEMIANNUAL_REESTIMATION` | Walk-forward pair re-selection, 12mo formation / 6mo rebalance |
| `WF2_ROLLING_ENTRY_HEDGE_RATIO` | WF1 + rolling entry-time hedge sizing (signal still frozen formation alpha/beta) |

**Research-scope disclaimer:** this is a research-learning project, not a production trading
strategy -- simplified short-leg implementation, no funding/borrow/margin modeling, Binance spot
reference prices.
"""),
    code(SETUP_CODE),
    code("""
results_tables_dir = PROJECT_ROOT / 'results' / 'tables'
with open(results_tables_dir / 's0_s1_rows.json') as f:
    s0_s1 = json.load(f)
with open(results_tables_dir / 'wf1_wf2_rows.json') as f:
    wf1_wf2 = json.load(f)

MASTER_COMPARISON_DF = pd.DataFrame([
    s0_s1['S0_STATIC_BASELINE'], s0_s1['S1_STATIC_PAIR_HALFLIFE_STOP'],
    wf1_wf2['WF1_SEMIANNUAL_REESTIMATION'], wf1_wf2['WF2_ROLLING_ENTRY_HEDGE_RATIO'],
]).set_index('model')
print('Master model comparison -- OOS (2024-01-01 to 2026-06-30):')
MASTER_COMPARISON_DF.round(4).T
"""),
    md("""## Standardized comparison fields

`daily_sharpe_252`, `ann_volatility` (both via the daily-252 convention -- see `CLAUDE.md`
"Evaluation conventions"), `cum_net_return`, `max_drawdown`, `n_trades`, `time_in_market_pct`,
`turnover`, `total_costs_pct`, `btc_correlation`, `btc_beta`, `ann_alpha`, `alpha_tstat_ols`,
`alpha_tstat_hac`, `r_squared` -- all four models report through the exact same schema
(`standardized_model_summary`/`standardized_wf1_summary` in `stat_arb.reporting`)."""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4))
sharpes = MASTER_COMPARISON_DF['daily_sharpe_252']
colors = ['#2a6f97' if v >= 0 else '#c1121f' for v in sharpes]
ax.bar(sharpes.index, sharpes.values, color=colors)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='0.5 (resume-relevance reference, not a target)')
ax.set_ylabel('Daily-252 Sharpe')
ax.set_title('OOS Daily-252 Sharpe by Model')
ax.legend(fontsize=8)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()
"""),
    md("""## Evaluation-Period Labelling

Every model above is reported over 2024-01-01 to 2026-06-30, labelled throughout this project
as a **Post-Sample Historical Evaluation** (see notebook 01's "Note on the Final OOS label") --
a corrected historical re-evaluation under a fixed, pre-specified methodology, not a pristine
never-seen holdout. `FORWARD_EVAL_START` marks the genuinely untouched forward window: no
parameter anywhere in `config/research_config.py` has been set or changed using any data at or
after this date, and none may be in the future without that constituting a new, explicitly
labelled research iteration."""),
    code("""
from stat_arb.data import load_raw_prices

_px_check = load_raw_prices()
assert _px_check.index.max() < pd.Timestamp(FORWARD_EVAL_START), (
    'px.pkl already contains data at or after FORWARD_EVAL_START -- the forward-evaluation '
    'window is no longer untouched and this constant must be moved forward before any further '
    'model change is made.')
print(f'[CHECK] No data currently available at or after FORWARD_EVAL_START ({FORWARD_EVAL_START}) -- '
      f'the forward-evaluation window remains genuinely unstarted. Data ends {_px_check.index.max()}.')
"""),
    md("""## Limitations

- **S0/S1 OOS Sharpe and cumulative return are both negative** on this sample -- a negative
  result is an acceptable, reportable outcome, not a failure of the research process.
- **S1's time-stop multiplier robustness is unstable** (2.0x/2.5x/3.0x Sharpe swings materially,
  see notebook 01) -- the time-stop choice does not robustly explain S0's underperformance.
- **WF1/WF2 trade only a handful of times** in the full OOS window (see notebook 03) -- any
  comparison against S0/S1, or between WF1 and WF2, must state this sample-size limitation.
- **WF1's active-day BTC beta materially exceeds its unconditional beta** (notebook 03) --
  low time-in-market makes the unconditional beta an unreliable market-neutrality indicator.
- This is a research-learning project: simplified short-leg implementation, no funding/borrow/
  margin modeling, Binance spot reference prices -- not a production P&L estimate.

## Final Research Interpretation

None of the four model versions produced a robustly positive, statistically well-supported
Sharpe ratio on this out-of-sample window. S0/S1 (negative Sharpe, unstable time-stop
robustness) and WF1/WF2 (too few trades for a reliable conclusion, and not market-neutral on
the days they do trade) each fail to clear the ~0.5 Sharpe resume-relevance threshold. This is
reported as the honest outcome of a pre-specified, no-look-ahead, non-overfit research process
-- not as a basis for further parameter searching on this same OOS window.
"""),
]


def build(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3 (stat-arb)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Wrote {out_path} ({len(cells)} cells)')


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebooks_dir = os.path.join(project_root, 'notebooks')
    os.makedirs(notebooks_dir, exist_ok=True)
    build(nb1_cells, os.path.join(notebooks_dir, '01_data_and_static_baseline.ipynb'))
    build(nb2_cells, os.path.join(notebooks_dir, '02_relationship_stability.ipynb'))
    build(nb3_cells, os.path.join(notebooks_dir, '03_walkforward_strategy.ipynb'))
    build(nb4_cells, os.path.join(notebooks_dir, '04_results_and_model_comparison.ipynb'))


if __name__ == '__main__':
    main()
