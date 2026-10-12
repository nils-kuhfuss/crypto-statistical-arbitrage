"""Central research configuration.

Every value here is copied verbatim from `legacy/Stat_Arb_cont_res_1.ipynb` -- this module
introduces no new parameter and changes no existing one (per REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md
Section 4). Where the notebook computed a derived value (e.g. `ROLLING_WINDOW_BARS`,
`WF_CONFIG['MIN_OBS_FOR_TEST']`), the same derivation is reproduced here rather than
hand-typing the resulting number, so the relationship stays visible and self-consistent.
"""

# --- S0/S1 static-model configuration (notebook Section 2, `CONFIG`) -------------------------
CONFIG = {
    # Sample boundaries (Section 4)
    'FORMATION_START': '2020-01-01',
    'FORMATION_END':   '2022-12-31',
    'VALIDATION_START': '2023-01-01',
    'VALIDATION_END':   '2023-12-31',
    'OOS_START': '2024-01-01',
    'OOS_END':   '2026-06-30',

    # Universe filtering
    'MANUAL_EXCLUDE': ['USDCUSDT', 'FDUSDUSDT', 'XMRUSDT'],  # stablecoins / not economically tradable pairs
    'MIN_HISTORY_CUTOFF': '2021-06-30',  # coins first observed after this date lack sufficient formation history

    # Bar frequency
    'BARS_PER_DAY': 6,         # 4h bars
    'BARS_PER_YEAR': 365 * 6,  # annualization factor

    # Cointegration screening
    'EG_ALPHA': 0.05,        # Engle-Granger significance level
    'FDR_METHOD': 'fdr_bh',  # Benjamini-Hochberg multiple-testing correction
    'MIN_OBS_FOR_TEST': 100, # minimum overlapping observations to run a cointegration test (S0's own formation-window-length threshold; distinct from WF_CONFIG['MIN_OBS_FOR_TEST'] below)

    # Static control model (reproduces a previously reported ICP/FIL result; see Section 3)
    'STATIC_BASELINE_PAIR': ('FILUSDT', 'ICPUSDT'),  # (X, Y), Y = alpha + beta*X + eps
    'STATIC_ALPHA': 0.1236,
    'STATIC_BETA': 0.9335,
    'BASELINE_EXPECTED_TRADES': 5,
    'BASELINE_EXPECTED_DURATIONS': [21, 86, 91, 36, 2],
    'BASELINE_EXPECTED_AVG_NET_PNL': 0.1026,  # prior cash/quantity report, approximate
    'BASELINE_PNL_TOLERANCE': 0.01,           # absolute return tolerance for the regression check

    # Trading rules (shared across the static and dynamic models unless an experiment
    # explicitly overrides one in isolation -- see Section 9)
    'ENTRY_Z': 1.5,
    'EXIT_Z': 0.5,
    'HARD_STOP_Z': 4.0,
    'TIME_STOP_BARS': 156,
    'COST_PER_LEG_BPS': 20,  # applied per unit of turnover, per leg

    # Dynamic-model parameters
    'ROLLING_WINDOW_DAYS': 90,  # converted to bars via BARS_PER_DAY

    # Tradability handling
    'FFILL_LIMIT_BARS': 2,  # bridges gaps of at most 2 bars (~8h); no unbounded forward-fill

    # Newey-West/HAC robustness lag length for the benchmark-regression alpha t-stat
    # (statsmodels' cov_kwds={'maxlags': ...}). Bare literal in the original notebook's
    # benchmark_regression_stats; centralized here (Phase B1).
    'HAC_MAXLAGS': 5,
}
CONFIG['ROLLING_WINDOW_BARS'] = CONFIG['ROLLING_WINDOW_DAYS'] * CONFIG['BARS_PER_DAY']

# --- Section 6 pair-quality filters (S0's cointegration screen) ------------------------------
MAX_HALF_LIFE_BARS = 300     # ~50 days on 4h bars -- generous upper bound, tune only via 2023 validation reasoning, not OOS
MIN_CROSSINGS = 15           # require enough mean-reversion cycles in-sample to trust the estimate

# --- Section 13 model-selection margin --------------------------------------------------------
SELECTION_GO_MARGIN = 0.10  # a candidate must beat every simpler model it replaces by more than this

# --- Section 13 rolling-window robustness grid ------------------------------------------------
WINDOW_CANDIDATES_DAYS = [90, 180, 365]

# --- S1 pair-specific half-life time stop ------------------------------------------------------
S1_TIME_STOP_MULTIPLIER = 2.5  # primary specification -- see Phase 9 for 2.0x/3.0x robustness only
S1_TIME_STOP_MULTIPLIER_ROBUSTNESS = [2.0, 2.5, 3.0]  # pre-specified sensitivity grid, S1 and WF1 both use this

# --- Section 26 forward-evaluation window label ------------------------------------------------
FORWARD_EVAL_START = '2026-07-01'

# --- Walk-forward configuration (notebook Section 18.5, `WF_CONFIG`) --------------------------
WF_CONFIG = {
    # Universe: only true stablecoins are manually excluded (feedback #8). Coverage +
    # active-at-rebalance rules (below) handle late listings and delistings point-in-time,
    # so no other coin needs to be hardcoded out.
    'MANUAL_EXCLUDE': ['USDCUSDT', 'FDUSDUSDT'],

    # True 12-month formation coverage (feedback #2). 12 months * 365.25/12 * 6 bars/day.
    'FORMATION_MONTHS': 12,
    'BARS_PER_DAY': CONFIG['BARS_PER_DAY'],
    'MIN_COVERAGE_FRACTION': 0.90,

    # Same statistical/pair-quality rules as S0's own screen (feedback #4), applied at every
    # rebalance instead of once. MAX_HALF_LIFE_BARS/MIN_CROSSINGS are copied verbatim from
    # Section 6 so both models use identical thresholds.
    'EG_ALPHA': CONFIG['EG_ALPHA'],
    'FDR_METHOD': CONFIG['FDR_METHOD'],
    'MAX_HALF_LIFE_BARS': MAX_HALF_LIFE_BARS,
    'MIN_CROSSINGS': MIN_CROSSINGS,
    'REQUIRE_POSITIVE_BETA': True,   # feedback #6 -- primary specification

    # Rebalance schedule (unchanged from the primary WF1 specification)
    'REBALANCE_MONTHS_PRIMARY': 6,
    'REBALANCE_MONTHS_ROBUSTNESS': 3,
    'TIME_STOP_MULTIPLIER_PRIMARY': 2.5,

    # Point-in-time active-at-rebalance tolerance (bars strictly before rebalance_date within
    # which a valid price still counts as "still trading"). Bare default in the original
    # notebook's check_active_at_rebalance(..., lookback_bars=3); centralized here (Phase B1)
    # so it is a visible, discoverable methodology parameter rather than a hidden function default.
    'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,

    # Maximum pairs selected per rebalance (select_pairs_at_rebalance's one-pair-per-coin cap).
    # Bare default (max_pairs=10) at every original call site; centralized here (Phase B1).
    'MAX_PAIRS_PER_REBALANCE': 10,

    # WF2's rolling entry-hedge-sizing window (compute_rolling_beta's window_bars). The original
    # notebook hardcoded this as an independently-specified literal (540 bars = 90 days) inside
    # compute_rolling_beta's own default argument, not derived from CONFIG['ROLLING_WINDOW_BARS']
    # (which numerically also happens to be 90 days, but is S0/S1's own model-selection rolling
    # window -- a conceptually distinct parameter). Kept as its own named constant here (Phase
    # B1) rather than silently coupling the two, to avoid implying a shared derivation that the
    # original research process never specified.
    'WF2_ROLLING_BETA_WINDOW_BARS': 540,
}
_expected_bars_per_year = 365.25 * WF_CONFIG['BARS_PER_DAY']
WF_CONFIG['FORMATION_WINDOW_BARS_NOMINAL'] = int(round(_expected_bars_per_year * (WF_CONFIG['FORMATION_MONTHS'] / 12)))
WF_CONFIG['MIN_OBS_FOR_TEST'] = int(round(WF_CONFIG['FORMATION_WINDOW_BARS_NOMINAL'] * WF_CONFIG['MIN_COVERAGE_FRACTION']))

# --- Model registry IDs (see CLAUDE.md "Model registry" table) --------------------------------
MODEL_IDS = {
    'S0': 'S0_STATIC_BASELINE',
    'S1': 'S1_STATIC_PAIR_HALFLIFE_STOP',
    'WF1': 'WF1_SEMIANNUAL_REESTIMATION',
    'WF2': 'WF2_ROLLING_ENTRY_HEDGE_RATIO',
    # Phase B2.5 exploratory ablation (docs/... B2_5_NO_OVERLAP_ABLATION_CLAUDE.md): NOT a
    # primary model, never promoted into the S0/S1/WF1/WF2 master comparison without an
    # explicit later decision. Removes only WF1's one-pair-per-coin selection constraint.
    'WF1_NO_OVERLAP': 'WF1_NO_OVERLAP_CONSTRAINT',
}

# --- Data file locations -----------------------------------------------------------------------
# Centralized here (docs/project_management/CLAUDE_FOLDER_CLEANUP_AND_NEXT_RESEARCH.md Section 3/11):
# every path any project code needs is built from PROJECT_ROOT here, once, rather than each
# caller hand-building a relative "../..." path.
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_RAW_DIR = _os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_CACHE_DIR = _os.path.join(PROJECT_ROOT, 'data', 'cache')
PX_PATH = _os.path.join(DATA_RAW_DIR, 'px.pkl')
RET_PATH = _os.path.join(DATA_RAW_DIR, 'ret.pkl')
