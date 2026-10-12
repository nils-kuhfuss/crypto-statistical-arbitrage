# Refactor Validation Report

Validates the modular refactor (`src/stat_arb/` + `notebooks/`) against the Golden Master
captured from the original, unmodified notebook (`legacy/Stat_Arb_cont_res_1.ipynb`), per
`REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md` Section 12.

## 1. Golden Master capture

`scripts/capture_golden_master.py` executed `legacy/Stat_Arb_cont_res_1.ipynb` end-to-end from
a fresh kernel (true Restart & Run All, not a re-run of an already-populated kernel) and
captured every field required by Section 2 into `results/golden_master.json` and
`results/golden_master_trades/*.json`. This required first regenerating the pair-stability
cache (`pair_stability_523179f2a1.pkl`), which was missing from the project directory along
with its documented generator (`build/run_stability_screen_v2.py`) — reconstructed verbatim as
`scripts/run_stability_screen.py` from the notebook's own `screen_pairs_at_date` cell. Its
recomputed cache matched the notebook's stored expected fingerprint (`523179f2a1`) and row
counts (26 rows across 5 rebalance dates: 1+3+3+4+15) exactly before any refactor code was
written — an early, independent confirmation that the reconstruction was faithful.

Captured headline figures (all confirmed against `CLAUDE.md`'s pre-existing documented values):

| Model | Daily-252 Sharpe | Cum. net return | Max drawdown |
|---|---:|---:|---:|
| S0_STATIC_BASELINE | -0.11552399 | -1.6482% | -11.1707% |
| S1_STATIC_PAIR_HALFLIFE_STOP | -0.12486232 | -1.8760% | -10.5320% |
| WF1_SEMIANNUAL_REESTIMATION | -0.06786708 | -3.7954% | -18.5844% |
| WF2_ROLLING_ENTRY_HEDGE_RATIO | -0.01850725 | -1.8789% | -15.6259% |

## 2. Numerical parity: fast direct-Python reconciliation

`scripts/validate_against_golden_master.py` runs the full S0/S1/WF1/WF2 pipeline via direct
`src/stat_arb` function calls (no notebook/kernel overhead) and compares every field of every
model's summary row, plus each model's final equity value, against `results/golden_master.json`
with a tolerance of `max(1e-6, |golden_value| * 1e-6)`.

**Result: ALL FOUR MODELS RECONCILE EXACTLY** — every field of every summary row matches, and
every final-equity value matches to within floating-point tolerance:

- `S0_STATIC_BASELINE`: ALL FIELDS MATCH, final_equity 0.98351845 == 0.98351845
- `S1_STATIC_PAIR_HALFLIFE_STOP`: ALL FIELDS MATCH, final_equity match confirmed
- `WF1_SEMIANNUAL_REESTIMATION`: ALL FIELDS MATCH, final_equity match confirmed
- `WF2_ROLLING_ENTRY_HEDGE_RATIO`: ALL FIELDS MATCH, final_equity match confirmed

`FROZEN_PAIRS` (the 12 pairs selected by `stat_arb.pair_selection.run_static_pair_screen`) also
match the Golden Master's captured `selected_pairs` exactly (set equality).

An independent, narrower check (`stat_arb.pair_selection.screen_pairs_at_date` called directly
for the 2024-01-01 rebalance date) reproduces the cached `pair_stability_df` rows for that date
exactly (15/15 rows, all columns matching to 1e-9) — confirming the walk-forward point-in-time
screen was ported correctly independent of the S0/S1/WF1/WF2 aggregate comparison above.

`stat_arb.cache.wf_cache_fingerprint`, called from the new modular code with
`config.research_config.WF_CONFIG`, reproduces the fingerprint `523179f2a1` exactly — confirming
`stat_arb.data.build_walkforward_universe`, `stat_arb.walkforward.generate_rebalance_dates`, and
the fingerprint function itself are all byte-identical to the original notebook's behavior.

## 3. Bugs found and fixed during reconciliation

Two real bugs were caught by the fast reconciliation script before it passed (both were config
key-path mismatches, not logic errors):

1. `stat_arb.pair_selection.run_static_pair_screen` originally read `config['MAX_HALF_LIFE_BARS']`
   / `config['MIN_CROSSINGS']` from the passed `CONFIG` dict, but those are bare top-level
   constants in `config/research_config.py` (matching the original notebook's own bare-global
   convention), not keys inside `CONFIG`. Fixed by making them explicit function parameters
   (`max_half_life_bars`, `min_crossings`), consistent with how `screen_pairs_at_date` already
   receives its equivalent thresholds.
2. The Golden Master reconciliation script and the notebook-01 builder both referenced
   `CONFIG['S1_TIME_STOP_MULTIPLIER']` / `CONFIG['S1_TIME_STOP_MULTIPLIER_ROBUSTNESS']`, which
   are likewise bare top-level constants, not `CONFIG` keys. Fixed by importing them directly.

A systematic grep of every `config['KEY']`/`CONFIG['KEY']` reference across `src/stat_arb/*.py`
against the actual contents of `CONFIG`/`WF_CONFIG` in `config/research_config.py` found no
further instances of this bug class after the fix.

## 4. Test suite

`pytest` (61 tests, `tests/test_*.py`): **61 passed**, covering execution-engine regression
(state machine, next-bar execution, fixed-quantity ledger, trade-level P&L reconciliation —
ported from the original notebook's synthetic-data cells), statistics primitives, point-in-time
eligibility, pair selection (one-pair-per-coin, positive-Beta, dedup), portfolio reconciliation
(flat-after-liquidation, sleeve allocation, repeated-instance keying), daily-return
reconciliation (flat days kept, 4h/daily equity parity), and cache-fingerprint methodology
sensitivity (every `WF_CONFIG` parameter change produces a different fingerprint).

## 5. Notebook execution status

`scripts/run_all_notebooks.py` performs a genuine Restart & Run All (fresh kernel per notebook,
via `nbclient`) on all four notebooks in order, verifying no hidden inter-notebook state
dependency beyond the explicit `results/tables/*.json` hand-off (notebook 01 -> 03 -> 04).

**Result: PASSED.** All four notebooks executed with no cell errors, from a fresh kernel each:

```
OK: 01_data_and_static_baseline.ipynb in 294.2s
OK: 02_relationship_stability.ipynb in 7.2s
OK: 03_walkforward_strategy.ipynb in 8.9s
OK: 04_results_and_model_comparison.ipynb in 3.4s
ALL_NOTEBOOKS_OK
```

The S0/S1 rows notebook 01 actually wrote to `results/tables/s0_s1_rows.json`, and the WF1/WF2
rows notebook 03 actually wrote to `results/tables/wf1_wf2_rows.json` (both real notebook
outputs, not the standalone reconciliation script from Section 2), were compared field-by-field
against `results/golden_master.json` — **EXACT MATCH, zero mismatched fields, for all four
models.** This confirms parity end-to-end through the actual notebook execution path, not just
the underlying `src/stat_arb` functions in isolation.

## 5b. Trade-log comparison

`scripts/compare_trade_logs.py` independently re-runs the full S0/S1/WF1/WF2 pipeline and
compares every individual trade's direction, entry timestamp, exit timestamp, exit reason, and
duration against `results/golden_master_trades/*.json`, trade-by-trade (not just the summary
statistics derived from them in Section 2).

**Result: ALL TRADE LOGS MATCH.**

- S0: 29/29 trades match exactly
- S1: 29/29 trades match exactly
- WF1: 2/2 trades match exactly
- WF2: 2/2 trades match exactly

## 6. Cache fingerprint verification

Confirmed independently in Section 2 above (fingerprint match) and covered by
`tests/test_cache_fingerprint.py`'s parametrized sweep over every `WF_CONFIG` field.

## 7. Known, pre-existing limitations ported as-is (not silently "fixed")

- **WF1/WF2 sleeve-allocation assertion is tautological**: `n_pairs * (1/n_pairs) <= 1` is
  algebraically true for any `n_pairs > 0` and can never actually catch an allocation
  violation. This was a pre-existing property of *both* `run_wf1_full` and `run_wf2_full` in
  the original notebook (spelled slightly differently in each, but equally tautological) — it
  is ported as-is per this task's "pure software refactor, no logic changes" scope, not
  silently corrected. See `stat_arb.walkforward`'s module docstring.
- **`PairPosition` class** was test-only scaffolding in the original notebook (used by one
  regression cell, not the real execution path) — ported to `tests/test_execution.py` rather
  than `stat_arb/execution.py`.
- **Model-selection derivation (S vs RZ vs RB, rolling-window search) is not re-executed** in
  the refactored `notebooks/01_data_and_static_baseline.ipynb` — it is a one-time historical
  decision, preserved in full in `legacy/Stat_Arb_cont_res_1.ipynb` Sections 12-13, whose
  already-decided outcome (Model S, frozen parameters) notebook 01 takes as its starting point.
  This does not affect numerical parity of any of the four reported models (verified in
  Section 2), since S0's OOS backtest depends only on `FROZEN_MODEL`'s parameters and
  `FROZEN_PAIRS`, both fully reproduced.

## 8. Consolidations performed (structure only, verified against Golden Master)

Per `REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md` Section 8's explicit prohibition on maintaining two
independent implementations of the same logic:

- `stat_arb.statistics.run_engle_granger_tests`/`apply_bh_fdr` replace what were two
  independent copies of the EG-test + BH-FDR loop (S0's Section 6 orchestration and
  `screen_pairs_at_date`).
- `stat_arb.walkforward._run_block`/`_run_full` replace what were four near-identical
  functions (`run_wf1_block`/`run_wf1_full`/`run_wf2_block`/`run_wf2_full`), parameterized by
  `use_rolling_beta`. Thin, identically-named wrappers preserve every existing call site.

Both consolidations are verified byte-for-byte correct by Section 2's reconciliation (which
exercises every one of these code paths for all four models) and were not the source of either
bug in Section 3.

## Conclusion

**The refactor is complete and numerically validated.** All four models (S0, S1, WF1, WF2)
reconcile exactly against the Golden Master, confirmed at three independent levels: (1) direct
`src/stat_arb` function calls (Section 2), (2) the actual four notebooks executed end-to-end
from fresh kernels (Section 5), and (3) the walk-forward cache fingerprint and point-in-time
screen output (Section 2). The full test suite (61 tests) passes. No unresolved numerical
discrepancy exists between the original notebook and the refactored codebase. Two real
config-key-path bugs were found and fixed during reconciliation (Section 3); no other
correctness issues surfaced. Pre-existing weaknesses in the original notebook (the tautological
WF1/WF2 sleeve-allocation assertion) were preserved as-is, per this task's pure-software-refactor
scope, rather than silently corrected (Section 7).
