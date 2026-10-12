# Phase B2 Validation Report — Selection Funnel Diagnostic

Scope: diagnostic only, per the user's B2 instructions. No threshold, ranking rule, entry rule,
portfolio rule, or cache-fingerprint methodology was changed.

## 1. What was built

- `src/stat_arb/pair_selection.py`: `screen_pairs_at_date` and `select_pairs_at_rebalance`
  extended with an additive, opt-in `collect_diagnostics: bool = False` parameter. Default
  behavior/return type is byte-for-byte unchanged (verified — see Section 3). When `True`, both
  functions return additional stage counts and a pair-level trail, captured by instrumenting the
  *same* filtering loop that produces the real result — never a parallel reimplementation.
- `src/stat_arb/funnel.py` (new): `build_selection_funnel()` merges the above with the **real**
  trades already produced by `run_wf1_full`/`run_wf2_full` (never re-simulated) into three
  tables: `funnel_df` (one row per rebalance date), `pair_level_df` (one row per testable pair),
  `conflicts_df` (one row per pair rejected by the one-pair-per-coin constraint).
- `scripts/run_selection_funnel.py` (new): standalone precompute mirroring
  `run_stability_screen.py`'s convention (expensive — re-runs the point-in-time screen with
  diagnostics enabled — so it is not a notebook cell). Caches to
  `data/cache/selection_funnel_<fingerprint>.pkl` and writes
  `results/tables/selection_funnel.json`, `selection_funnel_pair_level.json`,
  `selection_funnel_overlap_conflicts.json`.
- `notebooks/03_walkforward_strategy.ipynb`: new "Selection Funnel Diagnostic" section (7 cells)
  loading the precomputed tables and reporting the aggregate funnel, the one-pair-per-coin
  conflict table, the pair-level bottleneck table for 2024-01-01, and an interpretation.
- `tests/test_selection_funnel.py` (new, 7 tests): default-behavior-unchanged regression guard,
  monotonic sequential-survivor-count identities, pair-trail elimination-stage consistency,
  overlap-diagnostic winner correctness, max-pairs-cap diagnostics, a targeted tuple-ordering
  regression test (see Section 2), and trade-count reconciliation against a fabricated
  `all_trades` dict.

## 2. Bugs found and fixed during B2

1. **pandas tuple-index ambiguity**: `.loc[tuple_key, column]` on a DataFrame indexed by tuple
   values (not a MultiIndex) raised an internal pandas `AssertionError` on read (though not on
   write, which is why it wasn't caught immediately). Fixed by switching every scalar
   `pair_trail` access to `.at[]`, which has no such ambiguity.
2. **Pair-tuple-ordering mismatch** (a real instance of the exact unit-mixing risk the user's
   instructions warned about): `screen_pairs_at_date`'s pair-level trail is keyed by the
   alphabetically-sorted unordered pair (inherited from `run_engle_granger_tests`'s
   `tuple(sorted([c1, c2]))`), while `select_pairs_at_rebalance`'s `selected[i]['pair']` is
   `(x, y)` in the **winning regression direction** (matching the real pipeline's
   `WF1_ALL_TRADES` instance keys). These are different tuples whenever the winning direction
   isn't already alphabetical — e.g. `('OMUSDT', 'LDOUSDT')` vs. `('LDOUSDT', 'OMUSDT')`. This
   silently caused the winning pair's `selected`/`generated_entry_signal` flags to read as
   `False`/`null` in the pair-level table, even though the aggregate `funnel_df` counts (built
   from a separate, correct counting path) were unaffected throughout. Fixed by normalizing to
   the sorted-pair convention at every `pair_trail` lookup, with an explicit code comment
   documenting the two conventions so this cannot silently regress. A targeted regression test
   (`test_selected_pair_trail_row_matches_regardless_of_tuple_order`) now guards this.

Both bugs were caught by internal consistency checks (an aggregate count implying a selected
pair must exist, contradicted by the pair-level table showing none) before this report was
written — not discovered by the user.

## 3. Validation results

| Check | Result |
|---|---|
| `pytest` | **68/68 passed** (61 pre-B2 + 7 new selection-funnel tests) |
| Default (`collect_diagnostics=False`) behavior unchanged | Confirmed by `test_screen_pairs_at_date_default_unchanged_shape` (`pd.testing.assert_frame_equal` against the pre-B2 return) |
| All 4 notebooks, fresh kernel | **All OK** (notebook 03 grew from 19 to 26 cells) |
| `results/tables/*.json` (actual notebook output) vs Golden Master | **Exact match**, S0/S1/WF1/WF2 |
| `validate_against_golden_master.py` | **ALL MODELS RECONCILE EXACTLY** |
| `compare_trade_logs.py` | **ALL TRADE LOGS MATCH** — S0 29/29, S1 29/29, WF1 2/2, WF2 2/2 |
| Cache fingerprints | `pair_stability_523179f2a1.pkl`, `wf1_selections_6d2bb2d492.pkl`, `wf1_selections_3m_d99bee3e45.pkl`, `eg_results_b62b6e0656.pkl` all **unmodified** (timestamps predate B2) — only a **new** `selection_funnel_1e9d141c99.pkl` cache was added, its own namespace |

**No existing model output or trade history changed. B2 added diagnostics only.**

## 4. Findings — where does the opportunity set disappear?

At **2024-01-01** (WF1's richest rebalance date):

```
33 eligible assets -> 528 candidate pairs -> 528 testable -> 1,056 directed EG tests
  -> 15 FDR-significant pairs (after direction resolution)
  -> 15 pass positive-Beta -> 15 pass Half-Life -> 15 pass crossings
  -> 15 fully-qualifying pairs
  -> 1 selected (14 removed by one-pair-per-coin overlap, 0 by the max-pairs cap)
  -> 1 selected pair generates a real entry signal -> 2 executed trades
```

**Zero pairs are eliminated by the Beta, Half-Life, or crossing filters at this date** — every
one of the 15 FDR-significant pairs is fully statistically qualifying. The entire reduction from
15 to 1 happens at the **one-pair-per-coin portfolio constraint**, and it is driven by
**concentration on a single coin**: all 14 rejected pairs involve **LDOUSDT** paired with a
different counterpart (DOTUSDT, UNIUSDT, FTMUSDT, ATOMUSDT, VETUSDT, DOGEUSDT, FILUSDT, SHIBUSDT,
ICPUSDT, AVAXUSDT, BNBUSDT, ADAUSDT, APTUSDT, NEARUSDT), and the lowest-adjusted-p-value pair
(OMUSDT/LDOUSDT, `pvalue_adj`=0.00036) claims that coin first under the existing, unchanged
ranking rule.

At the **other four WF1 rebalance dates** (2024-07-01, 2025-01-01, 2025-07-01, 2026-01-01),
**zero pairs are even FDR-significant** — a completely different, purely statistical bottleneck
with no portfolio-constraint or execution-stage involvement at all.

Execution/state-machine constraints played no role at 2024-01-01 either: the one selected pair
generated a real entry signal and both of its signal-level trades were executed with no drop
between "entry signal" and "executed trade" — structurally identical counts in this
fixed-quantity, next-bar-execution engine (documented, not assumed, in `stat_arb.funnel`).

Full detail: [results/tables/selection_funnel.json](results/tables/selection_funnel.json),
[selection_funnel_pair_level.json](results/tables/selection_funnel_pair_level.json),
[selection_funnel_overlap_conflicts.json](results/tables/selection_funnel_overlap_conflicts.json).
