# Phase B2.5 Validation Report — One-Pair-Per-Coin Constraint Ablation

Scope: an **exploratory Post-Sample Historical ablation**, not a new primary model. Exactly one
assumption changes relative to `WF1_SEMIANNUAL_REESTIMATION`: the one-pair-per-coin selection
constraint. Every other rule (ranking, `MAX_PAIRS_PER_REBALANCE` cap, formation window,
EG/FDR/Beta/Half-Life/crossing filters, execution engine, equal-sleeve portfolio construction,
cost model, benchmark regression) is untouched and shared verbatim with WF1.

## 1. Exact implementation change

`select_pairs_at_rebalance` (`src/stat_arb/pair_selection.py`) gains one new parameter,
`enforce_one_pair_per_coin: bool = True`. It gates exactly two things in the existing greedy
accept/reject loop, nothing else:

1. The coin-overlap rejection check (`if enforce_one_pair_per_coin and (x_name in used_coins or
   y_name in used_coins): ... continue`) — when `False`, a fully-qualifying pair is never
   rejected merely because one of its coins already appears in another selected pair.
2. The post-loop one-pair-per-coin assertion, which only makes sense when the constraint is
   enforced.

The ranking order (ascending `pvalue_adj`), the `max_pairs` cap and its `break`, and every
statistical filter upstream in `screen_pairs_at_date` are byte-for-byte unchanged. Default
behavior (`enforce_one_pair_per_coin` absent or `True`) is unaffected — proven by
`test_default_enforces_one_pair_per_coin_unchanged`.

New code, all additive:
- `src/stat_arb/ablation.py` (new module): `run_no_overlap_selection_pipeline` (same
  cache-then-compute convention as `walkforward.run_wf1_selection_pipeline`, just with the new
  flag), `coin_cluster_pair_table`, `coin_cluster_return_correlation`.
- `src/stat_arb/portfolio.py`: `coin_concentration_diagnostics` — read-only summary of an
  already-built `build_coin_netted_portfolio` result (`coin_notional`/`equity`). No new
  coin-netting math; introduced because this is the first model variant where two selected
  pairs can legitimately share a coin.
- `config/research_config.py`: `MODEL_IDS['WF1_NO_OVERLAP'] = 'WF1_NO_OVERLAP_CONSTRAINT'`.
- `scripts/run_no_overlap_ablation.py` (new, mirrors `scripts/run_selection_funnel.py`'s
  standalone-precompute convention): computes and caches the no-overlap selections
  (`data/cache/wf1_no_overlap_selections_<fingerprint>.pkl`, its own fingerprint namespace via
  `wf_cache_fingerprint(..., extra=...,no_overlap_constraint)` — never touches the plain
  `wf1_selections` cache or fingerprint), backtests via the exact same `run_wf1_full` WF1 uses,
  and writes `results/tables/wf1_no_overlap_{summary,selected_pairs,coin_exposure,
  pair_diagnostics}.json`.
- `notebooks/03_walkforward_strategy.ipynb`: new "B2.5 Ablation" section (14 cells) placed after
  the WF1 rebalance-frequency robustness section and before WF2, recomputing everything live
  from the cache (fast — the only genuinely expensive step, the 5-rebalance-date screen, is
  cached) rather than loading the script's JSON, so there is exactly one implementation of the
  selection/backtest/reporting logic, not two.
- `tests/test_no_overlap_ablation.py` (new, 7 tests, all passing): default-unchanged regression
  guard, overlap-retention with a shared-coin synthetic fixture, ranking-order preservation,
  max-pairs-cap preservation, a **meaningful structural reconciliation** of multi-pair coin
  netting (replacing reliance on the pre-existing tautological `n * (1/n) <= 1` assertion —
  see `CLAUDE.md`'s note on that assertion), a `coin_concentration_diagnostics` reconciliation
  test, and a trade-count reconciliation test.

## 2. Proof that baseline models did not change

| Check | Result |
|---|---|
| `pytest` | **75/75 passed** (68 pre-B2.5 + 7 new ablation tests) |
| `pyflakes` on all new/modified files | clean |
| All 4 notebooks, fresh kernel (Restart & Run All) | **All OK** (notebook 03 grew from 26 to 40 cells; a first attempt hit a 1800s cell timeout in notebook 01 — an unrelated, unmodified S0 cell — under what was traced to transient system load after a long idle period; an immediate retry with zero code changes completed cleanly, confirming this was environmental, not a regression) |
| `results/tables/*.json` (actual notebook output) vs Golden Master | **Exact match**, S0/S1/WF1/WF2 |
| `validate_against_golden_master.py` | **ALL MODELS RECONCILE EXACTLY** |
| `compare_trade_logs.py` | **ALL TRADE LOGS MATCH** — S0 29/29, S1 29/29, WF1 2/2, WF2 2/2 |
| Cache fingerprints | `pair_stability_523179f2a1.pkl`, `eg_results_b62b6e0656.pkl`, `wf1_selections_6d2bb2d492.pkl`, `wf1_selections_3m_d99bee3e45.pkl`, `selection_funnel_1e9d141c99.pkl` all **unmodified** (timestamps unchanged) — only one **new** cache added, `wf1_no_overlap_selections_e52f5400ab.pkl`, its own fingerprint namespace |
| Notebook-live vs. standalone-script numbers | **Exact match** — the notebook's live `ablation_comparison_df` reproduces `scripts/run_no_overlap_ablation.py`'s output bit-for-bit, confirming there is one implementation, not two |

**No existing model (S0/S1/WF1/WF2) output or trade history changed. B2.5 adds one new,
clearly-labeled exploratory variant only.**

## 3. New model results — `WF1_NO_OVERLAP_CONSTRAINT` vs. `WF1_SEMIANNUAL_REESTIMATION`

Every other rebalance date besides 2024-01-01 selects **zero pairs** under both models — zero
pairs are even FDR-significant there (the same purely-statistical bottleneck the B2 funnel
already found), so the ablation cannot create opportunity that does not statistically exist.
All results below are driven entirely by the 2024-01-01 block.

| Metric | WF1 (baseline) | WF1_NO_OVERLAP (ablation) |
|---|---:|---:|
| Pairs selected @ 2024-01-01 | 1 | 10 (of 15 fully-qualifying; 5 excluded only by the unchanged `MAX_PAIRS_PER_REBALANCE=10` cap) |
| Total executed trades (OOS) | 2 | 31 |
| Distinct pairs traded | 1 | 10 |
| Time in market | 1.50% | 14.07% |
| Daily-252 Sharpe | -0.068 | -0.475 |
| Cumulative net return | -3.80% | -10.10% |
| Max drawdown | -18.58% | -12.91% |
| Turnover | 4.15x | 6.30x |
| Total costs | 0.83% | 1.26% |
| Unconditional BTC beta | 0.0141 | 0.0016 |
| Active-day BTC beta (n obs) | 0.781 (n=16) | 0.009 (n=133) |

## 4. Concentration findings (2024-01-01 block)

- LDOUSDT gross exposure peaks at **46.9% of portfolio equity**, i.e. **70.7% of the
  portfolio's own total gross exposure** at that moment (`max_share_of_total_gross_single_coin`).
- LDOUSDT is the single largest gross-exposure coin on **87.8%** of all active bars in the
  block.
- No other coin's max gross exposure exceeds ~6.5% of equity (next-largest: VETUSDT 6.54%,
  NEARUSDT 6.31%, DOGEUSDT 6.26%, UNIUSDT 6.19%) — LDOUSDT dominates by roughly 7x over any
  other single coin.

## 5. LDOUSDT-specific findings

All 10 no-overlap-selected pairs at 2024-01-01 involve LDOUSDT (confirming the B2 funnel
finding directly at the execution stage, not just the screening stage). Of the 15
fully-qualifying LDOUSDT pairs: 10 selected (all generated real entry signals, 31 trades
total), 5 excluded solely by the unchanged max-pairs cap (ICPUSDT, AVAXUSDT, BNBUSDT, ADAUSDT,
APTUSDT — all vs. LDOUSDT), 20 further LDOUSDT pairs existed as candidates but never passed FDR.

Pairwise 4h net-return correlation across the 10 traded pairs: mean **0.163**, median **0.135**,
range [-0.170, 0.568], 25 of 45 pairwise correlations above +0.10, only 1 of 45 meaningfully
negative — mild-to-moderate positive co-movement, well below what ten copies of one trade would
show, but clearly above independence.

Per-pair P&L (sleeve-basis, own initial_equity=1.0): 3 of 10 pairs contributed positively
(VETUSDT +0.204, ATOMUSDT +0.098, DOTUSDT +0.082), 7 negatively, with SHIBUSDT/LDOUSDT the
single largest loser (-0.511, 27.4% of total absolute P&L) and FTMUSDT/LDOUSDT next
(-0.267, 14.3%).

## 6. Transaction-cost impact

Total costs rise from 0.83% to 1.26% of equity (turnover 4.15x → 6.30x) — proportionate to the
2 → 31 trade increase, not a disproportionate cost blowup.

## 7. Interpretation

Removing the one-pair-per-coin constraint did not improve OOS performance in this episode: both
Sharpe and cumulative net return deteriorate materially, transaction costs rise, and the
portfolio becomes dominated by a single coin's exposure for most of its active life (peak 70.7%
of gross exposure, largest coin on 87.8% of active bars). The LDOUSDT-cluster pairs' P&L series
show mild-to-moderate positive correlation, not independence — consistent with a partially
common LDOUSDT-centered exposure expressed through different counterparties rather than ten
diversified relative-value opportunities. On the measured evidence, the constraint appears to
have been primarily suppressing **concentration risk**, not a materially better opportunity set,
in this one episode.

Max drawdown is a partial exception: it is smaller in magnitude under the ablation (-12.91% vs.
-18.58%), a genuine diversification effect that does not carry through to Sharpe or cumulative
return, and should not be read as a general endorsement of the unconstrained approach.

## 8. Limitations

- **Single-episode finding.** Every result here comes from one rebalance block (2024-01-01) and
  31 trades — an even smaller effective sample than WF1/WF2's own already-flagged low-trade-count
  limitation. This is not a robust statistical conclusion in either direction.
- **No new concentration cap was introduced**, per the task's explicit scope — concentration is
  reported as a measured output, not suppressed.
- **Active-day beta's larger sample (n=133 vs. n=16) is a byproduct of trading more, not
  independent evidence of market-neutrality.**
- `WF1_NO_OVERLAP_CONSTRAINT` is **not** substituted for WF1 in the master model-comparison
  table and is **not** promoted to Notebook 04 as a primary result. It is not described as a new
  "best model."

## Stop condition

B2.5 implemented and validated as scoped. Per instructions, stopping here — B3 (Relationship
Survival Analysis) not started.
