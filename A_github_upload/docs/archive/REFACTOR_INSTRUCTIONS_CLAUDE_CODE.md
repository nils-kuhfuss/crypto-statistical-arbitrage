# Claude Code Refactor Instructions — Crypto Statistical Arbitrage Project

## Objective

Refactor the existing Jupyter notebook:

`Stat_Arb_cont_res_1.ipynb`

into a modular, readable research project inside the folder:

`Stat Arb - Pari Trading`

This is a **pure software refactor**.

**Do not change the research methodology, model definitions, parameters, statistical tests, signal rules, transaction costs, execution timing, portfolio construction, or numerical results.**

The current notebook is the source of truth.

---

## 1. First: inspect and preserve the current project

Before changing anything:

1. Inspect the entire project folder and the full notebook.
2. Identify:
   - code cells,
   - markdown cells,
   - functions,
   - global variables,
   - configs,
   - cached files,
   - helper scripts,
   - datasets,
   - tests/assertions,
   - model outputs.
3. Preserve the original notebook unchanged.
4. Create the new modular project inside `Stat Arb - Pari Trading`.
5. Do not delete or overwrite existing files without checking them first.

Copy the unchanged notebook to:

`legacy/Stat_Arb_cont_res_1.ipynb`

---

## 2. Golden Master before refactoring

Before extracting code, capture the exact current model outputs.

Create:

`results/golden_master.json`

and, where useful:

`results/golden_master_trades/`

Capture at least for:

- `S0_STATIC_BASELINE`
- `S1_STATIC_PAIR_HALFLIFE_STOP`
- `WF1_SEMIANNUAL_REESTIMATION`
- `WF2_ROLLING_ENTRY_HEDGE_RATIO`

Store:

- cumulative gross return,
- cumulative net return,
- daily-252 Sharpe,
- annualized volatility,
- max drawdown,
- trade count,
- selected pairs,
- trade entry/exit timestamps,
- trade directions,
- trade returns,
- BTC correlation,
- BTC beta,
- active-day BTC beta,
- annualized alpha,
- alpha t-stat,
- HAC alpha t-stat if available,
- time in market,
- turnover,
- cost drag,
- final equity value.

Extract these from the executable notebook state. Do not rely only on manually typed documentation.

After the refactor, equivalent outputs must match the Golden Master. Any unexplained difference means the refactor is not complete.

---

## 3. Target project structure

Create:

```text
Stat Arb - Pari Trading/
│
├── CLAUDE.md
├── README.md
├── pyproject.toml
│
├── config/
│   ├── __init__.py
│   └── research_config.py
│
├── src/
│   └── stat_arb/
│       ├── __init__.py
│       ├── data.py
│       ├── diagnostics.py
│       ├── statistics.py
│       ├── pair_selection.py
│       ├── signals.py
│       ├── execution.py
│       ├── portfolio.py
│       ├── metrics.py
│       ├── walkforward.py
│       ├── cache.py
│       └── reporting.py
│
├── notebooks/
│   ├── 01_data_and_static_baseline.ipynb
│   ├── 02_relationship_stability.ipynb
│   ├── 03_walkforward_strategy.ipynb
│   └── 04_results_and_model_comparison.ipynb
│
├── tests/
│   ├── test_data_eligibility.py
│   ├── test_no_lookahead.py
│   ├── test_statistics.py
│   ├── test_execution.py
│   ├── test_portfolio_reconciliation.py
│   ├── test_daily_return_reconciliation.py
│   ├── test_pair_selection.py
│   └── test_cache_fingerprint.py
│
├── scripts/
│   └── run_stability_screen.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── results/
│   ├── golden_master.json
│   ├── golden_master_trades/
│   ├── tables/
│   └── latest_results.md
│
└── legacy/
    └── Stat_Arb_cont_res_1.ipynb
```

Do not duplicate large datasets unnecessarily. If existing data files should remain in place, centralize their paths in the config.

---

## 4. Central configuration

Move all research parameters into:

`config/research_config.py`

Include the existing values for:

- date splits,
- bar frequency,
- formation window,
- rebalance frequency,
- coverage requirements,
- active-at-rebalance rule,
- exclusions,
- entry threshold,
- exit band,
- hard stop,
- transaction costs,
- FDR alpha/method,
- maximum Half-Life,
- minimum crossings,
- positive-Beta rule,
- Half-Life multiplier,
- robustness multipliers,
- one-pair-per-coin rule,
- rolling hedge-ratio window,
- model IDs.

Do not introduce or modify any value.

---

## 5. Module responsibilities

### `data.py`
Only data-related logic:
- loading prices/returns,
- synchronization,
- universe construction,
- date slicing,
- point-in-time availability,
- formation coverage,
- active-at-rebalance checks.

### `diagnostics.py`
- missing-data diagnostics,
- coverage tables,
- history diagnostics,
- ADF/I(1) diagnostics,
- descriptive statistics.

### `statistics.py`
- OLS,
- Engle-Granger,
- spread construction,
- BH/FDR,
- Half-Life,
- mean crossings,
- correlation,
- reusable statistical helpers.

### `pair_selection.py`
- pair generation,
- directed tests,
- eligibility,
- FDR-qualified selection,
- positive-Beta rule,
- Half-Life/crossing filters,
- deterministic direction selection,
- ranking,
- one-pair-per-coin selection.

### `signals.py`
- trading spread,
- Z-score,
- entry crossings,
- exit band,
- hard-stop triggers,
- Time Stop triggers.

### `execution.py`
- state machine,
- delayed entry,
- delayed exit,
- fixed quantities,
- trade lifecycle,
- costs,
- trade P&L,
- MAE/MFE,
- period-end handling.

### `portfolio.py`
- pair sleeves,
- coin-level netting,
- gross/net returns,
- turnover,
- costs,
- gross/net exposure,
- active pair count,
- active flag,
- equity curve.

### `metrics.py`
- 4h-to-daily reconciliation,
- daily geometric aggregation,
- daily-252 Sharpe,
- daily-252 annualized volatility,
- cumulative return,
- drawdown,
- BTC correlation,
- BTC beta,
- active-day beta,
- alpha,
- alpha t-stat,
- HAC diagnostics,
- R-squared.

### `walkforward.py`
Orchestrate:
- rebalance dates,
- formation windows,
- point-in-time screening,
- parameter estimation,
- WF1,
- WF2,
- block concatenation.

Do not duplicate lower-level logic here.

### `cache.py`
- methodology fingerprint,
- cache paths,
- load/save,
- validation,
- stale-cache prevention.

### `reporting.py`
- standardized model summaries,
- comparison tables,
- trade tables,
- pair contribution,
- selection tables,
- results export.

---

## 6. Split the notebook into four research notebooks

The notebooks should become **research narratives**, not backtest engines.

Large reusable functions must live in `src/stat_arb/`.

### `01_data_and_static_baseline.ipynb`

Include:

- research setup,
- data loading,
- universe construction,
- data-quality analysis,
- ADF/I(1) diagnostics,
- sample/validation/post-sample definitions,
- static pair analysis,
- Engle-Granger + FDR,
- static pair filters,
- S0,
- S1,
- 2x / 2.5x / 3x Half-Life robustness,
- static benchmark metrics,
- interpretation.

### `02_relationship_stability.ipynb`

Include existing implemented logic for:

- walk-forward universe,
- 12-month formation windows,
- point-in-time eligibility,
- coverage,
- active-at-rebalance,
- XMR point-in-time treatment,
- repeated cointegration screens,
- FDR,
- positive Beta,
- Half-Life/crossing filters,
- qualifying pairs over time,
- pair persistence,
- Beta evolution,
- Half-Life evolution,
- adjusted p-value evolution,
- stability charts/tables.

Do not add new persistence methodology during this refactor.

### `03_walkforward_strategy.ipynb`

Include:

- WF1 methodology,
- rebalance schedule,
- formation estimates,
- pair ranking,
- one-pair-per-coin,
- fixed sleeves,
- WF1 trades,
- WF1 portfolio,
- WF1 benchmark metrics,
- Time Stop robustness,
- 6M vs 3M robustness already implemented,
- WF2 rolling entry hedge sizing,
- WF2 trades/results,
- WF1 vs WF2,
- low-trade-count caveats.

Do not describe WF2 as a rolling-spread model.

### `04_results_and_model_comparison.ipynb`

This should be the cleanest notebook for an external reviewer.

Include:

- research question,
- brief methodology,
- model registry,
- standardized S0/S1/WF1/WF2 table,
- equity curves,
- drawdowns,
- daily Sharpe,
- annualized volatility,
- trade counts,
- exposure,
- turnover,
- cost drag,
- BTC correlation/beta,
- active-day beta,
- alpha,
- alpha t-stat,
- HAC alpha t-stat,
- R-squared,
- pair contributions where already implemented,
- existing robustness results,
- limitations,
- final research conclusions.

No strategy implementation code should be duplicated here.

---

## 7. Move assertions/regression gates into tests

Convert current notebook checks into proper tests where possible.

Test:

### Point-in-time / eligibility
- formation window ends before trading period,
- minimum coverage,
- active-at-rebalance,
- XMR eligible before delisting if rules permit,
- delisted assets cannot re-enter afterward.

### Execution
- signal precedes execution,
- first P&L occurs only after execution,
- exit trigger precedes exit execution,
- costs are charged according to current methodology.

### Statistics
- correct directed EG family,
- BH/FDR behavior,
- deterministic direction,
- positive Beta,
- Half-Life filter,
- crossing filter.

### Portfolio
- one pair per coin,
- sleeves <= 100%,
- time in market <= 100%,
- repeated pair instances are not overwritten,
- portfolio cashflows reconcile.

### Returns
- 4h returns reconcile to equity,
- daily returns reconcile to 4h equity,
- daily Sharpe keeps flat days.

### Cache
- methodology changes alter the fingerprint,
- stale cache cannot silently load.

---

## 8. Existing expensive precompute scripts

If the project contains an existing script such as:

`build/run_stability_screen_v2.py`

preserve its behavior.

If appropriate, move/adapt it to:

`scripts/run_stability_screen.py`

but make it import the same core functions from `src/stat_arb/`.

Do not maintain a second independent implementation of:

- Engle-Granger,
- eligibility,
- Half-Life,
- FDR,
- pair selection.

---

## 9. Documentation

### `README.md`

Document:

- project objective,
- folder architecture,
- model registry,
- data assumptions,
- install/run instructions,
- notebook order,
- how to run tests,
- how to run expensive stability screens,
- results location,
- research-learning disclaimer.

Keep performance claims cautious.

### `CLAUDE.md`

Keep stable project governance here:

- no look-ahead,
- no optimization on evaluation data,
- one assumption at a time,
- pre-specified robustness tests only,
- negative results are valid,
- low trade counts require caveats,
- exact model definitions,
- project architecture,
- cache rules,
- coding conventions,
- test requirements.

Where practical, move volatile numerical outputs to:

`results/latest_results.md`

---

## 10. Coding rules

During this refactor:

- preserve calculation order,
- preserve Pandas alignment,
- preserve timestamps,
- preserve execution lag,
- preserve cost timing,
- preserve missing-value handling,
- preserve `ddof`,
- preserve regression settings,
- preserve numerical behavior,
- avoid hidden global state,
- prefer explicit inputs/outputs,
- add useful type hints,
- add concise docstrings,
- point-in-time functions must document what information they are allowed to use,
- avoid circular imports,
- avoid duplicate logic.

All code and notebook documentation should remain in English.

---

## 11. No new research in this task

Do **not** implement:

- new entry/exit thresholds,
- new FDR thresholds,
- new Half-Life filters,
- new crossing filters,
- new ranking rules,
- new persistence methodology,
- relationship-survival analysis,
- new weighting,
- new universe methodology,
- new rolling-spread model,
- new transaction-cost assumptions.

This task must isolate software changes from research changes.

---

## 12. Validation after refactor

After creating the modular project:

1. Run the full test suite.
2. Restart and Run All four notebooks from clean kernels.
3. Verify there is no hidden notebook-state dependency.
4. Recompute all existing model outputs.
5. Compare metrics to the Golden Master.
6. Compare selected pairs.
7. Compare trade logs.
8. Compare final equity values.
9. Verify cache fingerprints.
10. Produce:

`results/refactor_validation.md`

The validation report must state:

- tests passed/failed,
- notebook execution status,
- numerical parity by model,
- any differences,
- exact explanation for accepted machine-precision differences,
- unresolved issues.

Do not declare the refactor complete until numerical parity is confirmed.

---

## 13. Recommended execution order

Work incrementally:

1. Inspect project and notebook.
2. Build dependency inventory.
3. Capture Golden Master.
4. Create directory skeleton.
5. Centralize config.
6. Extract statistical helpers.
7. Extract execution.
8. Extract portfolio.
9. Extract metrics.
10. Extract data/diagnostics.
11. Extract pair selection.
12. Extract walk-forward orchestration.
13. Extract cache logic.
14. Extract reporting.
15. Build tests.
16. Build the four notebooks.
17. Run tests.
18. Restart/Run All notebooks.
19. Reconcile with Golden Master.
20. Update README, CLAUDE.md and results docs.

If a step creates a numerical discrepancy, stop and resolve it before continuing.

---

# Final instruction

Transform:

`Stat_Arb_cont_res_1.ipynb`

into a clean modular research codebase inside:

`Stat Arb - Pari Trading`

while preserving the **exact existing research methodology and numerical results**.

Do not improve the strategy during this task.

Improve only:

- structure,
- readability,
- modularity,
- reproducibility,
- testability,
- maintainability,
- documentation.
