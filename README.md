# Crypto Statistical Arbitrage

An independent Python research project investigating whether cryptocurrency pairs trading can produce convincing risk-adjusted returns beyond its formation period.

Author: Nils Kuhfuß

Started: June 2026

Status: Four strategy variants evaluated; research currently paused.

Research outcome: I did not identify convincing alpha in the tested specifications. The project documents the strategy design, comparisons and limitations behind that negative result.

Research question:

Can cointegration-based pairs trading remain effective when pairs and parameters are selected using only information available at the time, signals are executed on the next bar, and transaction costs are included?

The research compares a static baseline with changes to holding periods, pair re-selection and hedge sizing. The aim is to understand whether each modification improves the evidence for a trading edge.

Price source:	Binance spot reference prices
Frequency:	Four-hour bars
Initial formation period:	1 January 2020 – 31 December 2022
Validation period:	1 January 2023 – 31 December 2023
Final evaluation period:	1 January 2024 – 30 June 2026, labelled final out-of-sample in the project configuration
Static universe:	28 coins for S0 and S1, Walk-forward universe	45 coins, rebuilt separately from the static universe


Dates and parameters are defined in config/research_config.py. The walk-forward models use their own rolling formation and rebalancing schedule within the evaluation workflow.
The two universes differ. Changes in performance between static and walk-forward models cannot automatically be attributed only to the strategy modification.
Strategy variants

Model	Research design

- S0_STATIC_BASELINE	Fixed pairs, formation-estimated intercept and hedge ratio, and one global time stop.
- S1_STATIC_PAIR_HALFLIFE_STOP	S0 with a pair-specific time stop equal to the ceiling of 2.5 times the formation-period half-life.
- WF1_SEMIANNUAL_REESTIMATION	Walk-forward pair re-selection using a 12-month formation window and six-month rebalancing, with point-in-time eligibility rules.
- WF2_ROLLING_ENTRY_HEDGE_RATIO	WF1 with hedge sizing updated at trade entry. Signals still use the frozen formation-period intercept and hedge ratio; this is not a rolling-spread model.


Relationship-stability diagnostics are descriptive and are not themselves trading signals.

Research and implementation approach
- Estimate formation parameters before the corresponding evaluation period.
- Apply point-in-time eligibility rules in the walk-forward workflow.
- Use next-bar execution and a flat transaction-cost assumption for each leg.
- Compare model outputs using a standardized, benchmark-relative evaluation.
- Use predefined sensitivity checks rather than optimizing parameters on evaluation-period performance.
- Preserve pre-refactor reference outputs to check that restructuring the code does not change numerical results.

Findings and interpretation
The four tested variants did not provide convincing evidence of a usable trading edge under the project's assumptions. Negative results are retained as part of the research record.
The detailed comparison is documented in notebooks/04_results_and_model_comparison.ipynb; model-level headline metrics are stored in results/latest_results.md. The conclusion is specific to the models, samples and assumptions tested. It does not establish that all cryptocurrency pairs-trading strategies are ineffective.
No live trading performance is reported.

Limitations
- Simplified short positions: the short leg is represented as a signed notional position. It is not a fully modelled spot borrow or perpetual-futures contract.
- Incomplete implementation costs: funding, borrowing and margin are not modelled. Flat per-leg transaction costs do not capture the full variation in real execution costs.
- Reference-price backtest: Binance spot prices do not by themselves demonstrate that both legs could have been executed at the assumed prices.
- Sample and selection sensitivity: universe construction, changing pair relationships, repeated model comparisons and low trade counts in some specifications limit the strength of the evidence.
The outputs are research backtests, not estimates of deployable production P&L.

Repository structure

Location	Purpose

config/	Central research parameters and data/cache paths, defined from a single project root.
src/stat_arb/	Research engine: data handling, statistics, signals, execution, portfolio construction, metrics, selection, walk-forward orchestration, caching and reporting.
notebooks/	Four ordered research narratives that call the shared implementation.
tests/	Tests for eligibility, execution, statistics, return reconciliation and cache behaviour.
scripts/	Standalone stability precompute and supporting reference-output/notebook tools.
data/raw/	Required input files: px.pkl and ret.pkl.
data/cache/	Regeneratable intermediate outputs, keyed by methodology fingerprints and excluded from version control.
data/processed/	Reserved for future derived datasets; currently unused.
results/	Comparison tables, headline results, reference outputs and validation reports.
legacy/	Original notebook, Stat_Arb_cont_res_1.ipynb, retained as the pre-refactor reference.
docs/	Methodology, project-management and archived documentation.
backups/	Local safety snapshots, excluded from version control.


CLAUDE.md contains additional project conventions and research principles. The original notebook remains the reference for the refactored implementation.
Setup

Run the following from the project root in the intended Python environment:
pip install -e ".[dev]"
python -m ipykernel install --user --name python3
The documented development dependencies include pytest, nbformat, nbclient and ipykernel. The python3 kernel is required for headless notebook execution.
The workflow expects data/raw/px.pkl and data/raw/ret.pkl. Ensure those inputs are available before running the analysis; the commands above are environment-setup instructions, not a data-download workflow.
Notebook order
Run each notebook from a clean kernel using Restart & Run All.
Order	Notebook	Purpose

1	01_data_and_static_baseline.ipynb	Load data, construct the static universe, screen pairs, and evaluate S0, S1 and time-stop sensitivity.
2	02_relationship_stability.ipynb	Examine the walk-forward universe and point-in-time relationship stability.
3	03_walkforward_strategy.ipynb	Evaluate WF1, its sensitivity checks and WF2.
4	04_results_and_model_comparison.ipynb	Assemble the standardized comparison, limitations and research conclusion.


Notebook 01 writes results/tables/s0_s1_rows.json; notebook 03 writes results/tables/wf1_wf2_rows.json. Notebook 04 reads both outputs.
Before running notebooks 02 and 03, generate the required stability cache:
python scripts/run_stability_screen.py

This resumable precompute performs the directional Engle–Granger stability tests across rebalance dates. It writes data/cache/pair_stability_<fingerprint>.pkl. The fingerprint incorporates the universe, rebalance schedule and walk-forward configuration so that a changed methodology uses a different cache key.
Tests and reference outputs

Run the test suite from the project root:
pytest
The existing conftest.py adds src/ and the project root to the import path.
Relevant outputs include:
- results/golden_master.json and results/golden_master_trades/: reference outputs captured before the refactor.
- results/latest_results.md: headline metrics for each model.
- results/refactor_validation.md: numerical-parity validation for the modular refactor.
- results/folder_cleanup_validation.md: validation following folder reorganisation.
