# Latest Results

Out-of-sample window: 2024-01-01 to 2026-06-30. Sharpe/volatility use the daily-252 convention
(geometric compounding of 4h bars into UTC calendar days; flat days kept, not dropped -- see
`CLAUDE.md` "Evaluation conventions"). These figures are captured verbatim in
`results/golden_master.json` and reproduced exactly by the refactored `src/stat_arb` pipeline
(`results/refactor_validation.md`).

| Model | Daily-252 Sharpe | Cum. net return | Max drawdown | Trades | Time in market | BTC beta (unconditional) | BTC beta (active-day) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `S0_STATIC_BASELINE` | -0.1155 | -1.65% | -11.17% | 29 | 42.3% | 0.0020 | -- |
| `S1_STATIC_PAIR_HALFLIFE_STOP` | -0.1249 | -1.88% | -10.53% | 29 | 66.4% | -0.0020 | -- |
| `WF1_SEMIANNUAL_REESTIMATION` | -0.0679 | -3.80% | -18.58% | 2 | 1.50% | 0.0141 | 0.78 (see notebook 03) |
| `WF2_ROLLING_ENTRY_HEDGE_RATIO` | -0.0185 | -1.88% | -15.63% | 2 | 1.50% | 0.0146 | -- |

**None of the four models clears the ~0.5 Sharpe resume-relevance threshold on this sample.**
This is reported as an acceptable, negative research outcome, not a failure of the research
process -- see `CLAUDE.md`'s non-negotiable research principles.

## Caveats (do not drop when citing these numbers)

- **S1's time-stop multiplier is unstable**: the pre-specified 2.0x/2.5x/3.0x sensitivity grid
  produces Sharpe ratios ranging from roughly +0.34 to -0.20 across pairs (see
  `notebooks/01_data_and_static_baseline.ipynb`'s S1 robustness section) -- the time-stop
  choice does not robustly explain S0's underperformance; the instability itself is the
  finding.
- **WF1/WF2 trade only 2 times each** in the full OOS window, across a single pair
  (OM/LDO). Any comparison between them, or against S0/S1, must state this sample-size
  limitation rather than imply a robust conclusion.
- **WF1 is not market-neutral on the days it actually trades**: unconditional BTC beta (0.014)
  looks near zero only because the strategy is flat 98.5% of the time; its active-day beta is
  materially higher (see `notebooks/03_walkforward_strategy.ipynb`).
- This is a research-learning project: simplified short-leg implementation, no funding/borrow/
  margin modeling, Binance spot reference prices, flat per-leg cost assumption -- not a
  production P&L estimate.

## Source

Regenerate this table by running `notebooks/01_data_and_static_baseline.ipynb` and
`notebooks/03_walkforward_strategy.ipynb` in order (each writes its rows to
`results/tables/*.json`), then `notebooks/04_results_and_model_comparison.ipynb`.
