"""Statistical primitive tests, ported from `legacy/Stat_Arb_cont_res_1.ipynb` Section 5's
inline unit test plus new coverage for the consolidated EG/FDR functions."""
import numpy as np
import pandas as pd
import pytest

from stat_arb.statistics import (apply_bh_fdr, compute_half_life, compute_spread,
                                  count_crossings, fit_static_ols, run_engle_granger_tests,
                                  sanity_check_residual_mean)


def test_compute_spread_matches_definition():
    y = pd.Series([1.0, 2.0, 3.0])
    x = pd.Series([0.5, 1.0, 1.5])
    spread = compute_spread(y, x, alpha=0.1, beta=0.5)
    assert (spread == (y - 0.1 - 0.5 * x)).all()


def test_fit_static_ols_recovers_known_relationship():
    rng = np.random.default_rng(0)
    x = pd.Series(np.linspace(1, 10, 200))
    y = 0.3 + 1.2 * x + rng.normal(0, 1e-6, len(x))
    alpha, beta, _ = fit_static_ols(y, x)
    assert abs(alpha - 0.3) < 1e-3
    assert abs(beta - 1.2) < 1e-3


def test_sanity_check_residual_mean_passes_for_correct_direction():
    rng = np.random.default_rng(1)
    x = pd.Series(np.linspace(1, 10, 500))
    y = 0.1 + 0.9 * x + rng.normal(0, 0.05, len(x))
    alpha, beta, _ = fit_static_ols(y, x)
    assert sanity_check_residual_mean(y, x, alpha, beta, label='test')


def test_sanity_check_residual_mean_fails_for_biased_spread():
    x = pd.Series(np.linspace(1, 10, 100))
    y = 0.1 + 0.9 * x
    with pytest.raises(AssertionError):
        sanity_check_residual_mean(y, x, alpha=5.0, beta=0.9, tol=0.01, label='biased')


def test_compute_half_life_mean_reverting_series():
    rng = np.random.default_rng(2)
    n = 5000
    phi = 0.98  # implies a finite, positive half-life
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i-1] + rng.normal(0, 1)
    spread = pd.Series(s)
    hl = compute_half_life(spread)
    assert np.isfinite(hl) and hl > 0
    expected_hl = -np.log(2) / np.log(phi)
    assert abs(hl - expected_hl) / expected_hl < 0.15


def test_compute_half_life_non_mean_reverting_returns_inf():
    # Pure exponential growth: delta_t = 0.001 * level_{t-1} exactly, so the AR(1) slope k is
    # deterministically positive (k=0.001 > 0) -- the "k >= 0 implies no reversion" branch.
    n = 500
    spread = pd.Series([100.0 * (1.001 ** i) for i in range(n)])
    hl = compute_half_life(spread)
    assert hl == np.inf


def test_count_crossings_counts_sign_changes():
    spread = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
    assert count_crossings(spread) == 4
    flat = pd.Series([1.0, 1.0, 1.0])
    assert count_crossings(flat) == 0


def test_run_engle_granger_tests_both_directions():
    rng = np.random.default_rng(4)
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='4h')
    common = np.cumsum(rng.normal(0, 1, n))
    log_a = pd.Series(common + rng.normal(0, 0.1, n), index=idx)
    log_b = pd.Series(common * 1.5 + rng.normal(0, 0.1, n), index=idx)
    log_px = pd.DataFrame({'AUSDT': log_a, 'BUSDT': log_b})
    eg_df = run_engle_granger_tests(log_px, [('AUSDT', 'BUSDT')], min_obs=100)
    assert len(eg_df) == 2  # both directions tested
    assert set(eg_df['pair']) == {('AUSDT', 'BUSDT')}


def test_apply_bh_fdr_adds_columns_and_flags_significance():
    eg_df = pd.DataFrame({
        'pair': [('A', 'B'), ('A', 'B'), ('C', 'D'), ('C', 'D')],
        'y': ['A', 'B', 'C', 'D'], 'x': ['B', 'A', 'D', 'C'],
        'eg_stat': [-5, -5, -0.1, -0.1],
        'eg_pvalue': [0.0001, 0.0001, 0.9, 0.9],
        'n_obs': [500, 500, 500, 500],
    })
    out = apply_bh_fdr(eg_df, alpha=0.05, fdr_method='fdr_bh')
    assert 'pvalue_adj' in out.columns and 'significant_fdr' in out.columns
    assert out.loc[out['pair'] == ('A', 'B'), 'significant_fdr'].all()
    assert not out.loc[out['pair'] == ('C', 'D'), 'significant_fdr'].any()


def test_apply_bh_fdr_empty_input_returns_empty():
    empty = pd.DataFrame(columns=['pair', 'y', 'x', 'eg_stat', 'eg_pvalue', 'n_obs'])
    out = apply_bh_fdr(empty, alpha=0.05, fdr_method='fdr_bh')
    assert len(out) == 0
