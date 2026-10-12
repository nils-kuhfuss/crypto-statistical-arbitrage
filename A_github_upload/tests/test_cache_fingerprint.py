"""Cache fingerprint tests, ported/adapted from `legacy/Stat_Arb_cont_res_1.ipynb` Section 19
(`wf_cache_fingerprint`'s docstring: "changing any one of these automatically produces a
different fingerprint... a stale cache can never be silently reused under a new methodology")."""
import os

import pandas as pd
import pytest

from stat_arb.cache import eg_cache_fingerprint, load_or_assert, load_or_compute, wf_cache_fingerprint

BASE_WF_CONFIG = {
    'MANUAL_EXCLUDE': ['USDCUSDT', 'FDUSDUSDT'],
    'FORMATION_MONTHS': 12,
    'MIN_COVERAGE_FRACTION': 0.90,
    'EG_ALPHA': 0.05,
    'FDR_METHOD': 'fdr_bh',
    'MAX_HALF_LIFE_BARS': 300,
    'MIN_CROSSINGS': 15,
    'REQUIRE_POSITIVE_BETA': True,
    'MIN_OBS_FOR_TEST': 1973,
}
UNIVERSE = ['AUSDT', 'BUSDT', 'CUSDT']
REBALANCE_DATES = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-07-01')]


def test_fingerprint_stable_for_identical_inputs():
    fp1 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    fp2 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    assert fp1 == fp2


@pytest.mark.parametrize('changed_key,new_value', [
    ('MIN_OBS_FOR_TEST', 1000),
    ('MIN_COVERAGE_FRACTION', 0.5),
    ('FORMATION_MONTHS', 6),
    ('EG_ALPHA', 0.10),
    ('FDR_METHOD', 'bonferroni'),
    ('MAX_HALF_LIFE_BARS', 500),
    ('MIN_CROSSINGS', 20),
    ('REQUIRE_POSITIVE_BETA', False),
    ('MANUAL_EXCLUDE', ['USDCUSDT']),
])
def test_fingerprint_changes_with_each_methodology_parameter(changed_key, new_value):
    base_fp = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    changed_config = dict(BASE_WF_CONFIG)
    changed_config[changed_key] = new_value
    changed_fp = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=changed_config)
    assert base_fp != changed_fp, f'changing {changed_key} did not change the fingerprint'


def test_fingerprint_changes_with_universe():
    fp1 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    fp2 = wf_cache_fingerprint(universe_cols=UNIVERSE + ['DUSDT'], rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    assert fp1 != fp2


def test_fingerprint_changes_with_rebalance_dates():
    fp1 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG)
    fp2 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES + [pd.Timestamp('2025-01-01')],
                                wf_config=BASE_WF_CONFIG)
    assert fp1 != fp2


def test_fingerprint_changes_with_extra():
    fp1 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG, extra='a')
    fp2 = wf_cache_fingerprint(universe_cols=UNIVERSE, rebalance_dates=REBALANCE_DATES, wf_config=BASE_WF_CONFIG, extra='b')
    assert fp1 != fp2


def test_load_or_assert_raises_loudly_when_missing(tmp_path):
    missing_path = str(tmp_path / 'does_not_exist.pkl')
    with pytest.raises(AssertionError, match='custom message'):
        load_or_assert(missing_path, 'custom message: cache not found')


def test_load_or_compute_computes_and_writes_on_miss(tmp_path):
    path = str(tmp_path / 'cache.pkl')
    calls = []

    def compute():
        calls.append(1)
        return {'a': 1}

    result1 = load_or_compute(path, compute)
    assert result1 == {'a': 1}
    assert os.path.exists(path)
    assert len(calls) == 1

    result2 = load_or_compute(path, compute)
    assert result2 == {'a': 1}
    assert len(calls) == 1, 'compute_fn must not be called again once the cache exists'


def test_eg_cache_fingerprint_changes_with_formation_window():
    fp1 = eg_cache_fingerprint(UNIVERSE, '2020-01-01', '2022-12-31')
    fp2 = eg_cache_fingerprint(UNIVERSE, '2020-01-01', '2023-12-31')
    assert fp1 != fp2
