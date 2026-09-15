"""Every numerical example published in the three source papers, reproduced.

Where a paper rounds, the tolerance is the paper's last printed digit.
"""

import math

import pytest

from deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    to_period_sharpe,
    to_period_variance,
)

# ----------------------------------------------------------------------------
# Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", pp. 9-10.
# Annualised SR 2.5 on five years of daily data at 250 days a year, N = 100
# trials whose annualised Sharpe ratios have variance 1/2, skew -3, kurtosis 10.
# ----------------------------------------------------------------------------

PPY = 250
SR = to_period_sharpe(2.5, PPY)
V = to_period_variance(0.5, PPY)


def test_dsr_numerical_example():
    r = deflated_sharpe_ratio(sharpe=SR, n_obs=1250, n_trials=100, var_trials=V, skew=-3, kurtosis=10)
    assert r.sr0 == pytest.approx(0.1132, abs=5e-5)
    assert r.dsr == pytest.approx(0.9004, abs=5e-5)
    assert not r.passes(0.95)


def test_dsr_passes_after_46_trials():
    r = deflated_sharpe_ratio(sharpe=SR, n_obs=1250, n_trials=46, var_trials=V, skew=-3, kurtosis=10)
    assert r.dsr == pytest.approx(0.9505, abs=5e-5)
    assert r.passes(0.95)


def test_dsr_normal_returns_after_88_trials():
    r = deflated_sharpe_ratio(sharpe=SR, n_obs=1250, n_trials=88, var_trials=V, skew=0, kurtosis=3)
    assert r.dsr == pytest.approx(0.9505, abs=5e-5)


# ----------------------------------------------------------------------------
# Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
# section 3 and section 5. A hedge fund with a monthly SR of 0.458 over two
# years; skew -2.448 and kurtosis 10.164.
# ----------------------------------------------------------------------------


def test_psr_hedge_fund_normal():
    assert probabilistic_sharpe_ratio(0.458, 24) == pytest.approx(0.982, abs=5e-4)


def test_psr_hedge_fund_non_normal():
    assert probabilistic_sharpe_ratio(0.458, 24, skew=-2.448, kurtosis=10.164) == pytest.approx(0.913, abs=5e-4)


def test_psr_hedge_fund_three_years():
    # The paper prints 0.953; the value is 0.95351, so the paper truncates.
    assert probabilistic_sharpe_ratio(0.458, 36, skew=-2.448, kurtosis=10.164) == pytest.approx(0.953, abs=1e-3)


@pytest.mark.parametrize(
    "ppy, skew, kurt, years",
    [
        (252, 0.0, 3.0, 2.73),  # daily IID normal
        (52, 0.0, 3.0, 2.83),  # weekly
        (12, 0.0, 3.0, 3.24),  # monthly
        (12, -0.72, 5.78, 4.99),  # monthly, HFR aggregate index moments
    ],
)
def test_min_track_record_length_examples(ppy, skew, kurt, years):
    n = min_track_record_length(
        to_period_sharpe(2.0, ppy), to_period_sharpe(1.0, ppy), skew=skew, kurtosis=kurt, confidence=0.95
    )
    assert n / ppy == pytest.approx(years, abs=5e-3)


# ----------------------------------------------------------------------------
# Bailey, Borwein, Lopez de Prado and Zhu (2014), "Pseudo-Mathematics and
# Financial Charlatanism", Proposition 2.1 and Theorem 3.1.
# ----------------------------------------------------------------------------


def test_expected_max_of_ten_standard_normals():
    assert expected_max_sharpe(10, 1.0) == pytest.approx(1.57, abs=5e-3)


def test_min_backtest_length_five_years_allows_45_trials():
    assert min_backtest_length(45, 1.0) < 5.0 < min_backtest_length(46, 1.0)


def test_min_backtest_length_below_upper_bound():
    for n in (2, 10, 100, 1_000, 100_000):
        assert min_backtest_length(n, 1.0) < 2 * math.log(n)
