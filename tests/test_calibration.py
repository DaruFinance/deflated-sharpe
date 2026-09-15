"""Monte Carlo checks that the statistics behave as claimed under a true null.

These use the standard library random generator with fixed seeds, and sizes
small enough to run in a few seconds. The tolerances are wide enough to be
stable across seeds and narrow enough to catch a formula that is off by a
meaningful factor.
"""

import math
import random
import statistics

import pytest

from deflated_sharpe import (
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_ratio,
)


@pytest.mark.parametrize("n_trials, rel", [(10, 0.05), (100, 0.03), (1000, 0.03)])
def test_expected_max_matches_simulated_max_of_normals(n_trials, rel):
    rng = random.Random(7 + n_trials)
    sims = 3000 if n_trials < 1000 else 800
    maxima = [max(rng.gauss(0.0, 1.0) for _ in range(n_trials)) for _ in range(sims)]
    simulated = statistics.fmean(maxima)
    analytic = expected_max_sharpe(n_trials, 1.0)
    # The approximation is asymptotic in N and slightly overstates the maximum
    # for small N: by about 2 to 3% at N = 10 across seeds, under 1% by N = 100.
    assert analytic == pytest.approx(simulated, rel=rel)


def test_psr_false_positive_rate_under_null():
    """With zero true Sharpe ratio, PSR(0) > 0.95 should happen about 5% of the time."""
    rng = random.Random(11)
    sims, T = 4000, 250
    hits = 0
    for _ in range(sims):
        xs = [rng.gauss(0.0, 0.01) for _ in range(T)]
        st = return_stats(xs)
        hits += probabilistic_sharpe_ratio(st.sharpe, T, st.skew, st.kurtosis) > 0.95
    rate = hits / sims
    assert 0.035 < rate < 0.065


def test_min_track_record_length_is_where_psr_reaches_confidence():
    sr, bench, skew, kurt = 0.08, 0.02, -0.5, 6.0
    n = min_track_record_length(sr, bench, skew=skew, kurtosis=kurt, confidence=0.9)
    assert probabilistic_sharpe_ratio(sr, n, skew, kurt, bench) == pytest.approx(0.9, abs=1e-12)


def _corpus_max_and_sr0(rng, n_trials, T, rho, n_for_bar):
    """Zero-skill corpus with equicorrelation rho; return (max SR, SR0)."""
    common = [rng.gauss(0.0, 1.0) for _ in range(T)]
    srs = []
    for _ in range(n_trials):
        xs = [math.sqrt(rho) * c + math.sqrt(1.0 - rho) * rng.gauss(0.0, 1.0) for c in common]
        srs.append(sharpe_ratio(xs))
    var = statistics.variance(srs)
    return max(srs), expected_max_sharpe(n_for_bar, var)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_null_max_clears_sr0_about_half_the_time_with_distinct_trial_count(rho):
    """SR0 is an expected maximum. With V estimated from the trials themselves and
    N set to the number of distinct trials, the best null trial beats it about half
    the time, whether or not the trials are correlated."""
    rng = random.Random(23 if rho == 0 else 29)
    sims, n_trials, T = 250, 40, 200
    above = sum(
        mx > sr0 for mx, sr0 in (_corpus_max_and_sr0(rng, n_trials, T, rho, n_trials) for _ in range(sims))
    )
    assert 0.35 < above / sims < 0.65


def test_effective_trial_count_makes_the_bar_too_lenient():
    """Substituting a correlation-adjusted count for N corrects for correlation twice,
    because the trials' observed Sharpe variance has already shrunk. With rho = 0.5
    the participation ratio of 40 equicorrelated trials is about 3.9, and a null
    corpus clears that bar most of the time."""
    rng = random.Random(31)
    sims, n_trials, T, rho = 250, 40, 200, 0.5
    lam1, lam_rest = 1 + (n_trials - 1) * rho, 1 - rho
    n_eff = n_trials ** 2 / (lam1 ** 2 + (n_trials - 1) * lam_rest ** 2)
    above = sum(
        mx > sr0 for mx, sr0 in (_corpus_max_and_sr0(rng, n_trials, T, rho, n_eff) for _ in range(sims))
    )
    assert above / sims > 0.75
