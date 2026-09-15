"""The Probabilistic and Deflated Sharpe Ratios and the two length bounds.

Every function here works in per-period units: a Sharpe ratio of daily returns
is the mean daily return over the standard deviation of daily returns, and
``n_obs`` counts daily observations. Use :mod:`deflated_sharpe.frequency` to
move between annualised and per-period values.

Kurtosis is the raw (Pearson) kurtosis, equal to 3 for a normal distribution.
Pass ``excess_kurtosis=`` instead if the value to hand is already excess.

References
----------
Bailey, D. H. and M. Lopez de Prado (2012). The Sharpe Ratio Efficient Frontier.
    Journal of Risk 15(2), 3-44.
Bailey, D. H. and M. Lopez de Prado (2014). The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality. Journal of
    Portfolio Management 40(5), 94-107.
Bailey, D. H., J. M. Borwein, M. Lopez de Prado and Q. J. Zhu (2014).
    Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
    Overfitting on Out-of-Sample Performance. Notices of the AMS 61(5), 458-471.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

__all__ = [
    "EULER_MASCHERONI",
    "DSRResult",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "min_backtest_length",
    "sharpe_ratio_std_error",
]

EULER_MASCHERONI = 0.5772156649015329
_Z = NormalDist()


def _resolve_kurtosis(kurtosis: Optional[float], excess_kurtosis: Optional[float]) -> float:
    if kurtosis is not None and excess_kurtosis is not None:
        raise ValueError("pass either kurtosis (raw, normal = 3) or excess_kurtosis (normal = 0), not both")
    if excess_kurtosis is not None:
        kurtosis = excess_kurtosis + 3.0
    if kurtosis is None:
        kurtosis = 3.0
    if not math.isfinite(kurtosis):
        raise ValueError("kurtosis must be finite")
    if kurtosis < 1.0:
        # Raw kurtosis is bounded below by 1 for any distribution. A value under 1
        # is almost always excess kurtosis passed as raw.
        raise ValueError(
            f"kurtosis={kurtosis} is below 1, which no distribution can have; "
            "if this is excess kurtosis, pass excess_kurtosis= instead"
        )
    return float(kurtosis)


def _check_finite(**values: float) -> None:
    for name, v in values.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number, got {v!r}")


def _variance_term(sr: float, skew: float, kurtosis: float) -> float:
    """1 - skew * SR + (kurtosis - 1) / 4 * SR^2, the non-normality factor.

    It is the asymptotic variance of the Sharpe ratio estimator times the sample
    length. Very large positive skew combined with a large SR can drive it below
    zero, where the approximation behind PSR no longer holds.
    """
    v = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if v <= 0.0:
        raise ValueError(
            f"non-normality term 1 - skew*SR + (kurtosis-1)/4*SR^2 = {v:.4g} is not positive "
            f"(SR={sr}, skew={skew}, kurtosis={kurtosis}); the asymptotic approximation does not apply. "
            "Check that SR is per-period, not annualised."
        )
    return v


def sharpe_ratio_std_error(
    sharpe: float,
    n_obs: float,
    skew: float = 0.0,
    kurtosis: Optional[float] = None,
    *,
    excess_kurtosis: Optional[float] = None,
) -> float:
    """Standard error of a per-period Sharpe ratio estimate under non-normal returns.

    ``sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (n_obs - 1))``, the denominator
    of the Probabilistic Sharpe Ratio (Bailey and Lopez de Prado 2012).
    """
    k = _resolve_kurtosis(kurtosis, excess_kurtosis)
    _check_finite(sharpe=sharpe, n_obs=n_obs, skew=skew)
    if n_obs <= 1:
        raise ValueError("n_obs must be greater than 1")
    return math.sqrt(_variance_term(sharpe, skew, k) / (n_obs - 1.0))


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: float,
    skew: float = 0.0,
    kurtosis: Optional[float] = None,
    benchmark: float = 0.0,
    *,
    excess_kurtosis: Optional[float] = None,
) -> float:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark``.

    PSR(SR*) = Z[ (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurtosis-1)/4 * SR^2) ]

    Parameters
    ----------
    sharpe:
        Observed per-period Sharpe ratio of the strategy.
    n_obs:
        Number of return observations behind ``sharpe`` (T).
    skew, kurtosis:
        Sample skewness and raw kurtosis of those returns. Defaults describe a
        normal distribution.
    benchmark:
        Per-period Sharpe ratio to test against (SR*). Zero asks whether the
        strategy has any skill at all.
    excess_kurtosis:
        Alternative to ``kurtosis`` for values where normal = 0.
    """
    se = sharpe_ratio_std_error(sharpe, n_obs, skew, kurtosis, excess_kurtosis=excess_kurtosis)
    _check_finite(benchmark=benchmark)
    return _Z.cdf((sharpe - benchmark) / se)


def expected_max_sharpe(
    n_trials: float,
    var_trials: float,
    mean_trials: float = 0.0,
) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent zero-skill trials.

    SR0 = mean + sqrt(V) * [ (1 - g) * Zinv(1 - 1/N) + g * Zinv(1 - 1/(N e)) ]

    where g is the Euler-Mascheroni constant (Bailey and Lopez de Prado 2014, eq. 1).
    It is an expected maximum, not a quantile: under the null the best of N trials
    lands above it roughly half the time.

    Parameters
    ----------
    n_trials:
        Number of distinct strategy configurations the selected one was chosen
        from (N). See ``docs/trial-count.md`` before substituting an "effective"
        count.
    var_trials:
        Variance of the per-period Sharpe ratios across those trials, V[{SR_n}].
    mean_trials:
        Expected Sharpe ratio across trials under the null. The paper uses 0.
    """
    _check_finite(n_trials=n_trials, var_trials=var_trials, mean_trials=mean_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if var_trials < 0:
        raise ValueError("var_trials must be non-negative")
    if n_trials == 1:
        # A single trial involves no selection, so there is nothing to deflate.
        return float(mean_trials)
    g = EULER_MASCHERONI
    z = (1.0 - g) * _Z.inv_cdf(1.0 - 1.0 / n_trials) + g * _Z.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_trials + math.sqrt(var_trials) * z)


@dataclass(frozen=True)
class DSRResult:
    """Output of :func:`deflated_sharpe_ratio`. All Sharpe values are per period."""

    dsr: float
    """Deflated Sharpe Ratio: probability the true SR exceeds the selection threshold."""
    sr0: float
    """Expected maximum Sharpe ratio under the null (the rejection threshold)."""
    sharpe: float
    n_obs: float
    n_trials: float
    var_trials: float
    skew: float
    kurtosis: float
    psr_zero: float
    """Probabilistic Sharpe Ratio against a zero benchmark, for comparison."""

    def passes(self, confidence: float = 0.95) -> bool:
        """True when ``dsr`` exceeds ``confidence``."""
        return self.dsr > confidence


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: float,
    n_trials: float,
    var_trials: float,
    skew: float = 0.0,
    kurtosis: Optional[float] = None,
    mean_trials: float = 0.0,
    *,
    excess_kurtosis: Optional[float] = None,
) -> DSRResult:
    """Deflated Sharpe Ratio: PSR evaluated at the expected maximum of the trials.

    DSR = PSR(SR0), with SR0 from :func:`expected_max_sharpe`
    (Bailey and Lopez de Prado 2014, eq. 2).

    Example
    -------
    The paper's numerical example: annualised SR 2.5 on five years of daily data
    (250 per year), N = 100 trials whose annualised Sharpe ratios have variance
    1/2, skew -3, kurtosis 10.

    >>> from deflated_sharpe import deflated_sharpe_ratio, to_period_sharpe, to_period_variance
    >>> r = deflated_sharpe_ratio(
    ...     sharpe=to_period_sharpe(2.5, 250), n_obs=1250, n_trials=100,
    ...     var_trials=to_period_variance(0.5, 250), skew=-3, kurtosis=10)
    >>> round(r.sr0, 4), round(r.dsr, 4)
    (0.1132, 0.9004)
    """
    k = _resolve_kurtosis(kurtosis, excess_kurtosis)
    sr0 = expected_max_sharpe(n_trials, var_trials, mean_trials)
    dsr = probabilistic_sharpe_ratio(sharpe, n_obs, skew, k, benchmark=sr0)
    psr0 = probabilistic_sharpe_ratio(sharpe, n_obs, skew, k, benchmark=0.0)
    return DSRResult(
        dsr=dsr, sr0=sr0, sharpe=float(sharpe), n_obs=float(n_obs), n_trials=float(n_trials),
        var_trials=float(var_trials), skew=float(skew), kurtosis=k, psr_zero=psr0,
    )


def min_track_record_length(
    sharpe: float,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: Optional[float] = None,
    confidence: float = 0.95,
    *,
    excess_kurtosis: Optional[float] = None,
) -> float:
    """Minimum number of observations for PSR(benchmark) to reach ``confidence``.

    MinTRL = 1 + (1 - skew*SR + (kurtosis-1)/4 * SR^2) * (Zinv(confidence) / (SR - SR*))^2

    (Bailey and Lopez de Prado 2012, eq. 13.) The result counts observations at the
    frequency of ``sharpe``; divide by periods per year for years. It is infinite
    when the observed Sharpe ratio does not exceed the benchmark.
    """
    k = _resolve_kurtosis(kurtosis, excess_kurtosis)
    _check_finite(sharpe=sharpe, benchmark=benchmark, skew=skew, confidence=confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if sharpe <= benchmark:
        return math.inf
    v = _variance_term(sharpe, skew, k)
    return 1.0 + v * (_Z.inv_cdf(confidence) / (sharpe - benchmark)) ** 2


def min_backtest_length(n_trials: float, max_sharpe: float = 1.0) -> float:
    """Minimum backtest length, in years, so that the best of ``n_trials`` zero-skill
    strategies is not expected to show an annualised Sharpe ratio above ``max_sharpe``.

    MinBTL = ( ((1 - g) * Zinv(1 - 1/N) + g * Zinv(1 - 1/(N e))) / E[max_N] )^2

    (Bailey, Borwein, Lopez de Prado and Zhu 2014, Theorem 3.1.) Here
    ``max_sharpe`` is the annualised Sharpe ratio a researcher is willing to see
    by chance. The theorem assumes independent trials and unit-variance
    annualised Sharpe ratios per year of data.
    """
    _check_finite(n_trials=n_trials, max_sharpe=max_sharpe)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if max_sharpe <= 0:
        raise ValueError("max_sharpe must be positive")
    if n_trials == 1:
        return 0.0
    return (expected_max_sharpe(n_trials, 1.0) / max_sharpe) ** 2
