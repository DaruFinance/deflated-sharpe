"""Converting Sharpe ratios and their variances between annual and per-period units.

A Sharpe ratio scales with the square root of the number of periods per year, so
its variance across trials scales linearly. The conversions assume serially
independent returns, which is also what the square-root rule assumes.
"""

from __future__ import annotations

import math

__all__ = [
    "PERIODS_PER_YEAR",
    "to_period_sharpe",
    "to_annual_sharpe",
    "to_period_variance",
    "to_annual_variance",
    "ar1_effective_obs",
]

PERIODS_PER_YEAR = {
    "daily": 252,
    "calendar_daily": 365,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}
"""Common conventions. The numerical example in Bailey and Lopez de Prado (2014)
uses 250 trading days; pass the number your data actually has."""


def _ppy(periods_per_year: float) -> float:
    if not (isinstance(periods_per_year, (int, float)) and math.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValueError("periods_per_year must be a positive number")
    return float(periods_per_year)


def to_period_sharpe(annual_sharpe: float, periods_per_year: float) -> float:
    """Annualised Sharpe ratio to per-period: ``SR / sqrt(periods_per_year)``."""
    return annual_sharpe / math.sqrt(_ppy(periods_per_year))


def to_annual_sharpe(period_sharpe: float, periods_per_year: float) -> float:
    """Per-period Sharpe ratio to annualised: ``SR * sqrt(periods_per_year)``."""
    return period_sharpe * math.sqrt(_ppy(periods_per_year))


def to_period_variance(annual_variance: float, periods_per_year: float) -> float:
    """Variance of annualised Sharpe ratios across trials to per-period units."""
    return annual_variance / _ppy(periods_per_year)


def to_annual_variance(period_variance: float, periods_per_year: float) -> float:
    """Variance of per-period Sharpe ratios across trials to annualised units."""
    return period_variance * _ppy(periods_per_year)


def ar1_effective_obs(n_obs: float, phi: float) -> float:
    """Effective number of independent observations for AR(1) returns.

    ``n * (1 - phi) / (1 + phi)``, the sample size at which the variance of the
    mean of independent returns matches that of ``n`` returns with lag-1
    autocorrelation ``phi``. PSR, DSR and MinTRL assume independent returns;
    passing this as ``n_obs`` is a first-order correction when returns are
    positively autocorrelated, as they are for strategies that hold positions
    across bars. It is an approximation, not an exact result for the Sharpe
    ratio (see Lo 2002 for the full treatment).
    """
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie strictly between -1 and 1; |phi| >= 1 has no finite effective sample")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    return n_obs * (1.0 - phi) / (1.0 + phi)
