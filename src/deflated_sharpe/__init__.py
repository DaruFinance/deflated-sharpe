"""Deflated Sharpe Ratio, Probabilistic Sharpe Ratio and the track-record and
backtest-length bounds of Bailey and Lopez de Prado, with no dependencies.

All Sharpe ratios are per period unless a function name says otherwise; see
:mod:`deflated_sharpe.frequency` for conversions.
"""

from .core import (
    EULER_MASCHERONI,
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio_std_error,
)
from .frequency import (
    PERIODS_PER_YEAR,
    ar1_effective_obs,
    to_annual_sharpe,
    to_annual_variance,
    to_period_sharpe,
    to_period_variance,
)
from .returns import (
    ReturnStats,
    deflated_sharpe_from_returns,
    return_stats,
    sharpe_ratio,
    trial_sharpe_variance,
)

__version__ = "0.1.0"

__all__ = [
    "EULER_MASCHERONI",
    "DSRResult",
    "PERIODS_PER_YEAR",
    "ReturnStats",
    "__version__",
    "ar1_effective_obs",
    "deflated_sharpe_from_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_backtest_length",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "return_stats",
    "sharpe_ratio",
    "sharpe_ratio_std_error",
    "to_annual_sharpe",
    "to_annual_variance",
    "to_period_sharpe",
    "to_period_variance",
    "trial_sharpe_variance",
]
