"""Computing the inputs to PSR and DSR from return series.

These helpers use the plain sample moments the original papers use: the Sharpe
ratio is mean over standard deviation (with ``ddof`` selectable), and skewness
and kurtosis are the moment ratios m3 / m2^1.5 and m4 / m2^2 without small-sample
bias correction. Any sequence of floats works; NumPy arrays and pandas Series are
accepted but not required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .core import DSRResult, deflated_sharpe_ratio

__all__ = ["ReturnStats", "return_stats", "sharpe_ratio", "trial_sharpe_variance", "deflated_sharpe_from_returns"]


def _clean(returns: Iterable[float]) -> list:
    xs = [float(x) for x in returns]
    bad = [x for x in xs if not math.isfinite(x)]
    if bad:
        raise ValueError(f"returns contain {len(bad)} non-finite value(s); drop or fill them explicitly first")
    return xs


@dataclass(frozen=True)
class ReturnStats:
    """Per-period moments of one return series."""

    n_obs: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float
    """Raw (Pearson) kurtosis: 3 for a normal distribution."""

    @property
    def excess_kurtosis(self) -> float:
        return self.kurtosis - 3.0


def return_stats(returns: Iterable[float], risk_free: float = 0.0, ddof: int = 1) -> ReturnStats:
    """Sharpe ratio, skewness and raw kurtosis of a per-period return series.

    ``risk_free`` is subtracted from every return and must be in the same
    per-period units. ``ddof`` sets the standard deviation used for the Sharpe
    ratio (1 for the sample standard deviation, 0 for the population form).
    """
    xs = [x - risk_free for x in _clean(returns)]
    n = len(xs)
    if n < 3:
        raise ValueError("need at least 3 returns")
    mean = sum(xs) / n
    dev = [x - mean for x in xs]
    m2 = sum(d * d for d in dev) / n
    if m2 == 0.0:
        raise ValueError("returns have zero variance; the Sharpe ratio is undefined")
    m3 = sum(d ** 3 for d in dev) / n
    m4 = sum(d ** 4 for d in dev) / n
    std = math.sqrt(m2 * n / (n - ddof))
    return ReturnStats(
        n_obs=n, mean=mean, std=std, sharpe=mean / std,
        skew=m3 / m2 ** 1.5, kurtosis=m4 / (m2 * m2),
    )


def sharpe_ratio(returns: Iterable[float], risk_free: float = 0.0, ddof: int = 1) -> float:
    """Per-period Sharpe ratio of a return series."""
    return return_stats(returns, risk_free, ddof).sharpe


def trial_sharpe_variance(trial_sharpes: Sequence[float]) -> float:
    """Sample variance (ddof = 1) of the Sharpe ratios of every trial.

    Pass the Sharpe ratios of all the configurations tried, including the losers
    and the one finally selected, in the same units as the selected Sharpe ratio.
    """
    xs = _clean(trial_sharpes)
    if len(xs) < 2:
        raise ValueError("need at least 2 trial Sharpe ratios to estimate their variance")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def deflated_sharpe_from_returns(
    returns: Iterable[float],
    trial_sharpes: Sequence[float],
    n_trials: Optional[float] = None,
    risk_free: float = 0.0,
) -> DSRResult:
    """DSR of a selected strategy, given its returns and the Sharpe ratios of all trials.

    ``trial_sharpes`` must be per-period Sharpe ratios computed on the same
    frequency as ``returns``. ``n_trials`` defaults to ``len(trial_sharpes)``;
    override it only to collapse exact duplicates or to count trials whose Sharpe
    ratios were not kept (see ``docs/trial-count.md``).
    """
    st = return_stats(returns, risk_free)
    n = float(len(trial_sharpes)) if n_trials is None else float(n_trials)
    return deflated_sharpe_ratio(
        sharpe=st.sharpe, n_obs=st.n_obs, n_trials=n,
        var_trials=trial_sharpe_variance(trial_sharpes), skew=st.skew, kurtosis=st.kurtosis,
    )
