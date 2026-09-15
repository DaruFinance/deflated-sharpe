"""Every function in the package on one small example.

Run:  python examples/quickstart.py
"""

import random

from deflated_sharpe import (
    deflated_sharpe_from_returns,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_ratio,
    to_annual_sharpe,
)

rng = random.Random(42)
PPY = 252

# Forty variants of a strategy, three years of daily returns each, none with skill
# except the last one, which has a small true edge.
trials = [[rng.gauss(0.0, 0.01) for _ in range(3 * PPY)] for _ in range(39)]
trials.append([rng.gauss(0.0008, 0.01) for _ in range(3 * PPY)])

sharpes = [sharpe_ratio(t) for t in trials]
best = max(range(len(trials)), key=lambda i: sharpes[i])
st = return_stats(trials[best])

print(f"best of {len(trials)} trials: annualised Sharpe {to_annual_sharpe(st.sharpe, PPY):.2f}")
print(f"PSR against zero, ignoring selection: {probabilistic_sharpe_ratio(st.sharpe, st.n_obs, st.skew, st.kurtosis):.3f}")

result = deflated_sharpe_from_returns(trials[best], sharpes)
print(f"Deflated Sharpe Ratio: {result.dsr:.3f} (threshold {to_annual_sharpe(result.sr0, PPY):.2f} annualised)")

n = min_track_record_length(st.sharpe, 0.0, st.skew, st.kurtosis)
print(f"track record needed at 95%: {n / PPY:.1f} years")
print(f"backtest length needed for {len(trials)} trials: {min_backtest_length(len(trials)):.1f} years")
