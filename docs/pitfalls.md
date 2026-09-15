# Pitfalls

Each of these produces a number that looks reasonable and is wrong.

## Annualised values in per-period formulas

Every formula is written for per-period Sharpe ratios and per-period observation counts, so an annualised Sharpe ratio of 2 on daily data has to be passed as about 0.126 per day.

Passing 2 as the Sharpe ratio makes the non-normality term explode, and the package raises an error with a hint when that term turns non-positive, although it cannot catch every case. Explicit conversion before calling any function closes the gap:

```python
from deflated_sharpe import to_period_sharpe, to_period_variance

sr_daily = to_period_sharpe(2.0, 252)
var_daily = to_period_variance(0.5, 252)  # variance of annualised Sharpe ratios across trials
```

The variance of the trial Sharpe ratios scales with the number of periods per year, not its square root, which makes converting it with a square root a common slip.

Use the number of periods your data actually has. Crypto markets trade 365 days a year, and many equity datasets have fewer than 252 sessions in some years.

## Excess kurtosis passed as raw kurtosis

The formulas use raw kurtosis, 3 for a normal distribution, while most libraries report excess kurtosis by default: `scipy.stats.kurtosis` and `pandas.Series.kurt` both return 0 for a normal distribution. Passing excess kurtosis as raw understates tail risk by 3 and inflates PSR.

Use the `excess_kurtosis=` argument when the value is excess. A raw kurtosis below 1 is impossible for any distribution, so the package rejects it with a message pointing at this mistake, but an excess kurtosis above 1 passed as raw passes silently.

## Serial correlation

The standard error of the Sharpe ratio assumes independent returns. Strategies that hold positions across several bars produce positively autocorrelated returns, as do trend followers and any signal built from overlapping windows, and those returns carry less information than their count suggests.

For lag-1 autocorrelation `φ`, the variance of a sample mean is larger by `(1 + φ)/(1 − φ)`, so the zero-skill noise floor of a Sharpe ratio is wider by the square root of that factor:

| φ | 0.1 | 0.3 | 0.5 | 0.8 |
|---|---:|---:|---:|---:|
| floor multiplier | 1.11 | 1.36 | 1.73 | 3.00 |

A first-order correction is to pass the effective sample size as `n_obs`:

```python
from deflated_sharpe import ar1_effective_obs, probabilistic_sharpe_ratio

n_eff = ar1_effective_obs(n_obs=1260, phi=0.3)  # 678
psr = probabilistic_sharpe_ratio(sr, n_eff, skew, kurt)
```

This is an approximation for AR(1) returns. Lo (2002) derives the full correction for general autocorrelation. At `φ` close to 1 the returns behave like a random walk and no finite sample gives a reliable estimate.

## Overlapping evaluation windows

Sharpe ratios computed on rolling or overlapping windows are not independent draws. Averaging them does not shrink the error as the number of windows grows and their spread understates the true uncertainty, so the statistics belong on one non-overlapping series of returns.

## An incomplete set of trials

`var_trials` must come from every configuration tried, including the ones that performed badly. Discarding losers before measuring the variance narrows it and lowers `SR₀`. See [trial-count.md](trial-count.md).

## Treating the DSR as a certificate

A DSR above 0.95 says the selection is unlikely to be explained by the number of trials and the shape of the returns alone, which falls short of saying the strategy will perform out of sample. It does not account for transaction costs, capacity, regime change, look-ahead in the data or trials that were never recorded.

Nor does a DSR below 0.95 prove the absence of skill, particularly on short samples, so the statistic is one input to a decision rather than the decision itself.

The statistic applies to a single selected strategy. For a portfolio assembled from many strategies the selection step that matters is the assembly itself, where the relevant `N` is the count of candidates considered at that step.
