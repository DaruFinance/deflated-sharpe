# deflated-sharpe

[![tests](https://github.com/DaruFinance/deflated-sharpe/actions/workflows/ci.yml/badge.svg)](https://github.com/DaruFinance/deflated-sharpe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

The Deflated Sharpe Ratio, the Probabilistic Sharpe Ratio, the minimum track record length and the minimum backtest length of Bailey and López de Prado, in pure Python with no dependencies. Every published numerical example from the three source papers is reproduced as a test.

A backtest that looks good after many configurations were tried has been selected, and the best of many zero-skill strategies always looks good. The Deflated Sharpe Ratio estimates the probability that the selected strategy's true Sharpe ratio is above zero once the number of trials, the spread of their results, the length of the sample and the skewness and fat tails of its returns are taken into account.

## Install

```bash
pip install git+https://github.com/DaruFinance/deflated-sharpe
```

Python 3.10 or newer. Nothing else is installed.

## Quick start

The numerical example from Bailey and López de Prado (2014): a strategy with an annualised Sharpe ratio of 2.5 on five years of daily returns, selected from 100 configurations whose annualised Sharpe ratios have variance 0.5, with skewness −3 and kurtosis 10.

```python
from deflated_sharpe import deflated_sharpe_ratio, to_period_sharpe, to_period_variance

result = deflated_sharpe_ratio(
    sharpe=to_period_sharpe(2.5, 250),       # per-day Sharpe ratio
    n_obs=1250,                              # five years of daily returns
    n_trials=100,                            # configurations tried
    var_trials=to_period_variance(0.5, 250), # variance of their Sharpe ratios, per day
    skew=-3,
    kurtosis=10,                             # raw kurtosis: normal = 3
)
result.dsr          # 0.9004
result.sr0          # 0.1132, the Sharpe ratio the best of 100 null trials is expected to show
result.passes(0.95) # False
```

An annualised Sharpe ratio of 2.5 gives no more than 90% confidence of any skill after 100 trials with these return characteristics. The same strategy would pass at 95% had it been the best of 46 trials.

From return data directly:

```python
from deflated_sharpe import deflated_sharpe_from_returns, sharpe_ratio

trial_sharpes = [sharpe_ratio(r) for r in all_trial_returns]  # every configuration tried
result = deflated_sharpe_from_returns(selected_returns, trial_sharpes)
```

From the command line:

```console
$ dsr deflate --sharpe 2.5 --annual --periods-per-year 250 --n-obs 1250 \
      --trials 100 --trial-variance 0.5 --skew -3 --kurtosis 10
Deflated Sharpe Ratio   0.9004
Rejection threshold SR0 0.1132 per period (1.789 annualised)
PSR against zero        1.0000
Verdict at 95%           does not pass
```

## What it computes

| function | question it answers | source |
|---|---|---|
| `probabilistic_sharpe_ratio` | How likely is the true Sharpe ratio to exceed a benchmark? | Bailey and López de Prado (2012) |
| `min_track_record_length` | How many observations until that probability reaches a confidence level? | Bailey and López de Prado (2012) |
| `expected_max_sharpe` | What Sharpe ratio does the best of N zero-skill trials show? | Bailey et al. (2014), Bailey and López de Prado (2014) |
| `deflated_sharpe_ratio` | How likely is the selected strategy's true Sharpe ratio above zero, given the selection? | Bailey and López de Prado (2014) |
| `min_backtest_length` | How many years of data before N trials stop producing a spurious Sharpe ratio? | Bailey et al. (2014) |
| `return_stats`, `sharpe_ratio`, `trial_sharpe_variance` | The inputs above, from return series | |
| `to_period_sharpe`, `to_period_variance`, `ar1_effective_obs` | Unit conversions and a serial-correlation adjustment | |

The formulas, their assumptions and the references are in [docs/methodology.md](docs/methodology.md).

## Getting the inputs right

Most wrong DSRs come from inputs, not arithmetic. Check these before trusting a result.

1. **Per-period units.** Every Sharpe ratio, and the variance of the trial Sharpe ratios, is per observation. Convert annualised values with `to_period_sharpe` and `to_period_variance`. The variance scales with periods per year, not its square root.
2. **Raw kurtosis.** Normal is 3. `scipy.stats.kurtosis` and `pandas.Series.kurt` return excess kurtosis (normal is 0); pass those through `excess_kurtosis=`.
3. **The distinct number of trials.** Count every configuration evaluated, including abandoned ones. Do not replace it with a correlation-adjusted effective count when the trial variance comes from the trials themselves: that double-counts the correlation and lets most zero-skill corpora through. [docs/trial-count.md](docs/trial-count.md) shows why, with a simulation anyone can rerun.
4. **Every trial in the variance.** The spread of Sharpe ratios must include the losers.
5. **Independent returns.** Positions held across bars and overlapping windows make returns autocorrelated. `ar1_effective_obs` gives a first-order correction for `n_obs`.

[docs/pitfalls.md](docs/pitfalls.md) covers each of these in detail.

## Validation

Every numerical example printed in the source papers is a test.

| example | paper | package |
|---|---|---|
| DSR, 100 trials, skew −3, kurtosis 10 | SR₀ ≈ 0.1132, DSR = 0.9004 | 0.1132, 0.9004 |
| DSR, same strategy after 46 trials | 0.9505 | 0.9505 |
| DSR, normal returns after 88 trials | 0.9505 | 0.9505 |
| PSR, monthly SR 0.458 over 24 months, normal | 0.982 | 0.982 |
| PSR, same, skew −2.448, kurtosis 10.164 | 0.913 | 0.913 |
| PSR, same moments over 36 months | 0.953 | 0.9535 |
| MinTRL, SR 2 vs 1 at 95%, daily / weekly / monthly | 2.73 / 2.83 / 3.24 years | 2.73 / 2.83 / 3.24 |
| MinTRL, monthly, skew −0.72, kurtosis 5.78 | 4.99 years | 4.99 |
| Expected maximum of 10 standard normals | 1.57 | 1.575 |
| MinBTL, trials allowed by five years of data | 45 | 45 |

Monte Carlo tests then check the behaviour the formulas promise. PSR crosses 0.95 on about 5% of zero-skill samples and the expected maximum matches simulated maxima, while MinTRL lands exactly where PSR reaches its confidence level. The best strategy in a zero-skill corpus also clears `SR₀` about half the time, whether or not the trials are correlated.

```bash
pip install -e ".[dev]"
pytest
```

The suite runs in a few seconds on Python 3.10 to 3.13.

## Documentation

- [Methodology](docs/methodology.md): formulas, assumptions and references
- [Choosing the number of trials](docs/trial-count.md)
- [Pitfalls](docs/pitfalls.md)
- [Command-line reference](docs/cli.md)
- [Examples](examples/)

## Citing

If this package is used in research, please cite the software and the papers whose methods it implements. Citation metadata is in [CITATION.cff](CITATION.cff), which GitHub renders under "Cite this repository".

```bibtex
@software{gatto_deflated_sharpe,
  author  = {Gatto, Daniel V.},
  title   = {deflated-sharpe: Deflated and Probabilistic Sharpe Ratio in Python},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/DaruFinance/deflated-sharpe},
  license = {MIT}
}

@article{bailey2014deflated,
  author  = {Bailey, David H. and L{\'o}pez de Prado, Marcos},
  title   = {The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality},
  journal = {Journal of Portfolio Management},
  volume  = {40},
  number  = {5},
  pages   = {94--107},
  year    = {2014},
  doi     = {10.3905/jpm.2014.40.5.094}
}

@article{bailey2012frontier,
  author  = {Bailey, David H. and L{\'o}pez de Prado, Marcos},
  title   = {The Sharpe Ratio Efficient Frontier},
  journal = {Journal of Risk},
  volume  = {15},
  number  = {2},
  pages   = {3--44},
  year    = {2012},
  doi     = {10.21314/JOR.2012.255}
}

@article{bailey2014pseudo,
  author  = {Bailey, David H. and Borwein, Jonathan M. and L{\'o}pez de Prado, Marcos and Zhu, Qiji Jim},
  title   = {Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance},
  journal = {Notices of the American Mathematical Society},
  volume  = {61},
  number  = {5},
  pages   = {458--471},
  year    = {2014},
  doi     = {10.1090/noti1105}
}
```

## License

MIT. See [LICENSE](LICENSE).
