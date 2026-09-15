# Methodology

This page states every formula the package implements, the assumptions behind it and the source it comes from. All Sharpe ratios below are **per period**: the mean of per-period returns divided by their standard deviation, with no annualisation.

Notation: `T` is the number of return observations, `γ₃` the sample skewness, `γ₄` the sample raw kurtosis (3 for a normal distribution) and `Z` the standard normal CDF, with `Z⁻¹` its inverse.

## Probabilistic Sharpe Ratio

Bailey and López de Prado (2012) show that, for independent and identically distributed returns with finite fourth moment, the estimated Sharpe ratio is asymptotically normal around the true value with variance

```
V[SR] ≈ (1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / (T − 1)
```

The Probabilistic Sharpe Ratio is the probability, under that distribution, that the true Sharpe ratio exceeds a benchmark `SR*`:

```
PSR(SR*) = Z[ (SR − SR*) · √(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) ]
```

Negative skewness and fat tails widen the distribution and lower PSR for a positive Sharpe ratio. With normal returns (`γ₃ = 0`, `γ₄ = 3`) the factor reduces to `1 + SR²/2`, the classical result of Lo (2002) for IID returns.

`probabilistic_sharpe_ratio()` implements this equation, and `sharpe_ratio_std_error()` returns the square root of the variance above.

## Minimum track record length

Solving `PSR(SR*) = 1 − α` for `T` gives the number of observations needed before the estimated Sharpe ratio is significantly above the benchmark (Bailey and López de Prado 2012, eq. 13):

```
MinTRL = 1 + (1 − γ₃·SR + (γ₄ − 1)/4 · SR²) · ( Z⁻¹(1 − α) / (SR − SR*) )²
```

It is measured in observations at the frequency of the Sharpe ratio. When the observed Sharpe ratio does not exceed the benchmark no track record is long enough, and `min_track_record_length()` returns infinity. The authors note that the asymptotic approximation needs samples of at least about 30 observations regardless of what MinTRL returns.

## Expected maximum Sharpe ratio

If `N` independent strategies with no skill are tested, and their estimated Sharpe ratios have variance `V`, the best of them is expected to show (Bailey, Borwein, López de Prado and Zhu 2014, Proposition 2.1; Bailey and López de Prado 2014, eq. 1)

```
E[max SR] ≈ E[SR] + √V · ( (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) )
```

where `γ ≈ 0.5772` is the Euler–Mascheroni constant. Under the null hypothesis of no skill the mean `E[SR]` is zero, and `expected_max_sharpe()` uses zero by default.

The approximation is asymptotic in `N`. Against simulation it overstates the mean maximum of `N` standard normals by 2.5% at `N = 10`, 0.9% at `N = 100` and 0.8% at `N = 1,000` (20,000 draws at `N` = 10 and 100, 4,000 at 1,000), so at small `N` it is slightly conservative.

It is an expected value, not a quantile. Under the null the best of `N` trials lands above it roughly half the time, which is also the test this package uses to check the input conventions (see [trial-count.md](trial-count.md)).

## Deflated Sharpe Ratio

The Deflated Sharpe Ratio (Bailey and López de Prado 2014, eq. 2) is the Probabilistic Sharpe Ratio with the benchmark set to the expected maximum of the trials:

```
SR₀ = √V[{SRₙ}] · ( (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) )
DSR = PSR(SR₀)
```

It answers the question: given that this strategy was the best of `N` tried, how likely is it that its true Sharpe ratio is above zero? It corrects for two separate sources of inflation at once, the selection of a maximum and the non-normality of returns.

`deflated_sharpe_ratio()` returns a `DSRResult` holding `dsr`, the threshold `sr0`, the undeflated `psr_zero` for comparison and every input.

### The five inputs

| input | meaning | where it comes from |
|---|---|---|
| `sharpe` | per-period Sharpe ratio of the selected strategy | its return series |
| `n_obs` | number of returns behind that Sharpe ratio | its return series |
| `skew`, `kurtosis` | third and fourth moments of those returns | its return series |
| `n_trials` | number of distinct configurations it was selected from | the research log |
| `var_trials` | variance of the per-period Sharpe ratios of all those configurations | every trial's return series |

`var_trials` must be in the same per-period units as `sharpe`. An annualised variance of `V` becomes `V / periods_per_year` (see `to_period_variance()`).

## Minimum backtest length

Rescaling Proposition 2.1 to annualised Sharpe ratios estimated on `y` years of data gives an expected maximum of `y^(−1/2)` times the bracket above, and solving for `y` gives Theorem 3.1 of Bailey, Borwein, López de Prado and Zhu (2014):

```
MinBTL ≈ ( ((1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))) / E[max_N] )²  <  2·ln(N) / E[max_N]²
```

`E[max_N]` is the annualised Sharpe ratio a researcher is prepared to see by chance, 1 by default in `min_backtest_length()`. With five years of data, no more than 45 independent configurations can be tried before the best one is expected to show an annualised Sharpe ratio of 1 with no skill behind it.

The theorem assumes independent trials whose annualised Sharpe ratios have unit variance per year of data, which holds for normal returns with zero true Sharpe ratio, and it gives a necessary condition against overfitting rather than a sufficient one.

## Moments from returns

`return_stats()` computes the plain sample moments the papers use:

```
mean = Σx / T
m₂ = Σ(x − mean)² / T,   m₃ = Σ(x − mean)³ / T,   m₄ = Σ(x − mean)⁴ / T
SR = mean / s,   s = √(m₂ · T / (T − ddof))
γ₃ = m₃ / m₂^1.5,   γ₄ = m₄ / m₂²
```

Skewness and kurtosis carry no small-sample bias correction. The standard deviation in the Sharpe ratio uses `ddof = 1` by default. At the sample sizes these statistics need, the choice changes the Sharpe ratio by a factor of `√(T/(T−1))`, well inside the estimate's own standard error.

## Assumptions shared by every formula

- **Serially independent returns.** Positive autocorrelation shrinks the information in a sample. See [pitfalls.md](pitfalls.md#serial-correlation) for a first-order correction.
- **Stationary returns.** Moments estimated over a sample that mixes regimes describe the mixture.
- **Asymptotic normality of the Sharpe ratio estimator.** Short samples, and very large Sharpe ratios combined with strong positive skew, fall outside it. The package raises an error when `1 − γ₃·SR + (γ₄ − 1)/4 · SR²` is not positive, which usually signals an annualised Sharpe ratio passed where a per-period one belongs.

## References

- Bailey, D. H. and M. López de Prado (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2), 3–44. [SSRN 1821643](https://ssrn.com/abstract=1821643)
- Bailey, D. H. and M. López de Prado (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management* 40(5), 94–107. [SSRN 2460551](https://ssrn.com/abstract=2460551)
- Bailey, D. H., J. M. Borwein, M. López de Prado and Q. J. Zhu (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the American Mathematical Society* 61(5), 458–471. [SSRN 2308659](https://ssrn.com/abstract=2308659)
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4), 36–52.
