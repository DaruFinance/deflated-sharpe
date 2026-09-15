# Choosing the number of trials

`n_trials` is the input most often set wrong, and the error is silent: a DSR computed with the wrong count looks exactly as authoritative as one computed with the right count.

## What counts as a trial

A trial is one strategy configuration whose performance was observed before the final one was chosen. Every parameter value on a grid, every signal variant, every universe filter and every period that was looked at and then discarded is a trial, including configurations abandoned early, because they informed the choice.

The Sharpe ratios of all those trials, winners and losers, feed `var_trials`. Keeping only the survivors understates the variance and lowers the bar.

When the research log is incomplete, a conservative upper estimate of the count is better than a guess from memory, because the bar grows only logarithmically with `N`: doubling the count moves `SR₀` far less than halving the variance does.

## Distinct strategies, not an effective count

Strategy variants are usually correlated, and it is tempting to replace `N` by an "effective number of independent trials", for example the eigenvalue participation ratio of the strategies' return correlation matrix. When `var_trials` is estimated from the trials' own Sharpe ratios, that substitution corrects for correlation twice.

Where the double counting comes from is the variance term. Write each trial's standardised return as a common component shared by all of them plus an independent part:

```
xᵢ = √ρ · common + √(1 − ρ) · εᵢ
```

Because the common component moves every trial's Sharpe ratio by the same amount, it adds nothing to their spread across trials. The observed variance of the trial Sharpe ratios is already about `(1 − ρ)` times what independent trials would show, while the maximum is `√(1 − ρ)` times the maximum of the independent parts.

Correlation therefore sits inside `√V` already, and shrinking `N` as well lowers the bar a second time.

`examples/trial_count_calibration.py` checks this directly. It builds 300 corpora of 60 strategies with no skill under four correlation structures and counts how often the best strategy in each corpus clears `SR₀`, scoring both bars on the same corpora.

Because `SR₀` is an expected maximum, a correctly built bar is cleared about half the time.

| structure | participation ratio | bar from distinct count | bar from participation ratio |
|---|---:|---:|---:|
| independent | 60.0 | 45% | 45% |
| equicorrelated, ρ = 0.2 | 17.9 | 52% | 79% |
| equicorrelated, ρ = 0.5 | 3.8 | 44% | 85% |
| 6 families plus a market factor | 20.4 | 50% | 73% |

The distinct count stays near one half across every structure. The participation ratio lets the best null strategy through three to four times out of four once the strategies are correlated.

## When a smaller count is right

- **Exact duplicates.** Two configurations that produce identical returns are one trial. Collapse clones before counting.
- **A variance that was not estimated from the trials.** If `var_trials` comes from theory rather than from the observed spread of the trials, for example the unit variance assumed by the minimum backtest length, the correlation is no longer inside `V`, and the source papers suggest reducing `N` with a dimension-reduction method such as principal components. The same count cannot be carried back into a DSR whose variance is estimated from the trials.

## A check for any counting rule

Simulate a corpus of zero-skill strategies with the same size, sample length and correlation structure as the real research, then count how often its best strategy clears the bar. A rule that nothing null can clear is too strict while a rule that almost everything null clears is too lenient, and a rate near one half means the rule is calibrated.

## Estimating a participation ratio correctly

A participation ratio is still a useful redundancy diagnostic, for example to judge how much diversification a portfolio of strategies offers, provided it is computed on the per-strategy return series. Computed on a matrix of per-window Sharpe ratios it cannot exceed the number of windows whatever the number of strategies, so it reports a count close to the window count for any pool.
