# Command-line reference

Installing the package adds a `dsr` command. `python -m deflated_sharpe` runs the same program.

Sharpe ratios are per period by default. With `--annual --periods-per-year P`, every Sharpe ratio and the trial variance given on the command line are read as annualised and converted, and every command accepts `--json` for machine-readable output.

Kurtosis is raw by default (`--kurtosis`, normal = 3). Use `--excess-kurtosis` for values where normal = 0.

## `dsr deflate`

Deflated Sharpe Ratio from summary statistics.

```console
$ dsr deflate --sharpe 2.5 --annual --periods-per-year 250 --n-obs 1250 \
      --trials 100 --trial-variance 0.5 --skew -3 --kurtosis 10
Deflated Sharpe Ratio   0.9004
Rejection threshold SR0 0.1132 per period (1.789 annualised)
PSR against zero        1.0000
Verdict at 95%           does not pass
```

| option | meaning |
|---|---|
| `--sharpe` | Sharpe ratio of the selected strategy |
| `--n-obs` | number of return observations |
| `--trials` | number of distinct configurations tried |
| `--trial-variance` | variance of Sharpe ratios across all trials |
| `--skew`, `--kurtosis` / `--excess-kurtosis` | return moments |
| `--confidence` | level for the verdict, default 0.95 |

## `dsr psr`

```console
$ dsr psr --sharpe 0.458 --n-obs 24 --skew -2.448 --kurtosis 10.164
Probabilistic Sharpe Ratio 0.9134
```

`--benchmark` sets the Sharpe ratio to test against (default 0).

## `dsr mintrl`

```console
$ dsr mintrl --sharpe 2 --benchmark 1 --annual --periods-per-year 252
Minimum track record length 688.2 observations (2.73 years)
```

## `dsr minbtl`

```console
$ dsr minbtl --trials 45
Minimum backtest length 5.00 years (upper bound 2 ln N / E[max]^2 = 7.61)
```

`--max-sharpe` sets the annualised Sharpe ratio acceptable by chance (default 1).

## `dsr returns`

Compute everything from data: a CSV of the selected strategy's per-period returns and a text file with one Sharpe ratio per line for every trial.

```console
$ dsr returns strategy.csv --column ret --trials-file trial_sharpes.txt --periods-per-year 252
```

| option | meaning |
|---|---|
| `returns_csv` | CSV with one column of returns, or several columns and `--column` |
| `--column` | name of the returns column |
| `--trials-file` | one Sharpe ratio per line, same frequency as the returns |
| `--trials-annual` | the trial Sharpe ratios are annualised (needs `--periods-per-year`) |
| `--n-trials` | override the count, default the number of lines in `--trials-file` |

A CSV without a header is read as numbers from its first column, or its last column when there are several.

## Exit codes

`0` on success, `2` for invalid inputs, with the reason on standard error.
