"""Command-line interface: ``dsr <command> ...``.

Commands
--------
deflate   Deflated Sharpe Ratio from summary statistics
psr       Probabilistic Sharpe Ratio from summary statistics
mintrl    Minimum track record length
minbtl    Minimum backtest length
returns   DSR from a CSV of returns and a file of trial Sharpe ratios

Sharpe ratios are per period unless ``--annual`` is given together with
``--periods-per-year``, in which case every Sharpe ratio and the trial variance
on the command line are read as annualised and converted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import List, Optional, Sequence

from . import __version__
from .core import (
    deflated_sharpe_ratio,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from .frequency import to_annual_sharpe, to_period_sharpe, to_period_variance
from .returns import deflated_sharpe_from_returns


def _add_moments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skew", type=float, default=0.0, help="skewness of returns (default 0)")
    k = p.add_mutually_exclusive_group()
    k.add_argument("--kurtosis", type=float, default=None, help="raw kurtosis, normal = 3 (default 3)")
    k.add_argument("--excess-kurtosis", type=float, default=None, help="excess kurtosis, normal = 0")


def _add_frequency(p: argparse.ArgumentParser) -> None:
    p.add_argument("--annual", action="store_true", help="read Sharpe ratios and trial variance as annualised")
    p.add_argument("--periods-per-year", type=float, default=None, help="observations per year, e.g. 252 or 12")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")


def _period(args: argparse.Namespace, sharpe: float) -> float:
    if args.annual:
        if args.periods_per_year is None:
            raise SystemExit("--annual needs --periods-per-year")
        return to_period_sharpe(sharpe, args.periods_per_year)
    return sharpe


def _emit(args: argparse.Namespace, payload: dict, lines: List[str]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(lines))


def _fmt_obs(n: float, ppy: Optional[float]) -> str:
    if math.isinf(n):
        return "infinite (the Sharpe ratio does not exceed the benchmark)"
    s = f"{n:,.1f} observations"
    if ppy:
        s += f" ({n / ppy:.2f} years)"
    return s


def cmd_deflate(args: argparse.Namespace) -> int:
    sr = _period(args, args.sharpe)
    var = to_period_variance(args.trial_variance, args.periods_per_year) if args.annual else args.trial_variance
    r = deflated_sharpe_ratio(
        sharpe=sr, n_obs=args.n_obs, n_trials=args.trials, var_trials=var,
        skew=args.skew, kurtosis=args.kurtosis, excess_kurtosis=args.excess_kurtosis,
    )
    ppy = args.periods_per_year
    payload = {
        "dsr": r.dsr, "sr0_per_period": r.sr0, "sharpe_per_period": r.sharpe, "psr_zero": r.psr_zero,
        "n_obs": r.n_obs, "n_trials": r.n_trials, "var_trials_per_period": r.var_trials,
        "skew": r.skew, "kurtosis": r.kurtosis, "confidence": args.confidence,
        "passes": r.passes(args.confidence),
    }
    if ppy:
        payload["sr0_annual"] = to_annual_sharpe(r.sr0, ppy)
        payload["sharpe_annual"] = to_annual_sharpe(r.sharpe, ppy)
    thr = f"{r.sr0:.4f} per period" + (f" ({to_annual_sharpe(r.sr0, ppy):.3f} annualised)" if ppy else "")
    lines = [
        f"Deflated Sharpe Ratio   {r.dsr:.4f}",
        f"Rejection threshold SR0 {thr}",
        f"PSR against zero        {r.psr_zero:.4f}",
        f"Verdict at {args.confidence:.0%}           {'passes' if r.passes(args.confidence) else 'does not pass'}",
    ]
    _emit(args, payload, lines)
    return 0


def cmd_psr(args: argparse.Namespace) -> int:
    sr = _period(args, args.sharpe)
    bench = _period(args, args.benchmark)
    p = probabilistic_sharpe_ratio(sr, args.n_obs, args.skew, args.kurtosis, bench, excess_kurtosis=args.excess_kurtosis)
    _emit(args, {"psr": p, "sharpe_per_period": sr, "benchmark_per_period": bench, "n_obs": args.n_obs},
          [f"Probabilistic Sharpe Ratio {p:.4f}"])
    return 0


def cmd_mintrl(args: argparse.Namespace) -> int:
    sr = _period(args, args.sharpe)
    bench = _period(args, args.benchmark)
    n = min_track_record_length(sr, bench, args.skew, args.kurtosis, args.confidence, excess_kurtosis=args.excess_kurtosis)
    payload = {"min_track_record_obs": None if math.isinf(n) else n}
    if args.periods_per_year and not math.isinf(n):
        payload["min_track_record_years"] = n / args.periods_per_year
    _emit(args, payload, [f"Minimum track record length {_fmt_obs(n, args.periods_per_year)}"])
    return 0


def cmd_minbtl(args: argparse.Namespace) -> int:
    y = min_backtest_length(args.trials, args.max_sharpe)
    _emit(args, {"min_backtest_years": y, "n_trials": args.trials, "max_sharpe": args.max_sharpe},
          [f"Minimum backtest length {y:.2f} years (upper bound 2 ln N / E[max]^2 = "
           f"{2 * math.log(args.trials) / args.max_sharpe ** 2:.2f})" if args.trials > 1 else "Minimum backtest length 0 years (one trial)"])
    return 0


def _read_column(path: str, column: Optional[str]) -> List[float]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    header = rows[0]
    has_header = any(_not_number(c) for c in header)
    if column is not None:
        if not has_header or column not in header:
            raise SystemExit(f"column {column!r} not found in {path}")
        idx, body = header.index(column), rows[1:]
    else:
        body = rows[1:] if has_header else rows
        idx = 0 if len(header) == 1 else len(header) - 1
    out = []
    for row in body:
        if len(row) <= idx or row[idx].strip() == "":
            continue
        out.append(float(row[idx]))
    return out


def _not_number(s: str) -> bool:
    try:
        float(s)
        return False
    except ValueError:
        return True


def cmd_returns(args: argparse.Namespace) -> int:
    rets = _read_column(args.returns_csv, args.column)
    trials = _read_column(args.trials_file, None)
    if args.trials_annual:
        if args.periods_per_year is None:
            raise SystemExit("--trials-annual needs --periods-per-year")
        trials = [to_period_sharpe(t, args.periods_per_year) for t in trials]
    r = deflated_sharpe_from_returns(rets, trials, n_trials=args.n_trials)
    ppy = args.periods_per_year
    payload = {"dsr": r.dsr, "sr0_per_period": r.sr0, "sharpe_per_period": r.sharpe, "psr_zero": r.psr_zero,
               "n_obs": r.n_obs, "n_trials": r.n_trials, "var_trials_per_period": r.var_trials,
               "skew": r.skew, "kurtosis": r.kurtosis, "passes": r.passes(args.confidence)}
    ann = f" ({to_annual_sharpe(r.sharpe, ppy):.3f} annualised)" if ppy else ""
    lines = [
        f"Returns                 {int(r.n_obs):,} observations, Sharpe {r.sharpe:.4f} per period{ann}",
        f"Skew / kurtosis         {r.skew:.3f} / {r.kurtosis:.3f}",
        f"Trials                  {int(r.n_trials):,}, Sharpe variance {r.var_trials:.3g}",
        f"Deflated Sharpe Ratio   {r.dsr:.4f}",
        f"Verdict at {args.confidence:.0%}           {'passes' if r.passes(args.confidence) else 'does not pass'}",
    ]
    _emit(args, payload, lines)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsr", description="Deflated and Probabilistic Sharpe Ratio calculator.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("deflate", help="Deflated Sharpe Ratio from summary statistics")
    p.add_argument("--sharpe", type=float, required=True, help="Sharpe ratio of the selected strategy")
    p.add_argument("--n-obs", type=float, required=True, help="number of return observations")
    p.add_argument("--trials", type=float, required=True, help="number of distinct configurations tried")
    p.add_argument("--trial-variance", type=float, required=True, help="variance of Sharpe ratios across trials")
    p.add_argument("--confidence", type=float, default=0.95)
    _add_moments(p)
    _add_frequency(p)
    p.set_defaults(func=cmd_deflate)

    p = sub.add_parser("psr", help="Probabilistic Sharpe Ratio")
    p.add_argument("--sharpe", type=float, required=True)
    p.add_argument("--n-obs", type=float, required=True)
    p.add_argument("--benchmark", type=float, default=0.0)
    _add_moments(p)
    _add_frequency(p)
    p.set_defaults(func=cmd_psr)

    p = sub.add_parser("mintrl", help="minimum track record length")
    p.add_argument("--sharpe", type=float, required=True)
    p.add_argument("--benchmark", type=float, default=0.0)
    p.add_argument("--confidence", type=float, default=0.95)
    _add_moments(p)
    _add_frequency(p)
    p.set_defaults(func=cmd_mintrl)

    p = sub.add_parser("minbtl", help="minimum backtest length in years")
    p.add_argument("--trials", type=float, required=True)
    p.add_argument("--max-sharpe", type=float, default=1.0, help="annualised Sharpe ratio acceptable by chance")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_minbtl)

    p = sub.add_parser("returns", help="DSR from a returns CSV and a file of trial Sharpe ratios")
    p.add_argument("returns_csv", help="CSV with one column of per-period returns, or use --column")
    p.add_argument("--trials-file", required=True, help="one per-period Sharpe ratio per line, one per trial")
    p.add_argument("--column", default=None, help="returns column name when the CSV has several")
    p.add_argument("--n-trials", type=float, default=None, help="override the trial count (default: lines in --trials-file)")
    p.add_argument("--trials-annual", action="store_true", help="trial Sharpe ratios are annualised")
    p.add_argument("--periods-per-year", type=float, default=None)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_returns)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"dsr: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
