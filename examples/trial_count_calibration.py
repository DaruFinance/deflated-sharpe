"""How often does the best of N zero-skill strategies clear SR0?

SR0 is an expected maximum, so a correctly specified bar is cleared by the null
maximum about half the time. This script builds corpora of strategies with no
skill at all, under different correlation structures, and compares the bar built
from the number of distinct strategies with the bar built from a
correlation-adjusted "effective" count (the eigenvalue participation ratio).

Run:  python examples/trial_count_calibration.py
Takes about a minute with the defaults and uses only the standard library.
"""

from __future__ import annotations

import math
import random
import statistics

from deflated_sharpe import expected_max_sharpe, sharpe_ratio

SIMS = 300  # corpora per structure
N = 60  # strategies per corpus
T = 250  # return observations per strategy


def participation_ratio(eigenvalues):
    return sum(eigenvalues) ** 2 / sum(v * v for v in eigenvalues)


def corpus(rng, rho, n_families=None):
    """Zero-skill returns. rho sets a market factor shared by every strategy; with
    n_families, strategies inside a family also share a family factor."""
    market = [rng.gauss(0, 1) for _ in range(T)]
    families = [[rng.gauss(0, 1) for _ in range(T)] for _ in range(n_families or 0)]
    out = []
    for i in range(N):
        fam = families[i % n_families] if n_families else None
        row = []
        for t in range(T):
            x = math.sqrt(rho) * market[t]
            if fam is not None:
                x += math.sqrt(0.3) * fam[t] + math.sqrt(max(0.0, 1 - rho - 0.3)) * rng.gauss(0, 1)
            else:
                x += math.sqrt(1 - rho) * rng.gauss(0, 1)
            row.append(x)
        out.append(row)
    return out


def analytic_n_eff(rho, n_families=None):
    """Participation ratio of the population correlation matrix."""
    if not n_families:
        return participation_ratio([1 + (N - 1) * rho] + [1 - rho] * (N - 1))
    per = N // n_families
    within = rho + 0.3
    fam_eig = 1 + (per - 1) * within
    lam_market = fam_eig + (N - per) * rho
    eig = [lam_market] + [fam_eig - per * rho] * (n_families - 1) + [1 - within] * (N - n_families)
    return participation_ratio(eig)


def rates(structure, rng):
    """Score both bars on the same corpora, so any gap comes from the count alone."""
    rho, fams = structure
    n_eff = analytic_n_eff(rho, fams)
    hits_n = hits_eff = 0
    for _ in range(SIMS):
        srs = [sharpe_ratio(r) for r in corpus(rng, rho, fams)]
        v, best = statistics.variance(srs), max(srs)
        hits_n += best > expected_max_sharpe(N, v)
        hits_eff += best > expected_max_sharpe(n_eff, v)
    return hits_n / SIMS, hits_eff / SIMS, n_eff


def main():
    rng = random.Random(2026)
    structures = [
        ("independent", (0.0, None)),
        ("equicorrelated 0.2", (0.2, None)),
        ("equicorrelated 0.5", (0.5, None)),
        ("6 families + market 0.1", (0.1, 6)),
    ]
    print(f"{SIMS} zero-skill corpora each, N = {N} strategies, T = {T} observations\n")
    print(f"{'structure':<26}{'N_eff':>7}{'bar from N':>13}{'bar from N_eff':>17}")
    for name, st in structures:
        distinct, eff, n_eff = rates(st, rng)
        print(f"{name:<26}{n_eff:>7.1f}{distinct:>13.0%}{eff:>17.0%}")
    print("\nAbout 50% is correct: SR0 is the expected maximum, not a quantile.")


if __name__ == "__main__":
    main()
