import json
import math

import pytest

from deflated_sharpe import (
    PERIODS_PER_YEAR,
    ar1_effective_obs,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_ratio_std_error,
    to_annual_sharpe,
    to_annual_variance,
    to_period_sharpe,
    to_period_variance,
    trial_sharpe_variance,
)
from deflated_sharpe.cli import main


# ---- moments ----------------------------------------------------------------


def test_return_stats_known_values():
    st = return_stats([1.0, 2.0, 3.0, 4.0, 10.0])
    assert st.n_obs == 5
    assert st.mean == pytest.approx(4.0)
    assert st.std == pytest.approx(math.sqrt(12.5))
    assert st.sharpe == pytest.approx(4.0 / math.sqrt(12.5))
    # deviations -3, -2, -1, 0, 6: m2 = 10, m3 = 36, m4 = 278.8
    assert st.skew == pytest.approx(36 / 10 ** 1.5)
    assert st.kurtosis == pytest.approx(278.8 / 100.0)
    assert st.excess_kurtosis == pytest.approx(st.kurtosis - 3)


def test_return_stats_risk_free_and_ddof():
    a = return_stats([0.02, 0.01, 0.03], risk_free=0.01)
    assert a.mean == pytest.approx(0.01)
    b = return_stats([0.02, 0.01, 0.03], ddof=0)
    assert b.std < return_stats([0.02, 0.01, 0.03]).std


def test_return_stats_rejects_bad_input():
    with pytest.raises(ValueError):
        return_stats([1.0, 2.0])
    with pytest.raises(ValueError):
        return_stats([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        return_stats([1.0, float("nan"), 2.0])


def test_trial_variance():
    assert trial_sharpe_variance([0.1, 0.2, 0.3]) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        trial_sharpe_variance([0.1])


# ---- frequency --------------------------------------------------------------


def test_frequency_round_trips():
    assert to_annual_sharpe(to_period_sharpe(1.7, 252), 252) == pytest.approx(1.7)
    assert to_annual_variance(to_period_variance(0.4, 12), 12) == pytest.approx(0.4)
    assert PERIODS_PER_YEAR["monthly"] == 12
    with pytest.raises(ValueError):
        to_period_sharpe(1.0, 0)


def test_ar1_effective_obs():
    assert ar1_effective_obs(1000, 0.0) == pytest.approx(1000)
    assert ar1_effective_obs(1000, 0.5) == pytest.approx(1000 / 3)
    with pytest.raises(ValueError):
        ar1_effective_obs(1000, 1.0)


# ---- core behaviour ---------------------------------------------------------


def test_kurtosis_conventions_agree():
    a = probabilistic_sharpe_ratio(0.1, 500, skew=-1, kurtosis=7)
    b = probabilistic_sharpe_ratio(0.1, 500, skew=-1, excess_kurtosis=4)
    assert a == pytest.approx(b)
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 500, kurtosis=7, excess_kurtosis=4)
    with pytest.raises(ValueError, match="excess_kurtosis"):
        probabilistic_sharpe_ratio(0.1, 500, kurtosis=0.2)


def test_psr_monotone_and_bounds():
    assert probabilistic_sharpe_ratio(0.0, 100) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.1, 200) > probabilistic_sharpe_ratio(0.1, 100)
    assert probabilistic_sharpe_ratio(0.1, 100, benchmark=0.05) < probabilistic_sharpe_ratio(0.1, 100)
    # negative skew and fat tails lower confidence in a positive SR
    assert probabilistic_sharpe_ratio(0.1, 100, skew=-2, kurtosis=9) < probabilistic_sharpe_ratio(0.1, 100)


def test_std_error_normal_case():
    assert sharpe_ratio_std_error(0.0, 101) == pytest.approx(0.1)


def test_expected_max_edge_cases():
    assert expected_max_sharpe(1, 0.3) == 0.0
    assert expected_max_sharpe(1, 0.3, mean_trials=0.2) == 0.2
    assert expected_max_sharpe(50, 0.0) == 0.0
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))
    assert expected_max_sharpe(1000, 1.0) > expected_max_sharpe(100, 1.0)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            expected_max_sharpe(bad, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1.0)


def test_dsr_with_one_trial_equals_psr():
    r = deflated_sharpe_ratio(0.08, 400, n_trials=1, var_trials=0.5, skew=-0.3, kurtosis=4)
    assert r.sr0 == 0.0
    assert r.dsr == pytest.approx(probabilistic_sharpe_ratio(0.08, 400, -0.3, 4))
    assert r.dsr == pytest.approx(r.psr_zero)


def test_non_normality_term_guard():
    # A huge annualised Sharpe passed as per-period trips the guard with a hint.
    with pytest.raises(ValueError, match="per-period"):
        probabilistic_sharpe_ratio(5.0, 100, skew=3.0)


def test_min_track_record_length_edge_cases():
    assert math.isinf(min_track_record_length(0.05, benchmark=0.05))
    assert math.isinf(min_track_record_length(0.02, benchmark=0.05))
    with pytest.raises(ValueError):
        min_track_record_length(0.1, confidence=1.0)


def test_min_backtest_length_edge_cases():
    assert min_backtest_length(1) == 0.0
    assert min_backtest_length(100, 2.0) == pytest.approx(min_backtest_length(100, 1.0) / 4)
    with pytest.raises(ValueError):
        min_backtest_length(100, 0.0)


def test_from_returns_matches_summary_path():
    import random

    rng = random.Random(3)
    rets = [rng.gauss(0.0005, 0.01) for _ in range(600)]
    trials = [rng.gauss(0.0, 0.04) for _ in range(30)]
    a = deflated_sharpe_from_returns(rets, trials)
    st = return_stats(rets)
    b = deflated_sharpe_ratio(st.sharpe, st.n_obs, 30, trial_sharpe_variance(trials), st.skew, st.kurtosis)
    assert a == b
    c = deflated_sharpe_from_returns(rets, trials, n_trials=10)
    assert c.n_trials == 10 and c.dsr > a.dsr


# ---- CLI --------------------------------------------------------------------


def test_cli_deflate_json(capsys):
    code = main([
        "deflate", "--sharpe", "2.5", "--annual", "--periods-per-year", "250", "--n-obs", "1250",
        "--trials", "100", "--trial-variance", "0.5", "--skew", "-3", "--kurtosis", "10", "--json",
    ])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dsr"] == pytest.approx(0.9004, abs=5e-5)
    assert out["passes"] is False


def test_cli_text_commands(capsys):
    assert main(["psr", "--sharpe", "0.458", "--n-obs", "24", "--skew", "-2.448", "--kurtosis", "10.164"]) == 0
    assert "0.913" in capsys.readouterr().out
    assert main(["mintrl", "--sharpe", "2", "--benchmark", "1", "--annual", "--periods-per-year", "252"]) == 0
    assert "2.73 years" in capsys.readouterr().out
    assert main(["minbtl", "--trials", "45"]) == 0
    assert "5.00 years" in capsys.readouterr().out


def test_cli_returns_file(tmp_path, capsys):
    import random

    rng = random.Random(5)
    r = tmp_path / "returns.csv"
    r.write_text("date,ret\n" + "\n".join(f"d{i},{rng.gauss(0.001, 0.01)}" for i in range(300)))
    t = tmp_path / "trials.txt"
    t.write_text("\n".join(str(rng.gauss(0, 0.05)) for _ in range(20)))
    assert main(["returns", str(r), "--column", "ret", "--trials-file", str(t), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_obs"] == 300 and out["n_trials"] == 20


def test_cli_reports_value_errors(capsys):
    code = main(["psr", "--sharpe", "0.1", "--n-obs", "100", "--kurtosis", "0.3"])
    assert code == 2
    assert "excess_kurtosis" in capsys.readouterr().err
