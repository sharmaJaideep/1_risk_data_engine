import math

import numpy as np
import polars as pl
import pytest
import yaml

from src.metrics.stability import (
    NULL_BIN,
    OTHER_BIN,
    StabilityMetrics,
    bin_distribution,
    drift_label,
    run_stability,
    split_by_time,
)

N = 20_000


def _normal(mean=0.0, n=N, seed=0):
    return pl.Series("x", np.random.default_rng(seed).normal(mean, 1.0, n))


# ---- the maths ---------------------------------------------------------------------------

def test_hand_computed_categorical_psi():
    baseline = pl.Series("c", ["A"] * 50 + ["B"] * 50)
    target = pl.Series("c", ["A"] * 25 + ["B"] * 75)
    # (0.25-0.5)ln(0.25/0.5) + (0.75-0.5)ln(0.75/0.5) = 0.17329 + 0.10137
    assert StabilityMetrics.psi(baseline, target) == pytest.approx(0.27465, abs=1e-4)


def test_same_distribution_is_stable_and_identical_is_zero():
    assert StabilityMetrics.psi(_normal(seed=1), _normal(seed=2)) < 0.01
    same = _normal(seed=3)
    assert StabilityMetrics.psi(same, same) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_injected_shift_and_crosses_thresholds():
    psis = [StabilityMetrics.psi(_normal(0, seed=1), _normal(shift, seed=2)) for shift in (0.0, 0.25, 0.5, 1.0)]
    assert psis == sorted(psis) and len(set(psis)) == 4  # strictly increasing
    assert drift_label(psis[0]) == "STABLE"
    assert drift_label(psis[-1]) == "SIGNIFICANT_DRIFT"
    assert psis[-1] > 0.25


def test_variance_change_without_mean_change_is_detected():
    wider = pl.Series("x", np.random.default_rng(5).normal(0, 2.0, N))
    assert StabilityMetrics.psi(_normal(seed=1), wider) > 0.25


def test_target_values_outside_baseline_range_are_counted():
    # regression: out-of-range target values used to be silently dropped by the histogram
    baseline = pl.Series("x", np.linspace(0, 1, 1000))
    target = pl.Series("x", [5.0] * 1000)
    dist, _ = bin_distribution(baseline, target)
    assert dist["target_pct"].sum() == pytest.approx(1.0)
    assert dist.filter(pl.col("bin").str.starts_with(">"))["target_pct"][0] == pytest.approx(1.0)
    assert StabilityMetrics.psi(baseline, target) > 1.0


def test_missing_value_shift_registers_through_null_bin():
    baseline = _normal(seed=1)
    target = pl.Series("x", [None if i % 10 < 3 else v for i, v in enumerate(_normal(seed=1))])  # 30% missing
    csi = StabilityMetrics.csi(baseline, target)
    assert StabilityMetrics.psi(baseline, target) > 0.1
    assert csi.sort("contribution", descending=True)["bin"][0] == NULL_BIN


def test_unseen_category_lands_in_other_bin():
    baseline = pl.Series("c", ["A"] * 500 + ["B"] * 500)
    target = pl.Series("c", ["A"] * 350 + ["B"] * 350 + ["NEW"] * 300)
    csi = StabilityMetrics.csi(baseline, target)
    assert csi.sort("contribution", descending=True)["bin"][0] == OTHER_BIN
    assert StabilityMetrics.psi(baseline, target) > 0.25


def test_discrete_numeric_flag_is_binned_by_value():
    baseline = pl.Series("flag", [0] * 900 + [1] * 100)
    target = pl.Series("flag", [0] * 700 + [1] * 300)
    _, feature_type = bin_distribution(baseline, target)
    assert feature_type == "discrete"
    assert StabilityMetrics.psi(baseline, target) > 0.1


def test_csi_contributions_sum_to_psi_and_shares_are_distributions():
    baseline, target = _normal(0, seed=1), _normal(0.7, seed=2)
    csi = StabilityMetrics.csi(baseline, target)
    assert csi["contribution"].sum() == pytest.approx(StabilityMetrics.psi(baseline, target))
    assert csi["baseline_pct"].sum() == pytest.approx(1.0) and csi["target_pct"].sum() == pytest.approx(1.0)
    assert (csi["contribution"] >= 0).all()


def test_degenerate_inputs():
    assert math.isnan(StabilityMetrics.psi(_normal(), pl.Series("x", [], dtype=pl.Float64)))
    assert math.isnan(StabilityMetrics.psi(pl.Series("x", [None, None], dtype=pl.Float64), _normal()))
    constant = pl.Series("x", [3.0] * 100)
    assert StabilityMetrics.psi(constant, constant) == pytest.approx(0.0, abs=1e-12)
    assert StabilityMetrics.psi(constant, pl.Series("x", [4.0] * 100)) > 1.0
    with_inf = pl.Series("x", [1.0, 2.0, float("inf"), float("nan")] * 50)
    assert not math.isnan(StabilityMetrics.psi(with_inf, with_inf))


# ---- cohorts ----------------------------------------------------------------------------

def test_split_by_time_keeps_ties_together_and_drops_null_times():
    df = pl.DataFrame({"t": [1] * 50 + [2] * 30 + [3] * 20 + [None] * 5, "v": range(105)})
    baseline, target = split_by_time(df, "t", 0.7)
    assert baseline["t"].max() <= 2 < target["t"].min()
    assert (baseline.height, target.height) == (80, 20)  # all the 2s stay in the baseline


# ---- end to end -------------------------------------------------------------------------

def _write_inputs(tmp_path):
    rng = np.random.default_rng(0)
    train = pl.DataFrame({
        "SK_ID_CURR": range(N), "TARGET": rng.integers(0, 2, N),
        "stable": rng.normal(0, 1, N), "drifty": rng.normal(0, 1, N),
        "segment": rng.choice(["a", "b"], N, p=[0.8, 0.2]),
    })
    test = pl.DataFrame({
        "SK_ID_CURR": range(N, 2 * N),
        "stable": rng.normal(0, 1, N), "drifty": rng.normal(2, 1, N),
        "segment": rng.choice(["a", "b"], N, p=[0.8, 0.2]),
    })
    train.write_parquet(tmp_path / "train.parquet")
    test.write_parquet(tmp_path / "test.parquet")
    hist = pl.DataFrame({"SK_ID_CURR": range(N), "DAYS_X": rng.integers(-1000, 0, N), "amt": rng.normal(0, 1, N)})
    hist.with_columns(
        pl.when(pl.col("DAYS_X") > -300).then(pl.col("amt") + 3).otherwise(pl.col("amt")).alias("amt")  # recent rows shifted
    ).write_parquet(tmp_path / "hist.parquet")


def _config(tmp_path, hist_alert=True):
    path = tmp_path / "stability.yaml"
    path.write_text(yaml.safe_dump({
        "exclude_prefixes": ["SK_ID"], "exclude_columns": ["TARGET"],
        "analyses": [
            {"name": "train_vs_test", "baseline": "train", "target": "test", "score_columns": ["stable"]},
            {"name": "hist_by_recency", "table": "hist", "time_column": "DAYS_X", "alert": hist_alert},
            {"name": "missing_table", "table": "nope", "time_column": "t"},
        ],
    }))
    return path


def test_run_stability_detects_injected_drift_end_to_end(tmp_path):
    _write_inputs(tmp_path)
    report = run_stability(tmp_path, _config(tmp_path))
    by = {(r["analysis"], r["feature_name"]): r for r in report.iter_rows(named=True)}

    assert by[("train_vs_test", "drifty")]["drift_status"] == "SIGNIFICANT_DRIFT"
    assert by[("train_vs_test", "stable")]["drift_status"] == "STABLE"
    assert by[("train_vs_test", "segment")]["drift_status"] == "STABLE" and by[("train_vs_test", "segment")]["feature_type"] == "categorical"
    assert by[("train_vs_test", "stable")]["role"] == "score" and by[("train_vs_test", "drifty")]["role"] == "characteristic"
    assert by[("hist_by_recency", "amt")]["drift_status"] == "SIGNIFICANT_DRIFT"

    scored = set(by)
    assert not any(f in ("SK_ID_CURR", "TARGET", "DAYS_X") for _, f in scored)  # ids, target, time column excluded
    assert not any(a == "missing_table" for a, _ in scored)  # skipped, not crashed
    assert (tmp_path / "portfolio_stability_report.parquet").exists() and (tmp_path / "portfolio_stability_report.csv").exists()


def test_alert_flag_propagates_to_report(tmp_path):
    _write_inputs(tmp_path)
    report = run_stability(tmp_path, _config(tmp_path, hist_alert=False))
    flags = dict(report.group_by("analysis").agg(pl.col("alerting").first()).iter_rows())
    assert flags == {"train_vs_test": True, "hist_by_recency": False}


def test_committed_config_is_well_formed():
    config = yaml.safe_load(open("config/stability.yaml"))
    names = [a["name"] for a in config["analyses"]]
    assert len(names) == len(set(names)) >= 2
    for a in config["analyses"]:
        assert ("baseline" in a and "target" in a) or ("table" in a and "time_column" in a)
        assert 0 < a.get("baseline_fraction", 0.7) < 1
