import json

import numpy as np
import polars as pl
import pytest

from src.modeling.pd_model import (
    DEV_SCORES, METRICS_FILE, MODEL_FILE, TEST_SCORES, calibration_table, feature_columns, ks_statistic, score_frame, score_test, train_model,
)

N = 4000


def _mar(n=N, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    segment = rng.choice(["a", "b", "c"], n)
    logit = -2.5 + 1.4 * signal + np.where(segment == "c", 1.0, 0.0)
    frame = {
        "SK_ID_CURR": np.arange(n) + 100_000,
        "signal": np.where(rng.random(n) < 0.1, np.nan, signal),  # 10% missing
        "noise": noise,
        "segment": segment,
    }
    if with_target:
        frame["TARGET"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pl.DataFrame(frame).with_columns(pl.col("signal").fill_nan(None))


@pytest.fixture
def trained(tmp_path):
    mar = tmp_path / "mar.parquet"
    _mar().write_parquet(mar)
    metrics = train_model(mar, tmp_path, tmp_path / "models")
    return tmp_path, metrics


def test_train_writes_model_metrics_and_development_scores(trained):
    tmp, m = trained
    assert (tmp / "models" / MODEL_FILE).exists() and (tmp / "models" / METRICS_FILE).exists()
    assert m["auc"] > 0.72 and 0 < m["ks"] < 1 and m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert m["n_train"] + m["n_holdout"] == N and m["n_holdout"] == pytest.approx(0.2 * N, abs=2)
    assert json.loads((tmp / "models" / METRICS_FILE).read_text())["auc"] == m["auc"]
    dev = pl.read_parquet(tmp / DEV_SCORES)
    assert dev.height == m["n_holdout"] and dev["score"].min() >= 0 and dev["score"].max() <= 1


def test_model_finds_the_real_driver_not_the_noise(trained):
    _, m = trained
    importance = {f["feature"]: f["auc_drop"] for f in m["top_features"]}
    assert importance["signal"] > importance.get("noise", 0) + 0.01


def test_id_and_target_are_never_features(trained):
    tmp, _ = trained
    assert feature_columns(_mar().schema) == ["signal", "noise", "segment"]
    import joblib

    assert set(joblib.load(tmp / "models" / MODEL_FILE)["features"]) == {"signal", "noise", "segment"}


def test_holdout_does_not_depend_on_physical_row_order(tmp_path):
    # regression: the MAR's row order is not guaranteed, and a positional split used to change the hold-out
    ordered, shuffled = tmp_path / "a.parquet", tmp_path / "b.parquet"
    mar = _mar()
    mar.write_parquet(ordered)
    mar.sample(fraction=1.0, shuffle=True, seed=7).write_parquet(shuffled)
    a = train_model(ordered, tmp_path, tmp_path / "ma")
    b = train_model(shuffled, tmp_path, tmp_path / "mb")
    assert a["auc"] == pytest.approx(b["auc"], abs=1e-12) and a["ks"] == pytest.approx(b["ks"], abs=1e-12)


def test_holdout_is_stratified(trained):
    tmp, m = trained
    defaults = int(_mar()["TARGET"].sum())
    held_out_defaults = int(pl.read_parquet(tmp / DEV_SCORES)["TARGET"].sum())
    # stratified: the hold-out gets 20% of the defaulters (to within one account), not "about" 20%
    assert abs(held_out_defaults - 0.2 * defaults) <= 1


def test_score_frame_is_robust_to_unseen_categories_and_missing_values(trained):
    tmp, _ = trained
    fresh = pl.DataFrame({"SK_ID_CURR": [1, 2, 3], "signal": [None, 0.5, -3.0], "noise": [0.0, None, 1.0], "segment": ["a", "NEW", None]})
    scored = score_frame(fresh, tmp / "models")
    assert scored["SK_ID_CURR"].to_list() == [1, 2, 3] and scored["score"].is_between(0, 1).all()


def test_score_frame_rejects_missing_feature_columns(trained):
    tmp, _ = trained
    with pytest.raises(ValueError, match="feature columns missing"):
        score_frame(pl.DataFrame({"SK_ID_CURR": [1], "signal": [0.1]}), tmp / "models")


def test_score_test_scores_every_application_once(trained):
    tmp, _ = trained
    test = _mar(500, seed=3, with_target=False)
    test.write_parquet(tmp / "mar_test.parquet")
    scores = score_test(tmp / "mar_test.parquet", tmp, tmp / "models")
    assert scores.height == 500 and scores["SK_ID_CURR"].to_list() == test["SK_ID_CURR"].to_list()
    assert pl.read_parquet(tmp / TEST_SCORES).height == 500


# ---- metrics ------------------------------------------------------------------------------

def test_ks_statistic_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])) == 1.0  # perfect separation
    assert ks_statistic(y, np.array([0.5] * 8)) == 0.0  # no information
    assert ks_statistic(y, np.array([0.1, 0.6, 0.3, 0.7, 0.2, 0.8, 0.4, 0.9])) == pytest.approx(0.5)


def test_calibration_table_partitions_accounts_and_orders_by_score():
    rng = np.random.default_rng(0)
    score = rng.random(1003)
    y = (rng.random(1003) < score).astype(int)  # perfectly calibrated by construction
    table = calibration_table(y, score, bins=10)
    assert sum(r["accounts"] for r in table) == 1003 and len(table) == 10
    means = [r["mean_score"] for r in table]
    assert means == sorted(means)
    assert all(abs(r["mean_score"] - r["observed_rate"]) < 0.12 for r in table)
