import polars as pl

from src.metrics.stability import StabilityMetrics


def test_psi_returns_non_negative_value() -> None:
    actual = pl.Series("actual", [0, 0, 1, 1])
    expected = pl.Series("expected", [0, 1, 0, 1])
    value = StabilityMetrics.psi(actual, expected, bins=2)
    assert value >= 0.0


def test_csi_matches_psi() -> None:
    actual = pl.Series("actual", [0, 0, 1, 1])
    expected = pl.Series("expected", [0, 1, 0, 1])
    assert StabilityMetrics.csi(actual, expected, bins=2) == StabilityMetrics.psi(actual, expected, bins=2)
