"""Population stability: PSI (score) and CSI (characteristic) drift detection.

PSI compares how a variable is distributed in a baseline cohort versus a target cohort:

    PSI = sum over bins of (target% - baseline%) * ln(target% / baseline%)

Bins come from the baseline only, so the target is measured against a fixed yardstick:
- continuous numeric: baseline quantile bins with open-ended outer edges (target values
  outside the baseline range are counted, not dropped)
- discrete numeric (few distinct values) and strings: one bin per baseline value, with an
  `__OTHER__` bin for unseen/rare values
- missing values get their own `__NULL__` bin, so a change in missing-data rate registers

`csi` returns the per-bin contributions (which bins drive a shift); PSI is their sum.
Conventionally "PSI" refers to a model score and "CSI" to its input characteristics; the
maths is identical, and `score_columns` in config/stability.yaml tags which is which.

Cohorts are defined per analysis in config/stability.yaml: either two tables (e.g. train vs
test) or one table split by a time column (older vs more recent rows). An analysis with
`alert: false` is reported but does not raise alerts (see the config for why).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

EPS = 1e-4  # floor for empty bins so ln() stays finite
MAX_CATEGORIES = 50
OTHER_BIN, NULL_BIN = "__OTHER__", "__NULL__"
CONFIG_PATH = Path("config/stability.yaml")

REPORT_SCHEMA = {
    "analysis": pl.String,
    "feature_name": pl.String,
    "feature_type": pl.String,
    "role": pl.String,
    "psi_score": pl.Float64,
    "drift_status": pl.String,
    "baseline_rows": pl.Int64,
    "target_rows": pl.Int64,
    "top_driver": pl.String,
    "alerting": pl.Boolean,  # False = informational: reported, but not used to raise alerts
}


def drift_label(psi: float) -> str:
    if np.isnan(psi):
        return "UNKNOWN"
    if psi < 0.1:
        return "STABLE"
    if psi < 0.25:
        return "MODERATE_SHIFT"
    return "SIGNIFICANT_DRIFT"


def _clean(series: pl.Series) -> tuple[pl.Series, int]:
    """Drop nulls (and NaN/inf for floats); return the clean values and how many were missing."""
    values = series.drop_nulls()
    if values.dtype.is_float():
        values = values.filter(values.is_finite())
    return values, len(series) - len(values)


def bin_distribution(baseline: pl.Series, target: pl.Series, bins: int = 10) -> tuple[pl.DataFrame, str] | None:
    """Per-bin shares for both cohorts, plus the feature type. None if either cohort is empty."""
    if len(baseline) == 0 or len(target) == 0:
        return None
    b, b_missing = _clean(baseline)
    t, t_missing = _clean(target)
    if len(b) == 0:
        return None  # nothing to build bins from

    if b.dtype.is_numeric() and b.n_unique() > bins:
        feature_type = "numeric"
        b_np, t_np = b.cast(pl.Float64).to_numpy(), t.cast(pl.Float64).to_numpy()
        edges = np.unique(np.quantile(b_np, np.linspace(0, 1, bins + 1)[1:-1]))
        b_counts = np.bincount(np.searchsorted(edges, b_np, side="left"), minlength=len(edges) + 1)
        t_counts = np.bincount(np.searchsorted(edges, t_np, side="left"), minlength=len(edges) + 1)
        labels = [f"<= {edges[0]:.4g}"] + [f"({lo:.4g}, {hi:.4g}]" for lo, hi in zip(edges[:-1], edges[1:])] + [f"> {edges[-1]:.4g}"]
    else:
        feature_type = "categorical" if b.dtype == pl.String else "discrete"
        top = b.value_counts(sort=True).head(MAX_CATEGORIES)
        cats = top[b.name].to_list()
        b_map = dict(zip(top[b.name].to_list(), top["count"].to_list()))
        t_vc = t.value_counts()
        t_map = dict(zip(t_vc[t.name].to_list(), t_vc["count"].to_list()))
        b_counts = np.array([b_map[c] for c in cats] + [len(b) - sum(b_map.values())])
        t_counts = np.array([t_map.get(c, 0) for c in cats] + [len(t) - sum(t_map.get(c, 0) for c in cats)])
        labels = [str(c) for c in cats] + [OTHER_BIN]

    if b_missing or t_missing:
        b_counts, t_counts = np.append(b_counts, b_missing), np.append(t_counts, t_missing)
        labels.append(NULL_BIN)

    dist = pl.DataFrame(
        {
            "bin": labels,
            "baseline_pct": b_counts / b_counts.sum(),
            "target_pct": t_counts / t_counts.sum(),
        }
    )
    return dist, feature_type


def _contributions(dist: pl.DataFrame) -> pl.DataFrame:
    b = dist["baseline_pct"].to_numpy().clip(min=EPS)
    t = dist["target_pct"].to_numpy().clip(min=EPS)
    return dist.with_columns(pl.Series("contribution", (t - b) * np.log(t / b)))


class StabilityMetrics:
    """Scalar PSI and per-bin CSI for a single variable."""

    @staticmethod
    def csi(baseline: pl.Series, target: pl.Series, bins: int = 10) -> pl.DataFrame:
        """Per-bin shares and contributions; `contribution` sums to the PSI."""
        result = bin_distribution(baseline, target, bins)
        if result is None:
            return pl.DataFrame(schema={"bin": pl.String, "baseline_pct": pl.Float64, "target_pct": pl.Float64, "contribution": pl.Float64})
        return _contributions(result[0])

    @staticmethod
    def psi(baseline: pl.Series, target: pl.Series, bins: int = 10) -> float:
        contributions = StabilityMetrics.csi(baseline, target, bins)
        return float(contributions["contribution"].sum()) if contributions.height else float("nan")


def split_by_time(df: pl.DataFrame, time_column: str, baseline_fraction: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Older rows (time <= the quantile cut) form the baseline; later rows the target.

    Cutting on the value rather than row position keeps tied timestamps in the same cohort.
    """
    df = df.filter(pl.col(time_column).is_not_null())
    cut = df[time_column].quantile(baseline_fraction, interpolation="lower")
    return df.filter(pl.col(time_column) <= cut), df.filter(pl.col(time_column) > cut)


def analyze_frames(
    analysis: str,
    baseline: pl.DataFrame,
    target: pl.DataFrame,
    features: list[str],
    bins: int = 10,
    score_columns: set[str] | frozenset[str] = frozenset(),
    alerting: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for col in features:
        result = bin_distribution(baseline[col], target[col], bins)
        if result is None:
            continue
        dist, feature_type = result
        contributions = _contributions(dist)
        psi = float(contributions["contribution"].sum())
        rows.append(
            {
                "analysis": analysis,
                "feature_name": col,
                "feature_type": feature_type,
                "role": "score" if col in score_columns else "characteristic",
                "psi_score": psi,
                "drift_status": drift_label(psi),
                "baseline_rows": len(baseline),
                "target_rows": len(target),
                "top_driver": contributions.sort("contribution", descending=True)["bin"][0],
                "alerting": alerting,
            }
        )
    return rows


def _eligible(columns: list[str], exclude_columns: set[str], exclude_prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in columns if c not in exclude_columns and not c.startswith(exclude_prefixes)]


def run_stability(data_dir: str | Path = "data/processed", config_path: str | Path = CONFIG_PATH) -> pl.DataFrame:
    """Run every analysis in the config and write portfolio_stability_report.{parquet,csv}."""
    data_dir = Path(data_dir)
    config = yaml.safe_load(Path(config_path).read_text())
    bins = config.get("bins", 10)
    base_cols = set(config.get("exclude_columns", []))
    base_prefixes = tuple(config.get("exclude_prefixes", []))
    rows: list[dict[str, Any]] = []

    for spec in config["analyses"]:
        name = spec["name"]
        cols = base_cols | set(spec.get("exclude_columns", []))
        prefixes = base_prefixes + tuple(spec.get("exclude_prefixes", []))
        scores = frozenset(spec.get("score_columns", []))
        start = time.perf_counter()

        if "time_column" in spec:
            path = data_dir / f"{spec['table']}.parquet"
            if not path.exists():
                logger.warning("%s: %s not found; skipping", name, path)
                continue
            lf = pl.scan_parquet(path)
            time_col = spec["time_column"]
            features = _eligible(lf.collect_schema().names(), cols | {time_col}, prefixes)
            baseline, target = split_by_time(lf.select([time_col, *features]).collect(), time_col, spec.get("baseline_fraction", 0.7))
        else:
            paths = [data_dir / f"{spec['baseline']}.parquet", data_dir / f"{spec['target']}.parquet"]
            if not all(p.exists() for p in paths):
                logger.warning("%s: missing input parquet; skipping", name)
                continue
            lfs = [pl.scan_parquet(p) for p in paths]
            common = [c for c in lfs[0].collect_schema().names() if c in set(lfs[1].collect_schema().names())]
            features = _eligible(common, cols, prefixes)
            baseline, target = (lf.select(features).collect() for lf in lfs)

        found = analyze_frames(name, baseline, target, features, bins, scores, spec.get("alert", True))
        rows += found
        logger.info("%s: %d features, baseline=%d target=%d rows (%.1fs)", name, len(found), len(baseline), len(target), time.perf_counter() - start)

    report = pl.DataFrame(rows, schema=REPORT_SCHEMA).sort(["analysis", "psi_score"], descending=[False, True])
    report.write_parquet(data_dir / "portfolio_stability_report.parquet")
    report.write_csv(data_dir / "portfolio_stability_report.csv")
    logger.info("Wrote stability report (%d rows) to %s", report.height, data_dir)
    return report


if __name__ == "__main__":
    result = run_stability()
    for r in result.filter(pl.col("psi_score") >= 0.25).iter_rows(named=True):
        print(f"[{r['analysis']}] {r['feature_name']}: PSI={r['psi_score']:.4f}")
