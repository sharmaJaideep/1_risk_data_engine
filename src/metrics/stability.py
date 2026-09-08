import time
import logging
from typing import Tuple, List

import duckdb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class StabilityMetrics:
    """Compatibility wrapper for PSI/CSI calculations used by tests and analytics modules."""

    @staticmethod
    def psi(actual: pl.Series, expected: pl.Series, bins: int = 10) -> float:
        return calculate_feature_psi(
            np.asarray(actual.to_numpy(), dtype=float),
            np.asarray(expected.to_numpy(), dtype=float),
            num_bins=max(2, int(bins)),
        )

    @staticmethod
    def csi(actual: pl.Series, expected: pl.Series, bins: int = 10) -> float:
        return StabilityMetrics.psi(actual, expected, bins=bins)


def split_cohorts(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split the dataframe into baseline (top 70% historical) and target (bottom 30% recent).

    Heuristics:
    - If a date-like column exists (contains 'date' or 'apply' in name), use it to sort ascending.
    - Otherwise, use the existing row order.
    """
    # find a date-like column
    date_col = None
    lc = [c.lower() for c in df.columns]
    for i, name in enumerate(lc):
        if "date" in name or "apply" in name or "time" in name:
            date_col = df.columns[i]
            break

    if date_col is not None:
        df_sorted = df.sort(date_col)
    else:
        # preserve existing order
        df_sorted = df

    n = df_sorted.height
    if n == 0:
        return df_sorted, df_sorted

    cut = int(np.floor(0.7 * n))
    baseline = df_sorted[:cut]
    target = df_sorted[cut:]
    return baseline, target


def calculate_feature_psi(baseline_vec: np.ndarray, target_vec: np.ndarray, num_bins: int = 10) -> float:
    """Calculate PSI for a single feature using baseline-derived quantile bins.

    - baseline_vec, target_vec: 1d numeric arrays
    - num_bins: number of quantile bins (deciles by default)
    """
    eps = 1e-6
    b = np.asarray(baseline_vec, dtype=float)
    t = np.asarray(target_vec, dtype=float)
    b = b[np.isfinite(b)]
    t = t[np.isfinite(t)]

    if b.size == 0 or t.size == 0:
        return float('nan')

    quantiles = np.linspace(0.0, 1.0, max(2, int(num_bins)) + 1)
    edges = np.unique(np.quantile(b, quantiles))
    if edges.size < 2:
        edges = np.array([b.min(), b.max()])
    if edges[0] == edges[-1]:
        edges = np.array([edges[0] - 0.5, edges[-1] + 0.5])

    baseline_counts, _ = np.histogram(b, bins=edges)
    target_counts, _ = np.histogram(t, bins=edges)

    baseline_pct = baseline_counts.astype(float) + eps
    baseline_pct /= baseline_pct.sum()

    target_pct = target_counts.astype(float) + eps
    target_pct /= target_pct.sum()

    psi_components = (target_pct - baseline_pct) * np.log(target_pct / baseline_pct)
    psi = np.sum(psi_components)
    return float(psi)


def _drift_label(psi: float) -> str:
    if np.isnan(psi):
        return 'UNKNOWN'
    if psi < 0.1:
        return 'STABLE'
    if psi < 0.25:
        return 'MODERATE_SHIFT'
    return 'SIGNIFICANT_DRIFT'


def calculate_portfolio_stability(data_path: str) -> pl.DataFrame:
    """Calculate PSI for a set of numeric features and return a Polars DataFrame report.

    Uses Polars LazyFrame to read the parquet and DuckDB to validate counts.
    """
    # Use DuckDB to validate record count quickly
    try:
        con = duckdb.connect(database=':memory:')
        count = con.execute(f"SELECT COUNT(*) as cnt FROM read_parquet('{data_path}')").fetchdf()['cnt'].iloc[0]
        logger.info("DuckDB record count: %d", int(count))
    except Exception:
        logger.warning("DuckDB count failed, proceeding with Polars read")

    # Read lazily with Polars
    lf = pl.scan_parquet(data_path)
    df = lf.collect()

    # choose candidate numeric features
    candidate_cols = [
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'DAYS_BIRTH',
        'AMT_ANNUITY',
        'CNT_CHILDREN'
    ]

    # add bureau-like aggregates by selecting columns that contain common substrings
    for col in df.columns:
        low = col.lower()
        if any(x in low for x in ('bureau', 'delinq', 'delinquency', 'dpd', 'overdue')):
            candidate_cols.append(col)

    # also include all numeric columns as fallback
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )
    numeric_cols = [c for c, dtype in zip(df.columns, df.dtypes) if dtype in numeric_types]
    candidate_cols.extend(numeric_cols)

    # deduplicate and keep those actually present
    candidate_cols = [c for c in dict.fromkeys(candidate_cols) if c in df.columns]

    baseline, target = split_cohorts(df)

    rows: List[dict] = []
    for col in candidate_cols:
        try:
            b_arr = baseline[col].to_numpy()
            t_arr = target[col].to_numpy()
        except Exception:
            continue

        # ensure numeric conversion
        try:
            b_arr = b_arr.astype(float)
            t_arr = t_arr.astype(float)
        except Exception:
            continue

        psi = calculate_feature_psi(b_arr, t_arr, num_bins=10)
        rows.append({'feature_name': col, 'psi_score': psi, 'drift_status': _drift_label(psi)})

    report = pl.DataFrame(rows)

    # Save outputs
    out_parquet = 'data/processed/portfolio_stability_report.parquet'
    out_csv = 'data/processed/portfolio_stability_report.csv'
    try:
        report.write_parquet(out_parquet)
        report.write_csv(out_csv)
        logger.info('Wrote stability report to %s and %s', out_parquet, out_csv)
    except Exception as e:
        logger.exception('Failed to write reports: %s', e)

    return report


if __name__ == '__main__':
    start = time.time()
    data_path = 'data/processed/master_analytical_record.parquet'
    logger.info('Starting portfolio stability run against %s', data_path)
    report = calculate_portfolio_stability(data_path)
    significant = report.filter(pl.col('psi_score') >= 0.25)
    if significant.height:
        logger.info('Features with significant drift:')
        for r in significant.select(['feature_name', 'psi_score']).rows():
            print(f"{r[0]}: PSI={r[1]:.4f}")
    else:
        logger.info('No features with PSI >= 0.25')

    duration = time.time() - start
    logger.info('Completed in %.2fs', duration)
