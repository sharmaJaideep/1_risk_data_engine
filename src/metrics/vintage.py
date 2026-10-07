"""Roll-rate and vintage analysis on bureau_balance.

bureau_balance holds one row per (SK_ID_BUREAU, MONTHS_BALANCE) where 0 is the
most recent month and earlier months are negative. STATUS is one of:
C = closed, X = unknown, 0 = current, 1..5 = DPD buckets (1-30, 31-60, 61-90,
91-120, 120+/written off). Rows with STATUS 'X' carry no information and are
excluded from both analyses.

Caveat: history is truncated at MONTHS_BALANCE = -96, so an account's first
observed month is its true opening month only if it opened inside that window.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.db.duckdb_engine import DuckDBEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUCKET_LABELS = {
    "C": "CLOSED",
    "0": "CURRENT",
    "1": "DPD_1_30",
    "2": "DPD_31_60",
    "3": "DPD_61_90",
    "4": "DPD_91_120",
    "5": "DPD_120_PLUS",
}
_BUCKET_CASE = "CASE STATUS " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in BUCKET_LABELS.items()) + " END"

# Accounts at or beyond this bucket are treated as "bad" for vintage curves (61+ DPD).
BAD_STATUSES = ("3", "4", "5")
MONTHS_PER_COHORT = 3


def _connect(path: str | Path) -> tuple[DuckDBEngine, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bureau_balance parquet not found: {path}")
    return DuckDBEngine(":memory:"), str(path)


def compute_roll_rates(bureau_balance_path: str | Path) -> pl.DataFrame:
    """Month-over-month transition matrix between delinquency buckets.

    Returns from_bucket, to_bucket, accounts (transition count) and roll_rate
    (share of the from_bucket's transitions that land in to_bucket).
    """
    conn, path = _connect(bureau_balance_path)
    sql = f"""
    WITH b AS (
        SELECT SK_ID_BUREAU, MONTHS_BALANCE, {_BUCKET_CASE} AS bucket
        FROM read_parquet('{path}') WHERE STATUS <> 'X'
    ),
    t AS (
        SELECT cur.bucket AS from_bucket, nxt.bucket AS to_bucket
        FROM b cur
        JOIN b nxt ON nxt.SK_ID_BUREAU = cur.SK_ID_BUREAU AND nxt.MONTHS_BALANCE = cur.MONTHS_BALANCE + 1
    )
    SELECT from_bucket, to_bucket, COUNT(*) AS accounts,
           COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY from_bucket) AS roll_rate
    FROM t GROUP BY from_bucket, to_bucket
    ORDER BY from_bucket, to_bucket
    """
    try:
        return conn.query(sql)
    finally:
        conn.close()


def compute_vintage(bureau_balance_path: str | Path, months_per_cohort: int = MONTHS_PER_COHORT) -> pl.DataFrame:
    """Cumulative bad-rate curves by opening cohort and months-on-book (MOB).

    Cohort = account's first observed month bucketed into `months_per_cohort`-month
    groups (cohort_start_month is relative to now, e.g. -12 = 12 months ago).
    For each MOB the rate is accounts that have gone bad (61+ DPD) by then, over
    accounts in the cohort observed at that MOB, so right-censored accounts
    don't dilute later MOBs.
    """
    conn, path = _connect(bureau_balance_path)
    bad = ", ".join(f"'{s}'" for s in BAD_STATUSES)
    sql = f"""
    WITH b AS (
        SELECT SK_ID_BUREAU, MONTHS_BALANCE, STATUS
        FROM read_parquet('{path}') WHERE STATUS <> 'X'
    ),
    acct AS (
        SELECT SK_ID_BUREAU,
               MIN(MONTHS_BALANCE) AS open_month,
               MAX(MONTHS_BALANCE) - MIN(MONTHS_BALANCE) AS max_mob,
               MIN(CASE WHEN STATUS IN ({bad}) THEN MONTHS_BALANCE END) - MIN(MONTHS_BALANCE) AS first_bad_mob
        FROM b GROUP BY SK_ID_BUREAU
    ),
    cohort AS (
        SELECT *, CAST(FLOOR(open_month * 1.0 / {months_per_cohort}) * {months_per_cohort} AS INTEGER) AS cohort_start_month
        FROM acct
    ),
    mobs AS (SELECT UNNEST(range(0, 97)) AS mob)
    SELECT c.cohort_start_month, m.mob,
           COUNT(*) AS accounts_observed,
           SUM(CASE WHEN c.first_bad_mob <= m.mob THEN 1 ELSE 0 END) AS accounts_bad,
           SUM(CASE WHEN c.first_bad_mob <= m.mob THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS cumulative_bad_rate
    FROM cohort c JOIN mobs m ON c.max_mob >= m.mob
    GROUP BY c.cohort_start_month, m.mob
    ORDER BY c.cohort_start_month, m.mob
    """
    try:
        return conn.query(sql)
    finally:
        conn.close()


def build_vintage_reports(data_dir: str | Path = "data/processed") -> dict[str, pl.DataFrame]:
    """Compute both analyses and write them to parquet + csv under data_dir."""
    data_dir = Path(data_dir)
    source = data_dir / "bureau_balance.parquet"
    reports = {
        "roll_rate_matrix": compute_roll_rates(source),
        "vintage_curves": compute_vintage(source),
    }
    for name, frame in reports.items():
        frame.write_parquet(data_dir / f"{name}.parquet")
        frame.write_csv(data_dir / f"{name}.csv")
        logger.info("Wrote %s (%d rows) to %s", name, frame.height, data_dir)
    return reports


class VintageMetrics:
    """Thin class wrapper over the module functions."""

    roll_rate = staticmethod(compute_roll_rates)
    vintage_analysis = staticmethod(compute_vintage)


if __name__ == "__main__":
    build_vintage_reports()
