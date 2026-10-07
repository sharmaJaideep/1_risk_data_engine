"""Roll-rate and vintage analysis on monthly account snapshots.

Three sources are supported, all one row per (account, MONTHS_BALANCE) with
earlier months negative:

- bureau:      bureau_balance, keyed by SK_ID_BUREAU. STATUS is C = closed,
               X = unknown (excluded), 0 = current, 1..5 = DPD buckets
               (1-30, 31-60, 61-90, 91-120, 120+/written off).
- credit_card: credit_card_balance, keyed by SK_ID_PREV; buckets derived from SK_DPD.
- pos_cash:    POS_CASH_balance, keyed by SK_ID_PREV; buckets derived from SK_DPD.

For credit_card and pos_cash, NAME_CONTRACT_STATUS = 'Completed' maps to CLOSED
and SK_DPD days map to the same buckets as bureau STATUS, so matrices are
comparable across sources. "Bad" means 61+ days past due in every source.

Caveat: history is truncated at MONTHS_BALANCE = -96, so an account's first
observed month is its true opening month only if it opened inside that window.
"""
from __future__ import annotations

from dataclasses import dataclass
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
_BUREAU_BUCKET = "CASE STATUS " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in BUCKET_LABELS.items()) + " END"
_DPD_BUCKET = """CASE
    WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 'CLOSED'
    WHEN SK_DPD = 0 THEN 'CURRENT'
    WHEN SK_DPD <= 30 THEN 'DPD_1_30'
    WHEN SK_DPD <= 60 THEN 'DPD_31_60'
    WHEN SK_DPD <= 90 THEN 'DPD_61_90'
    WHEN SK_DPD <= 120 THEN 'DPD_91_120'
    ELSE 'DPD_120_PLUS' END"""


@dataclass(frozen=True)
class Source:
    """How to read one monthly-snapshot table as (acct, MONTHS_BALANCE, bucket, is_bad)."""

    file: str
    id_col: str
    bucket_sql: str
    bad_sql: str  # 61+ days past due
    where_sql: str  # drops rows with no usable delinquency information

    def snapshot_sql(self, path: str) -> str:
        return f"""
        SELECT {self.id_col} AS acct, MONTHS_BALANCE, {self.bucket_sql} AS bucket, ({self.bad_sql}) AS is_bad
        FROM read_parquet('{path}') WHERE {self.where_sql}"""


SOURCES = {
    "bureau": Source("bureau_balance.parquet", "SK_ID_BUREAU", _BUREAU_BUCKET, "STATUS IN ('3', '4', '5')", "STATUS <> 'X'"),
    "credit_card": Source("credit_card_balance.parquet", "SK_ID_PREV", _DPD_BUCKET, "SK_DPD > 60", "SK_DPD IS NOT NULL"),
    "pos_cash": Source("POS_CASH_balance.parquet", "SK_ID_PREV", _DPD_BUCKET, "SK_DPD > 60", "SK_DPD IS NOT NULL"),
}
MONTHS_PER_COHORT = 3


def _connect(path: str | Path) -> tuple[DuckDBEngine, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bureau_balance parquet not found: {path}")
    return DuckDBEngine(":memory:"), str(path)


def compute_roll_rates(snapshot_path: str | Path, source: str = "bureau") -> pl.DataFrame:
    """Month-over-month transition matrix between delinquency buckets.

    Returns source, from_bucket, to_bucket, accounts (transition count) and
    roll_rate (share of the from_bucket's transitions that land in to_bucket).
    """
    conn, path = _connect(snapshot_path)
    sql = f"""
    WITH b AS ({SOURCES[source].snapshot_sql(path)}),
    t AS (
        SELECT cur.bucket AS from_bucket, nxt.bucket AS to_bucket
        FROM b cur
        JOIN b nxt ON nxt.acct = cur.acct AND nxt.MONTHS_BALANCE = cur.MONTHS_BALANCE + 1
    )
    SELECT '{source}' AS source, from_bucket, to_bucket, COUNT(*) AS accounts,
           COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY from_bucket) AS roll_rate
    FROM t GROUP BY from_bucket, to_bucket
    ORDER BY from_bucket, to_bucket
    """
    try:
        return conn.query(sql)
    finally:
        conn.close()


def compute_vintage(
    snapshot_path: str | Path, months_per_cohort: int = MONTHS_PER_COHORT, source: str = "bureau"
) -> pl.DataFrame:
    """Cumulative bad-rate curves by opening cohort and months-on-book (MOB).

    Cohort = account's first observed month bucketed into `months_per_cohort`-month
    groups (cohort_start_month is relative to now, e.g. -12 = 12 months ago).
    For each MOB the rate is accounts that have gone bad (61+ DPD) by then, over
    accounts in the cohort observed at that MOB, so right-censored accounts
    don't dilute later MOBs.
    """
    conn, path = _connect(snapshot_path)
    sql = f"""
    WITH b AS ({SOURCES[source].snapshot_sql(path)}),
    acct AS (
        SELECT acct,
               MIN(MONTHS_BALANCE) AS open_month,
               MAX(MONTHS_BALANCE) - MIN(MONTHS_BALANCE) AS max_mob,
               MIN(CASE WHEN is_bad THEN MONTHS_BALANCE END) - MIN(MONTHS_BALANCE) AS first_bad_mob
        FROM b GROUP BY acct
    ),
    cohort AS (
        SELECT *, CAST(FLOOR(open_month * 1.0 / {months_per_cohort}) * {months_per_cohort} AS INTEGER) AS cohort_start_month
        FROM acct
    ),
    mobs AS (SELECT UNNEST(range(0, 97)) AS mob)
    SELECT '{source}' AS source, c.cohort_start_month, m.mob,
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
    """Compute both analyses for every available source; write parquet + csv under data_dir.

    Each output stacks the sources and carries a `source` column.
    """
    data_dir = Path(data_dir)
    roll_rates, vintages = [], []
    for name, spec in SOURCES.items():
        path = data_dir / spec.file
        if not path.exists():
            logger.warning("%s: %s not found; skipping", name, path)
            continue
        roll_rates.append(compute_roll_rates(path, name))
        vintages.append(compute_vintage(path, source=name))
        logger.info("%s: computed roll rates and vintage curves", name)
    if not roll_rates:
        raise FileNotFoundError(f"No snapshot parquet files found in {data_dir}")

    reports = {"roll_rate_matrix": pl.concat(roll_rates), "vintage_curves": pl.concat(vintages)}
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
