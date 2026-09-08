"""Hierarchical Aggregation Engine (Phase 2)

Builds Level-2 (monthly) and Level-1 (account) rollups using DuckDB
zero-copy SQL over Parquet in `data/processed/` and writes a single
Master Analytical Record (MAR) to
`data/processed/master_analytical_record.parquet`.

This module is intended to be run as a script and also provides the
`build_master_analytical_record` function for programmatic use.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import duckdb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _safe_parquet(path: str) -> str:
    return str(Path(path))


def build_master_analytical_record(data_dir: str = "data/processed") -> None:
    """Builds the Master Analytical Record (MAR) and writes it out.

    Parameters
    - data_dir: location of processed parquet files
    """
    data_dir = Path(data_dir)
    conn = duckdb.connect(database=":memory:")

    # File paths
    app_fp = data_dir / "application_train.parquet"
    bureau_fp = data_dir / "bureau.parquet"
    bureau_balance_fp = data_dir / "bureau_balance.parquet"
    installments_fp = data_dir / "installments_payments.parquet"
    previous_fp = data_dir / "previous_application.parquet"
    out_fp = data_dir / "master_analytical_record.parquet"

    # Warn for missing inputs
    inputs: List[Path] = [app_fp, bureau_fp, bureau_balance_fp, installments_fp, previous_fp]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        logger.warning("Missing input parquet(s): %s", ", ".join(str(p) for p in missing))

    # Level-2: bureau_balance by SK_ID_BUREAU
    if bureau_balance_fp.exists():
        sql_bb = f"""
        CREATE TEMP VIEW bureau_balance_lv2 AS
        SELECT
            SK_ID_BUREAU,
            COUNT(*) AS bb_snapshot_months,
            MAX(CASE WHEN TRY_CAST(STATUS AS INTEGER) IS NOT NULL THEN CAST(STATUS AS INTEGER) ELSE 0 END) AS bb_max_status,
            SUM(CASE WHEN TRY_CAST(STATUS AS INTEGER) > 0 THEN 1 ELSE 0 END) AS bb_delinquent_months
        FROM read_parquet('{_safe_parquet(str(bureau_balance_fp))}')
        GROUP BY SK_ID_BUREAU;
        """
        conn.execute(sql_bb)
        logger.info("Created Level-2 bureau_balance aggregation (by SK_ID_BUREAU)")
    else:
        logger.info("bureau_balance.parquet not found; skipping Level-2 bureau_balance aggregation")

    # Level-2: installments_payments by SK_ID_CURR
    if installments_fp.exists():
        sql_ip = f"""
        CREATE TEMP VIEW installments_lv2 AS
        SELECT
            SK_ID_CURR,
            MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS ip_max_payment_delay_days,
            SUM(COALESCE(AMT_INSTALMENT - AMT_PAYMENT, 0)) AS ip_total_payment_deficit,
            COUNT(*) AS ip_total_payments,
            SUM(CASE WHEN (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) > 0 THEN 1 ELSE 0 END) AS ip_late_payments
        FROM read_parquet('{_safe_parquet(str(installments_fp))}')
        GROUP BY SK_ID_CURR;
        """
        conn.execute(sql_ip)
        logger.info("Created Level-2 installments aggregation (by SK_ID_CURR)")
    else:
        logger.info("installments_payments.parquet not found; skipping Level-2 installments aggregation")

    # Level-1: bureau (accounts) aggregated by SK_ID_CURR incorporating Level-2
    if bureau_fp.exists():
        sql_bu = f"""
        CREATE TEMP VIEW bureau_lv1 AS
        SELECT
            b.SK_ID_CURR,
            COUNT(DISTINCT b.SK_ID_BUREAU) AS bureau_total_accounts,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS bureau_active_accounts,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS bureau_closed_accounts,
            SUM(COALESCE(b.AMT_CREDIT_SUM_DEBT, 0)) AS bureau_sum_credit_sum_debt,
            SUM(COALESCE(b.AMT_CREDIT_SUM, 0)) AS bureau_sum_credit_sum,
            SUM(COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0)) AS bureau_total_overdue_balance,
            SUM(COALESCE(bb.bb_delinquent_months, 0)) AS bureau_total_delinquent_months
        FROM read_parquet('{_safe_parquet(str(bureau_fp))}') b
        LEFT JOIN (SELECT * FROM bureau_balance_lv2) bb ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
        GROUP BY b.SK_ID_CURR;
        """
        conn.execute(sql_bu)
        logger.info("Created Level-1 bureau aggregation (by SK_ID_CURR)")
    else:
        logger.info("bureau.parquet not found; skipping Level-1 bureau aggregation")

    # Level-1: previous_application by SK_ID_CURR
    if previous_fp.exists():
        sql_prev = f"""
        CREATE TEMP VIEW previous_lv1 AS
        SELECT
            SK_ID_CURR,
            COUNT(*) AS prev_total_applications,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_approved_applications,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS prev_refused_applications,
            AVG(COALESCE(AMT_APPLICATION, 0)) AS prev_mean_amt_application
        FROM read_parquet('{_safe_parquet(str(previous_fp))}')
        GROUP BY SK_ID_CURR;
        """
        conn.execute(sql_prev)
        logger.info("Created Level-1 previous_application aggregation (by SK_ID_CURR)")
    else:
        logger.info("previous_application.parquet not found; skipping Level-1 previous_application aggregation")

    # Build Master Analytical Record by left-joining onto application_train
    if not app_fp.exists():
        logger.error("application_train.parquet not found in %s; aborting MAR build", str(data_dir))
        conn.close()
        return

    select_cols = ["a.*"]

    aggs = {
        'bureau_lv1': [
            'COALESCE(bureau_total_accounts,0) AS bureau_total_accounts',
            'COALESCE(bureau_active_accounts,0) AS bureau_active_accounts',
            'COALESCE(bureau_closed_accounts,0) AS bureau_closed_accounts',
            'COALESCE(bureau_sum_credit_sum_debt,0) AS bureau_sum_credit_sum_debt',
            'COALESCE(bureau_sum_credit_sum,0) AS bureau_sum_credit_sum',
            'COALESCE(bureau_total_overdue_balance,0) AS bureau_total_overdue_balance',
            'COALESCE(bureau_total_delinquent_months,0) AS bureau_total_delinquent_months'
        ],
        'installments_lv2': [
            'COALESCE(ip_max_payment_delay_days,0) AS ip_max_payment_delay_days',
            'COALESCE(ip_total_payment_deficit,0) AS ip_total_payment_deficit',
            'COALESCE(ip_total_payments,0) AS ip_total_payments',
            'COALESCE(ip_late_payments,0) AS ip_late_payments'
        ],
        'previous_lv1': [
            'COALESCE(prev_total_applications,0) AS prev_total_applications',
            'COALESCE(prev_approved_applications,0) AS prev_approved_applications',
            'COALESCE(prev_refused_applications,0) AS prev_refused_applications',
            'COALESCE(prev_mean_amt_application,0) AS prev_mean_amt_application'
        ]
    }

    for cols in aggs.values():
        select_cols.extend(cols)

    select_sql = f"SELECT {', '.join(select_cols)} FROM read_parquet('{_safe_parquet(str(app_fp))}') a"

    if bureau_fp.exists():
        select_sql += " LEFT JOIN (SELECT * FROM bureau_lv1) b ON a.SK_ID_CURR = b.SK_ID_CURR"
    if installments_fp.exists():
        select_sql += " LEFT JOIN (SELECT * FROM installments_lv2) ip ON a.SK_ID_CURR = ip.SK_ID_CURR"
    if previous_fp.exists():
        select_sql += " LEFT JOIN (SELECT * FROM previous_lv1) p ON a.SK_ID_CURR = p.SK_ID_CURR"

    out_path_str = _safe_parquet(str(out_fp))
    copy_sql = f"COPY ({select_sql}) TO '{out_path_str}' (FORMAT PARQUET, COMPRESSION 'ZSTD')"
    logger.info("Writing Master Analytical Record to %s", out_path_str)
    conn.execute(copy_sql)

    try:
        rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path_str}')").fetchone()[0]
        cols = len(conn.execute(f"SELECT * FROM read_parquet('{out_path_str}') LIMIT 1").description)
    except Exception:
        rows = None
        cols = None

    logger.info("MAR written: %s rows x %s cols", rows, cols)

    conn.close()


if __name__ == '__main__':
    start = time.time()
    root = Path("data/processed")
    candidates = [
        root / "application_train.parquet",
        root / "bureau.parquet",
        root / "bureau_balance.parquet",
        root / "installments_payments.parquet",
        root / "previous_application.parquet",
    ]
    input_sizes = {}
    for p in candidates:
        if p.exists():
            input_sizes[str(p.name)] = p.stat().st_size

    build_master_analytical_record(str(root))

    out_fp = root / "master_analytical_record.parquet"
    out_size = out_fp.stat().st_size if out_fp.exists() else 0
    elapsed = time.time() - start
    logger.info("Execution time: %.2f seconds", elapsed)
    logger.info("Input sizes (bytes): %s", input_sizes)
    logger.info("Output size (bytes): %s", out_size)
