"""Benchmark the Master Analytical Record build: pandas vs the DuckDB pipeline.

Both implementations compute the same 15 aggregates (bureau + bureau_balance, installments,
previous applications) and left-join them onto application_train. Each run executes in a
fresh subprocess so peak memory (RSS high-water mark) is isolated, and the pandas output is
checked against the pipeline output so the comparison is like-for-like.

Usage: python scripts/benchmark_pipeline.py [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["application_train", "bureau", "bureau_balance", "installments_payments", "previous_application"]
OUT_NAME = "master_analytical_record.parquet"


def run_pandas(data_dir: Path) -> None:
    import pandas as pd

    bb = pd.read_parquet(data_dir / "bureau_balance.parquet", columns=["SK_ID_BUREAU", "STATUS"])
    bb["delinq"] = pd.to_numeric(bb["STATUS"], errors="coerce").fillna(0) > 0
    bb = bb.groupby("SK_ID_BUREAU", as_index=False)["delinq"].sum().rename(columns={"delinq": "bb_delinq"})

    bureau = pd.read_parquet(
        data_dir / "bureau.parquet",
        columns=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_OVERDUE"],
    )
    bureau = bureau.merge(bb, on="SK_ID_BUREAU", how="left")
    del bb
    bureau["active"] = bureau["CREDIT_ACTIVE"] == "Active"
    bureau["closed"] = bureau["CREDIT_ACTIVE"] == "Closed"
    g = bureau.groupby("SK_ID_CURR")
    bureau_agg = pd.DataFrame(
        {
            "bureau_total_accounts": g["SK_ID_BUREAU"].nunique(),
            "bureau_active_accounts": g["active"].sum(),
            "bureau_closed_accounts": g["closed"].sum(),
            "bureau_sum_credit_sum_debt": bureau["AMT_CREDIT_SUM_DEBT"].fillna(0).groupby(bureau["SK_ID_CURR"]).sum(),
            "bureau_sum_credit_sum": bureau["AMT_CREDIT_SUM"].fillna(0).groupby(bureau["SK_ID_CURR"]).sum(),
            "bureau_total_overdue_balance": bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0).groupby(bureau["SK_ID_CURR"]).sum(),
            "bureau_total_delinquent_months": bureau["bb_delinq"].fillna(0).groupby(bureau["SK_ID_CURR"]).sum(),
        }
    )
    del bureau

    ip = pd.read_parquet(
        data_dir / "installments_payments.parquet",
        columns=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    )
    ip["delay"] = ip["DAYS_ENTRY_PAYMENT"] - ip["DAYS_INSTALMENT"]
    ip["deficit"] = (ip["AMT_INSTALMENT"] - ip["AMT_PAYMENT"]).fillna(0)
    ip["late"] = ip["delay"] > 0
    g = ip.groupby("SK_ID_CURR")
    ip_agg = pd.DataFrame(
        {
            "ip_max_payment_delay_days": g["delay"].max(),
            "ip_total_payment_deficit": g["deficit"].sum(),
            "ip_total_payments": g["delay"].size(),
            "ip_late_payments": g["late"].sum(),
        }
    )
    del ip

    prev = pd.read_parquet(
        data_dir / "previous_application.parquet", columns=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION"]
    )
    prev["approved"] = prev["NAME_CONTRACT_STATUS"] == "Approved"
    prev["refused"] = prev["NAME_CONTRACT_STATUS"] == "Refused"
    prev["amt"] = prev["AMT_APPLICATION"].fillna(0)
    g = prev.groupby("SK_ID_CURR")
    prev_agg = pd.DataFrame(
        {
            "prev_total_applications": g["amt"].size(),
            "prev_approved_applications": g["approved"].sum(),
            "prev_refused_applications": g["refused"].sum(),
            "prev_mean_amt_application": g["amt"].mean(),
        }
    )
    del prev

    app = pd.read_parquet(data_dir / "application_train.parquet")
    mar = app.join(bureau_agg, on="SK_ID_CURR").join(ip_agg, on="SK_ID_CURR").join(prev_agg, on="SK_ID_CURR")
    agg_cols = [*bureau_agg.columns, *ip_agg.columns, *prev_agg.columns]
    mar[agg_cols] = mar[agg_cols].fillna(0)
    mar.to_parquet(data_dir / OUT_NAME, compression="zstd")


def run_duckdb(data_dir: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from src.metrics.aggregators import build_master_analytical_record

    build_master_analytical_record(str(data_dir))


def child(impl: str) -> None:
    """Run one implementation in this process and print wall time + peak RSS as JSON."""
    import logging

    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in INPUTS:
            (work / f"{name}.parquet").symlink_to(ROOT / "data" / "processed" / f"{name}.parquet")
        start = time.perf_counter()
        {"pandas": run_pandas, "duckdb": run_duckdb}[impl](work)
        seconds = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # macOS reports bytes, Linux KB
        import polars as pl

        out = work / OUT_NAME
        rows, cols = pl.scan_parquet(out).select(pl.len()).collect().item(), len(pl.scan_parquet(out).collect_schema())
        saved = ROOT / "data" / "processed" / f"benchmark_{impl}.parquet"
        saved.write_bytes(out.read_bytes())
    print(json.dumps({"impl": impl, "seconds": seconds, "peak_mb": peak_mb, "rows": rows, "cols": cols}))


def verify() -> str:
    import polars as pl
    from polars.testing import assert_frame_equal

    a = pl.read_parquet(ROOT / "data/processed/benchmark_pandas.parquet").sort("SK_ID_CURR")
    b = pl.read_parquet(ROOT / "data/processed/benchmark_duckdb.parquet").sort("SK_ID_CURR")
    b = b.select(a.columns)
    num = [c for c, t in a.schema.items() if t.is_numeric()]
    assert_frame_equal(
        a.select(num).cast({c: pl.Float64 for c in num}), b.select(num).cast({c: pl.Float64 for c in num}),
        check_exact=False, rel_tol=1e-6, abs_tol=1e-4, check_dtypes=False,
    )
    return f"outputs match ({a.height} rows x {len(a.columns)} cols, numeric columns within tolerance)"


def driver(runs: int) -> None:
    results: dict[str, list[dict]] = {"pandas": [], "duckdb": []}
    for i in range(runs):
        for impl in ("pandas", "duckdb"):  # interleaved so neither benefits from a warmer page cache
            proc = subprocess.run([sys.executable, __file__, "--child", impl], capture_output=True, text=True, cwd=ROOT)
            if proc.returncode:
                sys.exit(f"{impl} run failed:\n{proc.stderr[-2000:]}")
            results[impl].append(json.loads(proc.stdout.strip().splitlines()[-1]))
            r = results[impl][-1]
            print(f"run {i + 1} {impl:7s} {r['seconds']:7.2f}s  peak {r['peak_mb']:8.0f} MB")
    check = verify()
    summary = {
        impl: {
            "median_seconds": statistics.median(r["seconds"] for r in rs),
            "median_peak_mb": statistics.median(r["peak_mb"] for r in rs),
        }
        for impl, rs in results.items()
    }
    p, d = summary["pandas"], summary["duckdb"]
    summary["speedup_x"] = p["median_seconds"] / d["median_seconds"]
    summary["memory_reduction_pct"] = 100 * (1 - d["median_peak_mb"] / p["median_peak_mb"])
    summary["verification"] = check
    summary["machine"] = {"platform": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()}
    summary["runs"] = runs
    (ROOT / "data/processed/benchmark_results.json").write_text(json.dumps(summary, indent=2))
    for name in ("pandas", "duckdb"):
        (ROOT / f"data/processed/benchmark_{name}.parquet").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--child", choices=["pandas", "duckdb"])
    args = ap.parse_args()
    child(args.child) if args.child else driver(args.runs)
