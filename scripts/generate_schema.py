"""Draft a table schema YAML from a processed parquet file.

Types come from the parquet schema; a column is `nullable: false` only if no nulls
are observed. Review the output before committing: it describes the data you
have today, not necessarily the contract you want.

Usage: python scripts/generate_schema.py <table> <pk_col> [<pk_col> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.schema_validator import infer_schema  # noqa: E402

if __name__ == "__main__":
    table, *pk = sys.argv[1:]
    schema = infer_schema(table, pl.scan_parquet(Path("data/processed") / f"{table}.parquet"), pk)
    out = Path("config/schemas") / f"{table}.yaml"
    out.write_text(yaml.safe_dump(schema, sort_keys=False))
    print(f"Wrote {out}")
