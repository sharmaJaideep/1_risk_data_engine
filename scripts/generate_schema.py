"""Draft a table schema YAML from a processed parquet file.

Types come from the parquet schema; a column is `nullable: false` only if no nulls
are observed. Review the output before committing: it describes the data you
have today, not necessarily the contract you want.

Writes config/schemas/<table>/v<N>.yaml. An existing table is never overwritten:
pass --bump to add the next version (and print what changed from the last one).

Usage: python scripts/generate_schema.py [--bump] <table> [<pk_col> ...]
       (pk columns default to the table's current primary key when bumping)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import schema_registry  # noqa: E402
from src.ingestion.schema_drift import compare_versions  # noqa: E402
from src.ingestion.schema_validator import SchemaValidator, infer_schema  # noqa: E402

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--bump"]
    bump = "--bump" in sys.argv[1:]
    table, pk = args[0], args[1:]
    if bump and not pk:
        pk = SchemaValidator.for_table(table).schema["strict_schema"]["primary_key"] or []
    schema = infer_schema(table, pl.scan_parquet(Path("data/processed") / f"{table}.parquet"), pk)
    path = schema_registry.write_version(table, schema, bump=bump)
    print(f"Wrote {path}")
    if bump:
        for f in compare_versions(table):
            print(f"  [{f.severity}] {f.kind} {f.column}: {f.baseline} -> {f.observed}")
