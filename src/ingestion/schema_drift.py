"""Schema drift detection.

Compares what a table looks like now against its baseline schema in
config/schemas/, and classifies every difference:

- breaking: downstream code or models may fail or silently misbehave
  (removed required column, type change, nulls in a non-nullable column,
  duplicate primary keys)
- warning:  worth review (new column, removed optional column, int -> float widening)
- info:     benign tightening (a nullable column is now never null)

`diff_schemas` compares any two schema specs; `compare_versions` compares two
stored versions of a table (e.g. to review a new version before adopting it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.ingestion import schema_registry
from src.ingestion.schema_registry import SCHEMA_DIR
from src.ingestion.schema_validator import SchemaValidator, infer_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = {"info": 0, "warning": 1, "breaking": 2}


@dataclass(frozen=True)
class DriftFinding:
    table: str
    kind: str
    column: str
    baseline: str
    observed: str
    severity: str


def diff_schemas(table: str, baseline: dict[str, Any], observed: dict[str, Any]) -> list[DriftFinding]:
    """Diff two `strict_schema` specs (baseline -> observed)."""
    base_cols, obs_cols = baseline["columns"], observed["columns"]
    required = set(baseline.get("required_columns", []))
    findings: list[DriftFinding] = []

    def add(kind: str, column: str, base: str, obs: str, severity: str) -> None:
        findings.append(DriftFinding(table, kind, column, base, obs, severity))

    for col in base_cols.keys() - obs_cols.keys():
        add("column_removed", col, "present", "absent", "breaking" if col in required else "warning")
    for col in obs_cols.keys() - base_cols.keys():
        add("column_added", col, "absent", obs_cols[col]["type"], "warning")

    for col in base_cols.keys() & obs_cols.keys():
        b, o = base_cols[col], obs_cols[col]
        if b["type"] != o["type"]:
            widening = (b["type"], o["type"]) == ("int", "float")
            add("type_changed", col, b["type"], o["type"], "warning" if widening else "breaking")
        if not b["nullable"] and o["nullable"]:
            add("nulls_introduced", col, "non-nullable", "nullable", "breaking")
        elif b["nullable"] and not o["nullable"]:
            add("nulls_absent", col, "nullable", "no nulls observed", "info")

    if baseline.get("primary_key") != observed.get("primary_key"):
        add("primary_key_changed", "", str(baseline.get("primary_key")), str(observed.get("primary_key")), "breaking")

    findings.sort(key=lambda f: (-SEVERITY_ORDER[f.severity], f.kind, f.column))
    return findings


def compare_versions(
    table: str, from_version: int | None = None, to_version: int | None = None, schema_dir: str | Path = SCHEMA_DIR
) -> list[DriftFinding]:
    """Diff two stored versions of a table's schema (defaults: previous latest -> latest)."""
    available = schema_registry.versions(table, schema_dir)
    if to_version is None:
        to_version = available[-1] if available else None
    if from_version is None:
        earlier = [v for v in available if to_version is not None and v < to_version]
        if not earlier:
            raise ValueError(f"'{table}' has no version before v{to_version} to compare against (versions: {available})")
        from_version = earlier[-1]
    old = SchemaValidator.for_table(table, from_version, schema_dir).schema["strict_schema"]
    new = SchemaValidator.for_table(table, to_version, schema_dir).schema["strict_schema"]
    return diff_schemas(table, old, new)


def detect_drift(
    table: str,
    frame: pl.DataFrame | pl.LazyFrame,
    schema_dir: str | Path = SCHEMA_DIR,
    version: int | None = None,
) -> list[DriftFinding]:
    """Compare a frame against a table's schema (latest version unless `version` is given)."""
    baseline = SchemaValidator.for_table(table, version, schema_dir).schema["strict_schema"]
    pk = baseline.get("primary_key") or []
    observed = infer_schema(table, frame, pk)["strict_schema"]
    findings = diff_schemas(table, baseline, observed)

    # Key uniqueness is a data property, not something the inferred schema carries.
    lf = frame.lazy()
    if pk and all(c in lf.collect_schema() for c in pk):
        rows, unique = lf.select(pl.len(), pl.struct(pk).n_unique().alias("u")).collect().row(0)
        if rows != unique:
            findings.append(
                DriftFinding(table, "primary_key_duplicates", ",".join(pk), "unique", f"{rows - unique} duplicate rows", "breaking")
            )
    return findings


def run_drift_report(data_dir: str | Path = "data/processed", schema_dir: str | Path = SCHEMA_DIR) -> pl.DataFrame:
    """Detect drift for every table that has both a schema and a parquet file; write the report."""
    data_dir = Path(data_dir)
    findings: list[DriftFinding] = []
    versions_used: dict[str, int] = {}
    for table in schema_registry.tables(schema_dir):
        parquet = data_dir / f"{table}.parquet"
        if not parquet.exists():
            logger.warning("%s: no parquet to check for drift", table)
            continue
        versions_used[table] = schema_registry.latest_version(table, schema_dir)
        table_findings = detect_drift(table, pl.scan_parquet(parquet), schema_dir)
        logger.info("%s (schema v%d): %d drift finding(s)", table, versions_used[table], len(table_findings))
        findings += table_findings

    report = pl.DataFrame(
        [asdict(f) for f in findings],
        schema={k: pl.String for k in DriftFinding.__dataclass_fields__},
    ).with_columns(pl.col("table").replace_strict(versions_used, default=None, return_dtype=pl.Int64).alias("schema_version"))
    report.write_parquet(data_dir / "schema_drift_report.parquet")
    report.write_csv(data_dir / "schema_drift_report.csv")
    logger.info("Wrote schema drift report (%d findings) to %s", report.height, data_dir)
    return report
