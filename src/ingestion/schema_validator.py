from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

SCHEMA_DIR = Path("config/schemas")
_TYPE_CHECKS = {
    "int": lambda dtype: dtype.is_integer(),
    "float": lambda dtype: dtype.is_float(),
    "string": lambda dtype: dtype == pl.String,
}


class SchemaValidator:
    """Validate input columns and data against a YAML schema definition."""

    def __init__(self, schema_path: str | Path | None = None) -> None:
        self.schema_path = Path(schema_path or SCHEMA_DIR / "application_train.yaml")
        self.schema = self._load_schema()

    @classmethod
    def for_table(cls, table: str, schema_dir: str | Path = SCHEMA_DIR) -> "SchemaValidator":
        """Load the schema for a table from config/schemas/<table>.yaml."""
        return cls(Path(schema_dir) / f"{table}.yaml")

    def _load_schema(self) -> dict[str, Any]:
        with self.schema_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def validate_columns(self, columns: list[str]) -> bool:
        strict_schema = self.schema["strict_schema"]
        expected = set(strict_schema["columns"].keys())
        provided = set(columns)
        required_columns = set(strict_schema.get("required_columns", []))

        if not required_columns.issubset(provided):
            return False

        return provided.issubset(expected)

    def validate_frame(self, frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
        """Check columns, types, nullability and primary-key uniqueness.

        Returns a list of human-readable violations (empty means valid).
        """
        spec = self.schema["strict_schema"]
        columns: dict[str, dict[str, Any]] = spec["columns"]
        lf = frame.lazy()
        actual = lf.collect_schema()
        errors: list[str] = []

        missing = [c for c in spec.get("required_columns", []) if c not in actual]
        unexpected = [c for c in actual if c not in columns]
        if missing:
            errors.append(f"missing required columns: {missing}")
        if unexpected:
            errors.append(f"unexpected columns: {unexpected}")

        for name, dtype in actual.items():
            if name not in columns:
                continue
            expected_type = columns[name]["type"]
            if not _TYPE_CHECKS[expected_type](dtype):
                errors.append(f"{name}: expected {expected_type}, found {dtype}")

        strict_cols = [c for c, s in columns.items() if not s.get("nullable", True) and c in actual]
        pk = [c for c in spec.get("primary_key") or [] if c in actual]
        has_pk = bool(spec.get("primary_key")) and len(pk) == len(spec["primary_key"])
        exprs = [pl.col(c).null_count().alias(f"nulls::{c}") for c in strict_cols]
        if has_pk:
            exprs.append(pl.struct(pk).n_unique().alias("pk::unique"))
            exprs.append(pl.len().alias("pk::rows"))
        if exprs:
            stats = lf.select(exprs).collect().row(0, named=True)
            errors += [f"{c}: {stats[f'nulls::{c}']} nulls in non-nullable column" for c in strict_cols if stats[f"nulls::{c}"]]
            if has_pk and stats["pk::unique"] != stats["pk::rows"]:
                errors.append(f"primary key {pk} has {stats['pk::rows'] - stats['pk::unique']} duplicate rows")
        return errors


def type_name(dtype: pl.DataType) -> str:
    """Map a polars dtype to the schema type vocabulary (int / float / string)."""
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if dtype == pl.String:
        return "string"
    raise ValueError(f"Unsupported dtype {dtype}")


def infer_schema(table: str, frame: pl.DataFrame | pl.LazyFrame, primary_key: list[str]) -> dict[str, Any]:
    """Describe a frame in the schema YAML format.

    A column is `nullable: false` only if no nulls are observed.
    """
    lf = frame.lazy()
    schema = lf.collect_schema()
    null_counts = lf.select(pl.all().null_count()).collect().row(0, named=True)
    columns = {c: {"type": type_name(t), "nullable": null_counts[c] > 0} for c, t in schema.items()}
    return {
        "table_name": table,
        "strict_schema": {
            "required_columns": [c for c, spec in columns.items() if not spec["nullable"]],
            "columns": columns,
            "primary_key": primary_key,
        },
    }
