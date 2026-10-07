from pathlib import Path

import polars as pl
import pytest

from src.ingestion import schema_registry as reg
from src.ingestion.schema_drift import compare_versions, run_drift_report
from src.ingestion.schema_validator import SchemaValidator

V1 = {
    "required_columns": ["id"],
    "columns": {"id": {"type": "int", "nullable": False}, "amount": {"type": "int", "nullable": True}},
    "primary_key": ["id"],
}
V2 = {
    "required_columns": ["id"],
    "columns": {
        "id": {"type": "int", "nullable": False},
        "amount": {"type": "float", "nullable": True},  # widened
        "channel": {"type": "string", "nullable": True},  # added
    },
    "primary_key": ["id"],
}


def test_first_version_is_v1_and_history_is_protected(tmp_path):
    path = reg.write_version("t", {"strict_schema": V1}, tmp_path)
    assert path == tmp_path / "t" / "v1.yaml"
    assert SchemaValidator(path).version == 1
    with pytest.raises(FileExistsError):
        reg.write_version("t", {"strict_schema": V2}, tmp_path)  # no silent overwrite
    assert reg.versions("t", tmp_path) == [1]


def test_bump_adds_next_version_and_latest_resolves(tmp_path):
    reg.write_version("t", {"strict_schema": V1}, tmp_path)
    reg.write_version("t", {"strict_schema": V2}, tmp_path, bump=True)
    assert reg.versions("t", tmp_path) == [1, 2]
    assert SchemaValidator.for_table("t", schema_dir=tmp_path).version == 2
    assert SchemaValidator.for_table("t", 1, tmp_path).version == 1
    assert "channel" not in SchemaValidator.for_table("t", 1, tmp_path).schema["strict_schema"]["columns"]


def test_unknown_table_or_version_raises(tmp_path):
    reg.write_version("t", {"strict_schema": V1}, tmp_path)
    with pytest.raises(FileNotFoundError, match="no version 9"):
        reg.schema_path("t", 9, tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.latest_version("missing", tmp_path)


def test_compare_versions_defaults_to_previous_vs_latest(tmp_path):
    reg.write_version("t", {"strict_schema": V1}, tmp_path)
    with pytest.raises(ValueError, match="no version before"):
        compare_versions("t", schema_dir=tmp_path)
    reg.write_version("t", {"strict_schema": V2}, tmp_path, bump=True)
    kinds = {(f.kind, f.column, f.severity) for f in compare_versions("t", schema_dir=tmp_path)}
    assert kinds == {("type_changed", "amount", "warning"), ("column_added", "channel", "warning")}
    assert compare_versions("t", 1, 1, tmp_path) == []


def test_drift_report_uses_latest_version_and_records_it(tmp_path):
    schemas, data = tmp_path / "schemas", tmp_path / "data"
    data.mkdir()
    reg.write_version("t", {"strict_schema": V1}, schemas)
    reg.write_version("t", {"strict_schema": V2}, schemas, bump=True)
    pl.DataFrame({"id": [1, 2], "amount": [1.5, None], "channel": ["a", None]}).write_parquet(data / "t.parquet")
    report = run_drift_report(data, schemas)
    assert report.height == 0  # data matches v2, even though it would drift from v1
    assert "schema_version" in report.columns


def test_every_committed_table_has_v1_and_matches_its_version_field():
    for table in reg.tables():
        for v in reg.versions(table):
            spec = SchemaValidator.for_table(table, v).schema
            assert spec["table_name"] == table and spec["version"] == v
        assert reg.versions(table)[0] == 1
    assert len(reg.tables()) == 8
