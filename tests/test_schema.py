from pathlib import Path

from src.ingestion.schema_validator import SchemaValidator


def test_schema_validator_detects_expected_columns() -> None:
    validator = SchemaValidator.for_table("application_train")
    columns = list(validator.schema["strict_schema"]["columns"])
    assert validator.validate_columns(columns) is True
    assert validator.validate_columns([*columns, "UNKNOWN_COLUMN"]) is False


def test_schema_validator_rejects_missing_columns() -> None:
    validator = SchemaValidator.for_table("application_train")
    columns = ["SK_ID_CURR", "TARGET"]
    assert validator.validate_columns(columns) is False


import polars as pl
import pytest

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "credit_card_balance",
    "installments_payments",
    "POS_CASH_balance",
    "previous_application",
]


@pytest.mark.parametrize("table", TABLES)
def test_table_schema_loads_with_valid_structure(table):
    spec = SchemaValidator.for_table(table).schema["strict_schema"]
    assert set(spec["required_columns"]) <= set(spec["columns"])
    assert set(spec["primary_key"]) <= set(spec["columns"])
    assert {c["type"] for c in spec["columns"].values()} <= {"int", "float", "string"}


@pytest.mark.parametrize("table", TABLES)
def test_processed_data_conforms_to_schema(table):
    path = Path("data/processed") / f"{table}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not generated")
    assert SchemaValidator.for_table(table).validate_frame(pl.scan_parquet(path)) == []


def test_validate_frame_reports_violations():
    validator = SchemaValidator.for_table("bureau_balance")
    frame = pl.DataFrame({"SK_ID_BUREAU": [1, 1, None], "MONTHS_BALANCE": [0.0, 0.0, 1.5], "EXTRA": [1, 2, 3]})
    errors = " | ".join(validator.validate_frame(frame))
    assert "missing required columns: ['STATUS']" in errors
    assert "unexpected columns: ['EXTRA']" in errors
    assert "MONTHS_BALANCE: expected int" in errors
    assert "SK_ID_BUREAU: 1 nulls" in errors
