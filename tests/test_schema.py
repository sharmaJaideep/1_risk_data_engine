from pathlib import Path

from src.ingestion.schema_validator import SchemaValidator


def test_schema_validator_detects_expected_columns() -> None:
    validator = SchemaValidator(Path("config/schema_config.yaml"))
    columns = ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL"]
    assert validator.validate_columns(columns) is True


def test_schema_validator_rejects_missing_columns() -> None:
    validator = SchemaValidator(Path("config/schema_config.yaml"))
    columns = ["SK_ID_CURR", "TARGET"]
    assert validator.validate_columns(columns) is False
