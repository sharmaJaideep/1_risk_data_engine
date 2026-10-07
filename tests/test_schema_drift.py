import polars as pl

from src.agents.diagnostic_agent import DiagnosticAgent
from src.ingestion.schema_drift import detect_drift, diff_schemas

BASE = {
    "required_columns": ["id", "status"],
    "columns": {
        "id": {"type": "int", "nullable": False},
        "status": {"type": "string", "nullable": False},
        "amount": {"type": "int", "nullable": True},
        "note": {"type": "string", "nullable": True},
    },
    "primary_key": ["id"],
}


def _kinds(findings):
    return {(f.kind, f.column, f.severity) for f in findings}


def test_diff_classifies_changes():
    observed = {
        "required_columns": ["id"],
        "columns": {
            "id": {"type": "int", "nullable": False},
            "status": {"type": "string", "nullable": True},  # nulls introduced
            "amount": {"type": "float", "nullable": True},  # int -> float widening
            "note": {"type": "int", "nullable": False},  # type break
            "extra": {"type": "string", "nullable": True},  # added
        },
        "primary_key": ["id"],
    }
    kinds = _kinds(diff_schemas("t", BASE, observed))
    assert ("nulls_introduced", "status", "breaking") in kinds
    assert ("type_changed", "amount", "warning") in kinds
    assert ("type_changed", "note", "breaking") in kinds
    assert ("nulls_absent", "note", "info") in kinds
    assert ("column_added", "extra", "warning") in kinds


def test_removed_required_column_is_breaking_optional_is_warning():
    observed = {"columns": {"id": BASE["columns"]["id"], "amount": BASE["columns"]["amount"]}, "primary_key": ["id"]}
    kinds = _kinds(diff_schemas("t", BASE, observed))
    assert ("column_removed", "status", "breaking") in kinds
    assert ("column_removed", "note", "warning") in kinds


def test_no_drift_on_identical_schema():
    assert diff_schemas("t", BASE, BASE) == []


def test_detect_drift_finds_duplicate_keys_and_nulls(tmp_path):
    from src.ingestion.schema_registry import write_version

    write_version("bureau_balance", {"strict_schema": {
        "required_columns": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        "columns": {
            "SK_ID_BUREAU": {"type": "int", "nullable": False},
            "MONTHS_BALANCE": {"type": "int", "nullable": False},
            "STATUS": {"type": "string", "nullable": False},
        },
        "primary_key": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    }}, tmp_path)
    frame = pl.DataFrame({"SK_ID_BUREAU": [1, 1], "MONTHS_BALANCE": [0, 0], "STATUS": ["C", None]})
    kinds = {(f.kind, f.severity) for f in detect_drift("bureau_balance", frame, tmp_path)}
    assert ("primary_key_duplicates", "breaking") in kinds
    assert ("nulls_introduced", "breaking") in kinds


def test_agent_escalates_on_breaking_drift(tmp_path):
    stability = tmp_path / "portfolio_stability_report.parquet"
    pl.DataFrame({"feature_name": ["A"], "psi_score": [0.01], "drift_status": ["STABLE"]}).write_parquet(stability)
    pl.DataFrame(
        [("bureau", "type_changed", "AMT_ANNUITY", "float", "string", "breaking")],
        schema=["table", "kind", "column", "baseline", "observed", "severity"],
        orient="row",
    ).write_parquet(tmp_path / "schema_drift_report.parquet")
    report = DiagnosticAgent(report_path=stability, use_llm=False).run()
    assert report.status == "critical"
    assert any("type_changed bureau.AMT_ANNUITY" in f for f in report.findings)
