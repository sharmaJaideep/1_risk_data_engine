import logging
import sys

import polars as pl
import pytest

import main as cli
from src.ingestion import schema_registry


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    cli.main()


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    return tmp_path


def _bureau_balance(path, rows):
    pl.DataFrame(rows, schema=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], orient="row").write_parquet(path)


# ---- orchestration -----------------------------------------------------------------------

def test_run_all_runs_every_step_in_order(monkeypatch, data_dir):
    calls = []
    monkeypatch.setattr(cli, "run_step", lambda step, d, a: calls.append(step))
    run_cli(monkeypatch, "run-all", "--data-dir", str(data_dir))
    assert calls == ["ingest", "drift", "validate", "aggregate", "stability", "vintage", "diagnose"]


def test_skip_ingest_drops_only_ingest(monkeypatch, data_dir):
    calls = []
    monkeypatch.setattr(cli, "run_step", lambda step, d, a: calls.append(step))
    run_cli(monkeypatch, "run-all", "--skip-ingest", "--data-dir", str(data_dir))
    assert "ingest" not in calls and calls[0] == "drift" and len(calls) == 6


def test_single_step_runs_alone_with_the_given_data_dir(monkeypatch, data_dir):
    seen = []
    monkeypatch.setattr(cli, "run_step", lambda step, d, a: seen.append((step, d, a.no_llm)))
    run_cli(monkeypatch, "vintage", "--data-dir", str(data_dir), "--no-llm")
    assert seen == [("vintage", data_dir, True)]


def test_unknown_command_is_rejected(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "not-a-step")
    assert exc.value.code == 2


# ---- individual steps (real code, tiny data) --------------------------------------------

def test_ingest_step_converts_raw_csvs(monkeypatch, data_dir):
    (data_dir / "raw" / "t.csv").write_text("id,v\n1,2\n")
    run_cli(monkeypatch, "ingest", "--data-dir", str(data_dir))
    assert pl.read_parquet(data_dir / "processed" / "t.parquet").to_dicts() == [{"id": 1, "v": 2}]


def test_vintage_step_writes_both_reports(monkeypatch, data_dir):
    _bureau_balance(data_dir / "processed" / "bureau_balance.parquet", [(1, -2, "0"), (1, -1, "1"), (1, 0, "3")])
    run_cli(monkeypatch, "vintage", "--data-dir", str(data_dir))
    for name in ("roll_rate_matrix", "vintage_curves"):
        assert (data_dir / "processed" / f"{name}.parquet").exists()
        assert (data_dir / "processed" / f"{name}.csv").exists()


def test_aggregate_step_builds_master_record(monkeypatch, data_dir):
    pl.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}).write_parquet(data_dir / "processed" / "application_train.parquet")
    run_cli(monkeypatch, "aggregate", "--data-dir", str(data_dir))
    assert pl.read_parquet(data_dir / "processed" / "master_analytical_record.parquet").height == 2


def test_stability_step_tolerates_missing_inputs_and_still_writes_a_report(monkeypatch, data_dir):
    run_cli(monkeypatch, "stability", "--data-dir", str(data_dir))
    report = pl.read_parquet(data_dir / "processed" / "portfolio_stability_report.parquet")
    assert report.height == 0 and "psi_score" in report.columns


def test_drift_step_logs_each_finding(monkeypatch, data_dir, caplog):
    # STATUS stored as an int instead of a string, and no other columns: both are drift
    pl.DataFrame({"SK_ID_BUREAU": [1], "MONTHS_BALANCE": [0], "STATUS": [5]}).write_parquet(data_dir / "processed" / "bureau_balance.parquet")
    with caplog.at_level(logging.WARNING):
        run_cli(monkeypatch, "drift", "--data-dir", str(data_dir))
    assert "[breaking] type_changed bureau_balance.STATUS" in caplog.text
    report = pl.read_parquet(data_dir / "processed" / "schema_drift_report.parquet")
    assert report.filter(pl.col("kind") == "type_changed").height == 1


def test_validate_step_passes_on_conforming_data(monkeypatch, data_dir, caplog):
    _bureau_balance(data_dir / "processed" / "bureau_balance.parquet", [(1, 0, "C"), (1, -1, "0")])
    with caplog.at_level(logging.INFO):
        run_cli(monkeypatch, "validate", "--data-dir", str(data_dir))
    assert "bureau_balance (v1): OK" in caplog.text and "no parquet to validate" in caplog.text


def test_validate_step_exits_nonzero_on_violations(monkeypatch, data_dir):
    _bureau_balance(data_dir / "processed" / "bureau_balance.parquet", [(1, 0, "C"), (1, 0, "0")])  # duplicate key
    with pytest.raises(SystemExit, match="Schema validation failed"):
        run_cli(monkeypatch, "validate", "--data-dir", str(data_dir))


def test_diagnose_step_writes_and_prints_the_brief(monkeypatch, data_dir, capsys):
    pl.DataFrame(
        {"analysis": ["train_vs_test"], "feature_name": ["A"], "feature_type": ["numeric"], "role": ["characteristic"],
         "psi_score": [0.5], "drift_status": ["SIGNIFICANT_DRIFT"], "top_driver": ["(0, 1]"], "alerting": [True]}
    ).write_parquet(data_dir / "processed" / "portfolio_stability_report.parquet")
    run_cli(monkeypatch, "diagnose", "--no-llm", "--data-dir", str(data_dir))
    brief = (data_dir / "processed" / "diagnostic_brief.md").read_text()
    assert "**Status:** CRITICAL" in brief and "[train_vs_test] A" in brief
    assert "**Status:** CRITICAL" in capsys.readouterr().out


# ---- schema-diff -------------------------------------------------------------------------

def _two_versions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schema_dir = tmp_path / "config" / "schemas"
    base = {"required_columns": ["id"], "columns": {"id": {"type": "int", "nullable": False}}, "primary_key": ["id"]}
    schema_registry.write_version("t", {"strict_schema": base}, schema_dir)
    grown = {**base, "columns": {**base["columns"], "extra": {"type": "string", "nullable": True}}}
    schema_registry.write_version("t", {"strict_schema": grown}, schema_dir, bump=True)


def test_schema_diff_prints_differences(monkeypatch, tmp_path, capsys):
    _two_versions(tmp_path, monkeypatch)
    run_cli(monkeypatch, "schema-diff", "--table", "t")
    out = capsys.readouterr().out
    assert "[warning] column_added extra: absent -> string" in out and "1 difference(s)" in out


def test_schema_diff_explicit_versions_and_identical_versions(monkeypatch, tmp_path, capsys):
    _two_versions(tmp_path, monkeypatch)
    run_cli(monkeypatch, "schema-diff", "--table", "t", "--from-version", "1", "--to-version", "1")
    assert "0 difference(s)" in capsys.readouterr().out


def test_schema_diff_requires_a_table(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "schema-diff")
    assert exc.value.code == 2


def test_schema_diff_reports_a_friendly_error_when_there_is_nothing_to_compare(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "schema-diff", "--table", "bureau")  # committed schemas only have v1
    assert exc.value.code == 2
    assert "no version before" in capsys.readouterr().err
