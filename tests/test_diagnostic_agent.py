import polars as pl

from src.agents.diagnostic_agent import DiagnosticAgent


def _write(tmp_path, rows):
    path = tmp_path / "report.parquet"
    pl.DataFrame(rows, schema=["feature_name", "psi_score", "drift_status"], orient="row").write_parquet(path)
    return path


def test_flags_significant_drift(tmp_path):
    path = _write(tmp_path, [("A", 0.4, "SIGNIFICANT_DRIFT"), ("B", 0.01, "STABLE")])
    report = DiagnosticAgent(report_path=path, use_llm=False).run()
    assert report.status == "critical"
    assert [b["feature_name"] for b in report.breaches] == ["A"]


def test_stable_portfolio_is_ok(tmp_path):
    path = _write(tmp_path, [("A", 0.01, "STABLE")])
    assert DiagnosticAgent(report_path=path, use_llm=False).run().status == "ok"


def test_missing_report_returns_error(tmp_path):
    assert DiagnosticAgent(report_path=tmp_path / "nope.parquet", use_llm=False).run().status == "error"


def _roll_rate(tmp_path, roll_in):
    rows = [
        ("CURRENT", "CURRENT", 1.0 - roll_in),
        ("CURRENT", "DPD_1_30", roll_in),
        ("DPD_1_30", "CURRENT", 0.7),
        ("DPD_1_30", "DPD_31_60", 0.3),
    ]
    path = tmp_path / "rr.parquet"
    pl.DataFrame(rows, schema=["from_bucket", "to_bucket", "roll_rate"], orient="row").write_parquet(path)
    return path


def test_high_roll_in_escalates_to_warning(tmp_path):
    stability = _write(tmp_path, [("A", 0.01, "STABLE")])
    report = DiagnosticAgent(
        report_path=stability, roll_rate_path=_roll_rate(tmp_path, 0.10), vintage_path=tmp_path / "none", use_llm=False
    ).run()
    assert report.status == "warning"
    assert any("roll-in" in f for f in report.findings)


def test_worsening_vintage_flagged(tmp_path):
    stability = _write(tmp_path, [("A", 0.01, "STABLE")])
    rates = [0.01, 0.01, 0.01, 0.01, 0.01, 0.03]
    rows = [(-96 + 3 * i, 12, 5000, r) for i, r in enumerate([0.05] + rates)]
    vint = tmp_path / "v.parquet"
    pl.DataFrame(rows, schema=["cohort_start_month", "mob", "accounts_observed", "cumulative_bad_rate"], orient="row").write_parquet(vint)
    report = DiagnosticAgent(
        report_path=stability, roll_rate_path=tmp_path / "none", vintage_path=vint, use_llm=False
    ).run()
    assert report.status == "warning"
    assert any("worsening" in f for f in report.findings)
