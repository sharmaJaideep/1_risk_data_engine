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


def _vintage(tmp_path, latest_bad, source=None):
    rates = [0.01, 0.01, 0.01, 0.01, 0.01, 0.03]
    rows = [(-96 + 3 * i, 12, 5000, latest_bad if i == 6 else 50, r) for i, r in enumerate([0.05] + rates)]
    cols = ["cohort_start_month", "mob", "accounts_observed", "accounts_bad", "cumulative_bad_rate"]
    if source:
        rows = [(source, *r) for r in rows]
        cols = ["source", *cols]
    path = tmp_path / "v.parquet"
    pl.DataFrame(rows, schema=cols, orient="row").write_parquet(path)
    return path


def _agent(tmp_path, vint):
    stability = _write(tmp_path, [("A", 0.01, "STABLE")])
    return DiagnosticAgent(report_path=stability, roll_rate_path=tmp_path / "none", vintage_path=vint, use_llm=False)


def test_worsening_vintage_flagged(tmp_path):
    report = _agent(tmp_path, _vintage(tmp_path, latest_bad=150)).run()
    assert report.status == "warning"
    assert any("worsening" in f for f in report.findings)


def test_worsening_vintage_with_too_few_bad_accounts_not_flagged(tmp_path):
    report = _agent(tmp_path, _vintage(tmp_path, latest_bad=10)).run()
    assert report.status == "ok"
    assert any("not flagged" in f for f in report.findings)


def test_checks_run_per_source(tmp_path):
    cols = ["source", "cohort_start_month", "mob", "accounts_observed", "accounts_bad", "cumulative_bad_rate"]
    rows = []
    for source, last_rate in (("credit_card", 0.03), ("pos_cash", 0.01)):  # only credit_card worsens
        rates = [0.05, 0.01, 0.01, 0.01, 0.01, 0.01, last_rate]
        rows += [(source, -96 + 3 * i, 12, 5000, 150, r) for i, r in enumerate(rates)]
    path = tmp_path / "v.parquet"
    pl.DataFrame(rows, schema=cols, orient="row").write_parquet(path)
    report = _agent(tmp_path, path).run()
    assert any(f.startswith("[credit_card]") and "worsening" in f for f in report.findings)
    assert any(f.startswith("[pos_cash]") and "worsening" not in f for f in report.findings)


def _write_thresholds(tmp_path, text):
    path = tmp_path / "thresholds.yaml"
    path.write_text(text)
    return path


def _roll_rate_for_sources(tmp_path, roll_in):
    rows = []
    for source in ("bureau", "credit_card"):
        rows += [
            (source, "CURRENT", "CURRENT", 1.0 - roll_in),
            (source, "CURRENT", "DPD_1_30", roll_in),
            (source, "DPD_1_30", "CURRENT", 0.9),
            (source, "DPD_1_30", "DPD_31_60", 0.1),
        ]
    path = tmp_path / "rr.parquet"
    pl.DataFrame(rows, schema=["source", "from_bucket", "to_bucket", "roll_rate"], orient="row").write_parquet(path)
    return path


def test_per_source_roll_threshold_overrides_default(tmp_path):
    thresholds = _write_thresholds(
        tmp_path,
        "roll_rates:\n  default: {roll_in_max: 0.05, forward_max: 0.15}\n  sources:\n    credit_card: {roll_in_max: 0.03}\n",
    )
    report = DiagnosticAgent(
        report_path=_write(tmp_path, [("A", 0.01, "STABLE")]),
        roll_rate_path=_roll_rate_for_sources(tmp_path, 0.04),  # 4%: under the 5% default, over credit_card's 3%
        vintage_path=tmp_path / "none",
        thresholds_path=thresholds,
        use_llm=False,
    ).run()
    flagged = [f for f in report.findings if "exceeds" in f]
    assert len(flagged) == 1 and flagged[0].startswith("[credit_card]") and "3.0%" in flagged[0]


def test_thresholds_fall_back_to_builtins_and_merge_partial_overrides(tmp_path):
    missing = DiagnosticAgent(report_path=tmp_path / "r.parquet", thresholds_path=tmp_path / "nope.yaml", use_llm=False)
    assert missing.roll_thresholds("pos_cash") == {"roll_in_max": 0.05, "forward_max": 0.15}

    partial = DiagnosticAgent(
        report_path=tmp_path / "r.parquet",
        thresholds_path=_write_thresholds(tmp_path, "roll_rates:\n  sources:\n    pos_cash: {forward_max: 0.08}\n"),
        use_llm=False,
    )
    assert partial.roll_thresholds("pos_cash") == {"roll_in_max": 0.05, "forward_max": 0.08}
    assert partial.roll_thresholds("bureau") == {"roll_in_max": 0.05, "forward_max": 0.15}


def test_committed_thresholds_file_is_well_formed():
    agent = DiagnosticAgent(use_llm=False)
    for source in ("bureau", "credit_card", "pos_cash"):
        limits = agent.roll_thresholds(source)
        assert set(limits) == {"roll_in_max", "forward_max"} and all(0 < v < 1 for v in limits.values())
