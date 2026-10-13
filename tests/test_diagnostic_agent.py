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
    assert missing.roll_thresholds("pos_cash") == {
        "roll_in_max": 0.05, "forward_max": 0.15, "balance_roll_in_max": 0.05, "balance_forward_max": 0.15,
    }

    partial = DiagnosticAgent(
        report_path=tmp_path / "r.parquet",
        thresholds_path=_write_thresholds(tmp_path, "roll_rates:\n  sources:\n    pos_cash: {forward_max: 0.08}\n"),
        use_llm=False,
    )
    # balance limits follow the count limits unless set explicitly
    assert partial.roll_thresholds("pos_cash") == {
        "roll_in_max": 0.05, "forward_max": 0.08, "balance_roll_in_max": 0.05, "balance_forward_max": 0.08,
    }
    assert partial.roll_thresholds("bureau")["balance_forward_max"] == 0.15


def test_committed_thresholds_file_is_well_formed():
    agent = DiagnosticAgent(use_llm=False)
    for source in ("bureau", "credit_card", "pos_cash"):
        limits = agent.roll_thresholds(source)
        assert set(limits) == {"roll_in_max", "forward_max", "balance_roll_in_max", "balance_forward_max"}
        assert all(0 < v < 1 for v in limits.values())


def _drifting_report(tmp_path):
    return _write(tmp_path, [("AMT_CREDIT", 0.41, "SIGNIFICANT_DRIFT")])


def test_llm_narrative_used_when_key_present(tmp_path, monkeypatch):
    from pydantic_ai.models.test import TestModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    report = DiagnosticAgent(
        report_path=_drifting_report(tmp_path), model=TestModel(custom_output_text="Loan sizes shifted."), use_llm=True
    ).run()
    assert report.narrative == "Loan sizes shifted."
    assert "## Summary" in report.to_markdown()


def test_llm_skipped_without_key_and_failures_fall_back(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = DiagnosticAgent(report_path=_drifting_report(tmp_path), use_llm=True).run()
    assert report.status == "critical" and report.narrative is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-key")  # reaches the API and fails auth, or fails offline
    report = DiagnosticAgent(report_path=_drifting_report(tmp_path), use_llm=True).run()
    assert report.status == "critical" and report.narrative is None


def _stability_report(tmp_path, rows):
    path = tmp_path / "portfolio_stability_report.parquet"
    pl.DataFrame(
        rows,
        schema=["analysis", "feature_name", "psi_score", "drift_status", "top_driver", "alerting"],
        orient="row",
    ).write_parquet(path)
    return path


def test_informational_analyses_do_not_raise_alerts(tmp_path):
    path = _stability_report(
        tmp_path,
        [("train_vs_test", "A", 0.01, "STABLE", "(0, 1]", True), ("by_recency", "B", 3.0, "SIGNIFICANT_DRIFT", "__NULL__", False)],
    )
    report = DiagnosticAgent(report_path=path, use_llm=False, roll_rate_path=tmp_path / "x", vintage_path=tmp_path / "y").run()
    assert report.status == "ok"
    assert any("Informational" in f and "1 with PSI" in f for f in report.findings)


def test_alerting_analysis_escalates_and_names_analysis_and_driver(tmp_path):
    path = _stability_report(tmp_path, [("train_vs_test", "NAME_CONTRACT_TYPE", 0.4, "SIGNIFICANT_DRIFT", "Revolving loans", True)])
    report = DiagnosticAgent(report_path=path, use_llm=False, roll_rate_path=tmp_path / "x", vintage_path=tmp_path / "y").run()
    assert report.status == "critical"
    assert any("[train_vs_test] NAME_CONTRACT_TYPE" in f and "driven by bin Revolving loans" in f for f in report.findings)


# ---- remaining branches --------------------------------------------------------------------

def _drift_report(tmp_path, rows):
    path = tmp_path / "schema_drift_report.parquet"
    pl.DataFrame(rows, schema=["table", "kind", "column", "baseline", "observed", "severity"], orient="row").write_parquet(path)
    return path


def _agent_with(tmp_path, **paths):
    stability = _write(tmp_path, [("A", 0.01, "STABLE")])
    return DiagnosticAgent(
        report_path=stability, roll_rate_path=paths.get("roll", tmp_path / "x"), vintage_path=paths.get("vintage", tmp_path / "y"),
        schema_drift_path=paths.get("drift"), use_llm=False,
    ).run()


def test_missing_stability_report_is_an_error(tmp_path):
    report = DiagnosticAgent(report_path=tmp_path / "nope.parquet", use_llm=False).run()
    assert report.status == "error" and "python main.py run-all" in report.recommendations[0]


def test_schema_warnings_alone_give_warning_and_breaking_gives_critical(tmp_path):
    warn_only = _drift_report(tmp_path, [("bureau", "column_added", "NEW", "absent", "string", "warning")])
    assert _agent_with(tmp_path, drift=warn_only).status == "warning"
    breaking = _drift_report(tmp_path, [("bureau", "type_changed", "AMT", "float", "string", "breaking")])
    report = _agent_with(tmp_path, drift=breaking)
    assert report.status == "critical" and any("type_changed bureau.AMT" in f for f in report.findings)


def test_informational_schema_drift_does_not_escalate(tmp_path):
    info = _drift_report(tmp_path, [("bureau", "nulls_absent", "X", "nullable", "no nulls observed", "info")])
    assert _agent_with(tmp_path, drift=info).status == "ok"


def test_missing_optional_reports_are_noted_not_fatal(tmp_path):
    report = _agent_with(tmp_path, drift=tmp_path / "none")
    text = " | ".join(report.findings)
    assert report.status == "ok"
    assert "Schema drift report not found" in text and "Roll-rate matrix not found" in text and "Vintage curves not found" in text


def test_too_few_vintage_cohorts_is_reported(tmp_path):
    path = tmp_path / "v.parquet"
    rows = [(-96 + 3 * i, 12, 5000, 40, 0.01) for i in range(3)]
    pl.DataFrame(rows, schema=["cohort_start_month", "mob", "accounts_observed", "accounts_bad", "cumulative_bad_rate"], orient="row").write_parquet(path)
    report = _agent_with(tmp_path, vintage=path)
    assert report.status == "ok" and any("Too few cohorts" in f for f in report.findings)


def test_forward_roll_breach_warns_with_a_collections_recommendation(tmp_path):
    rows = [("CURRENT", "CURRENT", 0.99), ("CURRENT", "DPD_1_30", 0.01), ("DPD_1_30", "CURRENT", 0.5), ("DPD_1_30", "DPD_31_60", 0.5)]
    path = tmp_path / "rr.parquet"
    pl.DataFrame(rows, schema=["from_bucket", "to_bucket", "roll_rate"], orient="row").write_parquet(path)
    report = _agent_with(tmp_path, roll=path)
    assert report.status == "warning"
    assert any("forward roll 50.0% exceeds 15.0%" in f for f in report.findings)
    assert any("early-stage collections" in r for r in report.recommendations)


def test_markdown_brief_layout():
    from src.agents.diagnostic_agent import DiagnosticReport

    md = DiagnosticReport("warning", ["finding one"], ["do this"], narrative="Because.").to_markdown()
    assert md.index("## Summary") < md.index("## Findings") < md.index("## Recommendations")
    assert "**Status:** WARNING" in md and "- finding one" in md and "- do this" in md
    assert "## Summary" not in DiagnosticReport("ok", [], []).to_markdown()


# ---- balance-weighted roll rates -----------------------------------------------------------

def _balance_roll_rates(tmp_path, count_in, balance_in, source="credit_card", with_balance=True):
    rows = []
    for to_bucket, c, b in (("CURRENT", 1 - count_in, 1 - balance_in), ("DPD_1_30", count_in, balance_in)):
        rows.append((source, "CURRENT", to_bucket, c, b if with_balance else None))
    rows += [(source, "DPD_1_30", "CURRENT", 0.9, 0.9 if with_balance else None), (source, "DPD_1_30", "DPD_31_60", 0.1, 0.1 if with_balance else None)]
    path = tmp_path / "rr.parquet"
    pl.DataFrame(rows, schema=["source", "from_bucket", "to_bucket", "roll_rate", "balance_roll_rate"], orient="row").write_parquet(path)
    return path


def test_balance_weighted_breach_is_flagged_even_when_counts_look_fine(tmp_path):
    report = _agent_with(tmp_path, roll=_balance_roll_rates(tmp_path, count_in=0.01, balance_in=0.09))
    assert report.status == "warning"
    text = " | ".join(report.findings)
    assert "Balance-weighted: CURRENT->delinquent 9.0%" in text
    assert "Balance-weighted CURRENT->delinquent roll-in 9.0% exceeds 5.0% threshold" in text
    assert not any(f.startswith("[credit_card] CURRENT->delinquent roll-in") for f in report.findings)  # count view is fine


def test_balance_limits_can_be_set_separately(tmp_path):
    thresholds = _write_thresholds(tmp_path, "roll_rates:\n  default: {roll_in_max: 0.05, forward_max: 0.15, balance_roll_in_max: 0.10}\n")
    agent = DiagnosticAgent(
        report_path=_write(tmp_path, [("A", 0.01, "STABLE")]),
        roll_rate_path=_balance_roll_rates(tmp_path, count_in=0.01, balance_in=0.09),
        vintage_path=tmp_path / "y", thresholds_path=thresholds, use_llm=False,
    )
    assert agent.run().status == "ok"  # 9% is under the explicit 10% balance limit


def test_sources_without_balances_show_no_balance_line(tmp_path):
    report = _agent_with(tmp_path, roll=_balance_roll_rates(tmp_path, 0.01, 0.0, source="bureau", with_balance=False))
    assert report.status == "ok" and not any("Balance-weighted" in f for f in report.findings)


# ---- model health ---------------------------------------------------------------------------

def _model_metrics(tmp_path, **overrides):
    metrics = {"auc": 0.76, "gini": 0.52, "ks": 0.39, "mean_predicted": 0.0808, "base_rate": 0.0807, "n_holdout": 61503}
    path = tmp_path / "model_metrics.json"
    path.write_text(__import__("json").dumps({**metrics, **overrides}))
    return path


def _model_agent(tmp_path, metrics_path, thresholds=None):
    return DiagnosticAgent(
        report_path=_write(tmp_path, [("A", 0.01, "STABLE")]), model_metrics_path=metrics_path,
        roll_rate_path=tmp_path / "x", vintage_path=tmp_path / "y",
        thresholds_path=thresholds or tmp_path / "none.yaml", use_llm=False,
    ).run()


def test_healthy_model_is_reported_without_a_warning(tmp_path):
    report = _model_agent(tmp_path, _model_metrics(tmp_path))
    assert report.status == "ok"
    assert any("Model: AUC 0.760, Gini 0.520, KS 0.390" in f and "8.08% vs observed 8.07%" in f for f in report.findings)


def test_weak_discrimination_warns(tmp_path):
    report = _model_agent(tmp_path, _model_metrics(tmp_path, auc=0.62, ks=0.18))
    assert report.status == "warning"
    assert any("discrimination below floor" in f for f in report.findings) and any("Retrain" in r for r in report.recommendations)


def test_miscalibrated_model_warns(tmp_path):
    report = _model_agent(tmp_path, _model_metrics(tmp_path, mean_predicted=0.12))  # 49% over-prediction
    assert report.status == "warning" and any("calibration off by 49%" in f for f in report.findings)


def test_model_limits_are_configurable(tmp_path):
    thresholds = _write_thresholds(tmp_path, "model:\n  min_auc: 0.80\n")
    assert _model_agent(tmp_path, _model_metrics(tmp_path), thresholds).status == "warning"  # 0.76 < the stricter floor


def test_missing_model_metrics_are_noted_not_fatal(tmp_path):
    report = _model_agent(tmp_path, tmp_path / "nope.json")
    assert report.status == "ok" and any("Model metrics not found" in f for f in report.findings)
