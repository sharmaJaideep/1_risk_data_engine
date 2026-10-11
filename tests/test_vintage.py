import polars as pl
import pytest

from src.metrics.vintage import compute_roll_rates, compute_vintage


def _write(tmp_path, rows):
    path = tmp_path / "bb.parquet"
    pl.DataFrame(rows, schema=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], orient="row").write_parquet(path)
    return path


def test_roll_rates_sum_to_one_and_skip_unknown(tmp_path):
    # acct 1: 0 -> 1 -> 3 ; acct 2: 0 -> X -> 0 (X breaks the chain, no transitions)
    path = _write(tmp_path, [(1, -2, "0"), (1, -1, "1"), (1, 0, "3"), (2, -2, "0"), (2, -1, "X"), (2, 0, "0")])
    rr = compute_roll_rates(path)
    assert set(rr["from_bucket"]) == {"CURRENT", "DPD_1_30"}
    assert rr["accounts"].sum() == 2
    assert all(abs(g["roll_rate"].sum() - 1.0) < 1e-9 for _, g in rr.group_by("from_bucket"))


def test_vintage_cumulative_bad_rate(tmp_path):
    # both accounts open at -3; acct 1 goes bad at MOB 2, acct 2 never does
    path = _write(
        tmp_path,
        [(1, -3, "0"), (1, -2, "0"), (1, -1, "3"), (1, 0, "3"), (2, -3, "0"), (2, -2, "0"), (2, -1, "0"), (2, 0, "0")],
    )
    v = compute_vintage(path)
    assert v["cohort_start_month"].unique().to_list() == [-3]
    rates = dict(zip(v["mob"], v["cumulative_bad_rate"]))
    assert rates[1] == 0.0 and rates[2] == 0.5 and rates[3] == 0.5


def _write_dpd(tmp_path, name, rows):
    path = tmp_path / f"{name}.parquet"
    rows = [r if len(r) == 5 else (*r, 0.0) for r in rows]  # rows are (id, month, dpd, status[, balance])
    pl.DataFrame(
        rows, schema=["SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "NAME_CONTRACT_STATUS", "AMT_BALANCE"], orient="row"
    ).write_parquet(path)
    return path


def test_dpd_sources_map_days_to_buckets_and_completed_to_closed(tmp_path):
    # acct 1: 0 dpd -> 45 dpd -> 100 dpd -> 400 dpd ; acct 2: 0 dpd -> Completed
    path = _write_dpd(
        tmp_path,
        "cc",
        [
            (1, -4, 0, "Active"), (1, -3, 45, "Active"), (1, -2, 100, "Active"), (1, -1, 400, "Active"),
            (2, -2, 0, "Active"), (2, -1, 0, "Completed"),
        ],
    )
    for source in ("credit_card", "pos_cash"):
        rr = compute_roll_rates(path, source)
        assert set(rr["source"]) == {source}
        pairs = set(zip(rr["from_bucket"], rr["to_bucket"]))
        assert pairs == {
            ("CURRENT", "DPD_31_60"),
            ("DPD_31_60", "DPD_91_120"),
            ("DPD_91_120", "DPD_120_PLUS"),
            ("CURRENT", "CLOSED"),
        }


@pytest.mark.parametrize("source", ["credit_card", "pos_cash"])
def test_dpd_vintage_bad_means_over_60_days(tmp_path, source):
    # acct 1 hits 61 dpd at MOB 1; acct 2 peaks at exactly 60 dpd and is never bad
    path = _write_dpd(
        tmp_path,
        "pos",
        [(1, -2, 0, "Active"), (1, -1, 61, "Active"), (2, -2, 0, "Active"), (2, -1, 60, "Active")],
    )
    v = compute_vintage(path, source=source)
    rates = dict(zip(v["mob"], v["cumulative_bad_rate"]))
    assert rates[0] == 0.0 and rates[1] == 0.5


# ---- edge cases and the report builder --------------------------------------------------

import pytest

from src.metrics.vintage import VintageMetrics, build_vintage_reports


def _bureau(tmp_path, rows, name="bb"):
    path = tmp_path / f"{name}.parquet"
    pl.DataFrame(rows, schema=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], orient="row").write_parquet(path)
    return path


def test_missing_months_produce_no_transition(tmp_path):
    # months -3 and -1 are not consecutive, so there is no observed month-over-month move
    rr = compute_roll_rates(_bureau(tmp_path, [(1, -3, "0"), (1, -1, "1")]))
    assert rr.height == 0


def test_unknown_status_rows_are_ignored(tmp_path):
    rr = compute_roll_rates(_bureau(tmp_path, [(1, -2, "0"), (1, -1, "X"), (1, 0, "0")]))
    assert rr.height == 0  # the X month breaks the chain


def test_vintage_denominator_only_counts_accounts_observed_at_that_age(tmp_path):
    # acct 1 is observed for 6 months and goes bad at MOB 4; acct 2 is only observed for 3 months
    rows = [(1, m, "3" if m >= -1 else "0") for m in range(-5, 1)] + [(2, m, "0") for m in range(-5, -2)]
    v = compute_vintage(_bureau(tmp_path, rows))
    by_mob = {r["mob"]: r for r in v.iter_rows(named=True)}
    assert by_mob[2]["accounts_observed"] == 2 and by_mob[2]["cumulative_bad_rate"] == 0.0
    assert by_mob[4]["accounts_observed"] == 1 and by_mob[4]["cumulative_bad_rate"] == 1.0  # acct 2 no longer counted


def test_cohort_width_groups_opening_months(tmp_path):
    rows = [(1, -7, "0"), (1, -6, "0"), (2, -6, "0"), (2, -5, "0"), (3, -5, "0"), (3, -4, "0")]
    v = compute_vintage(_bureau(tmp_path, rows), months_per_cohort=3)
    cohorts = {r["cohort_start_month"]: r["accounts_observed"] for r in v.filter(pl.col("mob") == 0).iter_rows(named=True)}
    assert cohorts == {-9: 1, -6: 2}  # opening months -7 | -6 and -5 share a cohort


def test_missing_snapshot_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot parquet not found"):
        compute_roll_rates(tmp_path / "nope.parquet")


def test_build_reports_skips_missing_sources_and_tags_the_rest(tmp_path):
    _bureau(tmp_path, [(1, -2, "0"), (1, -1, "1"), (1, 0, "3")], name="bureau_balance")
    reports = build_vintage_reports(tmp_path)
    assert set(reports["roll_rate_matrix"]["source"]) == {"bureau"} and set(reports["vintage_curves"]["source"]) == {"bureau"}
    assert (tmp_path / "roll_rate_matrix.csv").exists() and (tmp_path / "vintage_curves.parquet").exists()


def test_build_reports_stacks_all_three_sources(tmp_path):
    _bureau(tmp_path, [(1, -1, "0"), (1, 0, "0")], name="bureau_balance")
    for name in ("credit_card_balance", "POS_CASH_balance"):
        _write_dpd(tmp_path, name, [(1, -2, 0, "Active"), (1, -1, 0, "Active")])
    reports = build_vintage_reports(tmp_path)
    assert set(reports["roll_rate_matrix"]["source"]) == {"bureau", "credit_card", "pos_cash"}


def test_build_reports_with_no_inputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No snapshot parquet files"):
        build_vintage_reports(tmp_path)


def test_vintage_metrics_wrappers_delegate(tmp_path):
    path = _bureau(tmp_path, [(1, -1, "0"), (1, 0, "1")])
    assert VintageMetrics.roll_rate(path).equals(compute_roll_rates(path))
    assert VintageMetrics.vintage_analysis(path).equals(compute_vintage(path))


# ---- balance-weighted roll rates ----------------------------------------------------------

def _transition(matrix, frm, to):
    return matrix.filter((pl.col("from_bucket") == frm) & (pl.col("to_bucket") == to)).row(0, named=True)


def test_balance_weighted_roll_rate_uses_opening_balance(tmp_path):
    # acct 1 (balance 100) rolls CURRENT -> 1-30 DPD; acct 2 (balance 300) stays CURRENT.
    # Next-month balances (999 / 0) must be ignored: exposure is what the account carried when it rolled.
    path = _write_dpd(
        tmp_path, "cc",
        [(1, -2, 0, "Active", 100.0), (1, -1, 10, "Active", 999.0), (2, -2, 0, "Active", 300.0), (2, -1, 0, "Active", 0.0)],
    )
    rr = compute_roll_rates(path, "credit_card")
    to_dpd, stay = _transition(rr, "CURRENT", "DPD_1_30"), _transition(rr, "CURRENT", "CURRENT")
    assert to_dpd["roll_rate"] == 0.5 and stay["roll_rate"] == 0.5  # by account count
    assert (to_dpd["balance"], stay["balance"]) == (100.0, 300.0)
    assert to_dpd["balance_roll_rate"] == pytest.approx(0.25) and stay["balance_roll_rate"] == pytest.approx(0.75)


def test_negative_and_null_balances_count_as_zero_exposure(tmp_path):
    rows = [
        (1, -2, 0, "Active", 100.0), (1, -1, 0, "Active", 0.0),     # stays current
        (2, -2, 0, "Active", -50.0), (2, -1, 10, "Active", 0.0),    # credit balance, rolls: no exposure
        (3, -2, 0, "Active", None), (3, -1, 10, "Active", 0.0),     # unknown balance, rolls: no exposure
    ]
    rr = compute_roll_rates(_write_dpd(tmp_path, "cc", rows), "credit_card")
    assert _transition(rr, "CURRENT", "DPD_1_30")["balance_roll_rate"] == 0.0
    assert _transition(rr, "CURRENT", "CURRENT")["balance_roll_rate"] == 1.0
    assert _transition(rr, "CURRENT", "DPD_1_30")["roll_rate"] == pytest.approx(2 / 3)  # counts unaffected


def test_bucket_with_no_balance_has_null_balance_rate_but_a_count_rate(tmp_path):
    rows = [(1, -2, 10, "Active", 0.0), (1, -1, 0, "Active", 0.0)]
    rr = compute_roll_rates(_write_dpd(tmp_path, "cc", rows), "credit_card")
    row = _transition(rr, "DPD_1_30", "CURRENT")
    assert row["roll_rate"] == 1.0 and row["balance_roll_rate"] is None  # no division by zero


def test_balance_rates_sum_to_one_within_each_from_bucket(tmp_path):
    rows = [(i, m, d, "Active", 10.0 * (i + 1)) for i, dpds in enumerate([[0, 0, 40, 0], [0, 10, 10, 70], [0, 0, 0, 0]]) for m, d in zip(range(-3, 1), dpds)]
    rr = compute_roll_rates(_write_dpd(tmp_path, "cc", rows), "credit_card")
    for _, group in rr.group_by("from_bucket"):
        assert group["balance_roll_rate"].sum() == pytest.approx(1.0)


def test_sources_without_balances_have_null_balance_columns(tmp_path):
    # pos_cash files may carry an AMT_BALANCE-like column; the source still defines no exposure
    pos = compute_roll_rates(_write_dpd(tmp_path, "pos", [(1, -2, 0, "Active", 500.0), (1, -1, 10, "Active", 500.0)]), "pos_cash")
    bureau = compute_roll_rates(_bureau(tmp_path, [(1, -2, "0"), (1, -1, "1")]))
    for rr in (pos, bureau):
        assert rr.height > 0 and rr["balance"].null_count() == rr.height and rr["balance_roll_rate"].null_count() == rr.height


def test_stacked_report_keeps_balance_columns_for_every_source(tmp_path):
    _bureau(tmp_path, [(1, -1, "0"), (1, 0, "1")], name="bureau_balance")
    for name in ("credit_card_balance", "POS_CASH_balance"):
        _write_dpd(tmp_path, name, [(1, -2, 0, "Active", 100.0), (1, -1, 10, "Active", 100.0)])
    rr = build_vintage_reports(tmp_path)["roll_rate_matrix"]
    by_source = dict(rr.group_by("source").agg(pl.col("balance_roll_rate").drop_nulls().len().alias("n")).iter_rows())
    assert by_source == {"bureau": 0, "pos_cash": 0, "credit_card": 1}  # one CURRENT->1-30 DPD transition
