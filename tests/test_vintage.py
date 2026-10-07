import polars as pl

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
    pl.DataFrame(
        rows, schema=["SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "NAME_CONTRACT_STATUS"], orient="row"
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


def test_dpd_vintage_bad_means_over_60_days(tmp_path):
    # acct 1 hits 61 dpd at MOB 1; acct 2 peaks at exactly 60 dpd and is never bad
    path = _write_dpd(
        tmp_path,
        "pos",
        [(1, -2, 0, "Active"), (1, -1, 61, "Active"), (2, -2, 0, "Active"), (2, -1, 60, "Active")],
    )
    v = compute_vintage(path, source="pos_cash")
    rates = dict(zip(v["mob"], v["cumulative_bad_rate"]))
    assert rates[0] == 0.0 and rates[1] == 0.5
