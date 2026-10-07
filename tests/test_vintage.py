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
