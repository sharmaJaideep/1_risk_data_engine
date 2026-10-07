import polars as pl
import pytest

from src.metrics.aggregators import build_master_analytical_record

MAR = "master_analytical_record.parquet"


def _write(data_dir, name, rows, schema):
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(data_dir / f"{name}.parquet")


def _write_inputs(d, skip=()):
    """Three applicants: 1 has everything, 2 has a single bureau account only, 3 has previous apps only."""
    tables = {
        "application_train": (
            [(1, 0), (2, 1), (3, 0)],
            {"SK_ID_CURR": pl.Int64, "TARGET": pl.Int64},
        ),
        "bureau": (
            [
                (1, 10, "Active", 100.0, 1000.0, 5.0),
                (1, 11, "Closed", None, 500.0, 0.0),
                (2, 12, "Active", 50.0, 200.0, None),
            ],
            {
                "SK_ID_CURR": pl.Int64,
                "SK_ID_BUREAU": pl.Int64,
                "CREDIT_ACTIVE": pl.String,
                "AMT_CREDIT_SUM_DEBT": pl.Float64,
                "AMT_CREDIT_SUM": pl.Float64,
                "AMT_CREDIT_SUM_OVERDUE": pl.Float64,
            },
        ),
        # 'C' and 'X' are not numeric and must count as non-delinquent
        "bureau_balance": (
            [(10, 0, "0"), (10, -1, "1"), (10, -2, "C"), (11, 0, "X"), (11, -1, "2")],
            {"SK_ID_BUREAU": pl.Int64, "MONTHS_BALANCE": pl.Int64, "STATUS": pl.String},
        ),
        "installments_payments": (
            [(1, -10.0, -12.0, 100.0, 100.0), (1, -5.0, 0.0, 100.0, 60.0)],
            {
                "SK_ID_CURR": pl.Int64,
                "DAYS_INSTALMENT": pl.Float64,
                "DAYS_ENTRY_PAYMENT": pl.Float64,
                "AMT_INSTALMENT": pl.Float64,
                "AMT_PAYMENT": pl.Float64,
            },
        ),
        "previous_application": (
            [
                (1, "Approved", 100.0),
                (1, "Refused", 200.0),
                (1, "Approved", 300.0),
                (3, "Canceled", None),
            ],
            {"SK_ID_CURR": pl.Int64, "NAME_CONTRACT_STATUS": pl.String, "AMT_APPLICATION": pl.Float64},
        ),
    }
    for name, (rows, schema) in tables.items():
        if name not in skip:
            _write(d, name, rows, schema)


@pytest.fixture
def mar(tmp_path):
    _write_inputs(tmp_path)
    build_master_analytical_record(str(tmp_path))
    return pl.read_parquet(tmp_path / MAR).sort("SK_ID_CURR")


def _row(mar, sk_id):
    return mar.filter(pl.col("SK_ID_CURR") == sk_id).row(0, named=True)


def test_one_row_per_applicant_and_base_columns_kept(mar):
    # joining children must not fan out the application grain
    assert mar["SK_ID_CURR"].to_list() == [1, 2, 3]
    assert mar["TARGET"].to_list() == [0, 1, 0]


def test_bureau_account_counts_and_amounts(mar):
    r = _row(mar, 1)
    assert (r["bureau_total_accounts"], r["bureau_active_accounts"], r["bureau_closed_accounts"]) == (2, 1, 1)
    assert r["bureau_sum_credit_sum_debt"] == 100.0  # NULL debt treated as 0
    assert r["bureau_sum_credit_sum"] == 1500.0
    assert r["bureau_total_overdue_balance"] == 5.0
    assert _row(mar, 2)["bureau_total_overdue_balance"] == 0.0  # NULL overdue treated as 0


def test_bureau_balance_delinquent_months_roll_up_without_fan_out(mar):
    # bureau 10: statuses 0,1,C -> 1 delinquent month; bureau 11: X,2 -> 1; X/C count as 0
    assert _row(mar, 1)["bureau_total_delinquent_months"] == 2
    # bureau 10 has three balance rows but must still be counted as one account
    assert _row(mar, 1)["bureau_total_accounts"] == 2
    assert _row(mar, 2)["bureau_total_delinquent_months"] == 0


def test_installment_delay_deficit_and_late_counts(mar):
    r = _row(mar, 1)
    assert r["ip_max_payment_delay_days"] == 5.0
    assert r["ip_total_payment_deficit"] == 40.0
    assert (r["ip_total_payments"], r["ip_late_payments"]) == (2, 1)


def test_previous_application_counts_and_mean(mar):
    r = _row(mar, 1)
    assert (r["prev_total_applications"], r["prev_approved_applications"], r["prev_refused_applications"]) == (3, 2, 1)
    assert r["prev_mean_amt_application"] == 200.0
    assert _row(mar, 3)["prev_mean_amt_application"] == 0.0  # all-NULL amounts -> 0


def test_applicants_without_children_get_zeros_not_nulls(mar):
    aggregate_cols = [c for c in mar.columns if c.startswith(("bureau_", "ip_", "prev_"))]
    assert len(aggregate_cols) == 15
    assert mar.select(aggregate_cols).null_count().row(0) == (0,) * len(aggregate_cols)
    r = _row(mar, 3)  # no bureau, no installments
    assert r["bureau_total_accounts"] == 0 and r["ip_total_payments"] == 0


def test_missing_application_train_writes_nothing(tmp_path):
    _write_inputs(tmp_path, skip=("application_train",))
    build_master_analytical_record(str(tmp_path))
    assert not (tmp_path / MAR).exists()


@pytest.mark.parametrize("missing", ["bureau", "bureau_balance", "installments_payments", "previous_application"])
def test_missing_optional_input_still_builds_mar(tmp_path, missing):
    _write_inputs(tmp_path, skip=(missing,))
    build_master_analytical_record(str(tmp_path))
    assert (tmp_path / MAR).exists()
    assert pl.read_parquet(tmp_path / MAR).height == 3
