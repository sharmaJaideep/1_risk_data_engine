import polars as pl

from src.db.duckdb_engine import DuckDBEngine


def test_in_memory_engine_queries_parquet_into_polars(tmp_path):
    path = tmp_path / "t.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)
    with DuckDBEngine(":memory:") as engine:
        result = engine.query(f"SELECT SUM(a) AS total FROM read_parquet('{path}')")
    assert result["total"][0] == 6
    assert engine.database_path is None


def test_file_engine_creates_parent_dir(tmp_path):
    db = tmp_path / "nested" / "x.duckdb"
    with DuckDBEngine(db) as engine:
        engine.execute("CREATE TABLE t AS SELECT 1 AS v")
    assert db.exists()
