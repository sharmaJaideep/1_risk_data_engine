from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

IN_MEMORY = ":memory:"


class DuckDBEngine:
    """DuckDB connector wrapper for the engine's SQL-over-Parquet workloads.

    Pass `":memory:"` for throwaway analytics (the pipeline default); with no
    argument it opens the persistent database at data/duckdb/risk_engine.duckdb.
    Usable as a context manager.
    """

    def __init__(self, database_path: str | Path | None = None) -> None:
        if str(database_path) == IN_MEMORY:
            self.database_path = None
            self.connection = duckdb.connect(IN_MEMORY)
        else:
            self.database_path = Path(database_path or "data/duckdb/risk_engine.duckdb")
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(str(self.database_path))

    def execute(self, query: str) -> duckdb.DuckDBPyConnection:
        return self.connection.execute(query)

    def query(self, query: str) -> pl.DataFrame:
        """Run a query and return the result as a Polars DataFrame."""
        return self.connection.execute(query).pl()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DuckDBEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
