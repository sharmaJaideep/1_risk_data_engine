from __future__ import annotations

from pathlib import Path

import duckdb


class DuckDBEngine:
    """Simple DuckDB connector wrapper for local analytics workloads."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path or "data/duckdb/risk_engine.duckdb")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database_path))

    def execute(self, query: str) -> duckdb.DuckDBPyConnection:
        return self.connection.execute(query)

    def close(self) -> None:
        self.connection.close()
