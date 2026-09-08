from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class SchemaValidator:
    """Validate input CSV columns against a YAML schema definition."""

    def __init__(self, schema_path: str | Path | None = None) -> None:
        self.schema_path = Path(schema_path or "config/schema_config.yaml")
        self.schema = self._load_schema()

    def _load_schema(self) -> dict[str, Any]:
        with self.schema_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def validate_columns(self, columns: list[str]) -> bool:
        strict_schema = self.schema["strict_schema"]
        expected = set(strict_schema["columns"].keys())
        provided = set(columns)
        required_columns = set(strict_schema.get("required_columns", []))

        if not required_columns.issubset(provided):
            return False

        return provided.issubset(expected)
