"""Versioned schema storage.

Layout: config/schemas/<table>/v<N>.yaml, each carrying a `version: N` field.
Versions are immutable once written; a schema change is a new version, and the
highest N is the current contract. Use `diff_versions` in schema_drift.py to
see what changed between two versions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SCHEMA_DIR = Path("config/schemas")
_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")


def tables(schema_dir: str | Path = SCHEMA_DIR) -> list[str]:
    """Names of all tables that have at least one schema version."""
    return sorted(p.name for p in Path(schema_dir).iterdir() if p.is_dir() and versions(p.name, schema_dir))


def versions(table: str, schema_dir: str | Path = SCHEMA_DIR) -> list[int]:
    table_dir = Path(schema_dir) / table
    if not table_dir.is_dir():
        return []
    return sorted(int(m.group(1)) for f in table_dir.iterdir() if (m := _VERSION_FILE.match(f.name)))


def latest_version(table: str, schema_dir: str | Path = SCHEMA_DIR) -> int:
    found = versions(table, schema_dir)
    if not found:
        raise FileNotFoundError(f"No schema versions for table '{table}' in {schema_dir}")
    return found[-1]


def schema_path(table: str, version: int | None = None, schema_dir: str | Path = SCHEMA_DIR) -> Path:
    """Path to a specific version of a table's schema (latest when version is None)."""
    version = latest_version(table, schema_dir) if version is None else version
    path = Path(schema_dir) / table / f"v{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Schema '{table}' has no version {version} (available: {versions(table, schema_dir)})")
    return path


def write_version(table: str, schema: dict[str, Any], schema_dir: str | Path = SCHEMA_DIR, bump: bool = False) -> Path:
    """Write `schema` as the next version of `table`.

    The first version is always allowed; later ones require `bump=True` so existing
    history is never overwritten by accident.
    """
    existing = versions(table, schema_dir)
    if existing and not bump:
        raise FileExistsError(f"'{table}' already has versions {existing}; pass bump=True to add v{existing[-1] + 1}")
    version = existing[-1] + 1 if existing else 1
    body = {k: v for k, v in schema.items() if k not in ("table_name", "version")}
    path = Path(schema_dir) / table / f"v{version}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"table_name": table, "version": version, **body}, sort_keys=False))
    return path
