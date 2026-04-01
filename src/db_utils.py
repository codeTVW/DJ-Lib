from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from database_path import resolve_database_path
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_path import resolve_database_path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(resolve_database_path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None
