from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from database_init import CREATE_CRATES_TABLE
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import CREATE_CRATES_TABLE


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dj_library.sqlite3"
_DB_PATH = DEFAULT_DB_PATH


def set_database_path(db_path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path).expanduser()


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_required_tables(connection: sqlite3.Connection) -> None:
    required_tables = (
        "tracks",
        "audio_features",
        "similarities",
        "umap_coordinates",
        "clusters",
        "cluster_metadata",
    )
    missing = [name for name in required_tables if not table_exists(connection, name)]
    if missing:
        raise RuntimeError(f"Ontbrekende tabellen: {', '.join(missing)}")


def ensure_crates_table(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_CRATES_TABLE)
    connection.commit()


def _audio_feature_crates(connection: sqlite3.Connection) -> dict[str, set[str]]:
    return {
        "Warm-up": {
            row["file_path"]
            for row in connection.execute(
                """
                SELECT file_path
                FROM audio_features
                WHERE energy < 0.4 AND bpm < 110
                """
            ).fetchall()
        },
        "Peak-time": {
            row["file_path"]
            for row in connection.execute(
                """
                SELECT file_path
                FROM audio_features
                WHERE energy > 0.75 AND bpm > 128
                """
            ).fetchall()
        },
        "Late night": {
            row["file_path"]
            for row in connection.execute(
                """
                SELECT file_path
                FROM audio_features
                WHERE energy < 0.4 AND bpm > 110
                """
            ).fetchall()
        },
        "Vocal tools": {
            row["file_path"]
            for row in connection.execute(
                """
                SELECT file_path
                FROM audio_features
                WHERE vocal_presence > 0.65
                """
            ).fetchall()
        },
        "Orphan tracks": {
            row["file_path"]
            for row in connection.execute(
                """
                SELECT c.file_path
                FROM clusters AS c
                WHERE c.cluster_id = -1
                """
            ).fetchall()
        },
    }


def _bridge_tracks(connection: sqlite3.Connection) -> set[str]:
    cluster_pairs = connection.execute(
        """
        SELECT a.cluster_id AS cluster_a_id, b.cluster_id AS cluster_b_id
        FROM cluster_metadata AS a
        CROSS JOIN cluster_metadata AS b
        WHERE a.cluster_id < b.cluster_id
          AND a.cluster_id != -1
          AND b.cluster_id != -1
        ORDER BY a.cluster_id, b.cluster_id
        """
    ).fetchall()

    bridge_paths: set[str] = set()
    for pair in cluster_pairs:
        rows = connection.execute(
            """
            SELECT sub_a.file_path_a
            FROM (
                SELECT DISTINCT s.file_path_a
                FROM similarities AS s
                JOIN clusters AS c
                    ON c.file_path = s.file_path_b
                WHERE c.cluster_id = ?
            ) AS sub_a
            INNER JOIN (
                SELECT DISTINCT s.file_path_a
                FROM similarities AS s
                JOIN clusters AS c
                    ON c.file_path = s.file_path_b
                WHERE c.cluster_id = ?
            ) AS sub_b
                ON sub_a.file_path_a = sub_b.file_path_a
            """,
            (int(pair["cluster_a_id"]), int(pair["cluster_b_id"])),
        ).fetchall()
        bridge_paths.update(str(row["file_path_a"]) for row in rows)

    return bridge_paths


def generate_crates() -> dict[str, int]:
    with get_connection() as connection:
        ensure_required_tables(connection)
        ensure_crates_table(connection)

        connection.execute("DELETE FROM crates")

        crates = _audio_feature_crates(connection)
        crates["Bridge tracks"] = _bridge_tracks(connection)

        rows_to_insert = [
            (crate_name, file_path)
            for crate_name, file_paths in crates.items()
            for file_path in sorted(file_paths)
        ]
        connection.executemany(
            """
            INSERT OR IGNORE INTO crates (crate_name, file_path)
            VALUES (?, ?)
            """,
            rows_to_insert,
        )
        connection.commit()

        return {crate_name: len(file_paths) for crate_name, file_paths in crates.items()}
