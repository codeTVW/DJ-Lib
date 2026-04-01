from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from database_init import CREATE_FEEDBACK_TABLE
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import CREATE_FEEDBACK_TABLE

try:
    from similarity_engine import classify_mix_type
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.similarity_engine import classify_mix_type


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dj_library.sqlite3"
_DB_PATH = DEFAULT_DB_PATH


def set_database_path(db_path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path).expanduser()


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_feedback_table(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_FEEDBACK_TABLE)
    connection.commit()


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_required_tables(connection: sqlite3.Connection) -> None:
    required_tables = ("audio_features", "similarities")
    missing = [name for name in required_tables if not table_exists(connection, name)]
    if missing:
        raise RuntimeError(f"Ontbrekende tabellen: {', '.join(missing)}")


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


def record_feedback(file_path_a: str, file_path_b: str, rating: int) -> None:
    if rating not in {1, -1}:
        raise ValueError("Rating moet 1 of -1 zijn.")

    with get_connection() as connection:
        ensure_feedback_table(connection)
        connection.execute(
            """
            INSERT INTO feedback (file_path_a, file_path_b, rating, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_path_a, file_path_b, rating, current_timestamp()),
        )
        connection.commit()


def _pair_row(connection: sqlite3.Connection, file_path_a: str, file_path_b: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT
            s.similarity,
            af_a.bpm AS bpm_a,
            af_b.bpm AS bpm_b,
            af_a.key AS key_a,
            af_a.mode AS mode_a,
            af_b.key AS key_b,
            af_b.mode AS mode_b
        FROM similarities AS s
        JOIN audio_features AS af_a
            ON af_a.file_path = s.file_path_a
        JOIN audio_features AS af_b
            ON af_b.file_path = s.file_path_b
        WHERE s.file_path_a = ?
          AND s.file_path_b = ?
        """,
        (file_path_a, file_path_b),
    ).fetchone()


def apply_feedback() -> int:
    with get_connection() as connection:
        ensure_feedback_table(connection)
        ensure_required_tables(connection)

        feedback_rows = connection.execute(
            """
            SELECT file_path_a, file_path_b, rating
            FROM feedback
            ORDER BY created_at, rowid
            """
        ).fetchall()

        updated_rows = 0
        for row in feedback_rows:
            pair_row = _pair_row(connection, str(row["file_path_a"]), str(row["file_path_b"]))
            if pair_row is None:
                continue

            current_similarity = float(pair_row["similarity"])
            delta = 0.05 if int(row["rating"]) == 1 else -0.1
            new_similarity = max(0.0, min(1.0, current_similarity + delta))
            new_mix_type = classify_mix_type(
                similarity=new_similarity,
                bpm_a=float(pair_row["bpm_a"]),
                bpm_b=float(pair_row["bpm_b"]),
                key_a=int(pair_row["key_a"]),
                mode_a=int(pair_row["mode_a"]),
                key_b=int(pair_row["key_b"]),
                mode_b=int(pair_row["mode_b"]),
            )
            connection.execute(
                """
                UPDATE similarities
                SET similarity = ?, mix_type = ?
                WHERE file_path_a = ? AND file_path_b = ?
                """,
                (
                    new_similarity,
                    new_mix_type,
                    str(row["file_path_a"]),
                    str(row["file_path_b"]),
                ),
            )
            updated_rows += 1

        connection.commit()

    print(f"Feedback toegepast: {updated_rows} rijen bijgewerkt.")
    return updated_rows


def get_feedback_stats() -> dict[str, Any]:
    with get_connection() as connection:
        ensure_feedback_table(connection)

        summary_row = connection.execute(
            """
            SELECT COUNT(*) AS total_records, AVG(rating) AS avg_rating
            FROM feedback
            """
        ).fetchone()

        top_rows = connection.execute(
            """
            SELECT
                f.file_path_a,
                f.file_path_b,
                COALESCE(s.similarity, 0.0) AS similarity
            FROM (
                SELECT DISTINCT file_path_a, file_path_b
                FROM feedback
            ) AS f
            LEFT JOIN similarities AS s
                ON s.file_path_a = f.file_path_a
               AND s.file_path_b = f.file_path_b
            ORDER BY similarity DESC, f.file_path_a, f.file_path_b
            LIMIT 10
            """
        ).fetchall()

        return {
            "total_records": int(summary_row["total_records"]) if summary_row else 0,
            "avg_rating": round(
                float(summary_row["avg_rating"]) if summary_row and summary_row["avg_rating"] is not None else 0.0,
                2,
            ),
            "top_pairs": [
                {
                    "file_path_a": str(row["file_path_a"]),
                    "file_path_b": str(row["file_path_b"]),
                    "similarity": float(row["similarity"]),
                }
                for row in top_rows
            ],
        }
