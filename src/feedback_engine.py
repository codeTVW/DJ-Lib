from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from database_init import CREATE_FEEDBACK_TABLE
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import CREATE_FEEDBACK_TABLE

try:
    from similarity_engine import classify_mix_type, cosine_similarity, normalize
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.similarity_engine import classify_mix_type, cosine_similarity, normalize


LOGGER = logging.getLogger("feedback_engine")
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def load_track_features(connection: sqlite3.Connection) -> dict[str, dict[str, float | int]]:
    rows = connection.execute(
        """
        SELECT
            file_path,
            bpm,
            key,
            mode,
            energy,
            loudness,
            spectral_centroid
        FROM audio_features
        WHERE bpm IS NOT NULL
          AND key IS NOT NULL
          AND mode IS NOT NULL
          AND energy IS NOT NULL
          AND loudness IS NOT NULL
          AND spectral_centroid IS NOT NULL
        ORDER BY file_path
        """
    ).fetchall()

    if not rows:
        return {}

    bpm_values = [float(row["bpm"]) for row in rows]
    loudness_values = [float(row["loudness"]) for row in rows]
    centroid_values = [float(row["spectral_centroid"]) for row in rows]

    bpm_min, bpm_max = min(bpm_values), max(bpm_values)
    loudness_min, loudness_max = min(loudness_values), max(loudness_values)
    centroid_min, centroid_max = min(centroid_values), max(centroid_values)

    features: dict[str, dict[str, float | int]] = {}
    for row in rows:
        bpm = float(row["bpm"])
        key_value = int(row["key"])
        mode = int(row["mode"])
        energy = float(row["energy"])
        loudness = float(row["loudness"])
        spectral_centroid = float(row["spectral_centroid"])

        features[row["file_path"]] = {
            "bpm": bpm,
            "key": key_value,
            "mode": mode,
            "vector": (
                normalize(bpm, bpm_min, bpm_max),
                key_value / 11.0,
                float(mode),
                energy,
                normalize(loudness, loudness_min, loudness_max),
                normalize(spectral_centroid, centroid_min, centroid_max),
            ),
        }

    return features


def base_similarity_for_pair(
    feature_map: dict[str, dict[str, float | int]],
    file_path_a: str,
    file_path_b: str,
) -> float | None:
    track_a = feature_map.get(file_path_a)
    track_b = feature_map.get(file_path_b)
    if track_a is None or track_b is None:
        return None

    vector_a = track_a["vector"]
    vector_b = track_b["vector"]
    return float(cosine_similarity(vector_a, vector_b))


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


def apply_feedback() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with get_connection() as connection:
        ensure_feedback_table(connection)
        ensure_required_tables(connection)

        feedback_rows = connection.execute(
            """
            SELECT file_path_a, file_path_b, rating, created_at
            FROM feedback
            ORDER BY created_at, rowid
            """
        ).fetchall()

        if not feedback_rows:
            LOGGER.info("Feedback toegepast: 0 similarity-rijen bijgewerkt.")
            return 0

        feature_map = load_track_features(connection)
        adjusted_pairs: dict[tuple[str, str], float] = {}

        for row in feedback_rows:
            file_path_a = str(row["file_path_a"])
            file_path_b = str(row["file_path_b"])
            pair = (file_path_a, file_path_b)

            if pair not in adjusted_pairs:
                base_similarity = base_similarity_for_pair(feature_map, file_path_a, file_path_b)
                if base_similarity is None:
                    continue
                existing_pair = connection.execute(
                    """
                    SELECT 1
                    FROM similarities
                    WHERE file_path_a = ? AND file_path_b = ?
                    """,
                    pair,
                ).fetchone()
                if existing_pair is None:
                    continue
                adjusted_pairs[pair] = base_similarity

            delta = 0.05 if int(row["rating"]) == 1 else -0.1
            adjusted_pairs[pair] = clamp(adjusted_pairs[pair] + delta, 0.0, 1.0)

        updated_rows = 0
        for (file_path_a, file_path_b), similarity in adjusted_pairs.items():
            track_a = feature_map[file_path_a]
            track_b = feature_map[file_path_b]
            mix_type = classify_mix_type(
                similarity=similarity,
                bpm_a=float(track_a["bpm"]),
                bpm_b=float(track_b["bpm"]),
                key_a=int(track_a["key"]),
                mode_a=int(track_a["mode"]),
                key_b=int(track_b["key"]),
                mode_b=int(track_b["mode"]),
            )
            connection.execute(
                """
                UPDATE similarities
                SET similarity = ?, mix_type = ?
                WHERE file_path_a = ? AND file_path_b = ?
                """,
                (similarity, mix_type, file_path_a, file_path_b),
            )
            updated_rows += 1

        connection.commit()
        LOGGER.info("Feedback toegepast: %s similarity-rijen bijgewerkt.", updated_rows)
        return updated_rows


def get_feedback_stats() -> dict[str, Any]:
    with get_connection() as connection:
        ensure_feedback_table(connection)

        summary_row = connection.execute(
            """
            SELECT COUNT(*) AS total_feedback_records, AVG(rating) AS average_rating
            FROM feedback
            """
        ).fetchone()

        top_rows = connection.execute(
            """
            SELECT
                f.file_path_a,
                f.file_path_b,
                COALESCE(s.similarity, 0.0) AS similarity,
                SUM(f.rating) AS rating_score
            FROM feedback AS f
            LEFT JOIN similarities AS s
                ON s.file_path_a = f.file_path_a
               AND s.file_path_b = f.file_path_b
            GROUP BY f.file_path_a, f.file_path_b
            ORDER BY rating_score DESC, similarity DESC, f.file_path_a, f.file_path_b
            LIMIT 10
            """
        ).fetchall()

        return {
            "total_feedback_records": int(summary_row["total_feedback_records"]) if summary_row else 0,
            "average_rating": float(summary_row["average_rating"]) if summary_row and summary_row["average_rating"] is not None else 0.0,
            "top_positive_pairs": [
                {
                    "file_path_a": row["file_path_a"],
                    "file_path_b": row["file_path_b"],
                    "similarity": float(row["similarity"]),
                }
                for row in top_rows
            ],
        }
