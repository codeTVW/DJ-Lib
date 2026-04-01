from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any


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
    required_tables = ("tracks", "audio_features", "similarities", "clusters", "cluster_metadata", "crates")
    missing = [name for name in required_tables if not table_exists(connection, name)]
    if missing:
        raise RuntimeError(f"Ontbrekende tabellen: {', '.join(missing)}")


def find_orphans() -> list[dict[str, Any]]:
    with get_connection() as connection:
        ensure_required_tables(connection)
        rows = connection.execute(
            """
            SELECT
                t.file_path,
                COALESCE(t.artist, '') AS artist,
                COALESCE(t.title, '') AS title,
                COALESCE((
                    SELECT AVG(similarity_value)
                    FROM (
                        SELECT similarity AS similarity_value
                        FROM similarities
                        WHERE file_path_a = t.file_path
                        UNION ALL
                        SELECT similarity AS similarity_value
                        FROM similarities
                        WHERE file_path_b = t.file_path
                    )
                ), 0.0) AS avg_similarity
            FROM clusters AS c
            JOIN tracks AS t
                ON t.file_path = c.file_path
            WHERE c.cluster_id = -1
            ORDER BY avg_similarity ASC, t.file_path
            """
        ).fetchall()

    return [
        {
            "file_path": str(row["file_path"]),
            "artist": str(row["artist"]),
            "title": str(row["title"]),
            "avg_similarity": float(row["avg_similarity"]),
        }
        for row in rows
    ]


def find_overcrowded_clusters() -> list[dict[str, Any]]:
    with get_connection() as connection:
        ensure_required_tables(connection)
        rows = connection.execute(
            """
            SELECT
                c.cluster_id,
                COALESCE(cm.name, 'Cluster ' || c.cluster_id) AS name,
                COUNT(*) AS track_count
            FROM clusters AS c
            LEFT JOIN cluster_metadata AS cm
                ON cm.cluster_id = c.cluster_id
            GROUP BY c.cluster_id, name
            HAVING COUNT(*) > 50
            ORDER BY track_count DESC, c.cluster_id
            """
        ).fetchall()

    return [
        {
            "cluster_id": int(row["cluster_id"]),
            "name": str(row["name"]),
            "track_count": int(row["track_count"]),
        }
        for row in rows
    ]


def find_missing_bridges() -> list[dict[str, Any]]:
    with get_connection() as connection:
        ensure_required_tables(connection)
        cluster_rows = connection.execute(
            """
            SELECT cluster_id, name
            FROM cluster_metadata
            WHERE cluster_id != -1
            ORDER BY cluster_id
            """
        ).fetchall()

        clusters = [(int(row["cluster_id"]), str(row["name"])) for row in cluster_rows]
        cluster_name_map = {cluster_id: name for cluster_id, name in clusters}

        missing_pairs: list[dict[str, Any]] = []
        for cluster_a_id, cluster_b_id in combinations(sorted(cluster_name_map), 2):
            bridge_row = connection.execute(
                """
                SELECT 1
                FROM similarities AS s
                JOIN clusters AS c1
                    ON c1.file_path = s.file_path_b
                WHERE c1.cluster_id = ?
                  AND s.mix_type IN ('safe', 'creative')
                  AND EXISTS (
                      SELECT 1
                      FROM similarities AS s2
                      JOIN clusters AS c2
                          ON c2.file_path = s2.file_path_b
                      WHERE s2.file_path_a = s.file_path_a
                        AND c2.cluster_id = ?
                        AND s2.mix_type IN ('safe', 'creative')
                  )
                LIMIT 1
                """,
                (cluster_a_id, cluster_b_id),
            ).fetchone()
            if bridge_row is not None:
                continue

            missing_pairs.append(
                {
                    "cluster_a_id": cluster_a_id,
                    "cluster_a_name": cluster_name_map[cluster_a_id],
                    "cluster_b_id": cluster_b_id,
                    "cluster_b_name": cluster_name_map[cluster_b_id],
                }
            )

    return missing_pairs


def get_collection_stats() -> dict[str, Any]:
    with get_connection() as connection:
        ensure_required_tables(connection)

        total_tracks = int(connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
        total_clusters = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT cluster_id)
                FROM clusters
                WHERE cluster_id != -1
                """
            ).fetchone()[0]
        )
        total_orphans = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM clusters
                WHERE cluster_id = -1
                """
            ).fetchone()[0]
        )
        crate_rows = connection.execute(
            """
            SELECT crate_name, COUNT(*) AS track_count
            FROM crates
            GROUP BY crate_name
            ORDER BY crate_name
            """
        ).fetchall()
        bpm_rows = connection.execute(
            """
            SELECT bpm
            FROM audio_features
            WHERE bpm BETWEEN 60 AND 229.999
            """
        ).fetchall()

    crate_percentages = {
        str(row["crate_name"]): round(
            (float(row["track_count"]) / total_tracks * 100.0) if total_tracks else 0.0,
            1,
        )
        for row in crate_rows
    }

    bpm_histogram = {f"{start}-{start + 9}": 0 for start in range(60, 221, 10)}
    for row in bpm_rows:
        bpm = float(row["bpm"])
        bucket_start = int(bpm // 10 * 10)
        if bucket_start < 60:
            continue
        if bucket_start > 220:
            bucket_start = 220
        label = f"{bucket_start}-{bucket_start + 9}"
        bpm_histogram[label] = bpm_histogram.get(label, 0) + 1

    return {
        "total_tracks": total_tracks,
        "total_clusters": total_clusters,
        "total_orphans": total_orphans,
        "cluster_count": total_clusters,
        "orphan_count": total_orphans,
        "crate_percentages": crate_percentages,
        "bpm_histogram": bpm_histogram,
    }
