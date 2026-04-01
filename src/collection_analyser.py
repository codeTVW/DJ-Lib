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
                COALESCE(t.artist, 'Unknown Artist') AS artist,
                COALESCE(t.title, 'Unknown Title') AS title,
                AVG(similarity_rows.similarity) AS avg_similarity
            FROM clusters AS c
            JOIN tracks AS t
                ON t.file_path = c.file_path
            LEFT JOIN (
                SELECT file_path_a AS file_path, similarity FROM similarities
                UNION ALL
                SELECT file_path_b AS file_path, similarity FROM similarities
            ) AS similarity_rows
                ON similarity_rows.file_path = c.file_path
            WHERE c.cluster_id = -1
            GROUP BY t.file_path, artist, title
            ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE
            """
        ).fetchall()

    return [
        {
            "file_path": row["file_path"],
            "artist": row["artist"],
            "title": row["title"],
            "avg_similarity": float(row["avg_similarity"]) if row["avg_similarity"] is not None else 0.0,
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
                COALESCE(cm.name, CASE WHEN c.cluster_id = -1 THEN 'Orphan' END, 'Unclustered') AS name,
                COUNT(*) AS track_count
            FROM clusters AS c
            LEFT JOIN cluster_metadata AS cm
                ON cm.cluster_id = c.cluster_id
            WHERE c.cluster_id != -1
            GROUP BY c.cluster_id, name
            HAVING COUNT(*) > 50
            ORDER BY track_count DESC, c.cluster_id
            """
        ).fetchall()

    return [
        {
            "cluster_id": int(row["cluster_id"]),
            "name": row["name"],
            "track_count": int(row["track_count"]),
        }
        for row in rows
    ]


def find_missing_bridges() -> list[dict[str, Any]]:
    with get_connection() as connection:
        ensure_required_tables(connection)

        cluster_rows = connection.execute(
            """
            SELECT DISTINCT
                c.cluster_id,
                COALESCE(cm.name, 'Cluster ' || c.cluster_id) AS name
            FROM clusters AS c
            LEFT JOIN cluster_metadata AS cm
                ON cm.cluster_id = c.cluster_id
            WHERE c.cluster_id != -1
            ORDER BY c.cluster_id
            """
        ).fetchall()
        clusters = [(int(row["cluster_id"]), row["name"]) for row in cluster_rows]

        safe_rows = connection.execute(
            """
            SELECT
                s.file_path_a,
                c.cluster_id
            FROM similarities AS s
            JOIN clusters AS c
                ON c.file_path = s.file_path_b
            WHERE s.mix_type = 'safe'
              AND c.cluster_id != -1
            GROUP BY s.file_path_a, c.cluster_id
            """
        ).fetchall()

    safe_cluster_map: dict[str, set[int]] = {}
    for row in safe_rows:
        safe_cluster_map.setdefault(row["file_path_a"], set()).add(int(row["cluster_id"]))

    bridge_pairs = {
        tuple(sorted((cluster_a, cluster_b)))
        for cluster_ids in safe_cluster_map.values()
        for cluster_a, cluster_b in combinations(sorted(cluster_ids), 2)
    }

    cluster_name_map = {cluster_id: name for cluster_id, name in clusters}
    missing_pairs = []
    for cluster_a_id, cluster_b_id in combinations(sorted(cluster_name_map), 2):
        if (cluster_a_id, cluster_b_id) in bridge_pairs:
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

        total_tracks_row = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()
        cluster_count_row = connection.execute(
            """
            SELECT COUNT(DISTINCT cluster_id)
            FROM clusters
            WHERE cluster_id != -1
            """
        ).fetchone()
        orphan_count_row = connection.execute(
            """
            SELECT COUNT(*)
            FROM clusters
            WHERE cluster_id = -1
            """
        ).fetchone()
        crate_rows = connection.execute(
            """
            SELECT crate_name, COUNT(*) AS track_count
            FROM crates
            GROUP BY crate_name
            ORDER BY crate_name COLLATE NOCASE
            """
        ).fetchall()
        bpm_rows = connection.execute(
            """
            SELECT bpm
            FROM audio_features
            WHERE bpm IS NOT NULL
            ORDER BY bpm
            """
        ).fetchall()

    total_tracks = int(total_tracks_row[0]) if total_tracks_row else 0
    cluster_count = int(cluster_count_row[0]) if cluster_count_row else 0
    orphan_count = int(orphan_count_row[0]) if orphan_count_row else 0

    crate_percentages = {
        row["crate_name"]: (float(row["track_count"]) / total_tracks * 100.0 if total_tracks else 0.0)
        for row in crate_rows
    }

    bpm_values = [float(row["bpm"]) for row in bpm_rows]
    bpm_histogram: dict[str, int] = {}
    if bpm_values:
        start = int(min(bpm_values) // 10 * 10)
        stop = int(max(bpm_values) // 10 * 10 + 10)
        for value in range(start, stop + 1, 10):
            label = f"{value}-{value + 9}"
            bpm_histogram[label] = 0
        for bpm in bpm_values:
            bucket_start = int(bpm // 10 * 10)
            label = f"{bucket_start}-{bucket_start + 9}"
            bpm_histogram[label] = bpm_histogram.get(label, 0) + 1

    return {
        "total_tracks": total_tracks,
        "cluster_count": cluster_count,
        "orphan_count": orphan_count,
        "crate_percentages": crate_percentages,
        "bpm_histogram": bpm_histogram,
    }
