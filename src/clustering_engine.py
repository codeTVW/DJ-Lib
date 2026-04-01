from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numpy as np
import umap

try:
    from domain_constants import (
        BPM_PEAKTIME_MIN,
        BPM_WARMUP_MAX,
        CLUSTER_RECALC_THRESHOLD,
        ENERGY_LOW_MAX,
        ENERGY_MID_MAX,
        HDBSCAN_MIN_CLUSTER_SIZE,
        HDBSCAN_MIN_SAMPLES,
        UMAP_MIN_DIST,
        UMAP_N_COMPONENTS,
        UMAP_N_NEIGHBORS,
        UMAP_RANDOM_STATE,
    )
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.domain_constants import (
        BPM_PEAKTIME_MIN,
        BPM_WARMUP_MAX,
        CLUSTER_RECALC_THRESHOLD,
        ENERGY_LOW_MAX,
        ENERGY_MID_MAX,
        HDBSCAN_MIN_CLUSTER_SIZE,
        HDBSCAN_MIN_SAMPLES,
        UMAP_MIN_DIST,
        UMAP_N_COMPONENTS,
        UMAP_N_NEIGHBORS,
        UMAP_RANDOM_STATE,
    )

try:
    from database_init import (
        CREATE_CLUSTERS_TABLE,
        CREATE_CLUSTER_METADATA_TABLE,
        CREATE_UMAP_COORDINATES_TABLE,
    )
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import (
        CREATE_CLUSTERS_TABLE,
        CREATE_CLUSTER_METADATA_TABLE,
        CREATE_UMAP_COORDINATES_TABLE,
    )


LOGGER = logging.getLogger("clustering_engine")


@dataclass(frozen=True)
class TrackPoint:
    file_path: str
    bpm: float
    energy: float
    loudness: float
    genre: str
    vector: list[float]


def normalize_min_max(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def ensure_required_tables(connection: sqlite3.Connection) -> None:
    required_tables = ("tracks", "audio_features", "similarities")
    for table_name in required_tables:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"De vereiste tabel '{table_name}' ontbreekt in de database.")


def ensure_output_tables(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_UMAP_COORDINATES_TABLE)
    connection.execute(CREATE_CLUSTERS_TABLE)
    connection.execute(CREATE_CLUSTER_METADATA_TABLE)
    connection.commit()


def load_track_points(connection: sqlite3.Connection) -> list[TrackPoint]:
    rows = connection.execute(
        """
        SELECT
            af.file_path,
            af.bpm,
            af.key,
            af.mode,
            af.energy,
            af.loudness,
            af.spectral_centroid,
            COALESCE(t.genre, '') AS genre
        FROM audio_features AS af
        JOIN tracks AS t
            ON t.file_path = af.file_path
        WHERE af.bpm IS NOT NULL
          AND af.key IS NOT NULL
          AND af.mode IS NOT NULL
          AND af.energy IS NOT NULL
          AND af.loudness IS NOT NULL
          AND af.spectral_centroid IS NOT NULL
        ORDER BY af.file_path
        """
    ).fetchall()

    if not rows:
        return []

    bpm_values = [float(row["bpm"]) for row in rows]
    loudness_values = [float(row["loudness"]) for row in rows]
    centroid_values = [float(row["spectral_centroid"]) for row in rows]

    bpm_max = max(bpm_values) if bpm_values else 0.0
    loudness_min, loudness_max = min(loudness_values), max(loudness_values)
    centroid_min, centroid_max = min(centroid_values), max(centroid_values)

    points: list[TrackPoint] = []
    for row in rows:
        bpm = float(row["bpm"])
        key_value = int(row["key"])
        mode = int(row["mode"])
        energy = float(row["energy"])
        loudness = float(row["loudness"])
        spectral_centroid = float(row["spectral_centroid"])

        points.append(
            TrackPoint(
                file_path=str(row["file_path"]),
                bpm=bpm,
                energy=energy,
                loudness=loudness,
                genre=str(row["genre"] or "").strip(),
                vector=[
                    (bpm / bpm_max) if bpm_max > 0.0 else 0.0,
                    key_value / 11.0,
                    float(mode),
                    energy,
                    normalize_min_max(loudness, loudness_min, loudness_max),
                    normalize_min_max(spectral_centroid, centroid_min, centroid_max),
                ],
            )
        )

    return points


def feature_matrix(points: list[TrackPoint]) -> np.ndarray:
    if not points:
        return np.empty((0, 6), dtype=float)
    return np.array([point.vector for point in points], dtype=float)


def should_recalculate(connection: sqlite3.Connection, current_track_count: int) -> bool:
    stored_row = connection.execute(
        "SELECT COUNT(*) AS coordinate_count FROM umap_coordinates"
    ).fetchone()
    stored_count = int(stored_row["coordinate_count"]) if stored_row else 0

    if current_track_count == 0:
        return False
    if stored_count == 0:
        return True

    difference_ratio = abs(current_track_count - stored_count) / stored_count
    return difference_ratio > CLUSTER_RECALC_THRESHOLD


def compute_umap(matrix: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(matrix)


def compute_labels(embedding: np.ndarray) -> list[int]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    clusterer.fit(embedding)
    return [int(label) for label in clusterer.labels_]


def cluster_name(cluster_id: int, avg_bpm: float, avg_energy: float) -> str:
    if cluster_id == -1:
        return "Orphan"
    if avg_energy > ENERGY_MID_MAX and avg_bpm > BPM_PEAKTIME_MIN:
        return "Peak-time"
    if avg_energy < ENERGY_LOW_MAX and avg_bpm < BPM_WARMUP_MAX:
        return "Warm-up"
    if avg_energy < ENERGY_LOW_MAX and avg_bpm > BPM_WARMUP_MAX:
        return "Late night"
    return f"Cluster {cluster_id}"


def most_common_genre(points: list[TrackPoint]) -> str:
    genre_counts = Counter(point.genre for point in points if point.genre)
    if not genre_counts:
        return "onbekend"
    return str(genre_counts.most_common(1)[0][0]).lower()


def cluster_description(avg_bpm: float, avg_energy: float, genre: str) -> str:
    if avg_energy > ENERGY_MID_MAX:
        energy_text = "hoge energie"
    elif avg_energy < ENERGY_LOW_MAX:
        energy_text = "lage energie"
    else:
        energy_text = "gemiddelde energie"
    return f"Gemiddeld {round(avg_bpm):.0f} BPM, {energy_text}, overwegend {genre}."


def clear_previous_outputs(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM umap_coordinates")
    connection.execute("DELETE FROM clusters")
    connection.execute("DELETE FROM cluster_metadata")


def store_outputs(
    connection: sqlite3.Connection,
    points: list[TrackPoint],
    embedding: np.ndarray,
    labels: list[int],
) -> None:
    clear_previous_outputs(connection)

    connection.executemany(
        """
        INSERT OR REPLACE INTO umap_coordinates (file_path, x, y)
        VALUES (?, ?, ?)
        """,
        [
            (point.file_path, float(coordinate[0]), float(coordinate[1]))
            for point, coordinate in zip(points, embedding)
        ],
    )

    connection.executemany(
        """
        INSERT OR REPLACE INTO clusters (cluster_id, file_path)
        VALUES (?, ?)
        """,
        [(label, point.file_path) for point, label in zip(points, labels)],
    )

    grouped_points: dict[int, list[TrackPoint]] = defaultdict(list)
    for point, label in zip(points, labels):
        grouped_points[label].append(point)

    metadata_rows: list[tuple[int, str, str, float, float, float]] = []
    for cluster_id in sorted(grouped_points):
        cluster_points = grouped_points[cluster_id]
        avg_bpm = sum(point.bpm for point in cluster_points) / len(cluster_points)
        avg_energy = sum(point.energy for point in cluster_points) / len(cluster_points)
        avg_loudness = sum(point.loudness for point in cluster_points) / len(cluster_points)
        genre = most_common_genre(cluster_points)
        metadata_rows.append(
            (
                cluster_id,
                cluster_name(cluster_id, avg_bpm, avg_energy),
                cluster_description(avg_bpm, avg_energy, genre),
                avg_bpm,
                avg_energy,
                avg_loudness,
            )
        )

    connection.executemany(
        """
        INSERT OR REPLACE INTO cluster_metadata (
            cluster_id,
            name,
            description,
            avg_bpm,
            avg_energy,
            avg_loudness
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        metadata_rows,
    )
    connection.commit()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voer UMAP en HDBSCAN clustering uit op audio features.")
    parser.add_argument("--db-path", required=True, help="Pad naar de SQLite database.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        ensure_required_tables(connection)
        ensure_output_tables(connection)

        points = load_track_points(connection)
        if not points:
            LOGGER.info("Geen complete audio feature vectors gevonden.")
            return

        if not should_recalculate(connection, len(points)):
            print("Clustering overgeslagen: verschil in trackaantal is niet groter dan 10 procent.")
            return

        matrix = feature_matrix(points)
        embedding = compute_umap(matrix)
        labels = compute_labels(embedding)
        store_outputs(connection, points, embedding, labels)
        LOGGER.info(
            "Clustering voltooid: %s tracks verwerkt, %s clusters opgeslagen.",
            len(points),
            len(set(labels)),
        )


if __name__ == "__main__":
    main()
