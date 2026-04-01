from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import hdbscan
import umap

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
    vector: tuple[float, float, float, float, float, float]


def normalize(value: float, minimum: float, maximum: float) -> float:
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
            af.spectral_centroid
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

    bpm_min, bpm_max = min(bpm_values), max(bpm_values)
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
                file_path=row["file_path"],
                bpm=bpm,
                energy=energy,
                loudness=loudness,
                vector=(
                    normalize(bpm, bpm_min, bpm_max),
                    key_value / 11.0,
                    float(mode),
                    energy,
                    normalize(loudness, loudness_min, loudness_max),
                    normalize(spectral_centroid, centroid_min, centroid_max),
                ),
            )
        )

    return points


def compute_umap(points: list[TrackPoint]) -> list[tuple[float, float]]:
    vectors = [point.vector for point in points]
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    embedding = reducer.fit_transform(vectors)
    return [(float(row[0]), float(row[1])) for row in embedding]


def compute_clusters(coordinates: list[tuple[float, float]]) -> list[int]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(coordinates)
    return [int(label) for label in labels]


def clear_previous_outputs(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM umap_coordinates")
    connection.execute("DELETE FROM clusters")
    connection.execute("DELETE FROM cluster_metadata")


def cluster_name(avg_bpm: float, avg_energy: float, cluster_id: int) -> str:
    if cluster_id == -1:
        return "Orphan"
    if avg_energy > 0.75 and avg_bpm > 128:
        return "Peak-time"
    if avg_energy < 0.4 and avg_bpm < 110:
        return "Warm-up"
    if avg_energy < 0.4 and avg_bpm > 110:
        return "Late night"
    return f"Cluster {cluster_id}"


def cluster_description(name: str, cluster_id: int, avg_bpm: float, avg_energy: float) -> str:
    if cluster_id == -1:
        return "Tracks die HDBSCAN niet aan een stabiel cluster kon toewijzen."
    if name == "Peak-time":
        return "Hoge energie en hoog tempo, geschikt voor drukke piekmomenten."
    if name == "Warm-up":
        return "Lagere energie en rustiger tempo voor openingsuren of opbouw."
    if name == "Late night":
        return "Lage energie met hoger tempo voor diepere of latere setmomenten."
    return f"Gemengde cluster met gemiddeld {avg_bpm:.1f} BPM en energy {avg_energy:.2f}."


def store_outputs(
    connection: sqlite3.Connection,
    points: list[TrackPoint],
    coordinates: list[tuple[float, float]],
    labels: list[int],
) -> None:
    clear_previous_outputs(connection)

    connection.executemany(
        """
        INSERT INTO umap_coordinates (file_path, x, y)
        VALUES (?, ?, ?)
        """,
        [
            (point.file_path, coordinate[0], coordinate[1])
            for point, coordinate in zip(points, coordinates)
        ],
    )

    connection.executemany(
        """
        INSERT INTO clusters (cluster_id, file_path)
        VALUES (?, ?)
        """,
        [(label, point.file_path) for point, label in zip(points, labels)],
    )

    grouped_points: dict[int, list[TrackPoint]] = defaultdict(list)
    for point, label in zip(points, labels):
        grouped_points[label].append(point)

    metadata_rows = []
    for cluster_id in sorted(grouped_points):
        cluster_points = grouped_points[cluster_id]
        avg_bpm = fmean(point.bpm for point in cluster_points)
        avg_energy = fmean(point.energy for point in cluster_points)
        avg_loudness = fmean(point.loudness for point in cluster_points)
        name = cluster_name(avg_bpm, avg_energy, cluster_id)
        description = cluster_description(name, cluster_id, avg_bpm, avg_energy)
        metadata_rows.append(
            (
                cluster_id,
                name,
                description,
                avg_bpm,
                avg_energy,
                avg_loudness,
            )
        )

    connection.executemany(
        """
        INSERT INTO cluster_metadata (
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

        coordinates = compute_umap(points)
        labels = compute_clusters(coordinates)
        store_outputs(connection, points, coordinates, labels)
        LOGGER.info(
            "Clustering voltooid: %s tracks verwerkt, %s clusters opgeslagen.",
            len(points),
            len(set(labels)),
        )


if __name__ == "__main__":
    main()
