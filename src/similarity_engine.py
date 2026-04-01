from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from database_init import CREATE_SIMILARITIES_TABLE
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import CREATE_SIMILARITIES_TABLE


LOGGER = logging.getLogger("similarity_engine")
TOP_N = 10
LAST_RUN_KEY = "last_run"
CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
)
"""

CAMELOT_MAP: dict[tuple[int, int], str] = {
    (0, 1): "8B",
    (1, 1): "3B",
    (2, 1): "10B",
    (3, 1): "5B",
    (4, 1): "12B",
    (5, 1): "7B",
    (6, 1): "2B",
    (7, 1): "9B",
    (8, 1): "4B",
    (9, 1): "11B",
    (10, 1): "6B",
    (11, 1): "1B",
    (0, 0): "5A",
    (1, 0): "12A",
    (2, 0): "7A",
    (3, 0): "2A",
    (4, 0): "9A",
    (5, 0): "4A",
    (6, 0): "11A",
    (7, 0): "6A",
    (8, 0): "1A",
    (9, 0): "8A",
    (10, 0): "3A",
    (11, 0): "10A",
}


@dataclass(frozen=True)
class TrackMetadata:
    file_path: str
    bpm: float
    key: int
    mode: int
    updated_at: datetime


@dataclass(frozen=True)
class TrackDataset:
    file_paths: list[str]
    file_path_to_index: dict[str, int]
    metadata_by_path: dict[str, TrackMetadata]
    matrix: np.ndarray


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def normalize_min_max(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def ensure_settings_table(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_SETTINGS_TABLE)
    connection.commit()


def ensure_similarities_table(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_SIMILARITIES_TABLE)
    connection.commit()


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def ensure_audio_features_schema(connection: sqlite3.Connection) -> None:
    columns = get_table_columns(connection, "audio_features")
    required_columns = {
        "file_path",
        "bpm",
        "key",
        "mode",
        "energy",
        "loudness",
        "spectral_centroid",
        "updated_at",
    }
    missing_columns = required_columns - columns
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"De audio_features tabel mist vereiste kolommen: {missing_list}")


def get_last_run(connection: sqlite3.Connection) -> datetime | None:
    row = connection.execute(
        """
        SELECT value
        FROM settings
        WHERE key = ?
        """,
        (LAST_RUN_KEY,),
    ).fetchone()
    if row is None or row["value"] in (None, ""):
        return None
    return parse_timestamp(str(row["value"]))


def set_last_run(connection: sqlite3.Connection, value: str) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO settings (key, value)
        VALUES (?, ?)
        """,
        (LAST_RUN_KEY, value),
    )
    connection.commit()


def load_dataset(connection: sqlite3.Connection) -> TrackDataset:
    rows = connection.execute(
        """
        SELECT
            file_path,
            bpm,
            key,
            mode,
            energy,
            loudness,
            spectral_centroid,
            updated_at
        FROM audio_features
        WHERE bpm IS NOT NULL
          AND key IS NOT NULL
          AND mode IS NOT NULL
          AND energy IS NOT NULL
          AND loudness IS NOT NULL
          AND spectral_centroid IS NOT NULL
          AND updated_at IS NOT NULL
        ORDER BY file_path
        """
    ).fetchall()

    if not rows:
        return TrackDataset([], {}, {}, np.empty((0, 6), dtype=float))

    bpm_values = [float(row["bpm"]) for row in rows]
    loudness_values = [float(row["loudness"]) for row in rows]
    centroid_values = [float(row["spectral_centroid"]) for row in rows]

    bpm_max = max(bpm_values) if bpm_values else 0.0
    loudness_min, loudness_max = min(loudness_values), max(loudness_values)
    centroid_min, centroid_max = min(centroid_values), max(centroid_values)

    file_paths: list[str] = []
    metadata_by_path: dict[str, TrackMetadata] = {}
    vectors: list[list[float]] = []

    for row in rows:
        file_path = str(row["file_path"])
        bpm = float(row["bpm"])
        key = int(row["key"])
        mode = int(row["mode"])
        energy = float(row["energy"])
        loudness = float(row["loudness"])
        spectral_centroid = float(row["spectral_centroid"])

        file_paths.append(file_path)
        metadata_by_path[file_path] = TrackMetadata(
            file_path=file_path,
            bpm=bpm,
            key=key,
            mode=mode,
            updated_at=parse_timestamp(str(row["updated_at"])),
        )
        vectors.append(
            [
                (bpm / bpm_max) if bpm_max > 0.0 else 0.0,
                key / 11.0,
                float(mode),
                energy,
                normalize_min_max(loudness, loudness_min, loudness_max),
                normalize_min_max(spectral_centroid, centroid_min, centroid_max),
            ]
        )

    return TrackDataset(
        file_paths=file_paths,
        file_path_to_index={file_path: index for index, file_path in enumerate(file_paths)},
        metadata_by_path=metadata_by_path,
        matrix=np.array(vectors, dtype=float),
    )


def changed_file_paths(dataset: TrackDataset, last_run: datetime | None) -> list[str]:
    if not dataset.file_paths:
        return []
    if last_run is None:
        return list(dataset.file_paths)
    return [
        file_path
        for file_path in dataset.file_paths
        if dataset.metadata_by_path[file_path].updated_at > last_run
    ]


def to_camelot_code(key: int, mode: int) -> str:
    return CAMELOT_MAP[(key, mode)]


def camelot_compatible(key_a: int, mode_a: int, key_b: int, mode_b: int) -> bool:
    code_a = to_camelot_code(key_a, mode_a)
    code_b = to_camelot_code(key_b, mode_b)

    number_a, letter_a = int(code_a[:-1]), code_a[-1]
    number_b, letter_b = int(code_b[:-1]), code_b[-1]

    if number_a == number_b:
        return True

    if abs(number_a - number_b) == 1 or abs(number_a - number_b) == 11:
        return True

    return number_a == number_b and letter_a != letter_b


def classify_mix_type(
    similarity: float,
    bpm_a: float,
    bpm_b: float,
    key_a: int,
    mode_a: int,
    key_b: int,
    mode_b: int,
) -> str:
    bpm_difference = abs(bpm_a - bpm_b)
    if similarity > 0.85 and bpm_difference < 4.0 and camelot_compatible(key_a, mode_a, key_b, mode_b):
        return "safe"
    if 0.65 <= similarity <= 0.85 and bpm_difference < 8.0:
        return "creative"
    return "risky"


def rebuild_similarities(
    connection: sqlite3.Connection,
    dataset: TrackDataset,
    source_paths: list[str],
) -> int:
    if not source_paths or dataset.matrix.size == 0:
        return 0

    similarity_matrix = cosine_similarity(dataset.matrix)
    rows_to_write: list[tuple[str, str, float, str]] = []

    connection.executemany(
        "DELETE FROM similarities WHERE file_path_a = ?",
        [(file_path,) for file_path in source_paths],
    )

    for source_path in source_paths:
        source_index = dataset.file_path_to_index[source_path]
        source_metadata = dataset.metadata_by_path[source_path]
        row = similarity_matrix[source_index]
        ranked_indices = np.argsort(-row)

        added = 0
        for target_index in ranked_indices:
            if int(target_index) == source_index:
                continue

            target_path = dataset.file_paths[int(target_index)]
            target_metadata = dataset.metadata_by_path[target_path]
            similarity = float(row[int(target_index)])
            mix_type = classify_mix_type(
                similarity=similarity,
                bpm_a=source_metadata.bpm,
                bpm_b=target_metadata.bpm,
                key_a=source_metadata.key,
                mode_a=source_metadata.mode,
                key_b=target_metadata.key,
                mode_b=target_metadata.mode,
            )
            rows_to_write.append((source_path, target_path, similarity, mix_type))
            added += 1
            if added >= TOP_N:
                break

    if rows_to_write:
        connection.executemany(
            """
            INSERT OR REPLACE INTO similarities (
                file_path_a,
                file_path_b,
                similarity,
                mix_type
            ) VALUES (?, ?, ?, ?)
            """,
            rows_to_write,
        )
        connection.commit()

    return len(rows_to_write)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bereken track similarities op basis van audio features.")
    parser.add_argument("--db-path", required=True, help="Pad naar de SQLite database.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        ensure_similarities_table(connection)
        ensure_settings_table(connection)
        ensure_audio_features_schema(connection)

        dataset = load_dataset(connection)
        if not dataset.file_paths:
            LOGGER.info("Geen tracks met complete audio features gevonden.")
            set_last_run(connection, current_timestamp())
            return

        last_run = get_last_run(connection)
        source_paths = changed_file_paths(dataset, last_run)
        if not source_paths:
            LOGGER.info("Geen gewijzigde tracks sinds de vorige similarity-run.")
            set_last_run(connection, current_timestamp())
            return

        stored_rows = rebuild_similarities(connection, dataset, source_paths)
        set_last_run(connection, current_timestamp())
        LOGGER.info(
            "Similarity-run voltooid: %s tracks herberekend, %s similarity-rijen opgeslagen.",
            len(source_paths),
            stored_rows,
        )


if __name__ == "__main__":
    main()
