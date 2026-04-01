from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from database_init import CREATE_SIMILARITIES_TABLE
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import CREATE_SIMILARITIES_TABLE


LOGGER = logging.getLogger("similarity_engine")
TOP_N = 10

CAMELot_MAP: dict[tuple[int, int], tuple[int, str]] = {
    (0, 1): (8, "B"),
    (1, 1): (3, "B"),
    (2, 1): (10, "B"),
    (3, 1): (5, "B"),
    (4, 1): (12, "B"),
    (5, 1): (7, "B"),
    (6, 1): (2, "B"),
    (7, 1): (9, "B"),
    (8, 1): (4, "B"),
    (9, 1): (11, "B"),
    (10, 1): (6, "B"),
    (11, 1): (1, "B"),
    (0, 0): (5, "A"),
    (1, 0): (12, "A"),
    (2, 0): (7, "A"),
    (3, 0): (2, "A"),
    (4, 0): (9, "A"),
    (5, 0): (4, "A"),
    (6, 0): (11, "A"),
    (7, 0): (6, "A"),
    (8, 0): (1, "A"),
    (9, 0): (8, "A"),
    (10, 0): (3, "A"),
    (11, 0): (10, "A"),
}


@dataclass(frozen=True)
class TrackFeatures:
    file_path: str
    bpm: float
    key: int
    mode: int
    updated_at: datetime
    vector: tuple[float, float, float, float, float, float]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)

    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def cosine_similarity(vector_a: tuple[float, ...], vector_b: tuple[float, ...]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


def camelot_compatible(key_a: int, mode_a: int, key_b: int, mode_b: int) -> bool:
    number_a, letter_a = CAMELot_MAP[(key_a, mode_a)]
    number_b, letter_b = CAMELot_MAP[(key_b, mode_b)]

    if number_a == number_b:
        return True

    if letter_a != letter_b:
        return False

    delta = (number_a - number_b) % 12
    return delta in {1, 11}


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
    is_camelot_compatible = camelot_compatible(key_a, mode_a, key_b, mode_b)

    if similarity > 0.85 and bpm_difference < 4.0 and is_camelot_compatible:
        return "safe"
    if 0.65 <= similarity <= 0.85 and bpm_difference < 8.0:
        return "creative"
    return "risky"


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def ensure_similarities_schema(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_SIMILARITIES_TABLE)
    connection.commit()


def ensure_audio_features_schema(connection: sqlite3.Connection) -> None:
    track_columns = get_table_columns(connection, "tracks")
    if not track_columns:
        raise RuntimeError("De tracks tabel ontbreekt in de database.")

    feature_columns = get_table_columns(connection, "audio_features")
    if not feature_columns:
        raise RuntimeError("De audio_features tabel ontbreekt in de database.")

    required_columns = {
        "file_path",
        "bpm",
        "key",
        "mode",
        "energy",
        "loudness",
        "spectral_centroid",
    }
    missing_columns = required_columns - feature_columns
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"De audio_features tabel mist vereiste kolommen: {missing_list}")

    if "updated_at" not in feature_columns:
        connection.execute("ALTER TABLE audio_features ADD COLUMN updated_at TEXT")

    connection.execute(
        """
        UPDATE audio_features
        SET updated_at = ?
        WHERE updated_at IS NULL OR TRIM(updated_at) = ''
        """,
        (utc_now_iso(),),
    )
    connection.commit()


def get_last_run_epoch(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def set_last_run_epoch(connection: sqlite3.Connection, epoch_seconds: int) -> None:
    connection.execute(f"PRAGMA user_version = {epoch_seconds}")


def load_tracks(connection: sqlite3.Connection) -> dict[str, TrackFeatures]:
    rows = connection.execute(
        """
        SELECT
            t.file_path,
            af.bpm,
            af.key,
            af.mode,
            af.energy,
            af.loudness,
            af.spectral_centroid,
            af.updated_at
        FROM tracks AS t
        JOIN audio_features AS af
            ON af.file_path = t.file_path
        WHERE af.bpm IS NOT NULL
          AND af.key IS NOT NULL
          AND af.mode IS NOT NULL
          AND af.energy IS NOT NULL
          AND af.loudness IS NOT NULL
          AND af.spectral_centroid IS NOT NULL
          AND af.updated_at IS NOT NULL
        ORDER BY t.file_path
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

    tracks: dict[str, TrackFeatures] = {}
    for row in rows:
        bpm = float(row["bpm"])
        key = int(row["key"])
        mode = int(row["mode"])
        energy = float(row["energy"])
        loudness = float(row["loudness"])
        spectral_centroid = float(row["spectral_centroid"])

        tracks[row["file_path"]] = TrackFeatures(
            file_path=row["file_path"],
            bpm=bpm,
            key=key,
            mode=mode,
            updated_at=parse_timestamp(str(row["updated_at"])),
            vector=(
                normalize(bpm, bpm_min, bpm_max),
                key / 11.0,
                float(mode),
                energy,
                normalize(loudness, loudness_min, loudness_max),
                normalize(spectral_centroid, centroid_min, centroid_max),
            ),
        )

    return tracks


def changed_tracks(
    connection: sqlite3.Connection,
    tracks: dict[str, TrackFeatures],
    last_run_epoch: int,
) -> list[str]:
    if not tracks:
        return []

    has_similarity_rows = connection.execute(
        "SELECT 1 FROM similarities LIMIT 1"
    ).fetchone()
    if last_run_epoch <= 0 or has_similarity_rows is None:
        return sorted(tracks)

    changed = {
        file_path
        for file_path, track in tracks.items()
        if int(track.updated_at.timestamp()) > last_run_epoch
    }

    if len(tracks) > 1:
        existing_sources = {
            row["file_path_a"]
            for row in connection.execute(
                "SELECT DISTINCT file_path_a FROM similarities"
            ).fetchall()
        }
        changed.update(file_path for file_path in tracks if file_path not in existing_sources)

    return sorted(changed)


def rebuild_similarities(
    connection: sqlite3.Connection,
    tracks: dict[str, TrackFeatures],
    sources_to_rebuild: list[str],
) -> int:
    if not sources_to_rebuild:
        return 0

    connection.executemany(
        "DELETE FROM similarities WHERE file_path_a = ?",
        [(file_path,) for file_path in sources_to_rebuild],
    )

    rows_to_insert: list[tuple[str, str, float, str]] = []
    all_tracks = list(tracks.values())

    for source_path in sources_to_rebuild:
        source_track = tracks[source_path]
        ranked_matches: list[tuple[str, str, float, str]] = []

        for target_track in all_tracks:
            if target_track.file_path == source_track.file_path:
                continue

            similarity = cosine_similarity(source_track.vector, target_track.vector)
            mix_type = classify_mix_type(
                similarity=similarity,
                bpm_a=source_track.bpm,
                bpm_b=target_track.bpm,
                key_a=source_track.key,
                mode_a=source_track.mode,
                key_b=target_track.key,
                mode_b=target_track.mode,
            )
            ranked_matches.append(
                (
                    source_track.file_path,
                    target_track.file_path,
                    similarity,
                    mix_type,
                )
            )

        ranked_matches.sort(key=lambda row: (-row[2], row[1]))
        rows_to_insert.extend(ranked_matches[:TOP_N])

    if rows_to_insert:
        connection.executemany(
            """
            INSERT INTO similarities (
                file_path_a,
                file_path_b,
                similarity,
                mix_type
            ) VALUES (?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    connection.commit()
    return len(rows_to_insert)


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
        ensure_similarities_schema(connection)
        ensure_audio_features_schema(connection)

        tracks = load_tracks(connection)
        if not tracks:
            LOGGER.info("Geen tracks met complete audio features gevonden.")
            return

        last_run_epoch = get_last_run_epoch(connection)
        sources_to_rebuild = changed_tracks(connection, tracks, last_run_epoch)
        if not sources_to_rebuild:
            LOGGER.info("Geen gewijzigde tracks sinds de vorige similarity-run.")
            set_last_run_epoch(connection, int(utc_now().timestamp()))
            connection.commit()
            return

        inserted_rows = rebuild_similarities(connection, tracks, sources_to_rebuild)
        set_last_run_epoch(connection, int(utc_now().timestamp()))
        connection.commit()
        LOGGER.info(
            "Similarity-run voltooid: %s tracks herberekend, %s similarity-rijen opgeslagen.",
            len(sources_to_rebuild),
            inserted_rows,
        )


if __name__ == "__main__":
    main()
