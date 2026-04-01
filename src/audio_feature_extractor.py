from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import librosa

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db


LOGGER = logging.getLogger("audio_feature_extractor")
HOP_LENGTH = 512
INTRO_OUTRO_THRESHOLD = 0.15
INTRO_STABLE_SECONDS = 3.0


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_schema(connection: sqlite3.Connection) -> None:
    columns = get_table_columns(connection, "audio_features")
    for column_name in (
        "percussiveness",
        "vocal_presence",
        "intro_length",
        "outro_length",
        "updated_at",
    ):
        if column_name not in columns:
            column_type = "TEXT" if column_name == "updated_at" else "REAL"
            connection.execute(
                f"ALTER TABLE audio_features ADD COLUMN {column_name} {column_type}"
            )

    connection.execute(
        """
        UPDATE audio_features
        SET updated_at = ?
        WHERE updated_at IS NULL OR TRIM(updated_at) = ''
        """,
        (current_timestamp(),),
    )

    connection.commit()


def pending_tracks(connection: sqlite3.Connection) -> list[tuple[str, bool]]:
    rows = connection.execute(
        """
        SELECT
            t.file_path,
            CASE WHEN af.file_path IS NULL THEN 1 ELSE 0 END AS needs_insert
        FROM tracks AS t
        LEFT JOIN audio_features AS af
            ON af.file_path = t.file_path
        WHERE af.file_path IS NULL
           OR af.percussiveness IS NULL
           OR af.vocal_presence IS NULL
           OR af.intro_length IS NULL
           OR af.outro_length IS NULL
        ORDER BY t.file_path
        """
    ).fetchall()
    return [(row[0], bool(row[1])) for row in rows]


def count_existing_vocal_presence(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM audio_features
        WHERE vocal_presence IS NOT NULL
        """
    ).fetchone()
    return int(row[0]) if row else 0


def infer_mode_from_chroma(chroma_mean, key_index: int) -> int:
    major_third = float(chroma_mean[(key_index + 4) % 12])
    minor_third = float(chroma_mean[(key_index + 3) % 12])
    return 1 if major_third >= minor_third else 0


def detect_intro_length(rms_values, sr: int) -> float:
    stable_frames = max(1, math.ceil((INTRO_STABLE_SECONDS * sr) / HOP_LENGTH))
    if len(rms_values) < stable_frames:
        return 0.0

    for start_index in range(0, len(rms_values) - stable_frames + 1):
        window = rms_values[start_index : start_index + stable_frames]
        if all(float(value) > INTRO_OUTRO_THRESHOLD for value in window):
            return float(start_index * HOP_LENGTH / sr)

    return 0.0


def detect_outro_length(rms_values, sr: int, duration_seconds: float) -> float:
    last_above_index = -1
    for index, value in enumerate(rms_values):
        if float(value) >= INTRO_OUTRO_THRESHOLD:
            last_above_index = index

    if last_above_index < 0 or last_above_index >= len(rms_values) - 1:
        return 0.0

    outro_start_seconds = float((last_above_index + 1) * HOP_LENGTH / sr)
    return max(0.0, duration_seconds - outro_start_seconds)


def compute_features(file_path: str) -> dict[str, float | int]:
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration_seconds = float(len(y) / sr) if sr else 0.0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_index = int(chroma_mean.argmax())
    mode = infer_mode_from_chroma(chroma_mean, key_index)

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    energy = float(rms.mean())
    loudness = float(librosa.amplitude_to_db(rms, ref=1.0).mean())
    spectral_centroid = float(
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH).mean()
    )
    percussiveness = float(librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH).mean())

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    vocal_presence_raw = float(mfcc[10:20].mean())

    intro_length = detect_intro_length(rms, sr)
    outro_length = detect_outro_length(rms, sr, duration_seconds)

    return {
        "bpm": float(tempo),
        "key": key_index,
        "mode": mode,
        "energy": energy,
        "loudness": loudness,
        "spectral_centroid": spectral_centroid,
        "percussiveness": percussiveness,
        "vocal_presence": vocal_presence_raw,
        "intro_length": intro_length,
        "outro_length": outro_length,
    }


def insert_features(connection: sqlite3.Connection, file_path: str, features: dict[str, float | int]) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO audio_features (
            file_path,
            bpm,
            key,
            mode,
            energy,
            loudness,
            spectral_centroid,
            percussiveness,
            vocal_presence,
            intro_length,
            outro_length,
            updated_at
        ) VALUES (
            :file_path,
            :bpm,
            :key,
            :mode,
            :energy,
            :loudness,
            :spectral_centroid,
            :percussiveness,
            :vocal_presence,
            :intro_length,
            :outro_length,
            :updated_at
        )
        """,
        {"file_path": file_path, "updated_at": current_timestamp(), **features},
    )
    connection.commit()


def update_extra_features(connection: sqlite3.Connection, file_path: str, features: dict[str, float | int]) -> None:
    connection.execute(
        """
        UPDATE audio_features
        SET percussiveness = :percussiveness,
            vocal_presence = :vocal_presence,
            intro_length = :intro_length,
            outro_length = :outro_length,
            updated_at = :updated_at
        WHERE file_path = :file_path
        """,
        {
            "file_path": file_path,
            "percussiveness": features["percussiveness"],
            "vocal_presence": features["vocal_presence"],
            "intro_length": features["intro_length"],
            "outro_length": features["outro_length"],
            "updated_at": current_timestamp(),
        },
    )
    connection.commit()


def normalize_vocal_presence(
    connection: sqlite3.Connection,
    processed_paths: list[str],
    normalize_all_rows: bool,
) -> None:
    if not processed_paths and not normalize_all_rows:
        return

    min_max_row = connection.execute(
        """
        SELECT MIN(vocal_presence), MAX(vocal_presence)
        FROM audio_features
        WHERE vocal_presence IS NOT NULL
        """
    ).fetchone()

    if not min_max_row or min_max_row[0] is None or min_max_row[1] is None:
        return

    minimum = float(min_max_row[0])
    maximum = float(min_max_row[1])

    if normalize_all_rows:
        if maximum <= minimum:
            connection.execute(
                """
                UPDATE audio_features
                SET vocal_presence = 0.0
                WHERE vocal_presence IS NOT NULL
                """
            )
        else:
            connection.execute(
                """
                UPDATE audio_features
                SET vocal_presence = (vocal_presence - ?) / ?
                WHERE vocal_presence IS NOT NULL
                """,
                (minimum, maximum - minimum),
            )
        connection.commit()
        return

    if not processed_paths:
        return

    placeholders = ", ".join("?" for _ in processed_paths)
    if maximum <= minimum:
        connection.execute(
            f"""
            UPDATE audio_features
            SET vocal_presence = 0.0
            WHERE file_path IN ({placeholders})
            """,
            processed_paths,
        )
    else:
        connection.execute(
            f"""
            UPDATE audio_features
            SET vocal_presence = (vocal_presence - ?) / ?
            WHERE file_path IN ({placeholders})
            """,
            [minimum, maximum - minimum, *processed_paths],
        )
    connection.commit()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bereken audio features voor tracks in SQLite.")
    parser.add_argument("--db-path", required=True, help="Pad naar de SQLite database.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_argument_parser()
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser()
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        existing_vocal_presence_count = count_existing_vocal_presence(connection)
        processed_paths: list[str] = []

        for file_path, needs_insert in pending_tracks(connection):
            try:
                features = compute_features(file_path)
                if needs_insert:
                    insert_features(connection, file_path, features)
                    LOGGER.info("Features opgeslagen voor %s", file_path)
                else:
                    update_extra_features(connection, file_path, features)
                    LOGGER.info("Extra features bijgewerkt voor %s", file_path)
                processed_paths.append(file_path)
            except Exception as exc:
                LOGGER.error("Fout bij verwerken van %s: %s", file_path, exc)

        normalize_vocal_presence(
            connection,
            processed_paths=processed_paths,
            normalize_all_rows=existing_vocal_presence_count == 0,
        )


if __name__ == "__main__":
    main()
