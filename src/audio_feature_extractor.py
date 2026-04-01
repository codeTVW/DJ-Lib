from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db


LOGGER = logging.getLogger("audio_feature_extractor")
HOP_LENGTH = 512
INTRO_OUTRO_THRESHOLD = 0.15
INTRO_STABLE_SECONDS = 3.0
STALE_DAYS = 7


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


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
           OR af.updated_at IS NULL
           OR julianday(REPLACE(af.updated_at, 'T', ' ')) <= julianday('now', ?)
        ORDER BY t.file_path
        """,
        (f"-{STALE_DAYS} days",),
    ).fetchall()
    return [(str(row[0]), bool(row[1])) for row in rows]


def extract_bpm(tempo: Any) -> float:
    try:
        return float(tempo[0])
    except (TypeError, IndexError, KeyError):
        return float(tempo)


def infer_mode_from_tonnetz(y, sr: int, chroma) -> int:
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
    tonnetz_mean = tonnetz.mean(axis=1)
    return 1 if float(tonnetz_mean[2]) >= 0.0 else 0


def detect_intro_length(rms_values, sr: int) -> float:
    if len(rms_values) == 0:
        return 0.0

    stable_frames = max(1, math.ceil((INTRO_STABLE_SECONDS * sr) / HOP_LENGTH))
    frame_times = librosa.frames_to_time(
        range(len(rms_values)),
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    for start_index in range(0, len(rms_values) - stable_frames + 1):
        if float(rms_values[start_index]) <= INTRO_OUTRO_THRESHOLD:
            continue

        window = rms_values[start_index : start_index + stable_frames]
        if not all(float(value) > INTRO_OUTRO_THRESHOLD for value in window):
            continue

        end_index = start_index + stable_frames - 1
        if float(frame_times[end_index] - frame_times[start_index]) >= INTRO_STABLE_SECONDS:
            return float(frame_times[start_index])

    return 0.0


def detect_outro_length(rms_values, sr: int, duration_seconds: float) -> float:
    if len(rms_values) == 0:
        return 0.0

    frame_times = librosa.frames_to_time(
        range(len(rms_values)),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    last_above_index = -1
    for index, value in enumerate(rms_values):
        if float(value) >= INTRO_OUTRO_THRESHOLD:
            last_above_index = index

    if last_above_index < 0 or last_above_index >= len(rms_values) - 1:
        return 0.0

    outro_start_index = last_above_index + 1
    if float(rms_values[outro_start_index]) >= INTRO_OUTRO_THRESHOLD:
        return 0.0

    return max(0.0, float(duration_seconds - frame_times[outro_start_index]))


def compute_features(file_path: str) -> dict[str, float | int]:
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration_seconds = float(len(y) / sr) if sr else 0.0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = extract_bpm(tempo)
    if bpm < 60.0 or bpm > 220.0:
        LOGGER.warning("Ongebruikelijke BPM voor %s: %.2f", file_path, bpm)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_index = int(chroma_mean.argmax())
    mode = infer_mode_from_tonnetz(y, sr, chroma)

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_mean = float(rms.mean())
    energy = rms_mean
    loudness = float(librosa.amplitude_to_db(rms_mean, ref=1.0))
    spectral_centroid = float(
        librosa.feature.spectral_centroid(y=y, sr=22050, hop_length=HOP_LENGTH).mean()
    )
    percussiveness = float(
        librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH).mean()
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    vocal_band_mean = mfcc[10:20].mean(axis=1)
    vocal_presence = float(vocal_band_mean.mean())

    intro_length = detect_intro_length(rms, sr)
    outro_length = detect_outro_length(rms, sr, duration_seconds)

    return {
        "bpm": bpm,
        "key": key_index,
        "mode": mode,
        "energy": energy,
        "loudness": loudness,
        "spectral_centroid": spectral_centroid,
        "percussiveness": percussiveness,
        "vocal_presence": vocal_presence,
        "intro_length": intro_length,
        "outro_length": outro_length,
    }


def insert_features(
    connection: sqlite3.Connection,
    file_path: str,
    features: dict[str, float | int],
) -> None:
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


def update_features(
    connection: sqlite3.Connection,
    file_path: str,
    features: dict[str, float | int],
) -> None:
    connection.execute(
        """
        UPDATE audio_features
        SET bpm = :bpm,
            key = :key,
            mode = :mode,
            energy = :energy,
            loudness = :loudness,
            spectral_centroid = :spectral_centroid,
            percussiveness = :percussiveness,
            vocal_presence = :vocal_presence,
            intro_length = :intro_length,
            outro_length = :outro_length,
            updated_at = :updated_at
        WHERE file_path = :file_path
        """,
        {"file_path": file_path, "updated_at": current_timestamp(), **features},
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

        for file_path, needs_insert in pending_tracks(connection):
            try:
                features = compute_features(file_path)
                if needs_insert:
                    insert_features(connection, file_path, features)
                    LOGGER.info("Features opgeslagen voor %s", file_path)
                else:
                    update_features(connection, file_path, features)
                    LOGGER.info("Features bijgewerkt voor %s", file_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Fout bij verwerken van %s: %s", file_path, exc)


if __name__ == "__main__":
    main()
