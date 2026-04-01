from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa

try:
    from domain_constants import UNUSUAL_BPM_MAX, UNUSUAL_BPM_MIN
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.domain_constants import UNUSUAL_BPM_MAX, UNUSUAL_BPM_MIN

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db


LOGGER = logging.getLogger("audio_feature_extractor")
HOP_LENGTH = 512
INTRO_OUTRO_THRESHOLD = 0.15
INTRO_STABLE_SECONDS = 3.0
STALE_DAYS = 7
COMMIT_INTERVAL = 25


@dataclass(frozen=True)
class TrackProcessingResult:
    file_path: str
    needs_insert: bool
    features: dict[str, float | int] | None
    warning: str | None = None
    error: str | None = None


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


def unusual_bpm_warning(file_path: str, bpm: float) -> str | None:
    if bpm < UNUSUAL_BPM_MIN or bpm > UNUSUAL_BPM_MAX:
        return f"Ongebruikelijke BPM voor {file_path}: {bpm:.2f}"
    return None


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


def process_track(task: tuple[str, bool]) -> TrackProcessingResult:
    file_path, needs_insert = task
    try:
        features = compute_features(file_path)
        warning = unusual_bpm_warning(file_path, float(features["bpm"]))
        return TrackProcessingResult(
            file_path=file_path,
            needs_insert=needs_insert,
            features=features,
            warning=warning,
        )
    except Exception as exc:  # noqa: BLE001
        return TrackProcessingResult(
            file_path=file_path,
            needs_insert=needs_insert,
            features=None,
            error=str(exc),
        )


def resolve_worker_count(requested_workers: int | None, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_workers is not None:
        return max(1, min(requested_workers, task_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, task_count))


def handle_result(connection: sqlite3.Connection, result: TrackProcessingResult) -> bool:
    if result.error is not None:
        LOGGER.error("Fout bij verwerken van %s: %s", result.file_path, result.error)
        return False

    if result.warning:
        LOGGER.warning("%s", result.warning)

    if result.features is None:
        LOGGER.error("Geen features ontvangen voor %s.", result.file_path)
        return False

    if result.needs_insert:
        insert_features(connection, result.file_path, result.features)
        LOGGER.info("Features opgeslagen voor %s", result.file_path)
    else:
        update_features(connection, result.file_path, result.features)
        LOGGER.info("Features bijgewerkt voor %s", result.file_path)
    return True


def iter_results_parallel(
    tasks: list[tuple[str, bool]],
    workers: int,
):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_track, task): task
            for task in tasks
        }
        for future in as_completed(future_map):
            file_path, needs_insert = future_map[future]
            try:
                yield future.result()
            except Exception as exc:  # noqa: BLE001
                yield TrackProcessingResult(
                    file_path=file_path,
                    needs_insert=needs_insert,
                    features=None,
                    error=str(exc),
                )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bereken audio features voor tracks in SQLite.")
    parser.add_argument("--db-path", required=True, help="Pad naar de SQLite database.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Aantal parallelle workers voor feature extraction. Standaard: aantal CPU-kernen.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_argument_parser()
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser()
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        tasks = pending_tracks(connection)
        if not tasks:
            LOGGER.info("Geen tracks gevonden die feature extraction nodig hebben.")
            return

        workers = resolve_worker_count(args.workers, len(tasks))
        LOGGER.info("Start feature extraction voor %s tracks met %s worker(s).", len(tasks), workers)

        processed_since_commit = 0
        if workers == 1:
            results = (process_track(task) for task in tasks)
        else:
            results = iter_results_parallel(tasks, workers)

        for result in results:
            try:
                if handle_result(connection, result):
                    processed_since_commit += 1
                    if processed_since_commit >= COMMIT_INTERVAL:
                        connection.commit()
                        processed_since_commit = 0
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Fout bij opslaan van %s: %s", result.file_path, exc)

        if processed_since_commit:
            connection.commit()


if __name__ == "__main__":
    main()
