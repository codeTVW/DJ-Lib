from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dj_library.sqlite3"
_DB_PATH = DEFAULT_DB_PATH
MIX_TYPE_ORDER = {"safe": 0, "creative": 1, "risky": 2}


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


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def ensure_required_tables(connection: sqlite3.Connection) -> None:
    required_tables = ("tracks", "audio_features", "similarities")
    missing = [name for name in required_tables if not table_exists(connection, name)]
    if missing:
        raise RuntimeError(f"Ontbrekende tabellen: {', '.join(missing)}")


def fetch_audio_features(
    connection: sqlite3.Connection,
    file_paths: list[str],
) -> dict[str, sqlite3.Row]:
    if not file_paths:
        return {}

    columns = table_columns(connection, "audio_features")
    has_vocal_presence = "vocal_presence" in columns
    vocal_select = ", vocal_presence" if has_vocal_presence else ", NULL AS vocal_presence"
    placeholders = ", ".join("?" for _ in file_paths)
    rows = connection.execute(
        f"""
        SELECT
            file_path,
            bpm,
            key,
            mode,
            energy,
            loudness,
            spectral_centroid
            {vocal_select}
        FROM audio_features
        WHERE file_path IN ({placeholders})
        """,
        file_paths,
    ).fetchall()
    return {row["file_path"]: row for row in rows}


def word_count(text: str) -> int:
    return len([part for part in text.replace(",", " ").replace(".", " ").split() if part])


def build_reason(source: sqlite3.Row, target: sqlite3.Row) -> str:
    phrases: list[str] = []
    bpm_difference = abs(float(source["bpm"]) - float(target["bpm"]))
    energy_difference = abs(float(source["energy"]) - float(target["energy"]))
    loudness_difference = abs(float(source["loudness"]) - float(target["loudness"]))
    centroid_difference = abs(
        float(source["spectral_centroid"]) - float(target["spectral_centroid"])
    )

    if int(source["key"]) == int(target["key"]) and int(source["mode"]) == int(target["mode"]):
        phrases.append("zelfde key")
    elif int(source["mode"]) == int(target["mode"]):
        phrases.append("zelfde mode")

    if bpm_difference <= 1:
        phrases.append("zelfde BPM")
    elif bpm_difference <= 4:
        phrases.append(f"BPM-verschil {int(round(bpm_difference))}")
    elif bpm_difference < 8:
        phrases.append("vergelijkbaar tempo")

    if energy_difference <= 0.12:
        phrases.append("vergelijkbare energie")
    if loudness_difference <= 3.0:
        phrases.append("vergelijkbare loudness")
    if centroid_difference <= 350.0:
        phrases.append("vergelijkbaar timbre")

    vocal_source = source["vocal_presence"]
    vocal_target = target["vocal_presence"]
    if (
        vocal_source is not None
        and vocal_target is not None
        and abs(float(vocal_source) - float(vocal_target)) <= 0.18
        and max(float(vocal_source), float(vocal_target)) >= 0.65
    ):
        phrases.append("zelfde vocal focus")

    if not phrases:
        phrases.append("goede stilistische match")

    selected: list[str] = []
    for phrase in phrases:
        candidate = ", ".join(selected + [phrase])
        if word_count(candidate) <= 10:
            selected.append(phrase)

    sentence = ", ".join(selected) if selected else phrases[0]
    sentence = sentence[:1].upper() + sentence[1:]
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def get_next_tracks(file_path: str, played_tracks: list[str], n: int = 5) -> list[dict[str, Any]]:
    if n <= 0:
        return []

    played = set(played_tracks)
    played.add(file_path)

    with get_connection() as connection:
        ensure_required_tables(connection)

        rows = connection.execute(
            """
            SELECT file_path_b, similarity, mix_type
            FROM similarities
            WHERE file_path_a = ?
            """,
            (file_path,),
        ).fetchall()

        filtered_rows = [row for row in rows if row["file_path_b"] not in played]
        filtered_rows.sort(
            key=lambda row: (
                MIX_TYPE_ORDER.get(str(row["mix_type"]), 99),
                -float(row["similarity"]),
                str(row["file_path_b"]),
            )
        )
        filtered_rows = filtered_rows[:n]

        feature_paths = [file_path] + [row["file_path_b"] for row in filtered_rows]
        features = fetch_audio_features(connection, feature_paths)
        source_features = features.get(file_path)
        if source_features is None:
            raise RuntimeError(f"Geen audio features gevonden voor {file_path}")

        recommendations: list[dict[str, Any]] = []
        for row in filtered_rows:
            target_path = row["file_path_b"]
            target_features = features.get(target_path)
            if target_features is None:
                continue

            recommendations.append(
                {
                    "file_path": target_path,
                    "similarity": float(row["similarity"]),
                    "mix_type": str(row["mix_type"]),
                    "reason": build_reason(source_features, target_features),
                }
            )

        return recommendations
