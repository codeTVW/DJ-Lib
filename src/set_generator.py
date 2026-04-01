from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dj_library.sqlite3"
CONFIG_PATH = Path.home() / ".djlibrary" / "config.json"


def _resolve_database_path() -> Path:
    local_database = Path(__file__).resolve().with_name("database.db")
    if local_database.exists():
        return local_database

    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            configured = data.get("database_path")
            if configured:
                configured_path = Path(configured).expanduser()
                if configured_path.exists():
                    return configured_path
        except (json.JSONDecodeError, OSError):
            pass

    return DEFAULT_DB_PATH


def _get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(_resolve_database_path())
    connection.row_factory = sqlite3.Row
    return connection


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _load_tracks() -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    with _get_connection() as connection:
        columns = _table_columns(connection, "audio_features")
        vocal_select = ", af.vocal_presence" if "vocal_presence" in columns else ", NULL AS vocal_presence"
        track_rows = connection.execute(
            f"""
            SELECT
                t.file_path,
                COALESCE(t.artist, 'Unknown Artist') AS artist,
                COALESCE(t.title, 'Unknown Title') AS title,
                af.bpm,
                af.energy
                {vocal_select}
            FROM tracks AS t
            JOIN audio_features AS af
                ON af.file_path = t.file_path
            WHERE af.bpm IS NOT NULL
              AND af.energy IS NOT NULL
            ORDER BY t.artist COLLATE NOCASE, t.title COLLATE NOCASE
            """
        ).fetchall()
        similarity_rows = connection.execute(
            """
            SELECT file_path_a, file_path_b, similarity
            FROM similarities
            """
        ).fetchall()

    tracks = [dict(row) for row in track_rows]
    similarity_map: dict[tuple[str, str], float] = {}
    for row in similarity_rows:
        pair = (row["file_path_a"], row["file_path_b"])
        reverse_pair = (row["file_path_b"], row["file_path_a"])
        similarity = float(row["similarity"])
        similarity_map[pair] = similarity
        similarity_map[reverse_pair] = max(similarity, similarity_map.get(reverse_pair, 0.0))

    return tracks, similarity_map


def _phase_counts(length: int, ratios: list[float]) -> list[int]:
    raw_counts = [length * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remainder = length - sum(counts)
    fractional_parts = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in fractional_parts[:remainder]:
        counts[index] += 1
    return counts


def _artist_allowed(candidate_artist: str, recent_artists: list[str]) -> bool:
    if len(recent_artists) < 2:
        return True
    return not (recent_artists[-1] == candidate_artist and recent_artists[-2] == candidate_artist)


def _similarity_with_previous(
    previous_track: dict[str, Any] | None,
    candidate_track: dict[str, Any],
    similarity_map: dict[tuple[str, str], float],
) -> float:
    if previous_track is None:
        return 0.0
    return float(
        similarity_map.get((previous_track["file_path"], candidate_track["file_path"]), 0.0)
    )


def _pick_track(
    candidates: list[dict[str, Any]],
    previous_track: dict[str, Any] | None,
    recent_artists: list[str],
    used_paths: set[str],
    similarity_map: dict[tuple[str, str], float],
    target_bpm: float | None = None,
    target_energy: float | None = None,
    prefer_low_vocals: bool = False,
) -> dict[str, Any] | None:
    available = [track for track in candidates if track["file_path"] not in used_paths]
    if not available:
        return None

    def sort_key(track: dict[str, Any]) -> tuple[float, float, float, float, str, str]:
        similarity = _similarity_with_previous(previous_track, track, similarity_map)
        bpm_score = -abs(float(track["bpm"]) - target_bpm) if target_bpm is not None else 0.0
        energy_score = -abs(float(track["energy"]) - target_energy) if target_energy is not None else 0.0
        vocal_value = float(track["vocal_presence"]) if track["vocal_presence"] is not None else 0.5
        vocal_score = -vocal_value if prefer_low_vocals else 0.0
        return (
            similarity,
            bpm_score,
            energy_score,
            vocal_score,
            str(track["artist"]).lower(),
            str(track["title"]).lower(),
        )

    available.sort(key=sort_key, reverse=True)

    for track in available:
        if _artist_allowed(str(track["artist"]), recent_artists):
            return track

    return available[0]


def _club_phases(length: int) -> list[dict[str, Any]]:
    counts = _phase_counts(length, [0.20, 0.25, 0.40, 0.15])
    return [
        {
            "name": "warm-up",
            "count": counts[0],
            "filter": lambda track: float(track["energy"]) < 0.5 and float(track["bpm"]) < 120,
            "target_bpm": lambda index, total: 100 + (18 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.30 + (0.18 * index / max(total - 1, 1)),
            "prefer_low_vocals": False,
        },
        {
            "name": "build-up",
            "count": counts[1],
            "filter": lambda track: 0.5 <= float(track["energy"]) <= 0.75 and 120 <= float(track["bpm"]) <= 128,
            "target_bpm": lambda index, total: 120 + (8 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.50 + (0.25 * index / max(total - 1, 1)),
            "prefer_low_vocals": False,
        },
        {
            "name": "peak-time",
            "count": counts[2],
            "filter": lambda track: float(track["energy"]) > 0.75 and float(track["bpm"]) > 128,
            "target_bpm": lambda index, total: 128 + (8 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.76 + (0.18 * index / max(total - 1, 1)),
            "prefer_low_vocals": False,
        },
        {
            "name": "closing",
            "count": counts[3],
            "filter": lambda track: 0.4 <= float(track["energy"]) <= 0.75,
            "target_bpm": lambda index, total: 126 - (10 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.75 - (0.35 * index / max(total - 1, 1)),
            "prefer_low_vocals": False,
        },
    ]


def _afterhours_phases(length: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "afterhours",
            "count": length,
            "filter": lambda track: float(track["energy"]) < 0.5 and 120 <= float(track["bpm"]) <= 130,
            "target_bpm": lambda index, total: 122 + (6 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.28 + (0.16 * index / max(total - 1, 1)),
            "prefer_low_vocals": True,
        }
    ]


def _warmup_phases(length: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "warm-up",
            "count": length,
            "filter": lambda track: float(track["energy"]) < 0.45 and 100 <= float(track["bpm"]) <= 118,
            "target_bpm": lambda index, total: 100 + (18 * index / max(total - 1, 1)),
            "target_energy": lambda index, total: 0.20 + (0.20 * index / max(total - 1, 1)),
            "prefer_low_vocals": False,
        }
    ]


def _phase_plan(trajectory: str, length: int) -> list[dict[str, Any]]:
    if trajectory == "club":
        return _club_phases(length)
    if trajectory == "afterhours":
        return _afterhours_phases(length)
    if trajectory == "warmup":
        return _warmup_phases(length)
    raise ValueError("Trajectory moet 'club', 'afterhours' of 'warmup' zijn.")


def generate_set(trajectory: str, length: int) -> list[dict[str, Any]]:
    if length <= 0:
        return []

    tracks, similarity_map = _load_tracks()
    phase_plan = _phase_plan(trajectory, length)

    selected_tracks: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    recent_artists: list[str] = []
    previous_track: dict[str, Any] | None = None

    for phase in phase_plan:
        phase_candidates = [track for track in tracks if phase["filter"](track)]
        for index in range(phase["count"]):
            target_bpm = phase["target_bpm"](index, phase["count"])
            target_energy = phase["target_energy"](index, phase["count"])
            chosen = _pick_track(
                candidates=phase_candidates,
                previous_track=previous_track,
                recent_artists=recent_artists,
                used_paths=used_paths,
                similarity_map=similarity_map,
                target_bpm=target_bpm,
                target_energy=target_energy,
                prefer_low_vocals=bool(phase["prefer_low_vocals"]),
            )
            if chosen is None:
                break

            used_paths.add(chosen["file_path"])
            recent_artists.append(str(chosen["artist"]))
            recent_artists = recent_artists[-2:]
            previous_track = chosen
            selected_tracks.append(
                {
                    "file_path": chosen["file_path"],
                    "artist": chosen["artist"],
                    "title": chosen["title"],
                    "bpm": float(chosen["bpm"]),
                    "energy": float(chosen["energy"]),
                    "phase": phase["name"],
                }
            )

    if len(selected_tracks) < length:
        remaining_tracks = [track for track in tracks if track["file_path"] not in used_paths]
        while len(selected_tracks) < length and remaining_tracks:
            chosen = _pick_track(
                candidates=remaining_tracks,
                previous_track=previous_track,
                recent_artists=recent_artists,
                used_paths=used_paths,
                similarity_map=similarity_map,
            )
            if chosen is None:
                break
            used_paths.add(chosen["file_path"])
            recent_artists.append(str(chosen["artist"]))
            recent_artists = recent_artists[-2:]
            previous_track = chosen
            selected_tracks.append(
                {
                    "file_path": chosen["file_path"],
                    "artist": chosen["artist"],
                    "title": chosen["title"],
                    "bpm": float(chosen["bpm"]),
                    "energy": float(chosen["energy"]),
                    "phase": "fallback",
                }
            )
            remaining_tracks = [track for track in remaining_tracks if track["file_path"] not in used_paths]

    return selected_tracks[:length]
