from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("set_generator")
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


def _load_similarity_map(connection: sqlite3.Connection) -> dict[tuple[str, str], float]:
    rows = connection.execute(
        """
        SELECT file_path_a, file_path_b, similarity
        FROM similarities
        """
    ).fetchall()

    similarity_map: dict[tuple[str, str], float] = {}
    for row in rows:
        similarity_map[(str(row["file_path_a"]), str(row["file_path_b"]))] = float(row["similarity"])
    return similarity_map


def _club_phase_plan(length: int) -> list[dict[str, Any]]:
    warmup_count = int(length * 0.20)
    buildup_count = int(length * 0.25)
    closing_count = int(length * 0.15)
    peaktime_count = length - warmup_count - buildup_count - closing_count

    return [
        {
            "name": "warm-up",
            "count": warmup_count,
            "where_clause": "af.energy < 0.5 AND af.bpm < 120",
            "target_bpm": 110.0,
            "target_energy": 0.40,
            "prefer_low_vocals": False,
        },
        {
            "name": "build-up",
            "count": buildup_count,
            "where_clause": "af.energy >= 0.5 AND af.energy <= 0.75 AND af.bpm >= 120 AND af.bpm <= 128",
            "target_bpm": 124.0,
            "target_energy": 0.62,
            "prefer_low_vocals": False,
        },
        {
            "name": "peak-time",
            "count": peaktime_count,
            "where_clause": "af.energy > 0.75 AND af.bpm > 128",
            "target_bpm": 132.0,
            "target_energy": 0.85,
            "prefer_low_vocals": False,
        },
        {
            "name": "closing",
            "count": closing_count,
            "where_clause": "af.energy >= 0.4 AND af.energy <= 0.75",
            "target_bpm": 120.0,
            "target_energy": 0.55,
            "prefer_low_vocals": False,
        },
    ]


def _afterhours_phase_plan(length: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "afterhours",
            "count": length,
            "where_clause": "af.energy < 0.5 AND af.bpm >= 120 AND af.bpm <= 130",
            "target_bpm": 125.0,
            "target_energy": 0.35,
            "prefer_low_vocals": True,
        }
    ]


def _warmup_phase_plan(length: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "warmup",
            "count": length,
            "where_clause": "af.energy < 0.45 AND af.bpm >= 100 AND af.bpm <= 118",
            "target_bpm": 109.0,
            "target_energy": 0.30,
            "prefer_low_vocals": False,
        }
    ]


def _phase_plan(trajectory: str, length: int) -> list[dict[str, Any]]:
    if trajectory == "club":
        return _club_phase_plan(length)
    if trajectory == "afterhours":
        return _afterhours_phase_plan(length)
    if trajectory == "warmup":
        return _warmup_phase_plan(length)
    raise ValueError("Trajectory moet 'club', 'afterhours' of 'warmup' zijn.")


def _query_phase_tracks(connection: sqlite3.Connection, where_clause: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"""
        SELECT
            t.file_path,
            COALESCE(t.artist, 'Unknown Artist') AS artist,
            COALESCE(t.title, 'Unknown Title') AS title,
            af.bpm,
            af.energy,
            af.vocal_presence
        FROM audio_features AS af
        JOIN tracks AS t
            ON t.file_path = af.file_path
        WHERE {where_clause}
        ORDER BY t.artist COLLATE NOCASE, t.title COLLATE NOCASE
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _query_all_tracks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            t.file_path,
            COALESCE(t.artist, 'Unknown Artist') AS artist,
            COALESCE(t.title, 'Unknown Title') AS title,
            af.bpm,
            af.energy,
            af.vocal_presence
        FROM audio_features AS af
        JOIN tracks AS t
            ON t.file_path = af.file_path
        WHERE af.bpm IS NOT NULL
          AND af.energy IS NOT NULL
        ORDER BY t.artist COLLATE NOCASE, t.title COLLATE NOCASE
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _similarity_with_previous(
    previous_track: dict[str, Any] | None,
    candidate_track: dict[str, Any],
    similarity_map: dict[tuple[str, str], float],
) -> float:
    if previous_track is None:
        return 0.0
    return float(
        similarity_map.get(
            (str(previous_track["file_path"]), str(candidate_track["file_path"])),
            0.0,
        )
    )


def _criteria_distance(track: dict[str, Any], target_bpm: float, target_energy: float) -> float:
    return abs(float(track["bpm"]) - target_bpm) + abs(float(track["energy"]) - target_energy)


def _creates_three_artist_streak(candidate_artist: str, selected_tracks: list[dict[str, Any]]) -> bool:
    if len(selected_tracks) < 2:
        return False
    return (
        str(selected_tracks[-1]["artist"]) == candidate_artist
        and str(selected_tracks[-2]["artist"]) == candidate_artist
    )


def _pick_candidate(
    candidates: list[dict[str, Any]],
    selected_paths: set[str],
    selected_tracks: list[dict[str, Any]],
    previous_track: dict[str, Any] | None,
    similarity_map: dict[tuple[str, str], float],
    target_bpm: float,
    target_energy: float,
    prefer_low_vocals: bool,
    fallback_mode: bool,
) -> dict[str, Any] | None:
    available = [candidate for candidate in candidates if candidate["file_path"] not in selected_paths]
    if not available:
        return None

    def ranking_key(track: dict[str, Any]) -> tuple[float, float, float, str, str]:
        similarity = _similarity_with_previous(previous_track, track, similarity_map)
        distance = _criteria_distance(track, target_bpm, target_energy)
        vocal_value = float(track["vocal_presence"]) if track["vocal_presence"] is not None else 0.5
        vocal_score = -vocal_value if prefer_low_vocals else 0.0
        if fallback_mode:
            return (
                -distance,
                similarity,
                vocal_score,
                str(track["artist"]).lower(),
                str(track["title"]).lower(),
            )
        return (
            similarity,
            -distance,
            vocal_score,
            str(track["artist"]).lower(),
            str(track["title"]).lower(),
        )

    ranked = sorted(available, key=ranking_key, reverse=True)
    for candidate in ranked:
        if not _creates_three_artist_streak(str(candidate["artist"]), selected_tracks):
            return candidate

    warning_candidate = ranked[0]
    LOGGER.warning(
        "Geen alternatieve artiest gevonden na %s; herhaling geaccepteerd voor %s.",
        str(selected_tracks[-1]["artist"]) if selected_tracks else "lege set",
        str(warning_candidate["file_path"]),
    )
    return warning_candidate


def _append_track(
    selected_tracks: list[dict[str, Any]],
    selected_paths: set[str],
    chosen: dict[str, Any],
    phase_name: str,
) -> None:
    selected_paths.add(str(chosen["file_path"]))
    selected_tracks.append(
        {
            "file_path": str(chosen["file_path"]),
            "artist": str(chosen["artist"]),
            "title": str(chosen["title"]),
            "bpm": float(chosen["bpm"]),
            "energy": float(chosen["energy"]),
            "phase": phase_name,
        }
    )


def generate_set(trajectory: str, length: int) -> list[dict[str, Any]]:
    if trajectory not in {"club", "afterhours", "warmup"}:
        raise ValueError("Trajectory moet 'club', 'afterhours' of 'warmup' zijn.")
    if length <= 0:
        return []

    with _get_connection() as connection:
        similarity_map = _load_similarity_map(connection)
        all_tracks = _query_all_tracks(connection)
        phase_plan = _phase_plan(trajectory, length)

        selected_tracks: list[dict[str, Any]] = []
        selected_paths: set[str] = set()

        for phase in phase_plan:
            phase_tracks = _query_phase_tracks(connection, str(phase["where_clause"]))
            previous_track = selected_tracks[-1] if selected_tracks else None

            for _ in range(int(phase["count"])):
                candidate = _pick_candidate(
                    candidates=phase_tracks,
                    selected_paths=selected_paths,
                    selected_tracks=selected_tracks,
                    previous_track=previous_track,
                    similarity_map=similarity_map,
                    target_bpm=float(phase["target_bpm"]),
                    target_energy=float(phase["target_energy"]),
                    prefer_low_vocals=bool(phase["prefer_low_vocals"]),
                    fallback_mode=False,
                )

                if candidate is None:
                    fallback_candidate = _pick_candidate(
                        candidates=all_tracks,
                        selected_paths=selected_paths,
                        selected_tracks=selected_tracks,
                        previous_track=previous_track,
                        similarity_map=similarity_map,
                        target_bpm=float(phase["target_bpm"]),
                        target_energy=float(phase["target_energy"]),
                        prefer_low_vocals=bool(phase["prefer_low_vocals"]),
                        fallback_mode=True,
                    )
                    if fallback_candidate is None:
                        LOGGER.warning(
                            "Geen tracks beschikbaar voor ontbrekende plek in fase %s.",
                            str(phase["name"]),
                        )
                        continue
                    LOGGER.warning(
                        "Fase %s had geen directe match; fallback gebruikt voor %s.",
                        str(phase["name"]),
                        str(fallback_candidate["file_path"]),
                    )
                    candidate = fallback_candidate

                _append_track(selected_tracks, selected_paths, candidate, str(phase["name"]))
                previous_track = selected_tracks[-1]

        return selected_tracks[:length]
