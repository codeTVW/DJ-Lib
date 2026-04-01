from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from db_utils import get_connection, table_exists
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.db_utils import get_connection, table_exists

try:
    from similarity_engine import to_camelot_code
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.similarity_engine import to_camelot_code


def ensure_required_tables(connection) -> None:
    required_tables = ("tracks", "audio_features", "similarities")
    missing = [name for name in required_tables if not table_exists(connection, name)]
    if missing:
        raise RuntimeError(f"Ontbrekende tabellen: {', '.join(missing)}")


def _word_count(text: str) -> int:
    return len([part for part in text.replace(",", " ").replace(".", " ").split() if part])


def _build_reason(row) -> str:
    facts: list[tuple[int, str]] = []

    bpm_difference = abs(float(row["source_bpm"]) - float(row["target_bpm"]))
    energy_difference = abs(float(row["source_energy"]) - float(row["target_energy"]))
    source_camelot = to_camelot_code(int(row["source_key"]), int(row["source_mode"]))
    target_camelot = to_camelot_code(int(row["target_key"]), int(row["target_mode"]))

    if source_camelot == target_camelot:
        facts.append((0, "zelfde Camelot key"))

    facts.append((1, f"BPM-verschil {int(round(bpm_difference))}"))
    facts.append((2, f"energieverschil {energy_difference:.2f}"))

    selected: list[str] = []
    for _, fact in sorted(facts, key=lambda item: item[0]):
        candidate = ", ".join(selected + [fact])
        if _word_count(candidate) <= 10:
            selected.append(fact)
        if len(selected) == 2:
            break

    reason = ", ".join(selected) if selected else "goede match"
    reason = reason[:1].upper() + reason[1:]
    if not reason.endswith("."):
        reason += "."
    return reason


def get_next_tracks(
    file_path: str,
    played_tracks: list[str] = [],
    n: int = 5,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []

    excluded_paths = list(played_tracks)
    excluded_paths.append(file_path)

    with get_connection(db_path) as connection:
        ensure_required_tables(connection)

        where_clause = "WHERE s.file_path_a = ?"
        parameters: list[Any] = [file_path]
        if excluded_paths:
            placeholders = ", ".join("?" for _ in excluded_paths)
            where_clause += f" AND s.file_path_b NOT IN ({placeholders})"
            parameters.extend(excluded_paths)
        parameters.append(n)

        rows = connection.execute(
            f"""
            SELECT
                s.file_path_b AS file_path,
                COALESCE(t.artist, '') AS artist,
                COALESCE(t.title, '') AS title,
                s.similarity,
                s.mix_type,
                af_source.bpm AS source_bpm,
                af_target.bpm AS target_bpm,
                af_source.key AS source_key,
                af_target.key AS target_key,
                af_source.mode AS source_mode,
                af_target.mode AS target_mode,
                af_source.energy AS source_energy,
                af_target.energy AS target_energy
            FROM similarities AS s
            JOIN tracks AS t
                ON t.file_path = s.file_path_b
            JOIN audio_features AS af_source
                ON af_source.file_path = s.file_path_a
            JOIN audio_features AS af_target
                ON af_target.file_path = s.file_path_b
            {where_clause}
            ORDER BY
                CASE s.mix_type
                    WHEN 'safe' THEN 0
                    WHEN 'creative' THEN 1
                    WHEN 'risky' THEN 2
                    ELSE 3
                END,
                s.similarity DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()

    recommendations: list[dict[str, Any]] = []
    for row in rows:
        recommendations.append(
            {
                "file_path": str(row["file_path"]),
                "artist": str(row["artist"]),
                "title": str(row["title"]),
                "similarity": float(row["similarity"]),
                "mix_type": str(row["mix_type"]),
                "reason": _build_reason(row),
            }
        )

    return recommendations
