from __future__ import annotations

import argparse
import logging
import os
import plistlib
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mutagen import File as MutagenFile

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db


LOGGER = logging.getLogger("library_importer")
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aiff", ".flac"}
APPLE_MUSIC_XML = os.path.expanduser("~/Music/Music/Music Library.xml")
REKORDBOX_DIR = os.path.expanduser("~/Library/Pioneer/rekordbox")
MUSIC_DIR = os.path.expanduser("~/Music")


def decode_file_uri(location: str | None) -> str | None:
    if not location or not location.startswith("file://"):
        return None

    parsed = urlparse(location)
    path_text = unquote(parsed.path)
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        path_text = f"/{parsed.netloc}{path_text}"

    return os.path.normpath(os.path.expanduser(path_text))


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def tag_text(tags: Any, *keys: str) -> str:
    if not tags:
        return ""

    for key in keys:
        value = tags.get(key)
        if not value:
            continue
        if isinstance(value, list):
            first = value[0] if value else ""
            return text_value(first)
        return text_value(value)

    return ""


def apple_music_tracks(xml_path: str) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []

    try:
        with open(xml_path, "rb") as handle:
            library = plistlib.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Fout bij lezen van iTunes XML %s: %s", xml_path, exc)
        return tracks

    for raw_track in library.get("Tracks", {}).values():
        location = raw_track.get("Location")
        if not isinstance(location, str) or not location.startswith("file://"):
            continue

        try:
            file_path = decode_file_uri(location)
            if not file_path:
                continue

            duration_ms = safe_float(raw_track.get("Total Time"))
            tracks.append(
                {
                    "file_path": file_path,
                    "artist": text_value(raw_track.get("Artist")),
                    "title": text_value(raw_track.get("Name")),
                    "album": text_value(raw_track.get("Album")),
                    "genre": text_value(raw_track.get("Genre")),
                    "year": safe_int(raw_track.get("Year")),
                    "duration": (duration_ms / 1000.0) if duration_ms is not None else None,
                    "source": "itunes",
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Fout bij verwerken van iTunes-track %s: %s", location, exc)

    return tracks


def rekordbox_tracks(xml_directory: str) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []

    for xml_path in sorted(Path(xml_directory).glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Fout bij lezen van rekordbox XML %s: %s", xml_path, exc)
            continue

        for track in root.findall(".//TRACK"):
            location = track.attrib.get("Location")
            if not isinstance(location, str) or not location.startswith("file://"):
                continue

            try:
                file_path = decode_file_uri(location)
                if not file_path:
                    continue

                tracks.append(
                    {
                        "file_path": file_path,
                        "artist": text_value(track.attrib.get("Artist")),
                        "title": text_value(track.attrib.get("Name")),
                        "album": text_value(track.attrib.get("Album")),
                        "genre": text_value(track.attrib.get("Genre")),
                        "year": safe_int(track.attrib.get("Year")),
                        "duration": safe_float(track.attrib.get("TotalTime")),
                        "source": "rekordbox",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Fout bij verwerken van rekordbox-track %s: %s", location, exc)

    return tracks


def filesystem_tracks(root_directory: str) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []

    for current_root, _, file_names in os.walk(root_directory):
        for file_name in file_names:
            file_path = os.path.join(current_root, file_name)
            if os.path.splitext(file_name)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                audio_file = MutagenFile(file_path, easy=True)
                tags = getattr(audio_file, "tags", None) or {}
                info = getattr(audio_file, "info", None)
                duration = float(getattr(info, "length", 0.0) or 0.0)

                tracks.append(
                    {
                        "file_path": os.path.abspath(file_path),
                        "artist": tag_text(tags, "artist"),
                        "title": tag_text(tags, "title"),
                        "album": tag_text(tags, "album"),
                        "genre": tag_text(tags, "genre"),
                        "year": safe_int(tag_text(tags, "year", "date")),
                        "duration": duration,
                        "source": "scan",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Fout bij verwerken van %s: %s", file_path, exc)

    return tracks


def detect_source() -> tuple[str | None, list[dict[str, Any]]]:
    if os.path.exists(APPLE_MUSIC_XML):
        print(f"Bron gevonden: iTunes XML op {APPLE_MUSIC_XML}")
        return "itunes", apple_music_tracks(APPLE_MUSIC_XML)

    if os.path.exists(REKORDBOX_DIR):
        rekordbox_xml_files = sorted(Path(REKORDBOX_DIR).glob("*.xml"))
        if rekordbox_xml_files:
            print(f"Bron gevonden: rekordbox XML bestanden in {REKORDBOX_DIR}")
            return "rekordbox", rekordbox_tracks(REKORDBOX_DIR)

    if os.path.exists(MUSIC_DIR):
        print(f"Bron gevonden: recursieve scan van {MUSIC_DIR}")
        return "scan", filesystem_tracks(MUSIC_DIR)

    print(
        "Geen ondersteunde muziekbron gevonden. "
        "Gecontroleerd: ~/Music/Music/Music Library.xml, "
        "~/Library/Pioneer/rekordbox/ en ~/Music/."
    )
    return None, []


def insert_tracks(connection: sqlite3.Connection, tracks: list[dict[str, Any]]) -> int:
    before = connection.total_changes
    connection.executemany(
        """
        INSERT OR IGNORE INTO tracks (
            file_path,
            artist,
            title,
            album,
            genre,
            year,
            duration,
            source
        ) VALUES (
            :file_path,
            :artist,
            :title,
            :album,
            :genre,
            :year,
            :duration,
            :source
        )
        """,
        tracks,
    )
    connection.commit()
    return connection.total_changes - before


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importeer een lokale DJ-library naar SQLite.")
    parser.add_argument("--db-path", required=True, help="Pad naar de SQLite database.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_argument_parser()
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    init_db(db_path)

    source, tracks = detect_source()
    if source is None:
        return

    with sqlite3.connect(db_path) as connection:
        inserted = insert_tracks(connection, tracks)

    print(f"Import voltooid: {inserted} nieuwe tracks toegevoegd.")


if __name__ == "__main__":
    main()
