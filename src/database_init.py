from __future__ import annotations

import sqlite3
from pathlib import Path


CREATE_TRACKS_TABLE = """
CREATE TABLE IF NOT EXISTS tracks (
    file_path TEXT PRIMARY KEY,
    artist TEXT,
    title TEXT,
    album TEXT,
    genre TEXT,
    year INTEGER,
    duration REAL,
    source TEXT
)
"""

CREATE_AUDIO_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS audio_features (
    file_path TEXT PRIMARY KEY,
    bpm REAL,
    key INTEGER,
    mode INTEGER,
    energy REAL,
    loudness REAL,
    spectral_centroid REAL,
    percussiveness REAL,
    vocal_presence REAL,
    intro_length REAL,
    outro_length REAL,
    updated_at TEXT,
    FOREIGN KEY (file_path) REFERENCES tracks(file_path)
)
"""

CREATE_SIMILARITIES_TABLE = """
CREATE TABLE IF NOT EXISTS similarities (
    file_path_a TEXT NOT NULL,
    file_path_b TEXT NOT NULL,
    similarity REAL NOT NULL,
    mix_type TEXT NOT NULL CHECK (mix_type IN ('safe', 'creative', 'risky')),
    PRIMARY KEY (file_path_a, file_path_b)
)
"""

CREATE_UMAP_COORDINATES_TABLE = """
CREATE TABLE IF NOT EXISTS umap_coordinates (
    file_path TEXT PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL
)
"""

CREATE_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER NOT NULL,
    file_path TEXT PRIMARY KEY
)
"""

CREATE_CLUSTER_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS cluster_metadata (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    avg_bpm REAL NOT NULL,
    avg_energy REAL NOT NULL,
    avg_loudness REAL NOT NULL
)
"""

CREATE_CRATES_TABLE = """
CREATE TABLE IF NOT EXISTS crates (
    crate_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (crate_name, file_path)
)
"""

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    file_path_a TEXT NOT NULL,
    file_path_b TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    created_at TEXT NOT NULL
)
"""

CREATE_TABLE_STATEMENTS = (
    CREATE_TRACKS_TABLE,
    CREATE_AUDIO_FEATURES_TABLE,
    CREATE_SIMILARITIES_TABLE,
    CREATE_UMAP_COORDINATES_TABLE,
    CREATE_CLUSTERS_TABLE,
    CREATE_CLUSTER_METADATA_TABLE,
    CREATE_CRATES_TABLE,
    CREATE_FEEDBACK_TABLE,
)

EXPECTED_COLUMNS: dict[str, dict[str, str]] = {
    "tracks": {
        "file_path": "TEXT",
        "artist": "TEXT",
        "title": "TEXT",
        "album": "TEXT",
        "genre": "TEXT",
        "year": "INTEGER",
        "duration": "REAL",
        "source": "TEXT",
    },
    "audio_features": {
        "file_path": "TEXT",
        "bpm": "REAL",
        "key": "INTEGER",
        "mode": "INTEGER",
        "energy": "REAL",
        "loudness": "REAL",
        "spectral_centroid": "REAL",
        "percussiveness": "REAL",
        "vocal_presence": "REAL",
        "intro_length": "REAL",
        "outro_length": "REAL",
        "updated_at": "TEXT",
    },
    "similarities": {
        "file_path_a": "TEXT",
        "file_path_b": "TEXT",
        "similarity": "REAL",
        "mix_type": "TEXT",
    },
    "umap_coordinates": {
        "file_path": "TEXT",
        "x": "REAL",
        "y": "REAL",
    },
    "clusters": {
        "cluster_id": "INTEGER",
        "file_path": "TEXT",
    },
    "cluster_metadata": {
        "cluster_id": "INTEGER",
        "name": "TEXT",
        "description": "TEXT",
        "avg_bpm": "REAL",
        "avg_energy": "REAL",
        "avg_loudness": "REAL",
    },
    "crates": {
        "crate_name": "TEXT",
        "file_path": "TEXT",
    },
    "feedback": {
        "file_path_a": "TEXT",
        "file_path_b": "TEXT",
        "rating": "INTEGER",
        "created_at": "TEXT",
    },
}


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_columns(
    connection: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, str],
) -> None:
    existing_columns = _table_columns(connection, table_name)
    for column_name, column_type in expected_columns.items():
        if column_name in existing_columns:
            continue
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def init_db(db_path: str | Path) -> None:
    database_path = Path(db_path).expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        for statement in CREATE_TABLE_STATEMENTS:
            connection.execute(statement)

        for table_name, columns in EXPECTED_COLUMNS.items():
            _ensure_columns(connection, table_name, columns)

        connection.commit()
