from __future__ import annotations

import json
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dj_library.sqlite3"
CONFIG_PATH = Path.home() / ".djlibrary" / "config.json"


def resolve_database_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser()

    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            configured = data.get("db_path") or data.get("database_path")
            if configured:
                configured_path = Path(str(configured)).expanduser()
                if configured_path.exists():
                    return configured_path
        except (json.JSONDecodeError, OSError):
            pass

    return DEFAULT_DB_PATH
