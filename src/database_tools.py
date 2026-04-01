from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db

try:
    from database_path import resolve_database_path
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_path import resolve_database_path


def escape_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def get_user_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def get_user_views(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def connect_database(db_path: str | Path | None = None) -> tuple[Path, sqlite3.Connection]:
    resolved_path = resolve_database_path(db_path)
    connection = sqlite3.connect(resolved_path)
    connection.row_factory = sqlite3.Row
    return resolved_path, connection


def inspect_database(db_path: str | Path | None = None, show_schema: bool = False) -> None:
    resolved_path, connection = connect_database(db_path)
    with connection:
        file_size = resolved_path.stat().st_size if resolved_path.exists() else 0
        tables = get_user_tables(connection)

        print(f"Database: {resolved_path}")
        print(f"Grootte: {file_size} bytes")
        print(f"Tabellen: {len(tables)}")

        for table_name in tables:
            count_row = connection.execute(
                f"SELECT COUNT(*) AS row_count FROM {escape_identifier(table_name)}"
            ).fetchone()
            row_count = int(count_row["row_count"]) if count_row else 0
            print(f"- {table_name}: {row_count} rijen")

            if show_schema:
                columns = connection.execute(
                    f"PRAGMA table_info({escape_identifier(table_name)})"
                ).fetchall()
                for column in columns:
                    column_name = str(column["name"])
                    column_type = str(column["type"] or "TEXT")
                    nullable = "NOT NULL" if int(column["notnull"]) else "NULL"
                    print(f"  - {column_name}: {column_type} {nullable}")


def export_database(db_path: str | Path | None, output_path: str | Path) -> Path:
    resolved_path, source_connection = connect_database(db_path)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix.lower() == ".sql":
        with source_connection:
            with destination.open("w", encoding="utf-8") as handle:
                for statement in source_connection.iterdump():
                    handle.write(f"{statement}\n")
    else:
        with source_connection:
            with sqlite3.connect(destination) as target_connection:
                source_connection.backup(target_connection)

    print(f"Database geëxporteerd van {resolved_path} naar {destination}")
    return destination


def reset_database(
    db_path: str | Path | None = None,
    force: bool = False,
    backup_path: str | Path | None = None,
) -> Path:
    if not force:
        raise RuntimeError("Reset is destructief. Gebruik --force om door te gaan.")

    resolved_path, connection = connect_database(db_path)
    connection.close()

    if backup_path is not None:
        export_database(resolved_path, backup_path)

    with sqlite3.connect(resolved_path) as reset_connection:
        reset_connection.row_factory = sqlite3.Row
        reset_connection.execute("PRAGMA foreign_keys = OFF")

        for view_name in get_user_views(reset_connection):
            reset_connection.execute(f"DROP VIEW IF EXISTS {escape_identifier(view_name)}")

        for table_name in get_user_tables(reset_connection):
            reset_connection.execute(f"DROP TABLE IF EXISTS {escape_identifier(table_name)}")

        reset_connection.commit()

    init_db(resolved_path)
    print(f"Database gereset en schema opnieuw aangemaakt: {resolved_path}")
    return resolved_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline tools voor inspectie, export en reset van de DJ-library SQLite database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Toon database-info en aantallen per tabel.")
    inspect_parser.add_argument("--db-path", help="Pad naar de SQLite database.")
    inspect_parser.add_argument(
        "--schema",
        action="store_true",
        help="Toon ook kolommen per tabel.",
    )

    export_parser = subparsers.add_parser("export", help="Exporteer de database naar .sqlite3 of .sql.")
    export_parser.add_argument("output", help="Doelbestand voor export.")
    export_parser.add_argument("--db-path", help="Pad naar de SQLite database.")

    reset_parser = subparsers.add_parser("reset", help="Reset de database en maak het schema opnieuw aan.")
    reset_parser.add_argument("--db-path", help="Pad naar de SQLite database.")
    reset_parser.add_argument("--backup-path", help="Maak eerst een backup/export naar dit pad.")
    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Bevestig dat de reset echt uitgevoerd mag worden.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_database(db_path=args.db_path, show_schema=bool(args.schema))
        return

    if args.command == "export":
        export_database(db_path=args.db_path, output_path=args.output)
        return

    if args.command == "reset":
        reset_database(
            db_path=args.db_path,
            force=bool(args.force),
            backup_path=args.backup_path,
        )
        return

    raise RuntimeError(f"Onbekend commando: {args.command}")


if __name__ == "__main__":
    main()
