from __future__ import annotations

"""
Visuele vereisten
- Kleurenpalet: achtergrond #0A0A0B, panelen/cards #111113, borders #1E1E22,
  primaire tekst #F0F0F0, secundaire tekst #6B6B72, accent #3D7EFF.
- Energy-kleurschaal: #1A3A5C via #3D7EFF naar #FF4D6D.
- Fonts: DM Mono voor labels, titels en BPM; Inter voor tabellen en lijsten.
- Sidebar: vaste navigatiekolom van 220px breed links, met icon-chip en label per sectie,
  geen standaard tabs bovenaan.
- EnergyBar: custom QWidget van 60x6 pixels, getekend met QPainter, met energie-afhankelijke kleur.

Pas de bestaande Qt stylesheet en EnergyBar toe op alle nieuwe elementen.
"""

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import (
    QAbstractTableModel,
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QSortFilterProxyModel,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontDatabase,
    QPainter,
    QPalette,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

try:
    from database_init import init_db
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.database_init import init_db

from collection_analyser import (
    find_missing_bridges,
    find_orphans,
    find_overcrowded_clusters,
    get_collection_stats,
    set_database_path as set_collection_database_path,
)
from crate_generator import generate_crates, set_database_path as set_crate_database_path
from feedback_engine import (
    apply_feedback,
    record_feedback,
    set_database_path as set_feedback_database_path,
)
from recommendation_engine import (
    get_next_tracks,
    set_database_path as set_recommendation_database_path,
)
from set_generator import generate_set


BACKGROUND = "#0A0A0B"
SURFACE = "#111113"
BORDER = "#1E1E22"
TEXT_PRIMARY = "#F0F0F0"
TEXT_SECONDARY = "#6B6B72"
ACCENT = "#3D7EFF"
HOVER_ROW = "#1A1A1F"
ENERGY_LOW = "#1A3A5C"
ENERGY_MID = "#3D7EFF"
ENERGY_HIGH = "#FF4D6D"
CONFIG_PATH = Path.home() / ".djlibrary" / "config.json"

CLUSTER_PALETTE = [
    "#3D7EFF",
    "#FF4D6D",
    "#00C2A8",
    "#FFC857",
    "#8A5CFF",
    "#5DD39E",
    "#F08A5D",
    "#54A9FF",
    "#E85D75",
    "#9BE564",
]


def hex_to_color(value: str) -> QColor:
    return QColor(value)


def interpolate_channel(start: int, end: int, factor: float) -> int:
    return int(round(start + (end - start) * factor))


def energy_color(value: float) -> QColor:
    clamped = max(0.0, min(1.0, value))
    low = hex_to_color(ENERGY_LOW)
    mid = hex_to_color(ENERGY_MID)
    high = hex_to_color(ENERGY_HIGH)

    if clamped <= 0.5:
        local = clamped / 0.5
        return QColor(
            interpolate_channel(low.red(), mid.red(), local),
            interpolate_channel(low.green(), mid.green(), local),
            interpolate_channel(low.blue(), mid.blue(), local),
        )

    local = (clamped - 0.5) / 0.5
    return QColor(
        interpolate_channel(mid.red(), high.red(), local),
        interpolate_channel(mid.green(), high.green(), local),
        interpolate_channel(mid.blue(), high.blue(), local),
    )


def format_bpm(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_energy(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def format_key(value: int | None) -> str:
    if value is None:
        return "--"
    return str(value)


def color_for_cluster(cluster_id: int) -> str:
    if cluster_id == -1:
        return "#4A4A52"
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(database_path: Path) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"database_path": str(database_path)}, indent=2),
        encoding="utf-8",
    )


def resolve_database_path() -> Path:
    local_database = Path(__file__).resolve().with_name("database.db")
    if local_database.exists():
        save_config(local_database)
        return local_database

    config = load_config()
    configured = config.get("database_path")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return configured_path

    selected, _ = QFileDialog.getOpenFileName(
        None,
        "Kies een DJ library database",
        str(Path.home()),
        "SQLite Databases (*.db *.sqlite *.sqlite3);;All Files (*)",
    )
    if not selected:
        raise RuntimeError("Geen database geselecteerd.")

    selected_path = Path(selected).expanduser()
    save_config(selected_path)
    return selected_path


def load_preferred_fonts() -> tuple[str, str]:
    font_db = QFontDatabase()
    search_paths = [
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ]
    hints = {
        "DM Mono": ["DMMono-Regular.ttf", "DM Mono Regular.ttf", "DMMono-Medium.ttf"],
        "Inter": ["Inter-Regular.ttf", "Inter.ttc", "InterVariable.ttf"],
    }

    for file_names in hints.values():
        for base_path in search_paths:
            for file_name in file_names:
                candidate = base_path / file_name
                if candidate.exists():
                    font_db.addApplicationFont(str(candidate))

    families = set(font_db.families())
    display_family = "DM Mono" if "DM Mono" in families else "Menlo"
    body_family = "Inter" if "Inter" in families else "Helvetica Neue"
    return display_family, body_family


class EnergyBar(QWidget):
    def __init__(self, energy: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._energy = max(0.0, min(1.0, energy))
        self.setFixedSize(60, 6)

    def set_energy(self, energy: float | None) -> None:
        self._energy = max(0.0, min(1.0, float(energy or 0.0)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        background = QColor(BORDER)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 3, 3)

        color = energy_color(self._energy)
        color.setAlpha(245)
        painter.setBrush(color)
        painter.drawRoundedRect(rect.adjusted(0, 0, 0, 0), 3, 3)


class SidebarButton(QPushButton):
    def __init__(self, icon_text: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.icon_text = icon_text
        self.label_text = label
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(52)
        self.setText("")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(220, 52)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 4, -10, -4)
        active = self.isChecked()
        hovered = self.underMouse()

        background = QColor(SURFACE if active else BACKGROUND)
        if hovered and not active:
            background = QColor(HOVER_ROW)

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 12, 12)

        if active:
            painter.setBrush(QColor(ACCENT))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRect(rect.left(), rect.top(), 3, rect.height()), 1, 1)

        chip_rect = QRect(rect.left() + 14, rect.top() + 10, 28, 28)
        chip_color = QColor(ACCENT if active or hovered else BORDER)
        chip_text = QColor(TEXT_PRIMARY if active or hovered else TEXT_SECONDARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(chip_color)
        painter.drawRoundedRect(chip_rect, 8, 8)

        chip_font = QFont(self.font())
        chip_font.setPointSize(9)
        chip_font.setBold(True)
        painter.setFont(chip_font)
        painter.setPen(chip_text)
        painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, self.icon_text)

        text_rect = QRect(chip_rect.right() + 12, rect.top(), rect.width() - 70, rect.height())
        text_font = QFont(self.font())
        text_font.setPointSize(11)
        text_font.setBold(True)
        painter.setFont(text_font)
        painter.setPen(QColor(TEXT_PRIMARY if active else TEXT_SECONDARY))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.label_text,
        )


class AnimatedPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.content = QFrame()
        self.content.setObjectName("PageContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(24, 24, 24, 24)
        self.content_layout.setSpacing(18)
        outer_layout.addWidget(self.content)

        self.opacity_effect = QGraphicsOpacityEffect(self.content)
        self.content.setGraphicsEffect(self.opacity_effect)
        self._tab_animation: QParallelAnimationGroup | None = None

    def play_enter_animation(self) -> None:
        base_geometry = self.content.geometry()
        start_geometry = QRect(
            base_geometry.x() + 20,
            base_geometry.y(),
            base_geometry.width(),
            base_geometry.height(),
        )

        self.content.setGeometry(start_geometry)
        self.opacity_effect.setOpacity(0.0)

        opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        opacity_animation.setDuration(150)
        opacity_animation.setStartValue(0.0)
        opacity_animation.setEndValue(1.0)
        opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        geometry_animation = QPropertyAnimation(self.content, b"geometry", self)
        geometry_animation.setDuration(150)
        geometry_animation.setStartValue(start_geometry)
        geometry_animation.setEndValue(base_geometry)
        geometry_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._tab_animation = QParallelAnimationGroup(self)
        self._tab_animation.addAnimation(opacity_animation)
        self._tab_animation.addAnimation(geometry_animation)
        self._tab_animation.start()


@dataclass
class TrackRecord:
    file_path: str
    artist: str
    title: str
    bpm: float | None
    key: int | None
    energy: float | None
    cluster_id: int | None
    cluster_name: str


class LibraryRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def fetch_tracks(self) -> list[TrackRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.file_path,
                    COALESCE(t.artist, 'Unknown Artist') AS artist,
                    COALESCE(t.title, 'Unknown Title') AS title,
                    af.bpm,
                    af.key,
                    af.energy,
                    c.cluster_id,
                    COALESCE(cm.name, CASE WHEN c.cluster_id = -1 THEN 'Orphan' END, 'Unclustered') AS cluster_name
                FROM tracks AS t
                LEFT JOIN audio_features AS af
                    ON af.file_path = t.file_path
                LEFT JOIN clusters AS c
                    ON c.file_path = t.file_path
                LEFT JOIN cluster_metadata AS cm
                    ON cm.cluster_id = c.cluster_id
                ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE
                """
            ).fetchall()

        return [
            TrackRecord(
                file_path=row["file_path"],
                artist=row["artist"],
                title=row["title"],
                bpm=float(row["bpm"]) if row["bpm"] is not None else None,
                key=int(row["key"]) if row["key"] is not None else None,
                energy=float(row["energy"]) if row["energy"] is not None else None,
                cluster_id=int(row["cluster_id"]) if row["cluster_id"] is not None else None,
                cluster_name=row["cluster_name"],
            )
            for row in rows
        ]

    def fetch_track_detail(self, file_path: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audio_features)").fetchall()
            }
            vocal_select = ", af.vocal_presence" if "vocal_presence" in columns else ", NULL AS vocal_presence"
            row = connection.execute(
                f"""
                SELECT
                    t.file_path,
                    COALESCE(t.artist, 'Unknown Artist') AS artist,
                    COALESCE(t.title, 'Unknown Title') AS title,
                    af.bpm,
                    af.key,
                    af.energy,
                    af.loudness,
                    COALESCE(cm.name, CASE WHEN c.cluster_id = -1 THEN 'Orphan' END, 'Unclustered') AS cluster_name
                    {vocal_select}
                FROM tracks AS t
                LEFT JOIN audio_features AS af
                    ON af.file_path = t.file_path
                LEFT JOIN clusters AS c
                    ON c.file_path = t.file_path
                LEFT JOIN cluster_metadata AS cm
                    ON cm.cluster_id = c.cluster_id
                WHERE t.file_path = ?
                """,
                (file_path,),
            ).fetchone()
        return dict(row) if row else None

    def fetch_cluster_points(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    uc.file_path,
                    uc.x,
                    uc.y,
                    COALESCE(t.artist, 'Unknown Artist') AS artist,
                    COALESCE(t.title, 'Unknown Title') AS title,
                    af.bpm,
                    COALESCE(c.cluster_id, -1) AS cluster_id
                FROM umap_coordinates AS uc
                JOIN tracks AS t
                    ON t.file_path = uc.file_path
                LEFT JOIN audio_features AS af
                    ON af.file_path = uc.file_path
                LEFT JOIN clusters AS c
                    ON c.file_path = uc.file_path
                ORDER BY t.artist COLLATE NOCASE, t.title COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_crate_names(self) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT crate_name
                FROM crates
                ORDER BY crate_name COLLATE NOCASE
                """
            ).fetchall()
        return [row["crate_name"] for row in rows]

    def fetch_tracks_for_crate(self, crate_name: str) -> list[TrackRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.file_path,
                    COALESCE(t.artist, 'Unknown Artist') AS artist,
                    COALESCE(t.title, 'Unknown Title') AS title,
                    af.bpm,
                    af.key,
                    af.energy,
                    c.cluster_id,
                    COALESCE(cm.name, CASE WHEN c.cluster_id = -1 THEN 'Orphan' END, 'Unclustered') AS cluster_name
                FROM crates AS cr
                JOIN tracks AS t
                    ON t.file_path = cr.file_path
                LEFT JOIN audio_features AS af
                    ON af.file_path = t.file_path
                LEFT JOIN clusters AS c
                    ON c.file_path = t.file_path
                LEFT JOIN cluster_metadata AS cm
                    ON cm.cluster_id = c.cluster_id
                WHERE cr.crate_name = ?
                ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE
                """,
                (crate_name,),
            ).fetchall()
        return [
            TrackRecord(
                file_path=row["file_path"],
                artist=row["artist"],
                title=row["title"],
                bpm=float(row["bpm"]) if row["bpm"] is not None else None,
                key=int(row["key"]) if row["key"] is not None else None,
                energy=float(row["energy"]) if row["energy"] is not None else None,
                cluster_id=int(row["cluster_id"]) if row["cluster_id"] is not None else None,
                cluster_name=row["cluster_name"],
            )
            for row in rows
        ]


class TrackTableModel(QAbstractTableModel):
    FilePathRole = Qt.ItemDataRole.UserRole + 1
    TrackRole = Qt.ItemDataRole.UserRole + 2
    SortValueRole = Qt.ItemDataRole.UserRole + 3

    def __init__(self, tracks: list[TrackRecord] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks = tracks or []

    def set_tracks(self, tracks: list[TrackRecord]) -> None:
        self.beginResetModel()
        self._tracks = tracks
        self.endResetModel()

    def track_at(self, row: int) -> TrackRecord:
        return self._tracks[row]

    def rowCount(self, parent=None) -> int:  # noqa: N802
        return 0 if parent and parent.isValid() else len(self._tracks)

    def columnCount(self, parent=None) -> int:  # noqa: N802
        return 0 if parent and parent.isValid() else 3

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.DisplayRole:
            return None
        return ["Track", "BPM", "Energy"][section]

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None

        track = self._tracks[index.row()]

        if role == self.FilePathRole:
            return track.file_path
        if role == self.TrackRole:
            return track
        if role == self.SortValueRole:
            if index.column() == 0:
                return f"{track.artist} {track.title}"
            if index.column() == 1:
                return track.bpm if track.bpm is not None else -1
            return track.energy if track.energy is not None else -1

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if index.column() == 0:
            return f"{track.artist}\n{track.title}"
        if index.column() == 1:
            return format_bpm(track.bpm)
        if index.column() == 2:
            return format_energy(track.energy)
        return None


class TrackFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.search_text = ""
        self.bpm_range: tuple[int, int] | None = None
        self.key_filter: int | None = None
        self.energy_filter: str | None = None

    def set_search_text(self, value: str) -> None:
        self.search_text = value.strip().lower()
        self.invalidateFilter()

    def set_bpm_range(self, value: tuple[int, int] | None) -> None:
        self.bpm_range = value
        self.invalidateFilter()

    def set_key_filter(self, value: int | None) -> None:
        self.key_filter = value
        self.invalidateFilter()

    def set_energy_filter(self, value: str | None) -> None:
        self.energy_filter = value
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, TrackTableModel):
            return True

        track = model.track_at(source_row)
        searchable = f"{track.artist} {track.title}".lower()
        if self.search_text and self.search_text not in searchable:
            return False

        if self.bpm_range is not None:
            if track.bpm is None:
                return False
            start, end = self.bpm_range
            if not (start <= track.bpm < end):
                return False

        if self.key_filter is not None and track.key != self.key_filter:
            return False

        if self.energy_filter == "low":
            return track.energy is not None and track.energy < 0.4
        if self.energy_filter == "mid":
            return track.energy is not None and 0.4 <= track.energy <= 0.75
        if self.energy_filter == "high":
            return track.energy is not None and track.energy > 0.75

        return True


class TrackTableDelegate(QStyledItemDelegate):
    def __init__(self, display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.display_font = display_font
        self.body_font = body_font

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        model = index.model()
        if isinstance(model, QSortFilterProxyModel):
            source_index = model.mapToSource(index)
            track = model.sourceModel().data(source_index, TrackTableModel.TrackRole)
        else:
            track = model.data(index, TrackTableModel.TrackRole)

        if track is None:
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        background = QColor(SURFACE if is_selected else (HOVER_ROW if is_hovered else BACKGROUND))
        painter.fillRect(rect, background)

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        if is_selected and index.column() == 0:
            painter.fillRect(QRect(rect.left(), rect.top(), 3, rect.height()), QColor(ACCENT))

        inner = rect.adjusted(14, 8, -14, -8)
        if index.column() == 0:
            artist_font = QFont(self.body_font, 11)
            artist_font.setBold(True)
            painter.setFont(artist_font)
            painter.setPen(QColor(TEXT_PRIMARY))
            artist_rect = QRect(inner.left(), inner.top(), inner.width(), 18)
            painter.drawText(artist_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, track.artist)

            title_font = QFont(self.body_font, 10)
            painter.setFont(title_font)
            painter.setPen(QColor(TEXT_SECONDARY))
            title_rect = QRect(inner.left(), inner.top() + 20, inner.width(), 16)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, track.title)

        elif index.column() == 1:
            bpm_font = QFont(self.display_font, 16)
            bpm_font.setBold(True)
            painter.setFont(bpm_font)
            painter.setPen(QColor(TEXT_PRIMARY))
            painter.drawText(inner, Qt.AlignmentFlag.AlignCenter, format_bpm(track.bpm))

        elif index.column() == 2:
            bar_rect = QRect(
                inner.center().x() - 30,
                inner.center().y() - 3,
                60,
                6,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(BORDER))
            painter.drawRoundedRect(bar_rect, 3, 3)
            painter.setBrush(energy_color(track.energy or 0.0))
            painter.drawRoundedRect(bar_rect, 3, 3)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        return QSize(option.rect.width(), 56)


class SuggestionCard(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.display_font = display_font
        self.body_font = body_font
        self.file_path = ""
        self._selected = False
        self.setObjectName("SuggestionCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("selected", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel("Geen suggestie")
        self.title_label.setObjectName("CardTitle")
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("CardMeta")
        self.reason_label = QLabel("")
        self.reason_label.setObjectName("CardReason")
        self.reason_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.meta_label)
        layout.addWidget(self.reason_label)

    def set_suggestion(self, suggestion: dict[str, Any] | None, track: TrackRecord | None) -> None:
        if suggestion is None or track is None:
            self.file_path = ""
            self.set_selected(False)
            self.title_label.setText("Geen suggestie beschikbaar")
            self.meta_label.setText("")
            self.reason_label.setText("")
            self.setEnabled(False)
            return

        self.file_path = track.file_path
        self.title_label.setText(f"{track.artist} — {track.title}")
        self.meta_label.setText(
            f"{suggestion['mix_type'].upper()}   •   {suggestion['similarity']:.2f}   •   {format_bpm(track.bpm)} BPM"
        )
        self.reason_label.setText(str(suggestion["reason"]))
        self.setEnabled(True)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected and self.isEnabled()
        self.setProperty("selected", self._selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.file_path:
            self.selected.emit(self.file_path)
        super().mousePressEvent(event)


class LibraryTab(AnimatedPage):
    track_selected = pyqtSignal(str)

    def __init__(self, tracks: list[TrackRecord], display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = TrackTableModel(tracks, self)
        self.proxy = TrackFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        header = QHBoxLayout()
        header.setSpacing(12)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Zoek artiest of titel")
        self.bpm_filter = QComboBox()
        self.key_filter = QComboBox()
        self.energy_filter = QComboBox()

        self._populate_filters(tracks)

        header.addWidget(self.search, 1)
        header.addWidget(self.bpm_filter)
        header.addWidget(self.key_filter)
        header.addWidget(self.energy_filter)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setItemDelegate(TrackTableDelegate(display_font, body_font, self.table))
        self.table.setMouseTracking(True)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.setSortingEnabled(False)
        self.table.setColumnWidth(0, 520)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(56)

        self.content_layout.addLayout(header)
        self.content_layout.addWidget(self.table, 1)

        self.search.textChanged.connect(self.proxy.set_search_text)
        self.bpm_filter.currentIndexChanged.connect(self._apply_bpm_filter)
        self.key_filter.currentIndexChanged.connect(self._apply_key_filter)
        self.energy_filter.currentIndexChanged.connect(self._apply_energy_filter)
        self.table.clicked.connect(self._handle_click)

    def _populate_filters(self, tracks: list[TrackRecord]) -> None:
        self.bpm_filter.addItem("Alle BPM", None)
        bpm_values = [track.bpm for track in tracks if track.bpm is not None]
        if bpm_values:
            start = int(min(bpm_values) // 10 * 10)
            stop = int(max(bpm_values) // 10 * 10 + 10)
            for value in range(start, stop + 1, 10):
                self.bpm_filter.addItem(f"{value}-{value + 9}", (value, value + 10))

        self.key_filter.addItem("Alle keys", None)
        for key_value in range(12):
            self.key_filter.addItem(str(key_value), key_value)

        self.energy_filter.addItem("Alle energy", None)
        self.energy_filter.addItem("Laag", "low")
        self.energy_filter.addItem("Midden", "mid")
        self.energy_filter.addItem("Hoog", "high")

    def _apply_bpm_filter(self) -> None:
        self.proxy.set_bpm_range(self.bpm_filter.currentData())

    def _apply_key_filter(self) -> None:
        self.proxy.set_key_filter(self.key_filter.currentData())

    def _apply_energy_filter(self) -> None:
        self.proxy.set_energy_filter(self.energy_filter.currentData())

    def _handle_click(self, index) -> None:
        source_index = self.proxy.mapToSource(index)
        file_path = self.model.data(source_index, TrackTableModel.FilePathRole)
        if file_path:
            self.track_selected.emit(str(file_path))


class ClusterMapTab(AnimatedPage):
    track_selected = pyqtSignal(str)

    def __init__(self, points: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points = points
        self.selected_file_path: str | None = None

        self.plot = pg.PlotWidget()
        self.plot.setBackground(BACKGROUND)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.showGrid(x=False, y=False)
        self.plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=True)

        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        self.content_layout.addWidget(self.plot, 1)

        self.scatter.sigClicked.connect(self._handle_clicked)
        if hasattr(self.scatter, "sigHovered"):
            self.scatter.sigHovered.connect(self._handle_hovered)

        self.refresh_points()

    def refresh_points(self, selected_file_path: str | None = None) -> None:
        if selected_file_path is not None:
            self.selected_file_path = selected_file_path

        spots = []
        for point in self.points:
            is_selected = point["file_path"] == self.selected_file_path
            color = QColor("#FFFFFF" if is_selected else color_for_cluster(int(point["cluster_id"])))
            size = 10 if is_selected else 6
            spots.append(
                {
                    "pos": (float(point["x"]), float(point["y"])),
                    "size": size,
                    "brush": color,
                    "pen": pg.mkPen(color if is_selected else background_pen(point["cluster_id"])),
                    "data": point,
                }
            )

        self.scatter.setData(spots)

    def _handle_clicked(self, scatter, points) -> None:
        if not points:
            return
        point = points[0].data()
        if point:
            self.track_selected.emit(str(point["file_path"]))

    def _handle_hovered(self, scatter, points, event=None) -> None:
        if not points:
            QToolTip.hideText()
            return
        point = points[0].data()
        if not point:
            return
        tooltip = (
            f"<div style='color:{TEXT_PRIMARY};'>"
            f"<b>{point['artist']}</b><br>{point['title']}<br>"
            f"<span style='color:{TEXT_SECONDARY};'>{format_bpm(point['bpm'])} BPM</span>"
            f"</div>"
        )
        QToolTip.showText(QCursor.pos(), tooltip, self.plot)


def background_pen(cluster_id: int) -> QColor:
    if cluster_id == -1:
        return QColor("#4A4A52")
    return QColor(color_for_cluster(int(cluster_id)))


class TrackDetailTab(AnimatedPage):
    add_to_set_requested = pyqtSignal(str)
    track_selected = pyqtSignal(str)

    def __init__(self, repository: LibraryRepository, track_lookup: dict[str, TrackRecord], display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.track_lookup = track_lookup
        self.current_file_path: str | None = None
        self.current_suggestions: list[dict[str, Any]] = []
        self.active_pair_file_path: str | None = None

        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        self.artist_label = QLabel("Selecteer een track")
        self.artist_label.setObjectName("DetailArtist")
        self.title_label = QLabel("")
        self.title_label.setObjectName("DetailTitle")
        title_block.addWidget(self.artist_label)
        title_block.addWidget(self.title_label)

        metrics = QHBoxLayout()
        metrics.setSpacing(18)
        self.bpm_card = self._metric_card("BPM", "--", display_font)
        self.key_card = self._metric_card("Key", "--", display_font)
        self.energy_card = self._energy_metric_card(display_font)
        self.cluster_card = self._metric_card("Cluster", "--", body_font)

        metrics.addWidget(self.bpm_card)
        metrics.addWidget(self.key_card)
        metrics.addWidget(self.energy_card)
        metrics.addWidget(self.cluster_card)

        rec_title = QLabel("Beste opvolgers")
        rec_title.setObjectName("SectionTitle")
        self.suggestion_cards = [SuggestionCard(display_font, body_font) for _ in range(3)]
        for card in self.suggestion_cards:
            card.selected.connect(self._set_active_pair)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.good_mix_button = QPushButton("Goede mix")
        self.good_mix_button.setProperty("feedback", "positive")
        self.good_mix_button.clicked.connect(lambda: self._handle_feedback(1))
        self.bad_mix_button = QPushButton("Slechte mix")
        self.bad_mix_button.setProperty("feedback", "negative")
        self.bad_mix_button.clicked.connect(lambda: self._handle_feedback(-1))
        self.good_mix_button.setEnabled(False)
        self.bad_mix_button.setEnabled(False)
        self.add_button = QPushButton("Voeg toe aan set")
        self.add_button.clicked.connect(self._handle_add_to_set)
        actions.addWidget(self.good_mix_button)
        actions.addWidget(self.bad_mix_button)
        actions.addStretch(1)
        actions.addWidget(self.add_button)

        self.content_layout.addLayout(title_block)
        self.content_layout.addLayout(metrics)
        self.content_layout.addWidget(rec_title)
        for card in self.suggestion_cards:
            self.content_layout.addWidget(card)
        self.content_layout.addStretch(1)
        self.content_layout.addLayout(actions)

    def _metric_card(self, label: str, value: str, font_family: str) -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel(label)
        title.setObjectName("MetricLabel")
        number = QLabel(value)
        number.setObjectName("MetricValue")
        number.setProperty("displayFamily", font_family)
        layout.addWidget(title)
        layout.addWidget(number)
        card.value_label = number  # type: ignore[attr-defined]
        return card

    def _energy_metric_card(self, display_font: str) -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel("Energy")
        title.setObjectName("MetricLabel")
        value = QLabel("--")
        value.setObjectName("MetricValue")
        value.setProperty("displayFamily", display_font)
        bar = EnergyBar()
        layout.addWidget(title)
        layout.addWidget(value)
        layout.addWidget(bar, 0, Qt.AlignmentFlag.AlignLeft)
        card.value_label = value  # type: ignore[attr-defined]
        card.energy_bar = bar  # type: ignore[attr-defined]
        return card

    def set_track(self, file_path: str) -> None:
        detail = self.repository.fetch_track_detail(file_path)
        if detail is None:
            return

        self.current_file_path = file_path
        self.artist_label.setText(str(detail["artist"]))
        self.title_label.setText(str(detail["title"]))
        self.bpm_card.value_label.setText(format_bpm(detail["bpm"]))  # type: ignore[attr-defined]
        self.key_card.value_label.setText(format_key(detail["key"]))  # type: ignore[attr-defined]
        self.energy_card.value_label.setText(format_energy(detail["energy"]))  # type: ignore[attr-defined]
        self.energy_card.energy_bar.set_energy(float(detail["energy"] or 0.0))  # type: ignore[attr-defined]
        self.cluster_card.value_label.setText(str(detail["cluster_name"]))  # type: ignore[attr-defined]

        suggestions = get_next_tracks(file_path, [], n=3)
        self.current_suggestions = suggestions
        self.active_pair_file_path = None
        for index, card in enumerate(self.suggestion_cards):
            suggestion = suggestions[index] if index < len(suggestions) else None
            suggestion_track = self.track_lookup.get(str(suggestion["file_path"])) if suggestion else None
            card.set_suggestion(suggestion, suggestion_track)
            card.set_selected(False)

        if suggestions:
            self._set_active_pair(str(suggestions[0]["file_path"]))
        else:
            self._update_feedback_buttons()

    def _handle_add_to_set(self) -> None:
        if self.current_file_path:
            self.add_to_set_requested.emit(self.current_file_path)

    def _set_active_pair(self, file_path: str) -> None:
        self.active_pair_file_path = file_path
        for card in self.suggestion_cards:
            card.set_selected(card.file_path == file_path)
        self._update_feedback_buttons()

    def _update_feedback_buttons(self) -> None:
        enabled = bool(self.current_file_path and self.active_pair_file_path)
        self.good_mix_button.setEnabled(enabled)
        self.bad_mix_button.setEnabled(enabled)

    def _handle_feedback(self, rating: int) -> None:
        if not self.current_file_path or not self.active_pair_file_path:
            return
        record_feedback(self.current_file_path, self.active_pair_file_path, rating)
        apply_feedback()
        self.set_track(self.current_file_path)


class SetBuilderTab(AnimatedPage):
    track_selected = pyqtSignal(str)

    def __init__(self, track_lookup: dict[str, TrackRecord], display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.track_lookup = track_lookup
        self.set_tracks: list[str] = []

        title = QLabel("Setbuilder")
        title.setObjectName("SectionTitle")
        generator_row = QHBoxLayout()
        generator_row.setSpacing(12)
        self.trajectory_combo = QComboBox()
        self.trajectory_combo.addItem("club")
        self.trajectory_combo.addItem("afterhours")
        self.trajectory_combo.addItem("warmup")
        self.length_spinbox = QSpinBox()
        self.length_spinbox.setRange(10, 50)
        self.length_spinbox.setValue(20)
        self.length_spinbox.setPrefix("Lengte ")
        self.generate_button = QPushButton("Genereer set")
        self.generate_button.clicked.connect(self.generate_set_from_trajectory)
        generator_row.addWidget(self.trajectory_combo)
        generator_row.addWidget(self.length_spinbox)
        generator_row.addWidget(self.generate_button)
        generator_row.addStretch(1)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._handle_item_clicked)

        suggestion_title = QLabel("Volgende suggestie")
        suggestion_title.setObjectName("SectionTitle")
        self.suggestion_card = SuggestionCard(display_font, body_font)
        self.suggestion_card.selected.connect(self.track_selected.emit)

        self.remove_button = QPushButton("Verwijder laatste")
        self.remove_button.clicked.connect(self.remove_last_track)

        self.content_layout.addWidget(title)
        self.content_layout.addLayout(generator_row)
        self.content_layout.addWidget(self.list_widget, 1)
        self.content_layout.addWidget(suggestion_title)
        self.content_layout.addWidget(self.suggestion_card)
        self.content_layout.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignLeft)

    def add_track(self, file_path: str) -> None:
        track = self.track_lookup.get(file_path)
        if track is None:
            return
        self.set_tracks.append(file_path)
        item = QListWidgetItem(f"{track.artist} — {track.title}")
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.list_widget.addItem(item)
        self.refresh_suggestion()

    def remove_last_track(self) -> None:
        if not self.set_tracks:
            return
        self.set_tracks.pop()
        self.list_widget.takeItem(self.list_widget.count() - 1)
        self.refresh_suggestion()

    def refresh_suggestion(self) -> None:
        if not self.set_tracks:
            self.suggestion_card.set_suggestion(None, None)
            return

        current = self.set_tracks[-1]
        suggestion = next(iter(get_next_tracks(current, self.set_tracks, n=1)), None)
        suggestion_track = self.track_lookup.get(str(suggestion["file_path"])) if suggestion else None
        self.suggestion_card.set_suggestion(suggestion, suggestion_track)

    def _handle_item_clicked(self, item: QListWidgetItem) -> None:
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.track_selected.emit(str(file_path))

    def generate_set_from_trajectory(self) -> None:
        trajectory = self.trajectory_combo.currentText()
        length = int(self.length_spinbox.value())
        generated_tracks = generate_set(trajectory, length)

        self.set_tracks = []
        self.list_widget.clear()

        for track in generated_tracks:
            self.set_tracks.append(str(track["file_path"]))
            item = QListWidgetItem(
                f"{str(track['phase']).title()}  •  {track['artist']} — {track['title']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(track["file_path"]))
            self.list_widget.addItem(item)

        self.refresh_suggestion()


class CratesTab(AnimatedPage):
    track_selected = pyqtSignal(str)

    def __init__(self, repository: LibraryRepository, crate_names: list[str], display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(220)
        self.list_widget.itemClicked.connect(self._load_crate)

        self.table_model = TrackTableModel([], self)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setItemDelegate(TrackTableDelegate(display_font, body_font, self.table_view))
        self.table_view.setMouseTracking(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.verticalHeader().hide()
        self.table_view.horizontalHeader().hide()
        self.table_view.setShowGrid(False)
        self.table_view.setColumnWidth(0, 420)
        self.table_view.setColumnWidth(1, 110)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.clicked.connect(self._handle_click)

        split = QHBoxLayout()
        split.setSpacing(18)
        split.addWidget(self.list_widget)
        split.addWidget(self.table_view, 1)
        self.content_layout.addLayout(split, 1)

        for name in crate_names:
            self.list_widget.addItem(name)

        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
            self._load_crate(self.list_widget.item(0))

    def _load_crate(self, item: QListWidgetItem) -> None:
        crate_name = item.text()
        self.table_model.set_tracks(self.repository.fetch_tracks_for_crate(crate_name))

    def _handle_click(self, index) -> None:
        file_path = self.table_model.data(index, TrackTableModel.FilePathRole)
        if file_path:
            self.track_selected.emit(str(file_path))


class CollectionAnalysisTab(AnimatedPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        header = QHBoxLayout()
        header.setSpacing(12)
        title = QLabel("Collectie-analyse")
        title.setObjectName("SectionTitle")
        self.analyze_button = QPushButton("Analyseer")
        self.analyze_button.clicked.connect(self.run_analysis)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.analyze_button)

        self.stats_label = QLabel("Klik op Analyseer om inzicht te laden.")
        self.stats_label.setObjectName("AnalysisSummary")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextFormat(Qt.TextFormat.PlainText)

        self.orphans_table = self._create_table(["Artiest", "Titel", "Gem. similarity"])
        self.overcrowded_table = self._create_table(["Cluster", "Naam", "Tracks"])
        self.bridges_table = self._create_table(["Cluster A", "Naam A", "Cluster B", "Naam B"])

        self.content_layout.addLayout(header)
        self.content_layout.addWidget(self.stats_label)
        self.content_layout.addWidget(self._section_label("Orphans"))
        self.content_layout.addWidget(self.orphans_table)
        self.content_layout.addWidget(self._section_label("Overcrowded clusters"))
        self.content_layout.addWidget(self.overcrowded_table)
        self.content_layout.addWidget(self._section_label("Ontbrekende bruggen"))
        self.content_layout.addWidget(self.bridges_table)

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _create_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setShowGrid(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        table.verticalHeader().setDefaultSectionSize(42)
        return table

    def _set_table_data(self, table: QTableWidget, rows: list[list[str]]) -> None:
        table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, column_index, item)

    def run_analysis(self) -> None:
        generate_crates()
        orphans = find_orphans()
        overcrowded = find_overcrowded_clusters()
        missing_bridges = find_missing_bridges()
        stats = get_collection_stats()

        crate_lines = [
            f"{crate_name}: {percentage:.1f}%"
            for crate_name, percentage in stats["crate_percentages"].items()
        ]
        bpm_lines = [
            f"{bucket}: {count}"
            for bucket, count in stats["bpm_histogram"].items()
        ]
        summary_lines = [
            f"Totaal tracks: {stats['total_tracks']}",
            f"Aantal clusters: {stats['cluster_count']}",
            f"Aantal orphans: {stats['orphan_count']}",
            "",
            "Percentage tracks per crate:",
            *(crate_lines if crate_lines else ["Geen crate-data beschikbaar."]),
            "",
            "BPM-verdeling:",
            *(bpm_lines if bpm_lines else ["Geen BPM-data beschikbaar."]),
        ]
        self.stats_label.setText("\n".join(summary_lines))

        self._set_table_data(
            self.orphans_table,
            [
                [
                    orphan["artist"],
                    orphan["title"],
                    f"{orphan['avg_similarity']:.2f}",
                ]
                for orphan in orphans
            ],
        )
        self._set_table_data(
            self.overcrowded_table,
            [
                [
                    str(cluster["cluster_id"]),
                    cluster["name"],
                    str(cluster["track_count"]),
                ]
                for cluster in overcrowded
            ],
        )
        self._set_table_data(
            self.bridges_table,
            [
                [
                    str(pair["cluster_a_id"]),
                    pair["cluster_a_name"],
                    str(pair["cluster_b_id"]),
                    pair["cluster_b_name"],
                ]
                for pair in missing_bridges
            ],
        )


class MainWindow(QMainWindow):
    def __init__(self, database_path: Path, display_font: str, body_font: str) -> None:
        super().__init__()
        self.database_path = database_path
        self.display_font = display_font
        self.body_font = body_font

        set_recommendation_database_path(database_path)
        set_crate_database_path(database_path)
        set_feedback_database_path(database_path)
        set_collection_database_path(database_path)
        generate_crates()

        self.repository = LibraryRepository(database_path)
        self.tracks = self.repository.fetch_tracks()
        self.track_lookup = {track.file_path: track for track in self.tracks}
        self.cluster_points = self.repository.fetch_cluster_points()
        self.crate_names = self.repository.fetch_crate_names()

        self.setWindowTitle("DJ Library Intelligence")
        self.resize(1480, 920)
        self.setMinimumSize(1280, 820)

        self._fade_animation: QPropertyAnimation | None = None
        self.sidebar_buttons: list[SidebarButton] = []

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(18, 22, 18, 22)
        sidebar_layout.setSpacing(10)

        brand = QLabel("DJ LIB")
        brand.setObjectName("BrandLabel")
        sidebar_layout.addWidget(brand)

        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        self.tabs.setDocumentMode(True)

        self.library_tab = LibraryTab(self.tracks, display_font, body_font)
        self.cluster_tab = ClusterMapTab(self.cluster_points)
        self.detail_tab = TrackDetailTab(self.repository, self.track_lookup, display_font, body_font)
        self.setbuilder_tab = SetBuilderTab(self.track_lookup, display_font, body_font)
        self.crates_tab = CratesTab(self.repository, self.crate_names, display_font, body_font)
        self.collection_analysis_tab = CollectionAnalysisTab()

        for page, label in [
            (self.library_tab, "Bibliotheek"),
            (self.cluster_tab, "Clustermap"),
            (self.detail_tab, "Trackdetail"),
            (self.setbuilder_tab, "Setbuilder"),
            (self.crates_tab, "Crates"),
            (self.collection_analysis_tab, "Collectie-analyse"),
        ]:
            self.tabs.addTab(page, label)

        for icon_text, label in [
            ("LB", "Bibliotheek"),
            ("CM", "Clustermap"),
            ("TD", "Trackdetail"),
            ("SB", "Setbuilder"),
            ("CR", "Crates"),
            ("CA", "Collectie-analyse"),
        ]:
            button = SidebarButton(icon_text, label)
            button.clicked.connect(self._make_tab_switcher(len(self.sidebar_buttons)))
            self.sidebar_buttons.append(button)
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)

        content_frame = QFrame()
        content_frame.setObjectName("ContentArea")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.tabs)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(content_frame, 1)

        self.library_tab.track_selected.connect(self.open_track_detail)
        self.cluster_tab.track_selected.connect(self.open_track_detail)
        self.detail_tab.track_selected.connect(self.open_track_detail)
        self.detail_tab.add_to_set_requested.connect(self.setbuilder_tab.add_track)
        self.setbuilder_tab.track_selected.connect(self.open_track_detail)
        self.crates_tab.track_selected.connect(self.open_track_detail)

        self.tabs.currentChanged.connect(self._animate_current_tab)
        self._apply_window_styling()
        self.switch_to_tab(0)

        if self.tracks:
            self.open_track_detail(self.tracks[0].file_path, switch_tab=False)

    def _make_tab_switcher(self, index: int):
        def switcher() -> None:
            self.switch_to_tab(index)

        return switcher

    def _apply_window_styling(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget {{
                background: {BACKGROUND};
                color: {TEXT_PRIMARY};
                font-family: "{self.body_font}";
                font-size: 11pt;
            }}
            QMainWindow {{
                background: {BACKGROUND};
            }}
            QFrame#Sidebar {{
                background: {BACKGROUND};
                border-right: 1px solid {BORDER};
            }}
            QLabel#BrandLabel {{
                font-family: "{self.display_font}";
                font-size: 18pt;
                color: {TEXT_PRIMARY};
                padding: 8px 4px 20px 4px;
            }}
            QFrame#ContentArea, QFrame#PageContent {{
                background: {BACKGROUND};
            }}
            QLineEdit, QComboBox, QSpinBox, QListWidget, QTableView {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
                color: {TEXT_PRIMARY};
                padding: 10px 12px;
                selection-background-color: {SURFACE};
                selection-color: {TEXT_PRIMARY};
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QListWidget:focus, QTableView:focus {{
                border: 1px solid {ACCENT};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                border: none;
                width: 18px;
                background: transparent;
            }}
            QTableWidget {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
                color: {TEXT_PRIMARY};
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 8px 10px;
                border: none;
            }}
            QTableWidget::item:hover {{
                background: {HOVER_ROW};
            }}
            QTableWidget::item:selected {{
                background: {SURFACE};
                border-left: 3px solid {ACCENT};
                color: {TEXT_PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            QComboBox QAbstractItemView {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                selection-background-color: {HOVER_ROW};
                color: {TEXT_PRIMARY};
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-radius: 8px;
            }}
            QListWidget::item:hover {{
                background: {HOVER_ROW};
            }}
            QListWidget::item:selected {{
                background: {SURFACE};
                border-left: 3px solid {ACCENT};
                color: {TEXT_PRIMARY};
            }}
            QHeaderView::section {{
                background: {SURFACE};
                border: none;
                color: {TEXT_SECONDARY};
                padding: 10px 12px;
            }}
            QTableView {{
                gridline-color: transparent;
                outline: none;
                alternate-background-color: {SURFACE};
                font-family: "{self.body_font}";
            }}
            QTableView::item {{
                padding: 0;
                border: none;
            }}
            QTableView::item:hover {{
                background: {HOVER_ROW};
            }}
            QTableView::item:selected {{
                background: {SURFACE};
                border-left: 3px solid {ACCENT};
            }}
            QPushButton {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 10px 16px;
                color: {TEXT_PRIMARY};
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {ACCENT};
                background: {HOVER_ROW};
            }}
            QPushButton:pressed {{
                background: {ACCENT};
            }}
            QPushButton[feedback="positive"] {{
                border-color: {ACCENT};
            }}
            QPushButton[feedback="positive"]:hover {{
                background: #13203A;
            }}
            QPushButton[feedback="negative"] {{
                border-color: #7A3140;
                color: #FFB7C2;
            }}
            QPushButton[feedback="negative"]:hover {{
                border-color: #FF4D6D;
                background: #261118;
            }}
            QLabel#SectionTitle {{
                font-family: "{self.display_font}";
                font-size: 16pt;
                color: {TEXT_PRIMARY};
            }}
            QLabel#DetailArtist {{
                color: {TEXT_SECONDARY};
                font-size: 11pt;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QLabel#DetailTitle {{
                font-family: "{self.display_font}";
                font-size: 28pt;
                color: {TEXT_PRIMARY};
            }}
            QFrame#MetricCard, QFrame#SuggestionCard {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 16px;
            }}
            QFrame#SuggestionCard[selected="true"] {{
                border: 1px solid {ACCENT};
                background: #141823;
            }}
            QLabel#MetricLabel {{
                color: {TEXT_SECONDARY};
                font-size: 10pt;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QLabel#MetricValue {{
                font-family: "{self.display_font}";
                font-size: 24pt;
                color: {TEXT_PRIMARY};
            }}
            QLabel#CardTitle {{
                font-size: 12pt;
                font-weight: 600;
                color: {TEXT_PRIMARY};
            }}
            QLabel#CardMeta {{
                color: {ACCENT};
                font-family: "{self.display_font}";
                font-size: 10pt;
            }}
            QLabel#CardReason {{
                color: {TEXT_SECONDARY};
                font-size: 10pt;
            }}
            QLabel#AnalysisSummary {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 16px;
                padding: 16px 18px;
                color: {TEXT_PRIMARY};
                line-height: 1.4;
            }}
            QToolTip {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                color: {TEXT_PRIMARY};
                padding: 8px 10px;
            }}
            """
        )

    def switch_to_tab(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)
        for button_index, button in enumerate(self.sidebar_buttons):
            button.setChecked(button_index == index)

    def _animate_current_tab(self, index: int) -> None:
        page = self.tabs.widget(index)
        if isinstance(page, AnimatedPage):
            page.play_enter_animation()

    def open_track_detail(self, file_path: str, switch_tab: bool = True) -> None:
        self.detail_tab.set_track(file_path)
        self.cluster_tab.refresh_points(file_path)
        if switch_tab:
            self.switch_to_tab(2)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._fade_animation is not None:
            return

        effect = QGraphicsOpacityEffect(self.centralWidget())
        self.centralWidget().setGraphicsEffect(effect)
        effect.setOpacity(0.0)

        self._fade_animation = QPropertyAnimation(effect, b"opacity", self)
        self._fade_animation.setDuration(300)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_animation.start()


def build_application() -> QApplication:
    app = QApplication(sys.argv)
    display_font, body_font = load_preferred_fonts()
    app.setApplicationName("DJ Library Intelligence")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_PRIMARY))
    app.setPalette(palette)
    app.setFont(QFont(body_font, 11))
    pg.setConfigOptions(antialias=True)
    app.display_font = display_font  # type: ignore[attr-defined]
    app.body_font = body_font  # type: ignore[attr-defined]
    return app


def main() -> None:
    app = build_application()
    display_font = getattr(app, "display_font")
    body_font = getattr(app, "body_font")

    try:
        database_path = resolve_database_path()
        init_db(database_path)
        window = MainWindow(database_path, display_font, body_font)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Database ontbreekt", str(exc))
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
