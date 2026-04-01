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
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
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
    QStandardItem,
    QStandardItemModel,
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
    QStackedWidget,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
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
)
from crate_generator import generate_crates
from feedback_engine import (
    apply_feedback,
    record_feedback,
)
from recommendation_engine import get_next_tracks
from set_generator import generate_set

try:
    from domain_constants import BPM_HISTOGRAM_BIN_SIZE, ENERGY_LOW_MAX, ENERGY_MID_MAX
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.domain_constants import BPM_HISTOGRAM_BIN_SIZE, ENERGY_LOW_MAX, ENERGY_MID_MAX


BACKGROUND = "#0A0A0B"
SURFACE = "#111113"
BORDER = "#1E1E22"
TEXT_PRIMARY = "#F0F0F0"
TEXT_SECONDARY = "#6B6B72"
ACCENT = "#3D7EFF"
HOVER_ROW = "#1A1A1F"
ENERGY_LOW = "#1A3A5C"
ENERGY_HIGH = "#FF4D6D"
CONFIG_PATH = Path.home() / ".djlibrary" / "config.json"

TRACK_ROLE = Qt.ItemDataRole.UserRole + 1
FILE_PATH_ROLE = Qt.ItemDataRole.UserRole + 2
BPM_ROLE = Qt.ItemDataRole.UserRole + 3
KEY_ROLE = Qt.ItemDataRole.UserRole + 4
ENERGY_ROLE = Qt.ItemDataRole.UserRole + 5

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


def interpolate_channel(start: int, end: int, factor: float) -> int:
    return int(round(start + (end - start) * factor))


def energy_color(value: float) -> QColor:
    clamped = max(0.0, min(1.0, float(value)))
    low = QColor(ENERGY_LOW)
    high = QColor(ENERGY_HIGH)
    return QColor(
        interpolate_channel(low.red(), high.red(), clamped),
        interpolate_channel(low.green(), high.green(), clamped),
        interpolate_channel(low.blue(), high.blue(), clamped),
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
    return f"{float(value):.2f}"


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
    os.makedirs(CONFIG_PATH.parent, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"db_path": str(database_path)}, indent=2),
        encoding="utf-8",
    )


def resolve_database_path() -> Path:
    config = load_config()
    configured = config.get("db_path") or config.get("database_path")
    if configured:
        configured_path = Path(str(configured)).expanduser()
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


def build_stylesheet(display_font: str, body_font: str) -> str:
    return f"""
    QMainWindow, QWidget {{
        background: {BACKGROUND};
    }}
    QLabel {{
        color: {TEXT_PRIMARY};
        font-family: "{display_font}";
    }}
    QLineEdit, QComboBox, QSpinBox, QListWidget, QTableWidget {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
        font-family: "{body_font}";
        border-radius: 10px;
        padding: 8px 12px;
    }}
    QTableView {{
        background: {SURFACE};
        gridline-color: {BORDER};
        color: {TEXT_PRIMARY};
        font-family: "{body_font}";
        border: 1px solid {BORDER};
        border-radius: 12px;
        outline: none;
    }}
    QTableView::item:hover {{
        background: {HOVER_ROW};
    }}
    QTableView::item:selected {{
        background: transparent;
        border-left: 3px solid {ACCENT};
    }}
    QTableWidget {{
        gridline-color: {BORDER};
    }}
    QTableWidget::item {{
        color: {TEXT_PRIMARY};
        font-family: "{body_font}";
    }}
    QTableWidget::item:hover {{
        background: {HOVER_ROW};
    }}
    QTableWidget::item:selected {{
        background: transparent;
        border-left: 3px solid {ACCENT};
    }}
    QPushButton {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
        padding: 8px 16px;
        border-radius: 10px;
        font-family: "{body_font}";
    }}
    QPushButton:hover {{
        background: {HOVER_ROW};
    }}
    QPushButton[navButton="true"] {{
        text-align: left;
        font-family: "{display_font}";
        padding: 10px 16px;
        min-height: 40px;
    }}
    QPushButton[navButton="true"]:checked {{
        background: {HOVER_ROW};
        border-color: {ACCENT};
    }}
    QListWidget {{
        font-family: "{body_font}";
    }}
    QListWidget::item {{
        padding: 10px 12px;
    }}
    QListWidget::item:hover {{
        background: {HOVER_ROW};
    }}
    QListWidget::item:selected {{
        background: transparent;
        border-left: 3px solid {ACCENT};
    }}
    QHeaderView::section {{
        background: {SURFACE};
        color: {TEXT_SECONDARY};
        border: none;
        padding: 8px 10px;
        font-family: "{body_font}";
    }}
    QToolTip {{
        background: {SURFACE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        padding: 8px 10px;
    }}
    """


class EnergyBar(QWidget):
    def __init__(self, value: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, float(value)))
        self.setFixedSize(60, 6)

    def set_value(self, value: float | None) -> None:
        self._value = max(0.0, min(1.0, float(value or 0.0)))
        self.update()

    def set_energy(self, value: float | None) -> None:
        self.set_value(value)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(60, 6)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(energy_color(self._value))
        painter.drawRoundedRect(self.rect(), 3, 3)


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
                file_path=str(row["file_path"]),
                artist=str(row["artist"]),
                title=str(row["title"]),
                bpm=float(row["bpm"]) if row["bpm"] is not None else None,
                key=int(row["key"]) if row["key"] is not None else None,
                energy=float(row["energy"]) if row["energy"] is not None else None,
                cluster_id=int(row["cluster_id"]) if row["cluster_id"] is not None else None,
                cluster_name=str(row["cluster_name"]),
            )
            for row in rows
        ]

    def fetch_track_detail(self, file_path: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    t.file_path,
                    COALESCE(t.artist, 'Unknown Artist') AS artist,
                    COALESCE(t.title, 'Unknown Title') AS title,
                    af.bpm,
                    af.key,
                    af.energy,
                    COALESCE(cm.name, CASE WHEN c.cluster_id = -1 THEN 'Orphan' END, 'Unclustered') AS cluster_name
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
        return [str(row["crate_name"]) for row in rows]

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
                file_path=str(row["file_path"]),
                artist=str(row["artist"]),
                title=str(row["title"]),
                bpm=float(row["bpm"]) if row["bpm"] is not None else None,
                key=int(row["key"]) if row["key"] is not None else None,
                energy=float(row["energy"]) if row["energy"] is not None else None,
                cluster_id=int(row["cluster_id"]) if row["cluster_id"] is not None else None,
                cluster_name=str(row["cluster_name"]),
            )
            for row in rows
        ]


def build_track_model(tracks: list[TrackRecord]) -> QStandardItemModel:
    model = QStandardItemModel(0, 3)
    model.setHorizontalHeaderLabels(["Track", "BPM", "Energy"])

    for track in tracks:
        item_track = QStandardItem(f"{track.artist} — {track.title}")
        item_bpm = QStandardItem(format_bpm(track.bpm))
        item_energy = QStandardItem("")

        for item in (item_track, item_bpm, item_energy):
            item.setEditable(False)
            item.setData(track, TRACK_ROLE)
            item.setData(track.file_path, FILE_PATH_ROLE)
            item.setData(track.bpm, BPM_ROLE)
            item.setData(track.key, KEY_ROLE)
            item.setData(track.energy, ENERGY_ROLE)

        model.appendRow([item_track, item_bpm, item_energy])

    return model


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
        if not isinstance(model, QStandardItemModel):
            return True

        index = model.index(source_row, 0, source_parent)
        track = index.data(TRACK_ROLE)
        if not isinstance(track, TrackRecord):
            return True

        search_value = f"{track.artist} {track.title}".lower()
        if self.search_text and self.search_text not in search_value:
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
            return track.energy is not None and track.energy < ENERGY_LOW_MAX
        if self.energy_filter == "mid":
            return track.energy is not None and ENERGY_LOW_MAX <= track.energy <= ENERGY_MID_MAX
        if self.energy_filter == "high":
            return track.energy is not None and track.energy > ENERGY_MID_MAX

        return True


class BpmDelegate(QStyledItemDelegate):
    def __init__(self, display_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.display_font = display_font

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        if option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, QColor(HOVER_ROW))

        font = QFont(self.display_font, 12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, str(index.data() or "--"))
        painter.restore()


class EnergyDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        if option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, QColor(HOVER_ROW))

        value = index.data(ENERGY_ROLE)
        bar = EnergyBar(float(value) if value is not None else 0.0)
        bar.resize(bar.sizeHint())
        pixmap = bar.grab()
        target = QPoint(
            option.rect.center().x() - pixmap.width() // 2,
            option.rect.center().y() - pixmap.height() // 2,
        )
        painter.drawPixmap(target, pixmap)
        painter.restore()


class SuggestionCard(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.display_font = display_font
        self.body_font = body_font
        self.file_path = ""
        self.setObjectName("SuggestionCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel("Geen suggestie beschikbaar")
        self.title_label.setObjectName("CardTitle")
        self.reason_label = QLabel("")
        self.reason_label.setObjectName("CardReason")
        self.reason_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.reason_label)

    def set_suggestion(self, suggestion: dict[str, Any] | None) -> None:
        if suggestion is None:
            self.file_path = ""
            self.title_label.setText("Geen suggestie beschikbaar")
            self.reason_label.setText("")
            self.setEnabled(False)
            return

        self.file_path = str(suggestion["file_path"])
        self.title_label.setText(f"{suggestion['artist']} — {suggestion['title']}")
        self.reason_label.setText(str(suggestion["reason"]))
        self.setEnabled(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.file_path:
            self.selected.emit(self.file_path)
        super().mousePressEvent(event)


class PageBase(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        self.content_layout = layout


class LibraryTab(PageBase):
    track_selected = pyqtSignal(str)

    def __init__(self, tracks: list[TrackRecord], display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tracks = tracks
        self.base_model = build_track_model(tracks)
        self.proxy_model = TrackFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.base_model)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Zoek artiest of titel")

        self.bpm_filter = QComboBox()
        self.key_filter = QComboBox()
        self.energy_filter = QComboBox()
        self._populate_filters()

        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(self.bpm_filter)
        controls.addWidget(self.key_filter)
        controls.addWidget(self.energy_filter)

        self.table = QTableView()
        self.table.setModel(self.proxy_model)
        self.table.setMouseTracking(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.setShowGrid(True)
        self.table.setColumnWidth(0, 540)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.setItemDelegateForColumn(1, BpmDelegate(display_font, self.table))
        self.table.setItemDelegateForColumn(2, EnergyDelegate(self.table))

        self.content_layout.addWidget(self.search)
        self.content_layout.addLayout(controls)
        self.content_layout.addWidget(self.table, 1)

        self.search.textChanged.connect(self.proxy_model.set_search_text)
        self.bpm_filter.currentIndexChanged.connect(
            lambda: self.proxy_model.set_bpm_range(self.bpm_filter.currentData())
        )
        self.key_filter.currentIndexChanged.connect(
            lambda: self.proxy_model.set_key_filter(self.key_filter.currentData())
        )
        self.energy_filter.currentIndexChanged.connect(
            lambda: self.proxy_model.set_energy_filter(self.energy_filter.currentData())
        )
        self.table.clicked.connect(self._handle_click)

    def _populate_filters(self) -> None:
        self.bpm_filter.addItem("Alle BPM", None)
        bpm_values = [track.bpm for track in self.tracks if track.bpm is not None]
        if bpm_values:
            start = int(min(bpm_values) // 10 * 10)
            stop = int(max(bpm_values) // BPM_HISTOGRAM_BIN_SIZE * BPM_HISTOGRAM_BIN_SIZE + BPM_HISTOGRAM_BIN_SIZE)
            for value in range(start, stop + 1, BPM_HISTOGRAM_BIN_SIZE):
                self.bpm_filter.addItem(
                    f"{value}-{value + BPM_HISTOGRAM_BIN_SIZE - 1}",
                    (value, value + BPM_HISTOGRAM_BIN_SIZE),
                )

        self.key_filter.addItem("Alle keys", None)
        for key_value in range(12):
            self.key_filter.addItem(str(key_value), key_value)

        self.energy_filter.addItem("Alle energy", None)
        self.energy_filter.addItem("Laag", "low")
        self.energy_filter.addItem("Midden", "mid")
        self.energy_filter.addItem("Hoog", "high")

    def _handle_click(self, index) -> None:
        source_index = self.proxy_model.mapToSource(index)
        file_path = self.base_model.index(source_index.row(), 0).data(FILE_PATH_ROLE)
        if file_path:
            self.track_selected.emit(str(file_path))


class ClusterMapTab(PageBase):
    track_selected = pyqtSignal(str)

    def __init__(self, points: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points = points
        self.selected_file_path: str | None = None

        self.plot = pg.PlotWidget()
        self.plot.setBackground(BACKGROUND)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setMenuEnabled(False)

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
            is_selected = str(point["file_path"]) == self.selected_file_path
            color = QColor("#FFFFFF" if is_selected else color_for_cluster(int(point["cluster_id"])))
            spots.append(
                {
                    "pos": (float(point["x"]), float(point["y"])),
                    "size": 10 if is_selected else 6,
                    "brush": color,
                    "pen": pg.mkPen(color),
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


class TrackDetailTab(PageBase):
    add_to_set_requested = pyqtSignal(str)

    def __init__(self, repository: LibraryRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.current_file_path: str | None = None
        self.active_pair_file_path: str | None = None

        self.artist_label = QLabel("Selecteer een track")
        self.artist_label.setObjectName("SectionTitle")
        self.title_label = QLabel("")
        self.bpm_label = QLabel("BPM: --")
        self.key_label = QLabel("Key: --")
        self.energy_label = QLabel("Energy: --")
        self.cluster_label = QLabel("Cluster: --")
        self.energy_bar = EnergyBar(0.0)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(10)
        for widget in (
            self.artist_label,
            self.title_label,
            self.bpm_label,
            self.key_label,
            self.energy_label,
            self.cluster_label,
            self.energy_bar,
        ):
            info_layout.addWidget(widget)

        suggestion_title = QLabel("Beste opvolgers")
        suggestion_title.setObjectName("SectionTitle")
        self.suggestion_cards = [SuggestionCard("DM Mono", "Inter", self) for _ in range(3)]
        for card in self.suggestion_cards:
            card.selected.connect(self._set_active_pair)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.good_mix_button = QPushButton("Goede mix")
        self.bad_mix_button = QPushButton("Slechte mix")
        self.add_button = QPushButton("Voeg toe aan set")
        self.good_mix_button.clicked.connect(lambda: self._handle_feedback(1))
        self.bad_mix_button.clicked.connect(lambda: self._handle_feedback(-1))
        self.add_button.clicked.connect(self._add_to_set)
        actions.addWidget(self.good_mix_button)
        actions.addWidget(self.bad_mix_button)
        actions.addStretch(1)
        actions.addWidget(self.add_button)

        self.content_layout.addLayout(info_layout)
        self.content_layout.addWidget(suggestion_title)
        for card in self.suggestion_cards:
            self.content_layout.addWidget(card)
        self.content_layout.addStretch(1)
        self.content_layout.addLayout(actions)

        self._update_feedback_buttons()

    def set_track(self, file_path: str) -> None:
        detail = self.repository.fetch_track_detail(file_path)
        if detail is None:
            return

        self.current_file_path = file_path
        self.active_pair_file_path = None
        self.artist_label.setText(str(detail["artist"]))
        self.title_label.setText(str(detail["title"]))
        self.bpm_label.setText(f"BPM: {format_bpm(detail['bpm'])}")
        self.key_label.setText(f"Key: {format_key(detail['key'])}")
        self.energy_label.setText(f"Energy: {format_energy(detail['energy'])}")
        self.cluster_label.setText(f"Cluster: {detail['cluster_name']}")
        self.energy_bar.set_value(float(detail["energy"] or 0.0))

        suggestions = get_next_tracks(file_path, [], n=3, db_path=self.repository.database_path)
        for index, card in enumerate(self.suggestion_cards):
            suggestion = suggestions[index] if index < len(suggestions) else None
            card.set_suggestion(suggestion)

        if suggestions:
            self.active_pair_file_path = str(suggestions[0]["file_path"])
        self._update_feedback_buttons()

    def _set_active_pair(self, file_path: str) -> None:
        self.active_pair_file_path = file_path
        self._update_feedback_buttons()

    def _update_feedback_buttons(self) -> None:
        enabled = bool(self.current_file_path and self.active_pair_file_path)
        self.good_mix_button.setEnabled(enabled)
        self.bad_mix_button.setEnabled(enabled)

    def _handle_feedback(self, rating: int) -> None:
        if not self.current_file_path or not self.active_pair_file_path:
            return
        record_feedback(
            self.current_file_path,
            self.active_pair_file_path,
            rating,
            db_path=self.repository.database_path,
        )
        apply_feedback(db_path=self.repository.database_path)
        self.set_track(self.current_file_path)

    def _add_to_set(self) -> None:
        if self.current_file_path:
            self.add_to_set_requested.emit(self.current_file_path)


class SetBuilderTab(PageBase):
    track_selected = pyqtSignal(str)

    def __init__(self, database_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.database_path = database_path
        self.set_tracks: list[str] = []

        self.trajectory_combo = QComboBox()
        self.trajectory_combo.addItems(["Club", "Afterhours", "Warmup"])
        self.length_spinbox = QSpinBox()
        self.length_spinbox.setRange(10, 50)
        self.length_spinbox.setValue(20)
        self.generate_button = QPushButton("Genereer set")
        self.generate_button.clicked.connect(self.generate_set_from_trajectory)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(self.trajectory_combo)
        controls.addWidget(self.length_spinbox)
        controls.addWidget(self.generate_button)
        controls.addStretch(1)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._handle_item_clicked)
        self.suggestion_title = QLabel("Volgende suggestie")
        self.suggestion_title.setObjectName("SectionTitle")
        self.suggestion_card = SuggestionCard("DM Mono", "Inter", self)
        self.suggestion_card.selected.connect(self.track_selected.emit)

        self.remove_button = QPushButton("Verwijder laatste")
        self.remove_button.clicked.connect(self.remove_last)

        self.content_layout.addLayout(controls)
        self.content_layout.addWidget(self.list_widget, 1)
        self.content_layout.addWidget(self.suggestion_title)
        self.content_layout.addWidget(self.suggestion_card)
        self.content_layout.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignLeft)

    def add_track(self, file_path: str) -> None:
        if file_path in self.set_tracks:
            return
        self.set_tracks.append(file_path)
        item = QListWidgetItem(file_path)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.list_widget.addItem(item)
        self.refresh_suggestion()

    def remove_last(self) -> None:
        if not self.set_tracks:
            return
        self.set_tracks.pop()
        self.list_widget.takeItem(self.list_widget.count() - 1)
        self.refresh_suggestion()

    def generate_set_from_trajectory(self) -> None:
        mapping = {
            "Club": "club",
            "Afterhours": "afterhours",
            "Warmup": "warmup",
        }
        generated_tracks = generate_set(
            mapping[self.trajectory_combo.currentText()],
            int(self.length_spinbox.value()),
            db_path=self.database_path,
        )

        self.set_tracks = []
        self.list_widget.clear()
        for track in generated_tracks:
            self.set_tracks.append(str(track["file_path"]))
            item = QListWidgetItem(
                f"{track['phase']}  •  {track['artist']} — {track['title']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(track["file_path"]))
            self.list_widget.addItem(item)

        self.refresh_suggestion()

    def refresh_suggestion(self) -> None:
        if not self.set_tracks:
            self.suggestion_card.set_suggestion(None)
            return
        current = self.set_tracks[-1]
        suggestion = next(
            iter(get_next_tracks(current, list(self.set_tracks), n=1, db_path=self.database_path)),
            None,
        )
        self.suggestion_card.set_suggestion(suggestion)

    def _handle_item_clicked(self, item: QListWidgetItem) -> None:
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.track_selected.emit(str(file_path))


class CratesTab(PageBase):
    track_selected = pyqtSignal(str)

    def __init__(self, repository: LibraryRepository, display_font: str, body_font: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository

        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(220)
        self.list_widget.itemClicked.connect(self._load_crate)

        self.model = build_track_model([])
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setMouseTracking(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.setShowGrid(True)
        self.table.setColumnWidth(0, 520)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.bpm_delegate = BpmDelegate(display_font, self.table)
        self.energy_delegate = EnergyDelegate(self.table)
        self.table.setItemDelegateForColumn(1, self.bpm_delegate)
        self.table.setItemDelegateForColumn(2, self.energy_delegate)
        self.table.clicked.connect(self._handle_click)

        split = QHBoxLayout()
        split.setSpacing(18)
        split.addWidget(self.list_widget)
        split.addWidget(self.table, 1)
        self.content_layout.addLayout(split, 1)

        self.reload_crates()

    def reload_crates(self) -> None:
        self.list_widget.clear()
        for name in self.repository.fetch_crate_names():
            self.list_widget.addItem(name)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
            self._load_crate(self.list_widget.item(0))

    def _load_crate(self, item: QListWidgetItem) -> None:
        tracks = self.repository.fetch_tracks_for_crate(item.text())
        self.model = build_track_model(tracks)
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(1, self.bpm_delegate)
        self.table.setItemDelegateForColumn(2, self.energy_delegate)

    def _handle_click(self, index) -> None:
        file_path = index.data(FILE_PATH_ROLE)
        if file_path:
            self.track_selected.emit(str(file_path))


class CollectionAnalysisTab(PageBase):
    def __init__(self, database_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.database_path = database_path

        self.analyze_button = QPushButton("Analyseer")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.stats_label = QLabel("Klik op Analyseer om inzicht te laden.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextFormat(Qt.TextFormat.PlainText)

        self.orphans_table = self._create_table(["Artiest", "Titel", "Gem. similarity"])
        self.overcrowded_table = self._create_table(["Cluster", "Naam", "Tracks"])
        self.bridges_table = self._create_table(["Cluster A", "Naam A", "Cluster B", "Naam B"])

        self.content_layout.addWidget(self.analyze_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.content_layout.addWidget(self.stats_label)
        self.content_layout.addWidget(QLabel("Orphan tracks"))
        self.content_layout.addWidget(self.orphans_table)
        self.content_layout.addWidget(QLabel("Overvolle clusters"))
        self.content_layout.addWidget(self.overcrowded_table)
        self.content_layout.addWidget(QLabel("Ontbrekende bruggen"))
        self.content_layout.addWidget(self.bridges_table)

    def _create_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _set_table_rows(self, table: QTableWidget, rows: list[list[str]]) -> None:
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, column_index, item)

    def run_analysis(self) -> None:
        generate_crates(db_path=self.database_path)
        orphans = find_orphans(db_path=self.database_path)
        overcrowded = find_overcrowded_clusters(db_path=self.database_path)
        missing_bridges = find_missing_bridges(db_path=self.database_path)
        stats = get_collection_stats(db_path=self.database_path)

        crate_lines = [f"{name}: {value:.1f}%" for name, value in stats["crate_percentages"].items()]
        bpm_lines = [f"{bucket}: {count}" for bucket, count in stats["bpm_histogram"].items()]
        self.stats_label.setText(
            "\n".join(
                [
                    f"Totaal tracks: {stats['total_tracks']}",
                    f"Totaal clusters: {stats['total_clusters']}",
                    f"Totaal orphans: {stats['total_orphans']}",
                    "",
                    "Crate-percentages:",
                    *(crate_lines or ["Geen crate-data."]),
                    "",
                    "BPM-histogram:",
                    *(bpm_lines or ["Geen BPM-data."]),
                ]
            )
        )

        self._set_table_rows(
            self.orphans_table,
            [[row["artist"], row["title"], f"{row['avg_similarity']:.2f}"] for row in orphans],
        )
        self._set_table_rows(
            self.overcrowded_table,
            [[str(row["cluster_id"]), row["name"], str(row["track_count"])] for row in overcrowded],
        )
        self._set_table_rows(
            self.bridges_table,
            [
                [
                    str(row["cluster_a_id"]),
                    row["cluster_a_name"],
                    str(row["cluster_b_id"]),
                    row["cluster_b_name"],
                ]
                for row in missing_bridges
            ],
        )


class MainWindow(QMainWindow):
    def __init__(self, database_path: Path, display_font: str, body_font: str) -> None:
        super().__init__()
        self.database_path = database_path
        self.display_font = display_font
        self.body_font = body_font
        self._pending_page_index: int | None = None
        self._fade_out: QPropertyAnimation | None = None
        self._fade_in: QPropertyAnimation | None = None

        generate_crates(db_path=database_path)

        self.repository = LibraryRepository(database_path)
        self.tracks = self.repository.fetch_tracks()
        self.cluster_points = self.repository.fetch_cluster_points()

        self.setWindowTitle("DJ Library Intelligence")
        self.resize(1480, 920)
        self.setMinimumSize(1280, 820)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(18, 22, 18, 22)
        sidebar_layout.setSpacing(10)

        brand = QLabel("DJ LIB")
        brand.setObjectName("BrandLabel")
        sidebar_layout.addWidget(brand)

        self.content_stack = QStackedWidget()
        self.content_opacity_effect = QGraphicsOpacityEffect(self.content_stack)
        self.content_opacity_effect.setOpacity(1.0)
        self.content_stack.setGraphicsEffect(self.content_opacity_effect)
        self.content_stack.setWindowOpacity(1.0)

        self.library_tab = LibraryTab(self.tracks, display_font, body_font)
        self.cluster_tab = ClusterMapTab(self.cluster_points)
        self.detail_tab = TrackDetailTab(self.repository)
        self.setbuilder_tab = SetBuilderTab(database_path)
        self.crates_tab = CratesTab(self.repository, display_font, body_font)
        self.collection_tab = CollectionAnalysisTab(database_path)

        self.pages = [
            ("Bibliotheek", self.library_tab),
            ("Clustermap", self.cluster_tab),
            ("Trackdetail", self.detail_tab),
            ("Setbuilder", self.setbuilder_tab),
            ("Crates", self.crates_tab),
            ("Collectie-analyse", self.collection_tab),
        ]

        for _, page in self.pages:
            self.content_stack.addWidget(page)

        self.nav_buttons: list[QPushButton] = []
        for index, (label, _) in enumerate(self.pages):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("navButton", True)
            button.clicked.connect(lambda checked=False, idx=index: self.switch_to_page(idx))
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button)
        sidebar_layout.addStretch(1)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.content_stack, 1)

        self.library_tab.track_selected.connect(self.open_track_detail)
        self.cluster_tab.track_selected.connect(self.open_track_detail)
        self.setbuilder_tab.track_selected.connect(self.open_track_detail)
        self.detail_tab.add_to_set_requested.connect(self.setbuilder_tab.add_track)
        self.crates_tab.track_selected.connect(self.open_track_detail)

        self._update_nav_buttons(0)
        self.content_stack.setCurrentIndex(0)
        if self.tracks:
            self.detail_tab.set_track(self.tracks[0].file_path)

    def _update_nav_buttons(self, active_index: int) -> None:
        for index, button in enumerate(self.nav_buttons):
            button.setChecked(index == active_index)

    def _sync_content_opacity(self, value: float) -> None:
        opacity = max(0.0, min(1.0, float(value)))
        self.content_opacity_effect.setOpacity(opacity)

    def switch_to_page(self, index: int) -> None:
        if index == self.content_stack.currentIndex():
            self._update_nav_buttons(index)
            return

        self._pending_page_index = index
        self._update_nav_buttons(index)

        if self._fade_out is not None:
            self._fade_out.stop()
        if self._fade_in is not None:
            self._fade_in.stop()

        self._fade_out = QPropertyAnimation(self.content_stack, b"windowOpacity", self)
        self._fade_out.setDuration(150)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_out.valueChanged.connect(self._sync_content_opacity)
        self._fade_out.finished.connect(self._finish_page_switch)
        self._fade_out.start()

    def _finish_page_switch(self) -> None:
        if self._pending_page_index is None:
            return

        self.content_stack.setCurrentIndex(self._pending_page_index)
        self.content_stack.setWindowOpacity(0.0)
        self.content_opacity_effect.setOpacity(0.0)

        self._fade_in = QPropertyAnimation(self.content_stack, b"windowOpacity", self)
        self._fade_in.setDuration(150)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_in.valueChanged.connect(self._sync_content_opacity)
        self._fade_in.start()

    def open_track_detail(self, file_path: str) -> None:
        self.detail_tab.set_track(file_path)
        self.cluster_tab.refresh_points(file_path)
        self.switch_to_page(2)


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
    app.setStyleSheet(build_stylesheet(display_font, body_font))
    app.display_font = display_font  # type: ignore[attr-defined]
    app.body_font = body_font  # type: ignore[attr-defined]
    pg.setConfigOptions(antialias=True)
    return app


def main() -> None:
    app = build_application()
    display_font = getattr(app, "display_font")
    body_font = getattr(app, "body_font")

    try:
        database_path = resolve_database_path()
        save_config(database_path)
        init_db(database_path)
        window = MainWindow(database_path, display_font, body_font)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Database ontbreekt", str(exc))
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
