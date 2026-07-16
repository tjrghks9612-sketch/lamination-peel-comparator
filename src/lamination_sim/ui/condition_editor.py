"""Condition and six-point trajectory editors."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core_bridge import get_value, sequence
from .theme import COLORS


PANEL_PRESETS = {
    "pro": ("Pro형", 71.5, 149.6),
    "pro_max": ("Pro Max형", 77.6, 163.0),
    "custom": ("사용자 지정", 71.5, 149.6),
}

CORNER_OPTIONS = {
    "bottom_left": "좌하단",
    "bottom_right": "우하단",
    "top_left": "좌상단",
    "top_right": "우상단",
}


def _spin(
    value: float,
    minimum: float,
    maximum: float,
    decimals: int = 3,
    suffix: str = "",
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSingleStep(0.1 if decimals else 1.0)
    if suffix:
        widget.setSuffix(f" {suffix}")
    widget.setKeyboardTracking(False)
    return widget


class ConditionEditor(QWidget):
    """Dense but approachable editor for one comparison condition."""

    changed = Signal()

    def __init__(self, label: str, accent: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.accent = accent
        self._loading = False
        self._initial_approach: list[dict[str, float]] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        header = QFrame()
        header.setProperty("card", True)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 11, 14, 11)
        badge = QLabel(label)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(30, 26)
        badge.setStyleSheet(
            f"background:{accent}; color:#071015; border-radius:7px; font-weight:800;"
        )
        title = QLabel(f"조건 {label}")
        title.setProperty("subheading", True)
        hint = QLabel("패널 · 필름 · 6포인트 경로")
        hint.setProperty("muted", True)
        header_layout.addWidget(badge)
        header_layout.addWidget(title)
        header_layout.addWidget(hint)
        header_layout.addStretch(1)
        root.addWidget(header)

        root.addWidget(self._build_panel_group())
        root.addWidget(self._build_film_group())
        root.addWidget(self._build_tape_group())
        root.addWidget(self._build_trajectory_group(), 1)

        self._connect_changes()

    def _build_panel_group(self) -> QGroupBox:
        box = QGroupBox("패널")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(9)
        layout.setVerticalSpacing(8)
        self.panel_preset = QComboBox()
        for key, (text, _width, _height) in PANEL_PRESETS.items():
            self.panel_preset.addItem(text, key)
        self.panel_width = _spin(71.5, 20, 300, 2, "mm")
        self.panel_height = _spin(149.6, 20, 400, 2, "mm")
        self.panel_thickness = _spin(0.7, 0.05, 20, 3, "mm")
        layout.addWidget(QLabel("크기 프리셋"), 0, 0)
        layout.addWidget(self.panel_preset, 0, 1, 1, 3)
        layout.addWidget(QLabel("폭"), 1, 0)
        layout.addWidget(self.panel_width, 1, 1)
        layout.addWidget(QLabel("길이"), 1, 2)
        layout.addWidget(self.panel_height, 1, 3)
        layout.addWidget(QLabel("두께"), 2, 0)
        layout.addWidget(self.panel_thickness, 2, 1)
        return box

    def _build_film_group(self) -> QGroupBox:
        box = QGroupBox("상·하면 필름")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(7)
        top = QLabel("상면")
        top.setStyleSheet(f"color:{COLORS['a']}; font-weight:700;")
        bottom = QLabel("하면")
        bottom.setStyleSheet(f"color:{COLORS['b']}; font-weight:700;")
        layout.addWidget(QLabel(""), 0, 0)
        layout.addWidget(top, 0, 1)
        layout.addWidget(bottom, 0, 2)
        self.top_pet = _spin(50, 1, 1000, 1, "µm")
        self.bottom_pet = _spin(50, 1, 1000, 1, "µm")
        self.top_psa = _spin(20, 1, 1000, 1, "µm")
        self.bottom_psa = _spin(20, 1, 1000, 1, "µm")
        self.top_adhesion = _spin(2.0, 0.001, 10000, 3, "gf")
        self.bottom_adhesion = _spin(1.5, 0.001, 10000, 3, "gf")
        for row, (text, first, second) in enumerate(
            (
                ("PET 두께", self.top_pet, self.bottom_pet),
                ("PSA 두께", self.top_psa, self.bottom_psa),
                ("점착력", self.top_adhesion, self.bottom_adhesion),
            ),
            start=1,
        ):
            layout.addWidget(QLabel(text), row, 0)
            layout.addWidget(first, row, 1)
            layout.addWidget(second, row, 2)
        note = QLabel("gf는 동일 시험조건의 상대 접착 저항으로 사용됩니다.")
        note.setWordWrap(True)
        note.setProperty("dim", True)
        layout.addWidget(note, 4, 0, 1, 3)
        return box

    def _build_tape_group(self) -> QGroupBox:
        box = QGroupBox("풀테이프")
        layout = QFormLayout(box)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        self.start_corner = QComboBox()
        for key, label in CORNER_OPTIONS.items():
            self.start_corner.addItem(label, key)
        dimensions = QWidget()
        dims_layout = QHBoxLayout(dimensions)
        dims_layout.setContentsMargins(0, 0, 0, 0)
        dims_layout.setSpacing(7)
        self.tape_width = _spin(10, 1, 100, 2, "mm")
        self.tape_length = _spin(10, 1, 100, 2, "mm")
        dims_layout.addWidget(self.tape_width)
        cross = QLabel("×")
        cross.setProperty("muted", True)
        dims_layout.addWidget(cross)
        dims_layout.addWidget(self.tape_length)
        layout.addRow("시작 코너", self.start_corner)
        layout.addRow("파지 폭 × 길이", dimensions)
        return box

    def _build_trajectory_group(self) -> QGroupBox:
        box = QGroupBox("박리 궤적 · 시작점 포함 6개")
        layout = QVBoxLayout(box)
        helper_row = QHBoxLayout()
        helper = QLabel(
            "속도는 각 waypoint의 목표속도입니다. 포인트 사이에서 선형 보간하며 "
            "P1=0은 정지 출발로 허용됩니다. Z는 패널 표면(Z=0) 기준 절대값입니다."
        )
        helper.setWordWrap(True)
        helper.setProperty("dim", True)
        import_button = QPushButton("CSV 불러오기")
        export_button = QPushButton("CSV 저장")
        import_button.setToolTip("x_mm, y_mm, z_mm, speed_mm_s 열을 가진 6행 CSV")
        export_button.setToolTip("현재 6개 궤적 포인트를 CSV로 저장")
        import_button.clicked.connect(self._import_trajectory_csv)
        export_button.clicked.connect(self._export_trajectory_csv)
        helper_row.addWidget(helper, 1)
        helper_row.addWidget(import_button)
        helper_row.addWidget(export_button)
        layout.addLayout(helper_row)
        self.trajectory_table = QTableWidget(6, 4)
        self.trajectory_table.setHorizontalHeaderLabels(
            ["X (mm)", "Y (mm)", "Z (mm)", "속도 (mm/s)"]
        )
        self.trajectory_table.setVerticalHeaderLabels([f"P{i}" for i in range(1, 7)])
        self.trajectory_table.setAlternatingRowColors(True)
        self.trajectory_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.trajectory_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.trajectory_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trajectory_table.verticalHeader().setDefaultSectionSize(32)
        self.trajectory_table.setMinimumHeight(238)
        defaults = [
            (0, 0, 0, 5),
            (4, 6, 1.5, 8),
            (16, 30, 3.5, 15),
            (34, 70, 6, 25),
            (54, 112, 9, 35),
            (72, 150, 12, 35),
        ]
        for row, values in enumerate(defaults):
            for column, value in enumerate(values):
                item = QTableWidgetItem(f"{value:.3f}".rstrip("0").rstrip("."))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.trajectory_table.setItem(row, column, item)
        layout.addWidget(self.trajectory_table)
        return box

    def _import_trajectory_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"조건 {self.label} 궤적 CSV 불러오기",
            "",
            "Trajectory CSV (*.csv);;모든 파일 (*)",
        )
        if not filename:
            return
        try:
            from lamination_sim.project_io import load_trajectory_csv

            points = load_trajectory_csv(filename)
            self._loading = True
            for row, point in enumerate(points):
                for column, field in enumerate(("x_mm", "y_mm", "z_mm", "speed_mm_s")):
                    item = self.trajectory_table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self.trajectory_table.setItem(row, column, item)
                    item.setText(f"{float(getattr(point, field)):.6g}")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
        except Exception as exc:
            QMessageBox.critical(self, "궤적 CSV를 불러올 수 없습니다", str(exc))
            return
        finally:
            self._loading = False
        self.changed.emit()

    def _export_trajectory_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"조건 {self.label} 궤적 CSV 저장",
            f"trajectory_{self.label}.csv",
            "Trajectory CSV (*.csv)",
        )
        if not filename:
            return
        try:
            from lamination_sim.project_io import save_trajectory_csv

            points = self.condition_payload()["trajectory"]
            save_trajectory_csv(points, filename)
        except Exception as exc:
            QMessageBox.critical(self, "궤적 CSV를 저장할 수 없습니다", str(exc))
            return
        QMessageBox.information(self, "궤적 CSV 저장", f"조건 {self.label} 궤적을 저장했습니다.\n{filename}")

    def _connect_changes(self) -> None:
        self.panel_preset.currentIndexChanged.connect(self._on_preset_changed)
        for widget in (
            self.panel_width,
            self.panel_height,
            self.panel_thickness,
            self.top_pet,
            self.top_psa,
            self.top_adhesion,
            self.bottom_pet,
            self.bottom_psa,
            self.bottom_adhesion,
            self.tape_width,
            self.tape_length,
        ):
            widget.valueChanged.connect(self._emit_changed)
        self.start_corner.currentIndexChanged.connect(self._emit_changed)
        self.trajectory_table.itemChanged.connect(self._emit_changed)

    def _emit_changed(self, *_args) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_preset_changed(self, _index: int) -> None:
        if self._loading:
            return
        key = self.panel_preset.currentData()
        custom = key == "custom"
        self.panel_width.setReadOnly(not custom)
        self.panel_height.setReadOnly(not custom)
        if not custom:
            _label, width, height = PANEL_PRESETS[key]
            self.panel_width.setValue(width)
            self.panel_height.setValue(height)
        self.changed.emit()

    def condition_payload(self) -> dict[str, Any]:
        trajectory = []
        for row in range(6):
            values: list[float] = []
            for column in range(4):
                item = self.trajectory_table.item(row, column)
                text = item.text().strip().replace(",", ".") if item else "0"
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(f"조건 {self.label} P{row + 1}의 값이 숫자가 아닙니다: {text}") from exc
                values.append(value)
            if values[3] < 0:
                raise ValueError(f"조건 {self.label} P{row + 1} 속도는 음수일 수 없습니다.")
            trajectory.append(
                {
                    "x_mm": values[0],
                    "y_mm": values[1],
                    "z_mm": values[2],
                    "speed_mm_s": values[3],
                }
            )
        payload = {
            "name": f"Condition {self.label}",
            "panel": {
                "preset": self.panel_preset.currentData(),
                "width_mm": self.panel_width.value(),
                "height_mm": self.panel_height.value(),
                "thickness_mm": self.panel_thickness.value(),
                "corner_radius_mm": 0.0,
            },
            "top_film": {
                "pet_thickness_um": self.top_pet.value(),
                "psa_thickness_um": self.top_psa.value(),
                "adhesion_gf": self.top_adhesion.value(),
            },
            "bottom_film": {
                "pet_thickness_um": self.bottom_pet.value(),
                "psa_thickness_um": self.bottom_psa.value(),
                "adhesion_gf": self.bottom_adhesion.value(),
            },
            "pull_tape": {
                "start_corner": self.start_corner.currentData(),
                "width_mm": self.tape_width.value(),
                "length_mm": self.tape_length.value(),
            },
            "trajectory": trajectory,
        }
        if self._initial_approach is not None:
            payload["initial_approach"] = self._initial_approach
        return payload

    def set_condition(self, condition: Any) -> None:
        self._loading = True
        try:
            approach = get_value(condition, "initial_approach", default=None)
            self._initial_approach = (
                [
                    {
                        "x_mm": float(get_value(point, "x_mm", "x", default=0)),
                        "y_mm": float(get_value(point, "y_mm", "y", default=0)),
                        "z_mm": float(get_value(point, "z_mm", "z", default=0)),
                        "speed_mm_s": float(
                            get_value(point, "speed_mm_s", "speed", default=0)
                        ),
                    }
                    for point in sequence(approach)
                ]
                if approach is not None
                else None
            )
            panel = get_value(condition, "panel", default={})
            preset = str(get_value(panel, "preset", default="pro"))
            index = self.panel_preset.findData(preset)
            self.panel_preset.setCurrentIndex(max(0, index))
            self.panel_width.setValue(float(get_value(panel, "width_mm", default=71.5)))
            self.panel_height.setValue(float(get_value(panel, "height_mm", default=149.6)))
            self.panel_thickness.setValue(float(get_value(panel, "thickness_mm", default=0.7)))

            for prefix in ("top", "bottom"):
                film = get_value(condition, f"{prefix}_film", default={})
                getattr(self, f"{prefix}_pet").setValue(
                    float(get_value(film, "pet_thickness_um", default=50))
                )
                getattr(self, f"{prefix}_psa").setValue(
                    float(get_value(film, "psa_thickness_um", default=20))
                )
                default_adhesion = 2.0 if prefix == "top" else 1.5
                getattr(self, f"{prefix}_adhesion").setValue(
                    float(get_value(film, "adhesion_gf", default=default_adhesion))
                )

            tape = get_value(condition, "pull_tape", "tape", default={})
            corner = str(get_value(tape, "start_corner", default="bottom_left"))
            corner_index = self.start_corner.findData(corner)
            self.start_corner.setCurrentIndex(max(0, corner_index))
            self.tape_width.setValue(float(get_value(tape, "width_mm", default=10)))
            self.tape_length.setValue(float(get_value(tape, "length_mm", default=10)))

            trajectory = sequence(get_value(condition, "trajectory", "points", default=[]))
            for row, point in enumerate(trajectory[:6]):
                values = (
                    get_value(point, "x_mm", "x", default=0),
                    get_value(point, "y_mm", "y", default=0),
                    get_value(point, "z_mm", "z", default=0),
                    get_value(point, "speed_mm_s", "speed", default=1),
                )
                for column, value in enumerate(values):
                    item = self.trajectory_table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self.trajectory_table.setItem(row, column, item)
                    item.setText(f"{float(value):.4f}".rstrip("0").rstrip("."))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        finally:
            self._loading = False
        self._on_preset_changed(self.panel_preset.currentIndex())
        self.changed.emit()

    def set_shared_inputs(self, source: "ConditionEditor") -> None:
        """Copy all non-trajectory inputs from another editor."""

        payload = source.condition_payload()
        current = self.condition_payload()
        payload["trajectory"] = current["trajectory"]
        if "initial_approach" in current:
            payload["initial_approach"] = current["initial_approach"]
        else:
            payload.pop("initial_approach", None)
        self.set_condition(payload)
