"""Advanced editor for pull-tape tension sensitivity assumptions."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core_bridge import get_value, sequence
from ..models import (
    PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
    PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
)


class TensionSettingsDialog(QDialog):
    def __init__(
        self,
        config: Any,
        *,
        run_material_uncertainty: bool,
        material_samples: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("풀테이프 장력 민감도 고급 설정")
        self.resize(720, 570)
        self._run_material = run_material_uncertainty
        self._material_samples = material_samples
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Equal preload — A/B P1 초기장력 동일", "equal_preload")
        self.mode.addItem("Shared rest length — 공통 자연 길이", "shared_rest_length")
        self.mode.currentIndexChanged.connect(self._update_enabled)
        self.reference = QComboBox()
        self.reference.addItem("조건 A의 P1 길이를 기준", "condition_a")
        self.reference.addItem("조건 B의 P1 길이를 기준", "condition_b")
        self.reference.addItem("사용자 자연 길이", "custom")
        self.reference.currentIndexChanged.connect(self._update_enabled)
        self.custom_rest = QDoubleSpinBox()
        self.custom_rest.setRange(0.001, 10000.0)
        self.custom_rest.setDecimals(3)
        self.custom_rest.setSuffix(" mm")
        self.nested = QCheckBox("각 장력 조합에 재료 불확실성을 중첩")
        self.nested.toggled.connect(self._update_estimate)
        form.addRow("해석 모드", self.mode)
        form.addRow("자연 길이 기준", self.reference)
        form.addRow("사용자 자연 길이", self.custom_rest)
        form.addRow("재료 불확실성", self.nested)
        root.addLayout(form)

        tables = QHBoxLayout()
        self.preloads = self._level_group(
            "초기장력 수준 (N)", [("Low", 0.0), ("Mid", 0.5), ("High", 1.5)]
        )
        self.stiffnesses = self._level_group(
            "등가 인장강성 수준 (N/mm)",
            [
                (
                    PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
                    PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
                )
            ],
            editable=False,
        )
        tables.addWidget(self.preloads[0])
        tables.addWidget(self.stiffnesses[0])
        root.addLayout(tables, 1)

        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setProperty("muted", True)
        root.addWidget(self.estimate)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._load(config)

    def _level_group(
        self,
        title: str,
        defaults: list[tuple[str, float]],
        *,
        editable: bool = True,
    ) -> tuple[QGroupBox, QTableWidget]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["이름", "값"])
        table.horizontalHeader().setStretchLastSection(True)
        for label, value in defaults:
            self._append_level(table, label, value)
        table.setEnabled(editable)
        layout.addWidget(table)
        if editable:
            controls = QHBoxLayout()
            add = QPushButton("추가")
            remove = QPushButton("선택 삭제")
            add.clicked.connect(lambda: self._append_level(table, "New", 0.0))
            remove.clicked.connect(lambda: self._remove_selected(table))
            table.itemChanged.connect(self._update_estimate)
            controls.addWidget(add)
            controls.addWidget(remove)
            controls.addStretch(1)
            layout.addLayout(controls)
        else:
            note = QLabel(
                "Literature-based PET estimate is fixed in the model; "
                "only initial preload is swept."
            )
            note.setWordWrap(True)
            note.setProperty("muted", True)
            layout.addWidget(note)
        return group, table

    @staticmethod
    def _append_level(table: QTableWidget, label: str, value: float) -> None:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(label))
        table.setItem(row, 1, QTableWidgetItem(f"{value:g}"))

    @staticmethod
    def _remove_selected(table: QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def _load(self, config: Any) -> None:
        mode = str(get_value(config, "mode", default="equal_preload"))
        self.mode.setCurrentIndex(max(0, self.mode.findData(mode)))
        reference = str(
            get_value(config, "rest_length_reference", default="condition_a")
        )
        self.reference.setCurrentIndex(max(0, self.reference.findData(reference)))
        self.custom_rest.setValue(
            float(get_value(config, "custom_rest_length_mm", default=1.0) or 1.0)
        )
        self.nested.setChecked(
            bool(get_value(config, "nest_material_uncertainty", default=False))
        )
        for table, name in ((self.preloads[1], "preload_levels"),):
            levels = sequence(get_value(config, name, default=[]))
            if levels:
                table.setRowCount(0)
                for level in levels:
                    self._append_level(
                        table,
                        str(get_value(level, "label", default="Level")),
                        float(get_value(level, "value", default=0.0)),
                    )
        self._update_enabled()

    def _levels(self, table: QTableWidget) -> list[dict[str, Any]]:
        levels: list[dict[str, Any]] = []
        for row in range(table.rowCount()):
            label_item = table.item(row, 0)
            value_item = table.item(row, 1)
            label = label_item.text().strip() if label_item else ""
            value = float(value_item.text()) if value_item else 0.0
            if not label:
                raise ValueError("수준 이름은 비워둘 수 없습니다.")
            if value < 0.0:
                raise ValueError("장력과 강성은 음수일 수 없습니다.")
            levels.append({"label": label, "value": value})
        if not levels:
            raise ValueError("각 축에는 적어도 하나의 수준이 필요합니다.")
        return levels

    def payload(self) -> dict[str, Any]:
        reference = str(self.reference.currentData())
        return {
            "enabled": True,
            "mode": str(self.mode.currentData()),
            "preload_levels": self._levels(self.preloads[1]),
            "tape_stiffness_n_per_mm": PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
            # Compatibility for v0.5.4 project readers; this is not a sweep axis.
            "stiffness_levels": [
                {
                    "label": PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
                    "value": PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
                }
            ],
            "rest_length_reference": reference,
            "custom_rest_length_mm": (
                self.custom_rest.value() if reference == "custom" else None
            ),
            "nest_material_uncertainty": self.nested.isChecked(),
        }

    def _update_enabled(self, *_args) -> None:
        shared = self.mode.currentData() == "shared_rest_length"
        self.reference.setEnabled(shared)
        self.custom_rest.setEnabled(shared and self.reference.currentData() == "custom")
        self._update_estimate()

    def _update_estimate(self, *_args) -> None:
        combinations = self.preloads[1].rowCount()
        if self._run_material and self.nested.isChecked():
            total = combinations * self._material_samples * 2
            expression = (
                f"장력 조합 {combinations}개 × 재료 가정 {self._material_samples}개 "
                f"× A/B 2조건 = 총 {total}회 해석"
            )
        else:
            material_extra = (
                max(self._material_samples - 1, 0) * 2 if self._run_material else 0
            )
            total = combinations * 2 + material_extra
            expression = f"장력 조합 {combinations}개 × A/B 2조건"
            if material_extra:
                expression += f" + 선택 조합 재료 민감도 {material_extra}회"
            expression += f" = 총 {total}회 해석"
        self.estimate.setText(expression)

    def _accept_validated(self) -> None:
        try:
            payload = self.payload()
            from lamination_sim.models import TensionSweepConfig

            TensionSweepConfig.model_validate(payload)
        except Exception as exc:
            QMessageBox.warning(self, "장력 설정 확인", str(exc))
            return
        self.accept()


__all__ = ["TensionSettingsDialog"]
