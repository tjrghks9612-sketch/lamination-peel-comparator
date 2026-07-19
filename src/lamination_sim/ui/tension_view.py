"""Result tab for paired pull-tape tension sensitivity."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core_bridge import get_value, sequence
from .theme import COLORS
from .visualization import LineChart


class TensionSweepView(QWidget):
    mode_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenarios: list[Any] = []
        self._syncing = False
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 10, 4, 4)
        header = QHBoxLayout()
        title = QLabel("풀테이프 장력 민감도")
        title.setProperty("subheading", True)
        self.mode = QComboBox()
        self.mode.addItem("Equal preload", "equal_preload")
        self.mode.addItem("Shared rest length", "shared_rest_length")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.summary = QLabel("아직 장력 스윕 결과가 없습니다.")
        self.summary.setProperty("muted", True)
        header.addWidget(title)
        header.addWidget(self.mode)
        header.addSpacing(10)
        header.addWidget(self.summary, 1)
        root.addLayout(header)

        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setProperty("muted", True)
        root.addWidget(self.verdict)

        body = QGridLayout()
        self.heatmap = QTableWidget(0, 0)
        self.heatmap.setMinimumHeight(230)
        self.heatmap.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.heatmap.cellClicked.connect(self._show_cell)
        self.detail = QLabel("히트맵 셀을 선택하면 상세 지표가 표시됩니다.")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.detail.setMinimumWidth(330)
        self.detail.setProperty("muted", True)
        body.addWidget(self.heatmap, 0, 0, 1, 2)
        body.addWidget(self.detail, 0, 2)
        body.setColumnStretch(0, 1)
        body.setColumnStretch(1, 1)
        body.setColumnStretch(2, 1)
        root.addLayout(body)

        charts = QHBoxLayout()
        self.risk_chart = LineChart("초기장력 조합 Peak Rtop", "sweep_value")
        self.peel_chart = LineChart("초기장력 조합 하면 박리율", "sweep_value")
        charts.addWidget(self.risk_chart)
        charts.addWidget(self.peel_chart)
        root.addLayout(charts)

    def set_comparison(self, comparison: Any) -> None:
        self._scenarios = sequence(
            get_value(comparison, "tension_scenario_results", default=[])
        )
        mode = str(get_value(comparison, "tension_mode", default="equal_preload"))
        self._syncing = True
        try:
            self.mode.setCurrentIndex(max(0, self.mode.findData(mode)))
        finally:
            self._syncing = False
        wins_a = int(get_value(comparison, "tension_wins_a", default=0))
        wins_b = int(get_value(comparison, "tension_wins_b", default=0))
        ties = int(get_value(comparison, "tension_ties", default=0))
        pending = int(get_value(comparison, "tension_inconclusive", default=0))
        a_rate = float(get_value(comparison, "tension_a_win_rate", default=0.0))
        b_rate = float(get_value(comparison, "tension_b_win_rate", default=0.0))
        self.summary.setText(
            f"A {wins_a}개 ({a_rate*100:.1f}%) · B {wins_b}개 ({b_rate*100:.1f}%) · "
            f"동률 {ties}개 · 보류 {pending}개"
        )
        classification = str(
            get_value(comparison, "tension_classification", default="inconclusive")
        )
        verdict = str(get_value(comparison, "tension_verdict", default=""))
        self.verdict.setText(f"최종 강건성: {classification} — {verdict}")
        self._populate_heatmap()
        self._populate_charts()

    def _populate_heatmap(self) -> None:
        preloads: list[tuple[str, float]] = []
        stiffnesses: list[tuple[str, float]] = []
        for item in self._scenarios:
            preload = (
                str(get_value(item, "preload_label", default="")),
                float(get_value(item, "initial_preload_n", default=0.0)),
            )
            stiffness = (
                str(get_value(item, "stiffness_label", default="")),
                float(get_value(item, "tape_stiffness_n_per_mm", default=0.0)),
            )
            if preload not in preloads:
                preloads.append(preload)
            if stiffness not in stiffnesses:
                stiffnesses.append(stiffness)
        self.heatmap.setRowCount(len(preloads))
        self.heatmap.setColumnCount(len(stiffnesses))
        self.heatmap.setVerticalHeaderLabels(
            [f"{label} · {value:g} N" for label, value in preloads]
        )
        self.heatmap.setHorizontalHeaderLabels(
            [f"{label} · {value:g} N/mm" for label, value in stiffnesses]
        )
        colors = {
            "a": QColor(COLORS["a_dark"]),
            "b": QColor(COLORS["b_dark"]),
            "tie": QColor(COLORS["accent_dark"]),
            "inconclusive": QColor("#3A3320"),
        }
        labels = {"a": "A", "b": "B", "tie": "동률", "inconclusive": "보류"}
        for scenario in self._scenarios:
            preload = (
                str(get_value(scenario, "preload_label", default="")),
                float(get_value(scenario, "initial_preload_n", default=0.0)),
            )
            stiffness = (
                str(get_value(scenario, "stiffness_label", default="")),
                float(get_value(scenario, "tape_stiffness_n_per_mm", default=0.0)),
            )
            row = preloads.index(preload)
            column = stiffnesses.index(stiffness)
            winner = str(get_value(scenario, "winner", default="inconclusive"))
            item = QTableWidgetItem(labels.get(winner, winner))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(colors.get(winner, colors["inconclusive"]))
            item.setData(Qt.ItemDataRole.UserRole, int(get_value(scenario, "index", default=0)))
            self.heatmap.setItem(row, column, item)
        self.heatmap.resizeColumnsToContents()
        if self._scenarios:
            self.heatmap.setCurrentCell(0, 0)
            self._show_cell(0, 0)

    def _show_cell(self, row: int, column: int) -> None:
        item = self.heatmap.item(row, column)
        if item is None:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        scenario = next(
            (
                value
                for value in self._scenarios
                if int(get_value(value, "index", default=-1)) == index
            ),
            None,
        )
        if scenario is None:
            return
        force_a = sequence(get_value(scenario, "max_abs_force_xyz_n_a", default=[]))
        force_b = sequence(get_value(scenario, "max_abs_force_xyz_n_b", default=[]))
        self.detail.setText(
            f"초기장력: {float(get_value(scenario, 'initial_preload_n', default=0.0)):.3f} N\n"
            f"등가 인장강성: {float(get_value(scenario, 'tape_stiffness_n_per_mm', default=0.0)):.3f} N/mm\n"
            f"P1 장력 A/B: {float(get_value(scenario, 'p1_tension_a_n', default=0.0)):.3f} / "
            f"{float(get_value(scenario, 'p1_tension_b_n', default=0.0)):.3f} N\n"
            f"하면 박리 A/B: {float(get_value(scenario, 'final_bottom_peel_ratio_a', default=0.0))*100:.1f}% / "
            f"{float(get_value(scenario, 'final_bottom_peel_ratio_b', default=0.0))*100:.1f}%\n"
            f"하면 gate A/B: {bool(get_value(scenario, 'bottom_gate_pass_a', default=False))} / "
            f"{bool(get_value(scenario, 'bottom_gate_pass_b', default=False))}\n"
            f"Peak Rtop A/B: {float(get_value(scenario, 'peak_top_risk_a', default=0.0)):.4g} / "
            f"{float(get_value(scenario, 'peak_top_risk_b', default=0.0)):.4g}\n"
            f"위험 면적 A/B: {float(get_value(scenario, 'max_top_risk_area_mm2_a', default=0.0)):.3f} / "
            f"{float(get_value(scenario, 'max_top_risk_area_mm2_b', default=0.0)):.3f} mm²\n"
            f"최종 damage A/B: {float(get_value(scenario, 'final_top_damage_area_mm2_a', default=0.0)):.3f} / "
            f"{float(get_value(scenario, 'final_top_damage_area_mm2_b', default=0.0)):.3f} mm²\n"
            f"임계 초과시간 A/B: {float(get_value(scenario, 'top_risk_exceedance_duration_s_a', default=0.0)):.3f} / "
            f"{float(get_value(scenario, 'top_risk_exceedance_duration_s_b', default=0.0)):.3f} s\n"
            f"최대 들림 A/B: {float(get_value(scenario, 'max_panel_lift_mm_a', default=0.0)):.5f} / "
            f"{float(get_value(scenario, 'max_panel_lift_mm_b', default=0.0)):.5f} mm\n"
            f"최대 비틀림 A/B: {float(get_value(scenario, 'max_panel_twist_mm_a', default=0.0)):.5f} / "
            f"{float(get_value(scenario, 'max_panel_twist_mm_b', default=0.0)):.5f} mm\n"
            f"최대 |Fx,Fy,Fz| A: {force_a}\n최대 |Fx,Fy,Fz| B: {force_b}\n"
            f"최대 장력 A/B: {float(get_value(scenario, 'max_tension_n_a', default=0.0)):.3f} / "
            f"{float(get_value(scenario, 'max_tension_n_b', default=0.0)):.3f} N\n"
            f"판정: {str(get_value(scenario, 'winner', default='inconclusive')).upper()}\n"
            f"근거: {get_value(scenario, 'decision_basis', default='')}"
        )

    def _populate_charts(self) -> None:
        count = len(self._scenarios)
        times = [float(index) for index in range(count)]
        risk_a = [float(get_value(item, "peak_top_risk_a", default=0.0)) for item in self._scenarios]
        risk_b = [float(get_value(item, "peak_top_risk_b", default=0.0)) for item in self._scenarios]
        peel_a = [float(get_value(item, "final_bottom_peel_ratio_a", default=0.0)) for item in self._scenarios]
        peel_b = [float(get_value(item, "final_bottom_peel_ratio_b", default=0.0)) for item in self._scenarios]
        end = max(float(count - 1), 1.0)
        for chart, values_a, values_b in (
            (self.risk_chart, risk_a, risk_b),
            (self.peel_chart, peel_a, peel_b),
        ):
            chart.set_time_range(0.0, end)
            chart.set_results(
                {"time_s": times, "sweep_value": values_a},
                {"time_s": times, "sweep_value": values_b},
            )
            chart.set_time(0.0, 0.0)

    def _mode_changed(self, _index: int) -> None:
        if not self._syncing:
            self.mode_requested.emit(str(self.mode.currentData()))


__all__ = ["TensionSweepView"]
