"""Synchronized comparison result workspace."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .core_bridge import RunBundle, get_value, scalar_metric
from .theme import COLORS
from .visualization import LineChart, PeelView


CLASSIFICATION_LABELS = {
    "robust_a": "A가 강건하게 유리",
    "robust_b": "B가 강건하게 유리",
    "weak_a": "A가 약하게 우세",
    "weak_b": "B가 약하게 우세",
    "tie": "동률",
    "inconclusive": "판정 보류",
}


def _readable_text(value: str) -> bool:
    if not value:
        return False
    return value.count("?") / max(1, len(value)) < 0.08


def _format_metric(value: float | None, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "—"
    if value != 0 and (abs(value) < 0.01 or abs(value) >= 10000):
        return f"{value:.2e}{suffix}"
    return f"{value:.{decimals}f}{suffix}"


class MetricCard(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        layout.setSpacing(3)
        self.title_label = QLabel(title)
        self.title_label.setProperty("muted", True)
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet("font-size:17px; font-weight:700;")
        self.detail_label = QLabel("")
        self.detail_label.setProperty("dim", True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str = "", tone: str | None = None) -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)
        if tone:
            self.value_label.setStyleSheet(f"font-size:17px; font-weight:700; color:{tone};")
        else:
            self.value_label.setStyleSheet("font-size:17px; font-weight:700;")


class SummaryBanner(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(12)
        self.badge = QLabel("대기")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setMinimumWidth(96)
        self.badge.setFixedHeight(30)
        self.badge.setStyleSheet(
            f"background:{COLORS['surface_raised']}; color:{COLORS['text_muted']}; "
            "border-radius:7px; font-weight:700; padding:0 10px;"
        )
        text_box = QWidget()
        text_layout = QVBoxLayout(text_box)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        self.title = QLabel("아직 비교 결과가 없습니다")
        self.title.setProperty("subheading", True)
        self.detail = QLabel("동일 가정으로 조건 A와 B를 계산하면 여기에 상대 우세가 표시됩니다.")
        self.detail.setWordWrap(True)
        self.detail.setProperty("muted", True)
        text_layout.addWidget(self.title)
        text_layout.addWidget(self.detail)
        layout.addWidget(self.badge)
        layout.addWidget(text_box, 1)

    def set_comparison(self, comparison: Any) -> None:
        winner = str(get_value(comparison, "winner", default="")).strip().upper()
        classification = str(get_value(comparison, "classification", default="")).strip()
        verdict = str(get_value(comparison, "verdict", default="")).strip()
        change = scalar_metric(
            comparison,
            "a_vs_b_peak_risk_change_percent",
            "peak_risk_change_percent",
            default=None,
        )
        if winner in {"A", "CONDITION A", "CONDITION_A"}:
            badge_text = "A 우세"
            tone = COLORS["a"]
            title = "조건 A가 상면 역박리 억제에 더 유리합니다"
        elif winner in {"B", "CONDITION B", "CONDITION_B"}:
            badge_text = "B 우세"
            tone = COLORS["b"]
            title = "조건 B가 상면 역박리 억제에 더 유리합니다"
        elif winner == "TIE":
            badge_text = "동률"
            tone = COLORS["accent"]
            title = "하면 gate와 Pareto 지표에서 두 조건이 동률입니다"
        else:
            badge_text = "판정 보류"
            tone = COLORS["warning"]
            title = "gate 또는 Pareto 지표가 교차하여 판정을 보류합니다"
        self.badge.setText(badge_text)
        self.badge.setStyleSheet(
            f"background:{tone}22; color:{tone}; border:1px solid {tone}55; "
            "border-radius:7px; font-weight:750; padding:0 10px;"
        )
        self.title.setText(title)
        details = [CLASSIFICATION_LABELS.get(classification, classification)]
        if _readable_text(verdict):
            details.append(verdict)
        if change is not None:
            details.append(f"A 대비 B peak Rtop 변화 {change:+.1f}%")
        self.detail.setText(" · ".join(details) or "명목 조건 비교가 완료되었습니다.")


class ResultsView(QWidget):
    """Result summary, paired visualization, synchronized timeline and charts."""

    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bundle: RunBundle | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(42)
        self._timer.timeout.connect(self._advance)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(1, 1, 7, 8)
        self.content_layout.setSpacing(10)

        self.summary = SummaryBanner()
        self.content_layout.addWidget(self.summary)
        self.content_layout.addLayout(self._build_metrics())
        self.content_layout.addLayout(self._build_views(), 1)
        self.content_layout.addWidget(self._build_timeline())
        self.content_layout.addWidget(self._build_charts())
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_metrics(self) -> QGridLayout:
        layout = QGridLayout()
        layout.setSpacing(8)
        self.risk_card = MetricCard("Peak Rtop · A / B")
        self.lift_card = MetricCard("최대 패널 들림 · A / B")
        self.peel_card = MetricCard("하면 최종 박리율 · A / B")
        self.robust_card = MetricCard("민감도 우세율")
        cards = [self.risk_card, self.lift_card, self.peel_card, self.robust_card]
        for index, card in enumerate(cards):
            layout.addWidget(card, 0, index)
            layout.setColumnStretch(index, 1)
        return layout

    def _build_views(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(9)
        self.view_a = PeelView("A", COLORS["a"])
        self.view_b = PeelView("B", COLORS["b"])
        self.view_a.progress_requested.connect(self._set_progress_from_view)
        self.view_b.progress_requested.connect(self._set_progress_from_view)
        layout.addWidget(self.view_a, 1)
        layout.addWidget(self.view_b, 1)
        return layout

    def _set_progress_from_view(self, progress: float) -> None:
        self.timeline.setValue(round(max(0.0, min(1.0, progress)) * 1000))

    def _build_timeline(self) -> QFrame:
        frame = QFrame()
        frame.setProperty("card", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(38)
        self.play_button.setToolTip("동기화 재생 / 일시정지")
        self.play_button.clicked.connect(self._toggle_playback)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.valueChanged.connect(self._on_timeline)
        self.timeline_label = QLabel("0%")
        self.timeline_label.setFixedWidth(45)
        self.timeline_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.timeline_label.setProperty("muted", True)
        layout.addWidget(self.play_button)
        layout.addWidget(QLabel("동기화 진행률"))
        layout.addWidget(self.timeline, 1)
        layout.addWidget(self.timeline_label)
        return frame

    def _build_charts(self) -> QTabWidget:
        tabs = QTabWidget()
        risk_tab = QWidget()
        risk_layout = QHBoxLayout(risk_tab)
        risk_layout.setContentsMargins(0, 8, 0, 0)
        risk_layout.setSpacing(8)
        self.risk_chart = LineChart("상면 최대 역박리 위험도", "top_risk")
        self.peel_chart = LineChart("하면 박리율", "bottom_peel")
        risk_layout.addWidget(self.risk_chart)
        risk_layout.addWidget(self.peel_chart)

        damage_tab = QWidget()
        damage_layout = QHBoxLayout(damage_tab)
        damage_layout.setContentsMargins(0, 8, 0, 0)
        damage_layout.setSpacing(8)
        self.risk_area_chart = LineChart(
            "Rtop 임계 초과 면적", "top_risk_area", " mm²"
        )
        self.damage_chart = LineChart(
            "상면 cohesive damage 면적", "top_damage", " mm²"
        )
        damage_layout.addWidget(self.risk_area_chart)
        damage_layout.addWidget(self.damage_chart)

        mechanics_tab = QWidget()
        mechanics_layout = QHBoxLayout(mechanics_tab)
        mechanics_layout.setContentsMargins(0, 8, 0, 0)
        mechanics_layout.setSpacing(8)
        self.lift_chart = LineChart("패널 최대 들림", "panel_lift", " mm")
        self.force_chart = LineChart("풀테이프 반력", "force", " N")
        mechanics_layout.addWidget(self.lift_chart)
        mechanics_layout.addWidget(self.force_chart)

        moment_tab = QWidget()
        moment_layout = QHBoxLayout(moment_tab)
        moment_layout.setContentsMargins(0, 8, 0, 0)
        moment_layout.setSpacing(8)
        self.twist_chart = LineChart("패널 비틀림", "twist", " mm")
        self.moment_chart = LineChart("등가 하중 모멘트", "moment", " N·mm")
        moment_layout.addWidget(self.twist_chart)
        moment_layout.addWidget(self.moment_chart)

        command_tab = QWidget()
        command_layout = QHBoxLayout(command_tab)
        command_layout.setContentsMargins(0, 8, 0, 0)
        command_layout.setSpacing(8)
        self.speed_chart = LineChart("실제 보간 속도", "speed", " mm/s")
        self.angle_chart = LineChart("실제 전선-파지 박리각", "peel_angle", "°")
        command_layout.addWidget(self.speed_chart)
        command_layout.addWidget(self.angle_chart)

        force_components_tab = QWidget()
        force_components_layout = QHBoxLayout(force_components_tab)
        force_components_layout.setContentsMargins(0, 8, 0, 0)
        force_components_layout.setSpacing(8)
        self.force_x_chart = LineChart("반력 Fx", "force_x", " N")
        self.force_y_chart = LineChart("반력 Fy", "force_y", " N")
        self.force_z_chart = LineChart("반력 Fz", "force_z", " N")
        force_components_layout.addWidget(self.force_x_chart)
        force_components_layout.addWidget(self.force_y_chart)
        force_components_layout.addWidget(self.force_z_chart)

        tabs.addTab(risk_tab, "위험도 · 박리")
        tabs.addTab(damage_tab, "위험 면적 · 손상")
        tabs.addTab(mechanics_tab, "들림 · 반력")
        tabs.addTab(moment_tab, "비틀림 · 모멘트")
        tabs.addTab(command_tab, "속도 · 박리각")
        tabs.addTab(force_components_tab, "Fx · Fy · Fz")
        return tabs

    def set_bundle(self, bundle: RunBundle) -> None:
        self.bundle = bundle
        comparison = bundle.comparison
        result_a = bundle.result_a or get_value(comparison, "result_a", "simulation_a")
        result_b = bundle.result_b or get_value(comparison, "result_b", "simulation_b")
        condition_a = get_value(bundle.project, "condition_a", "a")
        condition_b = get_value(bundle.project, "condition_b", "b")
        self.summary.set_comparison(comparison)
        self.view_a.set_data(condition_a, result_a)
        self.view_b.set_data(condition_b, result_b)
        for chart in (
            self.risk_chart,
            self.peel_chart,
            self.risk_area_chart,
            self.damage_chart,
            self.lift_chart,
            self.force_chart,
            self.twist_chart,
            self.moment_chart,
            self.speed_chart,
            self.angle_chart,
            self.force_x_chart,
            self.force_y_chart,
            self.force_z_chart,
        ):
            chart.set_results(result_a, result_b)
        self._set_metric_cards(comparison, result_a, result_b)
        self.timeline.setValue(0)

    def _set_metric_cards(self, comparison: Any, result_a: Any, result_b: Any) -> None:
        risk_a = scalar_metric(result_a, "peak_top_risk", "top_peak_risk", "peak_rtop")
        risk_b = scalar_metric(result_b, "peak_top_risk", "top_peak_risk", "peak_rtop")
        lift_a = scalar_metric(result_a, "max_panel_lift_mm", "panel_max_lift_mm", "max_lift_mm")
        lift_b = scalar_metric(result_b, "max_panel_lift_mm", "panel_max_lift_mm", "max_lift_mm")
        peel_a = scalar_metric(result_a, "final_bottom_peel_ratio", "bottom_peel_ratio")
        peel_b = scalar_metric(result_b, "final_bottom_peel_ratio", "bottom_peel_ratio")
        completion_threshold = scalar_metric(
            comparison, "bottom_completion_threshold", default=0.98
        )
        self.risk_card.set_value(f"{_format_metric(risk_a)} / {_format_metric(risk_b)}", "낮을수록 유리")
        self.lift_card.set_value(
            f"{_format_metric(lift_a, ' mm')} / {_format_metric(lift_b, ' mm')}",
            "패널-상면 상대 변위",
        )
        self.peel_card.set_value(
            f"{_format_metric(peel_a * 100 if peel_a is not None else None, '%', 1)} / "
            f"{_format_metric(peel_b * 100 if peel_b is not None else None, '%', 1)}",
            f"{(completion_threshold or 0.98) * 100:.0f}% 이상이면 완료 gate 통과",
        )
        rate_a = scalar_metric(
            comparison,
            "uncertainty_a_win_rate",
            "a_win_rate",
            "condition_a_win_rate",
            "a_wins_rate",
        )
        rate_b = scalar_metric(
            comparison,
            "uncertainty_b_win_rate",
            "b_win_rate",
            "condition_b_win_rate",
            "b_wins_rate",
        )
        counts = get_value(comparison, "uncertainty_counts", default={})
        if rate_a is None:
            rate_a = scalar_metric(counts, "a_rate", "a_win_rate")
        if rate_b is None:
            rate_b = scalar_metric(counts, "b_rate", "b_win_rate")
        if rate_a is not None and rate_a <= 1:
            rate_a *= 100
        if rate_b is not None and rate_b <= 1:
            rate_b *= 100
        uncertainty_enabled = bool(get_value(comparison, "uncertainty_enabled", default=False))
        if not uncertainty_enabled:
            self.robust_card.set_value("사용 안 함", "명목 조건만 계산")
        else:
            ties = scalar_metric(comparison, "uncertainty_ties", default=0) or 0
            scenarios = get_value(comparison, "scenario_results", default=[])
            count = len(scenarios) if isinstance(scenarios, (list, tuple)) else 0
            self.robust_card.set_value(
                f"A {_format_metric(rate_a, '%', 0)} · B {_format_metric(rate_b, '%', 0)}",
                f"{count}개 중 동률 {ties:.0f}회",
            )

    def _on_timeline(self, value: int) -> None:
        progress = value / 1000.0
        self.timeline_label.setText(f"{progress * 100:.0f}%")
        self.view_a.set_progress(progress)
        self.view_b.set_progress(progress)
        for chart in (
            self.risk_chart,
            self.peel_chart,
            self.risk_area_chart,
            self.damage_chart,
            self.lift_chart,
            self.force_chart,
            self.twist_chart,
            self.moment_chart,
        ):
            chart.set_progress(progress)

    def _toggle_playback(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("▶")
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.timeline.setValue(0)
            self._timer.start()
            self.play_button.setText("❚❚")

    def _advance(self) -> None:
        next_value = self.timeline.value() + 7
        if next_value >= self.timeline.maximum():
            self.timeline.setValue(self.timeline.maximum())
            self._timer.stop()
            self.play_button.setText("▶")
        else:
            self.timeline.setValue(next_value)
