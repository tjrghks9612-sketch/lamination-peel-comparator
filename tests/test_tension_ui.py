from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lamination_sim.models import TensionSweepConfig
from lamination_sim.ui.results_view import ResultsView
from lamination_sim.ui.tension_settings import TensionSettingsDialog
from lamination_sim.ui.tension_view import TensionSweepView


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _scenario(index: int, preload: float, stiffness: float) -> dict:
    return {
        "index": index,
        "preload_label": ("Low", "Mid", "High")[index // 3],
        "stiffness_label": ("Low", "Mid", "High")[index % 3],
        "initial_preload_n": preload,
        "tape_stiffness_n_per_mm": stiffness,
        "p1_tension_a_n": preload,
        "p1_tension_b_n": preload,
        "final_bottom_peel_ratio_a": 0.9,
        "final_bottom_peel_ratio_b": 0.8,
        "peak_top_risk_a": 1.0 + index,
        "peak_top_risk_b": 2.0 + index,
        "max_top_risk_area_mm2_a": 1.0,
        "max_top_risk_area_mm2_b": 2.0,
        "final_top_damage_area_mm2_a": 0.5,
        "final_top_damage_area_mm2_b": 0.7,
        "max_panel_lift_mm_a": 0.01,
        "max_panel_lift_mm_b": 0.02,
        "max_abs_force_xyz_n_a": [1.0, 2.0, 3.0],
        "max_abs_force_xyz_n_b": [2.0, 3.0, 4.0],
        "winner": "a" if index < 7 else "inconclusive",
        "decision_basis": "test basis",
    }


def test_tension_tab_builds_three_by_three_unit_labeled_heatmap(
    qt_app: QApplication,
) -> None:
    preloads = [0.0, 0.5, 1.5]
    stiffnesses = [0.05, 0.2, 1.0]
    scenarios = [
        _scenario(row * 3 + column, preload, stiffness)
        for row, preload in enumerate(preloads)
        for column, stiffness in enumerate(stiffnesses)
    ]
    view = TensionSweepView()
    view.set_comparison(
        {
            "tension_scenario_results": scenarios,
            "tension_mode": "equal_preload",
            "tension_wins_a": 7,
            "tension_inconclusive": 2,
            "tension_a_win_rate": 7 / 9,
            "tension_classification": "weak_a",
            "tension_verdict": "장력 가정에 대한 약한 A 우세",
        }
    )

    assert view.heatmap.rowCount() == 3
    assert view.heatmap.columnCount() == 3
    assert "1.5 N" in view.heatmap.verticalHeaderItem(2).text()
    assert "1 N/mm" in view.heatmap.horizontalHeaderItem(2).text()
    assert "Peak Rtop" in view.detail.text()


def test_advanced_settings_show_nested_432_run_estimate(qt_app: QApplication) -> None:
    dialog = TensionSettingsDialog(
        TensionSweepConfig(),
        run_material_uncertainty=True,
        material_samples=24,
    )
    dialog.nested.setChecked(True)

    assert "432" in dialog.estimate.text()
    assert len(dialog.payload()["preload_levels"]) == 3
    assert len(dialog.payload()["stiffness_levels"]) == 3


def test_results_workspace_contains_tension_sensitivity_tab(qt_app: QApplication) -> None:
    view = ResultsView()

    labels = [view.chart_tabs.tabText(index) for index in range(view.chart_tabs.count())]
    assert "풀테이프 장력 민감도" in labels
