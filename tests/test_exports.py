from __future__ import annotations

import csv

from lamination_sim.comparison import compare
from lamination_sim.exports import export_comparison_csv, export_html_report
from lamination_sim.presets import default_project, measured_project
from lamination_sim.ui.core_bridge import RunBundle


def test_csv_and_html_exports_are_nonempty(tmp_path):
    project = default_project()
    project.run_uncertainty = False
    comparison = compare(project)
    bundle = RunBundle(
        project=project,
        comparison=comparison,
        result_a=comparison.result_a,
        result_b=comparison.result_b,
    )

    csv_path = export_comparison_csv(bundle, tmp_path / "result.csv")
    html_path = export_html_report(bundle, tmp_path / "report.html")

    csv_text = csv_path.read_text(encoding="utf-8-sig")
    html_text = html_path.read_text(encoding="utf-8")
    assert "time_s_a" in csv_text
    assert "actual_speed_mm_s_a" in csv_text
    assert "peel_angle_deg_a" in csv_text
    assert "force_x_n_a" in csv_text
    assert "force_y_n_a" in csv_text
    assert "force_z_n_a" in csv_text
    assert "top_risk_a" in csv_text
    assert "top_damage_area_mm2_a" in csv_text
    assert "moment_n_mm_a" in csv_text
    assert "tension_n_a" in csv_text
    assert "tape_span_length_mm_a" in csv_text
    assert "tension_scenario_index" in csv_text
    assert "p1_tension_a_n" in csv_text
    assert "tension_winner" in csv_text
    assert "필름 역박리 A/B 비교 보고서" in html_text
    assert "실제 안전·불량" in html_text
    assert "풀테이프 장력 민감도" in html_text
    assert "등가 인장강성" in html_text


def test_measured_csv_exports_absolute_z_zero_speed_angle_and_force(tmp_path) -> None:
    project = measured_project()
    comparison = compare(project)
    path = export_comparison_csv(comparison, tmp_path / "measured.csv")

    with path.open(encoding="utf-8-sig", newline="") as stream:
        first = next(csv.DictReader(stream))

    assert float(first["grip_z_mm_a"]) == 4.0
    assert float(first["grip_z_mm_b"]) == 14.0
    assert float(first["actual_speed_mm_s_a"]) == 0.0
    assert float(first["actual_speed_mm_s_b"]) == 0.0
    assert float(first["peel_angle_deg_a"]) > 0.0
    assert float(first["peel_angle_deg_b"]) > 0.0
    assert float(first["force_z_n_a"]) != float(first["force_z_n_b"])
