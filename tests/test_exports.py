from __future__ import annotations

from lamination_sim.comparison import compare
from lamination_sim.exports import export_comparison_csv, export_html_report
from lamination_sim.presets import default_project
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
    assert "top_risk_a" in csv_text
    assert "필름 역박리 A/B 비교 보고서" in html_text
    assert "실제 안전·불량" in html_text

