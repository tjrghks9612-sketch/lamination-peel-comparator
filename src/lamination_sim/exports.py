"""Project and result export helpers.

The functions in this module have no Qt dependency so they can also be used from
batch scripts and tests.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .ui.core_bridge import RunBundle, get_value, result_series, scalar_metric, to_plain


def save_project_json(project: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_plain(project)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def load_project_json(path: str | Path) -> Any:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    try:
        from lamination_sim.models import ProjectV1

        if hasattr(ProjectV1, "model_validate"):
            return ProjectV1.model_validate(payload)
        return ProjectV1(**payload)
    except ImportError:
        return payload


def export_comparison_csv(bundle_or_comparison: Any, path: str | Path) -> Path:
    """Export aligned A/B time histories as a UTF-8 BOM CSV."""

    comparison, result_a, result_b = _unpack_results(bundle_or_comparison)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        ("time", "time_s"),
        ("position_x", "grip_x_mm"),
        ("position_y", "grip_y_mm"),
        ("position_z", "grip_z_mm"),
        ("speed", "actual_speed_mm_s"),
        ("peel_angle", "peel_angle_deg"),
        ("force_x", "force_x_n"),
        ("force_y", "force_y_n"),
        ("force_z", "force_z_n"),
        ("top_risk", "top_risk"),
        ("bottom_peel", "bottom_peel_ratio"),
        ("top_risk_area", "top_risk_area_mm2"),
        ("top_damage", "top_damage_area_mm2"),
        ("panel_lift", "panel_lift_mm"),
        ("twist", "panel_twist_mm"),
        ("force", "force_n"),
        ("moment", "moment_n_mm"),
        ("tension", "tension_n"),
        ("tape_span", "tape_span_length_mm"),
    )
    columns: dict[str, list[float]] = {}
    for condition, result in (("a", result_a), ("b", result_b)):
        for key, column in keys:
            columns[f"{column}_{condition}"] = result_series(result, key)
    tension_scenarios = list(
        get_value(comparison, "tension_scenario_results", default=[]) or []
    )
    tension_columns = (
        "tension_scenario_index",
        "tension_mode",
        "preload_label",
        "initial_preload_n",
        "stiffness_label",
        "tape_stiffness_n_per_mm",
        "p1_span_length_a_mm",
        "p1_span_length_b_mm",
        "p1_tension_a_n",
        "p1_tension_b_n",
        "final_bottom_peel_ratio_a",
        "final_bottom_peel_ratio_b",
        "bottom_gate_pass_a",
        "bottom_gate_pass_b",
        "peak_top_risk_a",
        "peak_top_risk_b",
        "max_top_risk_area_mm2_a",
        "max_top_risk_area_mm2_b",
        "top_risk_exceedance_duration_s_a",
        "top_risk_exceedance_duration_s_b",
        "final_top_damage_area_mm2_a",
        "final_top_damage_area_mm2_b",
        "max_panel_lift_mm_a",
        "max_panel_lift_mm_b",
        "max_panel_twist_mm_a",
        "max_panel_twist_mm_b",
        "max_abs_fx_n_a",
        "max_abs_fy_n_a",
        "max_abs_fz_n_a",
        "max_abs_fx_n_b",
        "max_abs_fy_n_b",
        "max_abs_fz_n_b",
        "max_tension_n_a",
        "max_tension_n_b",
        "tension_winner",
        "tension_decision_basis",
    )
    fieldnames = [*columns, *tension_columns]
    length = max(
        max((len(values) for values in columns.values()), default=0),
        len(tension_scenarios),
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(length):
            row = {
                name: values[index] if index < len(values) else ""
                for name, values in columns.items()
            }
            if index < len(tension_scenarios):
                row.update(_tension_csv_row(tension_scenarios[index]))
            writer.writerow(row)
    return destination


def export_html_report(
    project_or_bundle: Any,
    comparison_or_path: Any,
    path: str | Path | None = None,
) -> Path:
    """Write a standalone, audit-friendly HTML comparison report.

    Supported call forms are ``export_html_report(bundle, path)`` and
    ``export_html_report(project, comparison, path)``.
    """

    if isinstance(project_or_bundle, RunBundle) or hasattr(project_or_bundle, "comparison"):
        bundle = project_or_bundle
        project = get_value(bundle, "project")
        comparison = get_value(bundle, "comparison")
        result_a = get_value(bundle, "result_a") or get_value(comparison, "result_a")
        result_b = get_value(bundle, "result_b") or get_value(comparison, "result_b")
        destination = Path(comparison_or_path)
    else:
        project = project_or_bundle
        comparison = comparison_or_path
        result_a = get_value(comparison, "result_a", "simulation_a")
        result_b = get_value(comparison, "result_b", "simulation_b")
        if path is None:
            raise TypeError("path is required when project and comparison are passed separately")
        destination = Path(path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    project_plain = to_plain(project)
    comparison_plain = to_plain(comparison)
    condition_a = get_value(project, "condition_a", "a", default={})
    condition_b = get_value(project, "condition_b", "b", default={})
    winner = str(get_value(comparison, "winner", default="판정 보류"))
    verdict = str(get_value(comparison, "verdict", default=""))
    classification = str(get_value(comparison, "classification", default=""))
    metric_rows = _report_metrics(result_a, result_b)
    warnings = _warnings(comparison, result_a, result_b)
    tension_scenarios = list(
        get_value(comparison, "tension_scenario_results", default=[]) or []
    )
    report = _render_html(
        winner=winner,
        verdict=verdict,
        classification=classification,
        metric_rows=metric_rows,
        condition_a=to_plain(condition_a),
        condition_b=to_plain(condition_b),
        project=project_plain,
        comparison=comparison_plain,
        warnings=warnings,
        tension_scenarios=tension_scenarios,
        tension_summary={
            "wins_a": get_value(comparison, "tension_wins_a", default=0),
            "wins_b": get_value(comparison, "tension_wins_b", default=0),
            "ties": get_value(comparison, "tension_ties", default=0),
            "inconclusive": get_value(comparison, "tension_inconclusive", default=0),
            "a_rate": get_value(comparison, "tension_a_win_rate", default=0.0),
            "b_rate": get_value(comparison, "tension_b_win_rate", default=0.0),
            "verdict": get_value(comparison, "tension_verdict", default=""),
        },
    )
    destination.write_text(report, encoding="utf-8")
    return destination


# Friendly aliases for callers that do not need to know the original function names.
save_project = save_project_json
load_project = load_project_json
export_result_csv = export_comparison_csv
export_report_html = export_html_report


def _unpack_results(value: Any) -> tuple[Any, Any, Any]:
    comparison = get_value(value, "comparison", default=value)
    result_a = get_value(value, "result_a", default=None) or get_value(
        comparison, "result_a", "simulation_a", default=None
    )
    result_b = get_value(value, "result_b", default=None) or get_value(
        comparison, "result_b", "simulation_b", default=None
    )
    return comparison, result_a, result_b


def _tension_csv_row(scenario: Any) -> dict[str, Any]:
    force_a = list(get_value(scenario, "max_abs_force_xyz_n_a", default=[]) or [])
    force_b = list(get_value(scenario, "max_abs_force_xyz_n_b", default=[]) or [])
    force_a += [0.0] * (3 - len(force_a))
    force_b += [0.0] * (3 - len(force_b))
    return {
        "tension_scenario_index": get_value(scenario, "index"),
        "tension_mode": get_value(scenario, "mode"),
        "preload_label": get_value(scenario, "preload_label"),
        "initial_preload_n": get_value(scenario, "initial_preload_n"),
        "stiffness_label": get_value(scenario, "stiffness_label"),
        "tape_stiffness_n_per_mm": get_value(scenario, "tape_stiffness_n_per_mm"),
        "p1_span_length_a_mm": get_value(scenario, "p1_span_length_a_mm"),
        "p1_span_length_b_mm": get_value(scenario, "p1_span_length_b_mm"),
        "p1_tension_a_n": get_value(scenario, "p1_tension_a_n"),
        "p1_tension_b_n": get_value(scenario, "p1_tension_b_n"),
        "final_bottom_peel_ratio_a": get_value(scenario, "final_bottom_peel_ratio_a"),
        "final_bottom_peel_ratio_b": get_value(scenario, "final_bottom_peel_ratio_b"),
        "bottom_gate_pass_a": get_value(scenario, "bottom_gate_pass_a"),
        "bottom_gate_pass_b": get_value(scenario, "bottom_gate_pass_b"),
        "peak_top_risk_a": get_value(scenario, "peak_top_risk_a"),
        "peak_top_risk_b": get_value(scenario, "peak_top_risk_b"),
        "max_top_risk_area_mm2_a": get_value(scenario, "max_top_risk_area_mm2_a"),
        "max_top_risk_area_mm2_b": get_value(scenario, "max_top_risk_area_mm2_b"),
        "top_risk_exceedance_duration_s_a": get_value(scenario, "top_risk_exceedance_duration_s_a"),
        "top_risk_exceedance_duration_s_b": get_value(scenario, "top_risk_exceedance_duration_s_b"),
        "final_top_damage_area_mm2_a": get_value(scenario, "final_top_damage_area_mm2_a"),
        "final_top_damage_area_mm2_b": get_value(scenario, "final_top_damage_area_mm2_b"),
        "max_panel_lift_mm_a": get_value(scenario, "max_panel_lift_mm_a"),
        "max_panel_lift_mm_b": get_value(scenario, "max_panel_lift_mm_b"),
        "max_panel_twist_mm_a": get_value(scenario, "max_panel_twist_mm_a"),
        "max_panel_twist_mm_b": get_value(scenario, "max_panel_twist_mm_b"),
        "max_abs_fx_n_a": force_a[0],
        "max_abs_fy_n_a": force_a[1],
        "max_abs_fz_n_a": force_a[2],
        "max_abs_fx_n_b": force_b[0],
        "max_abs_fy_n_b": force_b[1],
        "max_abs_fz_n_b": force_b[2],
        "max_tension_n_a": get_value(scenario, "max_tension_n_a"),
        "max_tension_n_b": get_value(scenario, "max_tension_n_b"),
        "tension_winner": get_value(scenario, "winner"),
        "tension_decision_basis": get_value(scenario, "decision_basis"),
    }


def _report_metrics(result_a: Any, result_b: Any) -> list[tuple[str, str, str]]:
    specs = (
        ("Peak Rtop", ("peak_top_risk", "top_peak_risk", "peak_rtop"), ""),
        ("하면 최종 박리율", ("final_bottom_peel_ratio", "bottom_peel_ratio"), "%"),
        ("최대 패널 들림", ("max_panel_lift_mm", "panel_max_lift_mm"), " mm"),
        ("최대 패널 비틀림", ("max_panel_twist_mm", "panel_twist_mm"), " mm"),
        ("상면 최대 위험 면적", ("max_top_risk_area_mm2", "top_risk_area_mm2"), " mm²"),
        ("상면 최종 손상 면적", ("final_top_damage_area_mm2",), " mm²"),
        ("상면 임계 초과 지속시간", ("top_risk_exceedance_duration_s",), " s"),
        ("최대 하중 모멘트", ("max_moment_resultant_n_mm",), " N·mm"),
    )
    rows = []
    for label, names, unit in specs:
        first = scalar_metric(result_a, *names)
        second = scalar_metric(result_b, *names)
        if unit == "%":
            first = first * 100 if first is not None else None
            second = second * 100 if second is not None else None
        rows.append((label, _number(first, unit), _number(second, unit)))
    return rows


def _number(value: float | None, suffix: str) -> str:
    if value is None:
        return "—"
    if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
        return f"{value:.3e}{suffix}"
    return f"{value:.3f}{suffix}"


def _warnings(*objects: Any) -> list[str]:
    warnings: list[str] = []
    for obj in objects:
        candidate = get_value(obj, "warnings", default=[])
        if isinstance(candidate, str):
            candidate = [candidate]
        if isinstance(candidate, (list, tuple)):
            warnings.extend(
                str(item)
                for item in candidate
                if item and str(item).count("?") / max(1, len(str(item))) < 0.08
            )
    warnings = list(dict.fromkeys(warnings))
    if not warnings:
        warnings = [
            "이 결과는 가정 범위에서의 상대 순위이며 실제 불량률이나 안전 판정을 의미하지 않습니다.",
            "하면 필름 곡면과 대각선 박리 전선은 축약된 기구학 근사입니다.",
            "시험 폭·각도·속도가 확인되면 접착 에너지 환산값을 다시 보정해야 합니다.",
        ]
    return warnings


def _render_html(
    *,
    winner: str,
    verdict: str,
    classification: str,
    metric_rows: list[tuple[str, str, str]],
    condition_a: Any,
    condition_b: Any,
    project: Any,
    comparison: Any,
    warnings: list[str],
    tension_scenarios: list[Any],
    tension_summary: Mapping[str, Any],
) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    rows = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(a)}</td><td>{esc(b)}</td></tr>"
        for label, a, b in metric_rows
    )
    warning_html = (
        "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in warnings) + "</ul>"
        if warnings
        else "<p class='muted'>추가 경고 없음</p>"
    )
    tension_rows = "".join(
        "<tr>"
        f"<td>{esc(get_value(item, 'preload_label'))} "
        f"({float(get_value(item, 'initial_preload_n', default=0.0)):.3f} N)</td>"
        f"<td>{esc(get_value(item, 'stiffness_label'))} "
        f"({float(get_value(item, 'tape_stiffness_n_per_mm', default=0.0)):.3f} N/mm)</td>"
        f"<td>{float(get_value(item, 'final_bottom_peel_ratio_a', default=0.0))*100:.1f}% / "
        f"{float(get_value(item, 'final_bottom_peel_ratio_b', default=0.0))*100:.1f}%</td>"
        f"<td>{float(get_value(item, 'peak_top_risk_a', default=0.0)):.4g} / "
        f"{float(get_value(item, 'peak_top_risk_b', default=0.0)):.4g}</td>"
        f"<td>{float(get_value(item, 'p1_tension_a_n', default=0.0)):.3f} / "
        f"{float(get_value(item, 'p1_tension_b_n', default=0.0)):.3f} N</td>"
        f"<td>{esc(get_value(item, 'bottom_gate_pass_a', default=False))} / "
        f"{esc(get_value(item, 'bottom_gate_pass_b', default=False))}</td>"
        f"<td>{float(get_value(item, 'max_top_risk_area_mm2_a', default=0.0)):.3f} / "
        f"{float(get_value(item, 'max_top_risk_area_mm2_b', default=0.0)):.3f}</td>"
        f"<td>{float(get_value(item, 'top_risk_exceedance_duration_s_a', default=0.0)):.3f} / "
        f"{float(get_value(item, 'top_risk_exceedance_duration_s_b', default=0.0)):.3f}</td>"
        f"<td>{float(get_value(item, 'final_top_damage_area_mm2_a', default=0.0)):.3f} / "
        f"{float(get_value(item, 'final_top_damage_area_mm2_b', default=0.0)):.3f}</td>"
        f"<td>{float(get_value(item, 'max_panel_lift_mm_a', default=0.0)):.5f} / "
        f"{float(get_value(item, 'max_panel_lift_mm_b', default=0.0)):.5f}</td>"
        f"<td>{float(get_value(item, 'max_panel_twist_mm_a', default=0.0)):.5f} / "
        f"{float(get_value(item, 'max_panel_twist_mm_b', default=0.0)):.5f}</td>"
        f"<td>{esc(get_value(item, 'max_abs_force_xyz_n_a', default=[]))}<br>"
        f"{esc(get_value(item, 'max_abs_force_xyz_n_b', default=[]))}</td>"
        f"<td>{float(get_value(item, 'max_tension_n_a', default=0.0)):.3f} / "
        f"{float(get_value(item, 'max_tension_n_b', default=0.0)):.3f}</td>"
        f"<td>{esc(str(get_value(item, 'winner', default='inconclusive')).upper())}</td>"
        f"<td>{esc(get_value(item, 'decision_basis', default=''))}</td>"
        "</tr>"
        for item in tension_scenarios
    )
    tension_summary_text = (
        f"A 우세 {int(tension_summary.get('wins_a', 0))}개 "
        f"({float(tension_summary.get('a_rate', 0.0))*100:.1f}%), "
        f"B 우세 {int(tension_summary.get('wins_b', 0))}개 "
        f"({float(tension_summary.get('b_rate', 0.0))*100:.1f}%), "
        f"동률 {int(tension_summary.get('ties', 0))}개, "
        f"판정 보류 {int(tension_summary.get('inconclusive', 0))}개"
    )
    project_json = html.escape(json.dumps(project, ensure_ascii=False, indent=2))
    comparison_json = html.escape(json.dumps(comparison, ensure_ascii=False, indent=2))
    a_json = html.escape(json.dumps(condition_a, ensure_ascii=False, indent=2))
    b_json = html.escape(json.dumps(condition_b, ensure_ascii=False, indent=2))
    classification_labels = {
        "robust_a": "A가 강건하게 유리",
        "robust_b": "B가 강건하게 유리",
        "weak_a": "A가 약하게 우세",
        "weak_b": "B가 약하게 우세",
        "tie": "동률",
        "inconclusive": "판정 보류",
    }
    classification_text = classification_labels.get(classification, classification or "비교 결과")
    readable_verdict = verdict if verdict.count("?") / max(1, len(verdict)) < 0.08 else "명목 조건과 공통 불확실성 가정의 비교가 완료되었습니다."
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>필름 역박리 A/B 비교 보고서</title>
<style>
:root {{ color-scheme: dark; --bg:#0a0d12; --surface:#11161e; --line:#283241;
  --text:#f4f7fb; --muted:#99a6b8; --a:#60a5fa; --b:#f59e6a; --accent:#55d6be; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.55 "Segoe UI",system-ui,sans-serif }} main {{ max-width:1080px; margin:0 auto; padding:40px 24px 72px }}
h1 {{ font-size:26px; margin:0 0 8px }} h2 {{ font-size:16px; margin:0 0 14px }} p {{ margin:5px 0 }}
.eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase }}
.muted {{ color:var(--muted) }} .card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; margin-top:14px }}
.verdict {{ display:flex; gap:16px; align-items:center }} .badge {{ color:var(--accent); border:1px solid #55d6be66; background:#55d6be18; padding:7px 12px; border-radius:8px; font-weight:750; white-space:nowrap }}
table {{ width:100%; border-collapse:collapse }} th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:right }}
.table-scroll {{ overflow-x:auto }} .table-scroll table {{ min-width:1780px }}
th:first-child {{ text-align:left; color:var(--muted); font-weight:600 }} thead th {{ color:var(--muted) }} .a {{ color:var(--a) }} .b {{ color:var(--b) }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px }} pre {{ white-space:pre-wrap; word-break:break-word; color:#c3cede; font:12px/1.5 Consolas,monospace; max-height:480px; overflow:auto }}
details summary {{ cursor:pointer; color:var(--muted); font-weight:650 }} .notice {{ border-left:3px solid #f3c969; padding-left:12px; color:var(--muted) }}
@media(max-width:720px) {{ .grid {{ grid-template-columns:1fr }} .verdict {{ align-items:flex-start; flex-direction:column }} }}
</style>
</head>
<body><main>
<div class="eyebrow">Reverse Peel Comparator · v1</div>
<h1>필름 역박리 A/B 비교 보고서</h1>
<p class="muted">동일 가정 하에서 두 궤적의 상대 위험도를 비교한 결과입니다.</p>
<section class="card verdict"><div class="badge">{esc(winner.upper())}</div><div><h2>{esc(classification_text)}</h2><p>{esc(readable_verdict or '명목 조건 계산 완료')}</p></div></section>
<section class="card"><h2>핵심 지표</h2><table><thead><tr><th>지표</th><th class="a">조건 A</th><th class="b">조건 B</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="card"><h2>풀테이프 장력 민감도</h2><p>{esc(tension_summary_text)}</p><p class="muted">{esc(tension_summary.get('verdict', ''))}</p><div class="table-scroll"><table><thead><tr><th>초기장력</th><th>등가 인장강성</th><th>하면 A/B</th><th>Peak Rtop A/B</th><th>P1 장력 A/B</th><th>gate A/B</th><th>위험 면적 A/B (mm²)</th><th>초과시간 A/B (s)</th><th>damage A/B (mm²)</th><th>들림 A/B (mm)</th><th>비틀림 A/B (mm)</th><th>|Fx,Fy,Fz| A/B (N)</th><th>최대 장력 A/B (N)</th><th>판정</th><th>근거</th></tr></thead><tbody>{tension_rows}</tbody></table></div></section>
<section class="card"><h2>해석 경고</h2>{warning_html}<p class="notice">이 결과는 검증 전 상대 비교값이며 실제 안전·불량 또는 절대 불량률을 확정하지 않습니다.</p></section>
<section class="grid"><div class="card"><h2 class="a">조건 A 입력</h2><pre>{a_json}</pre></div><div class="card"><h2 class="b">조건 B 입력</h2><pre>{b_json}</pre></div></section>
<section class="card"><details><summary>전체 프로젝트 JSON</summary><pre>{project_json}</pre></details></section>
<section class="card"><details><summary>전체 비교 결과 JSON</summary><pre>{comparison_json}</pre></details></section>
</main></body></html>"""


__all__ = [
    "save_project_json",
    "load_project_json",
    "export_comparison_csv",
    "export_html_report",
    "save_project",
    "load_project",
    "export_result_csv",
    "export_report_html",
]
