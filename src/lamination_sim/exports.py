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
    )
    columns: dict[str, list[float]] = {}
    for condition, result in (("a", result_a), ("b", result_b)):
        for key, column in keys:
            columns[f"{column}_{condition}"] = result_series(result, key)
    length = max((len(values) for values in columns.values()), default=0)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for index in range(length):
            writer.writerow(
                {
                    name: values[index] if index < len(values) else ""
                    for name, values in columns.items()
                }
            )
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
