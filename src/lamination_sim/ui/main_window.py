"""Main PySide6 desktop window."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lamination_sim.exports import (
    export_comparison_csv,
    export_html_report,
    load_project_json,
    save_project_json,
)

from .condition_editor import ConditionEditor
from .core_bridge import RunBundle, build_project, get_value, sequence, to_plain
from .results_view import ResultsView
from .theme import COLORS, apply_theme
from .tension_settings import TensionSettingsDialog
from .worker import SimulationWorker


APP_TITLE = "Reverse Peel Comparator"
PROJECT_FILTER = "박리 비교 프로젝트 (*.peel.json *.json);;JSON (*.json)"


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_TITLE} · 필름 역박리 A/B 비교")
        self.setMinimumSize(1100, 760)
        self.resize(1440, 920)
        self._project_path: Path | None = None
        self._assumptions: Any = {}
        self._tension_sweep: Any = {}
        self._bundle: RunBundle | None = None
        self._dirty = False
        self._syncing = False
        self._thread: QThread | None = None
        self._worker: SimulationWorker | None = None

        self._create_actions()
        self._create_menu()
        self._build_ui()
        self.statusBar().showMessage("준비됨")
        self._load_default_project()

    def _create_actions(self) -> None:
        self.new_action = QAction("새 프로젝트", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self.new_project)
        self.open_action = QAction("열기…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_project)
        self.save_action = QAction("저장", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_project)
        self.save_as_action = QAction("다른 이름으로 저장…", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        self.export_csv_action = QAction("결과 CSV…", self)
        self.export_csv_action.triggered.connect(self.export_csv)
        self.export_html_action = QAction("HTML 보고서…", self)
        self.export_html_action.triggered.connect(self.export_html)
        self.quit_action = QAction("끝내기", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)
        self.run_action = QAction("A/B 비교 실행", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+Return"))
        self.run_action.triggered.connect(self.run_comparison)

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("파일")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_csv_action)
        file_menu.addAction(self.export_html_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        run_menu = self.menuBar().addMenu("해석")
        run_menu.addAction(self.run_action)
        help_menu = self.menuBar().addMenu("도움말")
        about_action = QAction("이 프로그램에 관하여", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 12)
        root.setSpacing(11)
        root.addWidget(self._build_header())

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        root.addWidget(self.progress_bar)

        self.tabs = QTabWidget()
        self.setup_page = self._build_setup_page()
        self.results_page = ResultsView()
        self.results_page.tension_mode_requested.connect(
            self._set_tension_mode_from_results
        )
        self.tabs.addTab(self.setup_page, "조건 입력")
        self.tabs.addTab(self.results_page, "비교 결과")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

    def _build_header(self) -> QFrame:
        header = QFrame()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(9)
        brand = QLabel("RP")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedSize(38, 38)
        brand.setStyleSheet(
            f"background:{COLORS['accent_dark']}; color:{COLORS['accent']}; "
            f"border:1px solid {COLORS['accent']}55; border-radius:10px; font-weight:800;"
        )
        titles = QWidget()
        title_layout = QVBoxLayout(titles)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        title = QLabel("필름 역박리 A/B 비교")
        title.setProperty("heading", True)
        subtitle = QLabel("동일한 물리 가정에서 두 박리 궤적의 상대 위험을 비교합니다")
        subtitle.setProperty("muted", True)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addWidget(brand)
        layout.addWidget(titles)
        layout.addStretch(1)

        open_button = QPushButton("열기")
        open_button.clicked.connect(self.open_project)
        save_button = QPushButton("저장")
        save_button.clicked.connect(self.save_project)
        self.export_button = QToolButton()
        self.export_button.setText("내보내기")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self.export_button)
        export_menu.addAction(self.export_csv_action)
        export_menu.addAction(self.export_html_action)
        self.export_button.setMenu(export_menu)
        self.run_button = QPushButton("A/B 비교 실행")
        self.run_button.setProperty("primary", True)
        self.run_button.setMinimumWidth(140)
        self.run_button.clicked.connect(self.run_comparison)
        layout.addWidget(open_button)
        layout.addWidget(save_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.run_button)
        return header

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(9)

        controls = QFrame()
        controls.setProperty("card", True)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(10)
        self.link_inputs = QCheckBox("패널·필름·풀테이프 입력 연결")
        self.link_inputs.setChecked(True)
        self.link_inputs.setToolTip("켜면 조건 A의 비궤적 입력을 조건 B에 동일하게 적용합니다.")
        self.link_inputs.toggled.connect(self._on_link_toggled)
        copy_button = QPushButton("A 입력을 B에 복사")
        copy_button.clicked.connect(self._copy_shared_inputs)
        self.tension_enabled = QCheckBox("풀테이프 장력 민감도")
        self.tension_enabled.setChecked(True)
        self.tension_enabled.setToolTip("초기장력 × 등가 인장강성 조합을 A/B에 paired 적용합니다.")
        self.tension_enabled.toggled.connect(self._on_tension_enabled)
        self.tension_mode = QComboBox()
        self.tension_mode.addItem("Equal preload", "equal_preload")
        self.tension_mode.addItem("Shared rest length", "shared_rest_length")
        self.tension_mode.currentIndexChanged.connect(self._on_tension_mode_changed)
        tension_settings = QPushButton("장력 고급 설정…")
        tension_settings.clicked.connect(self._open_tension_settings)
        self.run_uncertainty = QCheckBox("재료 불확실성")
        self.run_uncertainty.setChecked(False)
        self.run_uncertainty.setToolTip("PSA·패널 강성·시험 폭·각도 가정을 paired 적용합니다.")
        self.run_uncertainty.toggled.connect(self._mark_dirty)
        self.run_uncertainty.toggled.connect(self._update_run_estimate)
        self.run_estimate = QLabel()
        self.run_estimate.setProperty("dim", True)
        controls_layout.addWidget(self.link_inputs)
        controls_layout.addWidget(copy_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.tension_enabled)
        controls_layout.addWidget(self.tension_mode)
        controls_layout.addWidget(tension_settings)
        controls_layout.addWidget(self.run_uncertainty)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.run_estimate)
        layout.addWidget(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        editors = QWidget()
        editors_layout = QHBoxLayout(editors)
        editors_layout.setContentsMargins(1, 1, 7, 8)
        editors_layout.setSpacing(10)
        self.editor_a = ConditionEditor("A", COLORS["a"])
        self.editor_b = ConditionEditor("B", COLORS["b"])
        self.editor_a.changed.connect(self._on_a_changed)
        self.editor_b.changed.connect(self._on_b_changed)
        editors_layout.addWidget(self.editor_a, 1)
        editors_layout.addWidget(self.editor_b, 1)
        scroll.setWidget(editors)
        layout.addWidget(scroll, 1)
        return page

    def _load_default_project(self) -> None:
        try:
            from lamination_sim.presets import default_project

            project = default_project()
            self._set_project(project)
        except Exception:
            # The editor owns safe defaults, so the UI can still be inspected when
            # only the presentation package is installed.
            self._assumptions = {}
            self._dirty = False
            self._update_title()

    def _set_project(self, project: Any) -> None:
        self._syncing = True
        try:
            self.editor_a.set_condition(get_value(project, "condition_a", "a", default={}))
            self.editor_b.set_condition(get_value(project, "condition_b", "b", default={}))
            self._assumptions = get_value(project, "assumptions", "assumption_set", default={})
            self._tension_sweep = get_value(project, "tension_sweep", default={})
            self.tension_enabled.setChecked(
                bool(get_value(self._tension_sweep, "enabled", default=True))
            )
            mode = str(get_value(self._tension_sweep, "mode", default="equal_preload"))
            self.tension_mode.setCurrentIndex(max(0, self.tension_mode.findData(mode)))
            self.run_uncertainty.setChecked(
                bool(get_value(project, "run_uncertainty", default=False))
            )
        finally:
            self._syncing = False
        self._bundle = None
        self._dirty = False
        self._update_title()
        self.tabs.setCurrentWidget(self.setup_page)
        self._update_run_estimate()

    def _project_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "condition_a": self.editor_a.condition_payload(),
            "condition_b": self.editor_b.condition_payload(),
            "assumptions": to_plain(self._assumptions),
            "tension_sweep": to_plain(self._tension_sweep),
            "run_uncertainty": self.run_uncertainty.isChecked(),
        }

    def _on_tension_enabled(self, checked: bool) -> None:
        payload = to_plain(self._tension_sweep) or {}
        payload["enabled"] = bool(checked)
        self._tension_sweep = payload
        self.tension_mode.setEnabled(checked)
        self._update_run_estimate()
        self._mark_dirty()

    def _on_tension_mode_changed(self, _index: int) -> None:
        if self._syncing:
            return
        payload = to_plain(self._tension_sweep) or {}
        payload["mode"] = str(self.tension_mode.currentData())
        self._tension_sweep = payload
        self._update_run_estimate()
        self._mark_dirty()

    def _set_tension_mode_from_results(self, mode: str) -> None:
        index = self.tension_mode.findData(mode)
        if index < 0 or index == self.tension_mode.currentIndex():
            return
        self.tension_mode.setCurrentIndex(index)
        self.tabs.setCurrentWidget(self.setup_page)
        self.statusBar().showMessage("장력 모드가 변경되었습니다. 다시 실행해 결과를 갱신하세요.", 6000)

    def _open_tension_settings(self) -> None:
        samples = int(get_value(self._assumptions, "uncertainty_samples", default=24))
        dialog = TensionSettingsDialog(
            self._tension_sweep,
            run_material_uncertainty=self.run_uncertainty.isChecked(),
            material_samples=samples,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._tension_sweep = dialog.payload()
        self._syncing = True
        try:
            self.tension_enabled.setChecked(True)
            self.tension_mode.setCurrentIndex(
                max(0, self.tension_mode.findData(self._tension_sweep["mode"]))
            )
        finally:
            self._syncing = False
        self._update_run_estimate()
        self._mark_dirty()

    def _update_run_estimate(self, *_args) -> None:
        preloads = sequence(get_value(self._tension_sweep, "preload_levels", default=[]))
        stiffnesses = sequence(get_value(self._tension_sweep, "stiffness_levels", default=[]))
        combinations = max(1, len(preloads)) * max(1, len(stiffnesses))
        samples = int(get_value(self._assumptions, "uncertainty_samples", default=24))
        nested = bool(
            get_value(self._tension_sweep, "nest_material_uncertainty", default=False)
        )
        if self.run_uncertainty.isChecked() and nested:
            total = combinations * samples * 2
        else:
            total = combinations * 2
            if self.run_uncertainty.isChecked():
                total += max(samples - 1, 0) * 2
        self.run_estimate.setText(f"예상 {total}회")

    def current_project(self) -> Any:
        return build_project(self._project_payload())

    def _on_a_changed(self) -> None:
        if self._syncing:
            return
        if self.link_inputs.isChecked():
            self._copy_shared_inputs()
        self._mark_dirty()

    def _on_b_changed(self) -> None:
        if not self._syncing:
            self._mark_dirty()

    def _copy_shared_inputs(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.editor_b.set_shared_inputs(self.editor_a)
        except ValueError:
            pass
        finally:
            self._syncing = False
        self._mark_dirty()

    def _on_link_toggled(self, checked: bool) -> None:
        if checked:
            self._copy_shared_inputs()
        self._mark_dirty()

    def _mark_dirty(self, *_args) -> None:
        if self._syncing:
            return
        self._dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self._project_path.name if self._project_path else "새 프로젝트"
        marker = " •" if self._dirty else ""
        self.setWindowTitle(f"{name}{marker} — {APP_TITLE}")

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._project_path = None
        self._load_default_project()
        self.statusBar().showMessage("새 프로젝트를 만들었습니다.", 4000)

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        filename, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "", PROJECT_FILTER)
        if not filename:
            return
        try:
            project = load_project_json(filename)
            self._set_project(project)
        except Exception as exc:
            self._show_error("프로젝트를 열 수 없습니다", str(exc))
            return
        self._project_path = Path(filename)
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"{filename}을 열었습니다.", 5000)

    def save_project(self, _checked: bool = False, *, save_as: bool = False) -> bool:
        path = self._project_path
        if path is None or save_as:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "프로젝트 저장",
                str(path or Path.cwd() / "comparison.peel.json"),
                PROJECT_FILTER,
            )
            if not filename:
                return False
            path = Path(filename)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".peel.json")
        try:
            project = self.current_project()
            save_project_json(project, path)
        except Exception as exc:
            self._show_error("프로젝트를 저장할 수 없습니다", str(exc))
            return False
        self._project_path = path
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"{path}에 저장했습니다.", 5000)
        return True

    def run_comparison(self) -> None:
        if self._thread is not None:
            return
        try:
            project = self.current_project()
        except Exception as exc:
            self.tabs.setCurrentWidget(self.setup_page)
            self._show_error("입력값을 확인해주세요", str(exc))
            return

        self._set_busy(True, "계산 준비 중…")
        thread = QThread(self)
        worker = SimulationWorker(project)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(lambda message: self.statusBar().showMessage(message))
        worker.finished.connect(self._on_run_finished)
        worker.failed.connect(self._on_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_run_finished(self, bundle: RunBundle) -> None:
        self._bundle = bundle
        self.results_page.set_bundle(bundle)
        self.tabs.setCurrentWidget(self.results_page)
        self.statusBar().showMessage("A/B 비교 계산이 완료되었습니다.", 6000)
        self._set_busy(False)

    def _on_run_failed(self, message: str, details: str) -> None:
        self._set_busy(False)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("해석 실패")
        dialog.setText("시뮬레이션을 완료하지 못했습니다.")
        dialog.setInformativeText(message)
        dialog.setDetailedText(details)
        dialog.exec()
        self.statusBar().showMessage("해석 실패 — 입력과 상세 오류를 확인해주세요.", 6000)

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.run_button.setEnabled(not busy)
        self.run_action.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        if busy:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.run_button.setText("계산 중…")
            if message:
                self.statusBar().showMessage(message)
        else:
            self.progress_bar.hide()
            self.progress_bar.setRange(0, 100)
            self.run_button.setText("A/B 비교 실행")

    def export_csv(self) -> None:
        if self._bundle is None:
            self._show_notice("내보낼 결과가 없습니다", "먼저 A/B 비교를 실행해주세요.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "결과 CSV 내보내기",
            str(Path.cwd() / "comparison_result.csv"),
            "CSV (*.csv)",
        )
        if not filename:
            return
        try:
            destination = export_comparison_csv(self._bundle, filename)
        except Exception as exc:
            self._show_error("CSV를 내보낼 수 없습니다", str(exc))
            return
        self.statusBar().showMessage(f"{destination}에 결과를 내보냈습니다.", 5000)

    def export_html(self) -> None:
        if self._bundle is None:
            self._show_notice("내보낼 결과가 없습니다", "먼저 A/B 비교를 실행해주세요.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "HTML 보고서 내보내기",
            str(Path.cwd() / "comparison_report.html"),
            "HTML (*.html)",
        )
        if not filename:
            return
        try:
            destination = export_html_report(self._bundle, filename)
        except Exception as exc:
            self._show_error("HTML 보고서를 내보낼 수 없습니다", str(exc))
            return
        self.statusBar().showMessage(f"{destination}에 보고서를 내보냈습니다.", 5000)

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "저장하지 않은 변경",
            "변경 내용을 저장한 뒤 계속할까요?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._thread is not None:
            QMessageBox.information(
                self,
                "계산 진행 중",
                "현재 비교 계산이 끝난 후 프로그램을 종료해주세요.",
            )
            event.ignore()
            return
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            APP_TITLE,
            "<b>Reverse Peel Comparator v1</b><br><br>"
            "양면 필름 공정에서 두 박리 궤적의 상면 역박리 위험을 "
            "동일한 가정으로 비교하는 물리 기반 도구입니다.<br><br>"
            "결과는 검증 전 상대 비교값이며 안전·불량을 확정하지 않습니다.",
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_notice(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)


def run_app(argv: list[str] | None = None) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("Lamination Engineering")
    apply_theme(app)
    window = MainWindow()
    window.show()
    if owns_app:
        return app.exec()
    return 0


__all__ = ["MainWindow", "run_app"]
