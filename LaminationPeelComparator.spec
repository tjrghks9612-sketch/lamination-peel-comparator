# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the Windows desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPECPATH).resolve()

# The UI is implemented with Qt Widgets and QPainter. Keep the hidden-import
# list deliberately narrow so unrelated Qt Quick/QML/3D runtimes are not
# dragged into the onedir distributable.
qt_hidden_imports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

datas = collect_data_files("lamination_sim")

a = Analysis(
    [str(project_root / "src" / "lamination_sim" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=qt_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pyvista", "vtk", "torch", "jax", "PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LaminationPeelComparator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LaminationPeelComparator",
)
