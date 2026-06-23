import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).parents[1]
API_ROOT = ROOT / "python" / "local_api"
CORE_ROOT = ROOT / "python" / "teacherlm_core"
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(CORE_ROOT))

datas = [(str(ROOT / "generators_registry.json"), ".")]
binaries = []
hiddenimports = collect_submodules("local_api") + collect_submodules("teacherlm_core")

for package in (
    "fastembed",
    "onnxruntime",
    "sherpa_onnx",
    "llama_cloud",
    "pydantic",
    "pydantic_settings",
    "uvicorn",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

analysis = Analysis(
    [str(Path(SPECPATH) / "sidecar_entry.py")],
    pathex=[str(API_ROOT), str(CORE_ROOT), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="teacherlm-local-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="teacherlm-local-api",
)
