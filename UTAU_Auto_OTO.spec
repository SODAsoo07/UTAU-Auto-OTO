# -*- mode: python ; coding: utf-8 -*-

import os

import customtkinter


APP_DIR = os.path.abspath(os.path.dirname(__file__))

# ml/configs: include only the two files that are actually read at runtime.
# training-only files (dataset_build_default.yaml, lightgbm_default.yaml,
# training_data_roots.yaml) are intentionally excluded from the runtime bundle.
RUNTIME_DATA_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "assets", "models", "oto_ml"), "assets/models/oto_ml"),
    (os.path.join(APP_DIR, "assets", "models", "coarse_crnn"), "assets/models/coarse_crnn"),
    (os.path.join(APP_DIR, "assets", "bootstrap", "get-pip.py"), "assets/bootstrap"),
    (os.path.join(APP_DIR, "ml", "configs", "silence_reliability_profile.json"), "ml/configs"),
    (os.path.join(APP_DIR, "ml", "configs", "kr_vcv_anchor_profile.yaml"), "ml/configs"),
    (os.path.join(APP_DIR, "config.json"), "."),
    (os.path.join(APP_DIR, "ui", "ui_layout.json"), "ui"),
    # bundle_info.json is generated at build time; included when present.
    (os.path.join(APP_DIR, "bundle_info.json"), "."),
]

datas = [
    (os.path.dirname(customtkinter.__file__), "customtkinter/"),
]

for src, dst in RUNTIME_DATA_PATHS:
    if os.path.exists(src):
        datas.append((src, dst))


a = Analysis(
    ["main.py"],
    pathex=[APP_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=["textgrid", "customtkinter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "torchvision",
        "ml",
        "ml.scripts",
        "ml.tests",
        "pytest",
        "librosa",
        "scripts",
        "tests",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UTAU_Auto_OTO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
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
    upx=True,
    upx_exclude=[],
    name="UTAU_Auto_OTO",
)
