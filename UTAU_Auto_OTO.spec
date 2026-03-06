# -*- mode: python ; coding: utf-8 -*-

import os

import customtkinter


APP_DIR = os.path.abspath(os.path.dirname(__file__))
FFMPEG_BIN = os.path.join(APP_DIR, "build_assets", "ffmpeg", "bin")
RUNTIME_DATA_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "assets", "models"), "assets/models"),
    (os.path.join(APP_DIR, "ml", "configs"), "ml/configs"),
    (os.path.join(APP_DIR, "config.json"), "."),
]

datas = [
    (os.path.dirname(customtkinter.__file__), "customtkinter/"),
]
if os.path.isdir(FFMPEG_BIN):
    datas.append((FFMPEG_BIN, "ffmpeg/bin"))

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
    excludes=[],
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
