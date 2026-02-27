# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\oyh57\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\customtkinter', 'customtkinter/'), ('C:\\Users\\oyh57\\SODAsoo1\\Devs\\UTAU_Auto_OTO_v3\\Auto_OTO\\build_assets\\ffmpeg\\bin', 'ffmpeg/bin')],
    hiddenimports=['textgrid', 'customtkinter'],
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
    a.binaries,
    a.datas,
    [],
    name='UTAU_Auto_OTO',
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
