# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
import sys

block_cipher = None
project_root = os.path.abspath(os.path.join(SPECPATH, '..', '..'))

# Locate patchright driver for browser tool support.
# Without this, async_playwright().start() fails with FileNotFoundError.
_patchright_datas = []
_pr_spec = importlib.util.find_spec("patchright")
if _pr_spec and _pr_spec.submodule_search_locations:
    _pr_pkg = _pr_spec.submodule_search_locations[0]
    _driver = os.path.join(_pr_pkg, "driver")
    if os.path.isdir(_driver):
        _patchright_datas.append((_driver, "patchright/driver"))

a = Analysis(
    [os.path.join(project_root, 'agent_os', 'desktop', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'agent_os', 'agents', 'manifests'), 'agent_os/agents/manifests'),
        (os.path.join(project_root, 'agent_os', 'config'), 'agent_os/config'),
    ] + _patchright_datas,
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'agent_os.api.app',
        'agent_os.platform.macos',
        'agent_os.platform.macos.provider',
        'agent_os.platform.macos.sandbox',
        'agent_os.desktop.migration',
        'agent_os.desktop.tray',
        'multipart',
        'claude_agent_sdk',
        'claude_agent_sdk.types',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'IPython'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Orbital',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(project_root, 'assets', 'icon.icns'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Orbital',
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name='Orbital.app',
    icon=os.path.join(project_root, 'assets', 'icon.icns'),
    bundle_identifier='com.orbital.desktop',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '13.0',
        'CFBundleDisplayName': 'Orbital',
        'NSHumanReadableCopyright': 'Copyright © 2026 Orbital. GPL-3.0.',
    },
)
