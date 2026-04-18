# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.util

block_cipher = None
project_root = os.path.abspath(os.path.join(SPECPATH, '..', '..'))

# Locate patchright driver (node.exe + package/) for browser tool support.
# Without this, async_playwright().start() fails with FileNotFoundError.
_patchright_driver_dir = None
_patchright_spec = importlib.util.find_spec('patchright')
if _patchright_spec and _patchright_spec.origin:
    _candidate = os.path.join(os.path.dirname(_patchright_spec.origin), 'driver')
    if os.path.isdir(_candidate):
        _patchright_driver_dir = _candidate

a = Analysis(
    [os.path.join(project_root, 'agent_os', 'desktop', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'agent_os', 'agents', 'manifests'), 'agent_os/agents/manifests'),
        (os.path.join(project_root, 'agent_os', 'config'), 'agent_os/config'),
        (os.path.join(project_root, 'agent_os', 'default_skills'), 'agent_os/default_skills'),
        (os.path.join(project_root, 'agent_os', 'vendor', 'rg', 'rg.exe'), 'agent_os/vendor/rg'),
    ] + ([(_patchright_driver_dir, 'patchright/driver')] if _patchright_driver_dir else []),
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
        'agent_os.platform.windows',
        'agent_os.platform.windows.provider',
        'agent_os.desktop.migration',
        'agent_os.desktop.tray',
        'pystray',
        'pystray._win32',
        'PIL',
        'claude_agent_sdk',
        'claude_agent_sdk.types',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    icon=os.path.join(project_root, 'assets', 'icon.ico'),
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
