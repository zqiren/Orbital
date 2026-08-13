# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.util

from PyInstaller.utils.hooks import collect_all

block_cipher = None
project_root = os.path.abspath(os.path.join(SPECPATH, '..', '..'))

# httptools is a C-extension package (uvicorn's default HTTP parser). PyInstaller
# 6.x ships no hook for it, so plain analysis bundles only the compiled .pyd leaves
# and drops the pure-Python package glue (httptools/__init__.py,
# httptools/parser/__init__.py). At runtime `import httptools` then resolves to a
# PEP 420 namespace package and `httptools.HttpRequestParser` is undefined, so
# uvicorn crashes on startup ("module 'httptools' has no attribute
# 'HttpRequestParser'"). collect_all() pulls in the package modules + binaries.
_httptools_datas, _httptools_binaries, _httptools_hiddenimports = collect_all('httptools')

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
    binaries=_httptools_binaries,
    datas=[
        (os.path.join(project_root, 'agent_os', 'agents', 'manifests'), 'agent_os/agents/manifests'),
        # Composition assets for Orbital-installed sub-agents (dsh): the
        # on-demand installer copies these out of the bundle at install time.
        (os.path.join(project_root, 'agent_os', 'agents', 'assets'), 'agent_os/agents/assets'),
        (os.path.join(project_root, 'agent_os', 'config'), 'agent_os/config'),
        (os.path.join(project_root, 'agent_os', 'default_skills'), 'agent_os/default_skills'),
        (os.path.join(project_root, 'agent_os', 'vendor', 'rg', 'rg.exe'), 'agent_os/vendor/rg'),
    ] + ([(_patchright_driver_dir, 'patchright/driver')] if _patchright_driver_dir else [])
      + _httptools_datas,
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
    ] + _httptools_hiddenimports,
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
