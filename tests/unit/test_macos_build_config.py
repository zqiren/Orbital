"""Validate macOS build configuration files."""

import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture
def spec_content():
    spec_path = os.path.join(PROJECT_ROOT, 'agent_os', 'desktop', 'agentos-macos.spec')
    with open(spec_path) as f:
        return f.read()


@pytest.fixture
def build_script_content():
    script_path = os.path.join(PROJECT_ROOT, 'scripts', 'build-macos.sh')
    with open(script_path) as f:
        return f.read()


class TestMacOSSpec:
    """Validate the macOS PyInstaller spec."""

    def test_spec_file_exists(self):
        spec_path = os.path.join(PROJECT_ROOT, 'agent_os', 'desktop', 'agentos-macos.spec')
        assert os.path.isfile(spec_path), "agentos-macos.spec must exist"

    def test_spec_has_bundle_directive(self, spec_content):
        assert 'BUNDLE(' in spec_content, "Spec must contain BUNDLE() for .app output"

    def test_spec_uses_icns_icon(self, spec_content):
        assert 'icon.icns' in spec_content, "Spec must reference icon.icns"
        assert 'icon.ico' not in spec_content, "Spec must not reference icon.ico (Windows icon)"

    def test_spec_has_macos_hidden_imports(self, spec_content):
        assert 'agent_os.platform.macos' in spec_content, (
            "Spec must include agent_os.platform.macos in hidden imports"
        )
        assert 'agent_os.platform.windows' not in spec_content, (
            "Spec must not include agent_os.platform.windows"
        )

    def test_spec_bundle_identifier(self, spec_content):
        assert 'bundle_identifier' in spec_content, "Spec must define a bundle_identifier"

    def test_spec_minimum_version(self, spec_content):
        assert 'LSMinimumSystemVersion' in spec_content, (
            "Spec must set LSMinimumSystemVersion in info_plist"
        )
        assert "'13.0'" in spec_content or '"13.0"' in spec_content, (
            "LSMinimumSystemVersion must be 13.0 (Ventura)"
        )


class TestMacOSBuildScript:
    """Validate the macOS build script."""

    def test_build_script_checks_macos(self, build_script_content):
        assert 'uname' in build_script_content, (
            "Build script must check for Darwin via uname"
        )

    def test_build_script_creates_dmg(self, build_script_content):
        assert 'DMG' in build_script_content or 'dmg' in build_script_content, (
            "Build script must contain DMG creation logic"
        )
        assert 'hdiutil' in build_script_content, (
            "Build script must have hdiutil fallback for DMG creation"
        )

    def test_build_script_has_code_signing_placeholder(self, build_script_content):
        assert 'codesign' in build_script_content, (
            "Build script must contain commented-out codesign commands"
        )
