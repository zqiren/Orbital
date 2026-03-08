# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for PTY Reconciliation: CLIAdapter + provider wiring.

Verifies:
1. CLIAdapter provider path: start/read/send/stop/is_alive via mock provider
2. CLIAdapter fallback path: backward compatibility when provider=None
3. ProcessHandle new structure: stdin/stdout/stderr fields, _native_handles
4. NullProvider: functional run_process, stop_process, configure_network
5. SubAgentManager wiring: provider passed to CLIAdapter, no double stop_process
6. PlatformProvider.run_process() ABC contract
"""

import ast
import asyncio
import inspect
import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.adapters.base import AdapterConfig, AdapterError, OutputChunk
from agent_os.agent.adapters.cli_adapter import CLIAdapter, strip_ansi
from agent_os.platform.base import PlatformProvider
from agent_os.platform.types import ProcessHandle, NetworkRules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> AdapterConfig:
    defaults = {
        "command": "echo hello",
        "workspace": ".",
        "approval_patterns": [r"\(y/n\)"],
        "env": None,
    }
    defaults.update(overrides)
    return AdapterConfig(**defaults)


def _make_mock_provider():
    """Create a mock PlatformProvider for CLIAdapter tests."""
    provider = MagicMock()
    provider.configure_network = MagicMock()
    provider.stop_process = AsyncMock(return_value=True)
    return provider


def _make_mock_process_handle(stdout_data: bytes = b"mock output\n", alive: bool = True):
    """Create a ProcessHandle with BytesIO streams for testing."""
    stdin_buf = io.BytesIO()
    stdout_buf = io.BytesIO(stdout_data)

    handle = ProcessHandle(
        pid=9999,
        command="mock_cmd",
        stdin=stdin_buf,
        stdout=stdout_buf,
        stderr=None,
        _native_handles={
            "is_alive": lambda: alive,
        },
    )
    return handle


# ===========================================================================
# 1. CLIAdapter provider path
# ===========================================================================


class TestCLIAdapterProviderPath:
    """CLIAdapter with provider: start/read/send/stop/is_alive via provider."""

    @pytest.mark.asyncio
    async def test_start_calls_run_process_with_pty(self):
        """start() calls provider.run_process() with use_pty=True."""
        provider = _make_mock_provider()
        handle = _make_mock_process_handle()
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        config = _make_config(workspace=".")
        await adapter.start(config)

        provider.run_process.assert_awaited_once()
        call_kwargs = provider.run_process.call_args
        assert call_kwargs.kwargs.get("use_pty") is True or \
               (len(call_kwargs.args) > 5 and call_kwargs.args[5] is True)

    @pytest.mark.asyncio
    async def test_start_passes_project_id(self):
        """start() passes correct project_id to provider.run_process()."""
        provider = _make_mock_provider()
        handle = _make_mock_process_handle()
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="my_project",
        )
        await adapter.start(_make_config())

        call_kwargs = provider.run_process.call_args
        assert call_kwargs.kwargs.get("project_id") == "my_project" or \
               call_kwargs.args[0] == "my_project"

    @pytest.mark.asyncio
    async def test_send_writes_to_process_handle_stdin(self):
        """send() writes data to ProcessHandle.stdin."""
        provider = _make_mock_provider()
        stdin_buf = io.BytesIO()
        handle = ProcessHandle(
            pid=9999,
            command="mock",
            stdin=stdin_buf,
            stdout=io.BytesIO(b"data"),
            _native_handles={"is_alive": lambda: True},
        )
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        await adapter.start(_make_config())
        await adapter.send("hello world")

        stdin_buf.seek(0)
        written = stdin_buf.read()
        assert b"hello world\n" == written

    @pytest.mark.asyncio
    async def test_stop_calls_provider_stop_process(self):
        """stop() calls provider.stop_process(project_id)."""
        provider = _make_mock_provider()
        handle = _make_mock_process_handle()
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        await adapter.start(_make_config())
        await adapter.stop()

        provider.stop_process.assert_awaited_once_with("proj_1")

    @pytest.mark.asyncio
    async def test_is_alive_with_provider(self):
        """is_alive() returns True when provider process is running."""
        provider = _make_mock_provider()
        alive_flag = [True]
        handle = ProcessHandle(
            pid=9999,
            command="mock",
            stdin=io.BytesIO(),
            stdout=io.BytesIO(b"data"),
            _native_handles={"is_alive": lambda: alive_flag[0]},
        )
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        await adapter.start(_make_config())

        assert adapter.is_alive() is True

        alive_flag[0] = False
        assert adapter.is_alive() is False

    @pytest.mark.asyncio
    async def test_is_alive_false_after_stop(self):
        """is_alive() returns False after stop()."""
        provider = _make_mock_provider()
        handle = _make_mock_process_handle()
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        await adapter.start(_make_config())
        await adapter.stop()

        assert adapter.is_alive() is False

    def test_is_alive_uses_native_check_callable(self):
        """is_alive() calls _native_handles['is_alive'] callable."""
        called = []

        def mock_is_alive():
            called.append(True)
            return True

        handle = ProcessHandle(
            pid=9999,
            command="mock",
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            _native_handles={"is_alive": mock_is_alive},
        )

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=MagicMock(),
            project_id="proj_1",
        )
        adapter._proc_handle = handle

        result = adapter.is_alive()
        assert result is True
        assert len(called) == 1

    def test_provider_path_no_platform_branching(self):
        """Provider path methods must not reference _HAS_PTY."""
        import textwrap
        for method in (CLIAdapter._start_via_provider, CLIAdapter._read_provider):
            source = textwrap.dedent(inspect.getsource(method))
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "_HAS_PTY":
                    pytest.fail(
                        f"{method.__name__} references _HAS_PTY — "
                        "should be platform-agnostic"
                    )

    def test_using_provider_property(self):
        """_using_provider is True only when both provider and project_id are set."""
        adapter_both = CLIAdapter("h", "d", platform_provider=MagicMock(), project_id="p1")
        assert adapter_both._using_provider is True

        adapter_no_provider = CLIAdapter("h", "d", platform_provider=None, project_id="p1")
        assert adapter_no_provider._using_provider is False

        adapter_no_project = CLIAdapter("h", "d", platform_provider=MagicMock(), project_id=None)
        assert adapter_no_project._using_provider is False

        adapter_neither = CLIAdapter("h", "d")
        assert adapter_neither._using_provider is False

    @pytest.mark.asyncio
    async def test_read_stream_reads_from_process_handle_stdout(self):
        """read_stream() yields OutputChunks from ProcessHandle.stdout."""
        provider = _make_mock_provider()

        # Use a stdout that returns data once, then triggers loop exit
        call_count = [0]
        alive_flag = [True]

        class MockStdout:
            closed = False
            def read(self, n):
                call_count[0] += 1
                if call_count[0] == 1:
                    return b"Hello from provider\n"
                # After first read, mark process as dead to exit loop
                alive_flag[0] = False
                return b""
            def close(self):
                self.closed = True
            def flush(self):
                pass

        handle = ProcessHandle(
            pid=9999,
            command="mock",
            stdin=io.BytesIO(),
            stdout=MockStdout(),
            _native_handles={"is_alive": lambda: alive_flag[0]},
        )
        provider.run_process = AsyncMock(return_value=handle)

        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id="proj_1",
        )
        await adapter.start(_make_config())

        chunks = []
        async for chunk in adapter.read_stream():
            chunks.append(chunk)
            if "Hello from provider" in chunk.text:
                break

        assert any("Hello from provider" in c.text for c in chunks)
        await adapter.stop()


# ===========================================================================
# 2. CLIAdapter fallback path (provider is None)
# ===========================================================================


class TestCLIAdapterFallbackPath:
    """CLIAdapter without provider: backward compatibility."""

    def test_fallback_when_no_provider(self):
        """CLIAdapter works when platform_provider is None."""
        adapter = CLIAdapter(handle="test", display_name="Test")
        assert adapter._platform_provider is None
        assert adapter._using_provider is False

    def test_fallback_when_no_project_id(self):
        """CLIAdapter falls back when project_id is None even if provider is set."""
        provider = _make_mock_provider()
        adapter = CLIAdapter(
            handle="test",
            display_name="Test",
            platform_provider=provider,
            project_id=None,
        )
        assert adapter._using_provider is False

    @pytest.mark.asyncio
    async def test_start_fallback_uses_subprocess(self):
        """Fallback start() uses subprocess.Popen, not provider."""
        adapter = CLIAdapter(handle="test", display_name="Test")
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        with patch("agent_os.agent.adapters.cli_adapter.subprocess") as mock_sub:
            mock_sub.Popen.return_value = mock_process
            mock_sub.CREATE_NEW_PROCESS_GROUP = 0x200
            mock_sub.PIPE = -1

            config = _make_config()
            await adapter.start(config)

            mock_sub.Popen.assert_called_once()
            assert adapter.is_alive() is True

        await adapter.stop()

    def test_is_alive_false_without_process(self):
        """is_alive() is False when no process is started."""
        adapter = CLIAdapter(handle="test", display_name="Test")
        assert adapter.is_alive() is False


# ===========================================================================
# 3. ProcessHandle new structure
# ===========================================================================


class TestProcessHandleStructure:
    """ProcessHandle has correct fields after refactor."""

    def test_process_handle_has_io_fields(self):
        """ProcessHandle has stdin, stdout, stderr fields."""
        handle = ProcessHandle(pid=1, command="test")
        assert handle.stdin is None
        assert handle.stdout is None
        assert handle.stderr is None

    def test_process_handle_with_io(self):
        """ProcessHandle accepts file-like objects for I/O."""
        buf_in = io.BytesIO()
        buf_out = io.BytesIO(b"output")
        buf_err = io.BytesIO(b"error")

        handle = ProcessHandle(
            pid=42,
            command="test cmd",
            stdin=buf_in,
            stdout=buf_out,
            stderr=buf_err,
        )
        assert handle.stdin is buf_in
        assert handle.stdout is buf_out
        assert handle.stderr is buf_err

    def test_process_handle_native_handles(self):
        """_native_handles dict works correctly."""
        handle = ProcessHandle(
            pid=1,
            command="test",
            _native_handles={"process_handle": 12345, "thread_handle": 67890},
        )
        assert handle._native_handles["process_handle"] == 12345
        assert handle._native_handles["thread_handle"] == 67890

    def test_process_handle_native_handles_default(self):
        """_native_handles defaults to empty dict."""
        handle = ProcessHandle(pid=1, command="test")
        assert handle._native_handles == {}

    def test_process_handle_pty_mode_stderr_none(self):
        """In PTY mode, stderr is None (merged into stdout)."""
        handle = ProcessHandle(
            pid=1,
            command="test",
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            stderr=None,
        )
        assert handle.stderr is None

    def test_process_handle_no_old_fields(self):
        """ProcessHandle no longer has process_handle or thread_handle fields."""
        handle = ProcessHandle(pid=1, command="test")
        # These old fields should not exist as direct attributes
        assert not hasattr(handle, "process_handle") or handle.__class__.__dataclass_fields__.get("process_handle") is None
        assert not hasattr(handle, "thread_handle") or handle.__class__.__dataclass_fields__.get("thread_handle") is None

    def test_process_handle_repr_hides_native_handles(self):
        """_native_handles should be hidden from repr (repr=False)."""
        handle = ProcessHandle(
            pid=1,
            command="test",
            _native_handles={"secret": "value"},
        )
        repr_str = repr(handle)
        assert "secret" not in repr_str
        assert "_native_handles" not in repr_str

    def test_process_handle_backward_compat_positional(self):
        """Old positional args still work: ProcessHandle(pid, command)."""
        h = ProcessHandle(42, "ls -la")
        assert h.pid == 42
        assert h.command == "ls -la"
        assert h.stdin is None
        assert h.stdout is None
        assert h.stderr is None
        assert h._native_handles == {}

    def test_process_handle_native_handles_independent_per_instance(self):
        """Each ProcessHandle gets its own _native_handles dict (not shared)."""
        h1 = ProcessHandle(pid=1, command="a")
        h2 = ProcessHandle(pid=2, command="b")
        h1._native_handles["key"] = "value"
        assert "key" not in h2._native_handles


# ===========================================================================
# 4. NullProvider tests
# ===========================================================================


class TestNullProviderFunctional:
    """NullProvider: functional run_process, stop_process, configure_network."""

    @pytest.mark.asyncio
    async def test_null_provider_run_process_spawns_real_process(self, tmp_path):
        """NullProvider.run_process() creates a real subprocess."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle = await provider.run_process(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "import time; time.sleep(5)"],
            working_dir=str(tmp_path),
        )

        assert handle.pid > 0
        assert handle.stdout is not None
        assert handle.stdin is not None
        assert handle._native_handles.get("is_alive") is not None
        assert handle._native_handles["is_alive"]()

        await provider.stop_process("test_proj")

    @pytest.mark.asyncio
    async def test_null_provider_stop_process_terminates(self, tmp_path):
        """NullProvider.stop_process() terminates the process."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle = await provider.run_process(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            working_dir=str(tmp_path),
        )

        result = await provider.stop_process("test_proj")
        assert result is True

        # Process should be stopped
        result2 = await provider.stop_process("test_proj")
        assert result2 is False  # Already stopped

    @pytest.mark.asyncio
    async def test_null_provider_configure_network_noop(self):
        """NullProvider.configure_network() does not raise."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        # Should not raise RuntimeError anymore
        provider.configure_network(
            "test_proj",
            NetworkRules(mode="allowlist", domains=["example.com"]),
        )

    @pytest.mark.asyncio
    async def test_null_provider_setup_returns_success(self):
        """NullProvider.setup() returns success=True."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        result = await provider.setup()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_null_provider_is_setup_complete(self):
        """NullProvider.is_setup_complete() returns True."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        assert provider.is_setup_complete() is True

    @pytest.mark.asyncio
    async def test_null_provider_run_process_with_pipes(self, tmp_path):
        """NullProvider.run_process(use_pty=False) uses subprocess.PIPE."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle = await provider.run_process(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "print('hello_from_null')"],
            working_dir=str(tmp_path),
            use_pty=False,
        )

        assert handle.pid > 0
        assert handle.stdout is not None
        assert handle.stderr is not None  # PIPE mode has separate stderr

        await provider.stop_process("test_proj")

    @pytest.mark.asyncio
    async def test_null_provider_run_command(self, tmp_path):
        """NullProvider.run_command() captures output."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        result = await provider.run_command(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "print('captured_output')"],
            working_dir=str(tmp_path),
        )

        assert result.exit_code == 0
        assert "captured_output" in result.stdout

    def test_null_provider_get_capabilities(self):
        """get_capabilities() returns setup_complete=True, isolation_method='none'."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        caps = provider.get_capabilities()

        assert caps.setup_complete is True
        assert caps.isolation_method == "none"
        assert caps.platform == "null"
        assert caps.supports_network_restriction is False
        assert caps.supports_folder_access is False
        assert caps.sandbox_username is None

    @pytest.mark.asyncio
    async def test_null_provider_handle_has_working_io(self, tmp_path):
        """ProcessHandle from NullProvider has real file objects that produce data."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle = await provider.run_process(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "print('hello_from_null')"],
            working_dir=str(tmp_path),
            use_pty=False,
        )

        assert handle.stdout is not None
        assert handle.stdin is not None

        # Wait for process to produce output
        await asyncio.sleep(0.5)
        data = handle.stdout.read(4096)
        assert data is not None
        assert b"hello_from_null" in data

        await provider.stop_process("test_proj")

    @pytest.mark.asyncio
    async def test_null_provider_pipe_mode_has_all_three_streams(self, tmp_path):
        """use_pty=False gives stdin + stdout + stderr (all three non-None)."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle = await provider.run_process(
            project_id="test_proj",
            command=sys.executable,
            args=["-c", "import time; time.sleep(5)"],
            working_dir=str(tmp_path),
            use_pty=False,
        )

        assert handle.stdin is not None
        assert handle.stdout is not None
        assert handle.stderr is not None  # pipe mode gives all three

        await provider.stop_process("test_proj")

    @pytest.mark.asyncio
    async def test_null_provider_replaces_existing_process(self, tmp_path):
        """run_process() stops existing process for same project_id."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        handle1 = await provider.run_process(
            project_id="proj_1",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            working_dir=str(tmp_path),
        )
        pid1 = handle1.pid

        handle2 = await provider.run_process(
            project_id="proj_1",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            working_dir=str(tmp_path),
        )
        pid2 = handle2.pid

        # New process should have a different PID
        assert pid1 != pid2
        # Only one entry in _processes
        assert len(provider._processes) == 1

        await provider.stop_process("proj_1")

    @pytest.mark.asyncio
    async def test_null_provider_teardown_stops_processes(self, tmp_path):
        """NullProvider.teardown() stops all running processes."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        await provider.run_process(
            project_id="proj_1",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            working_dir=str(tmp_path),
        )
        await provider.run_process(
            project_id="proj_2",
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            working_dir=str(tmp_path),
        )

        result = await provider.teardown()
        assert result.success is True
        assert len(provider._processes) == 0


# ===========================================================================
# 5. SubAgentManager wiring
# ===========================================================================


class TestSubAgentManagerWiring:
    """SubAgentManager passes provider to CLIAdapter, no double stop."""

    @pytest.mark.asyncio
    async def test_start_passes_provider_to_adapter(self):
        """SubAgentManager.start() creates CLIAdapter with provider and project_id."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        provider = _make_mock_provider()
        pm = MagicMock()
        pm.start = AsyncMock()

        configs = {
            "claudecode": AdapterConfig(
                command="claude",
                workspace=".",
                approval_patterns=["Approve?"],
            )
        }

        mgr = SubAgentManager(
            process_manager=pm,
            adapter_configs=configs,
            platform_provider=provider,
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

            # Verify CLIAdapter was created with provider and project_id
            MockAdapter.assert_called_once_with(
                handle="claudecode",
                display_name="claudecode",
                platform_provider=provider,
                project_id="proj_1",
            )

    @pytest.mark.asyncio
    async def test_stop_delegates_to_adapter_only(self):
        """SubAgentManager.stop() calls adapter.stop(), not provider.stop_process() directly."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        provider = _make_mock_provider()
        pm = MagicMock()
        pm.start = AsyncMock()
        pm.stop = AsyncMock()

        configs = {
            "claudecode": AdapterConfig(
                command="claude",
                workspace=".",
                approval_patterns=[],
            )
        }

        mgr = SubAgentManager(
            process_manager=pm,
            adapter_configs=configs,
            platform_provider=provider,
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            await mgr.stop("proj_1", "claudecode")

            # adapter.stop() should be called
            mock_instance.stop.assert_awaited_once()
            # provider.stop_process() should NOT be called directly by SubAgentManager
            # (it's now handled inside CLIAdapter.stop())
            provider.stop_process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_configure_network_still_called(self):
        """Network rules are configured before adapter launch."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        provider = _make_mock_provider()
        pm = MagicMock()
        pm.start = AsyncMock()

        configs = {
            "claudecode": AdapterConfig(
                command="claude",
                workspace=".",
                approval_patterns=[],
            )
        }

        mgr = SubAgentManager(
            process_manager=pm,
            adapter_configs=configs,
            platform_provider=provider,
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")

            provider.configure_network.assert_called_once()
            call_args = provider.configure_network.call_args
            assert call_args[0][0] == "proj_1"  # project_id
            rules = call_args[0][1]
            assert rules.mode == "allowlist"

    @pytest.mark.asyncio
    async def test_start_without_provider_still_works(self):
        """SubAgentManager works when platform_provider is None."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        pm.start = AsyncMock()

        configs = {
            "aider": AdapterConfig(
                command="aider",
                workspace=".",
                approval_patterns=[],
            )
        }

        mgr = SubAgentManager(
            process_manager=pm,
            adapter_configs=configs,
            platform_provider=None,
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            MockAdapter.return_value = mock_instance

            result = await mgr.start("proj_1", "aider")
            assert "Started" in result

            MockAdapter.assert_called_once_with(
                handle="aider",
                display_name="aider",
                platform_provider=None,
                project_id="proj_1",
            )


# ===========================================================================
# 6. PlatformProvider ABC signature
# ===========================================================================


class TestPlatformProviderABC:
    """PlatformProvider.run_process() has use_pty parameter."""

    def test_run_process_accepts_use_pty(self):
        """run_process() signature includes use_pty parameter."""
        sig = inspect.signature(PlatformProvider.run_process)
        assert "use_pty" in sig.parameters
        assert sig.parameters["use_pty"].default is False

    def test_windows_provider_run_process_accepts_use_pty(self):
        """WindowsPlatformProvider.run_process() signature includes use_pty."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        sig = inspect.signature(WindowsPlatformProvider.run_process)
        assert "use_pty" in sig.parameters

    def test_null_provider_run_process_accepts_use_pty(self):
        """NullProvider.run_process() signature includes use_pty."""
        from agent_os.platform.null import NullProvider
        sig = inspect.signature(NullProvider.run_process)
        assert "use_pty" in sig.parameters

    def test_abc_run_process_returns_process_handle(self):
        """run_process() return annotation is ProcessHandle."""
        sig = inspect.signature(PlatformProvider.run_process)
        assert sig.return_annotation is ProcessHandle

    def test_abc_run_process_required_params(self):
        """run_process() has required params: project_id, command, args, working_dir."""
        sig = inspect.signature(PlatformProvider.run_process)
        params = sig.parameters

        for name in ("project_id", "command", "args", "working_dir"):
            assert name in params, f"run_process() missing required parameter: {name}"
            assert params[name].default is inspect.Parameter.empty, \
                f"Parameter {name} should not have a default value"

    def test_abc_stop_process_is_abstract(self):
        """stop_process() is an abstract method on PlatformProvider."""
        assert hasattr(PlatformProvider.stop_process, "__isabstractmethod__")
        assert PlatformProvider.stop_process.__isabstractmethod__ is True

    def test_null_provider_is_concrete_subclass(self):
        """NullProvider is a concrete (non-abstract) subclass of PlatformProvider."""
        from agent_os.platform.null import NullProvider

        assert issubclass(NullProvider, PlatformProvider)
        # Should be instantiable
        provider = NullProvider()
        assert isinstance(provider, PlatformProvider)
