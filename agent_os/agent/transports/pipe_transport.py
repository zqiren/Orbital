# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipe-mode transport: spawn subprocess per message, capture output."""
import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field

from agent_os.agent.transports.base import AgentTransport, TransportEvent

logger = logging.getLogger(__name__)


@dataclass
class PipeTransportConfig:
    prompt_flag: str = "-p"
    resume_flag: str = "--resume"
    session_id_pattern: str = ""
    output_format_args: list[str] = field(default_factory=list)


CLAUDE_CODE_PIPE_CONFIG = PipeTransportConfig(
    prompt_flag="-p",
    resume_flag="--resume",
    session_id_pattern=r'"session_id":\s*"([^"]+)"',
    output_format_args=["--output-format", "stream-json", "--verbose"],
)


class PipeTransport(AgentTransport):
    """Pipe transport: spawn process per message, capture stdout."""

    def __init__(
        self,
        config: PipeTransportConfig | None = None,
        system_prompt: str | None = None,
        agent_slug: str | None = None,
    ):
        self._config = config or PipeTransportConfig()
        self._command: str = ""
        self._args: list[str] = []
        self._workspace: str = ""
        self._env: dict | None = None
        self._session_id: str | None = None
        self._running: bool = False
        # Optional system prompt. When set, written to a temp file under
        # {workspace}/orbital/.tmp/ on each send() and passed via
        # --append-system-prompt-file. Removed in finally.
        # We use the -file variant (not inline) because Windows OS argv
        # length limits (~32KB) are smaller than realistic sub-agent
        # context prompts. See FINDINGS-sub-agent-context-and-persistence.md
        # (Q7): -file is undocumented in `claude --help` but works on
        # claude-code 2.1.138.
        self._system_prompt: str | None = system_prompt
        self._agent_slug: str = agent_slug or "agent"

    async def start(self, command: str, args: list[str], workspace: str, env: dict | None = None) -> None:
        self._command = command
        self._args = list(args)
        self._workspace = workspace
        self._env = env

    def _write_system_prompt_tempfile(self) -> str | None:
        """Write self._system_prompt to a fresh temp file. Return abs path.

        The file is created under {workspace}/orbital/.tmp/ with a
        unique-per-dispatch suffix so concurrent dispatches do not race.
        Returns None if no prompt is set.
        """
        if not self._system_prompt or not self._workspace:
            return None
        from agent_os.agent.project_paths import ProjectPaths

        tmp_dir = ProjectPaths(self._workspace).tmp_dir
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                "PipeTransport: failed to create tmp dir %s: %s",
                tmp_dir, e,
            )
            return None
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", self._agent_slug or "agent")
        path = os.path.join(
            tmp_dir,
            f"system_prompt_{slug}_{uuid.uuid4().hex}.txt",
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._system_prompt)
        except OSError as e:
            logger.warning(
                "PipeTransport: failed to write system prompt to %s: %s",
                path, e,
            )
            return None
        return path

    async def send(self, message: str) -> str | None:
        args = list(self._args)
        if self._config.prompt_flag:
            args.extend([self._config.prompt_flag, message])
        if self._session_id and self._config.resume_flag:
            args.extend([self._config.resume_flag, self._session_id])

        # Inject system prompt via --append-system-prompt-file <abs-path>.
        # Cleaned up after dispatch in the finally block below.
        system_prompt_path: str | None = self._write_system_prompt_tempfile()
        if system_prompt_path is not None:
            args.extend(["--append-system-prompt-file", system_prompt_path])

        cmd = [self._command] + args
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        if self._env:
            env.update(self._env)

        self._running = True
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self._workspace, env=env, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return "Error: sub-agent timed out after 5 minutes"
        finally:
            self._running = False
            if system_prompt_path is not None:
                try:
                    os.remove(system_prompt_path)
                except OSError:
                    # Cleanup is best-effort — leftover temp files in
                    # .tmp/ are harmless.
                    pass

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        if result.returncode != 0 and not stdout:
            return f"Error: sub-agent failed (exit {result.returncode}): {stderr}"

        if self._config.session_id_pattern:
            match = re.search(self._config.session_id_pattern, stdout)
            if match:
                self._session_id = match.group(1)

        return self._parse_output(stdout)

    def _parse_output(self, raw: str) -> str:
        lines = raw.strip().split("\n")
        response_parts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "assistant":
                    message = obj.get("message", {})
                    for block in message.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            response_parts.append(block["text"])
                        elif isinstance(block, dict) and block.get("text"):
                            response_parts.append(block["text"])
                elif obj.get("type") == "tool_use":
                    tool_name = obj.get("name", "unknown")
                    response_parts.append(f"[Using tool: {tool_name}]")
                elif obj.get("type") == "result":
                    if obj.get("subtype") == "error":
                        response_parts.append(f"Error: {obj.get('error', 'unknown')}")
            except json.JSONDecodeError:
                if line and not line.startswith("\x1b"):
                    response_parts.append(line)
        return "\n".join(response_parts) if response_parts else "(no response)"

    async def read_stream(self):
        return
        yield  # make this a generator

    async def stop(self) -> None:
        self._session_id = None

    def is_alive(self) -> bool:
        return self._running

    @property
    def session_id(self) -> str | None:
        return self._session_id
