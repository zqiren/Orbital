# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapter abstract base classes and shared types for Component D."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class AdapterConfig:
    command: str
    workspace: str
    approval_patterns: list[str]
    env: dict[str, str] | None = None
    args: list[str] | None = None


@dataclass
class OutputChunk:
    text: str
    chunk_type: str  # "response" | "tool_activity" | "approval_request" | "status"
    timestamp: str = ""  # ISO 8601
    metadata: dict = field(default_factory=dict)  # transport-specific metadata


class AdapterError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AgentAdapter(ABC):
    agent_type: str
    handle: str
    display_name: str

    @abstractmethod
    async def start(self, config: AdapterConfig) -> None: ...

    @abstractmethod
    async def send(self, message: str) -> None: ...

    @abstractmethod
    async def read_stream(self) -> AsyncIterator[OutputChunk]: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def is_idle(self) -> bool: ...

    @abstractmethod
    def is_alive(self) -> bool: ...
