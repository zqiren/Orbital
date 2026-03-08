# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes for the tool suite."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result returned by every tool. Tools NEVER raise — errors go in content.

    content can be:
    - str: plain text (default, backward compatible)
    - list[dict]: multimodal content blocks (e.g. text + image for vision models)
    """
    content: str | list
    meta: dict | None = None


class Tool(ABC):
    """Abstract base class for all tools."""
    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    def execute(self, **arguments) -> ToolResult:
        """Execute the tool. NEVER raises. Returns ToolResult with error string in content on failure."""

    def schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
