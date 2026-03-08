# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transport abstraction layer for agent communication."""
from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.transports.pipe_transport import PipeTransport
from agent_os.agent.transports.pty_transport import PTYTransport
from agent_os.agent.transports.acp_transport import ACPTransport

__all__ = [
    "AgentTransport", "TransportEvent",
    "PipeTransport",
    "PTYTransport",
    "ACPTransport",
]

try:
    from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
    __all__.extend(["SDKTransport", "HAS_SDK"])
except ImportError:
    HAS_SDK = False
