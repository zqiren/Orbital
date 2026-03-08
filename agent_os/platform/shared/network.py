# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
HTTP/HTTPS forward proxy with domain-level filtering.

One proxy instance per project. Pure asyncio, no external dependencies.
Filters traffic at the domain level using allowlist/denylist rules.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from agent_os.platform.types import BlockedCallback, NetworkRules

logger = logging.getLogger("agent_os.platform.shared.network")

_MAX_FIRST_LINE = 8192  # reject first lines longer than 8 KB


class NetworkProxy:
    """Async HTTP/HTTPS forward proxy with domain-level filtering."""

    def __init__(
        self,
        project_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
        on_blocked: BlockedCallback | None = None,
    ) -> None:
        self._project_id = project_id
        self._host = host
        self._requested_port = port
        self._on_blocked = on_blocked

        self._rules: NetworkRules | None = None
        self._server: asyncio.AbstractServer | None = None
        self._port: int | None = None
        self._active_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the proxy server."""
        if self._server is not None:
            raise RuntimeError("Proxy is already running")

        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._requested_port
        )
        # Retrieve the actual bound port (important when port=0).
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        logger.info(
            "NetworkProxy started for project %s on %s:%d",
            self._project_id,
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        """Stop the proxy server and cancel all active connections."""
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None

        # Cancel all active connection tasks.
        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

        logger.info(
            "NetworkProxy stopped for project %s (was on port %d)",
            self._project_id,
            self._port,
        )
        self._port = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        """Return the actual bound port. Raises if not started."""
        if self._port is None:
            raise RuntimeError("Proxy has not been started")
        return self._port

    @property
    def proxy_url(self) -> str:
        """Return the proxy URL (e.g. http://127.0.0.1:12345)."""
        return f"http://{self._host}:{self.port}"

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def set_rules(self, rules: NetworkRules) -> None:
        """Update filtering rules. Thread-safe (GIL atomic reference assign)."""
        self._rules = rules
        logger.info(
            "Rules updated for project %s: mode=%s, %d domains",
            self._project_id,
            rules.mode,
            len(rules.domains),
        )

    def get_rules(self) -> NetworkRules | None:
        """Return current rules, or None if not set."""
        return self._rules

    # ------------------------------------------------------------------
    # Domain matching
    # ------------------------------------------------------------------

    def _is_allowed(self, domain: str) -> bool:
        """Check whether *domain* is allowed under current rules."""
        rules = self._rules
        if rules is None:
            return True  # no rules → allow all

        domain_lower = domain.lower()

        if rules.mode == "allowlist":
            return self._matches_any(domain_lower, rules.domains)
        else:  # denylist
            return not self._matches_any(domain_lower, rules.domains)

    @staticmethod
    def _matches_any(domain: str, patterns: list[str]) -> bool:
        """Return True if *domain* matches at least one pattern."""
        for pattern in patterns:
            pat = pattern.lower()
            if pat.startswith("*."):
                # Wildcard: *.github.com matches api.github.com
                # but NOT github.com itself.
                suffix = pat[1:]  # ".github.com"
                if domain.endswith(suffix) and domain != suffix.lstrip("."):
                    return True
            else:
                if domain == pat:
                    return True
        return False

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Entry point for every incoming connection from asyncio.start_server."""
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        try:
            await self._process_client(reader, writer)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Unhandled error in client handler", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if task is not None:
                self._active_tasks.discard(task)

    async def _process_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Process a single client connection."""
        # Read the first line of the request.
        try:
            first_line_raw = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            first_line_raw = exc.partial
            if not first_line_raw:
                return  # client disconnected immediately
        except asyncio.LimitOverrunError:
            # Line too long.
            self._send_error(writer, 400, "Request line too long")
            return
        except ConnectionError:
            return

        if len(first_line_raw) > _MAX_FIRST_LINE:
            self._send_error(writer, 400, "Request line too long")
            return

        first_line = first_line_raw.decode("latin-1", errors="replace").strip()
        if not first_line:
            return

        parts = first_line.split()
        if len(parts) < 2:
            self._send_error(writer, 400, "Malformed request")
            return

        method = parts[0].upper()
        target = parts[1]

        if method == "CONNECT":
            await self._handle_connect(first_line, target, reader, writer)
        else:
            await self._handle_http(first_line, first_line_raw, method, target, reader, writer)

    # ------------------------------------------------------------------
    # CONNECT handling (HTTPS tunneling)
    # ------------------------------------------------------------------

    async def _handle_connect(
        self,
        first_line: str,
        target: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an HTTPS CONNECT tunnel request."""
        # target is host:port
        host, port = self._parse_host_port(target, default_port=443)
        if host is None:
            self._send_error(client_writer, 400, "Malformed CONNECT target")
            return

        if not self._is_allowed(host):
            logger.info("Blocked CONNECT to %s for project %s", host, self._project_id)
            self._send_error(client_writer, 403, "Forbidden")
            self._fire_blocked(host, "CONNECT")
            return

        # Consume remaining request headers (up to blank line).
        await self._consume_headers(client_reader)

        # Open connection to the destination.
        try:
            remote_reader, remote_writer = await asyncio.open_connection(host, port)
        except (OSError, ConnectionError) as exc:
            logger.debug("Cannot connect to %s:%d: %s", host, port, exc)
            self._send_error(client_writer, 502, "Bad Gateway")
            return

        # Tell client the tunnel is established.
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()

        # Pipe data bidirectionally.
        await self._bidirectional_pipe(
            client_reader, client_writer, remote_reader, remote_writer
        )

    # ------------------------------------------------------------------
    # Plain HTTP handling
    # ------------------------------------------------------------------

    async def _handle_http(
        self,
        first_line: str,
        first_line_raw: bytes,
        method: str,
        target: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a plain HTTP request (forward proxy)."""
        parsed = urlparse(target)
        host = parsed.hostname
        if not host:
            self._send_error(client_writer, 400, "Missing host in request")
            return

        port = parsed.port or 80

        if not self._is_allowed(host):
            logger.info("Blocked %s to %s for project %s", method, host, self._project_id)
            self._send_error(client_writer, 403, "Forbidden")
            self._fire_blocked(host, method)
            # Consume remaining client data so the connection shuts down cleanly.
            await self._consume_headers(client_reader)
            return

        # Read the remaining headers from the client.
        remaining_headers = await self._read_headers(client_reader)

        # Rebuild the request with a relative path for the destination server.
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # Determine HTTP version from first line.
        line_parts = first_line.split()
        http_version = line_parts[2] if len(line_parts) >= 3 else "HTTP/1.1"

        request_line = f"{method} {path} {http_version}\r\n".encode("latin-1")

        # Open connection to destination.
        try:
            remote_reader, remote_writer = await asyncio.open_connection(host, port)
        except (OSError, ConnectionError) as exc:
            logger.debug("Cannot connect to %s:%d: %s", host, port, exc)
            self._send_error(client_writer, 502, "Bad Gateway")
            return

        try:
            # Send the reconstructed request line and headers.
            remote_writer.write(request_line)
            remote_writer.write(remaining_headers)
            await remote_writer.drain()

            # For plain HTTP: pipe the response back to the client.
            # (The request is already fully sent; no need for bidirectional pipe.)
            await self._pipe(remote_reader, client_writer)
        finally:
            for w in (remote_writer, client_writer):
                try:
                    w.close()
                    await w.wait_closed()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Bidirectional pipe
    # ------------------------------------------------------------------

    async def _bidirectional_pipe(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        remote_reader: asyncio.StreamReader,
        remote_writer: asyncio.StreamWriter,
    ) -> None:
        """Pipe data bidirectionally between client and remote."""
        task_c2r = asyncio.ensure_future(self._pipe(client_reader, remote_writer))
        task_r2c = asyncio.ensure_future(self._pipe(remote_reader, client_writer))

        try:
            done, pending = await asyncio.wait(
                {task_c2r, task_r2c}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            task_c2r.cancel()
            task_r2c.cancel()
            await asyncio.gather(task_c2r, task_r2c, return_exceptions=True)
            raise
        finally:
            for w in (client_writer, remote_writer):
                try:
                    w.close()
                    await w.wait_closed()
                except Exception:
                    pass

    @staticmethod
    async def _pipe(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Copy data from *reader* to *writer* until EOF or error."""
        try:
            while True:
                data = await reader.read(8192)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError, OSError):
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_host_port(
        target: str, default_port: int
    ) -> tuple[str | None, int]:
        """Parse 'host:port' or 'host' and return (host, port)."""
        if ":" in target:
            parts = target.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return None, default_port
        return target, default_port

    @staticmethod
    def _send_error(writer: asyncio.StreamWriter, code: int, reason: str) -> None:
        """Send a minimal HTTP error response."""
        body = f"{code} {reason}\r\n"
        response = (
            f"HTTP/1.1 {code} {reason}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: text/plain\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        try:
            writer.write(response.encode("latin-1"))
        except Exception:
            pass

    @staticmethod
    async def _consume_headers(reader: asyncio.StreamReader) -> None:
        """Read and discard headers until the blank line."""
        try:
            while True:
                line = await reader.readuntil(b"\n")
                if line.strip() == b"":
                    break
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            pass

    @staticmethod
    async def _read_headers(reader: asyncio.StreamReader) -> bytes:
        """Read all header lines including the blank terminator."""
        buf = bytearray()
        try:
            while True:
                line = await reader.readuntil(b"\n")
                buf.extend(line)
                if line.strip() == b"":
                    break
        except asyncio.IncompleteReadError as exc:
            buf.extend(exc.partial)
        except (asyncio.LimitOverrunError, ConnectionError):
            pass
        return bytes(buf)

    def _fire_blocked(self, domain: str, method: str) -> None:
        """Invoke the on_blocked callback if configured."""
        if self._on_blocked is not None:
            try:
                self._on_blocked(self._project_id, domain, method)
            except Exception:
                logger.debug("on_blocked callback raised", exc_info=True)
