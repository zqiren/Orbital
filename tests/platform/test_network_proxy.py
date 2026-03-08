# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C4: NetworkProxy (asyncio HTTP/HTTPS forward proxy).

10 tests from TASK-isolation-C4-network-proxy.md spec.
Uses pytest-asyncio. Starts a local test HTTP server as destination
to avoid external network dependencies.
"""

import asyncio
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest
import pytest_asyncio

from tests.platform.conftest import skip_not_windows


# --- Local test HTTP server ---


class _TestHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that returns 200 with the request path."""

    timeout = 5  # Per-connection socket timeout

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"OK: {self.path}".encode())

    def do_CONNECT(self):
        # This handler shouldn't receive CONNECT in normal operation
        # (CONNECT is handled by the proxy, not the destination)
        self.send_response(200)
        self.end_headers()

    def handle(self):
        try:
            super().handle()
        except (TimeoutError, OSError):
            pass

    def log_message(self, format, *args):
        # Suppress noisy log output during tests
        pass


@pytest.fixture
def local_http_server():
    """Start a local HTTP server on a random port. Returns (host, port)."""
    server = HTTPServer(("127.0.0.1", 0), _TestHandler)
    server.timeout = 0.5
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, args=(0.5,), daemon=True)
    thread.start()
    yield ("127.0.0.1", port)
    server.shutdown()
    thread.join(timeout=3)


@pytest_asyncio.fixture
async def proxy():
    """Create and start a NetworkProxy, stop it after test."""
    from agent_os.platform.shared.network import NetworkProxy

    p = NetworkProxy(project_id="test_project")
    await p.start()
    yield p
    await p.stop()


@pytest_asyncio.fixture
async def proxy_with_callback():
    """Create a proxy with a blocked callback tracker."""
    from agent_os.platform.shared.network import NetworkProxy

    blocked_calls: list[tuple[str, str, str]] = []

    def on_blocked(project_id: str, domain: str, method: str):
        blocked_calls.append((project_id, domain, method))

    p = NetworkProxy(
        project_id="test_callback_project",
        on_blocked=on_blocked,
    )
    await p.start()
    yield p, blocked_calls
    await p.stop()


async def _http_get_via_proxy(proxy_host: str, proxy_port: int, url: str) -> tuple[int, str]:
    """Make an HTTP GET request through the proxy. Returns (status_code, body)."""
    # Parse the target URL
    # url format: http://host:port/path
    from urllib.parse import urlparse

    parsed = urlparse(url)
    target_host = parsed.hostname
    target_port = parsed.port or 80
    path = parsed.path or "/"

    reader, writer = await asyncio.open_connection(proxy_host, proxy_port)

    # Send HTTP request with absolute URL (proxy style)
    request = (
        f"GET {url} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    writer.write(request.encode())
    await writer.drain()

    # Read full response until EOF
    chunks = []
    try:
        while True:
            data = await asyncio.wait_for(reader.read(65536), timeout=10)
            if not data:
                break
            chunks.append(data)
    except (asyncio.TimeoutError, ConnectionError):
        pass
    response = b"".join(chunks)
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    response_text = response.decode("utf-8", errors="replace")

    # Parse status code from first line
    first_line = response_text.split("\r\n")[0]
    parts = first_line.split(" ", 2)
    status_code = int(parts[1]) if len(parts) >= 2 else 0

    # Get body (after double CRLF)
    body = ""
    if "\r\n\r\n" in response_text:
        body = response_text.split("\r\n\r\n", 1)[1]

    return status_code, body


async def _connect_via_proxy(proxy_host: str, proxy_port: int, target_host: str, target_port: int) -> tuple[int, str]:
    """Send a CONNECT request through the proxy. Returns (status_code, status_line)."""
    reader, writer = await asyncio.open_connection(proxy_host, proxy_port)

    request = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        f"\r\n"
    )
    writer.write(request.encode())
    await writer.drain()

    # Read the proxy response
    response_line = await asyncio.wait_for(reader.readline(), timeout=10)
    response_text = response_line.decode("utf-8", errors="replace").strip()

    parts = response_text.split(" ", 2)
    status_code = int(parts[1]) if len(parts) >= 2 else 0

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    return status_code, response_text


@pytest.mark.asyncio
class TestNetworkProxy:
    async def test_start_stop(self):
        """Start proxy, verify port assigned, stop, verify server closed."""
        from agent_os.platform.shared.network import NetworkProxy

        p = NetworkProxy(project_id="test_start_stop")
        await p.start()

        assert p.port > 0
        assert "127.0.0.1" in p.proxy_url
        assert str(p.port) in p.proxy_url

        await p.stop()

    async def test_http_allowed(self, proxy, local_http_server):
        """Set allowlist with local server's address, make HTTP GET -> succeeds."""
        from agent_os.platform.types import NetworkRules

        host, port = local_http_server
        target_domain = f"{host}:{port}"

        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=[host, target_domain],
        ))

        status, body = await _http_get_via_proxy(
            "127.0.0.1", proxy.port,
            f"http://{host}:{port}/test_path",
        )
        assert status == 200
        assert "OK" in body

    async def test_http_blocked(self, proxy, local_http_server):
        """Set allowlist without target domain, make HTTP GET -> 403."""
        from agent_os.platform.types import NetworkRules

        host, port = local_http_server

        # Set allowlist with a different domain
        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=["only-this-domain.example.com"],
        ))

        status, body = await _http_get_via_proxy(
            "127.0.0.1", proxy.port,
            f"http://{host}:{port}/should_be_blocked",
        )
        assert status == 403

    async def test_https_connect_allowed(self, proxy, local_http_server):
        """CONNECT to allowed domain -> tunnel established (200)."""
        from agent_os.platform.types import NetworkRules

        host, port = local_http_server

        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=[host],
        ))

        status, _ = await _connect_via_proxy(
            "127.0.0.1", proxy.port,
            host, port,
        )
        assert status == 200

    async def test_https_connect_blocked(self, proxy):
        """CONNECT to blocked domain -> 403."""
        from agent_os.platform.types import NetworkRules

        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=["allowed.example.com"],
        ))

        status, _ = await _connect_via_proxy(
            "127.0.0.1", proxy.port,
            "blocked.example.com", 443,
        )
        assert status == 403

    async def test_blocked_callback(self, proxy_with_callback, local_http_server):
        """Make blocked request, verify on_blocked called with correct args."""
        from agent_os.platform.types import NetworkRules

        p, blocked_calls = proxy_with_callback
        host, port = local_http_server

        p.set_rules(NetworkRules(
            mode="allowlist",
            domains=["only-allowed.example.com"],
        ))

        # Make a request that will be blocked
        await _http_get_via_proxy(
            "127.0.0.1", p.port,
            f"http://{host}:{port}/blocked_path",
        )

        # Verify callback was called
        assert len(blocked_calls) > 0
        project_id, domain, method = blocked_calls[0]
        assert project_id == "test_callback_project"
        assert host in domain

    async def test_rules_update_live(self, proxy, local_http_server):
        """Start with domain blocked, update rules to allow, verify it works."""
        from agent_os.platform.types import NetworkRules

        host, port = local_http_server

        # Start with blocking everything
        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=["other.example.com"],
        ))

        status1, _ = await _http_get_via_proxy(
            "127.0.0.1", proxy.port,
            f"http://{host}:{port}/first_attempt",
        )
        assert status1 == 403

        # Update rules to allow
        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=[host],
        ))

        status2, body = await _http_get_via_proxy(
            "127.0.0.1", proxy.port,
            f"http://{host}:{port}/second_attempt",
        )
        assert status2 == 200

    async def test_wildcard_matching(self, proxy):
        """*.example.com matches sub.example.com, doesn't match example.com."""
        from agent_os.platform.types import NetworkRules

        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=["*.example.com"],
        ))

        # sub.example.com should match *.example.com → allowed (502 = can't connect, but not 403)
        status_sub, _ = await _connect_via_proxy(
            "127.0.0.1", proxy.port,
            "sub.example.com", 443,
        )
        assert status_sub != 403, "sub.example.com should match *.example.com (not blocked)"

        # example.com itself should NOT match *.example.com → blocked (403)
        status_root, _ = await _connect_via_proxy(
            "127.0.0.1", proxy.port,
            "example.com", 443,
        )
        assert status_root == 403

    async def test_no_rules_allows_all(self, local_http_server):
        """Proxy with no rules set -> all requests forwarded."""
        from agent_os.platform.shared.network import NetworkProxy

        p = NetworkProxy(project_id="test_no_rules")
        await p.start()

        try:
            host, port = local_http_server
            status, body = await _http_get_via_proxy(
                "127.0.0.1", p.port,
                f"http://{host}:{port}/any_path",
            )
            assert status == 200
            assert "OK" in body
        finally:
            await p.stop()

    async def test_concurrent_connections(self, proxy, local_http_server):
        """10 simultaneous requests, all handled without deadlock."""
        from agent_os.platform.types import NetworkRules

        host, port = local_http_server
        proxy.set_rules(NetworkRules(
            mode="allowlist",
            domains=[host],
        ))

        async def make_request(i: int):
            status, body = await _http_get_via_proxy(
                "127.0.0.1", proxy.port,
                f"http://{host}:{port}/concurrent_{i}",
            )
            return status, body

        # Launch 10 concurrent requests
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), (
                f"Request {i} raised: {result}"
            )
            status, body = result
            assert status == 200, f"Request {i} got status {status}"
