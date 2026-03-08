# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser safety module: URL validation, SSRF prevention, untrusted content wrapping, secret masking."""

import ipaddress
import re
import socket
from urllib.parse import urlparse

SECRET_PATTERN = re.compile(r"<secret:([a-zA-Z0-9_.]+)>")

_PRIVATE_NETWORKS_V4 = [
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("169.254.0.0/16"),
]

_PRIVATE_NETWORKS_V6 = [
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
]


def _is_private_ip(ip_str: str) -> bool:
    """Check whether an IP address string falls in a private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in net for net in _PRIVATE_NETWORKS_V4)
    if isinstance(addr, ipaddress.IPv6Address):
        return any(addr in net for net in _PRIVATE_NETWORKS_V6)
    return False


def _extract_host_ip(hostname: str) -> str | None:
    """If hostname is a literal IP (possibly bracketed IPv6), return it. Otherwise None."""
    # Strip IPv6 brackets
    stripped = hostname.strip("[]")
    try:
        ipaddress.ip_address(stripped)
        return stripped
    except ValueError:
        return None


def validate_url_pre_navigation(url: str) -> str | None:
    """Validate a URL before navigation. Returns None if allowed, or an error string if blocked."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL format."

    if not parsed.scheme:
        return "Invalid URL format."

    if parsed.scheme not in ("http", "https"):
        return f"Cannot navigate to {url}: only http/https URLs are allowed."

    if not parsed.netloc:
        return "Invalid URL format."

    hostname = parsed.hostname or ""
    if not hostname:
        return "Invalid URL format."

    # Check if hostname is a literal IP
    literal_ip = _extract_host_ip(hostname)
    if literal_ip is not None:
        if _is_private_ip(literal_ip):
            return f"Cannot navigate to {url}: private/internal addresses are blocked."
        return None

    # DNS resolution check for hostnames
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # DNS failure — let Playwright handle the error naturally
        return None

    resolved_ips = {info[4][0] for info in addrinfos}
    if resolved_ips and all(_is_private_ip(ip) for ip in resolved_ips):
        return f"Cannot navigate to {url}: hostname resolves to a private address."

    return None


def validate_url_post_navigation(final_url: str) -> str | None:
    """Validate the final URL after redirects. Returns None if allowed, or an error string if blocked."""
    try:
        parsed = urlparse(final_url)
    except Exception:
        return f"Navigation was redirected to a blocked address ({final_url}). The target site may have an open redirect vulnerability."

    if not parsed.scheme or not parsed.netloc:
        return f"Navigation was redirected to a blocked address ({final_url}). The target site may have an open redirect vulnerability."

    hostname = parsed.hostname or ""
    if not hostname:
        return f"Navigation was redirected to a blocked address ({final_url}). The target site may have an open redirect vulnerability."

    literal_ip = _extract_host_ip(hostname)
    if literal_ip is not None:
        if _is_private_ip(literal_ip):
            return f"Navigation was redirected to a blocked address ({final_url}). The target site may have an open redirect vulnerability."
        return None

    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return None

    resolved_ips = {info[4][0] for info in addrinfos}
    if resolved_ips and all(_is_private_ip(ip) for ip in resolved_ips):
        return f"Navigation was redirected to a blocked address ({final_url}). The target site may have an open redirect vulnerability."

    return None


def wrap_untrusted_content(content: str, source_url: str) -> str:
    """Wrap browser-fetched content with untrusted content markers."""
    return (
        f"[BROWSER CONTENT \u2014 UNTRUSTED \u2014 from: {source_url}]\n"
        f"{content}\n"
        f"[/BROWSER CONTENT]"
    )


def detect_secrets(text: str) -> list[str]:
    """Find all <secret:KEY> patterns in text. Return list of KEY names."""
    return SECRET_PATTERN.findall(text)


def substitute_secrets(text: str, resolver: "Callable[[str], str]") -> str:
    """Replace each <secret:KEY> with the actual value via resolver callable."""

    def replacer(match: re.Match) -> str:
        key = match.group(1)
        value = resolver(key)
        if value is None:
            raise ValueError(
                f"Secret '{key}' not found in credential store. "
                "Register it in project settings first."
            )
        return value

    return SECRET_PATTERN.sub(replacer, text)
