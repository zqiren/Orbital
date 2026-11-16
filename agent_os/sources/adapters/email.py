# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E-mail over IMAP (standard library ``imaplib``, always off the event loop).

Read-only by construction: the folder is opened with ``EXAMINE`` and mail is
read with ``BODY.PEEK``, so nothing is marked read, moved or deleted.

TLS: port 993 = TLS from the first byte; any other port = plain connection
upgraded with ``STARTTLS``, which is required. Certificates are verified.
The one exception is a server on this computer (127.0.0.1 / ::1 /
localhost, e.g. a mail bridge): traffic never leaves the machine, so its
self-signed certificate is accepted and, if it offers no STARTTLS, it is
used as is.

New mail: the source keeps a cursor ``(folder, UIDVALIDITY, last UID)`` in its
state. The first poll (and any poll after the folder or UIDVALIDITY
changed) only moves the cursor to the folder's end, so nothing already in
the mailbox is sent, also after the server renumbered it. Later polls read
UIDs above the cursor (unread ones only when ``unread_only``), at most
``FETCH_CAP`` per poll, oldest first. Item id = ``<folder>:<UIDVALIDITY>:<UID>``.
"""

from __future__ import annotations

import base64
import imaplib
import ipaddress
import re
import socket
import ssl
from datetime import timezone
from email import message_from_bytes
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser

from agent_os.sources.adapters import Adapter
from agent_os.sources.model import Choice, Item, SourceFailure, Verified, VerifyError

TIMEOUT_S = 20
FETCH_CAP = 25
PEEK_BYTES = 262_144
EXCERPT_CAP = 4000
TLS_PORT = 993


# ── text decoding ──────────────────────────────────────────────────────────

_CHARSET_ALIASES = {
    "gb2312": "gb18030", "gbk": "gb18030", "x-gbk": "gb18030", "cp936": "gb18030",
    "gb_2312-80": "gb18030", "euc-cn": "gb18030", "x-euc-cn": "gb18030", "chinese": "gb18030",
    "big5": "big5hkscs", "x-big5": "big5hkscs",
    "ks_c_5601-1987": "cp949", "euc-kr": "cp949",
    "shift_jis": "cp932", "x-sjis": "cp932", "sjis": "cp932",
    "unicode-1-1-utf-7": "utf-7", "utf8": "utf-8", "x-unknown": "utf-8", "unknown-8bit": "utf-8",
}


def decode_bytes(data: bytes, charset: str | None = None) -> str:
    """Bytes in a declared (possibly wrong or unknown) charset → text. Never raises."""
    if not data:
        return ""
    tried = []
    cs = (charset or "").strip().strip('"').lower()
    if cs:
        tried.append(_CHARSET_ALIASES.get(cs, cs))
    tried += ["utf-8", "gb18030"]
    for enc in tried:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode(tried[0] if _known(tried[0]) else "utf-8", errors="replace")


def _known(enc: str) -> bool:
    try:
        "".encode(enc)
        return True
    except LookupError:
        return False


def decode_header_value(value) -> str:
    """A header (RFC 2047 encoded words, or raw 8-bit bytes such as an
    unencoded GBK subject) → one line of text."""
    if value is None:
        return ""
    raw = str(value)
    if any(0xDC80 <= ord(c) <= 0xDCFF for c in raw):
        # compat32 kept undecodable 8-bit bytes as surrogates: raw bytes.
        raw = decode_bytes(raw.encode("ascii", "surrogateescape"))
    try:
        parts = decode_header(raw)
    except Exception:
        return " ".join(raw.split())
    out = []
    for text, cs in parts:
        if isinstance(text, bytes):
            out.append(decode_bytes(text, cs))
        else:
            out.append(text)
    # decode_header leaves the spaces between plain and encoded parts on the plain parts
    return " ".join("".join(out).split())


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "head", "title"}
    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table",
              "blockquote", "section", "article", "hr", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    try:
        p.feed(html)
        p.close()
        text = "".join(p.parts)
    except Exception:
        text = unescape(re.sub(r"<[^>]+>", " ", html))
    return _tidy(text)


def _tidy(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r", "").split("\n")]
    out, blank = [], False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()


def body_text(msg) -> str:
    """The readable text of a message: the first text/plain part, else the
    first text/html part as text. Attachments are skipped."""
    plain = html = None
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.is_multipart():
            continue
        disp = (part.get("Content-Disposition") or "").lower()
        if disp.startswith("attachment"):
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True) or b""
        except Exception:
            payload = b""
        text = decode_bytes(payload, part.get_content_charset())
        if ctype == "text/plain" and plain is None:
            plain = text
        elif ctype == "text/html" and html is None:
            html = text
        if plain is not None:
            break
    if plain is not None and plain.strip():
        return _tidy(plain)
    if html:
        return html_to_text(html)
    return ""


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CAP:
        return text
    return text[:EXCERPT_CAP].rstrip() + " …"


# ── IMAP modified UTF-7 (RFC 3501 §5.1.3) ──────────────────────────────────

def mutf7_decode(name: str) -> str:
    def _seg(m):
        s = m.group(1)
        if not s:
            return "&"
        b64 = s.replace(",", "/")
        b64 += "=" * (-len(b64) % 4)
        try:
            return base64.b64decode(b64).decode("utf-16-be")
        except Exception:
            return m.group(0)
    return re.sub(r"&([A-Za-z0-9+,]*)-", _seg, name)


def mutf7_encode(name: str) -> str:
    out, buf = [], []

    def flush():
        if buf:
            b = base64.b64encode("".join(buf).encode("utf-16-be")).decode().rstrip("=")
            out.append("&" + b.replace("/", ",") + "-")
            buf.clear()
    for ch in name:
        if 0x20 <= ord(ch) <= 0x7E:
            flush()
            out.append("&-" if ch == "&" else ch)
        else:
            buf.append(ch)
    flush()
    return "".join(out)


def _imap_folder(folder: str) -> str:
    """User text or a raw name from the folder list → a quoted IMAP mailbox."""
    f = (folder or "").strip() or "INBOX"
    if any(ord(c) > 0x7E for c in f):
        f = mutf7_encode(f)
    return '"' + f.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ── connection ─────────────────────────────────────────────────────────────

def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Fail(Exception):
    """Internal: (reason, field, failure code, retryable, message)."""

    def __init__(self, reason, field, code, retryable, message):
        super().__init__(message)
        self.reason, self.field, self.code, self.retryable, self.message = \
            reason, field, code, retryable, message


def _settings(ctx) -> tuple[str, int, str, str]:
    server = (ctx.config.get("server") or "").strip()
    if "://" in server:
        server = server.split("://", 1)[1]
    server = server.strip("/").strip()
    try:
        port = int(str(ctx.config.get("port") or TLS_PORT).strip())
    except ValueError:
        raise VerifyError("invalid", "port", "the port must be a number")
    if not 0 < port < 65536:
        raise VerifyError("invalid", "port", "the port must be a number")
    acct = (ctx.config.get("acct") or "").strip()
    if not server:
        raise VerifyError("invalid", "server", "enter the IMAP server")
    if not acct:
        raise VerifyError("invalid", "acct", "enter the mail address")
    return server, port, acct, ctx.secret("pass") or ""


def _connect(server: str, port: int):
    loop = _is_loopback(server)
    tls = ssl.create_default_context()
    if loop:
        tls.check_hostname = False
        tls.verify_mode = ssl.CERT_NONE
    try:
        if port == TLS_PORT:
            return imaplib.IMAP4_SSL(server, port, ssl_context=tls, timeout=TIMEOUT_S)
        conn = imaplib.IMAP4(server, port, timeout=TIMEOUT_S)
        if "STARTTLS" in conn.capabilities:
            conn.starttls(ssl_context=tls)
        elif not loop:
            _close(conn)
            raise _Fail("invalid", "port", "unreachable", True,
                        "this server offers no encrypted connection on that port")
        return conn
    except _Fail:
        raise
    except socket.gaierror:
        raise _Fail("not_found", "server", "unreachable", True, f"cannot find {server}")
    except ssl.SSLCertVerificationError:
        raise _Fail("invalid", "server", "unreachable", True, f"{server} has no valid certificate")
    except ssl.SSLError:
        raise _Fail("invalid", "port", "unreachable", True, "that port does not speak TLS")
    except ConnectionRefusedError:
        raise _Fail("unreachable", "port", "unreachable", True, f"nothing answers on port {port}")
    except (socket.timeout, TimeoutError):
        raise _Fail("unreachable", "server", "unreachable", True, f"{server} did not answer")
    except (OSError, imaplib.IMAP4.error):
        raise _Fail("unreachable", "server", "unreachable", True, f"{server} did not answer")


def _close(conn) -> None:
    try:
        conn.logout()
    except Exception:
        try:
            conn.shutdown()
        except Exception:
            pass


def _login(conn, acct: str, password: str) -> None:
    try:
        conn.login(acct, password)
    except imaplib.IMAP4.abort:
        raise _Fail("unreachable", "server", "unreachable", True, "the server closed the connection")
    except imaplib.IMAP4.error as e:
        text = str(e).upper()
        if any(w in text for w in ("[UNAVAILABLE]", "[LIMIT]", "TRY AGAIN", "TEMPORAR", "TOO MANY")):
            raise _Fail("unreachable", "server", "unreachable", True, "the server is busy; try again shortly")
        raise _Fail("rejected", "pass", "token_rejected", False, "the address or app password was refused")
    except (OSError, ssl.SSLError):
        raise _Fail("unreachable", "server", "unreachable", True, "the server closed the connection")


def _examine(conn, folder: str) -> tuple[int, int | None]:
    """Open read-only; → (UIDVALIDITY, UIDNEXT or None)."""
    try:
        typ, _ = conn.select(_imap_folder(folder), readonly=True)
    except imaplib.IMAP4.abort:
        raise _Fail("unreachable", "server", "unreachable", True, "the server closed the connection")
    except imaplib.IMAP4.error:
        typ = "NO"
    if typ != "OK":
        raise _Fail("not_found", "folder", "error", True, f"there is no folder {folder}")
    _, v = conn.response("UIDVALIDITY")
    _, n = conn.response("UIDNEXT")
    try:
        uidvalidity = int((v or [b"0"])[0] or 0)
    except (TypeError, ValueError):
        uidvalidity = 0
    try:
        uidnext = int((n or [None])[0]) if n and n[0] else None
    except (TypeError, ValueError):
        uidnext = None
    return uidvalidity, uidnext


def _search(conn, *criteria) -> list[int]:
    typ, data = conn.uid("SEARCH", *criteria)
    if typ != "OK":
        raise _Fail("error", None, "error", True, "the server refused the search")
    out = []
    for chunk in data or []:
        if isinstance(chunk, bytes):
            out += [int(x) for x in chunk.split() if x.isdigit()]
    return sorted(set(out))


_UID_RE = re.compile(rb"UID (\d+)")


def _fetch(conn, uids: list[int]) -> dict[int, bytes]:
    if not uids:
        return {}
    typ, data = conn.uid("FETCH", ",".join(str(u) for u in uids),
                         f"(UID BODY.PEEK[]<0.{PEEK_BYTES}>)")
    if typ != "OK":
        raise _Fail("error", None, "error", True, "the server refused to send the mail")
    # imaplib gives [(b'1 (UID 5 BODY[]<0> {n}', raw), b')', ...]; a server may
    # also put UID after the literal, in the bytes element that follows.
    rows = list(data or [])
    out = {}
    for i, row in enumerate(rows):
        if not (isinstance(row, tuple) and len(row) >= 2):
            continue
        m = _UID_RE.search(row[0] or b"")
        if m is None and i + 1 < len(rows) and isinstance(rows[i + 1], bytes):
            m = _UID_RE.search(rows[i + 1])
        if m is None and len(uids) == 1:
            out[uids[0]] = row[1] or b""
        elif m is not None:
            out[int(m.group(1))] = row[1] or b""
    return out


def _raw_header(msg, name: str):
    """The header as stored (undecodable 8-bit bytes kept as surrogates);
    ``msg.get`` would already have replaced them with U+FFFD."""
    for k, v in msg.raw_items():
        if k.lower() == name.lower():
            return v
    return None


def message_item(raw: bytes, item_id: str, folder: str) -> Item:
    msg = message_from_bytes(raw)
    subject = decode_header_value(_raw_header(msg, "Subject")) or "(no subject)"
    sender_name = sender_addr = ""
    try:
        pairs = getaddresses([decode_header_value(_raw_header(msg, "From"))])
        if pairs:
            sender_name, sender_addr = pairs[0]
    except Exception:
        pass
    sender = sender_name or sender_addr or None
    at = None
    try:
        dt = parsedate_to_datetime(msg.get("Date"))
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            at = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        at = None
    # The task header already names the sender; add the address it hides.
    head = [f"Address: {sender_addr}"] if sender_name and sender_addr else []
    if folder.upper() != "INBOX":
        head.append(f"Folder: {mutf7_decode(folder)}")
    text = _excerpt(body_text(msg))
    body = "\n\n".join(x for x in ("\n".join(head), text) if x)
    return Item(id=item_id, kind="new_mail", subject=item_id, title=subject, body=body,
                sender=sender, at=at)


# ── the adapter ────────────────────────────────────────────────────────────

class EmailAdapter(Adapter):
    mode = "poll"
    interval_s = 120

    @staticmethod
    def _params(ctx) -> tuple[str, bool]:
        p = ctx.kinds.get("new_mail") or {}
        folder = (p.get("folder") or "INBOX").strip() or "INBOX"
        return folder, bool(p.get("unread_only", True))

    # -- verify -------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        server, port, acct, password = _settings(ctx)
        if not password:
            raise VerifyError("invalid", "pass", "enter the app password")
        folder, _ = self._params(ctx)

        def _check():
            conn = _connect(server, port)
            try:
                _login(conn, acct, password)
                _examine(conn, folder)
            finally:
                _close(conn)
        try:
            await ctx.to_thread(_check)
        except _Fail as f:
            raise VerifyError(f.reason, f.field, f.message)
        return Verified(label=acct)

    # -- folders ------------------------------------------------------------

    async def choices(self, ctx, list_name, params):
        if list_name != "folders":
            return []
        server, port, acct, password = _settings(ctx)

        def _list():
            conn = _connect(server, port)
            try:
                _login(conn, acct, password)
                typ, data = conn.list()
                return data if typ == "OK" else []
            finally:
                _close(conn)
        try:
            rows = await ctx.to_thread(_list)
        except _Fail as f:
            raise VerifyError(f.reason, f.field, f.message)
        out, seen = [], set()
        for name, flags in _parse_list(rows):
            if "\\noselect" in flags or "\\nonexistent" in flags or name in seen:
                continue
            seen.add(name)
            out.append(Choice(name, mutf7_decode(name)))
        out.sort(key=lambda c: (c.value.upper() != "INBOX", c.label.lower()))
        return out

    # -- poll ---------------------------------------------------------------

    async def poll(self, ctx) -> list:
        try:
            server, port, acct, password = _settings(ctx)
        except VerifyError:
            raise SourceFailure("error")
        folder, unread_only = self._params(ctx)
        cursor = ctx.state.get("cursor") if isinstance(ctx.state.get("cursor"), dict) else {}

        def _run():
            conn = _connect(server, port)
            try:
                _login(conn, acct, password)
                validity, uidnext = _examine(conn, folder)
                if cursor.get("folder") != folder or cursor.get("uidvalidity") != validity:
                    # First look, another folder, or the server renumbered the
                    # folder: start from its end, send nothing already there.
                    if uidnext is not None:
                        last = uidnext - 1
                    else:
                        found = _search(conn, "ALL")
                        last = found[-1] if found else 0
                    return {"folder": folder, "uidvalidity": validity, "last_uid": last,
                            "renumbered": bool(cursor) and cursor.get("folder") == folder}, {}
                last = int(cursor.get("last_uid") or 0)
                rng = f"{last + 1}:*"
                above = [u for u in _search(conn, "UID", rng) if u > last]
                if not above:
                    return dict(cursor), {}
                wanted = [u for u in _search(conn, "UID", rng, "UNSEEN") if u > last] \
                    if unread_only else above
                batch = wanted[:FETCH_CAP]
                if len(wanted) > FETCH_CAP:
                    new_last = batch[-1]  # the rest next time, oldest first
                else:
                    new_last = above[-1]
                raws = _fetch(conn, batch)
                return {"folder": folder, "uidvalidity": validity, "last_uid": new_last}, \
                    {u: raws[u] for u in batch if u in raws}
            finally:
                _close(conn)
        try:
            new_cursor, raws = await ctx.to_thread(_run)
        except _Fail as f:
            if f.code == "token_rejected":
                raise SourceFailure("token_rejected", {"field": "pass"}, retryable=False)
            if f.code == "unreachable":
                raise SourceFailure("unreachable", {"host": server})
            if f.field == "folder":
                raise SourceFailure("folder_not_found", {"folder": mutf7_decode(folder)})
            raise SourceFailure("error")
        if new_cursor.pop("renumbered", False):
            ctx.log.info("sources: %s: folder renumbered (UIDVALIDITY changed); "
                         "restarting from its end", ctx.source_id)
        ctx.state["cursor"] = new_cursor
        validity = new_cursor["uidvalidity"]
        return [message_item(raw, f"{folder}:{validity}:{uid}", folder) for uid, raw in sorted(raws.items())]


_LIST_RE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?P<delim>"(?:[^"\\]|\\.)*"|NIL)\s+(?P<name>.*)$', re.I)


def _parse_list(rows) -> list[tuple[str, set]]:
    out = []
    for row in rows or []:
        if isinstance(row, tuple):  # a literal: (b'(\\HasNoChildren) "/" {5}', b'Inbox')
            head = row[0].decode("utf-8", "replace") if isinstance(row[0], bytes) else str(row[0])
            name = row[1].decode("utf-8", "replace") if isinstance(row[1], bytes) else str(row[1])
            m = _LIST_RE.match(head.strip())
            if m:
                out.append((name, {f.lower() for f in m.group("flags").split()}))
            continue
        if not isinstance(row, bytes):
            continue
        line = row.decode("utf-8", "replace").strip()
        m = _LIST_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name.startswith('"') and name.endswith('"') and len(name) >= 2:
            name = re.sub(r'\\(.)', r'\1', name[1:-1])
        out.append((name, {f.lower() for f in m.group("flags").split()}))
    return out


ADAPTER = EmailAdapter
