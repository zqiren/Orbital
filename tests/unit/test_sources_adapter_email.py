# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E-mail source adapter against a local IMAP server stub (a real socket,
the real ``imaplib`` client).

The stub speaks the IMAP4rev1 subset the adapter uses, with response lines
shaped after RFC 3501's examples (§6.3.1 SELECT/EXAMINE with ``[UIDVALIDITY]``
/ ``[UIDNEXT]`` / ``[READ-ONLY]``, §6.3.8 LIST, §6.4.4 SEARCH, §6.4.5 FETCH
with a partial ``BODY[]<0>`` literal, §7.1 ``NO [AUTHENTICATIONFAILED]``
from RFC 5530, which Gmail and Outlook send). It also refuses (and records)
any command that would change the mailbox — STORE, COPY, MOVE, EXPUNGE,
SELECT, or a FETCH without PEEK — so the read-only rule is tested on the wire.
The mails are hand-built: a multipart GBK mail with RFC 2047 GBK/GB2312
headers as Chinese webmail sends them, an unencoded 8-bit GB18030 subject,
an HTML-only mail, an unknown charset and an oversized body.
"""

from __future__ import annotations

import base64
import re
import socket
import threading

import pytest

from agent_os.sources.adapters.email import (
    EmailAdapter, decode_bytes, decode_header_value, html_to_text, message_item,
    mutf7_decode, mutf7_encode,
)
from agent_os.sources.context import SourceContext
from agent_os.sources.model import SourceFailure, VerifyError

USER, PASSWORD = "mira@example.cn", "abcd efgh ijkl mnop"


# ── the stub ───────────────────────────────────────────────────────────────

def _tokens(s: str) -> list[str]:
    out, i = [], 0
    while i < len(s):
        if s[i] == " ":
            i += 1
        elif s[i] == '"':
            j, buf = i + 1, []
            while j < len(s) and s[j] != '"':
                if s[j] == "\\":
                    j += 1
                buf.append(s[j])
                j += 1
            out.append("".join(buf))
            i = j + 1
        elif s[i] == "(":
            j = s.index(")", i)
            out.append(s[i:j + 1])
            i = j + 1
        else:
            j = s.find(" ", i)
            j = len(s) if j < 0 else j
            out.append(s[i:j])
            i = j
    return out


class ImapStub:
    """Folders: name → {"uidvalidity": int, "uidnext": int, "msgs": [[uid, raw, seen]]}."""

    def __init__(self, user=USER, password=PASSWORD):
        self.user, self.password = user, password
        self.folders = {"INBOX": {"uidvalidity": 1700000001, "uidnext": 1, "msgs": []}}
        self.violations: list[str] = []
        self.commands: list[str] = []
        self.login_reply = None  # e.g. "NO [UNAVAILABLE] try later"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def add(self, raw: bytes, folder="INBOX", seen=False) -> int:
        f = self.folders[folder]
        uid = f["uidnext"]
        f["uidnext"] += 1
        f["msgs"].append([uid, raw, seen])
        return uid

    def renumber(self, folder="INBOX", start=1):
        """A server rebuild: new UIDVALIDITY, every UID reassigned."""
        f = self.folders[folder]
        f["uidvalidity"] += 1
        for i, m in enumerate(f["msgs"]):
            m[0] = start + i
        f["uidnext"] = start + len(f["msgs"])

    def close(self):
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile("rb")
        send = lambda b: conn.sendall(b if isinstance(b, bytes) else b.encode())
        send("* OK [CAPABILITY IMAP4rev1 IDLE] stub ready\r\n")
        folder = None
        try:
            while True:
                line = f.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                self.commands.append(text)
                tag, _, rest = text.partition(" ")
                cmd, _, args = rest.partition(" ")
                cmd = cmd.upper()
                if cmd == "CAPABILITY":
                    send("* CAPABILITY IMAP4rev1 IDLE\r\n" + f"{tag} OK done\r\n")
                elif cmd == "LOGIN":
                    u, p = _tokens(args)[:2]
                    if self.login_reply:
                        send(f"{tag} {self.login_reply}\r\n")
                    elif (u, p) == (self.user, self.password):
                        send(f"{tag} OK logged in\r\n")
                    else:
                        send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials (Failure)\r\n")
                elif cmd == "LIST":
                    for name in self.folders:
                        send(f'* LIST (\\HasNoChildren) "/" "{name}"\r\n')
                    send('* LIST (\\Noselect \\HasChildren) "/" "[Gmail]"\r\n')
                    send(f"{tag} OK LIST done\r\n")
                elif cmd in ("EXAMINE", "SELECT"):
                    if cmd == "SELECT":
                        self.violations.append(text)
                    name = _tokens(args)[0]
                    if name not in self.folders:
                        send(f"{tag} NO [NONEXISTENT] Unknown Mailbox: {name}\r\n")
                        continue
                    folder = self.folders[name]
                    send(f"* {len(folder['msgs'])} EXISTS\r\n* 0 RECENT\r\n"
                         f"* OK [UIDVALIDITY {folder['uidvalidity']}] UIDs valid\r\n"
                         f"* OK [UIDNEXT {folder['uidnext']}] Predicted next UID\r\n"
                         f"{tag} OK [READ-ONLY] {cmd} completed\r\n")
                elif cmd == "UID":
                    sub, _, a = args.partition(" ")
                    sub = sub.upper()
                    if sub == "SEARCH":
                        send(self._search(folder, a) + f"{tag} OK SEARCH completed\r\n")
                    elif sub == "FETCH":
                        if "BODY[" in a.upper() and "PEEK" not in a.upper():
                            self.violations.append(text)
                        send(self._fetch(folder, a))
                        send(f"{tag} OK FETCH completed\r\n")
                    else:
                        self.violations.append(text)
                        send(f"{tag} BAD read-only stub\r\n")
                elif cmd == "NOOP":
                    send(f"{tag} OK\r\n")
                elif cmd == "LOGOUT":
                    send(f"* BYE bye\r\n{tag} OK LOGOUT completed\r\n")
                    return
                else:
                    self.violations.append(text)
                    send(f"{tag} BAD not in this stub\r\n")
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _search(folder, criteria: str) -> str:
        msgs = folder["msgs"] if folder else []
        uids = [m[0] for m in msgs]
        toks = criteria.upper().split()
        if "UID" in toks:
            lo, hi = toks[toks.index("UID") + 1].split(":")
            top = max(uids) if uids else 0
            lo = int(lo)
            hi = top if hi == "*" else int(hi)
            if lo > top and uids:      # RFC 3501: "n:*" with n > max still matches the max
                lo = top
            uids = [u for u in uids if lo <= u <= hi]
        if "UNSEEN" in toks:
            seen = {m[0] for m in msgs if m[2]}
            uids = [u for u in uids if u not in seen]
        return "* SEARCH" + "".join(f" {u}" for u in uids) + "\r\n"

    @staticmethod
    def _fetch(folder, a: str) -> bytes:
        uidset, _, items = a.partition(" ")
        want = {int(x) for x in uidset.split(",")}
        m = re.search(r"<0\.(\d+)>", items)
        cap = int(m.group(1)) if m else None
        out = b""
        for seq, (uid, raw, _seen) in enumerate(folder["msgs"], 1):
            if uid in want:
                data = raw[:cap] if cap else raw
                out += f"* {seq} FETCH (UID {uid} BODY[]<0> {{{len(data)}}}\r\n".encode() + data + b")\r\n"
        return out


@pytest.fixture
def imap():
    s = ImapStub()
    yield s
    s.close()


def _ctx(imap, *, password=PASSWORD, folder="INBOX", unread_only=True, state=None, acct=USER):
    secrets = {"pass": password}
    return SourceContext(source_id="src_m", type_="email",
                         config={"server": "127.0.0.1", "port": str(imap.port), "acct": acct},
                         kinds={"new_mail": {"folder": folder, "unread_only": unread_only}},
                         account=None, secrets_fn=secrets.get,
                         state=state if state is not None else {})


# ── mails ──────────────────────────────────────────────────────────────────

def _b64(text, cs):
    return base64.b64encode(text.encode(cs)).decode()


GBK_MULTIPART = (
    "From: =?GB2312?B?" + _b64("李娜", "gb2312") + "?= <li.na@example.cn>\r\n"
    "To: mira@example.cn\r\n"
    "Subject: =?GBK?B?" + _b64("关于下周发布的问题", "gbk") + "?=\r\n"
    "Date: Mon, 16 Nov 2026 15:20:00 +0800\r\n"
    "Message-ID: <a1@example.cn>\r\n"
    "MIME-Version: 1.0\r\n"
    'Content-Type: multipart/alternative; boundary="b1"\r\n\r\n'
    "--b1\r\n"
    'Content-Type: text/plain; charset="gbk"\r\n'
    "Content-Transfer-Encoding: base64\r\n\r\n"
    + _b64("你好 Mira，\r\n购物车删除商品后总价没变，刷新才对。\r\n", "gbk") + "\r\n"
    "--b1\r\n"
    'Content-Type: text/html; charset="gbk"\r\n'
    "Content-Transfer-Encoding: base64\r\n\r\n"
    + _b64("<p>你好 Mira，</p><p>购物车删除商品后总价没变。</p>", "gbk") + "\r\n"
    "--b1--\r\n"
).encode("ascii")

RAW_GB18030_SUBJECT = (
    b"From: ops@example.cn\r\nSubject: " + "服务器告警：磁盘 95%".encode("gb18030") + b"\r\n"
    b"Date: Mon, 16 Nov 2026 07:00:00 +0000\r\n"
    b"Content-Type: text/plain; charset=gb18030\r\n\r\n" + "磁盘快满了 𠀀".encode("gb18030") + b"\r\n"
)

HTML_ONLY = (
    b"From: \"Shop\" <noreply@shop.example>\r\nSubject: =?UTF-8?Q?Order_=E2=9C=93_shipped?=\r\n"
    b"Content-Type: text/html; charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n"
    b"<html><head><style>p{color:red}</style><script>alert(1)</script></head>"
    b"<body><p>Your order&nbsp;#42</p><p>has <b>shipped</b> &amp; is on its way.</p></body></html>\r\n"
)

UNKNOWN_CHARSET = (
    b"From: a@b.c\r\nSubject: =?x-made-up?B?" + base64.b64encode("hello 世界".encode()) + b"?=\r\n"
    b"Content-Type: text/plain; charset=x-made-up\r\n\r\n" + "body 世界".encode() + b"\r\n"
)

HUGE = (b"From: a@b.c\r\nSubject: big\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
        + ("word " * 60000).encode())


# ── decoding (no server) ───────────────────────────────────────────────────

def test_header_and_charset_decoding():
    assert decode_header_value("=?GBK?B?" + _b64("关于下周发布的问题", "gbk") + "?=") == "关于下周发布的问题"
    assert decode_header_value("Re: =?utf-8?B?" + _b64("你好", "utf-8") + "?= there") == "Re: 你好 there"
    assert decode_bytes("磁盘".encode("gbk"), "gb2312") == "磁盘"
    assert decode_bytes("𠀀".encode("gb18030"), "gbk") == "𠀀"  # gbk alias decodes as the superset
    assert decode_bytes(b"\xff\xfe\xfa", "utf-8")  # garbage never raises
    assert decode_bytes("ok".encode(), "no-such-charset") == "ok"


def test_html_to_text_drops_script_and_style():
    t = html_to_text("<style>x{}</style><script>evil()</script><p>a&amp;b</p><br>c")
    assert t == "a&b\n\nc"


def test_modified_utf7_round_trip():
    assert mutf7_decode("&UXZO1mWHTvZZOQ-") == "其他文件夹"
    assert mutf7_encode("其他文件夹") == "&UXZO1mWHTvZZOQ-"
    assert mutf7_decode("A&-B") == "A&B" and mutf7_encode("A&B") == "A&-B"


def test_message_items():
    it = message_item(GBK_MULTIPART, "INBOX:1:5", "INBOX")
    assert it.title == "关于下周发布的问题"
    assert it.sender == "李娜"
    assert it.body.startswith("Address: li.na@example.cn\n\n") and "购物车删除商品后总价没变，刷新才对。" in it.body
    assert "<p>" not in it.body
    assert it.at == "2026-11-16T07:20:00Z" and it.kind == "new_mail" and it.subject == "INBOX:1:5"
    assert message_item(RAW_GB18030_SUBJECT, "i", "INBOX").title == "服务器告警：磁盘 95%"
    assert "磁盘快满了 𠀀" in message_item(RAW_GB18030_SUBJECT, "i", "INBOX").body
    h = message_item(HTML_ONLY, "i", "INBOX")
    assert h.title == "Order ✓ shipped" and h.sender == "Shop"
    assert "Your order #42" in h.body and "shipped & is on its way." in h.body
    assert "alert" not in h.body and "color" not in h.body
    u = message_item(UNKNOWN_CHARSET, "i", "INBOX")
    assert u.title == "hello 世界" and "body 世界" in u.body
    big = message_item(HUGE, "i", "INBOX")
    assert len(big.body) < 4200 and big.body.endswith("…")


# ── against the stub ───────────────────────────────────────────────────────

async def test_verify_logs_in_and_examines_the_folder(imap):
    v = await EmailAdapter().verify(_ctx(imap))
    assert v.label == USER
    assert any(c.split(" ")[1].upper() == "EXAMINE" for c in imap.commands)
    assert imap.violations == []


async def test_verify_maps_failures_to_fields(imap, monkeypatch):
    real_gai = socket.getaddrinfo

    def no_dns(h, *a, **k):  # the resolver's own failure, without asking it
        if (h.decode() if isinstance(h, bytes) else h) == "imap.no-such-host.invalid":
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")
        return real_gai(h, *a, **k)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(_ctx(imap, password="wrong"))
    assert (e.value.reason, e.value.field) == ("rejected", "pass")
    assert "wrong" not in e.value.message and PASSWORD not in e.value.message
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(_ctx(imap, folder="Nope"))
    assert (e.value.reason, e.value.field) == ("not_found", "folder")
    ctx = _ctx(imap)
    ctx.config["server"] = "imap.no-such-host.invalid"
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(ctx)
    assert (e.value.reason, e.value.field) == ("not_found", "server")
    ctx = _ctx(imap)
    ctx.config["port"] = "9"  # nothing listens
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(ctx)
    assert (e.value.reason, e.value.field) == ("unreachable", "port")
    ctx = _ctx(imap)
    ctx.config["port"] = "abc"
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(ctx)
    assert e.value.field == "port"


async def test_tls_port_against_a_plain_server_is_a_port_problem(imap, monkeypatch):
    import agent_os.sources.adapters.email as mod
    monkeypatch.setattr(mod, "TLS_PORT", imap.port)  # pretend the stub's port is 993
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().verify(_ctx(imap))
    assert e.value.field in ("port", "server") and e.value.reason in ("invalid", "unreachable")


async def test_first_poll_sends_nothing_already_there_then_new_mail_arrives(imap):
    imap.add(b"From: a@b.c\r\nSubject: old\r\n\r\nold\r\n")
    imap.add(b"From: a@b.c\r\nSubject: old2\r\n\r\nold\r\n")
    state = {}
    a = EmailAdapter()
    assert await a.poll(_ctx(imap, state=state)) == []
    assert state["cursor"] == {"folder": "INBOX", "uidvalidity": 1700000001, "last_uid": 2}
    uid = imap.add(GBK_MULTIPART)
    imap.add(b"From: a@b.c\r\nSubject: read elsewhere\r\n\r\nx\r\n", seen=True)
    items = await a.poll(_ctx(imap, state=state))
    assert [i.id for i in items] == [f"INBOX:1700000001:{uid}"]
    assert items[0].title == "关于下周发布的问题"
    assert state["cursor"]["last_uid"] == 4  # moved past the read one too
    assert await a.poll(_ctx(imap, state=state)) == []  # n:* quirk returns the max: filtered
    # unread_only off: read mail counts too
    imap.add(b"From: a@b.c\r\nSubject: already read\r\n\r\nx\r\n", seen=True)
    items = await a.poll(_ctx(imap, state=state, unread_only=False))
    assert [i.title for i in items] == ["already read"]
    assert imap.violations == []  # never SELECT, STORE, or a FETCH without PEEK


async def test_uidvalidity_change_does_not_resend_the_mailbox(imap):
    for n in range(3):
        imap.add(f"From: a@b.c\r\nSubject: m{n}\r\n\r\nx\r\n".encode())
    state = {}
    a = EmailAdapter()
    await a.poll(_ctx(imap, state=state))
    imap.renumber(start=100)
    assert await a.poll(_ctx(imap, state=state)) == []  # renumbered: start again from the end
    assert state["cursor"] == {"folder": "INBOX", "uidvalidity": 1700000002, "last_uid": 102}
    uid = imap.add(b"From: a@b.c\r\nSubject: after\r\n\r\nx\r\n")
    items = await a.poll(_ctx(imap, state=state))
    assert [i.id for i in items] == [f"INBOX:1700000002:{uid}"]


async def test_another_folder_starts_from_its_end(imap):
    imap.folders["Work"] = {"uidvalidity": 9, "uidnext": 1, "msgs": []}
    imap.add(b"Subject: w-old\r\n\r\nx\r\n", folder="Work")
    state = {}
    a = EmailAdapter()
    await a.poll(_ctx(imap, state=state))
    assert await a.poll(_ctx(imap, state=state, folder="Work")) == []
    uid = imap.add(b"Subject: w-new\r\n\r\nx\r\n", folder="Work")
    items = await a.poll(_ctx(imap, state=state, folder="Work"))
    assert [i.id for i in items] == [f"Work:9:{uid}"] and "Folder: Work" in items[0].body


async def test_a_burst_is_read_oldest_first_a_batch_per_poll(imap):
    state = {}
    a = EmailAdapter()
    await a.poll(_ctx(imap, state=state))
    for n in range(30):
        imap.add(f"Subject: n{n}\r\n\r\nx\r\n".encode())
    first = await a.poll(_ctx(imap, state=state))
    second = await a.poll(_ctx(imap, state=state))
    assert [i.title for i in first] == [f"n{n}" for n in range(25)]
    assert [i.title for i in second] == [f"n{n}" for n in range(25, 30)]


async def test_poll_failures(imap):
    state = {}
    await EmailAdapter().poll(_ctx(imap, state=state))
    with pytest.raises(SourceFailure) as f:
        await EmailAdapter().poll(_ctx(imap, state=state, password="revoked"))
    assert f.value.code == "token_rejected" and f.value.retryable is False
    assert f.value.params == {"field": "pass"}
    imap.login_reply = "NO [UNAVAILABLE] Temporary server problem, try again later"
    with pytest.raises(SourceFailure) as f:
        await EmailAdapter().poll(_ctx(imap, state=state))
    assert f.value.code == "unreachable" and f.value.retryable
    imap.login_reply = None
    with pytest.raises(SourceFailure) as f:
        await EmailAdapter().poll(_ctx(imap, state=state, folder="Gone"))
    assert f.value.code == "folder_not_found" and f.value.params == {"folder": "Gone"}
    port = imap.port
    imap.close()
    ctx = _ctx(imap, state=state)
    ctx.config["port"] = str(port)
    with pytest.raises(SourceFailure) as f:
        await EmailAdapter().poll(ctx)
    assert f.value.code == "unreachable" and f.value.params == {"host": "127.0.0.1"}


async def test_folder_choices(imap):
    imap.folders["&UXZO1mWHTvZZOQ-"] = {"uidvalidity": 3, "uidnext": 1, "msgs": []}
    imap.folders["Sent Items"] = {"uidvalidity": 4, "uidnext": 1, "msgs": []}
    items = await EmailAdapter().choices(_ctx(imap), "folders", {})
    assert [(c.value, c.label) for c in items] == [
        ("INBOX", "INBOX"), ("Sent Items", "Sent Items"), ("&UXZO1mWHTvZZOQ-", "其他文件夹")]
    # a folder picked by its raw name, or typed in Chinese, both open
    await EmailAdapter().verify(_ctx(imap, folder="&UXZO1mWHTvZZOQ-"))
    await EmailAdapter().verify(_ctx(imap, folder="其他文件夹"))
    await EmailAdapter().verify(_ctx(imap, folder="Sent Items"))
    with pytest.raises(VerifyError) as e:
        await EmailAdapter().choices(_ctx(imap, password="x"), "folders", {})
    assert e.value.field == "pass"
