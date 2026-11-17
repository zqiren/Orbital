# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The source-type catalogue: one connect descriptor per type.

The panel renders its connect list and every form from this, so all thirteen
built-in types are here even before their adapter module exists; a type
whose adapter is missing reports ``available: false`` (``not_built``).

An adapter module may define ``DESCRIPTOR`` (a dict) to override or extend
its entry here without touching this file (see ``ADAPTERS.md``).
"""

from __future__ import annotations

import copy
import sys

ALL_PLATFORMS = ["darwin", "win32", "linux"]


def _f(key, label_key, *, secret=False, optional=False, default=None, placeholder=None,
       pair=False):
    return {"key": key, "label_key": label_key, "secret": secret, "optional": optional,
            "default": default, "placeholder": placeholder, "pair": pair}


def _k(key, label_key, *, default=False, params=None):
    return {"key": key, "label_key": label_key, "default": default, "params": params or []}


def _p(key, label_key, *, type="choice", choices=None, options=None, depends_on=None,
       required=True, default=None):
    return {"key": key, "label_key": label_key, "type": type, "choices": choices,
            "options": options, "depends_on": depends_on, "required": required,
            "default": default}


BOT_KINDS = [_k("dm", "pick.dm", default=True), _k("at", "pick.at", default=True)]

# ── the ready-made prompt a user can copy to any agent (prototype text, zh source) ──

_PROMPT_EXPLICIT = {
    "telegram": {
        "zh": "帮我把一个 Telegram 机器人接入 Orbital，作为任务来源。\n{skill_zh}\n1. 通过 @BotFather 用 /newbot 新建一个机器人（可以叫 Orbital），拿到机器人令牌。\n2. 要在群里@它：在 @BotFather 用 /setprivacy 选 Disable，再把它拉进群。\n3. 用 orbital-sources 技能注册一个 telegram 来源，令牌直接交给技能保存；收进工作台：私聊机器人、群里@机器人。\n4. 告诉我机器人的用户名。\n不要把令牌或任何密钥打印到对话里。需要我在 Telegram 里操作的地方，一步一步告诉我。",
        "en": "Connect a Telegram bot to Orbital as a task source.\n{skill_en}\n1. Create a bot with @BotFather (/newbot; it can be called Orbital) and get its token.\n2. For @mentions in groups: in @BotFather, /setprivacy → Disable, then add the bot to the group.\n3. Register a telegram source with the orbital-sources skill. Hand the token straight to the skill. Bring in: messages to the bot, @bot in groups.\n4. Tell me the bot's username.\nNever print the token or any secret into this chat. Where I have to act in Telegram, walk me through it step by step.",
    },
    "slack": {
        "zh": "帮我把一个 Slack 机器人接入 Orbital，作为任务来源。\n{skill_zh}\n1. 用应用清单新建一个 Slack 应用：开启 Socket Mode；机器人权限 app_mentions:read、im:history、users:read、channels:read；订阅事件 app_mention、message.im；在 App Home 打开 Messages 标签并允许发消息；安装到我的工作区。\n2. 生成带 connections:write 的应用级令牌（xapp-），并取得机器人令牌（xoxb-）。\n3. 用 orbital-sources 技能注册一个 slack 来源，两个令牌直接交给技能保存；收进工作台：私聊机器人、群里@机器人。\n不要把令牌或任何密钥打印到对话里。需要我在浏览器里确认的地方，一步一步告诉我。",
        "en": "Connect a Slack bot to Orbital as a task source.\n{skill_en}\n1. Create a Slack app from a manifest: Socket Mode on; bot scopes app_mentions:read, im:history, users:read, channels:read; events app_mention, message.im; in App Home turn on the Messages tab and allow messages; install it to my workspace.\n2. Generate an app-level token with connections:write (xapp-) and get the bot token (xoxb-).\n3. Register a slack source with the orbital-sources skill. Hand both tokens straight to the skill. Bring in: messages to the bot, @bot in groups.\nNever print a token or any secret into this chat. Where I have to confirm in the browser, walk me through it step by step.",
    },
    "email": {
        "zh": "帮我把我的邮箱接入 Orbital，作为任务来源。\n{skill_zh}\n1. 先问我用哪个邮箱地址。\n2. 确认它开启了 IMAP，并在邮箱的账号安全设置里生成一个应用专用密码。\n3. 用 orbital-sources 技能注册一个 email 来源：IMAP 服务器、端口 993、邮箱地址、应用专用密码，文件夹 INBOX，只收未读。密码直接交给技能保存。\n不要把密码或任何密钥打印到对话里。需要我登录或确认的地方，一步一步告诉我。",
        "en": "Connect my mailbox to Orbital as a task source.\n{skill_en}\n1. Ask me which address to use.\n2. Make sure IMAP is on, and create an app password in the account's security settings.\n3. Register an email source with the orbital-sources skill: IMAP server, port 993, address, app password, folder INBOX, unread only. Hand the password straight to the skill.\nNever print the password or any secret into this chat. Where I have to sign in or confirm, walk me through it step by step.",
    },
}

_TPL = {
    "zh": "帮我把{svc_}接入 Orbital，作为任务来源。\n{skill_zh}\n1. 在{svc_}那边：{setup}\n2. 用 orbital-sources 技能注册一个 {type} 来源，{creds_}直接交给技能保存；收进工作台：{picks}。\n不要把令牌或任何密钥打印到对话里。需要我操作的地方，一步一步告诉我。",
    "en": "Connect {svc} to Orbital as a task source.\n{skill_en}\n1. On {svc}: {setup}\n2. Register a {type} source with the orbital-sources skill. Hand the {creds} straight to the skill. Bring in: {picks}.\nNever print a token or any secret into this chat. Where I have to act, walk me through it step by step.",
}
_TPL_PARTS = {
    # type: (svc zh, svc en, setup zh, setup en, creds zh, creds en, picks zh, picks en)
    "discord": ("Discord", "Discord",
                "在开发者门户新建应用并添加机器人，把机器人邀请进我的服务器。",
                "create an application with a bot in the Developer Portal, invite the bot to my server.",
                "机器人令牌", "Bot token", "私聊机器人、群里@机器人", "Messages to the bot, @bot in groups"),
    "feishu": ("飞书", "Feishu",
               "在开放平台新建企业自建应用，开启机器人能力并开通“接收单聊消息”“接收群里@机器人的消息”权限，订阅“接收消息”事件并选择长连接，发布版本。",
               "create a custom app on the Open Platform, add the bot capability with permissions to receive direct messages and @bot group messages, subscribe to the receive-message event over a long connection, publish a version.",
               "App ID、App Secret", "App ID, App Secret", "私聊机器人、群里@机器人", "Messages to the bot, @bot in groups"),
    "dingtalk": ("钉钉", "DingTalk",
                 "在开发者后台新建应用，添加机器人能力并开启 Stream 模式，发布。",
                 "create an app in the developer console, add the robot capability with Stream mode on, publish it.",
                 "Client ID (AppKey)、Client Secret", "Client ID (AppKey), Client Secret",
                 "私聊机器人、群里@机器人", "Messages to the bot, @bot in groups"),
    "wecom": ("企业微信", "WeCom",
              "新建一个智能机器人，选择 API 模式和长连接，设置可见范围。",
              "create a smart bot in API mode with a long connection and set who can see it.",
              "Bot ID、Secret", "Bot ID, Secret", "私聊机器人、群里@机器人", "Messages to the bot, @bot in groups"),
    "jira": ("Jira", "Jira",
             "在 Atlassian 账号的安全设置里创建一个 API 令牌，并问我 Jira 站点地址和登录邮箱。",
             "create an API token in the Atlassian account security settings, and ask me for the Jira site and login e-mail.",
             "站点、邮箱、API 令牌", "Site, E-mail, API token", "指派给我、提到我", "Assigned to me, Mentions me"),
}


# How the agent finds the skill: the running daemon serves it. ``{skill_url}``
# is filled in when the catalogue is served (the daemon's actual port).
SKILL_LINE = {
    "zh": "先读 Orbital 的 orbital-sources 技能：curl -s {skill_url}",
    "en": "First read Orbital's orbital-sources skill: curl -s {skill_url}",
}


def with_skill_url(prompt: dict | None, skill_url: str) -> dict | None:
    if not prompt:
        return prompt
    return {lang: text.replace("{skill_url}", skill_url) for lang, text in prompt.items()}


def _zh_pad(word: str, *, left: bool = True) -> str:
    """A Latin word in Chinese text gets a space on each side it meets a
    Chinese character («帮我把 Discord 接入», «Bot ID、Secret 直接交给»)."""
    if not word or not word[-1].isascii():
        return word
    return (" " if left and word[0].isascii() else "") + word + " "


def _tpl_prompt(type_: str) -> dict:
    svc_zh, svc_en, set_zh, set_en, cr_zh, cr_en, pk_zh, pk_en = _TPL_PARTS[type_]
    return {
        "zh": _TPL["zh"].replace("{svc_}", _zh_pad(svc_zh)).replace("{setup}", set_zh)
        .replace("{type}", type_).replace("{creds_}", _zh_pad(cr_zh, left=False))
        .replace("{picks}", pk_zh),
        "en": _TPL["en"].replace("{svc}", svc_en).replace("{setup}", set_en)
        .replace("{type}", type_).replace("{creds}", cr_en).replace("{picks}", pk_en),
    }


def _prompt(type_: str) -> dict | None:
    if type_ in _PROMPT_EXPLICIT:
        p = dict(_PROMPT_EXPLICIT[type_])
    elif type_ in _TPL_PARTS:
        p = _tpl_prompt(type_)
    else:
        return None
    return {lang: text.replace("{skill_" + lang + "}", SKILL_LINE[lang]) for lang, text in p.items()}


def _entry(type_, family, connect, kinds, *, interval_s=None, lossy=False, platforms=None,
           setup_link=None):
    return {
        "type": type_, "family": family, "platforms": platforms or list(ALL_PLATFORMS),
        "interval_s": interval_s, "lossy": lossy, "connect": connect, "kinds": kinds,
        "setup_link": setup_link, "agent_prompt": _prompt(type_),
    }


def current_platform() -> str:
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


# The one-line install per platform (Homebrew on a Mac, winget on Windows);
# elsewhere none, and the panel links to ``install_url`` instead.
_INSTALL = {
    "gh": {"darwin": "brew install gh", "win32": "winget install --id GitHub.cli"},
    "glab": {"darwin": "brew install glab", "win32": "winget install glab.glab"},
}


def install_command(cli: str, platform: str | None = None) -> str | None:
    return _INSTALL.get(cli, {}).get(platform or current_platform())


def _cli(cli, host, install_url):
    return {"method": "cli_account", "cli": cli, "host": host,
            "sign_in_command": f"{cli} auth login", "install_command": install_command(cli),
            "install_url": install_url}


def _fields(*fields, method="fields"):
    return {"method": method, "fields": list(fields)}


BUILTIN_ORDER = ["github", "gitlab", "jira", "email", "calendar", "website",
                 "telegram", "slack", "discord", "feishu", "dingtalk", "wecom"]

_BASE: dict[str, dict] = {
    "github": _entry("github", "account", _cli("gh", "github.com", "https://cli.github.com"), [
        _k("assigned", "pick.assigned", default=True),
        _k("mention", "pick.mention"),
        _k("label", "pick.label", params=[
            _p("repo", "repo.ph", choices="repos"),
            _p("label", "label.ph", choices="labels", depends_on="repo"),
        ]),
    ], interval_s=60),
    "gitlab": _entry("gitlab", "account", _cli("glab", "gitlab.com", "https://gitlab.com/gitlab-org/cli"), [
        _k("assigned", "pick.assigned", default=True),
        _k("mention", "pick.mention"),
        _k("review", "pick.review"),
    ], interval_s=300),
    "jira": _entry("jira", "account", _fields(
        _f("site", "fl.site", placeholder="your-team.atlassian.net"),
        _f("mail", "fl.mail", placeholder="you@company.com"),
        _f("token", "fl.jtoken", secret=True, placeholder="ATATT3x…"),
    ), [_k("assigned", "pick.assigned", default=True), _k("mention", "pick.mention")],
        interval_s=300, setup_link="https://id.atlassian.com/manage-profile/security/api-tokens"),
    "email": _entry("email", "account", _fields(
        _f("acct", "fl.acct", placeholder="you@example.com"),
        _f("pass", "fl.pass", secret=True, placeholder="xxxx xxxx xxxx xxxx"),
        _f("server", "fl.server", placeholder="imap.example.com", pair=True),
        _f("port", "fl.port", default="993", placeholder="993", pair=True),
    ), [_k("new_mail", "pick.newmail", default=True, params=[
        _p("folder", "fl.folder", choices="folders", default="INBOX", required=False),
        _p("unread_only", "pick.unread", type="bool", default=True, required=False),
    ])], interval_s=120),
    # ``pick``: one source-level multi-select both kinds read (no key stored
    # = every calendar; see ``SourcesCore._validate_config``).
    "calendar": _entry("calendar", "account", {"method": "none", "permission": "calendar",
                                               "pick": {"key": "calendars", "choices": "calendars",
                                                        "multi": True}}, [
        _k("new_event", "pick.newev"),
        _k("before", "pick.before", default=True, params=[
            _p("minutes", "mins", options=[10, 15, 30, 60], default=30, required=False),
        ]),
    ], interval_s=60, platforms=["darwin"]),
    "website": _entry("website", "check", _fields(
        _f("url", "web.addr", placeholder="https://"),
        _f("must_contain", "web.has", optional=True, placeholder="web.opt"),
    ), [_k("down", "web.down", default=True)], interval_s=120),
    "telegram": _entry("telegram", "bot", _fields(
        _f("token", "fl.btoken", secret=True, placeholder="123456789:AAH…"),
    ), BOT_KINDS, lossy=False, setup_link="https://t.me/BotFather"),
    "slack": _entry("slack", "bot", _fields(
        _f("app", "fl.app", secret=True, placeholder="xapp-1-…"),
        _f("bot", "fl.btoken", secret=True, placeholder="xoxb-…"),
    ), BOT_KINDS, lossy=True, setup_link="https://api.slack.com/apps?new_app=1"),
    "discord": _entry("discord", "bot", _fields(
        _f("token", "fl.btoken", secret=True, placeholder="MTI4…"),
    ), BOT_KINDS, lossy=True, setup_link="https://discord.com/developers/applications"),
    "feishu": _entry("feishu", "bot", _fields(
        _f("appid", "fl.appid", placeholder="cli_…"),
        _f("appsec", "fl.appsec", secret=True),
        method="qr",
    ), BOT_KINDS, lossy=True, setup_link="https://open.feishu.cn/app"),
    "dingtalk": _entry("dingtalk", "bot", _fields(
        _f("cid", "fl.cid", placeholder="ding…"),
        _f("csec", "fl.csec", secret=True),
    ), BOT_KINDS, lossy=True, setup_link="https://open-dev.dingtalk.com"),
    "wecom": _entry("wecom", "bot", _fields(
        _f("bid", "fl.bid", placeholder="aib…"),
        _f("bsec", "fl.bsec", secret=True),
    ), BOT_KINDS, lossy=True, setup_link="https://work.weixin.qq.com/wework_admin/frame"),
    # Registered from outside (an agent, through the orbital-sources skill).
    # Never in the connect list.
    "command": _entry("command", "external", {"method": "none"}, [
        _k("item", "kind.ext", default=True)], interval_s=300),
}

ALL_TYPES = BUILTIN_ORDER + ["command"]


def base_entry(type_: str) -> dict | None:
    e = _BASE.get(type_)
    return copy.deepcopy(e) if e is not None else None


def merged_entry(type_: str, override: dict | None) -> dict | None:
    """The catalogue entry with an adapter module's ``DESCRIPTOR`` laid over
    it (top-level keys replace)."""
    e = base_entry(type_)
    if e is None:
        if not override:
            return None
        e = {"type": type_}
    if override:
        for k, v in override.items():
            if k != "type":
                e[k] = copy.deepcopy(v)
    return e


def secret_fields(entry: dict) -> list[str]:
    return [f["key"] for f in (entry.get("connect") or {}).get("fields", []) if f.get("secret")]


def plain_fields(entry: dict) -> list[dict]:
    return [f for f in (entry.get("connect") or {}).get("fields", []) if not f.get("secret")]


def multi_pick(entry: dict) -> dict | None:
    """The connect's source-level multi-select (calendar: which calendars)."""
    pick = (entry.get("connect") or {}).get("pick")
    return pick if isinstance(pick, dict) and pick.get("multi") and pick.get("key") else None


def default_kinds(entry: dict) -> dict:
    out = {}
    for k in entry.get("kinds", []):
        if k.get("default"):
            out[k["key"]] = {p["key"]: p.get("default") for p in k.get("params", [])
                             if p.get("default") is not None}
    return out
