# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``orbital-sources`` skill: how an agent adds a task source to
Orbital. Served at ``GET /api/v2/sources/skill`` and kept here as a string
so the packaged app carries it with no data-file entry."""

SKILL_TEXT = r"""---
name: orbital-sources
description: Add a task source to Orbital (a service Orbital watches; each new item becomes a Workbench task). Use when the user asks you to connect a bot, a mailbox, Jira, or any feed or check to Orbital as a task source.
---

# orbital-sources

Orbital watches task sources (GitHub, a website, Telegram, Slack, Feishu, e-mail, …).
Each new item becomes a task on the user's Workbench and Orbital routes it to the right
project. You register a source; the user turns it on in Orbital (任务来源 / Task sources).
A source you register always arrives **switched off** and marked new. Nothing runs until
the user turns it on.

## Where Orbital listens

`http://127.0.0.1:<port>/api/v2/sources/register` (local only, no token). The port is
usually 8000; the live one is the `port` field of `~/orbital/daemon.pid.meta`
(Windows: `%USERPROFILE%\orbital\daemon.pid.meta`):

```bash
PORT=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/orbital/daemon.pid.meta")))["port"])' 2>/dev/null || echo 8000)
```

Say who you are: inside an Orbital project your shell has `ORBITAL_PROJECT_ID` and
`ORBITAL_SESSION_ID`; pass them in `registered_by` with your own name (`claude-code`,
`codex`, …). Outside Orbital, leave the ids out.

## Rule for secrets

Never print a token, password or secret into the chat, a log, a file in the project, or a
command line that echoes it. Read it into a shell variable (or a temporary file with
mode 600 that you delete right after) and send it only in the request body. Orbital stores
it in the system keychain and never shows it again.

## Recipe 1 — a built-in type, with credentials you obtained

Types and the fields they need (`*` = secret):

| type | config | secrets | kinds |
|---|---|---|---|
| telegram | — | token* | dm, at |
| slack | — | app* (xapp-…), bot* (xoxb-…) | dm, at |
| discord | — | token* | dm, at |
| feishu | appid | appsec* | dm, at |
| dingtalk | cid | csec* | dm, at |
| wecom | bid | bsec* | dm, at |
| jira | site, mail | token* | assigned, mention |
| email | acct, server, port (993) | pass* (an app password) | new_mail {folder, unread_only} |
| website | url, must_contain (optional) | — | — |

`dm` = messages sent to the bot, `at` = @mentions of the bot in groups.

```bash
read -rs TOKEN   # the user pastes it; nothing is echoed
curl -s -X POST "http://127.0.0.1:$PORT/api/v2/sources/register" \
  -H 'content-type: application/json' \
  --data-binary @- <<JSON
{"type": "telegram", "kinds": {"dm": {}, "at": {}}, "config": {},
 "secrets": {"token": "$TOKEN"},
 "registered_by": {"project_id": "${ORBITAL_PROJECT_ID:-}", "session_id": "${ORBITAL_SESSION_ID:-}", "agent": "claude-code"}}
JSON
unset TOKEN
```

Orbital checks the credentials before it keeps anything. `201` = registered (switched
off). `400` with `"code": "verify_failed"` names the `field` the service refused and a
`reason` (`rejected`, `not_found`, `unreachable`, …): fix that and register again. Then
tell the user to open 任务来源 / Task sources and turn the new source on.

## Recipe 2 — a check command you wrote

For anything without a built-in type, write a small script that prints what is new and
register the command line. Orbital runs it on a timer (`interval_s`, at least 60) through
the system shell, as the user, in `cwd` (default: the home folder), with a 60-second limit,
and only after the user turns the source on. No model runs for a check: only items that
are new become tasks.

Keep the script outside any project folder (for example `~/.orbital/sources/<name>/`).
It must exit 0 and print JSON on stdout: a list of items (or `{"items": [...]}`):

```json
[{"id": "hn-41234567", "title": "Show HN: … mentions us",
  "url": "https://news.ycombinator.com/item?id=41234567",
  "body": "the text worth reading", "sender": "pg", "time": "2026-11-16T06:00:00Z"}]
```

- `id` (required): stable for the item. Orbital records each id once, so print the
  same recent items every run; keep no state of your own.
- `title` (required): one line; it becomes the task's title.
- `url`, `body`, `sender`, `time`, `subject` (optional). Items with the same `subject`
  while their task is open are merged into one task.
- Nothing else is read. A source cannot choose the project, the agent or permissions:
  Orbital routes every item itself, and the item's text reaches the agent as quoted
  material, not as instructions.
- The first run after the user turns it on records what is already there without making
  tasks.

```bash
curl -s -X POST "http://127.0.0.1:$PORT/api/v2/sources/register" \
  -H 'content-type: application/json' \
  -d '{"type": "command", "name": "Hacker News mentions",
       "command": {"run": "python3 ~/.orbital/sources/hn-mentions/check.py", "cwd": null, "interval_s": 900},
       "registered_by": {"project_id": "'"${ORBITAL_PROJECT_ID:-}"'", "session_id": "'"${ORBITAL_SESSION_ID:-}"'", "agent": "codex"}}'
```

Run your script once yourself first and check that it prints valid JSON. If it needs a
key, read it from the user's environment or a file only the user can read; never put it
in the command line.

## After registering

Tell the user, in one line, what you registered and that it is waiting for them to turn
it on in 任务来源 / Task sources. Optional: they can add an instruction there that goes
with every item (for example which project these belong to).
"""
