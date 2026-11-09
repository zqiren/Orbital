# 从脚本发任务到工作台 / Send tasks to the Workbench from a script

## 中文

`POST /api/v2/workbench/tasks` 做的事和工作台输入框一样：把一段文字变成一个任务。Orbital 新开一个会话、发出消息，卡片出现在工作台顶部，所有设备都能看到。GitHub、Linear、邮件、定时任务……谁来监听是你自己的脚本；Orbital 只负责「任务进来，会话开始，卡片出现」。

```bash
curl -s -X POST http://127.0.0.1:8000/api/v2/workbench/tasks \
  -H 'content-type: application/json' \
  -d '{"text": "#123 登录页空密码报 500\n\n<issue 正文>", "project_id": "proj_ab12"}'
# → {"project_id": "proj_ab12", "session_id": "…", "agent": "orbital", "by": "given", "inject": {"status": "started"}}
```

字段：

- `text`（必填）：任务内容。第一行就是卡片标题。
- `project_id`（可选）：发到哪个项目。项目 id 在项目设置里，或 `GET /api/v2/projects`。不填时 Orbital 用自己的模型选项目（每次约几千 token，返回里的 `input_tokens` 会写明）；没有 API key 时进 Quick Tasks。返回的 `by` 说明项目是怎么定的：`given`（你指定的）、`model`（模型选的）、`fallback`（进了 Quick Tasks，`reason` 写原因）。
- `agent`（可选）：`orbital`，或某个子代理，比如 `claude-code`、`codex`（必须在这个项目里可用）。不填就用项目当前的默认代理，和输入框的「自动」一样。只影响这一个任务，不会改项目的固定代理。
- `attachments`（可选）：和聊天发送的附件格式一样。

错误：`422` 文字为空；`404` 项目不存在或不在工作台上；`400` 代理不可用或附件无效；`502` 没填项目且自动选项目失败（什么都没发，请带上 `project_id` 重试）。这些错误都在开会话之前返回，不会留下任何东西。代理启动失败这类发送后的错误，会照常显示在那个会话里。

要看结果，轮询 `GET /api/v2/workbench/feed`（每行有 `status`、`result_line`、`ask`），或读会话的聊天记录。超时后重试可能会开出两个会话，按返回的 `session_id` 自己去重。

它在本机运行，和其它本地 API 一样，没有额外的鉴权或令牌。

示例：把带 `orbital` 标签的 GitHub issue 各发成一个任务：

```bash
gh issue list --label orbital --json number,title,body,url \
  | jq -c '.[] | {project_id: "proj_ab12", text: "#\(.number) \(.title)\n\(.url)\n\n\(.body)"}' \
  | while read -r task; do
      curl -s -X POST http://127.0.0.1:8000/api/v2/workbench/tasks \
        -H 'content-type: application/json' -d "$task" \
        | jq -r '"\(.by) \(.project_id) \(.session_id)"'
    done
```

## English

`POST /api/v2/workbench/tasks` does what the Workbench composer does: it turns text into a task. Orbital starts a session and sends the message, and the card shows up at the top of the Workbench on every device. Your own script does the watching (GitHub, Linear, email, a cron job). Orbital's side is "a task comes in, a session starts, a card appears".

```bash
curl -s -X POST http://127.0.0.1:8000/api/v2/workbench/tasks \
  -H 'content-type: application/json' \
  -d '{"text": "#123 Login page 500s on empty password\n\n<issue body>", "project_id": "proj_ab12"}'
# → {"project_id": "proj_ab12", "session_id": "…", "agent": "orbital", "by": "given", "inject": {"status": "started"}}
```

Fields:

- `text` (required): the task. Its first line becomes the card title.
- `project_id` (optional): the project to send it to. Project ids are in project settings, or in `GET /api/v2/projects`. If you leave it out, Orbital picks the project with its own model (a few thousand tokens per call, shown as `input_tokens` in the response). With no API key, the task goes to Quick Tasks. `by` says how the project was chosen: `given` (you named it), `model` (the model picked it) or `fallback` (it went to Quick Tasks, and `reason` says why).
- `agent` (optional): `orbital`, or a worker such as `claude-code` or `codex` that is available in that project. If you leave it out, the project's current agent handles it, the same as the composer's Auto. This applies to this one task and doesn't change the project's pinned agent.
- `attachments` (optional): the same format a chat send uses.

Errors: `422` for empty text. `404` if the project doesn't exist or isn't on the Workbench. `400` if the agent isn't available or an attachment is invalid. `502` if no project was given and routing failed (nothing was sent, so retry with a `project_id`). All of these come back before a session is created, so nothing is left behind. A failure after sending (a worker that cannot start, say) shows up in that session like any failed run.

To follow a task, poll `GET /api/v2/workbench/feed` (each row has `status`, `result_line` and `ask`) or read the session's chat. A retry after a timeout can create two sessions, so dedupe on the `session_id` you got back.

The endpoint runs on this machine and is reachable the same way as the rest of the local API, with no extra auth or token.

Example: send one task per GitHub issue labelled `orbital`:

```bash
gh issue list --label orbital --json number,title,body,url \
  | jq -c '.[] | {project_id: "proj_ab12", text: "#\(.number) \(.title)\n\(.url)\n\n\(.body)"}' \
  | while read -r task; do
      curl -s -X POST http://127.0.0.1:8000/api/v2/workbench/tasks \
        -H 'content-type: application/json' -d "$task" \
        | jq -r '"\(.by) \(.project_id) \(.session_id)"'
    done
```
