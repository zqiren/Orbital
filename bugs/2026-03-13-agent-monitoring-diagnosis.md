# Agent Monitoring Diagnosis Report — 2026-03-13

**Date:** 2026-03-13 02:00–02:45 UTC+8
**Agents monitored:** Quick Tasks (ClawExam v1 & v2), x-manager (Twitter/Reddit)
**Observer:** Claude Code (message injection + approval only)

---

## Issue 1: Shell Tool Broken in Sandbox — `CreateProcessWithLogonW error 267`

**Severity:** Critical — completely blocks all shell/curl usage
**Affected agents:** All agents using the `shell` tool in sandbox mode

### Symptoms

Every `shell` tool call fails immediately:
```
Error: CreateProcessWithLogonW failed with error code 267
```
After 3 consecutive failures, the system disables the shell tool for the session:
```
[SYSTEM] The shell tool is currently unavailable (repeated error: ...)
```

### Root Cause

Win32 error 267 = `ERROR_DIRECTORY` ("The directory name is invalid").

`CreateProcessWithLogonW` is called in `agent_os/platform/windows/process.py:246` with `working_dir` set to the project workspace. The directory exists and has full-control ACLs for `AgentOS-Worker`:

```
D:\orbital-public\.agent-os-data\scratch
  AgentOS-Worker:(OI)(CI)(F)   ← Full Control, inherited
```

The `os.path.isdir()` check at line 228 passes (the directory exists for the current user), but `CreateProcessWithLogonW` runs under the `AgentOS-Worker` account which may lack a user profile or have a different drive mapping. Error 267 can occur when:

1. The sandbox user's profile hasn't been created yet (`LOGON_NO_PROFILE` flag at line 250 skips profile load, but the directory resolution may still require profile state)
2. The working directory path is valid for the current user but unresolvable under the sandbox user's context
3. The sandbox user password was recently reset (noted: `Password last set: 2026/3/13 2:20:55`) — the process launcher re-created the user but the profile/environment state is stale

### Impact

- **v2 exam (failed):** 4 shell errors → agent fell back to browser UI navigation → wasted 30 minutes → exam token expired → total failure
- **v1 exam (succeeded with intervention):** 1 shell error → agent was pre-instructed to use browser `fetch()` → completed exam via fallback

### Recommendation

1. **Investigate** why `CreateProcessWithLogonW` rejects the directory when ACLs are correct. Test with a minimal reproduction: call `launch()` with a simple command (`cmd /c echo hello`) and the scratch workspace.
2. **Consider** using `CreateProcessAsUserW` instead, which has different profile-loading behavior.
3. **Fallback path:** When shell fails, the system already emits a helpful system message after 3 failures. Consider auto-injecting a guidance message: "Use browser JavaScript fetch() for HTTP requests."

---

## Issue 2: Chinese Text Encoding Lost in CLI Message Injection

**Severity:** Medium — messages become unreadable but agents can often infer intent

### Symptoms

Chinese characters in injected messages are replaced with `?`:
```
Stored in session: ??? https://exam.clawhome.cc/exam/v1.md ??????????
Original intended:  请阅读 https://exam.clawhome.cc/exam/v1.md 并按照其中的指引完成
```

### Root Cause

The inject API (`POST /api/v2/agents/{id}/inject`) correctly handles UTF-8 — verified by injecting Chinese via Python `urllib` with explicit UTF-8 encoding, which round-trips correctly.

The corruption occurs in the **bash terminal layer**. The Windows terminal uses cp1252/cp936 encoding, not UTF-8. When `curl` receives the JSON body from bash heredoc or inline string, non-ASCII bytes are mangled before reaching the HTTP client.

Proof:
```bash
echo '{"content":"测试"}' | cat -v
# Output: M-fM-5M-^K... (corrupted bytes)
```

Meanwhile, Python with explicit UTF-8:
```python
urllib.request.Request(url, json.dumps({'content': '测试中文'}).encode('utf-8'))
# Result: stored correctly with CJK preserved
```

Messages sent from the desktop app UI are correctly encoded (the v2 exam's original user message was `请阅读...` with CJK intact).

### Impact

- The v1 exam agent received garbled instructions but successfully inferred the task from the URL and context
- Agent confusion was minimal because the English fallback message (entry 7) provided clear instructions

### Recommendation

1. **For CLI injection:** Use Python scripts instead of `curl` for messages containing non-ASCII text
2. **In the inject API:** Consider adding an encoding validation/normalization layer that detects and rejects/replaces invalid UTF-8 sequences with a warning
3. **In the desktop app:** Not affected — the UI correctly sends UTF-8

---

## Issue 3: `check_in` Autonomy Creates Excessive Approval Prompts

**Severity:** Medium — slows agent execution significantly, requires constant human attention

### Symptoms

During the v1 exam (11 minutes), the Quick Tasks agent required **~20 manual approvals** — one for every browser `evaluate` call (JavaScript `fetch()`). Each approval paused the agent for 3-5 seconds while waiting for the approval loop to cycle.

Approval sequence pattern:
```
[02:30:53] APPROVING browser (browser:6)   ← check exam data
[02:31:08] APPROVING browser (browser:7)   ← register
[02:31:37] APPROVING browser (browser:9)   ← register retry
[02:32:02] APPROVING browser (browser:11)  ← submit Q1
[02:32:37] APPROVING browser (browser:15)  ← submit Q2
...
[02:38:38] APPROVING browser (browser:43)  ← submit Q15
```

### Root Cause

The Quick Tasks project has `autonomy: "check_in"`, which requires approval for every tool call. This is appropriate for destructive operations (file writes, system commands) but excessive for read-only browser fetch calls.

The approval system is binary — either all tool calls need approval or none do. There's no per-tool or per-session trust escalation.

### Impact

- **Time waste:** An 11-minute exam could have been ~5 minutes without approval delays
- **Human burden:** Requires dedicated human attention throughout the entire agent run
- **Automation gap:** Cannot leave `check_in` agents unattended even for safe operations

### Recommendation

1. **Tool-level autonomy:** Allow `check_in` to be configured per tool type (e.g., `browser.evaluate: hands_off`, `shell: check_in`, `file_write: check_in`)
2. **Session trust escalation:** "Approve all browser calls for this session" button
3. **Approve-all for safe tools:** The `approve_all` field exists in the API (`ApproveRequest.approve_all`) — expose this in the UI with appropriate guardrails

---

## Issue 4: Agent Gets Lost Without Shell — No Strategic Fallback

**Severity:** High — caused complete exam failure (v2 attempt)

### Symptoms

When the shell tool became unavailable during the v2 exam, the agent:
1. Attempted curl 3 times → all failed with error 267
2. System correctly disabled shell and told agent to use other tools
3. Agent switched to browser but **navigated the exam website UI** instead of using the API
4. Spent 30 minutes clicking around the website, parsing page content, trying to find exam questions visually
5. Exam token expired (30-minute limit)
6. Re-registered, got one question, then API key died

### Root Cause

The agent (Kimi K2.5) lacks a robust fallback strategy when its primary tool (shell/curl) is unavailable. The system message says "Work with the tools that are available" but doesn't provide guidance on HOW to use the browser for API calls.

In contrast, the v1 exam succeeded because the injection message explicitly instructed: "Use browser JavaScript fetch() for ALL API calls" with exact code templates.

### Evidence

| Metric | v2 (no guidance) | v1 (with guidance) |
|--------|------------------|--------------------|
| Session entries | 183 | 93 |
| Shell errors | 4 | 1 |
| Invalid token errors | 12 | 0 |
| Exam completed | No | Yes (94/100) |
| Time to first answer | Never | ~2 min |

### Recommendation

1. **Enhanced fallback message:** When shell is disabled, inject a richer guidance message:
   ```
   Shell is unavailable. For HTTP requests, use browser JavaScript:
   fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
   body:JSON.stringify(data)}).then(r=>r.json())
   ```
2. **Tool capability metadata:** Expose to the agent what each tool can do as a fallback for another (browser can replace curl, etc.)
3. **Agent-level retry guidance in system prompt:** Add to project goals or user directives: "If shell fails, use browser evaluate with JavaScript fetch() for HTTP requests"

---

## Issue 5: API Key Loss Across Sessions

**Severity:** High — causes agent death mid-task and triggers setup wizard

### Symptoms

Both agents (x-manager and Quick Tasks) died mid-session with:
```
LLM error (non-recoverable): Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}
```

The setup wizard appears every time the desktop app opens because `GET /api/v2/settings` returns `api_key_set: false`.

### Root Cause

The API key storage path works correctly:
- **Backend:** `PUT /api/v2/settings/api-key` → `keyring.set_password("agent-os", "llm-api-key", key)` → Windows Credential Manager ✓
- **Read-back:** `keyring.get_password("agent-os", "llm-api-key")` returns stored key ✓
- **Daemon restart:** Key persists in keyring across daemon restarts ✓

The key was simply **not in keyring** at the start of this session. Verified by live testing:
1. Entered key in wizard → stored correctly → survived daemon restart
2. Previously the key was absent — either never saved or cleared by an uninstall (`--teardown-sandbox` deletes keyring entry)

### Impact

- x-manager's monitoring run died at entry 123/124 after completing most of its work
- Quick Tasks v2 exam died after re-registering and receiving first question
- User had to re-enter key in wizard

### Recommendation

1. **Redundant storage:** Write API key to both keyring and an encrypted local file as backup
2. **Graceful degradation:** When LLM returns 401, pause the agent and notify user rather than killing the session — allow key re-entry without losing session state
3. **Startup validation:** On daemon start, verify the API key is valid (make a test call) and warn immediately rather than failing mid-task

---

## Summary Table

| # | Issue | Severity | Root Cause | Quick Fix |
|---|-------|----------|------------|-----------|
| 1 | Shell broken (error 267) | Critical | `CreateProcessWithLogonW` rejects workspace dir for sandbox user | Investigate sandbox user profile; add fetch() fallback guidance |
| 2 | Chinese encoding lost | Medium | Bash terminal cp1252 encoding corrupts UTF-8 before curl sends | Use Python for non-ASCII injection; add API-level validation |
| 3 | Excessive approvals | Medium | `check_in` is all-or-nothing per session | Add per-tool autonomy levels or session trust escalation |
| 4 | No strategic fallback | High | Agent doesn't know browser can replace shell for HTTP | Enrich shell-failure system message with fetch() templates |
| 5 | API key loss | High | Key absent from keyring; 401 kills agent instantly | Add backup storage + graceful pause on auth failure |

---

## Appendix: Intervention Log

| Time | Agent | Action | Reason |
|------|-------|--------|--------|
| 02:25 | x-manager | Injected "continue routine" message | Agent died mid-report with 401; resumed with fresh API key |
| 02:25 | Quick Tasks | Injected exam task (v1) with fetch() instructions | Previous v2 attempt failed; provided explicit API guidance |
| 02:26 | Quick Tasks | Approved shell command (shell:1) | Agent tried curl; expected sandbox failure |
| 02:28 | Quick Tasks | Injected confirmation + fetch() templates | Agent stopped waiting for user confirmation per exam rules |
| 02:30 | Quick Tasks | Approved browser:5 (register fetch) | First browser evaluate call |
| 02:30-02:41 | Quick Tasks | Auto-approved ~18 browser evaluate calls | Routine fetch() calls for exam Q&A submission |
| — | x-manager | No intervention needed after initial resume | Completed report + PROJECT_STATE update autonomously |
