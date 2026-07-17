# LOG — Awesome-list submissions for Orbital

Date: 2026-07-08 · GitHub account used: **`zqiren`** (the Orbital repo owner, so the author-disclosure line is truthful).

**Positioning decision (agreed with Qiren):** every entry leads with Orbital's
differentiator — *a plain-markdown folder as the agent's persistent memory* — because the
relevant sections are saturated with "Claude Code / Codex orchestrator" peers and a
generic orchestration line would read as duplicative.

**Status: 5 draft PRs opened, 1 skipped, 2 blocked (surfaced as questions). Zero PRs marked ready-for-review.**

All facts sourced only from the task's Facts section. No claims of network isolation,
Linux support, or sub-agent containment anywhere. Sandbox language is scoped to the
"built-in shell" only.

---

## 1. kyrolabs/awesome-agents  →  ✅ DRAFT PR

- **Canonical check:** ✅ active & canonical — 2,574★, default branch `main`, last push 2026-07-07.
- **CONTRIBUTING rules:**
  - Open-source only; must be high-quality, maintained, show traction; PR (not issue).
  - **Place new items at the _bottom_** of the list; criteria are auto-managed.
  - Top auto-close triggers: *brand-new repo with no history, brand-new user, wrong place in list.* (Orbital = 4-month repo w/ history; `zqiren` account = 5 yrs old → clears both. Modest 43★ traction is a soft risk — noted.)
- **Action:** Draft PR → https://github.com/kyrolabs/awesome-agents/pull/614
- **Section:** Software Development (appended at end, alongside Dorothy/Maestro/Bernstein/h5i).
- **Exact line added:**
  ```
  - [Orbital](https://github.com/zqiren/Orbital): Local-first desktop agent runtime that uses a folder of plain markdown as the agent's persistent memory (a two-tier injected index plus on-demand files), then dispatches Claude Code and Codex as sub-agents in parallel with approval workflows, per-project budgets, and OS-sandboxed built-in shell commands (Seatbelt on macOS, low-privilege user on Windows). ![GitHub Repo stars](https://img.shields.io/github/stars/zqiren/Orbital?style=social)
  ```

---

## 2. aloth/awesome-ai-agents  →  ⛔ SKIPPED (fails eligibility gate)

- **Canonical check:** active (last push 2026-07-05) but a very small/new list (6★).
- **CONTRIBUTING rules:** actively-maintained only; alphabetical; ≤15-word description.
- **Failing criterion (quoted, from "Quality Checks"):**
  > "Project has >100 GitHub stars OR is from a major organization"

  Orbital has **43★** and `zqiren` is not a major organization → does not meet the gate.
- **Action:** No PR opened. Recorded and moved on per the eligibility rule.

---

## 3. e2b-dev/awesome-ai-agents  →  ⚠️ DRAFT PR (staleness flagged)

- **Canonical check:** ✅ canonical (THE e2b list, 28,652★) **but STALE — last commit 2025-02-26 (~17 months).** PR may sit unmerged; flagged for Qiren.
- **CONTRIBUTING rules (from README, no CONTRIBUTING.md):**
  - "Create a pull request … keep the alphabetical order and in the correct category."
  - Per-project **structured block** format: `## [Name](url)` + tagline + `<details>` with `### Category`, `### Description`, `### Links`.
- **Action:** Draft PR → https://github.com/e2b-dev/awesome-ai-agents/pull/1217
- **Section:** Open-source projects (inserted alphabetically between "Open Interpreter" and "Pezzo").
- **Exact block added:**
  ```
  ## [Orbital](https://github.com/zqiren/Orbital)
  Local-first desktop agent runtime with a plain-markdown folder as persistent memory

  <details>

  ### Category
  Coding, Multi-agent

  ### Description
  - Uses a local folder of plain markdown as the agent's persistent memory — a two-tier model with an always-injected index plus on-demand files.
  - Dispatches Claude Code (official Agent SDK) and Codex (App Server protocol) as sub-agents on your existing subscription, with parallel fan-out.
  - Approval workflows, per-project budgets, and a cron/task queue.
  - Built-in agent shell commands run under OS sandboxing (Seatbelt on macOS; a dedicated low-privilege user on Windows).
  - Desktop app for macOS (Apple Silicon) and Windows. Open source (GPL-3.0).

  ### Links
  - [GitHub](https://github.com/zqiren/Orbital)

  </details>
  ```

---

## 4. Jenqyang/Awesome-AI-Agents  →  ✅ DRAFT PR

- **Canonical check:** ✅ active & canonical — 1,182★, `main`, last push 2026-07-06.
- **CONTRIBUTING rules:**
  - Standard OSS license required; "license clear in GitHub metadata **or the repo itself**."
  - One entry per PR; neutral one-line description; no ad-style wording.
  - Format: `- [Name](url) - Neutral description. ![GitHub Repo stars …?style=social]`
  - **License note:** GitHub reports Orbital's license as "Other/NOASSERTION" (custom header before the GPL text), but the `LICENSE` file body **is** GPL-3.0 → satisfies the "or the repo itself" clause. The PR body states this explicitly to pre-empt a reviewer license check.
- **Action:** Draft PR → https://github.com/Jenqyang/Awesome-AI-Agents/pull/362
- **Section:** Applications → Autonomous Agent Task Solver Projects (appended at end).
- **Exact line added:**
  ```
  - [Orbital](https://github.com/zqiren/Orbital) - Local-first desktop agent runtime that uses a folder of plain markdown as the agent's persistent memory, then dispatches Claude Code and Codex as sub-agents in parallel with approvals, per-project budgets, a task queue, and OS-sandboxed built-in shell commands (Seatbelt on macOS, low-privilege user on Windows). ![GitHub Repo stars](https://img.shields.io/github/stars/zqiren/Orbital?style=social)
  ```

---

## 5. slavakurilyak/awesome-ai-agents  →  ❓ BLOCKED (question for Qiren)

- **Canonical check:** the canonical Slava list (1,613★) **but STALE — last push 2025-09-09 (~10 months).**
- **CONTRIBUTING rules:** **none** — no CONTRIBUTING.md and no "Contributing" section in the README.
- **Why blocked (ambiguity, not guessed):** every entry carries an identical machine stamp
  (`⭐ N stars (Updated: 2025-07-30)` + baked-in star badge + a computed "Top 10 / Rising 10"),
  strongly indicating the list is **script-generated**. A hand-authored PR would be
  formatting-inconsistent and would likely be **overwritten on the next regeneration**, and
  there's no stated process inviting external PRs.
- **Open question for Qiren:** Do you want to (a) skip this list, or (b) reach out to the
  maintainer (Slava) via issue/DM to ask whether manual additions are accepted before we
  invest in a matching entry? No PR opened.

---

## 6. hesreallyhim/awesome-claude-code  →  ❓ BLOCKED (question for Qiren)

- **Canonical check:** ✅ canonical & very active (49,308★, last push 2026-07-08). Most topically-relevant list (Claude Code ecosystem).
- **Why blocked — the list forbids the PR mechanism (quoted):**
  > "Do not open a PR. Just fill out the form." … "ALL RECOMMENDATIONS MUST BE MADE USING THE WEB UI ISSUE FORM TEMPLATE, OR YOU RISK BEING RESTRICTED FROM INTERACTING WITH THIS REPOSITORY TEMPORARILY." … "It is **not** possible to submit a resource recommendation using the `gh` CLI." … "resource recommendations must be created by **human beings**."
  - Also: **recommendations are currently paused** ("I'm disabling recommendations for a little while"), and the maintainer explicitly discourages list-submission-as-promotion.
- **Action:** No PR (a PR here would risk a temporary interaction ban). This one needs a
  **human** (you) to submit via the web issue form — and only once recommendations reopen.
  Submission link for when you're ready:
  https://github.com/hesreallyhim/awesome-claude-code/issues/new?template=recommend-resource.yml

---

## 7. jaywcjlove/awesome-mac  →  ✅ DRAFT PR

- **Canonical check:** ✅ active & canonical — 106,870★, default branch `master`, last push 2026-07-07.
- **CONTRIBUTING rules (`docs/CONTRIBUTING.md`):**
  - Title-cased `[Name](link)`; **alphabetical within category**; individual PRs; one-sentence description.
  - Entries are **synced across README.md / README-zh.md / README-ja.md / README-ko.md**.
  - Icons: `[![Open-Source Software][OSS Icon]](repo)` + `![Freeware][Freeware Icon]` (Orbital: open-source + free; **not** Native — it's a WKWebView app).
  - **Decision (agreed):** ship EN + zh now; PR body offers ja/ko or defers them to the maintainer skill.
- **Action:** Draft PR → https://github.com/jaywcjlove/awesome-mac/pull/2272
- **Section:** AI Tools (alphabetically between "Off Grid AI Desktop" and "Orchard"), in README.md and README-zh.md.
- **Exact lines added:**
  ```
  (README.md)
  * [Orbital](https://github.com/zqiren/Orbital) - Local-first desktop agent runtime that keeps its memory in a plain-markdown folder and dispatches Claude Code and Codex as sub-agents with approvals, per-project budgets, and an OS-sandboxed built-in shell. [![Open-Source Software][OSS Icon]](https://github.com/zqiren/Orbital) ![Freeware][Freeware Icon]

  (README-zh.md)
  * [Orbital](https://github.com/zqiren/Orbital) - 本地优先的桌面智能体运行时，以纯 Markdown 文件夹作为智能体的持久记忆，并将 Claude Code 与 Codex 作为子智能体并行调度，内置审批、按项目预算和受操作系统沙箱限制的内置 shell。 [![Open-Source Software][OSS Icon]](https://github.com/zqiren/Orbital) ![Freeware][Freeware Icon]
  ```

---

## 8. Awesome-Windows/Awesome → 0PandaDEV/awesome-windows  →  ✅ DRAFT PR

- **Canonical check:** the task's `Awesome-Windows/Awesome` **404s (dead org).** Canonical successor
  is **`0PandaDEV/awesome-windows`** (2,566★, active, last push 2026-07-07). Verified the list is
  Windows-only and Orbital ships a Windows build → eligible.
- **CONTRIBUTING rules:**
  - Title-cased `* [Name](link)`; **alphabetical within category**; individual PRs; "no vibecoded slop".
  - Format uses `[![Open-Source Software][oss]](repo)`.
- **Section decision (agreed):** **Developer Utilities** rather than "Local AI" (that section is
  local-*inference* tools like Jan/LM Studio/Ollama; Orbital is local-first but uses cloud models).
- **Action:** Draft PR → https://github.com/0PandaDEV/awesome-windows/pull/212
- **Section:** Developer Utilities (alphabetically between "Mamp" and "Pieces").
- **Exact line added:**
  ```
  * [Orbital](https://github.com/zqiren/Orbital) - Local-first desktop agent runtime that keeps its memory in a plain-markdown folder and dispatches Claude Code and Codex as sub-agents with approvals, budgets, and an OS-sandboxed built-in shell. [![Open-Source Software][oss]](https://github.com/zqiren/Orbital)
  ```

---

## Side-finding — RESOLVED ✅

**Orbital's `LICENSE` was not detected by GitHub as GPL-3.0** (reported "Other / NOASSERTION")
because the file prepended a 16-line program notice before the standard GPL-3.0 text, so GitHub's
license detector (Licensee) couldn't match it. **Fixed:** `LICENSE` is now the verbatim canonical
GPL-3.0 text and the Orbital program notice + copyright line moved to a new `NOTICE` file.
- PR: https://github.com/zqiren/Orbital/pull/46 — **merged to `main`** (squash).
- Verified: `gh api repos/zqiren/Orbital` now returns `license.spdx_id = GPL-3.0`.
- This also strengthens the open-source signal on the license-checking lists (Jenqyang, e2b, kyrolabs).

---

## ✅ Checklist — draft PRs awaiting Qiren's review (mark ready only after your review)

- [ ] kyrolabs/awesome-agents — https://github.com/kyrolabs/awesome-agents/pull/614
- [ ] Jenqyang/Awesome-AI-Agents — https://github.com/Jenqyang/Awesome-AI-Agents/pull/362
- [ ] e2b-dev/awesome-ai-agents — https://github.com/e2b-dev/awesome-ai-agents/pull/1217  *(repo stale ~17 mo — decide whether to ever mark ready)*
- [ ] jaywcjlove/awesome-mac — https://github.com/jaywcjlove/awesome-mac/pull/2272

- [ ] 0PandaDEV/awesome-windows — https://github.com/0PandaDEV/awesome-windows/pull/212

## Not-opened lists — SKIPPED per Qiren's instruction

- **aloth/awesome-ai-agents** — skipped (fails the >100★ / major-org gate; see #2).
- **slavakurilyak/awesome-ai-agents** — skipped (auto-generated list; no PR pursued).
- **hesreallyhim/awesome-claude-code** — skipped (PRs forbidden; would need a human web-form submission when recommendations reopen — left to Qiren if/when desired).
