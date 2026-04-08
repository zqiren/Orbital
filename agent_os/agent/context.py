# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Context window assembly for the management agent.

Owned by Component A. Builds the message list sent to the LLM on each iteration.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from datetime import datetime

logger = logging.getLogger(__name__)

from agent_os.agent.prompt_builder import PromptContext
from agent_os.agent.token_utils import estimate_message_tokens


# Layer 1 workspace file keys (matching WorkspaceFileManager.FILE_NAMES)
# paired with their on-disk filenames, in the order they should appear
# in the context window.
_LAYER1_FILES: tuple[tuple[str, str], ...] = (
    ("state", "PROJECT_STATE.md"),
    ("decisions", "DECISIONS.md"),
    ("lessons", "LESSONS.md"),
)


class ContextManager:
    """Assembles context for each LLM call: system prompt + layers + sliding window."""

    def __init__(
        self,
        session,
        prompt_builder,
        base_prompt_context: PromptContext,
        model_context_limit: int = 128_000,
        response_reserve: int = 20_000,
        workspace_files=None,
        sub_agent_provider=None,
    ):
        self._session = session
        self._prompt_builder = prompt_builder
        self._base_ctx = base_prompt_context
        self._model_context_limit = model_context_limit
        self._response_reserve = response_reserve
        self._workspace_files = workspace_files
        self._sub_agent_provider = sub_agent_provider  # callable() -> list[dict]
        self._cold_resume_injected: bool = False
        self._last_usage_pct: float = 0.0
        self._window_factor: float = 1.0
        self._recovery_injected: bool = False

    @property
    def usage_percentage(self) -> float:
        return self._last_usage_pct

    @property
    def model_context_limit(self) -> int:
        return self._model_context_limit

    def should_compact(self) -> bool:
        return self._last_usage_pct > 0.80

    def reduce_window(self, factor: float = 0.5) -> None:
        self._window_factor *= factor

    def _inject_recovery_context(self) -> None:
        """Inject context from the most recent archived session if Layer 1 files are incomplete."""
        self._recovery_injected = True

        # Only inject into fresh sessions (no messages yet)
        if len(self._session.get_messages()) > 0:
            return

        workspace = self._base_ctx.workspace
        agent_os_dir = os.path.join(workspace, "orbital")
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        state_path = os.path.join(agent_os_dir, "PROJECT_STATE.md")

        # If both exist, normal cold resume - no injection needed
        if os.path.isfile(goals_path) and os.path.isfile(state_path):
            return

        # Find archived session files
        sessions_dir = os.path.join(agent_os_dir, "sessions")
        if not os.path.isdir(sessions_dir):
            return

        current_session_id = getattr(self._session, "session_id", None)
        session_files = []
        for fname in os.listdir(sessions_dir):
            if not fname.endswith(".jsonl"):
                continue
            # Exclude current session's file
            stem = fname[:-6]  # strip .jsonl
            if current_session_id and stem == current_session_id:
                continue
            fpath = os.path.join(sessions_dir, fname)
            session_files.append((os.path.getmtime(fpath), fpath))

        if not session_files:
            return

        # Sort by mtime descending, pick most recent
        session_files.sort(key=lambda x: x[0], reverse=True)
        most_recent_path = session_files[0][1]

        # Read all messages from the most recent archived session
        messages = []
        with open(most_recent_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        if not messages:
            return

        # Take last 20 messages
        tail = messages[-20:]

        # Format messages
        formatted_lines = []
        for msg in tail:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    names = ", ".join(
                        tc.get("function", {}).get("name", "unknown")
                        for tc in tool_calls
                    )
                    formatted_lines.append(f"Assistant used: {names}")
                elif content:
                    formatted_lines.append(f"Assistant: {content}")
            # Skip "tool" and "system" roles

        if not formatted_lines:
            return

        injection_text = (
            "[RECOVERY CONTEXT] Your previous session ended unexpectedly before you "
            "could save your notes. Here are the last messages from that session for "
            "continuity:\n\n"
            + "\n".join(formatted_lines)
            + "\n\nUse this context to continue where you left off. If project_goals.md "
            "doesn't exist yet, create it based on what was discussed."
        )

        self._session.append_system(injection_text)

    def prepare(self) -> list[dict]:
        """Build context: system prompt + layer injections + sliding window."""
        if not self._recovery_injected:
            self._inject_recovery_context()

        available_budget = self._model_context_limit - self._response_reserve

        # Update transient fields in PromptContext
        active_subs = self._sub_agent_provider() if self._sub_agent_provider else []
        ctx = replace(
            self._base_ctx,
            datetime_now=datetime.now().isoformat(),
            context_usage_pct=self._last_usage_pct,
            active_sub_agents=active_subs,
        )

        # Build system prompt
        cached_prefix, dynamic_suffix = self._prompt_builder.build(ctx)
        system_prompt = cached_prefix + "\n\n" + dynamic_suffix
        system_tokens = self._estimate_tokens(system_prompt)

        # Read Layer 1 files from disk
        workspace = self._base_ctx.workspace
        agent_os_dir = os.path.join(workspace, "orbital")

        layer_messages: list[dict] = []

        # Layer 1: PROJECT_STATE.md, DECISIONS.md, LESSONS.md
        # Prefer WorkspaceFileManager (knows the project-namespaced path).
        # Fall back to the flat path only in scratch mode (no workspace_files).
        for key, filename in _LAYER1_FILES:
            if self._workspace_files is not None:
                content = self._workspace_files.read(key)
            else:
                filepath = os.path.join(agent_os_dir, filename)
                content = self._read_file(filepath)
            if content:
                layer_messages.append({
                    "role": "system",
                    "content": f"[{filename}]\n{content}",
                })

        # Layer 3: instructions/*.md
        instructions_dir = os.path.join(agent_os_dir, "instructions")
        instructions_content = self._read_instructions(instructions_dir)
        if instructions_content:
            layer_messages.append({
                "role": "system",
                "content": f"[Project Instructions]\n{instructions_content}",
            })

        # Cold resume injection (first call only)
        cold_resume_messages: list[dict] = []
        if not self._cold_resume_injected and self._workspace_files is not None:
            resume_ctx = self._workspace_files.build_cold_resume_context()
            if resume_ctx:
                cold_resume_messages.append({
                    "role": "system",
                    "content": f"[WORKSPACE MEMORY — Resume Context]\n\n{resume_ctx}",
                })
            self._cold_resume_injected = True

        # Estimate layer tokens
        layer_tokens = sum(
            estimate_message_tokens(m) for m in layer_messages
        )
        cold_resume_tokens = sum(
            estimate_message_tokens(m) for m in cold_resume_messages
        )

        # Calculate remaining budget for sliding window
        remaining = available_budget - system_tokens - layer_tokens - cold_resume_tokens
        remaining = max(0, int(remaining * self._window_factor))

        # Get sliding window from session
        sliding_window = self._session.get_recent(remaining)

        # Remap non-standard roles for LLM compatibility
        sliding_window = self._sanitize_roles(sliding_window)

        # Apply tool result pruning on old messages
        sliding_window = self._prune_old_tool_results(sliding_window)

        # Assemble final context
        result: list[dict] = []

        # System prompt
        result.append({"role": "system", "content": system_prompt})

        # Layer 3 instructions come before Layer 1 state
        for msg in layer_messages:
            if "[Project Instructions]" in msg.get("content", ""):
                result.append(msg)

        # Layer 1 state files
        for msg in layer_messages:
            if "[Project Instructions]" not in msg.get("content", ""):
                result.append(msg)

        # Cold resume context (workspace memory)
        result.extend(cold_resume_messages)

        # Sliding window
        result.extend(sliding_window)

        # Validate tool results: ensure every tool_call has a matching result
        result = self._validate_tool_results(result)

        # Update usage percentage
        total_tokens = sum(estimate_message_tokens(m) for m in result)
        if available_budget > 0:
            self._last_usage_pct = total_tokens / available_budget
        else:
            self._last_usage_pct = 1.0

        # Reset window factor after use
        self._window_factor = 1.0

        return result

    _VALID_LLM_ROLES = {"system", "user", "assistant", "tool"}

    @staticmethod
    def _sanitize_roles(messages: list[dict]) -> list[dict]:
        """Filter non-standard roles for LLM API compatibility.

        v5: Sub-agent output goes to transcripts, not the management session.
        role='agent' messages should no longer appear in the sliding window.
        If one is found, log a warning and drop it.
        """
        result = []
        for msg in messages:
            role = msg.get("role")
            if role in ContextManager._VALID_LLM_ROLES:
                result.append(msg)
            elif role == "agent":
                source = msg.get("source", "sub-agent")
                logger.warning(
                    "Unexpected role='agent' message from %s in management session "
                    "(v5 transcript isolation should prevent this). Dropping.",
                    source,
                )
            # else: drop messages with unknown roles
        return result

    @staticmethod
    def _validate_tool_results(messages: list[dict]) -> list[dict]:
        """Ensure every assistant tool_call has a matching tool result,
        and strip orphaned tool results that appear in wrong positions.

        Pass 1: If a result is missing, inject a synthetic error result to
        prevent LLM rejection of the conversation.

        Pass 2: Remove any role:"tool" message whose tool_call_id does not
        match a tool_calls entry in the immediately preceding assistant
        message block. This handles CANCELLED results appended far from
        their originating assistant message (e.g. after daemon crash).
        """
        # --- Pass 1: inject missing results ---
        result = list(messages)
        i = 0
        while i < len(result):
            msg = result[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {tc.get("id", "") for tc in msg["tool_calls"] if tc.get("id")}
                # Look ahead for tool results
                found_ids: set[str] = set()
                j = i + 1
                while j < len(result):
                    r = result[j]
                    if r.get("role") == "tool" and r.get("tool_call_id") in expected_ids:
                        found_ids.add(r["tool_call_id"])
                    elif r.get("role") == "assistant":
                        break  # Next assistant message — stop looking
                    j += 1

                missing = expected_ids - found_ids
                for tc_id in missing:
                    # Insert synthetic result right after the last found result
                    insert_pos = i + 1 + len(found_ids)
                    result.insert(insert_pos, {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "Error: Tool result was lost. The tool may not have executed.",
                    })
            i += 1

        # --- Pass 2: strip orphaned tool results ---
        valid_tool_ids: set[str] = set()
        cleaned: list[dict] = []
        for msg in result:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                valid_tool_ids = {
                    tc.get("id", "") for tc in msg["tool_calls"] if tc.get("id")
                }
                cleaned.append(msg)
            elif role == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id in valid_tool_ids:
                    valid_tool_ids.discard(tc_id)
                    cleaned.append(msg)
                # else: orphaned tool result — drop it
            else:
                # user, system, or assistant without tool_calls.
                # Do NOT reset valid_tool_ids — tool results can appear
                # after user messages (e.g. approval flow appends result
                # after a user message was injected).  valid_tool_ids is
                # only reset when the next assistant-with-tool_calls is
                # encountered (line 342), which is consistent with Pass 1's
                # look-ahead that also spans past user messages.
                cleaned.append(msg)

        return cleaned

    def _prune_old_tool_results(self, messages: list[dict]) -> list[dict]:
        """Prune tool results older than 5 LLM turns if >500 chars. In-memory only.

        Browser snapshots get aggressive 2-turn TTL. Browser screenshots
        are always replaced with a one-liner.

        "Turns" are counted as assistant messages (LLM responses), not raw
        message indices, so multi-message tool batches within one turn are
        not over-pruned.
        """
        if len(messages) <= 2:
            return messages

        # Pre-compute the LLM turn index for each message.
        # A "turn" increments each time an assistant message appears.
        total_turns = sum(1 for m in messages if m.get("role") == "assistant")
        turn_at: list[int] = []
        current_turn = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                current_turn += 1
            turn_at.append(current_turn)

        pruned = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                pruned.append(msg)
                continue

            llm_turns_ago = total_turns - turn_at[i]
            meta = msg.get("_meta", {})

            # Browser screenshot: always replace (no useful text content)
            if meta.get("screenshot_path") and not meta.get("snapshot_stats"):
                new_msg = dict(msg)
                new_msg["content"] = f"[Screenshot saved to {meta['screenshot_path']}]"
                pruned.append(new_msg)
                continue

            # Browser snapshot: aggressive 2-turn TTL
            if meta.get("snapshot_stats") and llm_turns_ago > 2:
                stats = meta["snapshot_stats"]
                url = meta.get("url", "unknown")
                ref_count = stats.get("refs", 0)
                new_msg = dict(msg)
                new_msg["content"] = f"[Snapshot of {url} — {ref_count} elements. Run snapshot for current.]"
                pruned.append(new_msg)
                continue

            # Image-bearing tool results: aggressive 1-turn TTL
            content = msg.get("content", "")
            if llm_turns_ago > 1 and isinstance(content, list):
                has_image = any(
                    isinstance(b, dict) and b.get("type") in ("image_url", "image")
                    for b in content
                )
                if has_image:
                    text_parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    summary = " ".join(text_parts)[:200] or "[image]"
                    image_path = meta.get("image_path", meta.get("screenshot_path", ""))
                    if image_path:
                        ref = f"[Image: {image_path}] {summary}"
                    else:
                        ref = f"[Image analyzed] {summary}"
                    new_msg = dict(msg)
                    new_msg["content"] = ref
                    pruned.append(new_msg)
                    continue

            # Default pruning: 5-turn / 500-char rule
            if llm_turns_ago > 5:
                if isinstance(content, list):
                    # Multimodal content: replace with text summary
                    text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    summary = " ".join(text_parts)[:200] or "[multimodal content]"
                    new_msg = dict(msg)
                    new_msg["content"] = f"[Pruned] {summary}"
                    pruned.append(new_msg)
                    continue
                if isinstance(content, str) and len(content) > 500:
                    tool_name = msg.get("tool_call_id", "unknown")
                    first_line = content.split("\n")[0][:100]
                    new_msg = dict(msg)
                    new_msg["content"] = f"[Truncated] {tool_name}: {first_line}... ({len(content)} chars)"
                    pruned.append(new_msg)
                    continue

            pruned.append(msg)
        return pruned

    @staticmethod
    def _estimate_tokens(text: str) -> float:
        """Approximate token count: len(text) / 4."""
        return len(text) / 4

    @staticmethod
    def _read_file(filepath: str) -> str | None:
        """Read a file, return None if missing."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    @staticmethod
    def _read_instructions(instructions_dir: str) -> str | None:
        """Read and concatenate all .md files in instructions dir, sorted alphabetically."""
        if not os.path.isdir(instructions_dir):
            return None
        files = sorted(
            f for f in os.listdir(instructions_dir)
            if f.endswith(".md") and f not in ("project_goals.md", "user_directives.md")
        )
        if not files:
            return None
        parts = []
        for filename in files:
            filepath = os.path.join(instructions_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    parts.append(f.read())
            except OSError:
                pass
        return "\n\n".join(parts) if parts else None
