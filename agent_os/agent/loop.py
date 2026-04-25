# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main agent execution loop.

Owned by Component A. Core while loop: prepare context -> call LLM ->
process response -> execute tools -> repeat.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from uuid import uuid4

from agent_os.agent.providers.types import (
    StreamAccumulator,
    StreamChunk,
    LLMResponse,
    TokenUsage,
    ContextOverflowError,
    LLMError,
    ErrorCategory,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tool_result_filters import dispatch_prefilter
from agent_os.agent.tool_result_lifecycle import truncate_consumed_tool_results

logger = logging.getLogger(__name__)

# Default cost rates ($/1K tokens) used when no per-model pricing is available.
_DEFAULT_COST_PER_1K_INPUT = 0.003
_DEFAULT_COST_PER_1K_OUTPUT = 0.015


def _estimate_cost_usd(usage: TokenUsage,
                       cost_per_1k_input: float = _DEFAULT_COST_PER_1K_INPUT,
                       cost_per_1k_output: float = _DEFAULT_COST_PER_1K_OUTPUT) -> float:
    """Estimate USD cost from token usage."""
    return (usage.input_tokens / 1000) * cost_per_1k_input + \
           (usage.output_tokens / 1000) * cost_per_1k_output


def normalize_tool_call(tc_raw: dict) -> dict:
    """Normalize tool call from nested/flat/string-args formats."""
    if "function" in tc_raw:
        func = tc_raw["function"]
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return {"id": tc_raw.get("id", ""), "name": func.get("name", ""), "arguments": args}
    args = tc_raw.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    return {"id": tc_raw.get("id", ""), "name": tc_raw.get("name", ""), "arguments": args}


class AgentLoop:
    """Core agent execution loop."""

    def __init__(
        self,
        session,
        provider,
        tool_registry,
        context_manager,
        interceptor=None,
        utility_provider=None,
        fallback_providers=None,
        max_iterations: int = 0,
        token_budget: int = 100_000_000,
        budget_limit_usd: float | None = None,
        budget_action: str = "ask",
        budget_spent_usd: float = 0.0,
        on_cost_update=None,
        cost_per_1k_input: float = _DEFAULT_COST_PER_1K_INPUT,
        cost_per_1k_output: float = _DEFAULT_COST_PER_1K_OUTPUT,
        on_session_end=None,
    ):
        self._session = session
        self._provider = provider
        self._tool_registry = tool_registry
        self._context_manager = context_manager
        self._interceptor = interceptor
        self._utility_provider = utility_provider
        self._fallback_providers = fallback_providers or []
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._budget_limit_usd = budget_limit_usd
        self._budget_action = budget_action
        self._budget_spent_usd = budget_spent_usd
        self._on_cost_update = on_cost_update
        self._cost_per_1k_input = cost_per_1k_input
        self._cost_per_1k_output = cost_per_1k_output
        self._running = False
        self._llm_failed = False
        self._on_session_end = on_session_end

        # Cancellation state (cancel_turn / terminate)
        self._task: asyncio.Task | None = None
        self._inflight_stream: asyncio.Task | None = None
        self._stream_accumulator: StreamAccumulator | None = None
        self._turn_cancelled: bool = False
        self._cancellation_marker_appended_this_turn: bool = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def budget_spent_usd(self) -> float:
        return self._budget_spent_usd

    async def _stream_response(self, context, tool_schemas) -> LLMResponse:
        """Stream LLM response from the primary provider.

        Convenience wrapper preserved for tests that monkey-patch this
        method. Delegates to the unified internal helper.
        """
        return await self._stream_response_internal(
            self._provider, context, tool_schemas,
        )

    async def _stream_response_with(self, provider, context, tool_schemas) -> LLMResponse:
        """Stream LLM response using a specific provider (rotation).

        Routes the primary provider through _stream_response so any test
        monkey-patches on that method continue to fire; otherwise goes
        directly to the unified internal helper. Both paths share the
        same accumulator-publishing semantics so cancel_turn() can rely
        on a consistent contract regardless of which provider is active.
        """
        if provider is self._provider:
            return await self._stream_response(context, tool_schemas)
        return await self._stream_response_internal(
            provider, context, tool_schemas,
        )

    async def _stream_response_internal(self, provider, context, tool_schemas) -> LLMResponse:
        """Single source-of-truth streaming implementation.

        Publishes the accumulator on self._stream_accumulator before any
        await so cancel_turn() can inspect partial state. The async
        iteration is wrapped in a try/finally so CancelledError reaches
        the underlying generator's cleanup path (its httpx context).
        """
        accumulator = StreamAccumulator()
        self._stream_accumulator = accumulator
        try:
            async for chunk in provider.stream(context, tools=tool_schemas):
                accumulator.add(chunk)
                self._session.notify_stream(chunk)
            return accumulator.finalize()
        finally:
            # Do not clear self._stream_accumulator here — cancel_turn()
            # may still need to read it on its way out. The next turn
            # will reassign it at the top of the next call.
            pass

    @staticmethod
    def _rotate_provider(providers, current_idx, cooldowns):
        """Find next available provider not in cooldown. Returns index or None."""
        now = time.monotonic()
        n = len(providers)
        for offset in range(1, n):
            candidate = (current_idx + offset) % n
            if now >= cooldowns.get(candidate, 0):
                return candidate
        return None

    async def run(self, initial_message: str | None = None,
                  initial_nonce: str | None = None) -> None:
        """Main loop entry point."""
        self._running = True
        # Capture our own task handle so terminate() can cancel us.
        self._task = asyncio.current_task()
        try:
            # Resolve pending if not resuming from approval pause
            if self._session._paused_for_approval:
                self._session._paused_for_approval = False  # reset here, not in resume()
            else:
                self._session.resolve_pending_tool_calls()

            # Append initial user message
            if initial_message is not None:
                msg: dict = {
                    "role": "user",
                    "content": initial_message,
                    "source": "user",
                }
                if initial_nonce:
                    msg["nonce"] = initial_nonce
                self._session.append(msg)
                # Fresh user input on this run ⇒ clear sticky cancel flag.
                self._cancellation_marker_appended_this_turn = False

            # Reset per-run state on all tools (e.g. send counters)
            self._tool_registry.reset_run_state()

            # Reset loop guards
            iteration = 0
            cumulative_tokens = 0
            action_hashes: deque = deque(maxlen=15)
            consecutive_overflows = 0
            llm_retries = 0

            # Fallback provider rotation state
            _all_providers = [self._provider] + list(self._fallback_providers)
            _current_idx = 0
            _cooldowns: dict[int, float] = {}
            _COOLDOWN_SECONDS = 60.0
            _retries_on_current = 0

            # Ping-pong detection state
            _pair_hashes: deque = deque(maxlen=20)
            _prev_action_hash: str | None = None

            # Circuit breaker state (reset per run / new user message)
            error_tracker: dict[str, dict] = {}  # tool_name -> {"error": str, "count": int}
            blocked_tools: set[str] = set()

            while True:
                # Drain queued user messages before preparing context
                queued = self._session.pop_queued_messages()
                if queued:
                    # New user input ⇒ a brand new "turn" begins. Reset the
                    # cancellation marker flag so subsequent cancel_turn()
                    # calls write a fresh marker.
                    self._cancellation_marker_appended_this_turn = False
                    for q_item in queued:
                        if isinstance(q_item, tuple):
                            q_content, q_nonce = q_item
                        else:
                            q_content, q_nonce = q_item, None
                        q_msg: dict = {
                            "role": "user",
                            "content": q_content,
                            "source": "user",
                        }
                        if q_nonce:
                            q_msg["nonce"] = q_nonce
                        self._session.append(q_msg)
                    # Reset circuit breaker on new user messages
                    error_tracker.clear()
                    blocked_tools.clear()

                # Check iteration cap (0 = unlimited)
                if self._max_iterations and iteration >= self._max_iterations:
                    self._session.append_system(
                        "Maximum iteration limit reached. Save your current state and stop."
                    )
                    break

                # Check token budget
                if cumulative_tokens >= self._token_budget:
                    self._session.append_system(
                        "Token budget exceeded. Save your current state and stop."
                    )
                    break

                iteration += 1

                # Reset per-iteration cancellation flags. The
                # cancellation-marker flag is sticky across cancelled
                # iterations: it's only reset when a new user message has
                # been drained (in the queue-drain block above) — so a
                # cancel_turn() called twice with no intervening user
                # message is idempotent.
                self._turn_cancelled = False

                # Prepare context
                context = self._context_manager.prepare()

                # Get tool schemas
                tool_schemas = self._tool_registry.schemas()

                # Stream LLM response (using active provider from rotation).
                # Wrap in asyncio.Task so cancel_turn() can interrupt it.
                active_provider = _all_providers[_current_idx]
                self._inflight_stream = asyncio.create_task(
                    self._stream_response_with(
                        active_provider, context, tool_schemas,
                    ),
                    name=f"stream-{self._session.session_id}-{iteration}",
                )
                try:
                    response = await self._inflight_stream
                    consecutive_overflows = 0
                    llm_retries = 0
                    _retries_on_current = 0
                except asyncio.CancelledError:
                    # cancel_turn() invoked while streaming. Marker + cost
                    # debit handled by cancel_turn() before propagating.
                    self._inflight_stream = None
                    if self._session.is_stopped():
                        # terminate() also fired — let outer cleanup run.
                        raise
                    # Bare cancel_turn: continue the outer while loop.
                    continue
                except ContextOverflowError:
                    self._inflight_stream = None
                    consecutive_overflows += 1
                    if consecutive_overflows >= 3:
                        self._session.append_system(
                            "Context overflow cannot be resolved. Save state and stop."
                        )
                        break
                    self._context_manager.reduce_window(0.5)
                    continue
                except LLMError as e:
                    self._inflight_stream = None
                    category = e.category

                    # Abort: non-recoverable errors (401, 403, 400)
                    if category == ErrorCategory.ABORT:
                        self._llm_failed = True
                        self._session.append_system(
                            f"LLM error (non-recoverable): {e.message}. Stopping."
                        )
                        break

                    # Transient: rotate to next fallback provider
                    if category == ErrorCategory.TRANSIENT:
                        _cooldowns[_current_idx] = time.monotonic() + _COOLDOWN_SECONDS
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            model_name = _all_providers[rotated].model
                            logger.info("Rotating to fallback model: %s", model_name)
                            self._session.append_system(
                                f"Primary model unavailable ({e.message}). "
                                f"Switching to {model_name}."
                            )
                            continue

                    # Retry same provider (500, or transient with no fallback)
                    _retries_on_current += 1
                    llm_retries += 1
                    if _retries_on_current >= 3:
                        # Exhausted retries on current — try rotating
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            continue
                        # All providers exhausted
                        self._llm_failed = True
                        self._session.append_system(
                            f"LLM error after {llm_retries} retries: "
                            f"{e.message}. Stopping."
                        )
                        break
                    await asyncio.sleep(2 ** _retries_on_current)
                    continue

                # Successful stream completion — clear inflight task handle.
                self._inflight_stream = None

                # Track token usage
                if response.usage:
                    cumulative_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Budget tracking (always active for spend persistence)
                if response.usage:
                    delta = _estimate_cost_usd(
                        response.usage, self._cost_per_1k_input, self._cost_per_1k_output,
                    )
                    self._budget_spent_usd += delta
                    if self._on_cost_update is not None:
                        self._on_cost_update(delta, self._budget_spent_usd)
                    # Per-project budget limit check
                    if (self._budget_limit_usd is not None
                            and self._budget_spent_usd >= self._budget_limit_usd):
                        if self._budget_action == "stop":
                            self._session.append_system(
                                f"Budget limit exceeded (${self._budget_spent_usd:.2f} / "
                                f"${self._budget_limit_usd:.2f}). Stopping."
                            )
                            break
                        else:  # "ask" — pause for user approval
                            self._session.append_system(
                                f"Budget limit exceeded (${self._budget_spent_usd:.2f} / "
                                f"${self._budget_limit_usd:.2f}). Pausing for approval."
                            )
                            self._session.pause()
                            break

                # Text-only response: append and exit
                if not response.has_tool_calls:
                    self._session.append({
                        "role": "assistant",
                        "content": response.text,
                        "source": "management",
                    })
                    truncate_consumed_tool_results(
                        self._session, response.text, iteration,
                    )
                    break

                # Extract status
                status = response.status_text
                if not status and response.tool_calls:
                    first_tc = normalize_tool_call(response.tool_calls[0])
                    status = f"Using {first_tc['name']}"

                # Append raw assistant message
                raw_msg = dict(response.raw_message)
                raw_msg["source"] = "management"
                if status:
                    raw_msg["_status"] = status
                self._session.append(raw_msg)

                # Truncate tool results consumed by this LLM response
                truncate_consumed_tool_results(
                    self._session, response.text, iteration,
                )

                # Normalize and execute tool calls
                tool_calls = [normalize_tool_call(tc) for tc in response.tool_calls]
                exit_outer = False
                intercepted_tc_id = None

                for tc in tool_calls:
                    # Cancellation check: if cancel_turn() fired between or
                    # during tool execution, skip remaining tool calls.
                    if self._turn_cancelled:
                        break

                    # Post-normalization validation
                    if not tc["id"]:
                        tc["id"] = uuid4().hex
                        logger.warning("Tool call missing ID, generated: %s", tc["id"])

                    if not tc["name"]:
                        self._session.append_tool_result(
                            tc["id"],
                            "Error: LLM returned tool call with no function name",
                        )
                        continue

                    # Interceptor check (fail-closed)
                    if self._interceptor is not None:
                        try:
                            should_intercept = self._interceptor.should_intercept(tc)
                        except Exception:
                            should_intercept = True

                        if should_intercept:
                            reasoning = response.text if response.text and response.text.strip() else None
                            try:
                                self._interceptor.on_intercept(
                                    tc, self._session.recent_activity(),
                                    reasoning=reasoning,
                                )
                            except Exception as e:
                                # on_intercept failed (e.g. WS broadcast hook
                                # crashed). Without this guard the exception
                                # would escape run(), leave the orphan tool
                                # call as CANCELLED, and end the loop task in
                                # error state — leaving the user with no
                                # approval card and no recovery path.
                                # Treat the failure as a denial: append an
                                # error tool result so the agent can react,
                                # and continue the dispatch loop.
                                logger.exception(
                                    "Interceptor.on_intercept raised for "
                                    "tool_call_id=%s tool=%s",
                                    tc.get("id"), tc.get("name"),
                                )
                                self._session.append_tool_result(
                                    tc["id"],
                                    f"Error requesting approval: {e}. "
                                    "The tool was not executed.",
                                )
                                continue
                            intercepted_tc_id = tc["id"]
                            self._session._paused_for_approval = True
                            self._session.pause()
                            exit_outer = True
                            break

                    # Circuit breaker: block tool if it has repeated errors
                    if tc["name"] in blocked_tools:
                        blocked_msg = (
                            f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                            f"(repeated error: {error_tracker[tc['name']]['error']}). "
                            f"Work with the tools that are available or ask the user for help."
                        )
                        self._session.append_tool_result(tc["id"], blocked_msg)
                        continue

                    # Execute tool (async-aware)
                    try:
                        if self._tool_registry.is_async(tc["name"]):
                            result: ToolResult = await self._tool_registry.execute_async(
                                tc["name"],
                                tc["arguments"],
                            )
                        else:
                            result: ToolResult = await asyncio.to_thread(
                                self._tool_registry.execute,
                                tc["name"],
                                tc["arguments"],
                            )
                        # Pre-filter tool result before storage
                        filtered_content = dispatch_prefilter(
                            tc["name"], tc["arguments"], result.content,
                        )
                        result = ToolResult(
                            content=filtered_content, meta=result.meta,
                        )

                        self._session.append_tool_result(
                            tc["id"], result.content, meta=result.meta,
                            context_limit=self._context_manager.model_context_limit,
                        )
                        tool_content = result.content
                        # Store result for observation-aware hash
                        _exec_result = result

                        # Pause if credential was requested (wait for user input)
                        if result.meta and result.meta.get("credential_request"):
                            self._session.pause()
                            exit_outer = True
                            break
                    except Exception as e:
                        tool_content = f"Error executing tool: {e}"
                        self._session.append_tool_result(
                            tc["id"], tool_content
                        )
                        _exec_result = None

                    # Observation-aware repetition detection (post-execution)
                    # Browser tools: skip hash detection (handled by advisory counter in browser tool)
                    if tc["name"] != "browser":
                        result_prefix = str(tool_content)[:500] if tool_content else ""
                        action_hash = hashlib.md5(
                            (tc["name"] + str(tc["arguments"]) + result_prefix).encode()
                        ).hexdigest()

                        action_hashes.append(action_hash)
                        if list(action_hashes).count(action_hash) >= 5:
                            self._session.resolve_pending_tool_calls()
                            self._session.append_system(
                                "Repetitive action detected. Save your state and try a different approach."
                            )
                            exit_outer = True
                            break

                        # Ping-pong / cycle detection: track consecutive pairs
                        if _prev_action_hash is not None:
                            pair = (_prev_action_hash, action_hash)
                            _pair_hashes.append(pair)
                            if list(_pair_hashes).count(pair) >= 3:
                                self._session.resolve_pending_tool_calls()
                                self._session.append_system(
                                    f"Ping-pong pattern detected: you have been "
                                    f"alternating between the same tool calls for "
                                    f"multiple iterations with no progress. "
                                    f"Stop and explain what you are trying to "
                                    f"accomplish, then try a fundamentally "
                                    f"different approach."
                                )
                                exit_outer = True
                                break
                        _prev_action_hash = action_hash

                    # Circuit breaker: track consecutive identical errors
                    if isinstance(tool_content, str) and tool_content.startswith("Error"):
                        prev = error_tracker.get(tc["name"])
                        if prev and prev["error"] == tool_content:
                            prev["count"] += 1
                        else:
                            error_tracker[tc["name"]] = {"error": tool_content, "count": 1}
                        if error_tracker[tc["name"]]["count"] >= 2:
                            blocked_tools.add(tc["name"])
                            self._session.append_system(
                                f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                                f"(repeated error: {tool_content}). "
                                f"Work with the tools that are available or ask the user for help."
                            )
                    else:
                        # Successful execution clears error tracking for this tool
                        error_tracker.pop(tc["name"], None)

                # Drain deferred messages (lifecycle notifications)
                for msg in self._session.pop_deferred_messages():
                    self._session.append(msg)

                # Cancellation landed mid-tool-loop (cancel_turn fired between
                # tool calls or during a tool's execution). Append marker if
                # not already done, resolve any unstarted tool calls, then
                # continue to the top of the while loop. terminate() also
                # sets is_stopped(), so we let the next iteration of the
                # outer loop handle the stop check.
                if self._turn_cancelled:
                    if not self._cancellation_marker_appended_this_turn:
                        self._persist_cancellation_marker()
                    # Resolve any tool calls that never got a result — these
                    # are the ones we skipped via the per-tool break above.
                    self._session.resolve_pending_tool_calls()
                    continue

                # Resolve unprocessed tool calls from this batch on intercept-break
                if exit_outer and intercepted_tc_id is not None:
                    for tc in tool_calls:
                        tc_id = tc["id"]
                        if tc_id in self._session.pending_tool_calls and tc_id != intercepted_tc_id:
                            self._session.append_tool_result(
                                tc_id,
                                "CANCELLED: Earlier tool call requires approval. "
                                "This tool call will be retried after approval.",
                            )

                # Check if we need to exit the outer loop
                if exit_outer:
                    break

                # Check stopped
                if self._session.is_stopped():
                    self._session.resolve_pending_tool_calls()
                    break

                # Check paused (redundant with exit_outer but kept for safety)
                if self._session.is_paused():
                    break

                # Check compaction
                if self._context_manager.should_compact():
                    from agent_os.agent import compaction as compaction_mod

                    # Pre-compaction memory flush: give agent one turn to save state
                    try:
                        self._session.append_system(compaction_mod.MEMORY_FLUSH_PROMPT)
                        flush_llm = self._utility_provider or self._provider
                        flush_context = self._context_manager.prepare()
                        flush_response = await flush_llm.complete(flush_context)
                        flush_text = flush_response.text or ""
                        if not compaction_mod.is_silent_response(flush_text):
                            # Agent wants to save state — execute any tool calls
                            if flush_response.tool_calls:
                                raw_msg = dict(flush_response.raw_message)
                                raw_msg["source"] = "management"
                                self._session.append(raw_msg)
                                for tc_raw in flush_response.tool_calls:
                                    try:
                                        tc = normalize_tool_call(tc_raw)
                                        if not tc["id"]:
                                            tc["id"] = uuid4().hex
                                        if not tc["name"]:
                                            self._session.append_tool_result(
                                                tc["id"],
                                                "Error: tool call has no function name",
                                            )
                                            continue
                                        if self._tool_registry.is_async(tc["name"]):
                                            result = await self._tool_registry.execute_async(
                                                tc["name"], tc["arguments"],
                                            )
                                        else:
                                            result = await asyncio.to_thread(
                                                self._tool_registry.execute,
                                                tc["name"], tc["arguments"],
                                            )
                                        self._session.append_tool_result(tc["id"], result.content)
                                    except Exception as e:
                                        tc_id = tc.get("id") if isinstance(tc, dict) else tc_raw.get("id", uuid4().hex)
                                        self._session.append_tool_result(
                                            tc_id, f"Error: {e}",
                                        )
                            else:
                                self._session.append({
                                    "role": "assistant",
                                    "content": flush_text,
                                    "source": "management",
                                })
                    except (ContextOverflowError, LLMError):
                        # Context too full for flush — skip and proceed to compaction
                        pass
                    except Exception:
                        # Any other error during flush — don't crash, just compact
                        logger.warning("Pre-compaction flush failed, proceeding to compaction")

                    await compaction_mod.run(
                        self._session,
                        self._provider,
                        utility_provider=self._utility_provider,
                    )

                    # Post-compaction reorientation
                    workspace = getattr(
                        getattr(self._context_manager, "_base_ctx", None),
                        "workspace", None,
                    )
                    if workspace:
                        compaction_mod.inject_reorientation(workspace, self._session)

        finally:
            if not self._session._paused_for_approval:
                self._session.resolve_pending_tool_calls()
            # Drain any remaining deferred messages
            for msg in self._session.pop_deferred_messages():
                self._session.append(msg)
            # NOTE: fire-and-forget asyncio.create_task(self._on_session_end())
            # was previously here. Removed in TASK-cancel-arch-04: the
            # synchronous session-end inside agent_manager.new_session() is
            # now the sole authoritative summarization path. The fire-and-
            # forget had no strong reference and would emit
            # "Task was destroyed but it is pending" on cancel.
            self._running = False

    # ------------------------------------------------------------------
    # Cancellation API (cancel_turn / terminate)
    # ------------------------------------------------------------------

    async def cancel_turn(self) -> None:
        """Interrupt the current turn. Loop returns to idle, agent stays alive.

        Idempotent: safe to call when no turn is in flight, or twice in
        rapid succession. Exactly one cancellation marker is written per
        cancellation event regardless of when the cancel landed (during
        stream, between stream and tools, or mid-tool-loop).

        Reserved for the future /cancel HTTP verb. terminate() should be
        used by stop_agent / new_session callers — not cancel_turn().
        """
        # Idempotency: if we have already initiated a cancel for the
        # current turn (marker pending or appended), do nothing. The flag
        # is reset at the top of each loop iteration once a new turn
        # actually starts running.
        if self._cancellation_marker_appended_this_turn:
            return

        if self._inflight_stream is None or self._inflight_stream.done():
            # No turn in flight; possibly idle, possibly between LLM and
            # tool exec, possibly mid-tool-loop. Set the flag so the loop
            # body skips remaining tool calls and the post-tool block
            # appends the marker if appropriate.
            self._turn_cancelled = True
            return

        # Claim the inflight task and capture the accumulator BEFORE any
        # await. After we yield to the event loop in `wait_for` below, the
        # loop's CancelledError handler may `continue` into iteration N+1
        # which reassigns self._stream_accumulator. We must hold a local
        # reference to the cancelled-turn's accumulator so the post-await
        # cost debit reflects the cancelled turn — not iteration N+1's
        # fresh empty accumulator (C1 race).
        inflight = self._inflight_stream
        self._inflight_stream = None
        accumulator = self._stream_accumulator

        # Mark cancelled FIRST so any concurrent loop work observes it,
        # then mark "marker pending for this turn" to make further
        # cancel_turn() calls no-ops until a new turn starts.
        self._turn_cancelled = True
        self._cancellation_marker_appended_this_turn = True

        # Persist the marker SYNCHRONOUSLY before the await so its JSONL
        # position is adjacent to the cancelled turn — even if iteration
        # N+1 races ahead and appends its own assistant message during
        # our wait_for (C2 race).
        self._persist_cancellation_marker_inner()

        inflight.cancel()
        try:
            await asyncio.wait_for(inflight, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("inflight stream raised during cancel_turn")

        # Debit cost using the captured accumulator (immune to iteration
        # N+1's reassignment of self._stream_accumulator during the await).
        self._debit_cancelled_turn_cost(accumulator)

    async def terminate(self) -> None:
        """End the loop. Cancels current turn and exits the while True.

        Session JSONL preserved; agent task done; handle cleanup is the
        caller's responsibility (caller awaits self._task or shields it).
        Do NOT await self._task here — this would deadlock if terminate
        was invoked from within the loop's own task.
        """
        await self.cancel_turn()
        # Cooperative flag: loop's between-iterations check picks this up.
        self._session.stop()
        if self._task is not None and not self._task.done():
            # If we're called from a different task (typical for /stop),
            # cancel the loop task so it exits the inflight stream wait
            # via the CancelledError -> is_stopped() raise branch.
            cur = None
            try:
                cur = asyncio.current_task()
            except RuntimeError:
                pass
            if cur is not self._task:
                self._task.cancel()

    def _persist_cancellation_marker(self) -> None:
        """Append a system message indicating user cancellation.

        Called exactly once per cancellation event. Sets the per-turn
        flag so cancel_turn() and the post-tool-loop block do not double-
        append. The flag is reset at the top of each loop iteration.
        """
        if self._cancellation_marker_appended_this_turn:
            return
        self._cancellation_marker_appended_this_turn = True
        self._persist_cancellation_marker_inner()

    def _persist_cancellation_marker_inner(self) -> None:
        """Write the cancellation marker without re-checking the flag.

        Internal helper used by cancel_turn() once it has already claimed
        the per-turn marker slot via _cancellation_marker_appended_this_turn.
        """
        self._session.append({
            "role": "system",
            "content": "[cancelled by user]",
            "source": "management",
            "cancelled_by_user": True,
        })

    def _debit_cancelled_turn_cost(self, accumulator=None) -> None:
        """Debit the cost of partial output tokens generated before cancel.

        Reads the streaming accumulator's collected usage (or estimates
        from text_parts when usage is unavailable). Mirrors the normal
        cost path used after a successful turn (the on_cost_update
        callback at loop.py:103). Full debit per Moonshot's billing
        policy: tokens generated are billed even if undelivered.

        Takes the accumulator as an explicit parameter so callers can pass
        in a reference captured BEFORE any awaits — protecting against
        the C1 race where iteration N+1 reassigns self._stream_accumulator
        during cancel_turn's await wait_for. If no accumulator is supplied
        (legacy call sites) we fall back to self._stream_accumulator.
        """
        if accumulator is None:
            accumulator = self._stream_accumulator
        if accumulator is None:
            return

        # Prefer the usage emitted by the provider's final/usage chunk.
        usage = accumulator.usage
        if usage is None:
            # No final-usage chunk arrived (we cancelled before it). Fall
            # back to a rough output-token estimate from the partial text.
            partial_text = "".join(accumulator.text_parts)
            output_tokens = max(1, len(partial_text) // 4) if partial_text else 0
            if output_tokens == 0:
                return
            usage = TokenUsage(input_tokens=0, output_tokens=output_tokens)

        delta = _estimate_cost_usd(
            usage, self._cost_per_1k_input, self._cost_per_1k_output,
        )
        self._budget_spent_usd += delta
        if self._on_cost_update is not None:
            try:
                self._on_cost_update(delta, self._budget_spent_usd)
            except Exception:
                logger.exception("on_cost_update callback raised during cancel")
