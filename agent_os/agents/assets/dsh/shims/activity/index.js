// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ORBITAL TEMPORARY SHIM — tool activity over ACP.
 *
 * `@deepseek-ai/dsh-acp` forwards no tool events by design ("committed answers
 * only"): a turn that ran five bash commands reaches an ACP client as one
 * assistant message. The core records every one of them, so the gap is in the
 * adapter, not the harness. This plugin closes it from the composition, and is
 * built to be deleted the day upstream closes it properly.
 *
 * WHAT IT DOES
 *   Listens on the core's `session/event` bus for `tool/call` and writes the
 *   corresponding STANDARD ACP `session/update` notification (`tool_call`) on
 *   the stdio wire. Nothing here is an Orbital-private protocol — any ACP
 *   client gets the same rows.
 *
 * WHY IT WRITES TO stdout DIRECTLY
 *   The obvious implementation is `conn.sessionUpdate(...)`. It is not
 *   available: `dsh-acp` keeps its `AgentSideConnection` in a closure-local
 *   `conn`, publishes no Cordis service and stores nothing on `ctx`, so no
 *   sibling plugin can reach it. The alternative to this file is forking that
 *   530-line pre-1.0 package. We write frames instead.
 *
 *   That is safe because of how the ACP SDK writes: `ndJsonStream`'s sink does
 *   `JSON.stringify(message) + "\n"` in ONE `write()` per message
 *   (@agentclientprotocol/sdk/dist/stream.js). Node's Writable queues whole
 *   chunks in call order, so a complete line written here can never interleave
 *   inside one of the bridge's lines, in either direction. We only ADD frames;
 *   we never touch the bridge's own traffic, so a bug here cannot corrupt a
 *   turn — at worst it costs the tool rows this shim exists to provide.
 *
 * WHY START-ONLY, NO TERMINAL `tool_call_update`
 *   Because the consumer cannot show one and WOULD double every row. Orbital's
 *   post-hoc capsule appends one row per `tool_activity` chunk with no
 *   de-duplication by `toolCallId` (sub_agent_transcript.py:69-88) and renders
 *   each as its own "name · duration" line (chatTransform.ts:743-758). A
 *   start+terminal pair therefore renders `bash · 3.2s` / `bash · 0.0s` for
 *   every single call. Emitting only `tool_call` yields exactly one row per
 *   tool invocation — byte-for-byte the parity bar claude-code sets, which
 *   likewise emits one `[Using tool: X]` per tool-use block at start
 *   (sdk_transport.py:636). The capsule has no status affordance at all
 *   (`result_status` is hardcoded `'received'`), so the dropped terminal
 *   status costs the user nothing today. Restore it here the day the capsule
 *   grows per-call status, and de-duplicate on `toolCallId` in the same change.
 *
 * ORDERING / SCOPE
 *   `tool/call` fires only inside a turn, which is exactly when Orbital's
 *   transport accepts updates (it drops them for inactive sessions and
 *   unopened turns — acp_sdk_transport.py:436-448). The core session id IS the
 *   ACP session id: `dsh-acp` mints one uuid and passes it to
 *   `agents.create({ sessionId })`, so `session.header.id` needs no mapping.
 *   Sub-agent sessions dsh spawns internally carry their own ids, which the
 *   client drops as inactive.
 *
 * REMOVAL
 *   Delete the `orbital-acp-activity` block from `cordis.template.yml`, drop
 *   the `@orbital/dsh-acp-activity-shim` dependency, refresh the lockfile, and
 *   re-run the Task 8 smoke. No Orbital Python changes.
 *   INVARIANT: no Orbital Python may reference this plugin's id.
 */

export const name = 'orbital-acp-activity';

// Deliberately NO `inject`. Cordis gates service access on the declaration and
// defers the fiber until those services exist; this plugin needs neither
// `agents` nor `sessionPersistence`, only the event bus, and staying
// dependency-free keeps it mounted for the whole process lifetime.

/** ACP `ToolCallStatus` for a call that has been dispatched but not settled. */
const IN_PROGRESS = 'in_progress';

/**
 * Translate one core session event into an ACP `session/update` params object.
 *
 * Pure, and exported for the Node-side unit check: everything below it is I/O.
 *
 * @param sessionId - the ACP session id (the core session id, unchanged).
 * @param event - a core `session/event` payload (`{ type, data }`).
 * @returns `session/update` params, or `undefined` when the event is not a tool
 *   dispatch (the overwhelming majority — chunks, headers, steps, turns).
 */
export function toolUpdateFor(sessionId, event) {
  if (event?.type !== 'tool/call') return undefined;

  const callId = event.data?.callId;
  const toolName = event.data?.name;
  if (typeof callId !== 'string' || typeof toolName !== 'string') return undefined;
  if (toolName === '') return undefined;

  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call',
      toolCallId: callId,
      // The plain tool name, and only that. Orbital's post-hoc capsule renders
      // `title`; richer payloads have nowhere to go.
      title: toolName,
      status: IN_PROGRESS,
    },
  };
}

/**
 * Mount the shim.
 * @param ctx - Cordis context carrying the core's session event bus.
 */
export function apply(ctx) {
  ctx.on('session/event', (session, event) => {
    const sessionId = session?.header?.id;
    if (typeof sessionId !== 'string') return;

    let params;
    try {
      params = toolUpdateFor(sessionId, event);
    } catch {
      // Never let a malformed event fail the turn it is describing.
      return;
    }
    if (params === undefined) return;

    process.stdout.write(
      JSON.stringify({ jsonrpc: '2.0', method: 'session/update', params }) + '\n'
    );
  });
}
