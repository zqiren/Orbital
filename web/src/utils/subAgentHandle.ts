// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Fanout worker handles are minted as `worker:<fanout_id>-<index>`
 * (agent_os/daemon_v2/native_worker.py's `make_worker_handle`). Workers are
 * ephemeral task executors, not persistent collaborators (spec 009 §0.5-8):
 * they never appear in the chip bar (`SubAgentStatusBar`) and their live
 * turn output must never leak into the main chat as bubbles — the fanout
 * progress card and the one-shot join summary are the only sanctioned
 * surfaces for their results.
 */
const WORKER_HANDLE_PREFIX = 'worker:';

export function isWorkerHandle(handle: string | null | undefined): boolean {
  return typeof handle === 'string' && handle.startsWith(WORKER_HANDLE_PREFIX);
}
