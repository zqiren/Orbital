// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 185: the one answer to "is a turn in flight for the session I am
// showing", used by ChatView's seed guard and its cached-paint check. A turn
// is in flight while the manager is running or waiting, or while any worker
// of the viewed session is running. A pinned worker's turn keeps the manager
// idle, so the manager's status alone misses it.
//
// Only `running` counts for a worker. `background-running` means a command
// the worker started is still alive (a dev server can last for hours); its
// turn is over, and counting it would hold the screen stale indefinitely. A
// worker waiting on an approval is still `running`.

export interface WorkerStatusLike {
  status: string;
}

export function isTurnInFlight(
  managerStatus: string | undefined,
  workers: readonly WorkerStatusLike[],
): boolean {
  if (managerStatus === 'running' || managerStatus === 'waiting') return true;
  return workers.some((w) => w.status === 'running');
}
