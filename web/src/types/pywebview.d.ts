// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

interface PyWebViewApi {
  pick_folder(): Promise<string | null>;
  /** Desktop app: native multi-select open dialog; the picked files by
   *  absolute path (`[]` on cancel). Absent on an older shell. */
  pick_files?(): Promise<Array<{ path: string; name: string; size: number; mime: string }>>;
}

interface Window {
  pywebview?: { api: PyWebViewApi };
  /** Set by the desktop shell once it reports dropped files' real paths. */
  __orbitalNativeDrop?: boolean;
}
