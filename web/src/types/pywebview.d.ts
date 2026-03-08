// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

interface PyWebViewApi {
  pick_folder(): Promise<string | null>;
}

interface Window {
  pywebview?: { api: PyWebViewApi };
}
