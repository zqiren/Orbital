// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Spec 161 for Pi: the worker at the top of an Orbital session can hand work
// to other agents. Orbital passes this file with `-e` only to that worker
// (never written into the workspace); everything else is Pi's own built-in
// MCP client talking to the daemon's one dispatch endpoint, so the tool's
// name, description and schema come from the endpoint's tools/list.
//
// The URL and the per-spawn bearer token arrive in the environment. Both are
// removed from Pi's environment before any tool runs, so the commands Pi runs
// never see the token; it lives only in the MCP client's request headers.
//
// The extension never throws: Pi exits at startup when an extension fails to
// load, and dispatch must never cost the user the worker itself. It reports
// the outcome as a status record (key "orbital-dispatch"); Orbital tells the
// worker about the tool only after "mounted".

export default function orbitalDispatch(pi: any) {
  const url = process.env.ORBITAL_DISPATCH_URL;
  const token = process.env.ORBITAL_DISPATCH_TOKEN;
  delete process.env.ORBITAL_DISPATCH_URL;
  delete process.env.ORBITAL_DISPATCH_TOKEN;
  if (!url || !token) return;

  let status = "mounted";
  try {
    // Needs Pi >= 0.99 (built-in MCP). "direct": declared to the model like
    // a built-in tool, as mcp__orbital__dispatch.
    pi.registerMcpServer("orbital", {
      url,
      headers: { Authorization: `Bearer ${token}` },
      exposure: "direct",
    });
  } catch (err) {
    status = `failed: ${err instanceof Error ? err.message : String(err)}`;
  }
  try {
    pi.on("session_start", (_event: unknown, ctx: any) => {
      try {
        ctx.ui.setStatus("orbital-dispatch", status);
      } catch {
        // No status record: Orbital treats the tool as not mounted.
      }
    });
  } catch {
    // Same.
  }
}
