// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

export interface InstalledSubAgent {
  slug: string;
  name: string;
}

interface Props {
  /** Installed, non-built-in sub-agents available on this machine. */
  agents: InstalledSubAgent[];
  /** Slugs disabled for this project (project.disabled_sub_agents). */
  disabled: string[];
  /** Called with (slug, enabled) when a row is toggled. enabled=false means
   *  the slug should be added to disabled_sub_agents. */
  onToggle: (slug: string, enabled: boolean) => void;
}

/**
 * Per-project enable/disable list for installed sub-agents. A disabled
 * sub-agent is hidden from the management agent (and peers) for this project —
 * it won't be suggested or dispatched here. Install/login/params live in the
 * global Sub-Agent settings; this only scopes visibility per project.
 */
export default function SubAgentToggleList({ agents, disabled, onToggle }: Props) {
  if (agents.length === 0) {
    return (
      <p className="text-xs text-secondary/60 italic">
        No sub-agents installed. Install them from global Settings → Sub-Agents.
      </p>
    );
  }

  const disabledSet = new Set(disabled);

  return (
    <div className="flex flex-col gap-1.5">
      {agents.map((agent) => {
        const enabled = !disabledSet.has(agent.slug);
        return (
          <label
            key={agent.slug}
            className="flex items-center gap-2 cursor-pointer select-none text-sm text-primary"
          >
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => onToggle(agent.slug, e.target.checked)}
              aria-label={agent.name}
              className="w-3.5 h-3.5"
            />
            <span>{agent.name}</span>
            <span className="font-mono text-[11px] text-secondary/60">{agent.slug}</span>
          </label>
        );
      })}
    </div>
  );
}
