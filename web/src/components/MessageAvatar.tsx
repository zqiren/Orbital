// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { getAgentIcon } from '../utils/agentIcons';

interface MessageAvatarProps {
  variant: 'user' | 'agent';
  /**
   * For the user variant: up-to-2 uppercase chars shown inside the box.
   * Ignored for the agent variant (which always shows the ◐ glyph).
   */
  label?: string;
  /**
   * For the agent variant: a sub-agent handle (e.g. "claude-code"). When
   * present, the avatar shows that sub-agent's recognizable icon (a
   * brand-coloured monogram badge) instead of the default ◐ management glyph,
   * marking the row as a first-class peer agent. Omitted/empty → ◐.
   */
  agentHandle?: string;
}

/**
 * The 26×26 avatar box that leads each message row in the flat avatar-log
 * layout (Design §5). User avatars are a solid primary square with initials;
 * the management agent is an outlined square with the ◐ glyph; a sub-agent is
 * its brand-coloured monogram badge (same size/shape).
 */
export default function MessageAvatar({ variant, label, agentHandle }: MessageAvatarProps) {
  const isUser = variant === 'user';

  if (!isUser && agentHandle) {
    const icon = getAgentIcon(agentHandle);
    if (icon.src) {
      return (
        <img
          src={icon.src}
          alt={agentHandle}
          data-testid="message-avatar"
          data-variant="agent"
          data-agent-handle={agentHandle}
          className="shrink-0 w-[26px] h-[26px] rounded-[6px] object-contain"
        />
      );
    }
    return (
      <div
        data-testid="message-avatar"
        data-variant="agent"
        data-agent-handle={agentHandle}
        className="shrink-0 w-[26px] h-[26px] rounded-[6px] flex items-center justify-center font-mono text-[9px] font-bold text-white"
        style={{ backgroundColor: icon.color }}
      >
        {icon.monogram}
      </div>
    );
  }

  return (
    <div
      data-testid="message-avatar"
      data-variant={variant}
      className={
        'shrink-0 w-[26px] h-[26px] rounded-[6px] flex items-center justify-center font-mono text-[10px] font-semibold ' +
        (isUser
          ? 'bg-primary text-white'
          : 'bg-background border border-border text-primary')
      }
    >
      {isUser ? (label ?? 'ME') : '◐'}
    </div>
  );
}
