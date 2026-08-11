// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../utils/agentIcons';
import { useT } from '../i18n/useT';

interface MessageAvatarProps {
  variant: 'user' | 'agent';
  /**
   * For the user variant: up-to-2 uppercase chars shown inside the box.
   * Ignored for the agent variant (which is identified by its icon).
   */
  label?: string;
  /**
   * For the agent variant: a sub-agent handle (e.g. "claude-code"), which
   * selects that vendor's mark. Omitted/empty means the row belongs to
   * Orbital's own management agent, which resolves to the Orbital logo through
   * the same lookup — every agent row gets its identity from one mechanism.
   */
  agentHandle?: string;
}

/**
 * Every avatar shares one container: 26×26, rounded-md, flex-centered. The
 * variants differ only in fill — solid primary with initials (user), a white
 * bordered box with the vendor mark inset (agent), or a brand-coloured
 * monogram badge (unknown handle / failed image) — so the three read as one
 * family rather than three visual systems.
 */
const AVATAR_BOX = 'shrink-0 w-[26px] h-[26px] rounded-md flex items-center justify-center';

/**
 * The 26×26 avatar box that leads each message row in the flat avatar-log
 * layout (Design §5). User avatars are a solid primary square with initials;
 * every agent — Orbital itself included — shows its own mark, falling back to a
 * brand-coloured monogram badge of the same size and shape if the image is
 * missing or fails to load.
 */
export default function MessageAvatar({ variant, label, agentHandle }: MessageAvatarProps) {
  const t = useT();
  const [imageFailed, setImageFailed] = useState(false);

  if (variant === 'user') {
    return (
      <div
        data-testid="message-avatar"
        data-variant="user"
        className={`${AVATAR_BOX} text-2xs font-semibold bg-primary text-white`}
      >
        {label ?? t('chat.avatar.me')}
      </div>
    );
  }

  const handle = agentHandle || MAIN_AGENT_HANDLE;
  const icon = getAgentIcon(handle);

  if (icon.src && !imageFailed) {
    return (
      <div
        data-testid="message-avatar"
        data-variant="agent"
        data-agent-handle={agentHandle}
        className={`${AVATAR_BOX} bg-card border border-border/60 p-[3px] overflow-hidden`}
      >
        <img
          src={icon.src}
          alt={agentHandle ?? 'Orbital'}
          onError={() => setImageFailed(true)}
          className="w-full h-full object-contain"
        />
      </div>
    );
  }

  return (
    <div
      data-testid="message-avatar"
      data-variant="agent"
      data-agent-handle={agentHandle}
      className={`${AVATAR_BOX} text-3xs font-semibold text-white`}
      style={{ backgroundColor: icon.color }}
    >
      {icon.monogram}
    </div>
  );
}
