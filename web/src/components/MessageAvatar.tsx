// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

interface MessageAvatarProps {
  variant: 'user' | 'agent';
  /**
   * For the user variant: up-to-2 uppercase chars shown inside the box.
   * Ignored for the agent variant (which always shows the ◐ glyph).
   */
  label?: string;
}

/**
 * The 26×26 avatar box that leads each message row in the flat avatar-log
 * layout (Design §5). User avatars are a solid primary square with initials;
 * agent / sub-agent avatars are an outlined square with the ◐ glyph.
 */
export default function MessageAvatar({ variant, label }: MessageAvatarProps) {
  const isUser = variant === 'user';
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
