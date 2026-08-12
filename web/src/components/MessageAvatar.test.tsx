// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import MessageAvatar from './MessageAvatar';

describe('MessageAvatar', () => {
  it('user variant renders a solid primary box with the provided initials', () => {
    const { container } = render(<MessageAvatar variant="user" label="ME" />);
    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box).not.toBeNull();
    expect(box?.getAttribute('data-variant')).toBe('user');
    expect(box?.textContent).toBe('ME');
    expect(box?.className).toContain('bg-primary');
    expect(box?.className).toContain('text-white');
    expect(box?.className).toContain('w-[26px]');
    expect(box?.className).toContain('h-[26px]');
    expect(box?.className).toContain('rounded-md');
  });

  it('user variant falls back to "ME" when no label is given', () => {
    const { container } = render(<MessageAvatar variant="user" />);
    expect(container.querySelector('[data-testid="message-avatar"]')?.textContent).toBe('ME');
  });

  it('agent variant with no handle renders the Orbital logo (ignores label) in a bordered box', () => {
    const { container } = render(<MessageAvatar variant="agent" label="XX" />);
    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box?.getAttribute('data-variant')).toBe('agent');
    expect(box?.className).toContain('rounded-md');
    const img = box?.querySelector('img');
    expect(img?.getAttribute('src')).toBe('/icon-192.png');
    expect(box?.textContent).toBe('');
  });

  it('agent variant with a known handle renders that vendor mark', () => {
    const { container } = render(<MessageAvatar variant="agent" agentHandle="claude-code" />);
    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box?.querySelector('img')?.getAttribute('src')).toBe('/agents/claude-code.svg');
    expect(box?.getAttribute('data-agent-handle')).toBe('claude-code');
  });

  it('agent variant with an unknown handle renders the monogram badge', () => {
    const { container } = render(<MessageAvatar variant="agent" agentHandle="researcher" />);
    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box?.querySelector('img')).toBeNull();
    expect(box?.textContent).toBe('RE');
  });

  it('falls back to the monogram badge when the image fails to load', () => {
    const { container } = render(<MessageAvatar variant="agent" agentHandle="claude-code" />);
    fireEvent.error(container.querySelector('[data-testid="message-avatar"] img')!);

    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box?.querySelector('img')).toBeNull();
    expect(box?.textContent).toBe('CC');
  });
});
