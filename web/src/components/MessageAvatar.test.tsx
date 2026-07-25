// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render } from '@testing-library/react';
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
    expect(box?.className).toContain('rounded-sm');
  });

  it('user variant falls back to "ME" when no label is given', () => {
    const { container } = render(<MessageAvatar variant="user" />);
    expect(container.querySelector('[data-testid="message-avatar"]')?.textContent).toBe('ME');
  });

  it('agent variant renders an outlined box with the ◐ glyph (ignores label)', () => {
    const { container } = render(<MessageAvatar variant="agent" label="XX" />);
    const box = container.querySelector('[data-testid="message-avatar"]');
    expect(box?.getAttribute('data-variant')).toBe('agent');
    expect(box?.textContent).toBe('◐');
    expect(box?.className).toContain('border');
    expect(box?.className).toContain('bg-background');
    expect(box?.className).not.toContain('bg-primary');
  });
});
