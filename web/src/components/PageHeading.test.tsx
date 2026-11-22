// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import PageHeading from './PageHeading';

// Spec 137 round 3: beside the project column a page opens with a heading,
// not a bar — the column is the navigation.
describe('PageHeading', () => {
  it('a page under the project: the title alone, as a title-size heading — no Back, no bar, no line', () => {
    const { container } = render(<PageHeading title="Tasks" titleTestId="tasks-title" />);
    const h1 = screen.getByRole('heading', { level: 1 });
    expect(h1.textContent).toBe('Tasks');
    expect(h1).toHaveAttribute('data-testid', 'tasks-title');
    expect(h1.className).toContain('text-title');
    expect(h1.className).toContain('font-semibold');
    expect(screen.queryByRole('navigation')).toBeNull();
    expect(screen.queryByRole('button')).toBeNull();
    expect(container.querySelector('header')).toBeNull();
    expect(container.innerHTML).not.toMatch(/border|vt-/);
  });

  it('a nested page: its parent is a small link line above the title', () => {
    const onSettings = vi.fn();
    render(
      <PageHeading parents={[{ label: 'Settings', onClick: onSettings }]} title="Pricing table" titleTestId="pricing-title" />,
    );
    const nav = screen.getByRole('navigation', { name: 'Location' });
    expect(nav.textContent).toBe('Settings');
    expect(nav.className).toContain('text-caption');
    // Above the title, not beside it.
    const h1 = screen.getByRole('heading', { level: 1 });
    expect(nav.compareDocumentPosition(h1) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(nav.contains(h1)).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(onSettings).toHaveBeenCalledTimes(1);
  });
});
