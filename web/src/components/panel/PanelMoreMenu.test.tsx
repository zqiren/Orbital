// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 088 §6.9 — the panel's More menu: keyboard operable, named, closes on
 * Escape / outside click, gives focus back to its trigger, and never lets the
 * Escape reach anything that could collapse the workspace.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import PanelMoreMenu from './PanelMoreMenu';

afterEach(() => cleanup());

function renderMenu(overrides: Partial<React.ComponentProps<typeof PanelMoreMenu>> = {}) {
  const copy = vi.fn();
  const open = vi.fn();
  render(
    <>
      <PanelMoreMenu
        items={[
          { id: 'copy', label: 'Copy', onSelect: copy },
          { id: 'open', label: 'Open in Files', onSelect: open },
        ]}
        {...overrides}
      />
      <p data-testid="outside">outside</p>
    </>,
  );
  return { copy, open, trigger: screen.getByRole('button', { name: 'More actions' }) };
}

describe('PanelMoreMenu — trigger', () => {
  it('is a named menu button, closed at rest', () => {
    const { trigger } = renderMenu();
    expect(trigger).toHaveAttribute('aria-haspopup', 'menu');
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    expect(trigger).toHaveAttribute('title', 'More actions');
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('opens on click into a named menu and focuses the first item', () => {
    const { trigger } = renderMenu();
    fireEvent.click(trigger);
    const menu = screen.getByRole('menu', { name: 'More actions' });
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(trigger).toHaveAttribute('aria-controls', menu.id);
    const items = screen.getAllByRole('menuitem');
    expect(items.map((i) => i.textContent)).toEqual(['Copy', 'Open in Files']);
    expect(document.activeElement).toBe(items[0]);
  });

  it('ArrowDown on the trigger opens on the first item, ArrowUp on the last', () => {
    const { trigger } = renderMenu();
    fireEvent.keyDown(trigger, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(screen.getAllByRole('menuitem')[0]);
    fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });

    fireEvent.keyDown(trigger, { key: 'ArrowUp' });
    expect(document.activeElement).toBe(screen.getAllByRole('menuitem')[1]);
  });

  it('a second click on the trigger closes the menu', () => {
    const { trigger } = renderMenu();
    fireEvent.click(trigger);
    fireEvent.click(trigger);
    expect(screen.queryByRole('menu')).toBeNull();
  });
});

describe('PanelMoreMenu — keyboard inside the menu', () => {
  it('Arrow keys move and wrap; Home / End jump', () => {
    const { trigger } = renderMenu();
    fireEvent.click(trigger);
    const menu = screen.getByRole('menu');
    const [first, second] = screen.getAllByRole('menuitem');

    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(second);
    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(menu, { key: 'ArrowUp' });
    expect(document.activeElement).toBe(second);
    fireEvent.keyDown(menu, { key: 'Home' });
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(menu, { key: 'End' });
    expect(document.activeElement).toBe(second);
  });

  it('choosing an item runs it, closes, and returns focus to the trigger', () => {
    const { trigger, open } = renderMenu();
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('menuitem', { name: 'Open in Files' }));
    expect(open).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('Escape closes, returns focus, and never reaches a window listener', () => {
    const windowKeys = vi.fn();
    window.addEventListener('keydown', windowKeys);
    try {
      const { trigger, copy } = renderMenu();
      fireEvent.click(trigger);
      fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
      expect(screen.queryByRole('menu')).toBeNull();
      expect(document.activeElement).toBe(trigger);
      expect(copy).not.toHaveBeenCalled();
      expect(windowKeys).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('keydown', windowKeys);
    }
  });
});

describe('PanelMoreMenu — dismissal and item kinds', () => {
  it('closes on a pointer press outside the menu', () => {
    const { trigger, copy } = renderMenu();
    fireEvent.click(trigger);
    fireEvent.mouseDown(screen.getByTestId('outside'));
    expect(screen.queryByRole('menu')).toBeNull();
    expect(copy).not.toHaveBeenCalled();
  });

  it('a press inside the menu does not close it', () => {
    const { trigger } = renderMenu();
    fireEvent.click(trigger);
    fireEvent.mouseDown(screen.getByRole('menu'));
    expect(screen.getByRole('menu')).toBeInTheDocument();
  });

  it('items with `checked` are radio items carrying aria-checked', () => {
    renderMenu({
      items: [
        { id: 'write', label: 'Write', checked: true, onSelect: vi.fn() },
        { id: 'preview', label: 'Preview', checked: false, onSelect: vi.fn() },
      ],
    });
    fireEvent.click(screen.getByRole('button', { name: 'More actions' }));
    expect(screen.getByRole('menuitemradio', { name: 'Write' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('menuitemradio', { name: 'Preview' })).toHaveAttribute('aria-checked', 'false');
  });
});
