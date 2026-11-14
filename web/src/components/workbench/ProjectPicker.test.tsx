// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/** Spec 180 §5.3: one picker, three anchors — its rows and its placement. */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ProjectPicker, { AnchoredProjectPicker, placePopover } from './ProjectPicker';
import type { Project } from '../../types';

afterEach(cleanup);

const PROJECTS = [
  { project_id: 'qt', name: 'Quick Tasks', workspace: '/w/qt', is_scratch: true },
  { project_id: 'a', name: 'Marketing', workspace: '/w/a' },
] as unknown as Project[];

describe('ProjectPicker rows', () => {
  it('choose: 自动, a gap, the projects (Quick Tasks by its catalog name)', () => {
    const onAuto = vi.fn();
    render(<ProjectPicker projects={PROJECTS} selectedId={null} auto="choose" autoSelected onAuto={onAuto} onPick={vi.fn()} testId="pp" />);
    const pop = screen.getByTestId('pp');
    expect([...pop.children].map((c) => c.className)).toEqual(['wb-opt', 'wb-sep', 'wb-opt', 'wb-opt']);
    expect(screen.getByTestId('wb-opt-auto')).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('wb-opt-project-qt')).toHaveTextContent('Quick Tasks');
    fireEvent.click(screen.getByTestId('wb-opt-auto'));
    expect(onAuto).toHaveBeenCalledTimes(1);
  });

  it('keyless: 自动 is shown but off, with its note', () => {
    render(<ProjectPicker projects={PROJECTS} selectedId="a" auto="keyless" onPick={vi.fn()} />);
    const auto = screen.getByTestId('wb-opt-auto');
    expect(auto).toHaveAttribute('aria-disabled', 'true');
    expect(auto.className).toContain('is-off');
    expect(screen.getByTestId('wb-opt-project-a')).toHaveAttribute('aria-selected', 'true');
  });

  it('move: no 自动 row and no gap; the current project ticked; a pick names the project', () => {
    const onPick = vi.fn();
    render(<ProjectPicker projects={PROJECTS} selectedId="qt" onPick={onPick} testId="pp" />);
    expect(screen.queryByTestId('wb-opt-auto')).toBeNull();
    expect(screen.getByTestId('pp').querySelector('.wb-sep')).toBeNull();
    expect(screen.getByTestId('wb-opt-project-qt')).toHaveAttribute('aria-selected', 'true');
    fireEvent.click(screen.getByTestId('wb-opt-project-a'));
    expect(onPick).toHaveBeenCalledWith('a');
  });
});

describe('AnchoredProjectPicker', () => {
  it('lives in a portal, stops its clicks from reaching the row under it, closes on a press outside', () => {
    const anchor = document.createElement('button');
    document.body.appendChild(anchor);
    const onClose = vi.fn();
    const onRowClick = vi.fn();
    render(
      <div onClick={onRowClick}>
        <AnchoredProjectPicker anchor={anchor} projects={PROJECTS} selectedId="qt" onPick={vi.fn()} onClose={onClose} testId="pp" />
      </div>,
    );
    const pop = screen.getByTestId('pp');
    expect(pop.className).toContain('is-anchored');
    expect(pop.closest('.wb-pophost')!.parentElement).toBe(document.body);
    fireEvent.click(screen.getByTestId('wb-opt-project-a'));
    expect(onRowClick).not.toHaveBeenCalled();
    fireEvent.pointerDown(anchor);
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.pointerDown(document.body);
    expect(onClose).toHaveBeenCalledTimes(1);
    anchor.remove();
  });
});

describe('placePopover (the prototype\'s place)', () => {
  const view = { w: 1440, h: 900 };
  const rect = (l: number, t: number, w: number, h: number) =>
    ({ left: l, top: t, right: l + w, bottom: t + h, width: w, height: h, x: l, y: t, toJSON: () => ({}) }) as DOMRect;

  it('below the anchor when it fits', () => {
    expect(placePopover(rect(100, 100, 80, 20), { w: 260, h: 200 }, view, 'below')).toEqual({ left: 94, top: 126, maxHeight: undefined });
  });

  it('near the bottom with little room: above, scrolling if taller than the room', () => {
    const p = placePopover(rect(100, 850, 80, 20), { w: 260, h: 300 }, view, 'below');
    expect(p.top).toBe(850 - 6 - 300);
    expect(p.maxHeight).toBeUndefined();
  });

  it('reasonable room below (≥200): stays below and scrolls', () => {
    const p = placePopover(rect(100, 600, 80, 20), { w: 260, h: 400 }, view, 'below');
    expect(p.top).toBe(626);
    expect(p.maxHeight).toBe(900 - 8 - 626);
  });

  it('never past the right edge', () => {
    expect(placePopover(rect(1400, 100, 30, 20), { w: 260, h: 100 }, view, 'below').left).toBe(1440 - 260 - 8);
  });

  it('side: beside the row on a wide screen, below it when there is no room (a phone)', () => {
    expect(placePopover(rect(10, 300, 236, 28), { w: 260, h: 200 }, view, 'side')).toMatchObject({ left: 252, top: 294 });
    const phone = placePopover(rect(0, 300, 390, 28), { w: 260, h: 200 }, { w: 390, h: 844 }, 'side');
    expect(phone.top).toBe(334);
  });
});
