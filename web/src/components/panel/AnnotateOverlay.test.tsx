// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import AnnotateOverlay from './AnnotateOverlay';

/** A drag from (x1,y1) to (x2,y2) over the overlay surface. */
function drag(el: Element, x1: number, y1: number, x2: number, y2: number) {
  fireEvent.pointerDown(el, { clientX: x1, clientY: y1, pointerId: 1, button: 0 });
  fireEvent.pointerMove(el, { clientX: x2, clientY: y2, pointerId: 1 });
  fireEvent.pointerUp(el, { clientX: x2, clientY: y2, pointerId: 1 });
}

describe('AnnotateOverlay — drawing a box', () => {
  it('draws a box and hands it to onAdd with the typed note on Enter', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    const overlay = screen.getByTestId('annotate-overlay');

    drag(overlay, 20, 30, 120, 90);

    // The note field appears anchored to the box; nothing is added yet.
    const note = screen.getByTestId('annotate-note');
    expect(onAdd).not.toHaveBeenCalled();

    fireEvent.change(note, { target: { value: 'click this one' } });
    fireEvent.keyDown(note, { key: 'Enter' });

    expect(onAdd).toHaveBeenCalledTimes(1);
    expect(onAdd).toHaveBeenCalledWith({ x: 20, y: 30, w: 100, h: 60 }, 'click this one');
    // The pending box + field are cleared once committed.
    expect(screen.queryByTestId('annotate-note')).toBeNull();
  });

  it('normalizes a drag made up-and-left', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    drag(screen.getByTestId('annotate-overlay'), 120, 90, 20, 30);
    fireEvent.keyDown(screen.getByTestId('annotate-note'), { key: 'Enter' });
    expect(onAdd).toHaveBeenCalledWith({ x: 20, y: 30, w: 100, h: 60 }, '');
  });

  it('commits through the Add button as well as Enter', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    drag(screen.getByTestId('annotate-overlay'), 0, 0, 40, 40);
    fireEvent.change(screen.getByTestId('annotate-note'), { target: { value: 'here' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    expect(onAdd).toHaveBeenCalledWith({ x: 0, y: 0, w: 40, h: 40 }, 'here');
  });

  it('trims the note', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    drag(screen.getByTestId('annotate-overlay'), 0, 0, 40, 40);
    fireEvent.change(screen.getByTestId('annotate-note'), { target: { value: '  spaced  ' } });
    fireEvent.keyDown(screen.getByTestId('annotate-note'), { key: 'Enter' });
    expect(onAdd).toHaveBeenCalledWith(expect.anything(), 'spaced');
  });

  it('Esc cancels the pending box without adding it', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    drag(screen.getByTestId('annotate-overlay'), 10, 10, 60, 60);
    fireEvent.keyDown(screen.getByTestId('annotate-note'), { key: 'Escape' });
    expect(screen.queryByTestId('annotate-note')).toBeNull();
    expect(onAdd).not.toHaveBeenCalled();
  });

  it('ignores a drag under the 6px minimum — that is a click, not a box', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active boxes={[]} onAdd={onAdd} />);
    drag(screen.getByTestId('annotate-overlay'), 10, 10, 14, 30);
    expect(screen.queryByTestId('annotate-note')).toBeNull();
    expect(onAdd).not.toHaveBeenCalled();
  });

  it('does not let the drag reach the view underneath while annotating', () => {
    const onParentDown = vi.fn();
    render(
      <div onPointerDown={onParentDown}>
        <AnnotateOverlay active boxes={[]} onAdd={vi.fn()} />
      </div>,
    );
    fireEvent.pointerDown(screen.getByTestId('annotate-overlay'), {
      clientX: 5,
      clientY: 5,
      pointerId: 1,
    });
    expect(onParentDown).not.toHaveBeenCalled();
  });
});

describe('AnnotateOverlay — existing boxes and the inactive state', () => {
  const boxes = [
    { n: 1, box: { x: 10, y: 20, w: 30, h: 40 } },
    { n: 2, box: { x: 50, y: 60, w: 70, h: 80 } },
  ];

  it('renders a numbered pin per box', () => {
    render(<AnnotateOverlay active boxes={boxes} onAdd={vi.fn()} />);
    expect(screen.getByTestId('annotation-pin-1')).toHaveTextContent('1');
    expect(screen.getByTestId('annotation-pin-2')).toHaveTextContent('2');
    const box = screen.getByTestId('annotation-box-1');
    expect(box).toHaveStyle({ left: '10px', top: '20px', width: '30px', height: '40px' });
  });

  it('offers a labelled remove control only when onRemove is given', () => {
    const { unmount } = render(<AnnotateOverlay active boxes={boxes} onAdd={vi.fn()} />);
    expect(screen.queryAllByRole('button', { name: 'Remove annotation' })).toHaveLength(0);
    unmount();

    const onRemove = vi.fn();
    render(<AnnotateOverlay active boxes={boxes} onAdd={vi.fn()} onRemove={onRemove} />);
    const removes = screen.getAllByRole('button', { name: 'Remove annotation' });
    expect(removes).toHaveLength(2);
    fireEvent.click(removes[1]);
    expect(onRemove).toHaveBeenCalledWith(2);
  });

  it('is transparent to the pointer and draws nothing new when inactive', () => {
    const onAdd = vi.fn();
    render(<AnnotateOverlay active={false} boxes={boxes} onAdd={onAdd} />);
    const overlay = screen.getByTestId('annotate-overlay');
    expect(overlay).toHaveStyle({ pointerEvents: 'none' });

    drag(overlay, 10, 10, 100, 100);
    expect(screen.queryByTestId('annotate-note')).toBeNull();
    expect(screen.queryByTestId('annotate-drag')).toBeNull();
    expect(onAdd).not.toHaveBeenCalled();
    // The pins are still there — they are what the boxes look like at rest.
    expect(screen.getByTestId('annotation-pin-1')).toBeInTheDocument();
  });
});
