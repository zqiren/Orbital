// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 078 §5 / D7 — the panel bar: exactly a Files|Browser switch and an
 * Annotate toggle, and below it the one selected view.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import WorkspacePanel from './WorkspacePanel';

afterEach(() => cleanup());

function renderPanel(overrides: Partial<React.ComponentProps<typeof WorkspacePanel>> = {}) {
  const props = {
    view: 'files' as const,
    onViewChange: vi.fn(),
    annotating: false,
    onToggleAnnotate: vi.fn(),
    browser: <div data-testid="browser-body">browser body</div>,
    files: <div data-testid="files-body">files body</div>,
    ...overrides,
  };
  render(<WorkspacePanel {...props} />);
  return props;
}

describe('WorkspacePanel — the view switch', () => {
  it('renders exactly two tabs in a tablist', () => {
    renderPanel();
    expect(screen.getAllByRole('tab')).toHaveLength(2);
    expect(screen.getByRole('tab', { name: 'Files' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Browser' })).toBeInTheDocument();
    expect(screen.getByRole('tablist')).toBeInTheDocument();
  });

  it('marks the selected view with aria-selected', () => {
    renderPanel({ view: 'browser' });
    expect(screen.getByRole('tab', { name: 'Browser' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'false');
  });

  it('reports the picked view to the parent', () => {
    const { onViewChange } = renderPanel({ view: 'files' });
    fireEvent.click(screen.getByRole('tab', { name: 'Browser' }));
    expect(onViewChange).toHaveBeenCalledWith('browser');
  });

  it('shows only the selected view, never both', () => {
    const { rerender } = render(
      <WorkspacePanel
        view="files"
        onViewChange={vi.fn()}
        annotating={false}
        onToggleAnnotate={vi.fn()}
        browser={<div data-testid="browser-body" />}
        files={<div data-testid="files-body" />}
      />,
    );
    expect(screen.getByTestId('files-body')).toBeInTheDocument();
    expect(screen.queryByTestId('browser-body')).toBeNull();

    rerender(
      <WorkspacePanel
        view="browser"
        onViewChange={vi.fn()}
        annotating={false}
        onToggleAnnotate={vi.fn()}
        browser={<div data-testid="browser-body" />}
        files={<div data-testid="files-body" />}
      />,
    );
    expect(screen.getByTestId('browser-body')).toBeInTheDocument();
    expect(screen.queryByTestId('files-body')).toBeNull();
  });
});

describe('WorkspacePanel — the Annotate toggle', () => {
  it('reads "Annotate" when off and "Done" when on, with aria-pressed following', () => {
    const { rerender } = render(
      <WorkspacePanel
        view="browser"
        onViewChange={vi.fn()}
        annotating={false}
        onToggleAnnotate={vi.fn()}
        browser={null}
        files={null}
      />,
    );
    const off = screen.getByRole('button', { name: 'Annotate' });
    expect(off).toHaveAttribute('aria-pressed', 'false');

    rerender(
      <WorkspacePanel
        view="browser"
        onViewChange={vi.fn()}
        annotating
        onToggleAnnotate={vi.fn()}
        browser={null}
        files={null}
      />,
    );
    expect(screen.getByRole('button', { name: 'Done' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('calls onToggleAnnotate when pressed', () => {
    const { onToggleAnnotate } = renderPanel({ view: 'browser' });
    fireEvent.click(screen.getByTestId('panel-annotate'));
    expect(onToggleAnnotate).toHaveBeenCalledTimes(1);
  });
});

describe('WorkspacePanel — Annotate is Browser-only', () => {
  it('shows no Annotate button on the Files view (selection + Quote covers files)', () => {
    renderPanel({ view: 'files' });
    expect(screen.queryByRole('button', { name: 'Annotate' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Done' })).toBeNull();
  });
  it('shows the Annotate button on the Browser view', () => {
    renderPanel({ view: 'browser' });
    expect(screen.getByRole('button', { name: 'Annotate' })).toBeInTheDocument();
  });
});
