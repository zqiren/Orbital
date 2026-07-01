// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import MarkdownContent from './MarkdownContent';

const WS = '/Users/you/repo';

describe('MarkdownContent — kind-aware clickable paths (spec 002)', () => {
  it('renders a previewable-artifact markdown link as a card that opens the relative path', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="See the [Q3 Dashboard](reports/q3.html) for details."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const card = screen.getByRole('button', {
      name: 'Open reports/q3.html in the preview panel',
    });
    // Card shows the link label, the filename subtitle, and an "Open" affordance.
    expect(card.textContent).toContain('Q3 Dashboard');
    expect(card.textContent).toContain('q3.html');
    expect(card.textContent).toContain('Open');

    fireEvent.click(card);
    expect(onOpenPath).toHaveBeenCalledWith('reports/q3.html');
  });

  it('renders a source-file markdown link as an inline chip (not a card)', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="Edited [the script](src/main.py) just now."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const chip = screen.getByRole('button', {
      name: 'Open src/main.py in the preview panel',
    });
    // A chip is just the label — no "Open" affordance text, no filename subtitle.
    expect(chip.textContent).toBe('the script');

    fireEvent.click(chip);
    expect(onOpenPath).toHaveBeenCalledWith('src/main.py');
  });

  it('strips the workspace prefix from an absolute path in a markdown link', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="Wrote [report](/Users/you/repo/out/report.csv)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const card = screen.getByRole('button', {
      name: 'Open out/report.csv in the preview panel',
    });
    fireEvent.click(card);
    expect(onOpenPath).toHaveBeenCalledWith('out/report.csv');
  });

  it('renders a workspace path inside an inline-code span as a clickable chip', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="Check `src/utils/helpers.ts` for the helper."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const chip = screen.getByRole('button', {
      name: 'Open src/utils/helpers.ts in the preview panel',
    });
    fireEvent.click(chip);
    expect(onOpenPath).toHaveBeenCalledWith('src/utils/helpers.ts');
  });

  it('does NOT linkify inline code that is not a path', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent content="Run `npm install` first." workspace={WS} onOpenPath={onOpenPath} />,
    );
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('does NOT linkify a path outside the workspace', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="See [hosts](/etc/hosts)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('does NOT linkify a path inside a fenced code block', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content={'```\nsrc/main.py\n```'}
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('renders bare (no linkification) when workspace/onOpenPath are omitted', () => {
    const { container } = render(
      <MarkdownContent content="See [Q3 Dashboard](reports/q3.html) and `src/main.py`." />,
    );
    expect(screen.queryByRole('button')).toBeNull();
    // The link is still a plain anchor; the inline code is still a <code>.
    expect(container.querySelector('a')).not.toBeNull();
    expect(container.querySelector('code')).not.toBeNull();
  });
});
