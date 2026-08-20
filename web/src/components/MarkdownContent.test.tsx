// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen, fireEvent } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
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

describe('MarkdownContent — external links open outside the app window (task 3)', () => {
  it('adds target=_blank and rel=noopener noreferrer to an http link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent
        content="See [the site](http://example.com/page)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const link = container.querySelector('a[href="http://example.com/page"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toContain('noopener');
    expect(link?.getAttribute('rel')).toContain('noreferrer');
  });

  it('adds target=_blank to an uppercase-scheme HTTP link (case-insensitive match)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent
        content="See [the site](HTTP://EXAMPLE.COM/page)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const link = container.querySelector('a');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toContain('noopener');
  });

  it('adds target=_blank and rel=noopener noreferrer to an https link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent
        content="See [the site](https://example.com/page)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const link = container.querySelector('a[href="https://example.com/page"]');
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toContain('noopener');
    expect(link?.getAttribute('rel')).toContain('noreferrer');
  });

  it('adds target=_blank to a protocol-relative link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent
        content="See [the site](//example.com/page)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const link = container.querySelector('a[href="//example.com/page"]');
    expect(link?.getAttribute('target')).toBe('_blank');
  });

  it('does NOT add target to a same-page fragment link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent content="See [section](#intro)." workspace={WS} onOpenPath={onOpenPath} />,
    );
    const link = container.querySelector('a[href="#intro"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBeNull();
  });

  it('does NOT add target to a mailto link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent
        content="Email [me](mailto:a@b.com)."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    const link = container.querySelector('a[href="mailto:a@b.com"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBeNull();
  });

  it('does NOT add target to a non-workspace relative link (kind-aware mode)', () => {
    const onOpenPath = vi.fn();
    const { container } = render(
      <MarkdownContent content="See [it](just-a-word)." workspace={WS} onOpenPath={onOpenPath} />,
    );
    const link = container.querySelector('a[href="just-a-word"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBeNull();
  });

  it('still renders a workspace-path link as a card, unaffected by the external-link change', () => {
    const onOpenPath = vi.fn();
    render(
      <MarkdownContent
        content="See the [Q3 Dashboard](reports/q3.html) for details."
        workspace={WS}
        onOpenPath={onOpenPath}
      />,
    );
    expect(
      screen.getByRole('button', { name: 'Open reports/q3.html in the preview panel' }),
    ).not.toBeNull();
  });

  it('adds target=_blank and rel=noopener noreferrer to an https link when workspace/onOpenPath are omitted', () => {
    const { container } = render(
      <MarkdownContent content="See [the site](https://example.com/page)." />,
    );
    const link = container.querySelector('a[href="https://example.com/page"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toContain('noopener');
    expect(link?.getAttribute('rel')).toContain('noreferrer');
  });

  it('does NOT add target to a fragment link when workspace/onOpenPath are omitted', () => {
    const { container } = render(<MarkdownContent content="See [section](#intro)." />);
    const link = container.querySelector('a[href="#intro"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBeNull();
  });

  it('does NOT add target to a relative link when workspace/onOpenPath are omitted', () => {
    const { container } = render(<MarkdownContent content="See [it](just-a-word)." />);
    const link = container.querySelector('a[href="just-a-word"]');
    expect(link).not.toBeNull();
    expect(link?.getAttribute('target')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Per-code-block copy (BACKLOG spec 068)
// ---------------------------------------------------------------------------

describe('MarkdownContent — fenced code block copy', () => {
  const writeText = vi.fn(async () => undefined);

  beforeEach(() => {
    writeText.mockClear();
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText }, configurable: true, writable: true,
    });
  });

  const FENCE = '```python\nimport os\nprint(os.getcwd())\n```';

  it('copies the fence body verbatim — no language tag, no backticks', () => {
    render(<MarkdownContent content={FENCE} />);
    fireEvent.click(screen.getByTestId('code-block-copy'));
    expect(writeText).toHaveBeenCalledWith('import os\nprint(os.getcwd())\n');
  });

  it('gives every fence its own button, each copying its own body', () => {
    render(
      <MarkdownContent content={'```\nfirst\n```\n\ntext between\n\n```\nsecond\n```'} />,
    );
    const buttons = screen.getAllByTestId('code-block-copy');
    expect(buttons).toHaveLength(2);
    fireEvent.click(buttons[1]);
    expect(writeText).toHaveBeenCalledWith('second\n');
  });

  it('works in the chat mode too (workspace + onOpenPath installed)', () => {
    render(
      <MarkdownContent content={FENCE} workspace={WS} onOpenPath={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('code-block-copy'));
    expect(writeText).toHaveBeenCalledWith('import os\nprint(os.getcwd())\n');
  });

  it('does not put a copy button on INLINE code', () => {
    render(<MarkdownContent content={'use `os.getcwd()` for that'} />);
    expect(screen.queryByTestId('code-block-copy')).toBeNull();
  });

  it('the button sits OUTSIDE the overflow-x <pre> so wide code cannot scroll it away', () => {
    const { container } = render(<MarkdownContent content={'```\nx\n```'} />);
    const btn = screen.getByTestId('code-block-copy');
    expect(container.querySelector('pre')?.contains(btn)).toBe(false);
    expect(btn.closest('.group.relative')?.querySelector('pre')).not.toBeNull();
  });
});
