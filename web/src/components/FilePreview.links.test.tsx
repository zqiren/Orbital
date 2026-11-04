// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 153 §4.4: a relative link inside a workspace markdown file opens the
 * file it names (resolved against the file's own folder) through the path
 * opener the panel and the drawer share. External links keep opening
 * outside; anything that is not a plain relative file link stays inert.
 */

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import FilePreview, { resolveRelativeLink } from './FilePreview';
import { OpenPathContext } from './filePreviewContext';
import type { FileContent } from '../types';

const INDEX = [
  '145 memories Claude Code kept about this project, imported from `~/.claude/projects/x/memory`',
  '',
  '- [Coupled steps](feedback_coupled_steps.md) — both sides of one fix in one commit',
  '- [Up a level](../README.md) — a sibling folder',
  '- [Site](https://example.com) — external',
  '- [Escape](../../../../etc/passwd) — leaves the workspace',
].join('\n');

function md(content: string, path: string): FileContent {
  return { path, content, size: content.length, truncated: false, type: 'text', mime: 'text/markdown' };
}

describe('resolveRelativeLink', () => {
  const from = 'orbital/imported/claude-code/INDEX.md';
  it('resolves against the file folder', () => {
    expect(resolveRelativeLink(from, 'a.md')).toBe('orbital/imported/claude-code/a.md');
    expect(resolveRelativeLink(from, './a.md#part')).toBe('orbital/imported/claude-code/a.md');
    expect(resolveRelativeLink(from, '../../INDEX.md')).toBe('orbital/INDEX.md');
    expect(resolveRelativeLink(from, 'my%20note.md')).toBe('orbital/imported/claude-code/my note.md');
    expect(resolveRelativeLink('README.md', 'docs/x.md')).toBe('docs/x.md');
  });
  it('refuses what is not a plain relative file link', () => {
    for (const href of ['https://x.y', '//x.y', 'mailto:a@b', '#top', '/etc/passwd', 'file:///x', 'C:/x.md', '../../../../x.md', '']) {
      expect(resolveRelativeLink(from, href)).toBeNull();
    }
  });
});

describe('FilePreview — relative markdown links', () => {
  it('opens the linked workspace file through the path opener', () => {
    const open = vi.fn();
    render(
      <OpenPathContext.Provider value={open}>
        <FilePreview fileContent={md(INDEX, 'orbital/imported/claude-code/INDEX.md')} loading={false} selectedPath="orbital/imported/claude-code/INDEX.md" />
      </OpenPathContext.Provider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Coupled steps' }));
    expect(open).toHaveBeenCalledWith('orbital/imported/claude-code/feedback_coupled_steps.md');
    fireEvent.click(screen.getByRole('button', { name: 'Up a level' }));
    expect(open).toHaveBeenLastCalledWith('orbital/imported/README.md');
    // External stays an external anchor; an escaping path stays inert text.
    expect(screen.getByRole('link', { name: 'Site' }).getAttribute('target')).toBe('_blank');
    expect(screen.queryByRole('button', { name: 'Escape' })).toBeNull();
    expect(screen.getByText('Escape').tagName).toBe('SPAN');
  });

  it('without an opener the links stay inert, as before', () => {
    render(<FilePreview fileContent={md(INDEX, 'INDEX.md')} loading={false} selectedPath="INDEX.md" />);
    expect(screen.queryByRole('button', { name: 'Coupled steps' })).toBeNull();
    expect(screen.getByText('Coupled steps').tagName).toBe('SPAN');
  });
});
