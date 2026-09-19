// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';

const { apiMock, relay, previewProps } = vi.hoisted(() => ({
  apiMock: vi.fn(async () => ({ entries: [] })),
  relay: { value: false },
  previewProps: { current: {} as Record<string, unknown> },
}));

vi.mock('../config', () => ({
  api: apiMock,
  BASE_URL: '',
  get isRelayMode() {
    return relay.value;
  },
}));

vi.mock('./FilePreview', () => ({
  default: (props: Record<string, unknown>) => {
    previewProps.current = props;
    return <div>File preview</div>;
  },
}));

import FileExplorer from './FileExplorer';

beforeEach(() => {
  // jsdom reports no platform; the reveal is offered on macOS and Windows.
  Object.defineProperty(window.navigator, 'platform', { value: 'MacIntel', configurable: true });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  apiMock.mockImplementation(async () => ({ entries: [] }));
  relay.value = false;
  previewProps.current = {};
  delete (window.navigator as unknown as { platform?: string }).platform;
});

describe('FileExplorer layout', () => {
  it('keeps the tree pane flex layout free of the conflicting desktop block utility', async () => {
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<FileExplorer projectId="project-1" />));
    });
    const treePane = container.firstElementChild?.firstElementChild;

    expect(treePane).toHaveClass('flex', 'flex-col', 'min-h-0');
    expect(treePane).not.toHaveClass('md:block');
  });
});

describe('FileExplorer initialPath (spec 078 D15)', () => {
  it('expands the ancestors and selects the file, fetching its content', async () => {
    apiMock.mockImplementation((async (url: string) => {
      if (url.includes('/files/content?')) return { path: 'docs/notes/plan.md', content: '# plan', type: 'text' };
      if (url.endsWith('/files')) return { entries: [{ name: 'docs', type: 'directory' }] };
      if (url.includes('path=docs%2Fnotes')) return { entries: [{ name: 'plan.md', type: 'file', size: 6 }] };
      if (url.includes('path=docs')) return { entries: [{ name: 'notes', type: 'directory' }] };
      return { entries: [] };
    }) as unknown as () => Promise<{ entries: never[] }>);
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<FileExplorer projectId="project-1" initialPath="docs/notes/plan.md" />));
    });
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });

    const calls = (apiMock.mock.calls as unknown as unknown[][]).map((c) => String(c[0]));
    expect(calls.some((u) => u.includes('path=docs') && !u.includes('notes'))).toBe(true);
    expect(calls.some((u) => u.includes('path=docs%2Fnotes'))).toBe(true);
    expect(calls.some((u) => u.includes('/files/content?path=docs%2Fnotes%2Fplan.md'))).toBe(true);
    // Spec 090: the Files tab renders documents, so it opts in to the envelope.
    expect(calls.some((u) => u.includes('/files/content?') && u.includes('document_preview=1'))).toBe(true);
    expect(container.textContent).toContain('plan.md');
  });
});

describe('FileExplorer — reveal in the file manager (spec 093)', () => {
  const TREE_URLS = (url: string) => {
    if (url.includes('/files/reveal')) return { revealed: true };
    if (url.includes('/files/content?')) return { path: 'README.md', content: 'hi', type: 'text', size: 2 };
    if (url.endsWith('/files')) {
      return { entries: [{ name: 'docs', type: 'directory' }, { name: 'README.md', type: 'file', size: 2 }] };
    }
    return { entries: [] };
  };

  function revealCalls(): unknown[] {
    return (apiMock.mock.calls as unknown as [string, RequestInit?][])
      .filter(([url]) => url.includes('/files/reveal'))
      .map(([url, init]) => {
        expect(url).toBe('/api/v2/projects/project-1/files/reveal');
        expect(init?.method).toBe('POST');
        return JSON.parse(String(init?.body)).path;
      });
  }

  async function renderTree() {
    apiMock.mockImplementation((async (url: string) => TREE_URLS(url)) as unknown as () => Promise<{ entries: never[] }>);
    await act(async () => {
      render(<FileExplorer projectId="project-1" />);
    });
  }

  it('file and directory rows each carry a reveal action that POSTs the row path', async () => {
    await renderTree();
    const docsRow = screen.getByRole('button', { name: 'docs' });
    const readmeRow = screen.getByRole('button', { name: 'README.md' });
    const [docsReveal] = Array.from(docsRow.querySelectorAll('button'));
    const [readmeReveal] = Array.from(readmeRow.querySelectorAll('button'));
    expect(docsReveal).toHaveAccessibleName('Reveal in Finder');
    expect(readmeReveal).toHaveAccessibleName('Reveal in Finder');

    await act(async () => { fireEvent.click(docsReveal); });
    await act(async () => { fireEvent.click(readmeReveal); });
    expect(revealCalls()).toEqual(['docs', 'README.md']);

    // stopPropagation: revealing neither expanded the folder nor opened the file.
    const urls = (apiMock.mock.calls as unknown as [string][]).map(([u]) => u);
    expect(urls.some((u) => u.includes('path=docs'))).toBe(false);
    expect(urls.some((u) => u.includes('/files/content?'))).toBe(false);
    expect(docsRow).toHaveAttribute('aria-expanded', 'false');
  });

  it('rows still expand and select on click, Enter and Space', async () => {
    await renderTree();
    const docsRow = screen.getByRole('button', { name: 'docs' });
    await act(async () => { fireEvent.click(docsRow); });
    expect((apiMock.mock.calls as unknown as [string][]).some(([u]) => u.includes('path=docs'))).toBe(true);
    expect(docsRow).toHaveAttribute('aria-expanded', 'true');
    await act(async () => { fireEvent.keyDown(docsRow, { key: 'Enter' }); });
    expect(docsRow).toHaveAttribute('aria-expanded', 'false');

    const readmeRow = screen.getByRole('button', { name: 'README.md' });
    await act(async () => { fireEvent.keyDown(readmeRow, { key: ' ' }); });
    expect(
      (apiMock.mock.calls as unknown as [string][]).some(([u]) => u.includes('/files/content?path=README.md')),
    ).toBe(true);
  });

  it('the footer opens the project folder itself', async () => {
    await renderTree();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open project folder' }));
    });
    expect(revealCalls()).toEqual(['']);
  });

  it('the preview gets onReveal, which POSTs the previewed path', async () => {
    await renderTree();
    const onReveal = previewProps.current.onReveal as (path: string) => void;
    expect(typeof onReveal).toBe('function');
    await act(async () => { onReveal('docs/plan.md'); });
    expect(revealCalls()).toEqual(['docs/plan.md']);
  });

  it('says so when the daemon could not open the file manager', async () => {
    await renderTree();
    apiMock.mockImplementation((async (url: string) => {
      if (url.includes('/files/reveal')) throw new Error('Path not found');
      return TREE_URLS(url);
    }) as unknown as () => Promise<{ entries: never[] }>);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open project folder' }));
    });
    expect(screen.getByText("Couldn't open the file manager")).toBeInTheDocument();
  });

  it('offers nothing through the relay', async () => {
    relay.value = true;
    await renderTree();
    expect(screen.queryByRole('button', { name: 'Reveal in Finder' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Open project folder' })).toBeNull();
    expect(previewProps.current.onReveal).toBeUndefined();
  });

  it('offers nothing on a client with no Finder / File Explorer', async () => {
    Object.defineProperty(window.navigator, 'platform', { value: 'iPhone', configurable: true });
    await renderTree();
    expect(screen.queryByRole('button', { name: 'Reveal in Finder' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Open project folder' })).toBeNull();
  });

  it('is labelled for File Explorer on Windows', async () => {
    Object.defineProperty(window.navigator, 'platform', { value: 'Win32', configurable: true });
    await renderTree();
    expect(screen.getAllByRole('button', { name: 'Show in File Explorer' })).toHaveLength(2);
  });
});
