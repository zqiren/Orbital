// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import DocPreview from './DocPreview';
import { TINY_GENERATED_DOC_BASE64 } from './fixtures/tinyGeneratedDoc';

function fixtureBytes(): ArrayBuffer {
  const binary = atob(TINY_GENERATED_DOC_BASE64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

// jsdom has no Worker, so these exercise the real parser in-thread.
describe('DocPreview (spec 090, legacy .doc)', () => {
  it('renders a generated Word 97 document into a sandboxed, CSP-locked frame', async () => {
    const onNav = vi.fn();
    const onError = vi.fn();
    const { container } = render(
      <DocPreview bytes={fixtureBytes()} format="doc" fileName="tiny.doc" onNav={onNav} onError={onError} />,
    );
    await waitFor(() => expect(container.querySelector('iframe')).not.toBeNull());
    const frame = container.querySelector('iframe')!;
    expect(frame.getAttribute('sandbox')).toBe('');
    const srcdoc = frame.getAttribute('srcdoc') ?? '';
    expect(srcdoc).toContain('Content-Security-Policy');
    expect(srcdoc).toContain('预览样例');
    expect(srcdoc).toContain('红色文字');
    expect(srcdoc).not.toMatch(/<script/i);
    expect(onError).not.toHaveBeenCalled();
    // Navigation is published from an effect, after the frame's commit.
    await waitFor(() =>
      expect(onNav).toHaveBeenCalledWith(expect.objectContaining({ kind: 'zoom', zoomPercent: 100 })),
    );
  });

  it('leaves the host bytes intact for Download', async () => {
    const bytes = fixtureBytes();
    const { container } = render(
      <DocPreview bytes={bytes} format="doc" fileName="tiny.doc" onNav={vi.fn()} onError={vi.fn()} />,
    );
    await waitFor(() => expect(container.querySelector('iframe')).not.toBeNull());
    expect(bytes.byteLength).toBeGreaterThan(0);
  });

  it('corrupt bytes report unreadable instead of rendering', async () => {
    const onError = vi.fn();
    const { container } = render(
      <DocPreview
        bytes={new TextEncoder().encode('not a Word file').buffer as ArrayBuffer}
        format="doc"
        fileName="corrupt.doc"
        onNav={vi.fn()}
        onError={onError}
      />,
    );
    await waitFor(() => expect(onError).toHaveBeenCalledWith('unreadable'));
    expect(container.querySelector('iframe')).toBeNull();
  });
});
