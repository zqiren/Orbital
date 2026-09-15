// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../config';
import { fileContentPath, useFileBytes, useFiles } from './useFiles';

const URL_A = '/api/v2/projects/p1/files/preview?path=a.pdf';
const URL_B = '/api/v2/projects/p1/files/preview?path=b.pdf';

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A fetch that never settles on its own and rejects when aborted. */
function pendingUntilAborted(_url: string, init?: RequestInit): Promise<Response> {
  return new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () =>
      reject(new DOMException('Aborted', 'AbortError')),
    );
  });
}

describe('file content requests opt in to the document envelope (spec 090)', () => {
  it('fileContentPath encodes both segments and adds document_preview=1', () => {
    expect(fileContentPath('p 1', 'docs/a b.pdf')).toBe(
      '/api/v2/projects/p%201/files/content?path=docs%2Fa%20b.pdf&document_preview=1',
    );
  });

  it('useFiles().getFileContent sends the flag', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ path: 'docs/a.pdf', type: 'document', format: 'pdf', content: '', size: 1, truncated: false }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    const { result } = renderHook(() => useFiles());
    let data: Awaited<ReturnType<typeof result.current.getFileContent>> = null;
    await act(async () => {
      data = await result.current.getFileContent('p1', 'docs/a.pdf');
    });
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain('/api/v2/projects/p1/files/content?path=docs%2Fa.pdf');
    expect(url).toContain('document_preview=1');
    expect(data).toMatchObject({ type: 'document', format: 'pdf' });
  });
});

describe('useFileBytes (spec 090)', () => {
  it('stays idle and fetches nothing for a null URL', () => {
    const { result } = renderHook(() => useFileBytes(null));
    expect(result.current.status).toBe('idle');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('fetches the preview URL as bytes', async () => {
    fetchMock.mockResolvedValue(
      new Response(new Uint8Array([37, 80, 68, 70]), {
        status: 200,
        headers: { 'content-type': 'application/pdf' },
      }),
    );
    const { result } = renderHook(() => useFileBytes(URL_A));
    expect(result.current.status).toBe('loading');
    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(new Uint8Array(result.current.bytes!)).toEqual(new Uint8Array([37, 80, 68, 70]));
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/api\/v2\/projects\/p1\/files\/preview\?path=a\.pdf$/);
  });

  it('decodes the base64 envelope a relayed request gets', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ encoding: 'base64', content: btoa(String.fromCharCode(0, 128, 255)) }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    const { result } = renderHook(() => useFileBytes(URL_A));
    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(new Uint8Array(result.current.bytes!)).toEqual(new Uint8Array([0, 128, 255]));
  });

  it('reports an HTTP failure with its status and detail', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: 'preview_unavailable: too_large' }), {
        status: 413,
        headers: { 'content-type': 'application/json' },
      }),
    );
    const { result } = renderHook(() => useFileBytes(URL_A));
    await waitFor(() => expect(result.current.status).toBe('error'));
    const error = result.current.error as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(413);
    expect(error.message).toBe('preview_unavailable: too_large');
  });

  it('aborts the in-flight fetch when the URL changes and on unmount', () => {
    fetchMock.mockImplementation(pendingUntilAborted);
    const { result, rerender, unmount } = renderHook(({ url }) => useFileBytes(url), {
      initialProps: { url: URL_A },
    });
    const first = fetchMock.mock.calls[0][1].signal as AbortSignal;
    expect(first.aborted).toBe(false);

    rerender({ url: URL_B });
    expect(first.aborted).toBe(true);
    expect(result.current.status).toBe('loading');

    const second = fetchMock.mock.calls[1][1].signal as AbortSignal;
    unmount();
    expect(second.aborted).toBe(true);
  });

  it('never shows the previous file’s bytes for a new URL', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(new Uint8Array([1]), { status: 200, headers: { 'content-type': 'application/pdf' } }),
    );
    const { result, rerender } = renderHook(({ url }) => useFileBytes(url), {
      initialProps: { url: URL_A },
    });
    await waitFor(() => expect(result.current.status).toBe('ready'));

    fetchMock.mockImplementation(pendingUntilAborted);
    rerender({ url: URL_B });
    expect(result.current.status).toBe('loading');
    expect(result.current.bytes).toBeNull();
  });
});
