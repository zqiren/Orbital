// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { humanSize, timestampedName, uploadFile } from './attachment-upload';

describe('timestampedName', () => {
  it('prefixes with a YYYY-MM-DDTHHMMSS timestamp', () => {
    const out = timestampedName('screenshot.png');
    expect(out).toMatch(/^\d{4}-\d{2}-\d{2}T\d{6}-screenshot\.png$/);
  });

  it('sanitises whitespace, parentheses, and other special chars', () => {
    const out = timestampedName('My Photo (2).jpg');
    expect(out).not.toContain(' ');
    expect(out).not.toContain('(');
    expect(out).not.toContain(')');
    expect(out.endsWith('.jpg')).toBe(true);
    expect(out).toMatch(/-My-Photo-2\.jpg$/);
  });

  it('strips path-traversal components', () => {
    const out = timestampedName('..\\..\\evil.txt');
    expect(out).not.toContain('..');
    expect(out).not.toContain('\\');
    expect(out.endsWith('.txt')).toBe(true);
  });

  it('caps the filename at 200 chars and preserves the extension', () => {
    const long = 'a'.repeat(300) + '.txt';
    const out = timestampedName(long);
    expect(out.length).toBeLessThanOrEqual(200);
    expect(out.endsWith('.txt')).toBe(true);
  });

  it('falls back to "file" when sanitization removes everything', () => {
    const out = timestampedName('???');
    expect(out).toMatch(/^\d{4}-\d{2}-\d{2}T\d{6}-file$/);
  });
});

describe('humanSize', () => {
  it.each([
    [0, '0 B'],
    [1, '1 B'],
    [1023, '1023 B'],
    [1024, '1.0 KB'],
    [1536, '1.5 KB'],
    [1024 * 1024, '1.0 MB'],
    [10 * 1024 * 1024, '10.0 MB'],
    [1024 * 1024 * 1024, '1.0 GB'],
  ])('humanSize(%i) -> %s', (input, expected) => {
    expect(humanSize(input)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// uploadFile — uses a mock XHR
// ---------------------------------------------------------------------------

interface MockXhrEvent {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: FormData | null;
  formFile: File | null;
  formFilename: string | null;
}

function installMockXhr(opts: {
  status: number;
  responseText: string;
  capture: MockXhrEvent;
}) {
  class MockXhr {
    upload = { onprogress: null as ((e: ProgressEvent) => void) | null };
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    status = 0;
    responseText = '';
    private _method = '';
    private _url = '';
    private _headers: Record<string, string> = {};
    open(method: string, url: string) {
      this._method = method;
      this._url = url;
    }
    setRequestHeader(name: string, value: string) {
      this._headers[name] = value;
    }
    send(body: FormData) {
      opts.capture.url = this._url;
      opts.capture.method = this._method;
      opts.capture.headers = this._headers;
      opts.capture.body = body;
      const file = body.get('file');
      if (file instanceof File) {
        opts.capture.formFile = file;
        opts.capture.formFilename = file.name;
      }
      // Fire load on next tick.
      setTimeout(() => {
        this.status = opts.status;
        this.responseText = opts.responseText;
        this.onload?.();
      }, 0);
    }
  }
  vi.stubGlobal('XMLHttpRequest', MockXhr as unknown as typeof XMLHttpRequest);
}

beforeEach(() => {
  // Quiet localStorage between tests.
  try {
    window.localStorage.clear();
  } catch {
    // ignore
  }
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('uploadFile', () => {
  it('POSTs multipart with the timestamped filename and resolves with the response body', async () => {
    const capture: MockXhrEvent = {
      url: '',
      method: '',
      headers: {},
      body: null,
      formFile: null,
      formFilename: null,
    };
    installMockXhr({
      status: 200,
      responseText: JSON.stringify({ path: 'uploads/x.txt', size: 5 }),
      capture,
    });

    const file = new File(['hello'], 'hello.txt', { type: 'text/plain' });
    const result = await uploadFile({
      projectId: 'proj_1',
      file,
      baseUrl: 'http://localhost:8000',
      isRelayMode: false,
    });

    expect(result).toEqual({ path: 'uploads/x.txt', size: 5 });
    expect(capture.method).toBe('POST');
    expect(capture.url).toBe(
      'http://localhost:8000/api/v2/projects/proj_1/files/upload?path=/uploads/',
    );
    // FormData should carry a file with the timestamped name.
    expect(capture.formFilename).toMatch(/^\d{4}-\d{2}-\d{2}T\d{6}-hello\.txt$/);
    // No Authorization header in non-relay mode.
    expect(capture.headers['Authorization']).toBeUndefined();
  });

  it('attaches the relay JWT in Authorization when isRelayMode is true', async () => {
    window.localStorage.setItem('relay_jwt', 'fake.jwt.token');
    const capture: MockXhrEvent = {
      url: '',
      method: '',
      headers: {},
      body: null,
      formFile: null,
      formFilename: null,
    };
    installMockXhr({
      status: 200,
      responseText: JSON.stringify({ path: 'uploads/y.txt', size: 3 }),
      capture,
    });

    const file = new File(['hi!'], 'hi.txt', { type: 'text/plain' });
    await uploadFile({
      projectId: 'proj_2',
      file,
      baseUrl: 'https://relay.example.com',
      isRelayMode: true,
    });

    expect(capture.headers['Authorization']).toBe('Bearer fake.jwt.token');
  });

  it('rejects with a useful message on a 413 response', async () => {
    const capture: MockXhrEvent = {
      url: '',
      method: '',
      headers: {},
      body: null,
      formFile: null,
      formFilename: null,
    };
    installMockXhr({
      status: 413,
      responseText: JSON.stringify({ detail: 'file too large' }),
      capture,
    });

    const file = new File(['x'.repeat(11 * 1024 * 1024)], 'big.bin', {
      type: 'application/octet-stream',
    });
    await expect(
      uploadFile({
        projectId: 'proj_3',
        file,
        baseUrl: 'http://localhost:8000',
        isRelayMode: false,
      }),
    ).rejects.toThrowError(/413.*file too large/);
  });
});
