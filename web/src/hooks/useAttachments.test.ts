// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Guards the state machine both composers share: caps, the too-big
// pre-flagging, retry, and the object-URL lifetime of image thumbnails.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import type React from 'react';

vi.mock('../lib/attachment-upload', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../lib/attachment-upload')>();
  return { ...actual, uploadFile: vi.fn() };
});

import { uploadFile } from '../lib/attachment-upload';
import {
  useAttachments,
  MAX_ATTACHMENTS,
  MAX_ATTACHMENT_BYTES,
} from './useAttachments';

const uploadMock = vi.mocked(uploadFile);

function textFile(name: string, size = 4) {
  const f = new File(['x'.repeat(size)], name, { type: 'text/plain' });
  return f;
}

function imageFile(name = 'shot.png') {
  return new File(['png'], name, { type: 'image/png' });
}

/** A File that reports an arbitrary size without allocating it. */
function hugeFile(name: string, size: number) {
  const f = new File(['x'], name, { type: 'application/pdf' });
  Object.defineProperty(f, 'size', { value: size });
  return f;
}

beforeEach(() => {
  uploadMock.mockReset();
  uploadMock.mockResolvedValue({ path: 'uploads/uploaded', size: 4 });
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
});

afterEach(() => vi.restoreAllMocks());

describe('useAttachments', () => {
  it('uploads an added file and records the returned path', async () => {
    uploadMock.mockResolvedValue({ path: 'uploads/notes.txt', size: 4 });
    const { result } = renderHook(() => useAttachments('p1'));

    act(() => result.current.addAttachments([textFile('notes.txt')]));

    await waitFor(() => expect(result.current.attachments[0].status).toBe('done'));
    expect(result.current.attachments[0].uploadedPath).toBe('uploads/notes.txt');
    expect(result.current.anyDone).toBe(true);
    expect(result.current.anyUploading).toBe(false);
    expect(uploadMock.mock.calls[0][0].projectId).toBe('p1');
  });

  it('flags an over-size file as an error without uploading it', async () => {
    const { result } = renderHook(() => useAttachments('p1'));

    act(() =>
      result.current.addAttachments([
        hugeFile('huge.pdf', MAX_ATTACHMENT_BYTES + 1),
      ]),
    );

    expect(result.current.attachments[0].status).toBe('error');
    expect(result.current.attachments[0].errorMessage).toBe(
      'File exceeds 10 MB limit',
    );
    expect(result.current.allError).toBe(true);
    // Give the deferred upload kick a tick to prove it never fires.
    await new Promise((r) => setTimeout(r, 5));
    expect(uploadMock).not.toHaveBeenCalled();
  });

  it('caps the staged set and reports the cap through onError', async () => {
    const onError = vi.fn();
    const { result } = renderHook(() => useAttachments('p1', { onError }));

    act(() =>
      result.current.addAttachments(
        Array.from({ length: MAX_ATTACHMENTS + 2 }, (_, i) =>
          textFile(`f${i}.txt`),
        ),
      ),
    );

    await waitFor(() =>
      expect(result.current.attachments).toHaveLength(MAX_ATTACHMENTS),
    );
    expect(onError).toHaveBeenCalledWith(
      `Only ${MAX_ATTACHMENTS} attachments allowed per message.`,
    );

    onError.mockClear();
    act(() => result.current.addAttachments([textFile('one-more.txt')]));
    expect(result.current.attachments).toHaveLength(MAX_ATTACHMENTS);
    expect(onError).toHaveBeenCalledWith(
      `Maximum of ${MAX_ATTACHMENTS} attachments per message.`,
    );
  });

  it('marks a failed upload with its message and retries on demand', async () => {
    uploadMock.mockRejectedValueOnce(new Error('Upload failed (500): boom'));
    const { result } = renderHook(() => useAttachments('p1'));

    act(() => result.current.addAttachments([textFile('notes.txt')]));
    await waitFor(() => expect(result.current.attachments[0].status).toBe('error'));
    expect(result.current.attachments[0].errorMessage).toBe(
      'Upload failed (500): boom',
    );

    uploadMock.mockResolvedValueOnce({ path: 'uploads/notes.txt', size: 4 });
    act(() => result.current.retryAttachment(result.current.attachments[0].id));

    await waitFor(() => expect(result.current.attachments[0].status).toBe('done'));
    expect(result.current.attachments[0].errorMessage).toBeUndefined();
    expect(uploadMock).toHaveBeenCalledTimes(2);
  });

  it('revokes an image thumbnail on remove and on clear', async () => {
    const revoke = vi.mocked(URL.revokeObjectURL);
    const { result } = renderHook(() => useAttachments('p1'));

    act(() => result.current.addAttachments([imageFile('a.png')]));
    await waitFor(() => expect(result.current.attachments[0].status).toBe('done'));
    expect(result.current.attachments[0].thumbnailUrl).toBe('blob:mock-url');

    act(() => result.current.removeAttachment(result.current.attachments[0].id));
    expect(result.current.attachments).toHaveLength(0);
    expect(revoke).toHaveBeenCalledWith('blob:mock-url');

    revoke.mockClear();
    act(() => result.current.addAttachments([imageFile('b.png')]));
    await waitFor(() => expect(result.current.attachments).toHaveLength(1));
    act(() => result.current.clearAttachments());
    expect(result.current.attachments).toHaveLength(0);
    expect(revoke).toHaveBeenCalledWith('blob:mock-url');
  });
});

describe('useAttachments — deferUpload (the Workbench stages, uploads at send)', () => {
  const opts = { deferUpload: true };

  it('stages added files as pending and uploads nothing', async () => {
    const { result } = renderHook(() => useAttachments('', opts));
    act(() => result.current.addAttachments([textFile('a.txt'), textFile('b.txt')]));
    await new Promise((r) => setTimeout(r, 5));
    expect(result.current.attachments.map((a) => a.status)).toEqual(['pending', 'pending']);
    expect(uploadMock).not.toHaveBeenCalled();
  });

  it('uploadAll uploads the staged ones in order into the given project and returns the payload', async () => {
    uploadMock
      .mockResolvedValueOnce({ path: 'uploads/a.txt', size: 4 })
      .mockResolvedValueOnce({ path: 'uploads/b.txt', size: 4 });
    const { result } = renderHook(() => useAttachments('', opts));
    act(() => result.current.addAttachments([textFile('a.txt'), textFile('b.txt')]));
    let payload: unknown;
    await act(async () => { payload = await result.current.uploadAll('proj-b'); });
    expect(payload).toEqual([
      { path: 'uploads/a.txt', mime: 'text/plain', size: 4 },
      { path: 'uploads/b.txt', mime: 'text/plain', size: 4 },
    ]);
    expect(uploadMock.mock.calls.map((c) => c[0].projectId)).toEqual(['proj-b', 'proj-b']);
    expect(result.current.attachments.map((a) => a.status)).toEqual(['done', 'done']);
  });

  it('uploadAll throws when one fails; the chips stay, the failed one in error', async () => {
    uploadMock
      .mockResolvedValueOnce({ path: 'uploads/a.txt', size: 4 })
      .mockRejectedValueOnce(new Error('disk full'));
    const { result } = renderHook(() => useAttachments('', opts));
    act(() => result.current.addAttachments([textFile('a.txt'), textFile('b.txt')]));
    await act(async () => {
      await expect(result.current.uploadAll('proj-b')).rejects.toThrow('disk full');
    });
    expect(result.current.attachments.map((a) => a.status)).toEqual(['done', 'error']);
    expect(result.current.attachments[1].errorMessage).toBe('disk full');
  });

  it('a retry goes back to staged, and the next uploadAll skips what that project already has', async () => {
    uploadMock
      .mockResolvedValueOnce({ path: 'uploads/a.txt', size: 4 })
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce({ path: 'uploads/b.txt', size: 4 });
    const { result } = renderHook(() => useAttachments('', opts));
    act(() => result.current.addAttachments([textFile('a.txt'), textFile('b.txt')]));
    await act(async () => { await result.current.uploadAll('proj-b').catch(() => {}); });
    act(() => result.current.retryAttachment(result.current.attachments[1].id));
    expect(result.current.attachments[1].status).toBe('pending');
    let payload: unknown;
    await act(async () => { payload = await result.current.uploadAll('proj-b'); });
    expect(uploadMock).toHaveBeenCalledTimes(3);
    expect(payload).toEqual([
      { path: 'uploads/a.txt', mime: 'text/plain', size: 4 },
      { path: 'uploads/b.txt', mime: 'text/plain', size: 4 },
    ]);
  });

  it('uploadAll with ids sends only those, and skips an over-size file', async () => {
    const { result } = renderHook(() => useAttachments('', opts));
    act(() => result.current.addAttachments([
      textFile('a.txt'), textFile('b.txt'), hugeFile('big.pdf', MAX_ATTACHMENT_BYTES + 1),
    ]));
    const [, b, big] = result.current.attachments;
    await act(async () => { await result.current.uploadAll('proj-a', [b.id, big.id]); });
    expect(uploadMock).toHaveBeenCalledTimes(1);
    expect(uploadMock.mock.calls[0][0].file.name).toBe('b.txt');
  });
});

// Attach by local path (desktop app, 2026-11-07): the native bridge picks
// files by absolute path; nothing is uploaded and there is no size cap.
describe('useAttachments — desktop app attach by path', () => {
  const entry = (name: string, size = 4) => ({
    path: `/Users/me/files/${name}`, name, size, mime: 'text/plain',
  });
  type Win = { pywebview?: unknown; __orbitalNativeDrop?: boolean };

  function desktop(pick = vi.fn(async () => [entry('a.txt'), entry('big.bin', 50 * 1024 * 1024)])) {
    (window as unknown as Win).pywebview = { api: { pick_folder: vi.fn(), pick_files: pick } };
    return pick;
  }

  afterEach(() => {
    delete (window as unknown as Win).pywebview;
    delete (window as unknown as Win).__orbitalNativeDrop;
  });

  it('the picker adds ready path chips, uploads nothing, no size cap', async () => {
    const pick = desktop();
    const browse = vi.fn();
    const { result } = renderHook(() => useAttachments('p1', { localPaths: true }));
    await act(async () => { await result.current.openPicker(browse); });
    expect(pick).toHaveBeenCalledTimes(1);
    expect(browse).not.toHaveBeenCalled();
    const [a, b] = result.current.attachments;
    expect(a).toMatchObject({ localPath: '/Users/me/files/a.txt', filename: 'a.txt', status: 'done' });
    expect(b).toMatchObject({ size: 50 * 1024 * 1024, status: 'done' });
    expect(result.current.anyDone).toBe(true);
    await new Promise((r) => setTimeout(r, 5));
    expect(uploadMock).not.toHaveBeenCalled();
  });

  it('uploadAll passes path chips straight through, marked local', async () => {
    desktop();
    const { result } = renderHook(() => useAttachments('', { localPaths: true, deferUpload: true }));
    await act(async () => { await result.current.openPicker(vi.fn()); });
    let out: unknown;
    await act(async () => { out = await result.current.uploadAll('proj-x'); });
    expect(out).toEqual([
      { path: '/Users/me/files/a.txt', mime: 'text/plain', size: 4, local: true },
      { path: '/Users/me/files/big.bin', mime: 'text/plain', size: 50 * 1024 * 1024, local: true },
    ]);
    expect(uploadMock).not.toHaveBeenCalled();
  });

  it('not the desktop app: the picker opens the page file input', async () => {
    const browse = vi.fn();
    const { result } = renderHook(() => useAttachments('p1', { localPaths: true }));
    await act(async () => { await result.current.openPicker(browse); });
    expect(browse).toHaveBeenCalledTimes(1);
    expect(result.current.attachments).toEqual([]);
  });

  it('a composer without localPaths (queue) keeps the file input even in the desktop app', async () => {
    const pick = desktop();
    const browse = vi.fn();
    const { result } = renderHook(() => useAttachments('p1'));
    await act(async () => { await result.current.openPicker(browse); });
    expect(browse).toHaveBeenCalledTimes(1);
    expect(pick).not.toHaveBeenCalled();
  });

  function dropEvent(files: File[]) {
    return {
      preventDefault: vi.fn(),
      dataTransfer: { files },
    } as unknown as React.DragEvent<HTMLElement>;
  }

  it('drop in the desktop app: files the shell sends a path for become path chips, the rest upload', async () => {
    desktop();
    (window as unknown as Win).__orbitalNativeDrop = true;
    const { result } = renderHook(() => useAttachments('p1', { localPaths: true }));
    act(() => result.current.handleDrop(dropEvent([textFile('a.txt'), textFile('web.txt')])));
    act(() => {
      window.dispatchEvent(new CustomEvent('orbital:native-drop', { detail: [entry('a.txt')] }));
    });
    await waitFor(() => expect(result.current.attachments).toHaveLength(2));
    const local = result.current.attachments.find((x) => x.localPath);
    expect(local).toMatchObject({ filename: 'a.txt', localPath: '/Users/me/files/a.txt' });
    await waitFor(() => expect(uploadMock).toHaveBeenCalledTimes(1));
    expect(uploadMock.mock.calls[0][0].file.name).toBe('web.txt');
  });

  it('drop with no native drop support uploads as before', async () => {
    desktop();
    const { result } = renderHook(() => useAttachments('p1', { localPaths: true }));
    act(() => result.current.handleDrop(dropEvent([textFile('a.txt')])));
    await waitFor(() => expect(uploadMock).toHaveBeenCalledTimes(1));
    expect(result.current.attachments[0].localPath).toBeUndefined();
  });

  it('paste always uploads, even in the desktop app', async () => {
    desktop();
    (window as unknown as Win).__orbitalNativeDrop = true;
    const { result } = renderHook(() => useAttachments('p1', { localPaths: true }));
    const file = imageFile();
    act(() => result.current.handlePaste({
      preventDefault: vi.fn(),
      clipboardData: { items: [{ kind: 'file', getAsFile: () => file }] },
    } as unknown as React.ClipboardEvent<HTMLElement>));
    await waitFor(() => expect(uploadMock).toHaveBeenCalledTimes(1));
  });
});
