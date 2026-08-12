// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// The composer uploads through the shared attachment hook. Stub only the
// network call — humanSize/timestampedName stay real (AttachmentChip uses
// humanSize to render the chip's size line).
vi.mock('../../lib/attachment-upload', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../lib/attachment-upload')>();
  return { ...actual, uploadFile: vi.fn() };
});

import { uploadFile } from '../../lib/attachment-upload';
import QueueComposer from '../QueueComposer';

const uploadMock = vi.mocked(uploadFile);

beforeEach(() => {
  uploadMock.mockReset();
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function pngFile(name = 'shot.png') {
  return new File(['binary'], name, { type: 'image/png' });
}

describe('QueueComposer', () => {
  it('submit is disabled when input is empty', () => {
    render(<QueueComposer projectId="p1" onSubmit={() => {}} />);
    const submit = screen.getByTestId('queue-composer-submit') as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
  });

  it('submits with default priority and review flags', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'task one' } });
    const submit = screen.getByTestId('queue-composer-submit');
    fireEvent.click(submit);
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('task one', {
      priority: 0,
      review: false,
      fileRefs: [],
    });
  });

  it('pin toggle sets priority=1 on submit', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'urgent' } });
    // Click "Pin to top"
    fireEvent.click(screen.getByLabelText(/Pin to top/));
    fireEvent.click(screen.getByTestId('queue-composer-submit'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('urgent', {
      priority: 1,
      review: false,
      fileRefs: [],
    });
  });

  it('review toggle sets review=true on submit', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'thing' } });
    fireEvent.click(screen.getByLabelText(/Review before advance/));
    fireEvent.click(screen.getByTestId('queue-composer-submit'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('thing', {
      priority: 0,
      review: true,
      fileRefs: [],
    });
  });

  it('Enter submits without Shift', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'enter-key' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('enter-key', {
      priority: 0,
      review: false,
      fileRefs: [],
    });
  });

  it('Shift+Enter does NOT submit', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'multiline' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // Enter that commits an IME (e.g. Pinyin) candidate must not submit the
  // message — it belongs to the input method, not to us.
  it('Enter does NOT submit while an IME is composing (isComposing)', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: '你好' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false, isComposing: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Enter does NOT submit while an IME is composing (legacy keyCode 229)', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: '你好' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false, keyCode: 229 });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('renders hint text when provided', () => {
    render(
      <QueueComposer
        projectId="p1"
        onSubmit={() => {}}
        hint="Chat freely — queue is stopped"
      />,
    );
    expect(screen.getByText(/Chat freely/)).toBeTruthy();
  });
});

type SubmitFn = (
  content: string,
  opts: { priority: number; review: boolean; fileRefs: string[] },
) => Promise<void>;

describe('QueueComposer — attachments', () => {
  it('a picked file uploads, shows a chip, and submits as bare content + fileRefs', async () => {
    uploadMock.mockResolvedValue({
      path: 'uploads/2026-08-11T101010-shot.png',
      size: 6,
    });
    const onSubmit = vi.fn<SubmitFn>(async () => {});
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);

    await userEvent.upload(
      screen.getByTestId('queue-attachment-file-input'),
      pngFile(),
    );

    // Chip appears and reaches the uploaded state.
    expect(await screen.findByTestId('attachment-chip')).toBeTruthy();
    await screen.findByTestId('chip-check');
    expect(uploadMock).toHaveBeenCalledTimes(1);
    expect(uploadMock.mock.calls[0][0].projectId).toBe('p1');

    fireEvent.change(screen.getByTestId('queue-composer-input'), {
      target: { value: 'crop this and add it to the deck' },
    });
    fireEvent.click(screen.getByTestId('queue-composer-submit'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('crop this and add it to the deck', {
      priority: 0,
      review: false,
      fileRefs: ['uploads/2026-08-11T101010-shot.png'],
    });
    // The dispatcher builds the <attached_files> block server-side; a
    // client-built one here would be delivered twice.
    expect(onSubmit.mock.calls[0][0]).not.toContain('attached_files');

    // Chips clear once the item is queued.
    await waitFor(() => expect(screen.queryByTestId('attachment-chip')).toBeNull());
  });

  it('pasting an image stages it as an attachment', async () => {
    uploadMock.mockResolvedValue({ path: 'uploads/pasted.png', size: 6 });
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);

    const file = pngFile('pasted.png');
    fireEvent.paste(screen.getByTestId('queue-composer-input'), {
      clipboardData: {
        items: [{ kind: 'file', type: 'image/png', getAsFile: () => file }],
      },
    });

    expect(await screen.findByTestId('attachment-chip')).toBeTruthy();
    await screen.findByTestId('chip-check');

    fireEvent.change(screen.getByTestId('queue-composer-input'), {
      target: { value: 'describe this screenshot' },
    });
    fireEvent.click(screen.getByTestId('queue-composer-submit'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('describe this screenshot', {
      priority: 0,
      review: false,
      fileRefs: ['uploads/pasted.png'],
    });
  });

  it('blocks submit while an upload is still in flight', async () => {
    // Never resolves — the chip stays in the uploading state.
    uploadMock.mockReturnValue(new Promise(() => {}));
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);

    fireEvent.change(screen.getByTestId('queue-composer-input'), {
      target: { value: 'wait for me' },
    });
    await userEvent.upload(
      screen.getByTestId('queue-attachment-file-input'),
      pngFile(),
    );
    await screen.findByTestId('chip-spinner');

    const submit = screen.getByTestId('queue-composer-submit') as HTMLButtonElement;
    await waitFor(() => expect(submit.disabled).toBe(true));
    fireEvent.click(submit);
    expect(onSubmit).not.toHaveBeenCalled();

    // Enter is gated by the same rule.
    fireEvent.keyDown(screen.getByTestId('queue-composer-input'), {
      key: 'Enter',
      shiftKey: false,
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('a failed upload offers retry and is excluded from the submitted fileRefs', async () => {
    uploadMock.mockRejectedValueOnce(new Error('Upload failed (500): boom'));
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer projectId="p1" onSubmit={onSubmit} />);

    await userEvent.upload(
      screen.getByTestId('queue-attachment-file-input'),
      pngFile('broken.png'),
    );

    const retry = await screen.findByLabelText('Retry upload');
    expect(retry).toBeTruthy();

    fireEvent.change(screen.getByTestId('queue-composer-input'), {
      target: { value: 'still a valid task' },
    });
    fireEvent.click(screen.getByTestId('queue-composer-submit'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('still a valid task', {
      priority: 0,
      review: false,
      fileRefs: [],
    });
  });

  it('retry re-uploads the failed chip', async () => {
    uploadMock.mockRejectedValueOnce(new Error('Upload failed (500): boom'));
    render(<QueueComposer projectId="p1" onSubmit={() => {}} />);

    await userEvent.upload(
      screen.getByTestId('queue-attachment-file-input'),
      pngFile('broken.png'),
    );
    const retry = await screen.findByLabelText('Retry upload');

    uploadMock.mockResolvedValueOnce({ path: 'uploads/broken.png', size: 6 });
    fireEvent.click(retry);

    await screen.findByTestId('chip-check');
    expect(uploadMock).toHaveBeenCalledTimes(2);
  });
});
