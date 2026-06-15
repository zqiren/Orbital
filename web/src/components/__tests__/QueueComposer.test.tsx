// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

afterEach(() => cleanup());
import QueueComposer from '../QueueComposer';

describe('QueueComposer', () => {
  it('submit is disabled when input is empty', () => {
    render(<QueueComposer onSubmit={() => {}} />);
    const submit = screen.getByTestId('queue-composer-submit') as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
  });

  it('submits with default priority and review flags', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'task one' } });
    const submit = screen.getByTestId('queue-composer-submit');
    fireEvent.click(submit);
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('task one', { priority: 0, review: false });
  });

  it('pin toggle sets priority=1 on submit', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'urgent' } });
    // Click "Pin to top"
    fireEvent.click(screen.getByLabelText(/Pin to top/));
    fireEvent.click(screen.getByTestId('queue-composer-submit'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('urgent', { priority: 1, review: false });
  });

  it('review toggle sets review=true on submit', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'thing' } });
    fireEvent.click(screen.getByLabelText(/Review before advance/));
    fireEvent.click(screen.getByTestId('queue-composer-submit'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('thing', { priority: 0, review: true });
  });

  it('Enter submits without Shift', async () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'enter-key' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('enter-key', { priority: 0, review: false });
  });

  it('Shift+Enter does NOT submit', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: 'multiline' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // Enter that commits an IME (e.g. Pinyin) candidate must not submit the
  // message — it belongs to the input method, not to us.
  it('Enter does NOT submit while an IME is composing (isComposing)', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: '你好' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false, isComposing: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Enter does NOT submit while an IME is composing (legacy keyCode 229)', () => {
    const onSubmit = vi.fn(() => Promise.resolve());
    render(<QueueComposer onSubmit={onSubmit} />);
    const input = screen.getByTestId('queue-composer-input');
    fireEvent.change(input, { target: { value: '你好' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false, keyCode: 229 });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('renders hint text when provided', () => {
    render(<QueueComposer onSubmit={() => {}} hint="Chat freely — queue is stopped" />);
    expect(screen.getByText(/Chat freely/)).toBeTruthy();
  });
});
