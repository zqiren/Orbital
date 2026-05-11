// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import AttachmentChip from './AttachmentChip';

describe('AttachmentChip', () => {
  it('renders filename and size', () => {
    render(
      <AttachmentChip
        filename="hello.txt"
        mime="text/plain"
        size={1024}
        status="done"
      />,
    );
    expect(screen.getByText('hello.txt')).toBeInTheDocument();
    expect(screen.getByText(/1\.0 KB/)).toBeInTheDocument();
  });

  it('renders an <img> when given an image mime and a thumbnailUrl', () => {
    const { container } = render(
      <AttachmentChip
        filename="cat.png"
        mime="image/png"
        size={2048}
        status="done"
        thumbnailUrl="blob:fake"
      />,
    );
    const img = container.querySelector('img');
    expect(img).not.toBeNull();
    expect(img?.getAttribute('src')).toBe('blob:fake');
  });

  it('renders a generic icon and no <img> for non-image mimes', () => {
    const { container } = render(
      <AttachmentChip
        filename="report.pdf"
        mime="application/pdf"
        size={1024}
        status="done"
      />,
    );
    expect(container.querySelector('img')).toBeNull();
  });

  it('renders a spinner when status is "uploading"', () => {
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="uploading"
      />,
    );
    expect(screen.getByTestId('chip-spinner')).toBeInTheDocument();
  });

  it('renders a check when status is "done"', () => {
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="done"
      />,
    );
    expect(screen.getByTestId('chip-check')).toBeInTheDocument();
  });

  it('applies a red border when status is "error"', () => {
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="error"
        errorMessage="boom"
      />,
    );
    const chip = screen.getByTestId('attachment-chip');
    expect(chip.className).toMatch(/border-red/);
  });

  it('invokes onRemove when the remove button is clicked', () => {
    const onRemove = vi.fn();
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="done"
        onRemove={onRemove}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /remove attachment/i }));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it('shows a retry button (only) when status is "error" and onRetry is provided', () => {
    const onRetry = vi.fn();
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="error"
        errorMessage="oops"
        onRetry={onRetry}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /retry upload/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('does not render a retry button when status is not "error"', () => {
    render(
      <AttachmentChip
        filename="x.txt"
        mime="text/plain"
        size={1}
        status="done"
        onRetry={() => {}}
      />,
    );
    expect(screen.queryByRole('button', { name: /retry upload/i })).toBeNull();
  });

  it('truncates a long filename but keeps the full name in the title attribute', () => {
    const longName = 'a'.repeat(60) + '.txt';
    const { container } = render(
      <AttachmentChip
        filename={longName}
        mime="text/plain"
        size={1}
        status="done"
      />,
    );
    // Find the inner span (truncate class) — the wrapper div also has a title
    // attribute, so we target the span specifically.
    const span = container.querySelector('span.truncate');
    expect(span).not.toBeNull();
    expect(span?.getAttribute('title')).toBe(longName);
    expect(span?.textContent?.includes('…')).toBe(true);
    expect(span?.textContent?.length).toBeLessThan(longName.length);
  });
});
