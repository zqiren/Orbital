// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ColdStartCard } from './ColdStartCard';

describe('ColdStartCard', () => {
  it('renders folder name and Scan/Skip', () => {
    render(<ColdStartCard folderName="my-repo" onScan={vi.fn()} onSkip={vi.fn()} />);
    expect(screen.getByText(/my-repo/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /scan/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /skip/i })).toBeTruthy();
  });

  it('calls onScan when Scan clicked', async () => {
    const onScan = vi.fn();
    render(<ColdStartCard folderName="r" onScan={onScan} onSkip={vi.fn()} />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /scan/i }));
    });
    expect(onScan).toHaveBeenCalledOnce();
  });
});
