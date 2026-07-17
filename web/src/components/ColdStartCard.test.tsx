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

  it('shows an inline error message when error is set (scan failure)', () => {
    render(
      <ColdStartCard
        folderName="r"
        onScan={vi.fn()}
        onSkip={vi.fn()}
        error="No API key configured. Add one in Settings."
      />,
    );
    expect(screen.getByTestId('cold-start-error').textContent).toContain(
      'No API key configured',
    );
    // Scan stays clickable so the user can retry after fixing settings.
    const scanBtn = screen.getByRole('button', { name: /scan/i }) as HTMLButtonElement;
    expect(scanBtn.disabled).toBe(false);
  });
});
