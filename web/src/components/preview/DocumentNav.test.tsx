// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import DocumentNav from './DocumentNav';
import type { PdfNavState, SheetNavState, ZoomNavState } from './types';

function pdfNav(overrides: Partial<PdfNavState> = {}): PdfNavState {
  return {
    kind: 'pdf',
    page: 2,
    pageCount: 5,
    zoomPercent: 80,
    fit: true,
    goTo: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    fitWidth: vi.fn(),
    ...overrides,
  };
}

describe('DocumentNav (spec 090)', () => {
  it('renders nothing without navigation', () => {
    const { container } = render(<DocumentNav nav={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('PDF: page x / y, prev / next, typed page and zoom actions', () => {
    const nav = pdfNav();
    render(<DocumentNav nav={nav} />);
    expect(screen.getByRole('textbox', { name: 'Page number' })).toHaveValue('2');
    expect(screen.getByText('of 5')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Previous page' }));
    fireEvent.click(screen.getByRole('button', { name: 'Next page' }));
    const input = screen.getByRole('textbox', { name: 'Page number' });
    fireEvent.change(input, { target: { value: '4' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(nav.goTo).toHaveBeenNthCalledWith(1, 1);
    expect(nav.goTo).toHaveBeenNthCalledWith(2, 3);
    expect(nav.goTo).toHaveBeenNthCalledWith(3, 4);

    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
    fireEvent.click(screen.getByRole('button', { name: 'Fit width' }));
    expect(nav.zoomIn).toHaveBeenCalledTimes(1);
    expect(nav.zoomOut).toHaveBeenCalledTimes(1);
    expect(nav.fitWidth).toHaveBeenCalledTimes(1);
  });

  it('PDF: the ends of the document disable prev / next', () => {
    render(<DocumentNav nav={pdfNav({ page: 5 })} />);
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled();
  });

  it('compact adds container-query classes to the secondary pieces', () => {
    render(<DocumentNav nav={pdfNav()} compact />);
    expect(screen.getByText('of 5').className).toContain('@max-[24rem]:hidden');
    expect(screen.getByText('80%').className).toContain('@max-[20rem]:hidden');
  });

  it('sheets: a compact selector that selects by index', () => {
    const nav: SheetNavState = { kind: 'sheet', sheets: ['汇总', 'Details'], active: 0, select: vi.fn() };
    render(<DocumentNav nav={nav} compact />);
    const select = screen.getByRole('combobox', { name: 'Sheets' });
    expect(select).toHaveValue('0');
    fireEvent.change(select, { target: { value: '1' } });
    expect(nav.select).toHaveBeenCalledWith(1);
  });

  it('a single sheet needs no selector', () => {
    const nav: SheetNavState = { kind: 'sheet', sheets: ['Only'], active: 0, select: vi.fn() };
    const { container } = render(<DocumentNav nav={nav} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('Word engines: zoom only', () => {
    const nav: ZoomNavState = {
      kind: 'zoom',
      zoomPercent: 100,
      fit: false,
      zoomIn: vi.fn(),
      zoomOut: vi.fn(),
      fitWidth: vi.fn(),
    };
    render(<DocumentNav nav={nav} />);
    expect(screen.queryByRole('textbox')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(nav.zoomIn).toHaveBeenCalledTimes(1);
  });
});
