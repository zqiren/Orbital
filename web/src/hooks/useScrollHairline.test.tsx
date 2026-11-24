// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect } from 'vitest';
import { render, fireEvent, act } from '@testing-library/react';
import { useRef, useState } from 'react';
import { useParentScrollHairline, useScrollHairline } from './useScrollHairline';

/** jsdom has no layout: give a box a scroll height, client height and top. */
function setScroll(el: HTMLElement, { top, height = 1000, client = 200 }: { top: number; height?: number; client?: number }) {
  Object.defineProperty(el, 'scrollHeight', { configurable: true, value: height });
  Object.defineProperty(el, 'clientHeight', { configurable: true, value: client });
  el.scrollTop = top;
}

function Pane({ resetKey, nested = false }: { resetKey?: string; nested?: boolean }) {
  const { scrolled, scrollProps } = useScrollHairline(resetKey);
  const inner = useScrollHairline();
  return (
    <div>
      <header data-testid="head" data-scrolled={scrolled ? 'true' : 'false'} />
      <div {...scrollProps}>
        <div data-testid="scroller" />
        {nested && (
          <div {...inner.scrollProps}>
            <header data-testid="inner-head" data-scrolled={inner.scrolled ? 'true' : 'false'} />
            <div data-testid="inner-scroller" />
          </div>
        )}
      </div>
    </div>
  );
}

describe('useScrollHairline', () => {
  it('shows the hairline only while the content under the header is scrolled', () => {
    const { getByTestId } = render(<Pane />);
    const sc = getByTestId('scroller');
    expect(getByTestId('head').dataset.scrolled).toBe('false');
    setScroll(sc, { top: 40 });
    fireEvent.scroll(sc);
    expect(getByTestId('head').dataset.scrolled).toBe('true');
    setScroll(sc, { top: 0 });
    fireEvent.scroll(sc);
    expect(getByTestId('head').dataset.scrolled).toBe('false');
  });

  it('ignores a box that only scrolls sideways (a wide code block, a grid)', () => {
    const { getByTestId } = render(<Pane />);
    const sc = getByTestId('scroller');
    setScroll(sc, { top: 40 });
    fireEvent.scroll(sc);
    const wide = document.createElement('pre');
    sc.appendChild(wide);
    setScroll(wide, { top: 0, height: 100, client: 100 });
    fireEvent.scroll(wide);
    expect(getByTestId('head').dataset.scrolled).toBe('true');
  });

  it('leaves a nested header its own scroller', () => {
    const { getByTestId } = render(<Pane nested />);
    const inner = getByTestId('inner-scroller');
    setScroll(inner, { top: 80 });
    fireEvent.scroll(inner);
    expect(getByTestId('inner-head').dataset.scrolled).toBe('true');
    expect(getByTestId('head').dataset.scrolled).toBe('false');
  });

  it('clears when what the header heads is replaced', () => {
    function Host() {
      const [k, setK] = useState('a');
      return (
        <>
          <button type="button" data-testid="swap" onClick={() => setK('b')} />
          <Pane resetKey={k} />
        </>
      );
    }
    const { getByTestId } = render(<Host />);
    const sc = getByTestId('scroller');
    setScroll(sc, { top: 40 });
    fireEvent.scroll(sc);
    expect(getByTestId('head').dataset.scrolled).toBe('true');
    act(() => { getByTestId('swap').click(); });
    expect(getByTestId('head').dataset.scrolled).toBe('false');
  });
});

describe('useParentScrollHairline', () => {
  function FileRow() {
    const ref = useRef<HTMLDivElement>(null);
    const scrolled = useParentScrollHairline(ref);
    return <div ref={ref} data-testid="file-head" data-scrolled={scrolled ? 'true' : 'false'} />;
  }
  function Panel() {
    const { scrolled, scrollProps } = useScrollHairline();
    return (
      <div>
        <header data-testid="panel-head" data-scrolled={scrolled ? 'true' : 'false'} />
        <div {...scrollProps}>
          <div>
            <FileRow />
            <div data-testid="doc" />
          </div>
        </div>
      </div>
    );
  }

  it("follows the scroller beside it, and the panel's own header leaves it alone", () => {
    const { getByTestId } = render(<Panel />);
    const doc = getByTestId('doc');
    setScroll(doc, { top: 120 });
    fireEvent.scroll(doc);
    expect(getByTestId('file-head').dataset.scrolled).toBe('true');
    expect(getByTestId('panel-head').dataset.scrolled).toBe('false');
    setScroll(doc, { top: 0 });
    fireEvent.scroll(doc);
    expect(getByTestId('file-head').dataset.scrolled).toBe('false');
  });
});
