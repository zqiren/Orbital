// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import InlineMarkdown, { plainInline } from './InlineMarkdown';

afterEach(cleanup);

describe('InlineMarkdown', () => {
  it('bold and inline code render as elements, with no paragraph', () => {
    const { container } = render(<InlineMarkdown text={'**a** `b`'} />);
    expect(container.querySelector('strong')).toHaveTextContent('a');
    expect(container.querySelector('code')).toHaveTextContent('b');
    expect(container.querySelector('p')).toBeNull();
    expect(container.textContent).toBe('a b');
  });

  it('a heading keeps its text and loses the block', () => {
    const { container } = render(<InlineMarkdown text="# x" />);
    expect(container.textContent).toBe('x');
    expect(container.querySelector('h1')).toBeNull();
  });

  it('paragraphs and list items join with a space; images are dropped', () => {
    const { container } = render(<InlineMarkdown text={'one\n\ntwo\n\n- three\n- four\n\n![pic](x.png)'} />);
    expect(container.textContent).toBe('one two three four');
    expect(container.querySelector('p, ul, li, img')).toBeNull();
  });

  it('keeps ①②③ and CJK text as written', () => {
    const { container } = render(<InlineMarkdown text={'选哪个：①**线上** ②`线下` ③都去'} />);
    expect(container.textContent).toBe('选哪个：①线上 ②线下 ③都去');
    expect(container.querySelector('strong')).toHaveTextContent('线上');
  });

  it('external links open outside; other hrefs are inert; links={false} keeps only the text', () => {
    const { container, rerender } = render(<InlineMarkdown text={'[site](https://example.com) [file](/etc/x)'} />);
    const a = container.querySelectorAll('a');
    expect(a).toHaveLength(1);
    expect(a[0]).toHaveAttribute('target', '_blank');
    rerender(<InlineMarkdown text={'[site](https://example.com)'} links={false} />);
    expect(container.querySelector('a')).toBeNull();
    expect(container.textContent).toBe('site');
  });
});

describe('plainInline', () => {
  it('strips bold, code, strikethrough and link marks', () => {
    expect(plainInline('看**这里** 和 `x` ~~旧~~ __u__ [doc](https://e.com)')).toBe('看这里 和 x 旧 u doc');
  });
});
