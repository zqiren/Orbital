// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One line of markdown, inline only: an ask's question (the agent writes
 * `**bold**`, `` `code` `` and links into it) drawn inside a title, a card
 * heading or a chat line. Bold, italic, strikethrough, inline code, links
 * and line breaks render; paragraphs are unwrapped (several are joined with
 * a space); headings, lists, quotes and code blocks keep their text but lose
 * the block; images are dropped. Links follow MarkdownContent's policy.
 *
 * Where only a string fits (a `title=`, an `aria-label`), use `plainInline`.
 */

import { memo } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { renderAnchor } from './MarkdownContent';

const ALLOWED = ['p', 'strong', 'em', 'del', 'code', 'a', 'br'];

const COMPONENTS: Components = {
  p: ({ children }) => <>{children}</>,
  a: ({ href, children }) => renderAnchor(href, children),
};

/** Inside a control (the Workbench card's title is a button) a link would
 *  be a control in a control: its text stays, the link does not. */
const COMPONENTS_NO_LINKS: Components = {
  p: COMPONENTS.p,
  a: ({ children }) => <>{children}</>,
};

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  children?: HastNode[];
}

/** Images go; a block left empty goes; the newlines between blocks become
 *  one space, none at the ends. Runs before the unwrap, so it sees every
 *  block level (lists, quotes). */
function rehypeInlineGaps() {
  const isGap = (c: HastNode) => c.type === 'text' && /^\s+$/.test(c.value ?? '');
  const walk = (node: HastNode) => {
    if (!node.children) return;
    node.children.forEach(walk);
    node.children = node.children.filter(
      (c) => c.tagName !== 'img'
        && !(c.type === 'element' && c.tagName !== 'br' && (c.children ?? []).every(isGap)),
    );
    if (node.type === 'root' || (node.tagName && !ALLOWED.includes(node.tagName))) {
      const kids = node.children.map((c) => (isGap(c) ? { ...c, value: ' ' } : c));
      while (kids.length && isGap(kids[0])) kids.shift();
      while (kids.length && isGap(kids[kids.length - 1])) kids.pop();
      node.children = kids;
    }
  };
  return (tree: HastNode) => walk(tree);
}

const REHYPE = [rehypeInlineGaps];

interface InlineMarkdownProps {
  text: string;
  className?: string;
  /** False inside a button or link: links render as their text. */
  links?: boolean;
}

export default memo(function InlineMarkdown({ text, className, links = true }: InlineMarkdownProps) {
  return (
    <span className={className ? `inline-md ${className}` : 'inline-md'}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={REHYPE}
        components={links ? COMPONENTS : COMPONENTS_NO_LINKS}
        allowedElements={ALLOWED}
        unwrapDisallowed
      >
        {text}
      </ReactMarkdown>
    </span>
  );
});

/** The same line with its marks taken out, for places that take only a
 *  string: `**`/`__`, backticks, `~~`, and `[text](url)` → `text`. */
// eslint-disable-next-line react-refresh/only-export-components -- the string twin of the component
export function plainInline(text: string): string {
  return text
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\*\*|__|~~|`/g, '');
}
