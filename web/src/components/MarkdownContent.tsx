// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ReactNode } from 'react';
import { isValidElement, useMemo } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, PanelRightOpen } from 'lucide-react';
import { detectWorkspacePath } from '../utils/pathDetection';
import { useT } from '../i18n/useT';
import CopyButton from './CopyButton';

interface MarkdownContentProps {
  content: string;
  /**
   * Absolute project workspace. Together with `onOpenPath`, enables clickable
   * workspace-relative paths (spec 002): markdown links to previewable
   * artifacts render as cards, source-file links + inline-code paths render as
   * chips, both opening the FilePreviewDrawer. When either is omitted (e.g. the
   * Files-tab FilePreview rendering a markdown file), no linkification happens.
   */
  workspace?: string;
  onOpenPath?: (path: string) => void;
}

/** http(s):// or protocol-relative `//` — anything that would leave the app. */
const EXTERNAL_HREF_RE = /^(?:https?:)?\/\//i;

/**
 * Render a plain anchor, externalizing http(s)/protocol-relative links so they
 * open in the system browser instead of navigating the desktop shell's single
 * webview frame (no back button — see agent_os/desktop/main.py's origin guard
 * for the defense-in-depth backstop on anything that slips past this).
 * In-page `#` anchors, relative hrefs, `mailto:`, and other schemes pass
 * through unchanged.
 */
function renderAnchor(href: string | undefined, children: ReactNode) {
  if (href && EXTERNAL_HREF_RE.test(href)) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  }
  return <a href={href}>{children}</a>;
}

/** Inline clickable chip for a workspace path (source files + inline-code spans). */
function PathChip({
  path,
  label,
  onOpen,
}: {
  path: string;
  label: ReactNode;
  onOpen: (p: string) => void;
}) {
  const t = useT();
  return (
    <button
      type="button"
      onClick={() => onOpen(path)}
      title={t('chat.path.openAria', { path })}
      aria-label={t('chat.path.openAria', { path })}
      className="font-mono text-[0.85em] text-accent bg-sidebar rounded px-[0.35em] py-[0.15em] hover:underline cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
    >
      {label}
    </button>
  );
}

/** Block card for a previewable artifact link (.html, images, .csv, .md, .json). */
function PathCard({
  path,
  label,
  onOpen,
}: {
  path: string;
  label: ReactNode;
  onOpen: (p: string) => void;
}) {
  const t = useT();
  const filename = path.split('/').pop() || path;
  return (
    <button
      type="button"
      onClick={() => onOpen(path)}
      aria-label={t('chat.path.openAria', { path })}
      className="my-2 flex w-full items-center gap-3 rounded-lg border border-border bg-sidebar px-3 py-2.5 text-left transition-colors hover:bg-card-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
    >
      <FileText size={18} className="shrink-0 text-secondary" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-medium text-primary">{label}</span>
        <span className="block truncate font-mono text-[11px] text-secondary">{filename}</span>
      </span>
      <span className="shrink-0 inline-flex items-center gap-1 text-[11px] font-medium text-accent">
        <PanelRightOpen size={13} />
        {t('chat.path.open')}
      </span>
    </button>
  );
}

/**
 * Flatten a react-markdown child tree back to its source text.
 *
 * The `<pre>` renderer receives the already-built `<code>` ELEMENT, not the raw
 * fence body, so the text has to be walked back out of it. Highlighters and the
 * inline-path renderer can nest elements arbitrarily deep, hence the recursion
 * rather than a `String(children)`, which would yield "[object Object]".
 */
function extractText(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (isValidElement(node)) {
    return extractText((node.props as { children?: ReactNode }).children);
  }
  return '';
}

/**
 * A fenced code block with its own copy button (BACKLOG spec 068).
 *
 * The button floats over the block's top-right corner and is hover/focus
 * revealed, so a page of code blocks does not become a page of buttons. It sits
 * on a wrapper rather than inside the `<pre>` because `<pre>` is
 * `overflow-x: auto` (index.css) — a child positioned there would scroll away
 * with wide code instead of staying pinned to the visible corner.
 */
function CodeBlock({ children }: { children?: ReactNode }) {
  const t = useT();
  const text = extractText(children);
  return (
    <div className="group relative">
      <CopyButton
        text={text}
        size={13}
        ariaLabel={t('chat.code.copyAria')}
        data-testid="code-block-copy"
        className="absolute right-1.5 top-1.5 z-10 border border-border bg-card/90 p-1 opacity-0 shadow-[0_1px_2px_rgb(0_0_0/0.04)] backdrop-blur-sm transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
      />
      <pre>{children}</pre>
    </div>
  );
}

export default function MarkdownContent({ content, workspace, onOpenPath }: MarkdownContentProps) {
  // Kind-aware path renderers are installed ONLY when a workspace + open
  // handler are supplied (i.e. in chat) — without them, react-markdown falls
  // back to the original bare behavior for paths, so reused surfaces
  // (Files-tab preview) are untouched. The external-link anchor renderer,
  // however, is installed in BOTH modes: external links must open outside the
  // app regardless of whether workspace linkification is active.
  const components = useMemo<Components>(() => {
    if (!workspace || !onOpenPath) {
      return {
        a({ href, children }) {
          return renderAnchor(href, children);
        },
        // Per-code-block copy is installed in BOTH modes, unlike the path
        // renderers above: copying a fence is useful wherever markdown renders
        // (chat and the Files-tab preview alike) and depends on no chat-only
        // prop.
        pre({ children }) {
          return <CodeBlock>{children}</CodeBlock>;
        },
      };
    }
    const open = onOpenPath;
    const ws = workspace;
    return {
      a({ href, children }) {
        const detected = href ? detectWorkspacePath(href, ws) : null;
        if (!detected) {
          // Non-workspace link → plain anchor, externalized if external.
          return renderAnchor(href, children);
        }
        // The markdown link text is the label (always present); the card also
        // shows the filename as a subtitle, so an empty label degrades cleanly.
        return detected.kind === 'card' ? (
          <PathCard path={detected.relativePath} label={children} onOpen={open} />
        ) : (
          <PathChip path={detected.relativePath} label={children} onOpen={open} />
        );
      },
      code({ className, children }) {
        const text = String(children ?? '');
        // Only INLINE code spans are linkified (spec §0.3): fenced blocks carry
        // a `language-*` class and/or newlines — never treat those as a path.
        const isFenced = !!className && /\blanguage-/.test(className);
        const isInline = !isFenced && !text.includes('\n');
        const detected = isInline ? detectWorkspacePath(text, ws) : null;
        if (!detected) {
          return <code className={className}>{children}</code>;
        }
        // Inline code always renders as a chip (never a card mid-sentence).
        return <PathChip path={detected.relativePath} label={children} onOpen={open} />;
      },
      pre({ children }) {
        return <CodeBlock>{children}</CodeBlock>;
      },
    };
  }, [workspace, onOpenPath]);

  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
