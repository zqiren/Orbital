// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.8 — FilePreview's third mode: a version's text rendered the way
 * the file renders today, its changes tinted in the colour of the agent that
 * made them (Docs per-editor colours). Inserted lines get a soft tint,
 * deleted lines the same tint struck through. The daemon computes every line
 * and its editor; this only paints.
 *
 * Markdown keeps its rendering: the view's lines (the version's plus the
 * deleted ones, in order) are joined into one document, and a remark plugin
 * wraps the text of each changed source line in a tint span. Everything else
 * is the source view with one span per changed line.
 */
import { useEffect, useMemo, useRef } from 'react';
import type { HistoryView } from '../../types';
import MarkdownContent from '../MarkdownContent';
import { editorColor, PALETTE_VARS, tintStyle } from './historyModel';
import { remarkHistoryTint, type LineTint, type MdNode } from './historyTint';

interface VersionViewProps {
  view: HistoryView;
  markdown: boolean;
  /** Extra remark plugins the normal view runs (FilePreview hides HTML comments). */
  remarkPlugins?: Array<() => (tree: MdNode) => void>;
  /** Changes when the selection changes: the view scrolls to its first change. */
  scrollKey: string;
}

export default function VersionView({ view, markdown, remarkPlugins, scrollKey }: VersionViewProps) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const tints = useMemo<LineTint[]>(
    () =>
      view.lines.map((line) =>
        line.op === 'same' ? null : { op: line.op, color: editorColor(line.editor === null ? null : view.editors[line.editor]) },
      ),
    [view],
  );
  const source = useMemo(() => view.lines.map((l) => l.text).join('\n'), [view]);
  const plugins = useMemo(
    () => [...(remarkPlugins ?? []), remarkHistoryTint(tints)],
    [remarkPlugins, tints],
  );

  // Docs scrolls to the first change when a version is selected.
  useEffect(() => {
    const host = scrollerRef.current;
    if (!host) return;
    const first = host.querySelector<HTMLElement>('[data-hist]');
    if (!first) {
      host.scrollTop = 0;
      return;
    }
    const y =
      first.getBoundingClientRect().top - host.getBoundingClientRect().top + host.scrollTop;
    host.scrollTop = Math.max(0, y - 120);
  }, [scrollKey, view]);

  return (
    <div
      ref={scrollerRef}
      data-testid="history-version-view"
      className={`doc-sheet flex-1 min-h-0 overflow-auto bg-sidebar ${PALETTE_VARS}`}
    >
      {markdown ? (
        <div className="p-4">
          <MarkdownContent content={source} remarkPlugins={plugins} />
        </div>
      ) : (
        <pre className="font-mono text-sm text-primary p-4 whitespace-pre-wrap break-words">
          {view.lines.map((line, i) => {
            const tint = tints[i];
            return (
              <span key={i}>
                {tint ? (
                  <span data-hist={tint.op} style={tintStyle(tint.op, tint.color)}>
                    {line.text || ' '}
                  </span>
                ) : (
                  line.text
                )}
                {'\n'}
              </span>
            );
          })}
        </pre>
      )}
    </div>
  );
}
