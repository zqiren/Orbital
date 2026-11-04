// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.8 — the remark plugin behind the tinted markdown view: the
 * text of every changed source line is wrapped in a tint span, so a version
 * renders exactly like the file does today plus the Docs-style colours.
 */
import { cssText, tintStyle } from './historyModel';

export interface MdNode {
  type: string;
  value?: string;
  children?: MdNode[];
  position?: { start: { line: number }; end: { line: number } };
  data?: Record<string, unknown>;
}

export type LineTint = { op: 'add' | 'del'; color: string } | null;

function tintSpan(tint: NonNullable<LineTint>, children: MdNode[]): MdNode {
  return {
    type: 'emphasis',
    data: { hName: 'span', hProperties: { style: cssText(tintStyle(tint.op, tint.color)), dataHist: tint.op } },
    children,
  };
}

/** Remark plugin: tint the text of every changed source line (1-based). */
export function remarkHistoryTint(tints: LineTint[]) {
  return () => (tree: MdNode) => {
    const at = (line: number) => tints[line - 1] ?? null;
    const walk = (node: MdNode) => {
      if (!node.children) return;
      const next: MdNode[] = [];
      for (const child of node.children) {
        const start = child.position?.start.line;
        if (child.type === 'text' && start !== undefined && child.value !== undefined) {
          const pieces = child.value.split('\n');
          pieces.forEach((piece, i) => {
            if (i > 0) next.push({ type: 'text', value: '\n' });
            if (piece === '') return;
            const tint = at(start + i);
            next.push(tint ? tintSpan(tint, [{ type: 'text', value: piece }]) : { type: 'text', value: piece });
          });
          continue;
        }
        if (child.type === 'inlineCode' && start !== undefined) {
          const tint = at(start);
          next.push(tint ? tintSpan(tint, [child]) : child);
          continue;
        }
        if (child.type === 'code' && child.position) {
          // A fenced block: tint it whole when any line inside changed;
          // struck only when every changed line is a deletion.
          const ops: Array<NonNullable<LineTint>> = [];
          for (let l = child.position.start.line; l <= child.position.end.line; l++) {
            const tint = at(l);
            if (tint) ops.push(tint);
          }
          if (ops.length > 0) {
            const op = ops.every((o) => o.op === 'del') ? 'del' : 'add';
            const color = (ops.find((o) => o.op === op) ?? ops[0]).color;
            next.push({ type: 'blockquote', data: { hName: 'div', hProperties: { style: cssText(tintStyle(op, color)), dataHist: op } }, children: [child] });
            continue;
          }
        }
        walk(child);
        next.push(child);
      }
      node.children = next;
    };
    walk(tree);
  };
}

