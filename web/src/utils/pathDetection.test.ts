// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { detectWorkspacePath } from './pathDetection';

const WS = '/Users/you/repo';

describe('detectWorkspacePath — relative workspace paths', () => {
  it('treats a relative source path as a chip-eligible workspace path', () => {
    expect(detectWorkspacePath('src/main.py', WS)).toEqual({
      relativePath: 'src/main.py',
      kind: 'chip',
    });
  });

  it('treats a bare filename with an extension as a chip path', () => {
    expect(detectWorkspacePath('package.json', WS)).toEqual({
      relativePath: 'package.json',
      kind: 'card', // .json is a previewable artifact
    });
  });

  it('strips a leading ./ from a relative path', () => {
    expect(detectWorkspacePath('./src/app.ts', WS)).toEqual({
      relativePath: 'src/app.ts',
      kind: 'chip',
    });
  });
});

describe('detectWorkspacePath — card kinds (previewable artifacts)', () => {
  it('marks an .html markdown-link target as card-eligible', () => {
    expect(detectWorkspacePath('reports/q3.html', WS)).toEqual({
      relativePath: 'reports/q3.html',
      kind: 'card',
    });
  });

  it.each(['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'csv', 'md', 'json', 'htm'])(
    'marks .%s as a card',
    (ext) => {
      const result = detectWorkspacePath(`out/file.${ext}`, WS);
      expect(result?.kind).toBe('card');
    },
  );

  it.each(['py', 'ts', 'tsx', 'js', 'go', 'rs', 'yaml', 'toml', 'txt', 'sh'])(
    'marks .%s as a chip',
    (ext) => {
      const result = detectWorkspacePath(`src/file.${ext}`, WS);
      expect(result?.kind).toBe('chip');
    },
  );
});

describe('detectWorkspacePath — absolute paths', () => {
  it('strips the workspace prefix from an absolute path inside the workspace', () => {
    expect(detectWorkspacePath('/Users/you/repo/reports/q3.html', WS)).toEqual({
      relativePath: 'reports/q3.html',
      kind: 'card',
    });
  });

  it('does NOT link an absolute path outside the workspace', () => {
    expect(detectWorkspacePath('/etc/hosts', WS)).toBeNull();
  });

  it('does NOT link the workspace root itself (no file)', () => {
    expect(detectWorkspacePath('/Users/you/repo', WS)).toBeNull();
  });

  it('matches the workspace prefix case-insensitively (macOS)', () => {
    const result = detectWorkspacePath('/users/YOU/Repo/docs/notes.md', WS);
    expect(result).toEqual({ relativePath: 'docs/notes.md', kind: 'card' });
  });
});

describe('detectWorkspacePath — Windows separators', () => {
  it('normalizes a backslash relative path to forward slashes', () => {
    expect(detectWorkspacePath('src\\components\\App.tsx', WS)).toEqual({
      relativePath: 'src/components/App.tsx',
      kind: 'chip',
    });
  });

  it('strips a Windows absolute workspace prefix', () => {
    const result = detectWorkspacePath(
      'C:\\Users\\you\\repo\\reports\\q3.html',
      'C:\\Users\\you\\repo',
    );
    expect(result).toEqual({ relativePath: 'reports/q3.html', kind: 'card' });
  });

  it('does NOT link a Windows absolute path outside the workspace', () => {
    expect(
      detectWorkspacePath('C:\\Windows\\System32\\drivers\\etc\\hosts', 'C:\\Users\\you\\repo'),
    ).toBeNull();
  });
});

describe('detectWorkspacePath — non-paths and false-positive guards', () => {
  it('does NOT match a plain-prose phrase', () => {
    expect(detectWorkspacePath('see the main file', WS)).toBeNull();
  });

  it('does NOT match a shell command in inline code', () => {
    expect(detectWorkspacePath('npm install', WS)).toBeNull();
  });

  it('does NOT match an identifier with no extension', () => {
    expect(detectWorkspacePath('useState', WS)).toBeNull();
  });

  it('does NOT match a bare directory with no file extension', () => {
    expect(detectWorkspacePath('src/components', WS)).toBeNull();
  });

  it('does NOT match an http(s) URL', () => {
    expect(detectWorkspacePath('https://example.com/a.html', WS)).toBeNull();
  });

  it('does NOT match a mailto: link', () => {
    expect(detectWorkspacePath('mailto:dev@example.com', WS)).toBeNull();
  });

  it('does NOT match an in-page anchor', () => {
    expect(detectWorkspacePath('#section', WS)).toBeNull();
  });

  it('rejects a path-traversal escape', () => {
    expect(detectWorkspacePath('../secret.txt', WS)).toBeNull();
  });

  it('rejects an absolute path that escapes via the workspace prefix boundary', () => {
    // "/Users/you/repo-evil/x.md" shares a string prefix with the workspace
    // but is a sibling directory, not inside it.
    expect(detectWorkspacePath('/Users/you/repo-evil/x.md', WS)).toBeNull();
  });

  it('trims surrounding whitespace before matching', () => {
    expect(detectWorkspacePath('  docs/notes.md  ', WS)).toEqual({
      relativePath: 'docs/notes.md',
      kind: 'card',
    });
  });

  it('returns null for an empty string', () => {
    expect(detectWorkspacePath('', WS)).toBeNull();
  });

  it.each(['3.14', '0.6.8', '1.0', '3.11', '10.15.7'])(
    'does NOT linkify the dotted-number version/python string %s',
    (n) => {
      expect(detectWorkspacePath(n, WS)).toBeNull();
    },
  );

  it('still linkifies real files whose names contain digits and dots', () => {
    expect(detectWorkspacePath('q3.html', WS)).toEqual({
      relativePath: 'q3.html',
      kind: 'card',
    });
  });
});

describe('detectWorkspacePath — :line / :line:col suffixes (spec 076)', () => {
  it('strips a trailing :line from a relative path', () => {
    expect(detectWorkspacePath('docs/notes.md:7', WS)).toEqual({
      relativePath: 'docs/notes.md',
      kind: 'card',
    });
  });

  it('strips a trailing :line:col from a relative path', () => {
    expect(detectWorkspacePath('src/main.py:7:12', WS)).toEqual({
      relativePath: 'src/main.py',
      kind: 'chip',
    });
  });

  it('strips a trailing :line from an absolute in-workspace path', () => {
    expect(detectWorkspacePath('/Users/you/repo/agent_output/README.md:7', WS)).toEqual({
      relativePath: 'agent_output/README.md',
      kind: 'card',
    });
  });

  it('does NOT strip a non-numeric suffix like :7x', () => {
    expect(detectWorkspacePath('docs/notes.md:7x', WS)).toBeNull();
  });

  it('still rejects a dotted-number string once the :line is stripped', () => {
    // `1.2:3` strips to `1.2`, which the all-numeric guard must still catch.
    expect(detectWorkspacePath('1.2:3', WS)).toBeNull();
  });

  it('does NOT turn a host:port into a path', () => {
    expect(detectWorkspacePath('localhost:8000', WS)).toBeNull();
  });

  it('leaves suffix-free paths untouched', () => {
    expect(detectWorkspacePath('q3.html', WS)).toEqual({
      relativePath: 'q3.html',
      kind: 'card',
    });
    expect(detectWorkspacePath('src/v2.py', WS)).toEqual({
      relativePath: 'src/v2.py',
      kind: 'chip',
    });
    expect(detectWorkspacePath('data/2024.csv', WS)).toEqual({
      relativePath: 'data/2024.csv',
      kind: 'card',
    });
    expect(detectWorkspacePath('2024.csv', WS)).toEqual({
      relativePath: '2024.csv',
      kind: 'card',
    });
  });
});
