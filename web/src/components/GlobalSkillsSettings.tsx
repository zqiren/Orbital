// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Global Settings → Skills: one list of every skill Orbital offers in all
 * projects — Orbital's own global skills, then the ones already in the
 * user's home folders (~/.agents/skills, ~/.claude/skills, ~/.codex/skills),
 * in the daemon's precedence order. Each row shows where it comes from by a
 * mark (Orbital, Claude Code, Codex, or a folder for the shared ~/.agents),
 * its folder on hover, and an on/off switch. Only Orbital's own rows can be
 * deleted; Orbital never deletes from the home folders.
 *
 * Data (skills contract v3): GET /api/v2/skills (four groups), POST
 * /api/v2/skills (multipart add), DELETE /api/v2/skills/{dir_name}, PUT
 * /api/v2/skills/enabled {id, enabled}. Rows and the upload target are the
 * project section's (SkillList).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Folder } from 'lucide-react';
import { api, ApiError } from '../config';
import type { GlobalSkill, GlobalSkillGroup, GlobalSkillsResponse } from '../types';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import { SkillRow, SkillUpload } from './SkillList';
import { AgentLogo } from './history/AgentLogo';
import HintSign from './HintSign';

/** SKILL.md path → the folder that holds it (either separator). */
export function skillFolder(path: string): string {
  return path.replace(/[\\/]SKILL\.md$/i, '');
}

/** The switch id: the daemon's, or the same `<source>:<lowercased name>`
 *  rule for a row from a daemon that predates it. */
export function skillId(s: GlobalSkill): string {
  return s.id ?? `${s.source}:${s.name.toLowerCase()}`;
}

const AGENT_OF: Partial<Record<GlobalSkill['source'], string>> = {
  global: 'orbital',
  claude: 'claude-code',
  codex: 'codex',
};

/** The agents a skill belongs to, as logo slugs: its own folder's agent,
 *  then any agent folder that links the same skill (`linked_from`). A skill
 *  only in the shared ~/.agents folder has none. */
export function skillOwners(skill: GlobalSkill): string[] {
  const sources = [skill.source, ...(skill.linked_from ?? [])];
  const slugs = sources.map((s) => AGENT_OF[s]).filter((s): s is string => Boolean(s));
  return Array.from(new Set(slugs));
}

/** Where a skill comes from, as a mark: the owning agents' logos (Orbital
 *  for its own), or a folder glyph when no agent owns it (~/.agents). */
function SourceMark({ skill }: { skill: GlobalSkill }) {
  const owners = skillOwners(skill);
  if (owners.length === 0) {
    return <Folder aria-hidden="true" className="h-[13px] w-[13px] shrink-0 text-muted" strokeWidth={1.75} />;
  }
  return (
    <span className="inline-flex items-center gap-0.5">
      {owners.map((slug) => (
        <AgentLogo key={slug} slug={slug} />
      ))}
    </span>
  );
}

/** Upload error → copy, keyed off the status code (the contract's rule). */
function uploadErrorKey(status: number): StringKey | null {
  if (status === 409) return 'global.skills.error.exists';
  if (status === 413) return 'global.skills.error.tooLarge';
  if (status === 400) return 'global.skills.error.invalid';
  return null;
}

export default function GlobalSkillsSettings() {
  const t = useT();
  // null = first load in flight.
  const [groups, setGroups] = useState<GlobalSkillGroup[] | null>(null);
  const [error, setError] = useState('');
  const [added, setAdded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  /** Latest switch save per skill id, so a slow earlier reply cannot win. */
  const toggleSeq = useRef<Record<string, number>>({});

  const refresh = useCallback(async () => {
    try {
      const data = await api<GlobalSkillsResponse>('/api/v2/skills');
      setGroups(Array.isArray(data?.groups) ? data.groups : []);
    } catch {
      setGroups((prev) => prev ?? []);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!added) return;
    const id = window.setTimeout(() => setAdded(false), 2000);
    return () => window.clearTimeout(id);
  }, [added]);

  function setEnabled(id: string, enabled: boolean) {
    setGroups(
      (prev) =>
        prev?.map((g) => ({
          ...g,
          skills: g.skills.map((s) => (skillId(s) === id ? { ...s, enabled } : s)),
        })) ?? prev,
    );
  }

  async function handleToggle(skill: GlobalSkill, enabled: boolean) {
    const id = skillId(skill);
    const before = skill.enabled !== false;
    const seq = (toggleSeq.current[id] ?? 0) + 1;
    toggleSeq.current[id] = seq;
    setError('');
    setEnabled(id, enabled);
    try {
      await api('/api/v2/skills/enabled', {
        method: 'PUT',
        body: JSON.stringify({ id, enabled }),
      });
      // Which same-name rows win over which (hidden_by) is the daemon's call.
      if (toggleSeq.current[id] === seq) await refresh();
    } catch {
      if (toggleSeq.current[id] !== seq) return;
      setEnabled(id, before);
      setError(t('global.skills.error.toggle'));
    }
  }

  async function handleUpload(file: File) {
    setError('');
    setAdded(false);
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      await api('/api/v2/skills', { method: 'POST', body: form });
      setAdded(true);
      await refresh();
    } catch (e) {
      const key = e instanceof ApiError ? uploadErrorKey(e.status) : null;
      setError(
        key ? t(key) : e instanceof ApiError && e.detail ? e.detail : t('settings.skills.uploadError'),
      );
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function handleDelete(skill: GlobalSkill) {
    setError('');
    try {
      await api(`/api/v2/skills/${encodeURIComponent(skill.dir_name)}`, { method: 'DELETE' });
    } catch (e) {
      // 404 = already gone: the refresh below shows the truth.
      if (!(e instanceof ApiError && e.status === 404)) {
        setError(t('settings.skills.deleteError'));
      }
    }
    await refresh();
  }

  // One list in the daemon's group order (its precedence order); a home
  // folder that does not exist simply contributes nothing.
  const rows = (groups ?? []).flatMap((g) => g.skills);

  return (
    <div data-testid="global-skills">
      {error && (
        <p className="text-caption text-error mb-2" role="alert" data-testid="global-skills-error">
          {error}
        </p>
      )}
      {added && <p className="text-caption text-success mb-2">{t('settings.skills.added')}</p>}

      {groups !== null &&
        (rows.length > 0 ? (
          <div className="space-y-2 mb-3" data-testid="global-skills-list">
            {rows.map((s) => {
              const id = skillId(s);
              const hidden = Boolean(s.hidden_by);
              const editable = s.editable && s.source === 'global';
              return (
                <SkillRow
                  key={id}
                  testId={`skill-${s.source}-${s.dir_name}`}
                  name={s.name}
                  description={s.description}
                  title={skillFolder(s.path)}
                  leading={<SourceMark skill={s} />}
                  enabled={s.enabled !== false}
                  onToggle={(on) => void handleToggle(s, on)}
                  toggleLabel={t('global.skills.toggleAria', { name: s.name })}
                  muted={hidden}
                  badge={
                    hidden ? (
                      <HintSign
                        label={t('global.skills.whyAria')}
                        testId={`skill-${s.source}-${s.dir_name}-why`}
                      >
                        {t('global.skills.hidden')}
                      </HintSign>
                    ) : undefined
                  }
                  onDelete={editable ? () => handleDelete(s) : undefined}
                />
              );
            })}
          </div>
        ) : (
          <p className="text-caption text-secondary mb-3" data-testid="global-skills-empty">
            {t('global.skills.empty')}
          </p>
        ))}

      <SkillUpload
        ref={fileInputRef}
        uploading={uploading}
        onFile={handleUpload}
        testId="global-skills-upload"
      />
    </div>
  );
}
