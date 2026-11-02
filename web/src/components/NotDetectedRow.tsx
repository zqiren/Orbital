// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import MessageAvatar from './MessageAvatar';
import { useT } from '../i18n/useT';

interface Props {
  slug: string;
  name: string;
  /** The launcher the user pasted earlier, when its file has since gone. */
  overridePath?: string | null;
  overrideMissing?: boolean;
  /** Refetch the list; after Save the row becomes the installed card. */
  onChanged?: () => void | Promise<void>;
  /** Spec 127 D3: shown under an Orbital-installed agent's card as the
   *  fallback. Replaces the "copy this into {name}" line, which would
   *  imply the agent itself can answer — an ACP server like dsh cannot. */
  fallbackHint?: string;
  /** The vendor app that ships this agent (DSH Desktop): the row leads with
   *  its download link, because installing the app is all it takes. */
  desktopApp?: { label: string; download_url?: string | null } | null;
}

interface LocateTestResult {
  ok: boolean;
  binary?: string;
  path_prepend?: string[];
  env_names?: string[];
  version_output?: string;
  error?: string;
}

/** The locate card: the row for a bring-your-own sub-agent whose CLI the
 *  daemon could not find.
 *
 *  Detection misses because the daemon's PATH is not the user's shell PATH,
 *  and the user does not know the path either. The agent does: it runs in
 *  the shell that works. So the row copies a prompt, the user pastes it into
 *  the agent, pastes the answer back, Test runs the agent's own version
 *  check with that path (and the PATH entries it named), and Save stores it.
 *  Save needs a Test on the exact text in the box, the same rule as the
 *  credential cards. A bare path pastes just as well as the YAML.
 *
 *  `overrideMissing`: the file the user chose earlier is gone (the agent
 *  updated itself, say). Same row, first line names the missing file. */
export default function NotDetectedRow({
  slug, name, overridePath, overrideMissing, onChanged, fallbackHint, desktopApp,
}: Props) {
  const t = useT();
  const [answer, setAnswer] = useState('');
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState<string | null>(null);
  const [promptText, setPromptText] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testedAnswer, setTestedAnswer] = useState<string | null>(null);
  const [result, setResult] = useState<LocateTestResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (copiedTimer.current) clearTimeout(copiedTimer.current);
  }, []);

  const fetchPrompt = useCallback(async (): Promise<string> => {
    if (promptText) return promptText;
    const data = await api<{ prompt: string }>(
      `/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/locate-prompt`,
    );
    setPromptText(data.prompt);
    return data.prompt;
  }, [promptText, slug]);

  const handleCopy = async () => {
    setCopyError(null);
    try {
      const text = await fetchPrompt();
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
      copiedTimer.current = setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      // No clipboard (an insecure context, a denied permission): show the
      // prompt so it can be selected by hand.
      setCopyError(e instanceof Error ? e.message : String(e));
    }
  };

  const trimmed = answer.trim();
  const tested = testedAnswer !== null && testedAnswer === trimmed && result?.ok === true;

  const handleTest = async () => {
    if (!trimmed || testing) return;
    setTesting(true);
    setSaveError(null);
    try {
      const data = await api<LocateTestResult>(
        `/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/locate/test`,
        { method: 'POST', body: JSON.stringify({ answer: trimmed }) },
      );
      setResult(data);
      setTestedAnswer(trimmed);
    } catch (e) {
      setResult({ ok: false, error: e instanceof Error ? e.message : String(e) });
      setTestedAnswer(trimmed);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!tested || saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      await api(
        `/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/locate`,
        { method: 'PUT', body: JSON.stringify({ answer: trimmed }) },
      );
      await onChanged?.();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const versionOf = (output: string | undefined): string => {
    const m = /(\d+\.\d+(?:\.\d+)?)/.exec(output ?? '');
    return m ? m[1] : (output ?? '').split('\n')[0];
  };

  return (
    <div
      data-testid={`sub-agent-not-detected-${slug}`}
      className="bg-sidebar/30 border border-border rounded-lg px-4 py-3"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <MessageAvatar variant="agent" agentHandle={slug} />
        <span className="text-base font-medium text-primary">{name}</span>
        <code className="text-xs text-secondary/80 font-mono">{slug}</code>
        <span className="px-2 py-0.5 rounded-full border text-[11px] bg-sidebar text-secondary border-muted">
          {t('subAgentCard.notDetected')}
        </span>
      </div>

      {overrideMissing && overridePath && (
        <p
          data-testid={`sub-agent-locate-gone-${slug}`}
          className="text-xs text-warning-text mt-1.5 break-all"
        >
          {t('subAgentCard.locate.gone', { path: overridePath })}
        </p>
      )}

      {desktopApp?.download_url && (
        <div
          data-testid={`sub-agent-desktop-install-${slug}`}
          className="flex items-center gap-2 flex-wrap mt-2"
        >
          <a
            href={desktopApp.download_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150"
          >
            {t('subAgentCard.desktop.install', { app: desktopApp.label })}
          </a>
          <span className="text-xs text-secondary">{t('subAgentCard.desktop.installHint')}</span>
        </div>
      )}

      <div className="flex items-center gap-2 flex-wrap mt-1.5">
        <p className="text-xs text-secondary">
          {fallbackHint ?? t('subAgentCard.locate.hint', { name })}
        </p>
        <button
          type="button"
          onClick={handleCopy}
          data-testid={`sub-agent-locate-copy-${slug}`}
          className="text-xs border border-border rounded px-2 py-0.5 text-secondary hover:text-primary transition-all duration-150"
        >
          {copied ? t('subAgentCard.locate.copied') : t('subAgentCard.locate.copy')}
        </button>
      </div>

      {copyError && promptText && (
        <pre
          data-testid={`sub-agent-locate-prompt-${slug}`}
          className="text-[11px] text-secondary bg-sidebar border border-border rounded p-2 mt-2 whitespace-pre-wrap break-words select-all"
        >
          {promptText}
        </pre>
      )}

      <textarea
        value={answer}
        onChange={e => setAnswer(e.target.value)}
        placeholder={t('subAgentCard.locate.placeholder')}
        rows={3}
        spellCheck={false}
        aria-label={t('subAgentCard.locate.placeholder')}
        data-testid={`sub-agent-locate-answer-${slug}`}
        className="w-full mt-2 text-xs font-mono bg-sidebar border border-border rounded px-2 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent"
      />

      <div className="flex items-center gap-2 flex-wrap mt-2">
        <button
          type="button"
          onClick={handleTest}
          disabled={!trimmed || testing}
          data-testid={`sub-agent-locate-test-${slug}`}
          className="text-xs border border-border rounded px-3 py-1.5 text-secondary hover:text-primary transition-all duration-150 disabled:opacity-50"
        >
          {testing ? t('subAgentCard.locate.testing') : t('subAgentCard.locate.test')}
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={!tested || saving}
          data-testid={`sub-agent-locate-save-${slug}`}
          className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
        >
          {saving ? t('subAgentCard.locate.saving') : t('subAgentCard.locate.save')}
        </button>
        {result && testedAnswer === trimmed && (
          <span
            data-testid={`sub-agent-locate-result-${slug}`}
            className={`text-xs break-all ${result.ok ? 'text-success-text' : 'text-error-text'}`}
          >
            {result.ok
              ? (result.version_output
                  ? t('subAgentCard.locate.found', {
                      name,
                      version: versionOf(result.version_output),
                      path: result.binary ?? '',
                    })
                  // Spec 127 D2: an agent with no check command passes on
                  // existence alone; say that rather than print a blank.
                  : t('subAgentCard.locate.foundNoVersion', {
                      name,
                      path: result.binary ?? '',
                    }))
              : (result.error ?? '')}
          </span>
        )}
        {saveError && (
          <span className="text-xs text-error-text break-all">{saveError}</span>
        )}
      </div>
    </div>
  );
}
