// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect } from 'react';
import { Key, Trash2, RotateCcw, Loader2, Plus, X } from 'lucide-react';
import { api } from '../config';

interface Credential {
  name: string;
  domain: string;
  fields: string[];
  created: string;
  use_count: number;
  last_used: string | null;
}

interface StoreCredentialRequest {
  name: string;
  domain: string;
  fields: Record<string, string>;
  project_id?: string;
}

export default function CredentialStore({ projectId }: { projectId?: string }) {
  const [credentials, setCredentials] = useState<Credential[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formName, setFormName] = useState('');
  const [formDomain, setFormDomain] = useState('');
  const [formFields, setFormFields] = useState<Array<{ key: string; value: string }>>([
    { key: 'username', value: '' },
    { key: 'password', value: '' },
  ]);
  const [formError, setFormError] = useState('');

  useEffect(() => {
    fetchCredentials();
  }, []);

  async function fetchCredentials() {
    try {
      const data = await api<Credential[]>('/api/v2/credentials');
      setCredentials(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(name: string) {
    setDeleting(name);
    try {
      await api(`/api/v2/credentials/${encodeURIComponent(name)}`, { method: 'DELETE' });
      setCredentials(prev => prev.filter(c => c.name !== name));
    } finally {
      setDeleting(null);
    }
  }

  async function handleRevoke(name: string) {
    await api(`/api/v2/credentials/${encodeURIComponent(name)}/revoke`, { method: 'POST' });
  }

  async function handleStore() {
    setFormError('');
    if (!formName.trim() || !formDomain.trim()) {
      setFormError('Name and domain are required.');
      return;
    }
    const fields: Record<string, string> = {};
    for (const f of formFields) {
      if (f.key.trim() && f.value.trim()) {
        fields[f.key.trim()] = f.value.trim();
      }
    }
    if (Object.keys(fields).length === 0) {
      setFormError('At least one field with a value is required.');
      return;
    }
    setSaving(true);
    try {
      const body: StoreCredentialRequest = {
        name: formName.trim(),
        domain: formDomain.trim(),
        fields,
      };
      if (projectId) body.project_id = projectId;
      await api('/api/v2/credentials', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setShowForm(false);
      setFormName('');
      setFormDomain('');
      setFormFields([{ key: 'username', value: '' }, { key: 'password', value: '' }]);
      await fetchCredentials();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : 'Failed to store credential');
    } finally {
      setSaving(false);
    }
  }

  function formatAge(isoDate: string): string {
    const diff = Date.now() - new Date(isoDate).getTime();
    const minutes = Math.floor(diff / 60_000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days}d ago`;
    return new Date(isoDate).toLocaleDateString();
  }

  const addButton = (
    <button
      type="button"
      onClick={() => setShowForm(true)}
      className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-all duration-150 mt-2"
    >
      <Plus className="w-3.5 h-3.5" />
      Add credential
    </button>
  );

  const formUI = showForm ? (
    <div className="border border-border rounded-lg px-4 py-3 bg-sidebar/50 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-primary">New Credential</span>
        <button onClick={() => { setShowForm(false); setFormError(''); }} className="p-1 text-secondary hover:text-primary">
          <X className="w-4 h-4" />
        </button>
      </div>
      {formError && <p className="text-xs text-error">{formError}</p>}
      <input
        type="text"
        placeholder="Name (e.g. GitHub)"
        value={formName}
        onChange={e => setFormName(e.target.value)}
        className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent"
      />
      <input
        type="text"
        placeholder="Domain (e.g. github.com)"
        value={formDomain}
        onChange={e => setFormDomain(e.target.value)}
        className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent"
      />
      {formFields.map((f, i) => (
        <div key={i} className="flex gap-2">
          <input
            type="text"
            placeholder="Field name"
            value={f.key}
            onChange={e => {
              const next = [...formFields];
              next[i] = { ...next[i], key: e.target.value };
              setFormFields(next);
            }}
            className="w-1/3 text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent"
          />
          <input
            type="password"
            placeholder="Value"
            value={f.value}
            onChange={e => {
              const next = [...formFields];
              next[i] = { ...next[i], value: e.target.value };
              setFormFields(next);
            }}
            className="flex-1 text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent"
          />
        </div>
      ))}
      <button
        type="button"
        onClick={() => setFormFields(prev => [...prev, { key: '', value: '' }])}
        className="text-xs text-secondary hover:text-primary transition-colors"
      >
        + Add field
      </button>
      <div className="flex gap-2 pt-1">
        <button
          type="button"
          onClick={handleStore}
          disabled={saving}
          className="bg-accent text-white text-sm font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          type="button"
          onClick={() => { setShowForm(false); setFormError(''); }}
          className="text-sm text-secondary border border-border rounded-lg px-4 py-2 hover:bg-sidebar transition-all duration-150"
        >
          Cancel
        </button>
      </div>
    </div>
  ) : null;

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-secondary py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading credentials...
      </div>
    );
  }

  if (credentials.length === 0) {
    return (
      <div className="space-y-3">
        <div className="text-sm text-secondary/70 py-2">
          No saved credentials. When an agent needs to log into a website, it will request credentials through a secure modal.
        </div>
        {formUI || addButton}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {credentials.map(cred => (
        <div
          key={cred.name}
          className="border border-border rounded-lg px-4 py-3 bg-sidebar/50"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <Key className="w-4 h-4 text-secondary/70 shrink-0 mt-0.5" />
              <div className="min-w-0">
                <div className="text-sm font-medium text-primary">{cred.name}</div>
                <div className="text-xs text-secondary mt-0.5">{cred.domain}</div>
              </div>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={() => handleRevoke(cred.name)}
                title="Revoke browser session"
                className="p-1.5 text-secondary hover:text-primary rounded-md hover:bg-primary/5 transition-all duration-150"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => handleDelete(cred.name)}
                disabled={deleting === cred.name}
                title="Delete credential"
                className="p-1.5 text-secondary hover:text-error rounded-md hover:bg-error/5 transition-all duration-150 disabled:opacity-50"
              >
                {deleting === cred.name
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <Trash2 className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3 mt-2 text-xs text-secondary/70">
            <span>{cred.fields.join(', ')}</span>
            <span className="text-secondary/30">|</span>
            {cred.use_count > 0 ? (
              <span>Used {cred.use_count} time{cred.use_count !== 1 ? 's' : ''}</span>
            ) : (
              <span>Never used</span>
            )}
            {cred.last_used && (
              <>
                <span className="text-secondary/30">|</span>
                <span>Last: {formatAge(cred.last_used)}</span>
              </>
            )}
          </div>
        </div>
      ))}
      {formUI || addButton}
    </div>
  );
}
