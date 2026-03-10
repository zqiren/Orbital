// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { api } from '../config';
import { useAgent } from '../hooks/useAgent';

export interface PendingCredential {
  tool_call_id: string;
  name: string;
  domain: string;
  fields: string[];
  reason: string;
  resolved?: boolean;
}

interface CredentialCardProps {
  credential: PendingCredential;
  projectId: string;
  onResolve?: (toolCallId: string) => void;
}

export default function CredentialCard({
  credential,
  projectId,
  onResolve,
}: CredentialCardProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [resolved, setResolved] = useState(false);
  const { approveToolCall, denyToolCall } = useAgent();

  if (resolved || credential.resolved) {
    return (
      <div className="mb-3 px-4 py-2 rounded-lg bg-sidebar text-sm">
        <span className="text-success">{'\u2713'} Credentials provided</span>{' '}
        <span className="text-secondary">for {credential.domain}</span>
      </div>
    );
  }

  async function handleSubmit() {
    setSubmitting(true);
    try {
      const fieldValues: Record<string, string> = {};
      for (const f of credential.fields) {
        fieldValues[f] = values[f] || '';
      }
      await api('/api/v2/credentials', {
        method: 'POST',
        body: JSON.stringify({
          name: credential.name,
          domain: credential.domain,
          fields: fieldValues,
          project_id: projectId,
          tool_call_id: credential.tool_call_id,
        }),
      });
      // Resolve the intercepted request_credential tool call
      await approveToolCall(projectId, credential.tool_call_id);
      setResolved(true);
      onResolve?.(credential.tool_call_id);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDeny() {
    setSubmitting(true);
    try {
      await denyToolCall(projectId, credential.tool_call_id, 'User declined to provide credentials');
      setResolved(true);
      onResolve?.(credential.tool_call_id);
    } finally {
      setSubmitting(false);
    }
  }

  const allFilled = credential.fields.every((f) => (values[f] || '').trim().length > 0);

  return (
    <div className="mb-3 rounded-lg border border-border overflow-hidden">
      <div className="bg-accent/10 px-4 py-1.5 border-b border-border">
        <span className="text-accent font-semibold text-sm">CREDENTIALS NEEDED</span>
      </div>

      <div className="px-4 py-3 space-y-3">
        <p className="text-sm">
          <span className="font-semibold">{credential.domain}</span>
          {' \u2014 '}
          {credential.reason}
        </p>

        <div className="space-y-2">
          {credential.fields.map((field) => (
            <div key={field}>
              <label className="block text-xs text-secondary mb-1 capitalize">
                {field}
              </label>
              <input
                type={field.toLowerCase().includes('password') || field.toLowerCase().includes('secret') ? 'password' : 'text'}
                value={values[field] || ''}
                onChange={(e) => setValues((prev) => ({ ...prev, [field]: e.target.value }))}
                placeholder={field}
                autoComplete={field.toLowerCase().includes('password') ? 'current-password' : field.toLowerCase().includes('user') || field.toLowerCase().includes('email') || field.toLowerCase().includes('account') ? 'username' : 'off'}
                className="w-full text-sm px-3 py-1.5 rounded-lg border border-border bg-sidebar focus:outline-none focus:border-accent max-md:min-h-[44px]"
              />
            </div>
          ))}
        </div>

        <p className="text-xs text-secondary">
          Credentials are stored securely and never sent through chat.
        </p>

        <div className="flex flex-col md:flex-row gap-2 md:justify-end">
          <button
            type="button"
            onClick={handleDeny}
            onTouchEnd={(e) => { e.preventDefault(); handleDeny(); }}
            disabled={submitting}
            className="px-4 py-1.5 text-sm rounded-lg border border-border text-secondary hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
          >
            Deny
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            onTouchEnd={(e) => { e.preventDefault(); handleSubmit(); }}
            disabled={submitting || !allFilled}
            className="px-4 py-1.5 text-sm rounded-lg bg-accent text-white hover:opacity-90 transition-opacity duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
          >
            {submitting ? 'Submitting...' : 'Provide Credentials'}
          </button>
        </div>
      </div>
    </div>
  );
}
