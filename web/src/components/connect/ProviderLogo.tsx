// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * A provider's mark, monochrome in currentColor so it follows the theme.
 * Sources and licences: web/src/assets/provider-logos/README.md. A provider
 * with no mark gets a monogram; Custom gets a server glyph; ChatGPT is the
 * OpenAI mark in a filled circle; TokenDance is the Watcha avatar.
 */
import { Server } from 'lucide-react';
import watchaLogo from '../../assets/watcha-logo.png';
import anthropic from '../../assets/provider-logos/anthropic.svg?raw';
import deepseek from '../../assets/provider-logos/deepseek.svg?raw';
import google from '../../assets/provider-logos/google.svg?raw';
import groq from '../../assets/provider-logos/groq.svg?raw';
import hunyuan from '../../assets/provider-logos/hunyuan.svg?raw';
import minimax from '../../assets/provider-logos/minimax.svg?raw';
import mistral from '../../assets/provider-logos/mistral.svg?raw';
import moonshot from '../../assets/provider-logos/moonshot.svg?raw';
import openai from '../../assets/provider-logos/openai.svg?raw';
import opencodeGo from '../../assets/provider-logos/opencode-go.svg?raw';
import opencodeZen from '../../assets/provider-logos/opencode-zen.svg?raw';
import openrouter from '../../assets/provider-logos/openrouter.svg?raw';
import qwen from '../../assets/provider-logos/qwen.svg?raw';
import together from '../../assets/provider-logos/together.svg?raw';
import xai from '../../assets/provider-logos/xai.svg?raw';
import zhipu from '../../assets/provider-logos/zhipu.svg?raw';
import { CHATGPT, CUSTOM } from './providerMeta';

const MARKS: Record<string, string> = {
  anthropic, deepseek, google, groq, hunyuan, minimax, mistral, moonshot, openai,
  'opencode-go': opencodeGo, 'opencode-zen': opencodeZen, openrouter, qwen, together, xai, zhipu,
};

/** Marks drawn edge to edge in their box (the LobeHub set): shown a little
 *  smaller so they sit at the same optical size as the padded opencode ones. */
const FULL_BLEED = new Set(['openai', 'deepseek', 'moonshot', 'minimax', 'google', 'xai']);

/** Inline SVG markup sized to the box. The files are ours (vetted on import). */
function Mark({ svg, size, inset = 1 }: { svg: string; size: number; inset?: number }) {
  return (
    <span
      aria-hidden="true"
      className="inline-flex shrink-0 items-center justify-center [&>svg]:h-[var(--mk)] [&>svg]:w-[var(--mk)]"
      style={{ width: size, height: size, ['--mk' as string]: `${Math.round(size * inset)}px` }}
      dangerouslySetInnerHTML={{ __html: svg.trim() }}
    />
  );
}

export default function ProviderLogo({
  id,
  name,
  size = 18,
}: {
  id: string;
  /** For the monogram fallback. */
  name?: string;
  size?: number;
}) {
  if (id === 'tokendance') {
    return (
      <img
        src={watchaLogo}
        alt=""
        aria-hidden="true"
        className="shrink-0 rounded-full"
        style={{ width: size, height: size }}
      />
    );
  }
  if (id === CHATGPT) {
    return (
      <span
        aria-hidden="true"
        className="inline-grid shrink-0 place-items-center rounded-full bg-primary text-card"
        style={{ width: size, height: size }}
      >
        <Mark svg={openai} size={Math.round(size * 0.62)} />
      </span>
    );
  }
  if (id === CUSTOM) {
    return <Server aria-hidden="true" className="shrink-0 text-secondary" style={{ width: size, height: size }} strokeWidth={1.6} />;
  }
  const svg = MARKS[id];
  if (svg) return <Mark svg={svg} size={size} inset={FULL_BLEED.has(id) ? 0.8 : 1} />;
  const letter = (name || id || '?').trim().charAt(0).toUpperCase();
  return (
    <span
      aria-hidden="true"
      className="inline-grid shrink-0 place-items-center rounded-[5px] bg-card-hover font-semibold leading-none text-secondary"
      style={{ width: size, height: size, fontSize: Math.round(size * 0.55) }}
    >
      {letter}
    </span>
  );
}
