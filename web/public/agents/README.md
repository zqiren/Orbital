# Agent avatar assets

Vendor marks served at `/agents/<slug>.svg` and referenced from
`web/src/utils/agentIcons.ts` (`AgentIcon.src`). `MessageAvatar` prefers these
images and falls back to the brand-coloured monogram badge if one fails to load,
so a missing or broken file degrades rather than breaking the row.

## Source

All five files come from **[lobehub/lobe-icons](https://github.com/lobehub/lobe-icons)**,
`packages/static-svg/icons/`, fetched from `raw.githubusercontent.com` at the
`master` tip on 2026-08-11. That repository is **MIT licensed** (© 2023 LobeHub).

| File | Upstream file | Mark |
|---|---|---|
| `claude-code.svg` | `claude.svg` | Anthropic / Claude |
| `codex.svg` | `openai.svg` | OpenAI |
| `cursor.svg` | `cursor.svg` | Cursor |
| `gemini.svg` | `gemini.svg` | Google Gemini |
| `grok.svg` | `grok.svg` | xAI / Grok |
| `dsh.svg` | `deepseek.svg` | DeepSeek (Harness) — fetched 2026-08-14; upstream ships `currentColor`/`1em`, normalized here to the baked brand fill `#4D6BFE` + 24px box to match the older vendored files |

## Local modifications

Each file was edited in exactly two mechanical ways; the path geometry is
untouched.

1. The root `fill="currentColor"` was pinned to a literal colour. **`currentColor`
   does not inherit through an `<img>` tag** — an SVG loaded as an image is an
   independent document with no access to the host page's CSS, so the keyword
   would resolve against the UA's initial `color` rather than ours. Pinning makes
   the render deterministic. Colours used: Claude `#D97757`, Gemini `#4285F4`,
   and `#000000` for Codex/Cursor/Grok (each vendor's own mark is black).
2. `width`/`height` changed from `1em` to `24` to match the `viewBox`, giving the
   `<img>` a sane intrinsic size.

Every file was validated on import: starts with `<svg`, contains no `<script>`,
no `onload`/`javascript:` attributes, and is under 50 KB.

## Trademark note

These are third-party trademarks used nominatively to identify which vendor's
agent a chat row belongs to. They are not Orbital marks and imply no endorsement.
If a vendor's brand guidelines require removal, delete the file — the monogram
fallback in `agentIcons.ts` takes over automatically with no code change.

Orbital's own management agent does not use this directory; it reuses the app
icon at `/icon-192.png`.

## Upstream licence (reproduced per the MIT notice requirement)

This file ships inside the built SPA alongside the assets it covers, which is
what satisfies "included in all copies or substantial portions of the Software".

```
MIT License

Copyright (c) 2023 LobeHub

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
