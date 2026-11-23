# Provider logos

Monochrome marks for the model-provider tiles and card rows
(`web/src/components/connect/ProviderLogo.tsx`). Each file is inlined at build
time (`?raw`) and drawn in `currentColor`, so it follows the theme: black on
light, light on dark. A provider without a file here gets a monogram mark.

Every file was normalized the same way: the root carries `fill="currentColor"`
(or `fill="none"` where the source was a stroke mark), any literal colour on a
path became `currentColor`, `<title>` was dropped, and the geometry is
untouched. On import each file was checked to contain only `path`/`g`-style
shapes: no `<script>`, no `on*=` attributes, no `href`.

## Sources

| File | Upstream | Licence |
|---|---|---|
| `openai.svg` | Orbital's own `web/public/agents/codex.svg` (LobeHub `openai.svg`) | MIT, LobeHub |
| `deepseek.svg` | Orbital's own `web/public/agents/dsh.svg` (LobeHub `deepseek.svg`) | MIT, LobeHub |
| `moonshot.svg` | Orbital's own `web/public/agents/kimi.svg` (LobeHub `kimi.svg`) | MIT, LobeHub |
| `minimax.svg` | Orbital's own `web/public/agents/minimax.svg` (LobeHub `minimax.svg`) | MIT, LobeHub |
| `google.svg` | Orbital's own `web/public/agents/gemini.svg` (LobeHub `gemini.svg`) | MIT, LobeHub |
| `xai.svg` | Orbital's own `web/public/agents/grok.svg` (LobeHub `grok.svg`) | MIT, LobeHub |
| `anthropic.svg` | opencode `packages/ui/src/components/provider-icons/sprite.svg`, symbol `anthropic` | MIT, opencode |
| `zhipu.svg` | opencode sprite, symbol `zhipuai` | MIT, opencode |
| `qwen.svg` | opencode sprite, symbol `alibaba` | MIT, opencode |
| `hunyuan.svg` | opencode sprite, symbol `tencent` | MIT, opencode |
| `mistral.svg` | opencode sprite, symbol `mistral` | MIT, opencode |
| `groq.svg` | opencode sprite, symbol `groq` | MIT, opencode |
| `together.svg` | opencode sprite, symbol `togetherai` | MIT, opencode |
| `openrouter.svg` | opencode sprite, symbol `openrouter` | MIT, opencode |
| `opencode-zen.svg` | opencode sprite, symbol `opencode` | MIT, opencode |
| `opencode-go.svg` | opencode sprite, symbol `opencode-go` | MIT, opencode |

TokenDance uses the existing `web/src/assets/watcha-logo.png`. ChatGPT is the
OpenAI mark inside a filled circle, drawn in code.

## Trademark note

These are third-party trademarks, used only to identify which provider a tile
or saved card belongs to. They imply no endorsement. Delete a file and the
monogram takes over with no code change beyond its import line.

## Upstream licences (reproduced per the MIT notice requirement)

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

```
MIT License

Copyright (c) 2025 opencode

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
