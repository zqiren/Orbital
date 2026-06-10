# i18n Maintainability — Is the CSV Approach Sustainable?

**Date:** 2026-06-10
**Context:** Evaluation requested while adding Simplified-Chinese support to the Orbital web UI, backed by `docs/i18n/ui-terms.zh-Hans.csv` (590 strings).

## Verdict

**The CSV is a fine *translator worksheet* but the wrong *runtime source of truth.*** Keep authoring/exchanging translations in the CSV if that's comfortable for translators, but compile it once into a typed module (`web/src/i18n/strings.ts`) that the app actually imports. That module — not the CSV — is what the build, the type-checker, and the running app depend on.

This is exactly the architecture now in place. The rest of this doc explains why, with the concrete problems this very migration hit.

## What actually went wrong with the CSV (observed, not hypothetical)

1. **Excel corrupts leading `+`/`=`/`-`/`@` into `#NAME?`.** Three rows arrived already destroyed — `app.newProject` (`+ New Project`), `credStore.form.addField` (`+ Add field`), `sessionSidebar.newSession` (`+ new session`) — because a spreadsheet read the leading `+` as a formula. The original English was **lost in the file**; it had to be recovered from the source code. A build cannot trust a source that silently eats data on save.
2. **Placeholder schemes don't survive a flat string.**
   - `activity.searchingFor` carried `Searching for "{pattern}"{in path}` — `{in path}` has a space, so it is *not* a valid `{word}` placeholder and would have rendered **literally** in the UI. It had to be split into `searchingFor` / `searchingForIn`, with the conditional chosen in code.
   - 12 `activityGroup.*` count labels had `{n}` in Chinese but not English (and 5 had it in neither), so the number would have vanished or mismatched. Fixed by making `{n}` explicit in both locales.
   - English plurals can't live in one cell: `BlockedBadge` needs "1 session" vs "5 sessions", which the CSV flattened to "session(s)". Fixed with `blocked.aria.one` / `.other`.
3. **Copy drift is invisible.** The sub-agent settings section was reworded in a recent commit, but the CSV still holds the *old* wording, so those strings silently fall back to English. A CSV has no way to tell you "this key no longer matches the code."
4. **No safety rails.** A raw CSV gives you: no type checking (a typo'd key is a runtime blank), no missing-key detection, no placeholder validation, a BOM at the front, and one giant file that is merge-conflict-hostile.

## Why the typed module fixes each of these

`web/src/i18n/strings.ts` (generated once, then committed and owned):

- **Compile-time key safety.** `t('app.loadng')` fails `tsc`. Autocomplete lists valid keys. Keys are grep-able and diffs are per-key.
- **No Excel round-trip in the hot path.** The CSV is parsed by `web/scripts/gen-i18n.mjs` (a real RFC-4180 parser, BOM-safe) — not by a spreadsheet — and the generator carries a `CORRECTIONS` safety net for the known Excel landmines.
- **A validator with teeth.** `web/scripts/check-i18n.mjs` errors on placeholder mismatches and missing English, and *warns* (never blocks) on missing Chinese — so the translation backlog is visible without gating merges.

## How this affects future feature work (the real question)

**Small, and crucially non-blocking**, *if* you follow two rules:

- **Ship English-first.** Every string resolves `zh → en → key`. A new feature with no Chinese still renders perfect English. **Never block a PR on translation.** Make `check-i18n` warn-only on missing `zh` (it already does).
- **Translate a surface when it stabilizes, not on every churn.** Partial coverage is a first-class state — a screen can be 100% English, 100% Chinese, or anywhere between, with zero breakage. The enemy isn't the framework; it's re-translating UI that's still moving.

Per-feature mechanics are cheap: write `t('your.key')` instead of a bare string, add one catalog entry (`en` required, `zh` optional). The recurring *cost* is translation labor, and the fallback decouples it from shipping.

## Patterns established here (reuse them)

- **Plurals:** two keys (`.one` / `.other`), chosen in code. No ICU dependency. (`BlockedBadge`)
- **Counts / word order:** bake `{n}` into the string so languages with different order place it correctly.
- **Non-React code** (utils, module-level helpers, class components) can't call the `useT()` hook. Thread an **optional translator param that defaults to English** (`(k, v) => translate('en', k, v)`), so callers/tests that omit it get byte-identical output. See `chatTransform.ts` (`ActivityTranslate`) and `ChatView.tsx` (`capsuleSummaryText`). Bind it to `locale` inside a `useMemo` (not the unstable `t`) to avoid re-render churn.
- **Dynamic/backend strings** (provider notes, file paths, model names, agent output) are **not** UI chrome — leave them untranslated.

## Recommendations

1. **Source of truth = `strings.ts`.** Treat the CSV as an import/export worksheet only.
2. **Never hand-edit the CSV in Excel.** Use a CSV-safe editor (VS Code, a text editor) or round-trip through `gen-i18n.mjs`. Excel mangles leading `+`/`=`/`-`/`@`.
3. **Wire `node web/scripts/check-i18n.mjs` into pre-commit / CI** as warn-on-missing-zh, error-on-structural.
4. **Adding a language** = one new column in the worksheet + one entry in `web/src/i18n/locales.ts` `LOCALES`. The dropdown is data-driven.
5. **When the catalog and code drift** (renamed/reworded copy), the string falls back to English — run `check-i18n` and re-sync the affected keys when you touch that surface.

## Tooling reference

- `web/scripts/gen-i18n.mjs` — CSV → `strings.ts` (run once; output committed).
- `web/scripts/check-i18n.mjs` — validate the catalog (warn on missing zh).
- `web/scripts/keys-for.mjs <File.tsx> …` — list the catalog rows for a component (used to wire/re-wire files).
