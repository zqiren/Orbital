# Frontend Test Gap

The `web/` package has no test framework configured (no vitest, jest, playwright, or
testing-library in devDependencies). Adding one for a single test would be overengineering.

Frontend changes in T05 (cancelMessage rewiring in ChatView, App) are verified by:

(a) `cd web && npx tsc --noEmit` — TypeScript type check, must produce zero errors.
(b) The live daemon smoke test the coordinator runs at end-of-batch.
(c) Manual smoke testing in the DMG installer.

This gap is tracked. A proper frontend test suite should be added when the component
count and change rate justify the investment.
