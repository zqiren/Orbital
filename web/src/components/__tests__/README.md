# Frontend Tests

Vitest + Testing Library are configured. Run:

```
cd web && npm run test:run    # one-shot
cd web && npm test            # watch mode
```

Tests live alongside the modules they cover: `web/src/**/*.test.{ts,tsx}`.

Setup file: `web/test-setup.ts` registers `@testing-library/jest-dom` matchers.

The TypeScript check (`cd web && npx tsc --noEmit`) and production build (`npm
run build`) still verify what tests can't. Manual smoke testing remains a
release gate.
