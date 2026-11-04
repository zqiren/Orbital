---
name: Coupled steps = one commit
description: Both sides of one fix land in one commit; rewrite old-contract tests
type: feedback
originSessionId: 00000000-0000-4000-8000-000000000001
---
When a fix touches both the producer and the consumer, commit them together.

**Why:** a half-landed contract breaks bisect.
