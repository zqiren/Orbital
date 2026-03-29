# Learning Capture

Log corrections, errors, and discoveries to orbital/LESSONS.md so future sessions avoid repeating mistakes.

## When to Use

Activate this skill whenever one of three triggers fires:

1. **User correction** -- the user says "no, do X instead" or "that's wrong."
   Log what you did, what the user wanted, and the corrected approach.

2. **Tool failure recovery** -- a tool call fails and you find a working
   alternative. Log the failing call, the error, and the fix.

3. **Non-obvious discovery** -- you learn something surprising about the
   project (e.g., a config file that overrides defaults, an undocumented
   API quirk). Log the finding and where you found it.

## Approach

### Read before act

Before starting any task type you have attempted in a previous session,
read `orbital/LESSONS.md` with the `read` tool and scan for entries
whose category or keywords match the current task. Apply any relevant
lessons before your first action.

### Entry format

Append one block per lesson. Use this exact structure:

```
### YYYY-MM-DD — <category>
**What happened:** <1-2 sentences>
**Do instead:** <1-2 sentences>
**Keywords:** <comma-separated tags for future search>
```

Categories (pick one): `correction`, `error-recovery`, `discovery`.

### Writing the entry

- Use the `edit` tool to append to `orbital/LESSONS.md`.
  If the file does not exist, create it with the `write` tool.
- Keep each entry under 100 words. Precision beats length.
- Include enough context that a future agent session can act on the
  lesson without re-reading chat history.

## Anti-patterns

- **Do NOT log trivial observations.** "The project uses Python 3.12"
  is not a lesson. Only log things that caused or would cause a mistake.
- **Do NOT log every tool failure.** Typos in a filename are not worth
  capturing. Only log failures where the recovery path was non-obvious.
- **Do NOT duplicate entries.** Read the file first; if a matching lesson
  already exists, skip or update it rather than appending a duplicate.
- **Do NOT include full stack traces.** Summarize the error in one line.
