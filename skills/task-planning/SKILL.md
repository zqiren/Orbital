# Task Planning

Match planning effort to task complexity. Check existing skills before inventing a new approach.

## When to Use

Apply before every non-trivial task. If you are about to make more than
one tool call, pause and classify the task first.

## Approach

### Complexity-scaled planning

| Complexity | Indicators                           | Planning required                  |
|------------|--------------------------------------|------------------------------------|
| Simple     | 1-2 steps, single file, obvious path | None. Just execute.                |
| Medium     | 3-5 steps, 2-3 files, some choices   | 1-2 sentence plan stating tools and order. |
| Complex    | 6+ steps, cross-file changes, dependencies | Numbered step list before first action. |

Write plans in your response text. Do NOT create plan files.

### Skill-check-first rule

Before starting any medium or complex task:

1. Scan the skills list (shown in the prompt) for a matching skill name.
2. If a match looks plausible, read the full `SKILL.md` with the `read`
   tool.
3. Follow the skill's steps rather than inventing your own approach.
4. If no skill matches, proceed with your own plan.

This takes seconds and can save minutes of wasted work.

### Simplicity bias

Before writing code, ask yourself: is there a simpler approach?

- About to write **>10 lines of code**? Stop. Check if a shell one-liner
  or existing tool would work.
- About to create **a new file** for intermediate processing? Stop. Can
  you pipe commands instead?
- About to install **a new dependency**? Stop. Can you use what is
  already available?

If a simpler approach exists, use it. If not, proceed — but note the
decision in `orbital/DECISIONS.md` so future sessions know why.

### Plan communication

- For simple tasks: no plan needed, just act.
- For medium tasks: state the plan in one line before acting.
  Example: "I'll use `edit` to update config.json, then `shell` to
  restart the service."
- For complex tasks: present a numbered list and wait for the user to
  confirm (unless in hands-off autonomy mode).

## Anti-patterns

- **Do NOT plan simple tasks.** Writing "Step 1: read the file" for a
  single read-and-respond is pure waste.
- **Do NOT skip planning complex tasks.** Jumping into a 10-step
  refactor without a plan leads to rework and wasted context.
- **Do NOT ignore existing skills.** If a skill covers your task, use
  it. Re-inventing the wheel wastes tokens and risks mistakes the skill
  already accounts for.
- **Do NOT over-plan.** A 20-line plan for a 5-step task is itself
  waste. Keep plans proportional to complexity.
