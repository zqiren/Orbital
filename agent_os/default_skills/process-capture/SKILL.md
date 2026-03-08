# Process Capture

Capture successful multi-step workflows as reusable skills so future sessions can replay them.

## When to Use

Activate after completing any task that meets ALL of these criteria:

1. It took **3 or more distinct steps** (tool calls).
2. It is **likely to recur** -- either the user said so, or the task type
   is common (deploy, report generation, data pipeline, etc.).
3. No existing skill already covers it (check `skills/` first).

After such a task, ask the user: "This took several steps. Want me to
save it as a reusable skill?"  If the user agrees (or you are in
hands-off autonomy mode), capture the skill.

## Approach

### Captured skill format

Create `skills/{task-name}/SKILL.md` with this structure:

```markdown
# Task Name

One-line description of what this skill does.

## When to Use
<1-3 trigger conditions>

## Steps
1. <action> — <tool and key arguments>
2. <action> — <tool and key arguments>
...

## Inputs
- <variable>: <description and default if any>

## Outputs
- <artifact>: <where it goes>

## Anti-patterns
- <what NOT to do>
```

Use RFC 2119 keywords (MUST, SHOULD, MAY) for clarity on which steps
are required vs. optional.

### Trigger-task pairing

When the user asks you to create a trigger (via `create_trigger`, if available), you
MUST also create a companion skill that documents how the triggered
task SHOULD be executed. The trigger fires the task; the skill defines
the task. Always pair them.

### Updating existing skills

If a captured workflow changes, update the existing SKILL.md rather
than creating a new skill. Use the `edit` tool to modify only the
changed sections.

## Anti-patterns

- **Do NOT capture trivial tasks.** A single-step file creation is not
  worth a skill. The 3-step minimum exists for a reason.
- **Do NOT duplicate existing skills.** Read `skills/` before creating.
  If a similar skill exists, extend it instead.
- **Do NOT include hardcoded paths or values.** Use placeholder names
  (e.g., `{input_file}`) in the Steps section so the skill is reusable
  across different inputs.
- **Do NOT write overly detailed skills.** Keep each skill under 80
  lines. If the workflow is that complex, break it into sub-skills.
