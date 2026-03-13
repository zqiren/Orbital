# Efficient Execution

Reduce token waste and task completion time by choosing the simplest tool for each job.

## When to Use

Apply these rules to every task. This is a standing discipline, not a
situational skill. Review the hierarchy below before each tool call.

## Approach

### Tool selection hierarchy (simplest first)

Pick the first option that works. Do not escalate without a reason.

| Task                  | Preferred tool                                | Avoid                              |
|-----------------------|-----------------------------------------------|------------------------------------|
| Create / overwrite file | `write` tool                                | Shell `echo` / `cat`, Python scripts |
| Edit existing file    | `edit` tool (find-and-replace)                | Rewriting the whole file with `write` |
| Create directories    | `shell` with `mkdir -p`                       | Python `os.makedirs`               |
| Quick fact lookup     | `browser(action="search", query="...")`       | Fetching full pages first          |
| Read a known URL      | `browser(action="fetch", url="...")`          | Full browser navigation            |
| Data processing       | `shell` with standard unix tools (`jq`, `awk`, `sort`, `cut`) | Writing a Python script  |
| Complex logic         | Python via `shell` only when unix tools are insufficient | —             |

### Token-saving rules

1. **Do not explain your reasoning** unless the user asks "why."
2. **Do not echo back** the user's request before acting.
3. **Do not read files you just wrote.** You already know the contents.
4. **Batch related operations** into a single tool call when possible
   (e.g., one `shell` call with `&&`-chained commands).
5. **Omit confirmation messages** like "Sure, I can do that." Just act.

### One-shot execution

If a task can be completed in one tool call, use one tool call. Common
examples:

- Creating a config file: one `write` call, not three shell commands.
- Renaming a file: one `shell` call with `mv`, not read + write + delete.
- Searching for a pattern: one `browser(action="search")` call, not
  navigating to Google and typing into the search box.

## Anti-patterns

- **Do NOT write a Python script to create a CSV.** Use the `write` tool
  with the CSV content directly.
- **Do NOT use `shell` to write files** (`echo "..." > file`). Use the
  `write` tool.
- **Do NOT fetch a full web page** when a search query would answer the
  question in fewer tokens.
- **Do NOT read a file, copy its contents, then write a new file** just
  to rename it. Use `shell` with `mv`.
- **Do NOT narrate each step** ("First I'll read the file, then I'll...")
  before doing it. Just do it.
