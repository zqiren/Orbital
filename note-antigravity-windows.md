# Antigravity worker (spec 193): Windows live check

Branch `feat/antigravity-worker`. The Mac session built and smoke-tested everything
it could, but its network exits in Hong Kong, where Google refuses Antigravity. So no
real reply has been seen yet. This check needs an **eligible network** (US, Japan or
Singapore exit, or no VPN outside the excluded regions). Do it on the owner's Windows
laptop against a dev daemon or a CI build of this branch.

## What changed (so you know what to look at)

- New manifest `agent_os/agents/manifests/antigravity.yaml` (`agy acp`, ACP transport).
  The old `gemini-cli` entry is gone.
- Login is the CLI's own `agy acp --login`. agy opens the browser itself. When the
  CLI asks for the code, the agent card shows one field ("Paste the code" / 粘贴授权码).
  Submit writes the code to the CLI's stdin. Orbital never opens the URL itself.
  The "Open the sign-in page" link is only a fallback for when no browser opened.
- Sign-in probe: `agy models` (exit 0 means signed in, exit 1 "Please sign in" means
  signed out).
- A turn whose whole output is agy's refusal ("… valid license … (#3501)" or
  "not eligible for Antigravity / not currently available in your location") ends as
  an error that reads "Antigravity isn't available in your region".
- Antigravity has no HTTP MCP, so a chat pinned to it cannot dispatch to other agents.
  That is expected in v1.

## Steps

1. **Install, then detect.** Orbital has no Install button for a bring-your-own CLI.
   The card shows "Not detected" plus the locate card, and the per-OS command reaches
   only the API's `setup_actions`. So in PowerShell run
   `irm https://antigravity.google/cli/install.ps1 | iex`. Then press Refresh in
   Settings → Sub-agents. The Antigravity card should read Installed with a version.
   - Check `GET /api/v2/settings/sub-agents`: before the install, the antigravity
     entry's `setup_actions` should carry the `irm …` command. That proves the per-OS
     install_command resolves on Windows.
   - Record the path `agy.exe` landed at. The installer script says
     `%LOCALAPPDATA%\agy\bin\agy.exe`, and that is the manifest's only `auto_detect`
     entry. If it landed somewhere else, or Orbital does not find it, write down the
     real path.
   - Record whether `agy install` added `%LOCALAPPDATA%\agy\bin` to the user PATH.
2. **Log in through the paste field.** Press Login on the card. A browser tab should
   open on Google sign-in, opened by agy, not by Orbital. Sign in. When Google shows
   the authorization code, copy it, paste it into the card's field and press Submit.
   - Pass: the card reads Logged in, with no "couldn't confirm" note.
   - Use the tab agy opened. A code from a second tab fails with "Invalid code verifier".
   - Record the code prompt and any other output the card's login log shows
     (`GET /api/v2/settings/sub-agents/login-jobs`). The code itself must show as
     `[redacted]` there and nowhere in `daemon.log`.
3. **Pin a chat and get PONG.** New session, switch the pin to Antigravity and send
   "Reply with the single word PONG".
   - Pass: an Antigravity bubble that says PONG.
   - Record whether a console window flashed up when `agy.exe` spawned. The spawn uses
     CREATE_NO_WINDOW; this is unverified for agy.
4. **File edit with an approval card.** Use a project in check-in mode, and set
   Antigravity's permission mode to `ask` in its card. Ask it to "create hello.txt
   containing hi".
   - Pass: an approval card appears. Approve it and the file exists with that content.
   - Then deny one and confirm the file is not written.
   - Record what the card shows (tool name, arguments) and whether agy asked at all in
     its `default` mode (spec unknown 7.2).
5. **Resume after a daemon restart.** Restart the daemon, then send "What word did you
   reply with earlier?" in the same session.
   - Pass: it answers PONG.
   - Pass: the session JSONL `sub_agent_thread` rows keep the same `session_id` with a
     new `proc_pid`.
6. **Switch the model in the picker.** In the pin menu, set Antigravity's model to
   another id the account offers (`agy models` lists them; the ACP option uses the
   bare id, e.g. `claude-sonnet-4-6`). Send one more message.
   - Pass: the next `sub_agent_thread` row reports the new model.
   - Pass: the process command line is `agy.exe --model <id> acp`.

## What to record

- The Windows install path, and whether Orbital found it without the locate card.
- Whether `agy.exe` spawns with no console window, for the ACP worker, for
  `agy models` during the status sweep, and for `agy acp --login`.
- How a real reply renders. Is the text in one bubble? Do tool calls show as capsule
  rows? Does the token counter move (ACP usage updates)? This is spec unknown 7.7.
- Whether permission requests arrive in `default` mode, and whether `auto` mode is
  silent.
- Any refusal or error text seen, verbatim, especially anything that slipped through
  as a normal reply.
- How long the status sweep takes with agy installed. `agy models` makes a network
  call, about 4 s on the Mac.
- Keychain or credential behaviour on Windows. Does the sign-in survive a reboot?

STATUS: PENDING
