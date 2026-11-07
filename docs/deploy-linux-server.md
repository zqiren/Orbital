# Running Orbital on a Linux server

There is no Linux desktop build, but the daemon and web UI run headless on a
server: `python -m agent_os.server` is the desktop app without its window and
tray. The Docker image in this repository adds everything the desktop normally
provides. It has been tested on arm64 (Oracle Linux 9 host) and contains nothing
architecture-specific.

## Before you expose it

**The daemon has no authentication of its own.** Anyone who can reach its port
can run commands as the agent. `docker-compose.yml` therefore publishes it on
loopback only. Put an authenticating proxy in front of it before making it
reachable from anywhere else, and have that proxy reject cross-origin requests,
because the API accepts any origin.

**Linux has no agent sandbox yet.** `GET /api/v2/platform/status` reports
`isolation_method: none`, so agents run as the same user as the daemon and can
read everything it can, stored API keys included. The container is the isolation
boundary. Don't mount your home folder, the Docker socket or other secrets into it.

## Quick start

```bash
mkdir -p data/home && sudo chown 10001:10001 data/home
docker compose up -d --build
```

Open <http://127.0.0.1:8000> on the server, or through an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 -L 6080:127.0.0.1:6080 user@server
```

Everything persists in `data/home`: settings, projects and the logins of
Claude Code, Codex and Gemini CLI, which are preinstalled. Create project folders
under `/home/orbital`. Logs are in `data/home/.orbital/logs`.

## What the container does

| Concern | How it is handled |
| --- | --- |
| Privileges | Every process runs as uid 10001 with no capabilities and `no-new-privileges`. `NET_ADMIN` and the other added capabilities are used only by the entrypoint, before it drops them. |
| Network | The entrypoint rejects outbound traffic to private, CGNAT and link-local ranges from the container's own network namespace, so agents can't reach the host, cloud metadata or a VPN. The container exits if the rules can't be applied. Set `ORBITAL_BLOCKED_NETS` to replace the list. |
| Credentials | With no OS keychain available, keys are stored by `keyrings.alt` in an unencrypted `0600` file under `data/home`. |
| Desktop browser | Xvfb and noVNC provide a virtual desktop at <http://127.0.0.1:6080/vnc.html>. |

## Signing in

Browser sign-in, connectors and CLI logins open a browser on the machine running
the daemon, and their OAuth callbacks go to that machine's `127.0.0.1`. Here that
is the container, so **finish these flows in the noVNC desktop**. A link opened on
your own computer can't complete the callback.

CLI logins also work from a terminal:

```bash
docker exec -it -u 10001 orbital claude auth login
```

```bash
docker exec -it -u 10001 orbital codex login --device-auth
```

## Remote access with Tailscale

[Tailscale Serve](https://tailscale.com/kb/1312/serve) gives you HTTPS inside your
tailnet and tells the proxy who is connecting. Keep it tailnet-only: don't use
Funnel. Run a proxy such as nginx in front of the daemon and forward `/vnc/` to
port 6080. It should allow only your login and same-origin requests:

```nginx
map $http_tailscale_user_login $identity_ok { default 0; "you@example.com" 1; }
map $http_origin $origin_ok { default 0; "" 1; "https://server.tailnet.ts.net:8443" 1; }
```

```bash
sudo tailscale serve --bg --https=8443 http://127.0.0.1:18765
```

Set `AGENT_OS_PUBLIC_HOST=server.tailnet.ts.net` so Settings shows that address
for phone access instead of the container's bridge IP.

## Differences from the desktop app

- Reveal in Finder / File Explorer is unavailable.
- The cloud relay is opt-in through `AGENT_OS_RELAY_URL`. Without it there are no
  push notifications; phones use the web UI directly.
- The `search` browser action may return no results, because search engines often
  challenge datacenter IP addresses. The agent falls back to a page snapshot.
