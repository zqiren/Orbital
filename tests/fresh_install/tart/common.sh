# shellcheck shell=bash
# Orbital fresh-install test — shared helpers for the Tart (local macOS VM) host.
# Sourced by prepare-golden.sh and run.sh. Never installs or runs Orbital on
# the host: everything Orbital happens inside a throwaway clone.

set -euo pipefail

: "${FRESH_BASE_IMAGE:=ghcr.io/cirruslabs/macos-sequoia-base:latest}"
: "${FRESH_GOLDEN:=orbital-fresh-golden}"
: "${FRESH_STATE_DIR:=$HOME/.cache/orbital-fresh-install}"  # rig SSH key + VM logs; never in the repo
: "${FRESH_VM_USER:=admin}"
: "${FRESH_VM_PASSWORD:=admin}"                        # Cirrus Labs image default

SSH_KEY="$FRESH_STATE_DIR/id_ed25519"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=10 -o ServerAliveInterval=30 -i "$SSH_KEY")

log() { printf '%s [tart] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

ensure_rig_key() {
  mkdir -p "$FRESH_STATE_DIR"
  chmod 700 "$FRESH_STATE_DIR"
  [[ -f "$SSH_KEY" ]] || ssh-keygen -q -t ed25519 -N '' -C orbital-fresh-install -f "$SSH_KEY"
}

# vm_boot <name> <logfile>: start headless, wait for an IP, echo it.
vm_boot() {
  local name=$1 logfile=$2 ip=""
  log "booting $name (headless)"
  nohup tart run --no-graphics --no-audio --no-clipboard "$name" >"$logfile" 2>&1 &
  echo $! >"$logfile.pid"
  for _ in $(seq 1 120); do
    ip=$(tart ip "$name" 2>/dev/null || true)
    [[ -n "$ip" ]] && break
    sleep 2
  done
  [[ -n "$ip" ]] || { log "no IP for $name"; return 1; }
  echo "$ip"
}

# wait_ssh <ip> [password-mode]: wait until sshd answers with the rig key
# (or, in password mode, until the port is open).
wait_ssh() {
  local ip=$1 mode=${2:-key}
  for _ in $(seq 1 90); do
    if [[ $mode == password ]]; then
      nc -z -G 3 "$ip" 22 >/dev/null 2>&1 && return 0
    else
      ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$FRESH_VM_USER@$ip" true >/dev/null 2>&1 && return 0
    fi
    sleep 2
  done
  log "ssh never came up on $ip"
  return 1
}

# install_rig_key <ip>: authorize the rig key using the image's default password.
install_rig_key() {
  local ip=$1
  FRESH_PW="$FRESH_VM_PASSWORD" expect >/dev/null <<EOF
set timeout 60
spawn ssh-copy-id -f -i "$SSH_KEY.pub" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $FRESH_VM_USER@$ip
expect {
  -re "assword:" { send "\$env(FRESH_PW)\r"; exp_continue }
  eof
}
EOF
}

# shellcheck disable=SC2029  # remote commands are built on purpose
vm_ssh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$FRESH_VM_USER@$ip" "$@"; }
vm_scp_to() { local ip=$1 src=$2 dst=$3; scp -q -r "${SSH_OPTS[@]}" "$src" "$FRESH_VM_USER@$ip:$dst"; }
vm_scp_from() { local ip=$1 src=$2 dst=$3; scp -q -r "${SSH_OPTS[@]}" "$FRESH_VM_USER@$ip:$src" "$dst"; }

# vm_wait_exit <logfile>: wait for the `tart run` started by vm_boot to end.
vm_wait_exit() {
  local pid
  pid=$(cat "$1.pid" 2>/dev/null || true)
  [[ -n "$pid" ]] || return 0
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || return 0; sleep 2; done
  return 1
}

vm_stop_delete() {
  local name=$1
  tart stop "$name" >/dev/null 2>&1 || true
  tart delete "$name" >/dev/null 2>&1 || true
}
