#!/bin/bash
# Build the golden macOS VM for the fresh-install test, once (re-run to
# refresh the worker CLIs). The golden image gets Node, Codex, DeepSeek
# Harness (dsh) and Playwright WebKit — and NOTHING of Orbital. Every test
# run clones it and throws the clone away (run.sh).
#
#   bash tests/fresh_install/tart/prepare-golden.sh [--force]
#
# Env: FRESH_BASE_IMAGE (default ghcr.io/cirruslabs/macos-sequoia-base:latest),
#      FRESH_GOLDEN (default orbital-fresh-golden), FRESH_STATE_DIR,
#      CODEX_VERSION / DSH_VERSION / PLAYWRIGHT_VERSION (guest-provision.sh).
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

if tart list --quiet 2>/dev/null | grep -qx "$FRESH_GOLDEN"; then
  if [[ ${1:-} != --force ]]; then
    log "$FRESH_GOLDEN already exists (pass --force to rebuild)"
    exit 0
  fi
  vm_stop_delete "$FRESH_GOLDEN"
fi

ensure_rig_key
mkdir -p "$FRESH_STATE_DIR/logs"
log "cloning $FRESH_BASE_IMAGE -> $FRESH_GOLDEN (pulls the image if it is not cached)"
tart clone "$FRESH_BASE_IMAGE" "$FRESH_GOLDEN"
tart set "$FRESH_GOLDEN" --cpu 4 --memory 8192 || true

trap 'tart stop "$FRESH_GOLDEN" >/dev/null 2>&1 || true' EXIT
IP=$(vm_boot "$FRESH_GOLDEN" "$FRESH_STATE_DIR/logs/golden-run.log")
log "golden VM at $IP; waiting for sshd"
wait_ssh "$IP" password
install_rig_key "$IP"
wait_ssh "$IP"

vm_scp_to "$IP" "$HERE/guest-provision.sh" /tmp/guest-provision.sh
vm_ssh "$IP" "CODEX_VERSION=${CODEX_VERSION:-latest} DSH_VERSION=${DSH_VERSION:-0.2.2-rc.1} \
  PLAYWRIGHT_VERSION=${PLAYWRIGHT_VERSION:-1.60.0} bash -l /tmp/guest-provision.sh" \
  | tee "$FRESH_STATE_DIR/logs/golden-provision.log"
vm_ssh "$IP" 'rm -f /tmp/guest-provision.sh; sync; sudo shutdown -h now' || true

vm_wait_exit "$FRESH_STATE_DIR/logs/golden-run.log" || log "VM did not power off in time; stopping it"
trap - EXIT
tart stop "$FRESH_GOLDEN" >/dev/null 2>&1 || true
log "golden image $FRESH_GOLDEN ready"
