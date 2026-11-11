#!/bin/bash
# One fresh-install run in a throwaway Tart macOS VM.
#
#   set -a; source <file with ORBITAL_TEST_OPENCODE_KEY>; set +a
#   bash tests/fresh_install/tart/run.sh --tag v0.21.1                 # release DMG
#   bash tests/fresh_install/tart/run.sh --run-id 42825776292          # CI artifact
#   bash tests/fresh_install/tart/run.sh --tag v0.21.1 --mode upgrade  # v0.21.0 -> v0.21.1
#
# Options:
#   --tag <vX.Y.Z> | --run-id <id>   which installer (release asset or CI artifact Orbital-macOS-dmg)
#   --mode fresh|upgrade              default fresh
#   --old-tag <vX.Y.Z>                upgrade from (default: the release published before --tag,
#                                     or the latest release for --run-id)
#   --out <dir>                       evidence root (default ./fresh-install-evidence)
#   --keep                            leave the clone running on failure (debugging)
#   --scenario-args "<args>"          extra args for scenario.mjs
#
# Steps: download installer(s) on the host (gh) -> clone the golden VM -> boot
# -> copy installer + tests/fresh_install in -> run scenario.mjs inside (key on
# stdin, never on a command line) -> scrub inside -> copy evidence out ->
# scrub-check again on the host -> stop + delete the clone.
HERE=$(cd "$(dirname "$0")" && pwd)
FI_DIR=$(cd "$HERE/.." && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

: "${FRESH_REPO:=zqiren/Orbital}"
TAG="" RUN_ID="" MODE=fresh OLD_TAG="" OUT="$PWD/fresh-install-evidence" KEEP=0 EXTRA=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag) TAG=$2; shift 2 ;;
    --run-id) RUN_ID=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --old-tag) OLD_TAG=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --scenario-args) EXTRA=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TAG$RUN_ID" ]] || { echo "need --tag or --run-id" >&2; exit 2; }
[[ -n "${ORBITAL_TEST_OPENCODE_KEY:-}" ]] || { echo "ORBITAL_TEST_OPENCODE_KEY is not set" >&2; exit 2; }
tart list --quiet | grep -qx "$FRESH_GOLDEN" || { echo "no golden VM $FRESH_GOLDEN: run prepare-golden.sh" >&2; exit 2; }

STAMP=$(date +%Y%m%d-%H%M%S)
LABEL="${TAG:-run$RUN_ID}-$MODE-$STAMP"
WORK="$FRESH_STATE_DIR/work/$LABEL"
EVID="$OUT/$LABEL"
mkdir -p "$WORK" "$EVID" "$FRESH_STATE_DIR/logs"
VM="orbital-fresh-$STAMP"

# ---- installers (host side; downloaded and cached, never installed here)
CACHE="$FRESH_STATE_DIR/installers"
fetch_release_dmg() {  # <tag> -> path of the cached DMG
  local dir="$CACHE/$1" f
  mkdir -p "$dir"
  if ! compgen -G "$dir/Orbital-*-macOS.dmg" >/dev/null; then
    log "downloading $1 macOS DMG"
    gh release download "$1" --repo "$FRESH_REPO" -p 'Orbital-*-macOS.dmg' -D "$dir" --clobber >&2
  fi
  f=("$dir"/Orbital-*-macOS.dmg)
  echo "${f[0]}"
}
if [[ -n "$TAG" ]]; then
  NEW_DMG=$(fetch_release_dmg "$TAG")
else
  dir="$CACHE/run-$RUN_ID"
  mkdir -p "$dir"
  if ! compgen -G "$dir/Orbital-*-macOS.dmg" >/dev/null; then
    log "downloading CI run $RUN_ID artifact Orbital-macOS-dmg"
    gh run download "$RUN_ID" --repo "$FRESH_REPO" -n Orbital-macOS-dmg -D "$dir"
  fi
  f=("$dir"/Orbital-*-macOS.dmg); NEW_DMG=${f[0]}
fi
SCEN_ARGS=(--installer "/Users/$FRESH_VM_USER/in/$(basename "$NEW_DMG")")
if [[ $MODE == upgrade ]]; then
  if [[ -z "$OLD_TAG" ]]; then
    OLD_TAG=$(gh release list --repo "$FRESH_REPO" --exclude-drafts --exclude-pre-releases --limit 50 \
      --json tagName -q '.[].tagName' | awk -v t="$TAG" 'found {print; exit} $0==t {found=1} t=="" {print; exit}')
  fi
  [[ -n "$OLD_TAG" ]] || { echo "could not find a release before ${TAG:-the CI build}" >&2; exit 2; }
  log "upgrade from $OLD_TAG"
  OLD_DMG=$(fetch_release_dmg "$OLD_TAG")
  SCEN_ARGS+=(--mode upgrade --old-installer "/Users/$FRESH_VM_USER/in/old-$(basename "$OLD_DMG")")
fi

# ---- throwaway clone
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  local rc=$?
  if [[ $KEEP == 1 && $rc != 0 ]]; then
    log "--keep: leaving $VM running (tart stop $VM && tart delete $VM when done)"
  else
    vm_stop_delete "$VM"
    log "deleted $VM"
  fi
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
log "cloning $FRESH_GOLDEN -> $VM"
tart clone "$FRESH_GOLDEN" "$VM"
IP=$(vm_boot "$VM" "$FRESH_STATE_DIR/logs/$VM.log")
log "$VM at $IP"
wait_ssh "$IP"

vm_ssh "$IP" "mkdir -p ~/in ~/fresh_install"
vm_scp_to "$IP" "$NEW_DMG" "in/$(basename "$NEW_DMG")"   # relative = remote home
[[ $MODE == upgrade ]] && vm_scp_to "$IP" "$OLD_DMG" "in/old-$(basename "$OLD_DMG")"
(cd "$FI_DIR" && tar --exclude node_modules -cf - .) | vm_ssh "$IP" "tar -xf - -C ~/fresh_install"

log "running the scenario inside $VM (this takes a while)"
set +e
printf '%s\n' "$ORBITAL_TEST_OPENCODE_KEY" | vm_ssh "$IP" "bash -lc '
  IFS= read -r K; export ORBITAL_TEST_OPENCODE_KEY=\"\$K\" AGENT_OS_TELEMETRY_DISABLED=1; unset K
  cd ~/fresh_install && npm ci --no-audit --no-fund >/tmp/npm-ci.log 2>&1 || { cat /tmp/npm-ci.log; exit 3; }
  node scenario.mjs ${SCEN_ARGS[*]} --evidence ~/evidence --require-key $EXTRA
'" 2>&1 | tee "$EVID/console.log"
RC=${PIPESTATUS[1]}
set -e
log "scenario exit code $RC"

# ---- evidence out (already scrubbed inside), then scrub-check on the host
vm_ssh "$IP" "cd ~ && tar -czf /tmp/evidence.tgz evidence" || true
vm_scp_from "$IP" /tmp/evidence.tgz "$WORK/evidence.tgz" || true
if [[ -f "$WORK/evidence.tgz" ]]; then
  tar -xzf "$WORK/evidence.tgz" -C "$EVID" --strip-components 1
fi
if ! node "$FI_DIR/scrub.mjs" "$EVID" --require-key; then
  log "KEY FOUND in evidence after scrub; deleting the copied evidence"
  rm -rf "$EVID"
  exit 4
fi
log "evidence: $EVID"
exit "$RC"
