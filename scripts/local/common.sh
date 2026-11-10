# Shared helpers of the local scripts (sourced).

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$LOCAL_DIR/../.." && pwd)

die() { echo "error: $*" >&2; exit 1; }

[ -f "$LOCAL_DIR/targets.local.sh" ] ||
  die "missing $LOCAL_DIR/targets.local.sh (copy targets.example.sh and adapt it)"
# shellcheck source=targets.example.sh
source "$LOCAL_DIR/targets.local.sh"

# resolve_target <target> -> MOUNT, SERVER_DIR (absolute, through the mount)
resolve_target() {
  local t=$1
  [ -n "${TARGET_MOUNT[$t]:-}" ] || die "unknown target '$t' (known: ${!TARGET_MOUNT[*]})"
  MOUNT=${TARGET_MOUNT[$t]}
  SERVER_DIR=$MOUNT/${TARGET_DIR[$t]}
  # An unmounted mount point is an empty local directory: writing there would go nowhere
  if ! mountpoint -q "$MOUNT"; then
    [ "${ALLOW_UNMOUNTED:-0}" = 1 ] || die "$MOUNT is not mounted (mount the server home first)"
  fi
}
