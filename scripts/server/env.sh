# Common environment of the server scripts (sourced).
#
# Directory layout on the server (GB_ROOT = parent of the synced src/):
#   $GB_ROOT/src/       synced sources (scripts/local/sync_to_server.sh)
#   $GB_ROOT/build/<machine>/<flow>/   (third-party sources fetched by CMake FetchContent per build)
#   $GB_ROOT/results/<campaign>/
#   $GB_ROOT/logs/

GB_SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GB_SRC=$(cd "$GB_SERVER_DIR/../.." && pwd)
GB_ROOT=$(cd "$GB_SRC/.." && pwd)
GB_BUILD_ROOT=$GB_ROOT/build
GB_RESULTS=$GB_ROOT/results
GB_LOGS=$GB_ROOT/logs

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Machine from the host name (GB_MACHINE overrides)
if [ -z "${GB_MACHINE:-}" ]; then
  case "$(hostname -s)" in
    mc-peak-*)  GB_MACHINE=peak ;;
    mc-blade-*) GB_MACHINE=blade ;;
    *) die "unknown host $(hostname -s): run on a compute node, or set GB_MACHINE" ;;
  esac
fi
[ -f "$GB_SERVER_DIR/machines/$GB_MACHINE.sh" ] || die "no machine profile machines/$GB_MACHINE.sh"
GB_BUILD=$GB_BUILD_ROOT/$GB_MACHINE

# Environment modules
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash

# -march string for the AOT flows from the ISA string in /proc/cpuinfo, keeping only the extensions
# the clang in PATH accepts. gb_native_march vec|novec (novec: without V and all zv* extensions).
gb_native_march() {
  local kind=$1 isa base e s
  isa=$(grep -m1 '^isa' /proc/cpuinfo | sed 's/^isa[[:space:]]*:[[:space:]]*//')
  [ -n "$isa" ] || die "no ISA string in /proc/cpuinfo"
  base=${isa%%_*}
  local exts=(); IFS=_ read -r -a exts <<< "${isa#"$base"}"; exts=("${exts[@]:1}")
  if [ "$kind" = novec ]; then
    base="rv64$(echo "${base#rv64}" | tr -d v)"
    local keep=(); for e in "${exts[@]}"; do case $e in zv*) ;; *) keep+=("$e") ;; esac; done
    exts=("${keep[@]}")
  fi
  march_ok() { clang -march="$1" -c -x c /dev/null -o /dev/null 2>/dev/null; }
  s=$base
  for e in "${exts[@]}"; do march_ok "${s}_$e" && s="${s}_$e"; done
  echo "-march=$s"
}

source "$GB_SERVER_DIR/machines/$GB_MACHINE.sh"
source "$GB_SERVER_DIR/flows.sh"
