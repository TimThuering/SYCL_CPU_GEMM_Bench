#!/usr/bin/env bash
# Builds (and tests) flows of this machine into $GB_ROOT/build/<machine>/<flow>.
# Run on a compute node (native ISA detection), interactively or through build.sbatch.
#
#   bash src/scripts/server/build.sh [flow ...]        (default: all flows of the machine profile)
#
# Environment:
#   GB_TESTS=0          build without tests (default 1: -DBENCH_BUILD_TESTS=ON + ctest)
#   GB_CLEAN=1          delete the build directory first
#   GB_TEST_THREADS=n   threads for ctest (default 8)
# Log per flow: $GB_ROOT/logs/build_<machine>_<flow>.log; summary table at the end.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

flows=("$@"); [ ${#flows[@]} -gt 0 ] || read -r -a flows <<< "$GB_FLOWS"
for f in "${flows[@]}"; do flow_check "$f"; done
mkdir -p "$GB_LOGS" "$GB_BUILD"
[ -f "$GB_SRC/SOURCE_VERSION" ] && log "sources: $(tr '\n' ' ' < "$GB_SRC/SOURCE_VERSION")"

declare -A RESULT
for f in "${flows[@]}"; do
  logf=$GB_LOGS/build_${GB_MACHINE}_$f.log
  dir=$GB_BUILD/$f
  log "$f -> $dir (log: $logf)"
  (
    set -e
    flow_load "$f"
    flow_cmake_args "$f"
    [ "${GB_CLEAN:-0}" = 1 ] && rm -rf "$dir"
    echo "== modules: $(module -t list 2>&1 | tr '\n' ' ')"
    echo "== cmake ${FLOW_CMAKE_ARGS[*]}"
    cmake -S "$GB_SRC" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
          "-DBENCH_BUILD_TESTS=$([ "${GB_TESTS:-1}" = 1 ] && echo ON || echo OFF)" \
          "${FLOW_CMAKE_ARGS[@]}" || exit 10
    cmake --build "$dir" -j "$(nproc)" || exit 11
    if [ "${GB_TESTS:-1}" = 1 ]; then
      cache=$(mktemp -d); trap 'rm -rf "$cache"' EXIT
      flow_run_env "$f" "${GB_TEST_THREADS:-8}" "$cache"
      (cd "$dir" && ctest --output-on-failure) || exit 12
    fi
  ) > "$logf" 2>&1
  case $? in
    0)  RESULT[$f]=OK ;;
    10) RESULT[$f]="FAIL (configure)" ;;
    11) RESULT[$f]="FAIL (build)" ;;
    12) RESULT[$f]="FAIL (tests)" ;;
    *)  RESULT[$f]="FAIL (see log)" ;;
  esac
  log "$f: ${RESULT[$f]}"
done

echo
echo "build summary ($GB_MACHINE):"
rc=0
for f in "${flows[@]}"; do
  printf '  %-22s %s\n' "$f" "${RESULT[$f]}"
  [ "${RESULT[$f]}" = OK ] || rc=1
done
exit $rc
