#!/usr/bin/env bash
# Copies campaign results from the server (through the sshfs mount) into the local analysis directory,
# sorted by the campaign tag (campaign name = <date>_<tag>_<name>):
#
#   S1 .. S4  -> $ANALYSIS_DIR/data/dev/S1_naive | S2_interchange | S3_regblock | S4_packed/<campaign>/
#   final     -> $ANALYSIS_DIR/data/final/<campaign>/
#   other     -> $ANALYSIS_DIR/data/dev/<tag>/<campaign>/     (e.g. smoke)
#
#   scripts/local/pull_results.sh <target> [--apply] [campaign ...]
#
#   (default)  dry run: lists what would be copied
#   --apply    copy
#   campaign   only these campaigns (default: all on the server)
#
# Never deletes anything (neither on the server nor locally); pulling a running campaign again later
# adds the new files.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

target=""; apply=0; only=()
for a in "$@"; do
  case $a in
    --apply) apply=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    -*) die "unknown option $a" ;;
    *) if [ -z "$target" ]; then target=$a; else only+=("$a"); fi ;;
  esac
done
[ -n "$target" ] || die "usage: $0 <target> [--apply] [campaign ...]"
resolve_target "$target"

results=$SERVER_DIR/results
[ -d "$results" ] || die "no results directory $results"
[ -n "${ANALYSIS_DIR:-}" ] || die "ANALYSIS_DIR not set in targets.local.sh"

subdir_for() {  # <campaign> -> data subdirectory
  local tag; tag=$(echo "$1" | cut -d_ -f2)
  case $tag in
    S1) echo dev/S1_naive ;;
    S2) echo dev/S2_interchange ;;
    S3) echo dev/S3_regblock ;;
    S4) echo dev/S4_packed ;;
    final) echo final ;;
    "") echo dev/other ;;
    *) echo "dev/$tag" ;;
  esac
}

if [ ${#only[@]} -gt 0 ]; then
  campaigns=("${only[@]}")
else
  mapfile -t campaigns < <(find "$results" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
[ ${#campaigns[@]} -gt 0 ] || { echo "no campaigns in $results"; exit 0; }

dry=(--dry-run); [ "$apply" = 1 ] && dry=()
[ "$apply" = 0 ] && echo "dry run (use --apply to copy)"
for c in "${campaigns[@]}"; do
  [ -d "$results/$c" ] || die "no campaign $c in $results"
  dest=$ANALYSIS_DIR/data/$(subdir_for "$c")/$c
  [ "$apply" = 1 ] && mkdir -p "$dest"
  n=$(rsync -rlt --modify-window=1 "${dry[@]}" --out-format='%n' "$results/$c/" "$dest/" | grep -vc '/$' || true)
  echo "  $c -> data/$(subdir_for "$c")/$c  ($n new or changed files)"
done
