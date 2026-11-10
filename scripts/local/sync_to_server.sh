#!/usr/bin/env bash
# Copies the sources of this repository to the server, through the sshfs mount (see targets.local.sh).
#
#   scripts/local/sync_to_server.sh <target> [--apply] [--final]
#
#   (default)  dry run: shows what would change, writes nothing
#   --apply    transfer
#   --final    refuse if the working tree has uncommitted changes (for final campaigns)
#
# What is transferred: only the whitelist below (no .git, no IDE / build directories, no dotfiles),
# into <server dir>/src/ with --delete (the server copy mirrors the whitelist exactly).
# Instead of .git, the version is written to src/SOURCE_VERSION (commit, describe, dirty), and for a
# dirty tree the uncommitted changes to src/SOURCE_PATCH. cmake/GitInfo.cmake reads SOURCE_VERSION.
# Before transferring, the whitelisted files are scanned for credential patterns (abort on a match).
# Git is only read.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHITELIST=(CMakeLists.txt README.md cmake docs include kernels scripts src tests)

target=""; apply=0; final=0
for a in "$@"; do
  case $a in
    --apply) apply=1 ;;
    --final) final=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) die "unknown option $a" ;;
    *) [ -z "$target" ] && target=$a || die "more than one target" ;;
  esac
done
[ -n "$target" ] || die "usage: $0 <target> [--apply] [--final]"
resolve_target "$target"

# ---- version of the working tree -----------------------------------------------------------------------
git_ro() { git --no-optional-locks -C "$REPO_DIR" "$@"; }
commit=$(git_ro rev-parse HEAD)
describe=$(git_ro describe --always --tags)
changes=$(git_ro status --porcelain -- "${WHITELIST[@]}")   # incl. untracked files in the whitelist
dirty=false; [ -n "$changes" ] && dirty=true

echo "repository: $REPO_DIR"
echo "version:    $describe ($commit), dirty=$dirty"
[ "$dirty" = true ] && echo "$changes" | sed 's/^/  /'
if [ "$final" = 1 ] && [ "$dirty" = true ]; then
  die "--final: uncommitted changes in the working tree (commit first)"
fi

# ---- credential scan of everything that would be transferred ------------------------------------------
# (this script itself contains the patterns and is excluded from the scan)
patterns='ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
patterns+='|-----BEGIN [A-Z ]*PRIVATE KEY-----|x-access-token:|https?://[^/@[:space:]]+:[^/@[:space:]]+@'
existing=(); for w in "${WHITELIST[@]}"; do [ -e "$REPO_DIR/$w" ] && existing+=("$w"); done
hits=$(cd "$REPO_DIR" && grep -rIlE --exclude-dir='.*' "$patterns" "${existing[@]}" |
       grep -v '^scripts/local/sync_to_server.sh$' || true)
[ -z "$hits" ] || die "possible credentials found, nothing transferred:
$hits"

# ---- transfer ---------------------------------------------------------------------------------------
# Filter: dotfiles/dirs never, SOURCE_* are written below (protected from --delete), then the whitelist
filter=(--exclude='.*' --exclude='/SOURCE_VERSION' --exclude='/SOURCE_PATCH'
        --exclude='/scripts/local/targets.local.sh')
for w in "${WHITELIST[@]}"; do
  if [ -d "$REPO_DIR/$w" ]; then filter+=(--include="/$w/***"); else filter+=(--include="/$w"); fi
done
filter+=(--exclude='*')

dest=$SERVER_DIR/src
# sshfs: no owner/group/permission transfer (-rlt), coarse timestamps (--modify-window)
rsync_opts=(-rlt --delete --modify-window=1 --itemize-changes "${filter[@]}")
if [ "$apply" = 0 ]; then
  echo; echo "dry run (use --apply to transfer) -> $dest"
  [ -d "$dest" ] || echo "  ($dest does not exist yet, will be created)"
  rsync "${rsync_opts[@]}" --dry-run "$REPO_DIR/" "$dest/"
  exit 0
fi

mkdir -p "$dest"
rsync "${rsync_opts[@]}" "$REPO_DIR/" "$dest/"

{
  echo "commit=$commit"
  echo "describe=$describe"
  echo "dirty=$dirty"
  echo "synced=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$dest/SOURCE_VERSION"

if [ "$dirty" = true ]; then
  {
    echo "# Uncommitted changes relative to $commit (git diff HEAD), synced $(date '+%F %T')"
    git_ro diff HEAD -- "${WHITELIST[@]}"
    untracked=$(git_ro ls-files --others --exclude-standard -- "${WHITELIST[@]}")
    if [ -n "$untracked" ]; then
      echo "# Untracked files (transferred, not part of the diff):"
      echo "$untracked" | sed 's/^/#   /'
    fi
  } > "$dest/SOURCE_PATCH"
else
  rm -f "$dest/SOURCE_PATCH"
fi
echo; echo "synced to $dest (dirty=$dirty)"
