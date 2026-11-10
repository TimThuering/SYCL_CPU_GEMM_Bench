#!/usr/bin/env bash
# Runs a campaign: all combinations of a campaign definition, one gemm_bench process per combination.
#
#   bash src/scripts/server/campaign.sh <definition>     (e.g. src/scripts/server/campaigns/smoke.sh)
#
# A definition is a bash file that sets:
#   CAMPAIGN_TAG    S1 | S2 | S3 | S4 | final | smoke | ...  (sorting key of pull_results.sh)
#   CAMPAIGN_NAME   short name, e.g. dev1
#   STAGES          e.g. "naive"
#   SIZES           array of "M N K", e.g. ("1000 1000 1000" "1024 1024 1024")
#   THREADS         e.g. "1 node"  (numbers, or node / socket from the machine profile)
#   FLOWS           optional, default: all flows of the machine profile
#   REPS            optional, process repetitions per combination (default 1)
#   GEMM_ARGS       optional array of further gemm_bench options, e.g. (--warmup 2 --reps 10)
#
# Output: $GB_ROOT/results/<date>_<tag>_<name>/
#   <host>/<flow>/<stage>_m<M>_n<N>_k<K>_t<threads>_r<rep>.json
#   campaign.json (definition, job, node, source version, exit code counts), runs.tsv, definition copy,
#   SOURCE_VERSION / SOURCE_PATCH copies
# Order: repetition > stage > size > threads > flow (flows innermost, so slow drifts spread over all flows).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

[ $# -eq 1 ] || die "usage: $0 <definition>"
def=$(realpath "$1"); [ -f "$def" ] || die "no definition $1"
FLOWS=$GB_FLOWS; REPS=1; GEMM_ARGS=(); SIZES=(); STAGES=""; THREADS=""; CAMPAIGN_TAG=""; CAMPAIGN_NAME=""
# shellcheck source=/dev/null
source "$def"
[ -n "$CAMPAIGN_TAG" ] && [ -n "$CAMPAIGN_NAME" ] || die "$def: CAMPAIGN_TAG and CAMPAIGN_NAME required"
[ -n "$STAGES" ] && [ ${#SIZES[@]} -gt 0 ] && [ -n "$THREADS" ] || die "$def: STAGES, SIZES and THREADS required"
[[ $CAMPAIGN_TAG$CAMPAIGN_NAME != *_* ]] || die "CAMPAIGN_TAG / CAMPAIGN_NAME must not contain '_'"
for f in $FLOWS; do flow_check "$f"; [ -x "$GB_BUILD/$f/gemm_bench" ] || die "flow $f not built"; done

threads_list=()
for t in $THREADS; do
  case $t in node) threads_list+=("$GB_THREADS_NODE") ;; socket) threads_list+=("$GB_THREADS_SOCKET") ;; *) threads_list+=("$t") ;; esac
done

name=$(date +%F)_${CAMPAIGN_TAG}_$CAMPAIGN_NAME
cdir=$GB_RESULTS/$name; i=2
while [ -e "$cdir" ]; do cdir=$GB_RESULTS/${name}-$i; i=$((i + 1)); done
name=$(basename "$cdir")
host=$(hostname -s)
mkdir -p "$cdir/$host"
cp "$def" "$cdir/definition.sh"
for f in SOURCE_VERSION SOURCE_PATCH; do [ -f "$GB_SRC/$f" ] && cp "$GB_SRC/$f" "$cdir/"; done
printf 'flow\tstage\tm\tn\tk\tthreads\trep\texit_code\tfile\n' > "$cdir/runs.tsv"

json_str() { printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }
write_campaign_json() {  # <end time or empty>
  {
    echo "{"
    echo "  \"name\": $(json_str "$name"),"
    echo "  \"tag\": $(json_str "$CAMPAIGN_TAG"),"
    echo "  \"machine\": $(json_str "$GB_MACHINE"),"
    echo "  \"host\": $(json_str "$host"),"
    echo "  \"slurm_job_id\": $(json_str "${SLURM_JOB_ID:-}"),"
    echo "  \"start\": $(json_str "$start"),"
    echo "  \"end\": $([ -n "$1" ] && json_str "$1" || echo null),"
    echo "  \"flows\": $(json_str "$FLOWS"),"
    echo "  \"stages\": $(json_str "$STAGES"),"
    echo "  \"sizes\": $(json_str "${SIZES[*]}"),"
    echo "  \"threads\": $(json_str "${threads_list[*]}"),"
    echo "  \"reps\": $REPS,"
    echo "  \"gemm_args\": $(json_str "${GEMM_ARGS[*]}"),"
    echo "  \"source_version\": $(json_str "$(tr '\n' ' ' < "$GB_SRC/SOURCE_VERSION" 2>/dev/null)"),"
    echo "  \"exit_codes\": {$(for c in "${!COUNT[@]}"; do printf '"%s": %s, ' "$c" "${COUNT[$c]}"; done | sed 's/, $//')}"
    echo "}"
  } > "$cdir/campaign.json"
}

declare -A COUNT
start=$(date '+%Y-%m-%dT%H:%M:%S%z')
write_campaign_json ""
total=$(( REPS * $(wc -w <<< "$STAGES") * ${#SIZES[@]} * ${#threads_list[@]} * $(wc -w <<< "$FLOWS") ))
log "campaign $name on $host: $total runs -> $cdir"

n=0
for ((rep = 0; rep < REPS; rep++)); do
  for stage in $STAGES; do
    for size in "${SIZES[@]}"; do
      read -r m nn k <<< "$size"
      for t in "${threads_list[@]}"; do
        for f in $FLOWS; do
          n=$((n + 1))
          rel=$host/$f/${stage}_m${m}_n${nn}_k${k}_t${t}_r${rep}.json
          bash "$GB_SERVER_DIR/run_one.sh" "$f" "$t" "$cdir/$rel" \
               -s "$stage" -m "$m" -n "$nn" -k "$k" "${GEMM_ARGS[@]}" 2>> "$cdir/stderr.log"
          rc=$?
          COUNT[$rc]=$(( ${COUNT[$rc]:-0} + 1 ))
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "$stage" "$m" "$nn" "$k" "$t" "$rep" "$rc" "$rel" >> "$cdir/runs.tsv"
          log "[$n/$total] $f $stage ${m}x${nn}x${k} t=$t r=$rep -> exit $rc"
        done
      done
    done
  done
done

write_campaign_json "$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo
echo "campaign $name done; exit codes (0 OK, 1 VERIFY_FAIL, 3 ERROR, 4 UNSUPPORTED, 124 timeout):"
for c in $(printf '%s\n' "${!COUNT[@]}" | sort -n); do echo "  $c: ${COUNT[$c]}"; done
