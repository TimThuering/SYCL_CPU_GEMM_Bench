#!/usr/bin/env bash
# One benchmark run of one flow: loads the flow's modules, sets its run-time environment for t threads,
# confines the process to the first t cores (taskset) and calls gemm_bench.
#
#   bash src/scripts/server/run_one.sh <flow> <threads> <out.json> [gemm_bench options ...]
#   e.g. run_one.sh acpp_omp_accel 64 /tmp/r.json -s naive --size 1000
#
# The flow name is passed as --label. JIT / kernel caches go to a fresh temporary directory, so the
# cold iteration always includes the kernel compilation. Exit code = the one of gemm_bench
# (0 OK, 1 VERIFY_FAIL, 2 usage, 3 ERROR, 4 UNSUPPORTED; 124 timeout, GB_RUN_TIMEOUT seconds, default 3600).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

[ $# -ge 3 ] || die "usage: $0 <flow> <threads> <out.json> [gemm_bench options ...]"
flow=$1 threads=$2 out=$3; shift 3
flow_check "$flow"
bin=$GB_BUILD/$flow/gemm_bench
[ -x "$bin" ] || die "$bin not built (src/scripts/server/build.sh $flow)"
mkdir -p "$(dirname "$out")"

cache=$(mktemp -d "${TMPDIR:-/tmp}/gemm-cache.XXXXXX")
trap 'rm -rf "$cache"' EXIT
(
  flow_load "$flow"
  flow_run_env "$flow" "$threads" "$cache"
  exec timeout "${GB_RUN_TIMEOUT:-3600}" taskset -c "$(gb_cpu_list "$threads")" \
       "$bin" "$@" --label "$flow" -q -o "$out"
)
