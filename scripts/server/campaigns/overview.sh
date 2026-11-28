# Overview of all stages and all flows on the whole node (64 threads on peak).
#   sbatch -p peak --time=12:00:00 src/scripts/server/campaign.sbatch src/scripts/server/campaigns/overview.sh
CAMPAIGN_TAG=overview
CAMPAIGN_NAME=all
STAGES="naive interchange regblock packed"
SIZES=("1000 1000 1000" "1024 1024 1024")
THREADS="node"
REPS=1
GEMM_ARGS=(--warmup 2 --reps 10)
# 40 min per run: room for the slow PoCL runs (interchange: ~6 / 19 min per run on blade); worst case
# 6 PoCL interchange runs x 40 min = 4 h + ~1 h for the rest, well within the walltime (submit with --time=12:00:00)
RUN_TIMEOUT=2400
